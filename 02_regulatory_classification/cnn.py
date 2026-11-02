#!/usr/bin/env python
# Sequence CNN baseline (one-hot input) at 1/10/100 % training data x seeds, run in parallel over GPUs.
import argparse
import copy
import json
import os
import queue
import random
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

FRACTIONS = [1.0, 0.1, 0.01]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 64, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.sigmoid(self.fc(x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(text):
    return text.replace("/", "__").replace(" ", "_")


def fraction_label(fraction):
    return f"{fraction:.2f}".rstrip("0").rstrip(".")


def result_path(task, out):
    return Path(out) / "jobs" / f"seed_{task['seed']}" / f"frac_{fraction_label(task['fraction'])}" / f"{safe_name(task['dataset'])}.csv"


def discover_tasks(data_root, fractions, seeds, limit=None):
    tasks = []
    for train_file in sorted(data_root.glob("*/*/train.csv")):
        ds_dir = train_file.parent
        rel = ds_dir.relative_to(data_root)
        category, sample = rel.parts[0], rel.parts[1]
        for seed in seeds:
            for fraction in fractions:
                tasks.append({"dataset_dir": str(ds_dir), "dataset": f"{category}/{sample}", "category": category,
                              "sample": sample, "fraction": float(fraction), "seed": int(seed)})
    if limit:
        tasks = tasks[:limit]
    return tasks


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"expected sequence and target columns: {csv_path}")
    return df.iloc[:, :2].copy()


def subsample_train(df, fraction, seed):
    if fraction >= 0.999999:
        return df.reset_index(drop=True)
    y = df.iloc[:, 1].astype(int).to_numpy()
    n = max(2, int(round(len(df) * fraction)))
    if np.min(np.bincount(y, minlength=2)) >= 2:
        sub, _ = train_test_split(df, train_size=n, stratify=y, random_state=seed)
        return sub.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def split_train_val(df, seed):
    y = df.iloc[:, 1].astype(int).to_numpy()
    if len(df) < 10 or np.min(np.bincount(y, minlength=2)) < 2:
        raise ValueError("not enough samples per class for train/val split")
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=y, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def encode_sequences(seq_series, max_length):
    seqs = seq_series.astype(str).str.upper().tolist()
    n = len(seqs)
    joined = "".join(s[:max_length].ljust(max_length, "N") for s in seqs)
    chars = np.frombuffer(joined.encode("ascii"), dtype=np.uint8).reshape(n, max_length)
    lut = np.full(256, 4, dtype=np.int64)
    for idx, nuc in enumerate(b"ACGTN"):
        lut[nuc] = idx
    channels = lut[chars]
    x = np.zeros((n, 5, max_length), dtype=np.float32)
    x[np.arange(n)[:, None], channels, np.arange(max_length)[None, :]] = 1.0
    return torch.from_numpy(x)


def make_loader(df, max_length, batch_size, shuffle, seed, num_workers, pin_memory):
    x = encode_sequences(df.iloc[:, 0], max_length)
    y = torch.tensor(df.iloc[:, 1].astype(float).to_numpy(), dtype=torch.float32).view(-1, 1)
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers, "pin_memory": pin_memory}
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(TensorDataset(x, y), **kwargs)


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(model(x.to(device, non_blocking=True)).detach().cpu().numpy().ravel())
            labels.append(y.numpy().ravel())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    preds = (probs > 0.5).astype(int)
    return {"MCC": float(matthews_corrcoef(labels, preds)), "AUC": float(roc_auc_score(labels, probs)),
            "F1-Score": float(f1_score(labels, preds)), "Accuracy": float(accuracy_score(labels, preds))}


def fit_one(task, gpu_id, args):
    out_file = result_path(task, args.out)
    if out_file.exists():
        return {"status": "skipped", "path": str(out_file), **task}

    set_seed(task["seed"])
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    ds_dir = Path(task["dataset_dir"])
    train_full = load_dataframe(ds_dir / "train.csv")
    test_df = load_dataframe(ds_dir / "test.csv")
    train_sub = subsample_train(train_full, task["fraction"], task["seed"])
    train_df, val_df = split_train_val(train_sub, task["seed"])

    max_length = max(train_full.iloc[:, 0].astype(str).str.len().max(), test_df.iloc[:, 0].astype(str).str.len().max())
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_df, max_length, args.batch_size, True, task["seed"], args.num_workers, pin_memory)
    val_loader = make_loader(val_df, max_length, args.batch_size, False, task["seed"], args.num_workers, pin_memory)
    test_loader = make_loader(test_df, max_length, args.batch_size, False, task["seed"], args.num_workers, pin_memory)

    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    epochs_run = 0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                val_loss += criterion(model(x), y).item() * x.size(0)
                val_n += x.size(0)
        val_loss /= max(1, val_n)
        epochs_run = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate(model, test_loader, device)
    row = {
        "model": "CNN",
        "dataset": task["dataset"],
        "category": task["category"],
        "sample": task["sample"],
        "classifier": "simple_cnn",
        "train_fraction": task["fraction"],
        "seed": task["seed"],
        "train_n_full": int(len(train_full)),
        "train_n_used": int(len(train_sub)),
        "val_n": int(len(val_df)),
        "test_n": int(len(test_df)),
        "train_pos_rate": float(train_sub.iloc[:, 1].mean()),
        "test_pos_rate": float(test_df.iloc[:, 1].mean()),
        "epochs_run": int(epochs_run),
        "best_val_loss": float(best_val_loss),
        "runtime_sec": round(time.time() - start, 3),
        "device": str(device),
        "params": json.dumps({"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr, "patience": args.patience,
                              "num_workers": args.num_workers, "architecture": "SimpleCNN"}, sort_keys=True),
        **metrics,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_file, index=False)
    return {"status": "done", "path": str(out_file), **row}


def worker_loop(gpu_id, task_queue, result_queue, args):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(args.omp_threads))
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break
        try:
            result = fit_one(task, gpu_id, args)
        except Exception as exc:
            result = {"status": "error", "error": repr(exc), **task}
        result_queue.put(result)


def combine_results(out):
    files = sorted((out / "jobs").rglob("*.csv")) if (out / "jobs").exists() else []
    if not files:
        return None
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    df = df.sort_values(["train_fraction", "seed", "category", "sample"])
    df.to_csv(out / "all_results.csv", index=False)

    metric_cols = ["AUC", "MCC", "F1-Score", "Accuracy"]
    agg = {**{m: ["mean", "std"] for m in metric_cols}, "dataset": "count"}
    by_model = df.groupby(["train_fraction", "model"], as_index=False).agg(agg)
    by_model.columns = ["_".join(col).rstrip("_") for col in by_model.columns.to_flat_index()]
    by_model.to_csv(out / "summary_by_fraction_model.csv", index=False)

    by_category = df.groupby(["train_fraction", "model", "category"], as_index=False).agg(agg)
    by_category.columns = ["_".join(col).rstrip("_") for col in by_category.columns.to_flat_index()]
    by_category.to_csv(out / "summary_by_fraction_model_category.csv", index=False)

    df.groupby(["train_fraction", "model"], as_index=False)[metric_cols].mean().to_csv(out / "compact_model_summary.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/cnn")
    parser.add_argument("--fractions", nargs="*", type=float, default=FRACTIONS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--omp-threads", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(Path(args.data), args.fractions, args.seeds, args.limit)
    print(f"Discovered {len(tasks)} CNN tasks; gpus={args.gpus}; fractions={args.fractions}; seeds={args.seeds}", flush=True)

    ctx = get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)

    workers = []
    gpu_list = args.gpus if torch.cuda.is_available() and args.gpus else [None]
    for gpu_id in gpu_list:
        proc = ctx.Process(target=worker_loop, args=(gpu_id, task_queue, result_queue, args))
        proc.start()
        workers.append(proc)

    done = 0
    status_counts = {}
    while done < len(tasks):
        try:
            result = result_queue.get(timeout=30)
        except queue.Empty:
            if not any(proc.is_alive() for proc in workers):
                break
            continue
        done += 1
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        print(f"[{done}/{len(tasks)}] {status} {result.get('dataset', '')} frac={result.get('train_fraction', '')} "
              f"seed={result.get('seed', '')} AUC={result.get('AUC', '')} {result.get('error', '')}", flush=True)

    for proc in workers:
        proc.join()
    for proc in workers:
        if proc.exitcode != 0:
            print(f"worker exitcode={proc.exitcode}", flush=True)

    df = combine_results(out)
    print(f"Status: {status_counts}", flush=True)
    if df is not None:
        print(df.groupby(["train_fraction", "model"])["AUC"].mean().unstack().round(4), flush=True)


if __name__ == "__main__":
    main()
