#!/usr/bin/env python3
"""Train the two-layer CNN probe on precomputed per-base embeddings (H5) and evaluate on the test split."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

LABEL_NAMES = [
    "NC",
    "CDS_F", "UTR5_F", "UTR3_F", "DF", "IF", "AF",
    "CDS_R", "UTR5_R", "UTR3_R", "DR", "IR", "AR",
]
NUM_CLASSES = len(LABEL_NAMES)

COLLAPSE_6 = {
    "CDS":      [1, 7],
    "5UTR":     [2, 8],
    "3UTR":     [3, 9],
    "Donor":    [4, 10],
    "Intron":   [5, 11],
    "Acceptor": [6, 12],
}
COLLAPSE_4 = {
    "Exon":     [1, 2, 3, 7, 8, 9],
    "Intron":   [5, 11],
    "Donor":    [4, 10],
    "Acceptor": [6, 12],
}


@dataclass
class TrainConfig:
    embedding_dir: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_size: int
    kernel_size: int
    seed: int
    gradient_accumulation_steps: int
    num_workers: int
    class_balanced: bool
    amp: bool
    max_train_records: Optional[int]
    max_valid_records: Optional[int]
    max_test_records: Optional[int]


class H5GeneFindingDataset(Dataset):
    def __init__(self, path: Path, max_records: Optional[int] = None):
        self.path = path
        with h5py.File(path, "r") as h5:
            if not bool(h5.attrs.get("complete", False)):
                raise RuntimeError(f"{path} is not marked complete")
            schema = h5.attrs.get("schema")
            if schema is not None and schema != "bend13":
                raise RuntimeError(f"{path} has schema={schema!r}, expected bend13")
            n = int(h5.attrs["num_records"])
            self.hidden_size = int(h5.attrs["hidden_size"])
            self.lengths = np.asarray(h5["meta/lengths"][:], dtype=np.int32)
        if max_records is not None:
            n = min(n, max_records)
            self.lengths = self.lengths[:n]
        self.n = n
        self._h5 = None

    def __len__(self): return self.n

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        h5 = self._file()
        emb = torch.from_numpy(h5[f"embeddings/{idx}"][:])
        labels = torch.from_numpy(h5[f"labels/{idx}"][:].astype(np.int64, copy=False))
        if emb.shape[0] != labels.numel():
            raise RuntimeError(f"{self.path}:{idx} emb/label length mismatch")
        return emb, labels


def collate(batch):
    embs, labs = zip(*batch)
    embs = pad_sequence(embs, batch_first=True, padding_value=0.0)
    labs = pad_sequence(labs, batch_first=True, padding_value=-100)
    return embs, labs


class Bend13CnnHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad)
        self.classifier = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        return self.classifier(x)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def compute_class_weights(loader: DataLoader) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, y in loader:
        flat = y.reshape(-1).numpy()
        flat = flat[flat != -100]
        counts += np.bincount(flat, minlength=NUM_CLASSES)
    inv = 1.0 / np.maximum(counts, 1)
    w = inv * (NUM_CLASSES / inv.sum())
    print(json.dumps({"event": "class_weights",
                      "counts": counts.tolist(),
                      "weights": [round(float(x), 4) for x in w]}), flush=True)
    return torch.tensor(w, dtype=torch.float32)


def per_class_mcc(yt: np.ndarray, yp: np.ndarray, k: int) -> List[float]:
    out = []
    for c in range(k):
        t = (yt == c).astype(np.int8)
        p = (yp == c).astype(np.int8)
        if t.sum() == 0 and p.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float(matthews_corrcoef(t, p)))
    return out


def collapse(y: np.ndarray, mapping: Dict[str, List[int]]) -> Tuple[np.ndarray, List[str]]:
    names = list(mapping.keys())
    out = np.zeros_like(y)
    for i, (n, cls) in enumerate(mapping.items(), start=1):
        out[np.isin(y, cls)] = i
    return out, ["background"] + names


def train_one_epoch(head, loader, optimizer, device, grad_accum, epoch,
                    loss_fn, log_every, amp_scaler):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    t0 = time.time()
    use_amp = amp_scaler is not None
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(x)
                loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            (loss / grad_accum).backward()
        else:
            logits = head(x)
            loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            (loss / grad_accum).backward()
        if step % grad_accum == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if log_every and step % log_every == 0:
            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(json.dumps({"event": "train_progress", "epoch": epoch, "step": step,
                              "steps": len(loader), "loss": float(np.mean(losses)),
                              "elapsed_sec": round(time.time() - t0, 1),
                              "peak_vram_gb": round(vram, 2)}), flush=True)
    return float(np.mean(losses)) if losses else math.nan


@torch.inference_mode()
def evaluate(head, loader, device, split, loss_fn):
    head.eval()
    losses: List[float] = []
    trues: List[np.ndarray] = []; preds: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = head(x)
        loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
        losses.append(float(loss.detach().cpu()))
        pred = logits.argmax(dim=-1)
        m = y != -100
        trues.append(y[m].detach().cpu().numpy().astype(np.int8))
        preds.append(pred[m].detach().cpu().numpy().astype(np.int8))
    yt = np.concatenate(trues); yp = np.concatenate(preds)
    mcc13 = float(matthews_corrcoef(yt, yp))
    pc13 = per_class_mcc(yt, yp, NUM_CLASSES)
    yt6, n6 = collapse(yt, COLLAPSE_6); yp6, _ = collapse(yp, COLLAPSE_6)
    mcc6 = float(matthews_corrcoef(yt6, yp6))
    pc6 = per_class_mcc(yt6, yp6, len(n6))
    yt4, n4 = collapse(yt, COLLAPSE_4); yp4, _ = collapse(yp, COLLAPSE_4)
    mcc4 = float(matthews_corrcoef(yt4, yp4))
    pc4 = per_class_mcc(yt4, yp4, len(n4))
    return {
        "split": split,
        "loss": float(np.mean(losses)) if losses else math.nan,
        "overall_mcc_13class": mcc13,
        "per_class_mcc_13class": dict(zip(LABEL_NAMES, pc13)),
        "overall_mcc_6class_collapsed": mcc6,
        "per_class_mcc_6class_collapsed": dict(zip(n6, pc6)),
        "overall_mcc_4class_bendlike": mcc4,
        "per_class_mcc_4class_bendlike": dict(zip(n4, pc4)),
        "n_positions": int(yt.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--class-balanced", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="Use bf16 autocast for forward/backward")
    ap.add_argument("--max-train-records", type=int, default=None)
    ap.add_argument("--max-valid-records", type=int, default=None)
    ap.add_argument("--max-test-records", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.embedding_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = H5GeneFindingDataset(emb_dir / "train.h5", args.max_train_records)
    valid_ds = H5GeneFindingDataset(emb_dir / "valid.h5", args.max_valid_records)
    test_ds  = H5GeneFindingDataset(emb_dir / "test.h5",  args.max_test_records)

    loader_kw = dict(num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  **loader_kw)
    valid_ld = DataLoader(valid_ds, args.batch_size, shuffle=False, **loader_kw)
    test_ld  = DataLoader(test_ds,  args.batch_size, shuffle=False, **loader_kw)

    in_dim = train_ds.hidden_size
    head = Bend13CnnHead(in_dim, args.hidden_size, args.kernel_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in head.parameters())
    print(json.dumps({"event": "model_ready", "input_size": in_dim,
                      "num_classes": NUM_CLASSES, "params": n_params,
                      "device": str(device), "amp": args.amp,
                      "batch_size": args.batch_size, "num_workers": args.num_workers}), flush=True)

    if args.class_balanced:
        weights = compute_class_weights(train_ld).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    cfg = TrainConfig(
        embedding_dir=str(emb_dir), output_dir=str(out_dir),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, hidden_size=args.hidden_size,
        kernel_size=args.kernel_size, seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers, class_balanced=args.class_balanced, amp=args.amp,
        max_train_records=args.max_train_records,
        max_valid_records=args.max_valid_records,
        max_test_records=args.max_test_records,
    )
    save_json(out_dir / "config.json", asdict(cfg))

    amp_scaler = True if (args.amp and device.type == "cuda") else None
    best_mcc = -1.0; best_epoch = -1; no_improve = 0
    history: List[Dict] = []
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        t0 = time.time()
        train_loss = train_one_epoch(head, train_ld, optimizer, device,
                                     args.gradient_accumulation_steps, epoch, loss_fn,
                                     args.log_every, amp_scaler)
        val = evaluate(head, valid_ld, device, "valid", loss_fn)
        epoch_time = round(time.time() - t0, 1)
        vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(json.dumps({"event": "epoch", "epoch": epoch,
                          "train_loss": train_loss, "valid_loss": val["loss"],
                          "valid_mcc13": val["overall_mcc_13class"],
                          "valid_mcc6": val["overall_mcc_6class_collapsed"],
                          "valid_mcc4": val["overall_mcc_4class_bendlike"],
                          "epoch_time_sec": epoch_time,
                          "peak_vram_gb": round(vram, 2)}), flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid": val,
                        "epoch_time_sec": epoch_time})
        save_json(out_dir / "metrics.json",
                  {"history": history, "best_mcc13": best_mcc, "best_epoch": best_epoch})
        if val["overall_mcc_13class"] > best_mcc:
            best_mcc = val["overall_mcc_13class"]; best_epoch = epoch; no_improve = 0
            torch.save({"head_state_dict": head.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "best_mcc": best_mcc,
                        "config": asdict(cfg), "label_names": LABEL_NAMES},
                       out_dir / "best.pt")
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(json.dumps({"event": "early_stop", "epoch": epoch,
                                  "best_epoch": best_epoch, "best_mcc13": best_mcc}), flush=True)
                break

    torch.save({"head_state_dict": head.state_dict(),
                "epoch": final_epoch, "config": asdict(cfg), "label_names": LABEL_NAMES},
               out_dir / "last.pt")
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    head.load_state_dict(ck["head_state_dict"])
    test_res = evaluate(head, test_ld, device, "test", loss_fn)
    save_json(out_dir / "test_metrics.json", {
        "best_epoch": best_epoch, "best_valid_mcc13": best_mcc, "test": test_res,
    })
    print(json.dumps({"event": "final_test",
                      "best_epoch": best_epoch, "best_valid_mcc13": best_mcc,
                      "test_mcc13": test_res["overall_mcc_13class"],
                      "test_mcc6": test_res["overall_mcc_6class_collapsed"],
                      "test_mcc4": test_res["overall_mcc_4class_bendlike"]}), flush=True)


if __name__ == "__main__":
    main()
