#!/usr/bin/env python3
# Random-forest heads on frozen embeddings: stratified 1/10/100 % training subsets x seeds, fixed test set.
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

# model -> (embedding subdir, train file, test file); subdirs are written by embed_<model>.py
SPECS = {
    "DNABERT2":  ("dnabert2",  "train_embed_dnabert_meanpool.csv",   "test_embed_dnabert_meanpool.csv"),
    "GENARator": ("generator", "train_embed_generator_meanpool.csv", "test_embed_generator_meanpool.csv"),
    "GENA-LM":   ("genalm",    "train_embed_genalm_meanpool.csv",    "test_embed_genalm_meanpool.csv"),
    "HyenaDNA":  ("hyenadna",  "train_embed_hyenadna_450k.csv",      "test_embed_hyenadna_450k.csv"),
    "NT-v2":     ("ntv2",      "train_embed_ntv2.csv",               "test_embed_ntv2.csv"),
    "Evo2":      ("evo2",      "train_embed_evo2.csv",               "test_embed_evo2.csv"),
}


def load_split(path):
    df = pd.read_csv(path, dtype=np.float32)
    y = df["target"].astype(np.int8).to_numpy(); x = df.drop(columns=["target"]).to_numpy(dtype=np.float32, copy=False)
    m = np.isfinite(x).all(axis=1) & np.isfinite(y); return x[m], y[m]


def subset_n(x, y, n_used, seed):
    if n_used >= len(y): return x, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=int(n_used), random_state=seed, stratify=y, shuffle=True)
    idx = np.sort(idx); return x[idx], y[idx]


def fit(x_tr, y_tr, x_te, y_te, workers, seed):
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, max_features="sqrt", min_samples_split=2, random_state=seed, n_jobs=workers)
    clf.fit(x_tr, y_tr); p = clf.predict_proba(x_te)[:, 1]; lab = (p >= 0.5).astype(np.int8)
    return {"MCC": matthews_corrcoef(y_te, lab), "AUC": roc_auc_score(y_te, p), "AP": average_precision_score(y_te, p),
            "F1-Score": f1_score(y_te, lab), "Accuracy": accuracy_score(y_te, lab)}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="embeddings"); ap.add_argument("--out", default="results/rf")
    ap.add_argument("--model", required=True, choices=SPECS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 17, 123, 997]); ap.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); root = Path(a.embeddings) / SPECS[a.model][0]; trn, tst = SPECS[a.model][1:]; out = Path(a.out)
    tasks = sorted(p.parent for p in root.rglob(trn)); print(f"[{a.model}] {len(tasks)} tasks; seeds={a.seeds}; fractions={a.fractions}", flush=True)
    for ti, td in enumerate(tasks, 1):
        rel = td.relative_to(root); category, sample = rel.parts[0], rel.parts[-1]
        todo = [(s, f) for s in a.seeds for f in a.fractions if not (out / "jobs" / a.model / f"seed_{s}" / f"frac_{f:g}" / f"{category}__{sample}.csv").exists()]
        if not todo: continue
        x_tr_full, y_tr_full = load_split(td / trn); x_te, y_te = load_split(td / tst)
        for s, f in todo:
            n_used = len(y_tr_full) if f >= 1 else int(len(y_tr_full) * f)
            x_tr, y_tr = subset_n(x_tr_full, y_tr_full, n_used, s)
            met, pred = fit(x_tr, y_tr, x_te, y_te, a.workers, s)
            row = {"model": a.model, "dataset": str(rel), "category": category, "sample": sample, "classifier": "rf", "train_fraction": f, "seed": s,
                   "train_n_full": len(y_tr_full), "train_n_used": len(y_tr), "test_n": len(y_te), "train_pos_rate": float(y_tr.mean()), "test_pos_rate": float(y_te.mean()),
                   **met, "params": json.dumps({"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2, "random_state": s})}
            jf = out / "jobs" / a.model / f"seed_{s}" / f"frac_{f:g}" / f"{category}__{sample}.csv"; pf = out / "preds" / a.model / f"seed_{s}" / f"frac_{f:g}" / f"{category}__{sample}.csv"
            jf.parent.mkdir(parents=True, exist_ok=True); pf.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(jf, index=False); pd.DataFrame({"True": y_te, "Pred": pred}).to_csv(pf, index=False)
            print(f"[{a.model} {ti}/{len(tasks)}] {rel} seed={s} frac={f:g} n={len(y_tr)} AUC={met['AUC']:.4f}", flush=True)


if __name__ == "__main__":
    main()
