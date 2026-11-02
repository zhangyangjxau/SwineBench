#!/usr/bin/env python
"""Leave-one-species-out random forest on alt-ref embedding differences: 3 windows x 6 models x 5 seeds."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
WINDOWS = ["flank3000_total6000", "flank512_total1024", "flank256_total512"]
MODELS = ["evo2", "ntv2", "genalm", "hyena", "dnabert2", "generator"]
SEEDS = [42, 13, 17, 123, 997]


def load_diff(d, model):
    pos = pd.read_csv(d / f"{model}_differences_pathogenic.csv", header=None, low_memory=False)
    neg = pd.read_csv(d / f"{model}_differences_common.csv", header=None, low_memory=False)
    pos["label"] = 1
    neg["label"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    n_emb = df.shape[1] - 1 - 5
    X = df.iloc[:, :n_emb].to_numpy(dtype=np.float32)
    species = df.iloc[:, n_emb + 4].astype(str).to_numpy()
    y = df["label"].to_numpy()
    return X, y, species, n_emb


def main():
    parser = argparse.ArgumentParser(description="LOSO random forest on OMIA embedding differences")
    parser.add_argument("--embeddings", default=ROOT / "embeddings", type=Path, help="<embeddings>/<window>/<model>_differences_*.csv")
    parser.add_argument("--out", default=ROOT / "results" / "rf", type=Path)
    parser.add_argument("--n-jobs", type=int, default=32)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "rf_metrics.csv"
    rows = []
    for window in WINDOWS:
        d = args.embeddings / window
        for model in MODELS:
            t0 = time.time()
            X, y, sp, n_emb = load_diff(d, model)
            print(f"[{window}] {model}: n={len(y)} dim={n_emb}", flush=True)
            for seed in SEEDS:
                for test_sp in sorted(np.unique(sp)):
                    m = sp == test_sp
                    if y[m].min() == y[m].max():
                        continue
                    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                random_state=seed, n_jobs=args.n_jobs)
                    rf.fit(X[~m], y[~m])
                    p = rf.predict_proba(X[m])[:, 1]
                    rows.append(dict(window=window, model=model, mode="diff",
                                     classifier="rf_fixed_300_sqrt", seed=seed,
                                     test_species=test_sp,
                                     auroc=roc_auc_score(y[m], p),
                                     auprc=average_precision_score(y[m], p),
                                     n_train=int((~m).sum()), n_test=int(m.sum())))
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    agg = (df.groupby(["window", "model", "test_species"])["auroc"]
             .agg(["mean", "std"]).reset_index()
             .rename(columns={"mean": "auroc_mean", "std": "auroc_sd"}))
    agg.to_csv(args.out / "rf_auroc_summary.csv", index=False)
    macro = df.groupby(["window", "model", "seed"])["auroc"].mean().groupby(["window", "model"]).agg(["mean", "std"])
    print(macro.round(4).to_string())


if __name__ == "__main__":
    main()
