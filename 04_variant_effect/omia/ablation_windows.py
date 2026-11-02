#!/usr/bin/env python3
"""Sequences for the window-size ablation: 1,024 and 512 bp centred on the variant, taken from the 6,000-bp dataset."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/omia_seqs_matched.csv")
    ap.add_argument("--out-dir", default="data/windows")
    ap.add_argument("--flanks", nargs="+", type=int, default=[512, 256])
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    length = df.ref_seq.str.len().iloc[0]
    assert (df.ref_seq.str.len() == length).all() and (df.alt_seq.str.len() == length).all()
    center = length // 2
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for flank in a.flanks:
        out = df.copy()
        out["ref_seq"] = out.ref_seq.str[center - flank:center + flank]
        out["alt_seq"] = out.alt_seq.str[center - flank:center + flank]
        path = Path(a.out_dir) / f"flank{flank}_total{2 * flank}.csv"
        out.to_csv(path, index=False)
        print(path, len(out), "rows", 2 * flank, "bp")


if __name__ == "__main__":
    main()
