#!/usr/bin/env python3
"""Build the cis-eGene coordinate table from the gene list and the eGene IDs."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", required=True, help="Gene coordinate CSV from prepare_genes.py")
    parser.add_argument("--egene-list", required=True, help="One Ensembl gene ID per line")
    parser.add_argument("--output", default="genes_cis_eGenes.csv")
    args = parser.parse_args()

    genes = pd.read_csv(args.genes, dtype=str).drop_duplicates("gene_id")
    required = {"gene_id", "chr", "start", "end", "strand", "tss"}
    missing = required - set(genes.columns)
    if missing:
        raise ValueError(f"Gene table is missing columns: {sorted(missing)}")

    egene_ids = {
        line.strip().split()[0]
        for line in Path(args.egene_list).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    selected = genes[genes["gene_id"].isin(egene_ids)].copy()
    selected = selected.sort_values(["chr", "tss", "gene_id"]).reset_index(drop=True)
    missing_ids = sorted(egene_ids - set(selected["gene_id"]))
    if missing_ids:
        print(f"[warning] {len(missing_ids)} eGenes were not present in the coordinate table")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(f"[done] wrote {len(selected)} cis-eGenes to {output}")


if __name__ == "__main__":
    main()
