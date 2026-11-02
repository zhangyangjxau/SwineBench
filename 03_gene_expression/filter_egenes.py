#!/usr/bin/env python3
"""Filter liver cis-eGenes (Bonferroni < 0.05, SNP within +/-1 Mb of the TSS) from eGWAS results and a GTF."""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eqtl", required=True, help="eGWAS result table, e.g. TS1_eGWAS_results.tsv.gz")
    parser.add_argument("--gtf", required=True, help="GTF file, e.g. Sus_scrofa.Sscrofa11.1.113.gtf")
    parser.add_argument("--outdir", default="cis_eGene_results")
    parser.add_argument("--tissue", default="Liver")
    parser.add_argument("--bonferroni_threshold", type=float, default=0.05)
    parser.add_argument("--cis_window", type=int, default=1_000_000, help="cis window around TSS in bp")
    return parser.parse_args()


def extract_gene_id(attr):
    if pd.isna(attr):
        return None
    m = re.search(r'gene_id "([^"]+)"', attr)
    return m.group(1) if m else None


def extract_gene_name(attr):
    if pd.isna(attr):
        return None
    m = re.search(r'gene_name "([^"]+)"', attr)
    return m.group(1) if m else None


def load_gtf_gene_tss(gtf_file):
    """Gene-level TSS: start on + strand, end on - strand."""
    print(f"[INFO] Reading GTF: {gtf_file}", file=sys.stderr)
    colnames = ["chr", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf = pd.read_csv(
        gtf_file, sep="\t", comment="#", header=None, names=colnames,
        dtype={"chr": str, "source": str, "feature": str, "start": int, "end": int,
               "score": str, "strand": str, "frame": str, "attribute": str},
        low_memory=False,
    )
    gene_df = gtf[gtf["feature"] == "gene"].copy()
    if gene_df.empty:
        raise ValueError("No 'gene' entries found in GTF file.")

    gene_df["ENSEMBLID"] = gene_df["attribute"].apply(extract_gene_id)
    gene_df["gene_name"] = gene_df["attribute"].apply(extract_gene_name)
    gene_df["TSS"] = gene_df.apply(lambda row: row["start"] if row["strand"] == "+" else row["end"], axis=1)

    gene_tss = gene_df[["ENSEMBLID", "gene_name", "chr", "strand", "start", "end", "TSS"]].copy()
    gene_tss = gene_tss.dropna(subset=["ENSEMBLID"]).drop_duplicates(subset=["ENSEMBLID"])
    print(f"[INFO] Extracted {len(gene_tss):,} unique genes with TSS from GTF.", file=sys.stderr)
    return gene_tss


def load_eqtl_table(eqtl_file):
    print(f"[INFO] Reading eQTL table: {eqtl_file}", file=sys.stderr)
    df = pd.read_csv(
        eqtl_file, sep="\t", compression="infer",
        dtype={"TISSUE": str, "ENSEMBLID": str, "CHR": str, "SNP": str, "POS": "Int64",
               "A1": str, "A2": str, "N": "Int64", "AF1": float, "BETA": float,
               "SE": float, "P": float, "bonferroni": float},
        low_memory=False,
    )
    missing = {"TISSUE", "ENSEMBLID", "CHR", "POS", "P", "bonferroni"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in eQTL table: {missing}")
    print(f"[INFO] Loaded {len(df):,} rows from eQTL table.", file=sys.stderr)
    return df


def normalize_chr(series):
    return series.astype(str).str.replace("^chr", "", regex=True)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.tissue.lower()

    gene_tss = load_gtf_gene_tss(args.gtf)
    gene_tss["chr_norm"] = normalize_chr(gene_tss["chr"])
    gene_tss.to_csv(outdir / "gene_tss_table.tsv", sep="\t", index=False)

    eqtl = load_eqtl_table(args.eqtl)
    eqtl["CHR_norm"] = normalize_chr(eqtl["CHR"])

    tissue_df = eqtl[eqtl["TISSUE"] == args.tissue].copy()
    print(f"[INFO] Rows for tissue '{args.tissue}': {len(tissue_df):,}", file=sys.stderr)
    if tissue_df.empty:
        raise ValueError(f"No rows found for tissue = {args.tissue}")

    sig_df = tissue_df[tissue_df["bonferroni"] < args.bonferroni_threshold].copy()
    print(f"[INFO] Significant rows (bonferroni < {args.bonferroni_threshold}): {len(sig_df):,}", file=sys.stderr)
    sig_df.to_csv(outdir / f"{prefix}_significant_eqtl.tsv", sep="\t", index=False)

    merged = sig_df.merge(gene_tss, how="left", on="ENSEMBLID", suffixes=("", "_gtf"))
    missing_annot = merged["TSS"].isna().sum()
    if missing_annot > 0:
        print(f"[WARNING] {missing_annot:,} rows could not be matched to GTF gene annotation by ENSEMBLID.", file=sys.stderr)
    merged = merged.dropna(subset=["TSS"]).copy()

    chr_match = merged["CHR_norm"] == merged["chr_norm"]
    n_chr_mismatch = (~chr_match).sum()
    if n_chr_mismatch > 0:
        print(f"[WARNING] {n_chr_mismatch:,} rows have chromosome mismatch between eQTL and GTF; they will be removed.", file=sys.stderr)
    merged = merged[chr_match].copy()

    merged["distance_to_TSS"] = (merged["POS"].astype(int) - merged["TSS"].astype(int)).abs()
    cis_df = merged[merged["distance_to_TSS"] <= args.cis_window].copy()
    print(f"[INFO] cis rows (distance_to_TSS <= {args.cis_window:,}): {len(cis_df):,}", file=sys.stderr)
    cis_df.to_csv(outdir / f"{prefix}_cis_eqtl.tsv", sep="\t", index=False)

    cis_genes = (
        cis_df[["ENSEMBLID", "gene_name"]]
        .drop_duplicates()
        .sort_values(["ENSEMBLID", "gene_name"], na_position="last")
        .reset_index(drop=True)
    )
    cis_genes["ENSEMBLID"].to_csv(outdir / f"{prefix}_cis_eGenes.txt", index=False, header=False)
    cis_genes.to_csv(outdir / f"{prefix}_cis_eGenes_table.tsv", sep="\t", index=False)

    # Top cis-SNP per gene (smallest P)
    idx = cis_df.groupby("ENSEMBLID")["P"].idxmin()
    top_cis = cis_df.loc[idx].sort_values(["CHR_norm", "POS", "ENSEMBLID"]).reset_index(drop=True)
    top_cis.to_csv(outdir / f"{prefix}_top_cis_eqtl_per_gene.tsv", sep="\t", index=False)

    print("\n========== SUMMARY ==========", file=sys.stderr)
    print(f"Tissue: {args.tissue}", file=sys.stderr)
    print(f"Significant eQTL rows: {len(sig_df):,}", file=sys.stderr)
    print(f"cis-eQTL rows: {len(cis_df):,}", file=sys.stderr)
    print(f"cis-eGenes: {cis_genes['ENSEMBLID'].nunique():,}", file=sys.stderr)
    print(f"Top cis-eQTL rows: {len(top_cis):,}", file=sys.stderr)
    print("Outputs written to:", outdir.resolve(), file=sys.stderr)


if __name__ == "__main__":
    main()
