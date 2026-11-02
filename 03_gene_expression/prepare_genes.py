#!/usr/bin/env python3
"""Extract expressed genes with TSS coordinates from a GTF and an expression matrix."""
import argparse
import re

import pandas as pd


def parse_gtf_genes(gtf_file, biotype_filter=None):
    records = []
    with open(gtf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            attrs = {m.group(1): m.group(2) for m in re.finditer(r'(\w+)\s+"([^"]*)"', fields[8])}
            gene_id = attrs.get('gene_id', '')
            gene_biotype = attrs.get('gene_biotype', '')
            if not gene_id:
                continue
            if biotype_filter and gene_biotype != biotype_filter:
                continue
            records.append({
                'gene_id': gene_id,
                'chr': fields[0],
                'start': int(fields[3]),
                'end': int(fields[4]),
                'strand': fields[6],
                'gene_name': attrs.get('gene_name', gene_id),
                'gene_biotype': gene_biotype,
            })
    df = pd.DataFrame(records)
    print(f"  GTF: {len(df)} genes parsed" + (f" (biotype={biotype_filter})" if biotype_filter else ""))
    return df


def filter_expressed_genes(gtf_genes, expr_file, min_mean_expr=None):
    expr_df = pd.read_csv(expr_file, sep='\t', index_col=0, usecols=[0])
    expressed_ids = set(expr_df.index.values)
    print(f"  Expression matrix: {len(expressed_ids)} genes")

    matched = gtf_genes[gtf_genes['gene_id'].isin(expressed_ids)].copy()
    print(f"  Matched (GTF ∩ Expression): {len(matched)} genes")

    if min_mean_expr is not None:
        expr_full = pd.read_csv(expr_file, sep='\t', index_col=0)
        gene_means = expr_full.loc[matched['gene_id']].mean(axis=1)
        keep = gene_means[gene_means >= min_mean_expr].index
        matched = matched[matched['gene_id'].isin(keep)]
        print(f"  After mean_expr >= {min_mean_expr}: {len(matched)} genes")
    return matched


def add_tss_column(genes_df):
    genes_df = genes_df.copy()
    genes_df['tss'] = genes_df.apply(lambda r: r['start'] if r['strand'] == '+' else r['end'], axis=1)
    return genes_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gtf', required=True, help='GTF annotation file')
    parser.add_argument('--expr', required=True, help='Expression matrix (tab-separated)')
    parser.add_argument('--output', default='genes_all.csv', help='Output gene list CSV')
    parser.add_argument('--biotype', default=None, help='Filter by gene_biotype (e.g. protein_coding)')
    parser.add_argument('--min-mean-expr', type=float, default=None, help='Minimum mean expression to include a gene')
    args = parser.parse_args()

    print("[1] Parsing GTF ...")
    gtf_genes = parse_gtf_genes(args.gtf, biotype_filter=args.biotype)
    print("[2] Filtering to expressed genes ...")
    expressed = filter_expressed_genes(gtf_genes, args.expr, args.min_mean_expr)
    print("[3] Adding TSS ...")
    expressed = add_tss_column(expressed)
    expressed = expressed.sort_values(['chr', 'start']).reset_index(drop=True)
    expressed.to_csv(args.output, index=False)
    print(f"\nSaved {len(expressed)} genes to {args.output}")

    print("\nChromosome distribution:")
    for chrom, count in expressed['chr'].value_counts().sort_index().items():
        print(f"  chr{chrom}: {count} genes")


if __name__ == '__main__':
    main()
