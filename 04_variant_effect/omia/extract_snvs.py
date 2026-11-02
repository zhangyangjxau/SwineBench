#!/usr/bin/env python3
"""Extract single-nucleotide variants of the target species from an OMIA variant table."""
import re
import sys

import pandas as pd

if len(sys.argv) < 2:
    sys.exit(f"Usage: python {sys.argv[0]} <omia_variants.csv>")
csv_filepath = sys.argv[1]

columns_to_keep = [
    'OMIA Variant ID', 'OMIA Phene-Species ID(s)', 'Species Name', 'Gene', 'Type of Variant',
    'Variant Effect', 'Reference Sequence', 'Chr.', 'g. or m.', 'EVA ID', 'Inferred EVA ID',
]
allowed_species = ["dog", "taurine cattle", "domestic cat", "pig", "sheep", "domestic horse",
                   "chicken", "rabbit", "goat"]
snv_pattern = re.compile(r'(?:g\.|:g\.)(\d+)([ACGTN])>([ACGTN])')


def parse_g_notation_for_snv(notation_str):
    if not isinstance(notation_str, str):
        return None, None, None
    match = snv_pattern.search(notation_str)
    return match.groups() if match else (None, None, None)


df = pd.read_csv(csv_filepath)
print(f"Read {len(df)} rows")
df = df[df['Species Name'].isin(allowed_species)].copy()
print(f"Kept {len(df)} rows for species {', '.join(allowed_species)}")
if df.empty:
    sys.exit(0)

missing_cols = [col for col in columns_to_keep if col not in df.columns]
if missing_cols:
    sys.exit(f"Missing columns: {', '.join(missing_cols)}")
df_selected = df[columns_to_keep].copy()

parse_results = df_selected['g. or m.'].astype(str).apply(parse_g_notation_for_snv)
df_selected['Position'] = [res[0] for res in parse_results]
df_selected['Ref Allele'] = [res[1] for res in parse_results]
df_selected['Alt Allele'] = [res[2] for res in parse_results]

df_final = df_selected[df_selected['Position'].notna()].copy()
print(f"{len(df_final)} rows match the SNV pattern")
df_final['Position'] = pd.to_numeric(df_final['Position'], errors='coerce')
df_final = df_final.drop(columns=['g. or m.']).rename(columns={'Chr.': 'Chr'})
cols_order = ['OMIA Variant ID', 'OMIA Phene-Species ID(s)', 'Species Name', 'Gene', 'Type of Variant',
              'Variant Effect', 'Reference Sequence', 'Chr', 'Position', 'Ref Allele', 'Alt Allele',
              'EVA ID', 'Inferred EVA ID']
df_final = df_final[[col for col in cols_order if col in df_final.columns]]

output_filename = f"{csv_filepath.rsplit('.', 1)[0]}_extracted_snvs.csv"
df_final.to_csv(output_filename, index=False)
print(f"Saved {output_filename}")
