#!/usr/bin/env python3
"""Extract ref/alt windows (default 6,000 bp) around sampled control SNVs from NCBI genome archives."""
import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

GENOME_NAME_TO_ACCESSION = {
    "GRCg6a": "GCF_000002315.6",
    "ARS-UCD1.2": "GCF_002263795.1",
    "ARS-UCD1.3": "GCF_002263795.2",
    "Sscrofa11.1": "GCF_000003025.6",
    "CanFam3.1": "GCF_000002285.3",
    "Dog10K_Boxer_Tasha": "GCF_000002285.5",
    "EquCab3.0": "GCF_002863925.1",
    "Felis_catus_9.0": "GCF_000181335.3",
    "Oar_rambouillet_v1.0": "GCF_002742125.1",
    "ROS_Cfam_1.0": "GCF_014441545.1",
    "UU_Cfam_GSD_1.0": "GCF_011100685.1",
    "Oar_v3.1": "GCF_000298735.1",
    "F.catus_Fca126_mat1.0": "GCF_018350175.1",
    "UMD_3.1.1": "GCF_000003055.6",
    "ARS-UI_Ramb_v3.0": "GCF_016772045.2",
    "ARS1.2": "GCF_001704415.2",
    "Oar_v4.0": "GCF_000298735.2",
    "Felis_catus 9.0": "GCF_000181335.3",
    "OryCun2.0": "GCF_000003625.3",
    "EquCab3": "GCF_002863925.1",
    "Oar_Rambouillet_v1.0": "GCF_002742125.1",
}


def build_chromosome_map(fasta_path):
    chr_map = {}
    with open(fasta_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            parts = line[1:].split(None, 1)
            if not parts:
                continue
            seq_id = parts[0]
            description = parts[1] if len(parts) > 1 else ""

            match = re.search(r'[Cc]hr(?:omosome)?\.?\s+([\w.-]+)', description)
            human_chr = match.group(1).rstrip(',.;') if match else None

            if human_chr is None and re.match(r'^chr[\w.-]+$|^[0-9XYMTZW]+$', seq_id, re.I):
                human_chr = seq_id

            chr_map[seq_id] = seq_id

            if human_chr:
                if human_chr not in chr_map:
                    chr_map[human_chr] = seq_id
                if human_chr.isdigit():
                    chr_map.setdefault(f"chr{human_chr}", seq_id)
                elif human_chr.lower().startswith('chr'):
                    chr_map.setdefault(human_chr.replace('chr', '', 1), seq_id)

    return chr_map


def resolve_chr(csv_chr, chr_map, fasta_obj):
    key = chr_map.get(csv_chr)
    if key:
        return key
    if csv_chr.startswith('chr'):
        key = chr_map.get(csv_chr.replace('chr', '', 1))
    else:
        key = chr_map.get(f"chr{csv_chr}")
    if key:
        return key
    if csv_chr in fasta_obj:
        return csv_chr
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract ref/alt sequence windows around control SNVs")
    parser.add_argument("input_csv", nargs="?", default="data/control_snvs_type_matched.csv")
    parser.add_argument("output_csv", nargs="?", default="data/control_snvs_type_matched_sequences.csv")
    parser.add_argument("--genome-dir", default="data/genomes", help="directory with NCBI GCF_*.zip archives")
    parser.add_argument("--temp-dir", default="data/genomes/fasta", help="where extracted FASTA files are kept")
    parser.add_argument("--window", type=int, default=6000, help="total window length in bp")
    args = parser.parse_args()
    input_csv, output_csv = args.input_csv, args.output_csv

    genome_dir = Path(args.genome_dir)
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, dtype={'Chr': str})
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').astype('Int64')
    df.dropna(subset=['Position'], inplace=True)
    print(f"Loaded {len(df)} variants from {input_csv}")

    df['NCBI_Accession'] = df['Reference Sequence'].str.strip().map(GENOME_NAME_TO_ACCESSION)
    unmapped = df['NCBI_Accession'].isna().sum()
    if unmapped:
        print(f"Warning: {unmapped} variants have unmapped Reference Sequence")
        print(df.loc[df['NCBI_Accession'].isna(), 'Reference Sequence'].unique())

    df['Ref_Sequence_Window'] = pd.NA
    df['Alt_Sequence_Window'] = pd.NA
    df['Processing_Status'] = 'Pending'

    window = args.window
    flank = window // 2  # variant at 0-based index `flank`
    genome_cache = {}

    success = 0
    fail = 0

    for accession, group in df[df['NCBI_Accession'].notna()].groupby('NCBI_Accession'):
        print(f"\n--- {accession}: {len(group)} variants ---")

        if accession not in genome_cache:
            fasta_path = temp_dir / f"{accession}.fna"
            if not fasta_path.is_file():
                gca = accession.replace('GCF', 'GCA')
                gca_path = temp_dir / f"{gca}.fna"
                if gca_path.is_file():
                    fasta_path = gca_path

            if not fasta_path.is_file():
                zip_path = genome_dir / f"{accession}.zip"
                if not zip_path.is_file():
                    zip_path = genome_dir / f"{accession.replace('GCF', 'GCA')}.zip"
                if zip_path.is_file():
                    print(f"Extracting FASTA from {zip_path}...")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            fna_files = [n for n in zf.namelist() if n.endswith('.fna')]
                            if fna_files:
                                zf.extract(fna_files[0], temp_dir)
                                extracted = temp_dir / fna_files[0]
                                extracted.rename(fasta_path)
                    except Exception as e:
                        print(f"Failed to extract: {e}")
                        df.loc[group.index, 'Processing_Status'] = 'Failed: ZIP extraction'
                        fail += len(group)
                        continue

            if not fasta_path.is_file():
                print(f"FASTA not found for {accession}")
                df.loc[group.index, 'Processing_Status'] = 'Failed: FASTA not found'
                fail += len(group)
                continue

            chr_map = build_chromosome_map(fasta_path)
            fasta_obj = Fasta(str(fasta_path), sequence_always_upper=True, rebuild=False)
            genome_cache[accession] = (fasta_obj, chr_map)
            print(f"Loaded FASTA with {len(chr_map)} chromosome mappings")

        fasta_obj, chr_map = genome_cache[accession]

        for idx, row in group.iterrows():
            csv_chr = str(row['Chr']).strip()
            pos = int(row['Position'])
            ref_allele = str(row['Ref Allele']).strip().upper()
            alt_allele = str(row['Alt Allele']).strip().upper()

            if len(ref_allele) != 1 or len(alt_allele) != 1:
                df.loc[idx, 'Processing_Status'] = 'Failed: Not SNV'
                fail += 1
                continue

            pyfaidx_key = resolve_chr(csv_chr, chr_map, fasta_obj)
            if pyfaidx_key is None:
                df.loc[idx, 'Processing_Status'] = f'Failed: Cannot map chr {csv_chr}'
                fail += 1
                continue

            try:
                chrom_len = len(fasta_obj[pyfaidx_key])
                start = max(1, pos - flank)
                end = min(pos + flank - 1, chrom_len)

                seq = fasta_obj.get_seq(pyfaidx_key, start, end).seq

                rel_pos = pos - start
                if rel_pos < 0 or rel_pos >= len(seq):
                    df.loc[idx, 'Processing_Status'] = 'Failed: Position out of bounds'
                    fail += 1
                    continue

                if seq[rel_pos] != ref_allele:
                    df.loc[idx, 'Processing_Status'] = f'Failed: Ref mismatch (exp {ref_allele}, got {seq[rel_pos]})'
                    fail += 1
                    continue

                alt_seq = seq[:rel_pos] + alt_allele + seq[rel_pos + 1:]

                pad_start = (pos - flank) - start if start > (pos - flank) else 0
                pad_end = (pos + flank - 1) - end if end < (pos + flank - 1) else 0
                ref_final = ('N' * pad_start) + seq + ('N' * pad_end)
                alt_final = ('N' * pad_start) + alt_seq + ('N' * pad_end)

                ref_final = (ref_final + 'N' * window)[:window]
                alt_final = (alt_final + 'N' * window)[:window]

                df.loc[idx, 'Ref_Sequence_Window'] = ref_final
                df.loc[idx, 'Alt_Sequence_Window'] = alt_final
                df.loc[idx, 'Processing_Status'] = 'Success'
                success += 1

            except Exception as e:
                df.loc[idx, 'Processing_Status'] = f'Failed: {type(e).__name__}: {e}'
                fail += 1

    for fasta_obj, _ in genome_cache.values():
        fasta_obj.close()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Success: {success}, Failed: {fail}; saved {output_csv}")


if __name__ == '__main__':
    main()
