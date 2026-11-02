#!/usr/bin/env python3
"""Extract ref/alt windows (default 6,000 bp) around OMIA SNVs from NCBI genome archives."""
import argparse
import gzip
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta, FastaIndexingError

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


def build_chromosome_map_from_fasta_headers(fasta_path):
    """Map chromosome names parsed from FASTA headers to the sequence IDs pyfaidx uses."""
    chr_map = {}
    if not fasta_path.is_file():
        return None
    open_func = gzip.open if str(fasta_path).endswith('.gz') else open
    processed_ids = set()
    with open_func(fasta_path, 'rt', encoding='latin-1', errors='ignore') as f_fasta:
        for line in f_fasta:
            if not line.startswith('>'):
                continue
            parts = line[1:].split(None, 1)
            if not parts:
                continue
            seq_id = parts[0]
            if seq_id in processed_ids:
                continue
            processed_ids.add(seq_id)
            header_description = parts[1] if len(parts) > 1 else ""

            match = re.search(r'[Cc]hr(?:omosome)?\.?\s+([\w.-]+)', header_description)
            human_chr = match.group(1).rstrip(',.;') if match else None
            if human_chr is None and re.match(r'^chr[\w.-]+$|^[0-9XYMTZW]+$', seq_id, re.I):
                human_chr = seq_id

            chr_map.setdefault(seq_id, seq_id)
            if not human_chr:
                continue
            if human_chr in chr_map and chr_map[human_chr] != seq_id:
                print(f"Warning: '{human_chr}' maps to both '{chr_map[human_chr]}' and '{seq_id}'; keeping first.",
                      file=sys.stderr)
            elif human_chr not in chr_map:
                chr_map[human_chr] = seq_id
            if human_chr.isdigit():
                chr_map.setdefault(f"chr{human_chr}", seq_id)
            elif human_chr.lower().startswith('chr'):
                chr_map.setdefault(human_chr.replace('chr', '', 1), seq_id)
            if human_chr.upper() == "MT":
                chr_map.setdefault("Mito", seq_id)
            elif human_chr.upper() == "MITO":
                chr_map.setdefault("MT", seq_id)
    return chr_map or None


def resolve_chr(csv_chr, chr_map, fasta_obj):
    key = chr_map.get(csv_chr)
    if key is not None:
        return key
    if csv_chr.startswith('chr') and csv_chr.replace('chr', '', 1) in chr_map:
        return chr_map[csv_chr.replace('chr', '', 1)]
    if not csv_chr.startswith('chr') and f"chr{csv_chr}" in chr_map:
        return chr_map[f"chr{csv_chr}"]
    if csv_chr == "Mito" and "MT" in chr_map:
        return chr_map["MT"]
    if csv_chr == "MT" and "Mito" in chr_map:
        return chr_map["Mito"]
    if csv_chr in fasta_obj:
        return csv_chr
    return None


def prepare_genome(accession_key, genome_dir, temp_dir):
    """Locate the GCF/GCA zip, extract its FASTA once and return (Fasta, chr_map) or (None, reason)."""
    accession = accession_key
    zip_path = genome_dir / f"{accession}.zip"
    if not zip_path.is_file():
        zip_path = genome_dir / f"{accession.replace('GCF', 'GCA')}.zip"
        accession = accession.replace('GCF', 'GCA')
    if not zip_path.is_file():
        return None, 'Failed: Genome ZIP not found'

    fasta_path = temp_dir / f"{accession}.fna"
    if not fasta_path.is_file():
        assembly_name = "UnknownAssembly"
        for name, acc in GENOME_NAME_TO_ACCESSION.items():
            if acc == accession_key or acc.replace('GCF', 'GCA') == accession_key:
                assembly_name = name.replace(" ", "_")
                break
        internal = f"ncbi_dataset/data/{accession}/{accession}_{assembly_name}_genomic.fna"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if internal not in zip_ref.namelist():
                    return None, 'Failed: FASTA path not in ZIP'
                zip_ref.extract(internal, temp_dir)
                (temp_dir / internal).rename(fasta_path)
        except Exception as e:
            print(f"Error extracting from {zip_path}: {e}", file=sys.stderr)
            return None, 'Failed: ZIP Extraction Error'

    chr_map = build_chromosome_map_from_fasta_headers(fasta_path)
    if chr_map is None:
        return None, 'Failed: Chromosome Map Build Error'
    try:
        fasta_obj = Fasta(str(fasta_path), sequence_always_upper=True, rebuild=False, read_ahead=None)
    except FastaIndexingError:
        return None, 'Failed: pyfaidx Indexing Error'
    except Exception:
        return None, 'Failed: pyfaidx Init Error'
    return (fasta_obj, chr_map), None


def extract_window(row, fasta_obj, chr_map, window):
    """Return (ref_window, alt_window, status) for one variant."""
    csv_chr = str(row['Chr']).strip()
    pos = int(row['Position'])
    ref_allele = str(row['Ref Allele']).strip().upper()
    alt_allele = str(row['Alt Allele']).strip().upper()
    if len(ref_allele) != 1 or len(alt_allele) != 1:
        return None, None, 'Failed: Not single base SNV'

    key = resolve_chr(csv_chr, chr_map, fasta_obj)
    if key is None:
        return None, None, f"Failed: Cannot map chromosome '{csv_chr}'"

    flank = window // 2
    start, end = pos - flank, pos + flank - 1  # variant at 0-based index `flank`
    try:
        chrom_len = len(fasta_obj[key])
        fetch_start, fetch_end = max(1, start), min(end, chrom_len)
        if fetch_start > fetch_end:
            return None, None, 'Failed: Position outside chr bounds'
        sequence = fasta_obj.get_seq(name=key, start=fetch_start, end=fetch_end).seq
    except (KeyError, ValueError, IndexError, FastaIndexingError):
        return None, None, 'Failed: pyfaidx extract error'
    if len(sequence) == 0:
        return None, None, 'Failed: Extracted empty sequence'

    rel = pos - fetch_start
    if not (0 <= rel < len(sequence)):
        return None, None, 'Failed: Position outside extracted bounds'
    if sequence[rel] != ref_allele:
        return None, None, f"Failed: Ref Mismatch (Exp {ref_allele}, Got {sequence[rel]})"

    alt_sequence = sequence[:rel] + alt_allele + sequence[rel + 1:]
    pad_start, pad_end = 'N' * (fetch_start - start), 'N' * (end - fetch_end)
    ref_final = (pad_start + sequence + pad_end + 'N' * window)[:window]
    alt_final = (pad_start + alt_sequence + pad_end + 'N' * window)[:window]
    return ref_final, alt_final, 'Success'


def main():
    parser = argparse.ArgumentParser(description="Extract ref/alt sequence windows around OMIA SNVs")
    parser.add_argument("--variants", default="data/omia_extracted_snvs.csv")
    parser.add_argument("--genome-dir", default="data/genomes", help="directory with NCBI GCF_*.zip archives")
    parser.add_argument("--temp-dir", default="data/genomes/fasta", help="where extracted FASTA files are kept")
    parser.add_argument("--output", default="data/omia_snv_sequences.csv")
    parser.add_argument("--window", type=int, default=6000, help="total window length in bp")
    args = parser.parse_args()

    genome_dir, temp_dir = Path(args.genome_dir), Path(args.temp_dir)
    if not genome_dir.is_dir():
        sys.exit(f"Genome directory not found: {genome_dir}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    variants_df = pd.read_csv(args.variants, dtype={'Chr': str})
    variants_df['Position'] = pd.to_numeric(variants_df['Position'], errors='coerce').astype('Int64')
    variants_df.dropna(subset=['Position'], inplace=True)
    variants_df['Ref Allele'] = variants_df['Ref Allele'].astype(str)
    variants_df['Alt Allele'] = variants_df['Alt Allele'].astype(str)
    print(f"Loaded {len(variants_df)} variants")

    variants_df['Ref_Sequence_Window'] = pd.NA
    variants_df['Alt_Sequence_Window'] = pd.NA
    variants_df['Processing_Status'] = 'Pending'
    variants_df['NCBI_Accession'] = variants_df['Reference Sequence'].str.strip().map(GENOME_NAME_TO_ACCESSION)
    unmapped = variants_df['NCBI_Accession'].isna()
    if unmapped.any():
        print(f"Unmapped assemblies: {variants_df.loc[unmapped, 'Reference Sequence'].unique()}", file=sys.stderr)
        variants_df.loc[unmapped, 'Processing_Status'] = 'Failed: Cannot map Reference Sequence to Accession'

    genomes = []
    for accession, group_df in variants_df[~unmapped].groupby('NCBI_Accession'):
        print(f"{accession}: {len(group_df)} variants")
        prepared, error = prepare_genome(accession, genome_dir, temp_dir)
        if prepared is None:
            print(f"  {error}", file=sys.stderr)
            variants_df.loc[group_df.index, 'Processing_Status'] = error
            continue
        fasta_obj, chr_map = prepared
        genomes.append(fasta_obj)
        for index, row in group_df.iterrows():
            try:
                ref_seq, alt_seq, status = extract_window(row, fasta_obj, chr_map, args.window)
            except ValueError:
                ref_seq, alt_seq, status = None, None, 'Failed: Data format/ValueError'
            except KeyError:
                ref_seq, alt_seq, status = None, None, 'Failed: Missing Column/KeyError'
            except Exception as e:
                print(f"Error at row {index}: {e}", file=sys.stderr)
                ref_seq, alt_seq, status = None, None, 'Failed: Unexpected Python Error'
            if ref_seq is not None:
                variants_df.loc[index, 'Ref_Sequence_Window'] = ref_seq
                variants_df.loc[index, 'Alt_Sequence_Window'] = alt_seq
            variants_df.loc[index, 'Processing_Status'] = status

    for fasta_obj in genomes:
        fasta_obj.close()

    original_cols = list(pd.read_csv(args.variants, nrows=0).columns)
    output_columns = original_cols + ['NCBI_Accession', 'Ref_Sequence_Window', 'Alt_Sequence_Window', 'Processing_Status']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    variants_df[[c for c in output_columns if c in variants_df.columns]].to_csv(args.output, index=False, na_rep='NA')

    status = variants_df['Processing_Status']
    print(f"Success: {(status == 'Success').sum()}; failed: {status.str.startswith('Failed', na=False).sum()}")
    failures = status[status.str.startswith('Failed', na=False)].value_counts()
    if not failures.empty:
        print(failures.to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
