#!/usr/bin/env python3
"""GraphPart homology-reduced train/valid/test split (80% protein identity, as in BEND) for a built dataset."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict

EXONIC_CHARS = set("123789")  # CDS + UTR label chars (both strands)
CDS_CHARS = set("17")         # CDS label chars (both strands)
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Standard genetic code (NCBI table 1).
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1].upper()


def translate(dna: str) -> str:
    """Translate frame-0 DNA; in-frame TGA -> 'U' (Sec), other internal stops -> 'X', terminal stop dropped."""
    aa = []
    n = len(dna) // 3
    for i in range(n):
        codon = dna[i * 3:i * 3 + 3]
        a = CODON_TABLE.get(codon, "X")
        if a == "*":
            if i == n - 1:
                break  # terminal stop codon: drop it
            aa.append("U" if codon == "TGA" else "X")  # in-frame TGA = Sec
            continue
        aa.append(a)
    return "".join(aa)


def gene_sequence(sequence: str, labels: str, strand: str, level: str) -> str:
    """Homology unit: translated CDS (level='protein') or spliced mRNA (level='mrna_nt')."""
    chars = CDS_CHARS if level == "protein" else EXONIC_CHARS
    sub = "".join(b for b, lab in zip(sequence, labels) if lab in chars)
    if strand == "-":
        sub = revcomp(sub)
    sub = sub.upper()
    return translate(sub) if level == "protein" else sub


def collect_genes(data_dir: Path, splits, level: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    dup = 0
    for split in splits:
        path = data_dir / f"{split}.jsonl.gz"
        with gzip.open(path, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                gid = rec["gene_id"]
                seq = gene_sequence(rec["sequence"], rec["labels"], rec["strand"], level)
                if not seq:
                    continue
                if gid in out:
                    dup += 1
                    continue
                out[gid] = seq
    if dup:
        print(json.dumps({"event": "warn", "msg": "duplicate gene_id across splits",
                          "count": dup}), flush=True)
    return out


def write_fasta(genes: Dict[str, str], out_fasta: Path) -> int:
    with open(out_fasta, "wt") as fh:
        for gene_id in sorted(genes):
            seq = genes[gene_id]
            fh.write(f">{gene_id}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")
    return len(genes)


def run_graphpart(fasta_file: Path, out_csv: Path, args) -> None:
    env = dict(os.environ)
    if args.mmseqs_dir:
        env["PATH"] = f"{args.mmseqs_dir}:{env.get('PATH', '')}"
    cmd = [
        "graphpart", "mmseqs2",
        "--fasta-file", str(fasta_file),
        "--threshold", str(args.threshold),
        "--transformation", "one-minus",
        "--test-ratio", str(args.test_ratio),
        "--val-ratio", str(args.val_ratio),
        "--denominator", args.denominator,
        "--out-file", str(out_csv),
    ]
    if args.seq_level == "mrna_nt":
        cmd.append("--nucleotide")
    print(json.dumps({"event": "graphpart_cmd", "cmd": cmd}), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_partitions(csv_path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(csv_path, "rt") as fh:
        reader = csv.DictReader(fh)
        fields = [f.strip() for f in (reader.fieldnames or [])]
        ac_col = next((c for c in fields if c.strip().lower() in ("ac", "accession", "id")), fields[0])
        part_col = next((c for c in fields if c.strip().lower() in ("cluster", "partition")), fields[-1])
        for row in reader:
            ac = row[ac_col].strip()
            out[ac] = int(float(row[part_col]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="built dataset dir defining the gene set")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--seq-level", choices=["protein", "mrna_nt"], default="protein",
                    help="homology unit: translated CDS or spliced mRNA")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="max cross-partition identity")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--denominator", default="shortest",
                    choices=["shortest", "longest", "n_aligned"])
    ap.add_argument("--mmseqs-dir", default=None,
                    help="dir containing the mmseqs binary to prepend to PATH")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = collect_genes(Path(args.data_dir), args.splits, args.seq_level)
    print(json.dumps({"event": "genes_collected", "n": len(genes),
                      "seq_level": args.seq_level}), flush=True)

    fasta_name = "proteins.fasta" if args.seq_level == "protein" else "transcripts.fasta"
    fasta_file = outdir / fasta_name
    n = write_fasta(genes, fasta_file)
    print(json.dumps({"event": "fasta_written", "n_sequences": n, "path": str(fasta_file)}), flush=True)

    part_csv = outdir / "graphpart_partitions.csv"
    run_graphpart(fasta_file, part_csv, args)

    partitions = parse_partitions(part_csv)
    sizes = Counter(partitions.values())
    ordered = [p for p, _ in sizes.most_common()]
    role = {}
    if ordered:
        role[ordered[0]] = "train"
    for p in ordered[1:]:
        role[p] = "valid" if list(role.values()).count("valid") == 0 else "test"

    split_map = {g: role.get(p, "train") for g, p in partitions.items()}
    counts = Counter(split_map.values())

    map_path = outdir / "split_map.tsv"
    with open(map_path, "wt") as fh:
        fh.write("gene_id\tsplit\tpartition\n")
        for g in sorted(split_map):
            fh.write(f"{g}\t{split_map[g]}\t{partitions[g]}\n")

    summary = {
        "seq_level": args.seq_level,
        "n_genes_dataset": len(genes),
        "n_genes_partitioned": len(partitions),
        "n_genes_unassigned": len(genes) - len(partitions),
        "threshold": args.threshold,
        "denominator": args.denominator,
        "partition_sizes": {str(k): v for k, v in sizes.items()},
        "partition_to_split": {str(k): v for k, v in role.items()},
        "split_counts": dict(counts),
        "outputs": {"fasta": str(fasta_file), "partitions": str(part_csv),
                    "split_map": str(map_path)},
    }
    with open(outdir / "graphpart_summary.json", "wt") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps({"event": "done", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
