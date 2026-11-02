#!/usr/bin/env python3
"""Build the 13-class (strand-aware CDS/UTR/intron/splice-site) gene-structure dataset from a genome FASTA and Ensembl GTF."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ATTR_RE = re.compile(r'(\S+)\s+"([^"]*)"')

LABEL_MAP = {
    0: "NC",
    1: "CDS_F", 2: "UTR5_F", 3: "UTR3_F", 4: "DF", 5: "IF", 6: "AF",
    7: "CDS_R", 8: "UTR5_R", 9: "UTR3_R", 10: "DR", 11: "IR", 12: "AR",
}
LABEL_CHARS = "0123456789ABC"
assert len(LABEL_CHARS) == 13

STD_CHROMS = [str(i) for i in range(1, 19)] + ["X", "Y"]


@dataclass
class FaiRecord:
    name: str
    length: int
    offset: int
    line_bases: int
    line_width: int


class IndexedFasta:
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self.fai_path = fasta_path + ".fai"
        if not os.path.exists(self.fai_path):
            raise FileNotFoundError(f"Missing FASTA index: {self.fai_path}; run samtools faidx first")
        self.records: Dict[str, FaiRecord] = {}
        with open(self.fai_path, "rt") as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                name, length, offset, line_bases, line_width = fields[:5]
                self.records[name] = FaiRecord(
                    name=name,
                    length=int(length),
                    offset=int(offset),
                    line_bases=int(line_bases),
                    line_width=int(line_width),
                )
        self._fh = open(fasta_path, "rb")

    def close(self) -> None:
        self._fh.close()

    def fetch(self, chrom: str, start: int, end: int) -> str:
        rec = self.records[chrom]
        if start < 0 or end < start or end > rec.length:
            raise ValueError(f"Invalid interval {chrom}:{start}-{end} for length {rec.length}")
        chunks = []
        pos = start
        remaining = end - start
        while remaining > 0:
            line_no = pos // rec.line_bases
            col = pos % rec.line_bases
            n = min(remaining, rec.line_bases - col)
            file_pos = rec.offset + line_no * rec.line_width + col
            self._fh.seek(file_pos)
            chunks.append(self._fh.read(n).decode("ascii"))
            pos += n
            remaining -= n
        return "".join(chunks).upper()


def parse_attrs(attr_text: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in ATTR_RE.finditer(attr_text)}


@dataclass
class Transcript:
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    gene_name: str = ""
    transcript_biotype: str = ""
    gene_biotype: str = ""
    tx_start: Optional[int] = None
    tx_end: Optional[int] = None
    exons: List[Tuple[int, int]] = field(default_factory=list)
    cds: List[Tuple[int, int]] = field(default_factory=list)
    utr5: List[Tuple[int, int]] = field(default_factory=list)
    utr3: List[Tuple[int, int]] = field(default_factory=list)

    def add_bounds(self, start: int, end: int) -> None:
        if self.tx_start is None or start < self.tx_start:
            self.tx_start = start
        if self.tx_end is None or end > self.tx_end:
            self.tx_end = end

    @property
    def exon_len(self) -> int:
        return sum(e - s for s, e in self.exons)

    @property
    def cds_len(self) -> int:
        return sum(e - s for s, e in self.cds)

    @property
    def span(self) -> Tuple[int, int]:
        if self.exons:
            return min(s for s, _ in self.exons), max(e for _, e in self.exons)
        if self.tx_start is None or self.tx_end is None:
            raise ValueError(f"Transcript {self.transcript_id} has no coordinates")
        return self.tx_start, self.tx_end


@dataclass
class GeneInterval:
    start: int
    end: int
    gene_id: str


def parse_gtf(gtf_path: str, allowed_chroms: set) -> Tuple[Dict[str, Transcript], Dict[str, List[GeneInterval]]]:
    transcripts: Dict[str, Transcript] = {}
    gene_intervals_by_chrom: Dict[str, List[GeneInterval]] = defaultdict(list)
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for raw in fh:
            if not raw or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, source, feature, start_s, end_s, score, strand, frame, attrs_s = fields
            if chrom not in allowed_chroms or strand not in {"+", "-"}:
                continue
            start, end = int(start_s) - 1, int(end_s)  # GTF 1-based inclusive -> 0-based half-open
            attrs = parse_attrs(attrs_s)
            gene_id = attrs.get("gene_id", "")
            if not gene_id:
                continue
            if feature == "gene":
                gene_intervals_by_chrom[chrom].append(GeneInterval(start, end, gene_id))
                continue
            tx_id = attrs.get("transcript_id")
            if not tx_id:
                continue
            tx = transcripts.get(tx_id)
            if tx is None:
                tx = Transcript(
                    transcript_id=tx_id, gene_id=gene_id, chrom=chrom, strand=strand,
                    gene_name=attrs.get("gene_name", ""),
                    transcript_biotype=attrs.get("transcript_biotype", attrs.get("transcript_type", "")),
                    gene_biotype=attrs.get("gene_biotype", attrs.get("gene_type", "")),
                )
                transcripts[tx_id] = tx
            tx.add_bounds(start, end)
            if feature == "transcript":
                tx.transcript_biotype = attrs.get("transcript_biotype", attrs.get("transcript_type", tx.transcript_biotype))
                tx.gene_biotype = attrs.get("gene_biotype", attrs.get("gene_type", tx.gene_biotype))
                tx.gene_name = attrs.get("gene_name", tx.gene_name)
            elif feature == "exon":
                tx.exons.append((start, end))
            elif feature == "CDS":
                tx.cds.append((start, end))
            elif feature == "five_prime_utr":
                tx.utr5.append((start, end))
            elif feature == "three_prime_utr":
                tx.utr3.append((start, end))
    for chrom in gene_intervals_by_chrom:
        gene_intervals_by_chrom[chrom].sort(key=lambda x: (x.start, x.end))
    return transcripts, gene_intervals_by_chrom


def select_representative_transcripts(transcripts: Dict[str, Transcript]) -> Dict[str, Transcript]:
    by_gene: Dict[str, List[Transcript]] = defaultdict(list)
    for tx in transcripts.values():
        if tx.chrom is None or tx.strand not in {"+", "-"}:
            continue
        if tx.transcript_biotype != "protein_coding" or tx.gene_biotype != "protein_coding":
            continue
        if not tx.exons or tx.cds_len <= 0:
            continue
        if tx.cds_len % 3 != 0:
            # incomplete CDS: skip this isoform
            continue
        tx.exons = sorted(set(tx.exons))
        tx.cds = sorted(set(tx.cds))
        tx.utr5 = sorted(set(tx.utr5))
        tx.utr3 = sorted(set(tx.utr3))
        by_gene[tx.gene_id].append(tx)
    selected = {}
    for gene_id, txs in by_gene.items():
        tx = sorted(
            txs,
            key=lambda t: (t.cds_len, t.exon_len, t.span[1] - t.span[0], t.transcript_id),
            reverse=True,
        )[0]
        selected[gene_id] = tx
    return selected


def nearest_gene_limits(chrom_intervals, chrom_len, gene_id, start, end):
    left_limit = 0
    right_limit = chrom_len
    overlaps_other = False
    own_starts, own_ends = [], []
    for iv in chrom_intervals:
        if iv.gene_id == gene_id:
            own_starts.append(iv.start); own_ends.append(iv.end); continue
        if iv.end <= start:
            if iv.end > left_limit:
                left_limit = iv.end
        elif iv.start >= end:
            right_limit = min(right_limit, iv.start); break
        else:
            overlaps_other = True
    if own_starts and min(own_starts) < start:
        left_limit = max(left_limit, start)
    if own_ends and max(own_ends) > end:
        right_limit = min(right_limit, end)
    return left_limit, right_limit, overlaps_other


def choose_window(tx, chrom_len, gene_intervals, max_len, min_len, max_flank, rng):
    """Window = gene span + random flank in [1, max_flank] per side, bounded by neighbouring genes and max_len."""
    tx_start, tx_end = tx.span
    span = tx_end - tx_start
    if span > max_len:
        return None, None, "transcript_span_gt_max_len"
    left_limit, right_limit, overlaps = nearest_gene_limits(gene_intervals, chrom_len, tx.gene_id, tx_start, tx_end)
    if overlaps:
        return None, None, "overlaps_other_gene"
    left_room = max(0, tx_start - left_limit)
    right_room = max(0, right_limit - tx_end)

    cap = max(0, max_len - span)  # total flank budget under the hard length cap
    fl = min(rng.randint(1, max_flank), left_room, cap)
    fr = min(rng.randint(1, max_flank), right_room, cap - fl)

    deficit = min_len - (span + fl + fr)
    if deficit > 0:
        add_l = min(deficit, left_room - fl, cap - fl - fr)
        fl += add_l; deficit -= add_l
        add_r = min(deficit, right_room - fr, cap - fl - fr)
        fr += add_r; deficit -= add_r
        if deficit > 0:
            return None, None, "not_enough_clean_flank_for_min_len"
    return tx_start - fl, tx_end + fr, "ok"


def paint_labels_13(tx: Transcript, sample_start: int, sample_end: int) -> List[int]:
    """Paint 13-class labels onto [sample_start, sample_end): intron, then CDS/UTR per exon, then 2-bp donor/acceptor."""
    n = sample_end - sample_start
    labels = [0] * n
    if tx.strand == "+":
        cds_lab, utr5_lab, utr3_lab = 1, 2, 3
        donor_lab, intron_lab, accept_lab = 4, 5, 6
    else:
        cds_lab, utr5_lab, utr3_lab = 7, 8, 9
        donor_lab, intron_lab, accept_lab = 10, 11, 12

    tx_start, tx_end = tx.span
    for i in range(max(tx_start, sample_start), min(tx_end, sample_end)):
        labels[i - sample_start] = intron_lab

    cds_min = min(s for s, _ in tx.cds)
    cds_max = max(e for _, e in tx.cds)

    use_explicit_utr = bool(tx.utr5) or bool(tx.utr3)
    if use_explicit_utr:
        for es, ee in tx.exons:
            es_c = max(es, sample_start); ee_c = min(ee, sample_end)
            if es_c >= ee_c:
                continue
            ks = max(es_c, cds_min); ke = min(ee_c, cds_max)
            for i in range(ks, ke):
                labels[i - sample_start] = cds_lab
            lks = es_c; lke = min(ee_c, cds_min)
            left_lab = utr5_lab if tx.strand == "+" else utr3_lab
            for i in range(lks, lke):
                labels[i - sample_start] = left_lab
            rks = max(es_c, cds_max); rke = ee_c
            right_lab = utr3_lab if tx.strand == "+" else utr5_lab
            for i in range(rks, rke):
                labels[i - sample_start] = right_lab
        for s, e in tx.utr5:
            for i in range(max(s, sample_start), min(e, sample_end)):
                labels[i - sample_start] = utr5_lab
        for s, e in tx.utr3:
            for i in range(max(s, sample_start), min(e, sample_end)):
                labels[i - sample_start] = utr3_lab
    else:
        for es, ee in tx.exons:
            es_c = max(es, sample_start); ee_c = min(ee, sample_end)
            if es_c >= ee_c:
                continue
            ks = max(es_c, cds_min); ke = min(ee_c, cds_max)
            for i in range(ks, ke):
                labels[i - sample_start] = cds_lab
            left_lab = utr5_lab if tx.strand == "+" else utr3_lab
            for i in range(es_c, min(ee_c, cds_min)):
                labels[i - sample_start] = left_lab
            right_lab = utr3_lab if tx.strand == "+" else utr5_lab
            for i in range(max(es_c, cds_max), ee_c):
                labels[i - sample_start] = right_lab

    exons = sorted(tx.exons)
    for (s1, e1), (s2, e2) in zip(exons, exons[1:]):
        intron_start, intron_end = e1, s2
        if intron_end - intron_start < 4:
            continue
        if tx.strand == "+":
            donor = (intron_start, intron_start + 2)
            acceptor = (intron_end - 2, intron_end)
        else:
            donor = (intron_end - 2, intron_end)
            acceptor = (intron_start, intron_start + 2)
        for i in range(max(donor[0], sample_start), min(donor[1], sample_end)):
            labels[i - sample_start] = donor_lab
        for i in range(max(acceptor[0], sample_start), min(acceptor[1], sample_end)):
            labels[i - sample_start] = accept_lab

    return labels


def splice_motifs(tx: Transcript, fasta: IndexedFasta) -> Counter:
    motifs = Counter()
    exons = sorted(tx.exons)
    for (s1, e1), (s2, e2) in zip(exons, exons[1:]):
        intron_start, intron_end = e1, s2
        if intron_end - intron_start < 4:
            motifs["short_intron"] += 1
            continue
        if tx.strand == "+":
            d = fasta.fetch(tx.chrom, intron_start, intron_start + 2)
            a = fasta.fetch(tx.chrom, intron_end - 2, intron_end)
        else:
            d = revcomp(fasta.fetch(tx.chrom, intron_end - 2, intron_end))
            a = revcomp(fasta.fetch(tx.chrom, intron_start, intron_start + 2))
        motifs[f"{d}-{a}"] += 1
    return motifs


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1].upper()


def split_for_chrom(chrom, valid_chroms, test_chroms):
    if chrom in valid_chroms: return "valid"
    if chrom in test_chroms: return "test"
    return "train"


def write_record(out_fh, rec):
    out_fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


def percentile(xs, p):
    if not xs: return None
    ys = sorted(xs); k = (len(ys) - 1) * p
    lo = int(k); hi = min(lo + 1, len(ys) - 1); frac = k - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def file_info(path):
    st = os.stat(path)
    return {"path": path, "size_bytes": st.st_size, "mtime": st.st_mtime}


def encode_labels(labels: List[int]) -> str:
    return "".join(LABEL_CHARS[x] for x in labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=14000)
    ap.add_argument("--min-len", type=int, default=1433)
    ap.add_argument("--max-flank", type=int, default=1000,
                    help="random flank per side (1..max_flank nt)")
    ap.add_argument("--seed", type=int, default=20260519)
    ap.add_argument("--chroms", default=",".join(STD_CHROMS))
    ap.add_argument("--valid-chroms", default="17")
    ap.add_argument("--test-chroms", default="18,X")
    ap.add_argument("--split-mode", choices=["chrom", "graphpart"], default="chrom",
                    help="chrom: split by chromosome; graphpart: use a gene_id->split map from split_graphpart.py")
    ap.add_argument("--split-map", default=None,
                    help="TSV with header 'gene_id<TAB>split[...]' (required for graphpart)")
    ap.add_argument("--max-n-frac", type=float, default=0.001)
    ap.add_argument("--require-utr", action="store_true",
                    help="only keep transcripts with both 5'UTR and 3'UTR annotated")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    allowed_chroms = {x for x in args.chroms.split(",") if x}
    valid_chroms = {x for x in args.valid_chroms.split(",") if x}
    test_chroms = {x for x in args.test_chroms.split(",") if x}
    if valid_chroms & test_chroms:
        raise ValueError("valid and test chromosomes overlap")

    split_map = None
    if args.split_mode == "graphpart":
        if not args.split_map or not os.path.exists(args.split_map):
            raise ValueError("--split-mode graphpart requires an existing --split-map TSV")
        split_map = {}
        with open(args.split_map, "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            gi = header.index("gene_id"); si = header.index("split")
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) > max(gi, si):
                    split_map[parts[gi]] = parts[si]

    rng = random.Random(args.seed)
    fasta = IndexedFasta(args.fasta)
    chrom_lengths = {c: fasta.records[c].length for c in allowed_chroms if c in fasta.records}
    missing = allowed_chroms - set(chrom_lengths)
    if missing:
        raise ValueError(f"Chromosomes missing from FASTA: {sorted(missing)}")

    transcripts, gene_intervals = parse_gtf(args.gtf, allowed_chroms)
    selected = select_representative_transcripts(transcripts)

    n_sel = len(selected)
    n_has_5 = sum(1 for tx in selected.values() if tx.utr5)
    n_has_3 = sum(1 for tx in selected.values() if tx.utr3)
    n_has_both = sum(1 for tx in selected.values() if tx.utr5 and tx.utr3)
    n_none = sum(1 for tx in selected.values() if not tx.utr5 and not tx.utr3)

    out_paths = {s: os.path.join(args.outdir, f"{s}.jsonl.gz") for s in ["train", "valid", "test"]}
    out_fhs = {s: gzip.open(p, "wt") for s, p in out_paths.items()}

    stats = {
        "config": vars(args),
        "label_map": LABEL_MAP,
        "input_files": {"fasta": file_info(args.fasta), "gtf": file_info(args.gtf)},
        "counts": Counter(),
        "skipped": Counter(),
        "split_counts": Counter(),
        "split_lengths": defaultdict(list),
        "split_label_base_counts": {s: Counter() for s in ["train", "valid", "test"]},
        "splice_motifs": Counter(),
        "chrom_sample_counts": Counter(),
        "chrom_gene_counts_selected": Counter(),
        "utr_coverage_selected_reps": {
            "n_selected": n_sel,
            "with_5utr": n_has_5,
            "with_3utr": n_has_3,
            "with_both": n_has_both,
            "with_neither": n_none,
            "pct_with_5utr": round(100 * n_has_5 / max(1, n_sel), 2),
            "pct_with_3utr": round(100 * n_has_3 / max(1, n_sel), 2),
            "pct_with_both": round(100 * n_has_both / max(1, n_sel), 2),
        },
        "transcripts_used_explicit_utr": 0,
        "transcripts_used_computed_utr": 0,
    }

    manifest_path = os.path.join(args.outdir, "manifest.tsv")
    with open(manifest_path, "wt") as manifest:
        manifest.write("sample_id\tsplit\tchrom\tstart0\tend0\tstrand\tgene_id\ttranscript_id\tgene_name\tlength\tn_frac\thas_5utr\thas_3utr\n")
        for gene_id in sorted(selected):
            tx = selected[gene_id]
            stats["counts"]["selected_representative_transcripts"] += 1
            stats["chrom_gene_counts_selected"][tx.chrom] += 1
            if tx.chrom not in chrom_lengths:
                stats["skipped"]["chrom_not_allowed"] += 1
                continue
            if args.require_utr and not (tx.utr5 and tx.utr3):
                stats["skipped"]["require_utr_filtered"] += 1
                continue
            start, end, reason = choose_window(
                tx, chrom_lengths[tx.chrom], gene_intervals.get(tx.chrom, []),
                args.max_len, args.min_len, args.max_flank, rng,
            )
            if reason != "ok":
                stats["skipped"][reason] += 1
                continue
            seq = fasta.fetch(tx.chrom, start, end)
            if len(seq) != end - start:
                stats["skipped"]["fasta_fetch_length_mismatch"] += 1
                continue
            n_frac = seq.count("N") / len(seq)
            if n_frac > args.max_n_frac:
                stats["skipped"]["n_fraction_gt_threshold"] += 1
                continue
            labels = paint_labels_13(tx, start, end)
            if len(labels) != len(seq):
                stats["skipped"]["label_length_mismatch"] += 1
                continue
            if tx.utr5 or tx.utr3:
                stats["transcripts_used_explicit_utr"] += 1
            else:
                stats["transcripts_used_computed_utr"] += 1
            if split_map is not None:
                split = split_map.get(tx.gene_id)
                if split is None:
                    stats["skipped"]["not_in_split_map"] += 1
                    continue
            else:
                split = split_for_chrom(tx.chrom, valid_chroms, test_chroms)
            sample_id = f"pig_bend13_{split}_{stats['split_counts'][split]:06d}"
            labels_s = encode_labels(labels)
            rec = {
                "id": sample_id, "split": split, "chrom": tx.chrom,
                "start0": start, "end0": end, "strand": tx.strand,
                "gene_id": tx.gene_id, "transcript_id": tx.transcript_id, "gene_name": tx.gene_name,
                "sequence": seq, "labels": labels_s, "schema": "bend13",
            }
            write_record(out_fhs[split], rec)
            manifest.write(
                f"{sample_id}\t{split}\t{tx.chrom}\t{start}\t{end}\t{tx.strand}\t"
                f"{tx.gene_id}\t{tx.transcript_id}\t{tx.gene_name}\t{len(seq)}\t{n_frac:.6g}\t"
                f"{int(bool(tx.utr5))}\t{int(bool(tx.utr3))}\n"
            )
            stats["split_counts"][split] += 1
            stats["chrom_sample_counts"][tx.chrom] += 1
            stats["split_lengths"][split].append(len(seq))
            stats["split_label_base_counts"][split].update(labels_s)
            stats["splice_motifs"].update(splice_motifs(tx, fasta))

    for fh in out_fhs.values(): fh.close()
    fasta.close()

    summary = {
        "config": stats["config"],
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "label_chars": LABEL_CHARS,
        "input_files": stats["input_files"],
        "counts": dict(stats["counts"]),
        "skipped": dict(stats["skipped"]),
        "split_counts": dict(stats["split_counts"]),
        "chrom_sample_counts": dict(stats["chrom_sample_counts"]),
        "chrom_gene_counts_selected": dict(stats["chrom_gene_counts_selected"]),
        "splice_motifs_top50": dict(stats["splice_motifs"].most_common(50)),
        "length_stats": {},
        "label_base_counts": {},
        "outputs": {**out_paths, "manifest": manifest_path},
        "utr_coverage_selected_reps": stats["utr_coverage_selected_reps"],
        "transcripts_used_explicit_utr": stats["transcripts_used_explicit_utr"],
        "transcripts_used_computed_utr": stats["transcripts_used_computed_utr"],
    }
    for split, lengths in stats["split_lengths"].items():
        summary["length_stats"][split] = {
            "n": len(lengths), "min": min(lengths) if lengths else None,
            "p25": percentile(lengths, 0.25),
            "median": statistics.median(lengths) if lengths else None,
            "mean": statistics.mean(lengths) if lengths else None,
            "p75": percentile(lengths, 0.75),
            "max": max(lengths) if lengths else None,
        }
    for split, counts in stats["split_label_base_counts"].items():
        summary["label_base_counts"][split] = {
            LABEL_MAP[LABEL_CHARS.index(c)]: int(v)
            for c, v in sorted(counts.items(), key=lambda kv: LABEL_CHARS.index(kv[0]))
        }

    with open(os.path.join(args.outdir, "summary.json"), "wt") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.outdir, "label_map.json"), "wt") as fh:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, fh, indent=2)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
