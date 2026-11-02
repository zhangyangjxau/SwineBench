#!/usr/bin/env python3
"""Score AUGUSTUS GFF predictions against BEND-13 labels (13-class, 6-class and 4-class MCC)."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

LABEL_CHARS = "0123456789ABC"
LABEL_NAMES = [
    "NC",
    "CDS_F", "UTR5_F", "UTR3_F", "DF", "IF", "AF",
    "CDS_R", "UTR5_R", "UTR3_R", "DR", "IR", "AR",
]
NC, CDS_F, UTR5_F, UTR3_F, DF, IF, AF, CDS_R, UTR5_R, UTR3_R, DR, IR, AR = range(13)

EXON_FEATURES = {
    "exon", "initial", "internal", "terminal", "single",
    "five_prime_utr", "three_prime_utr", "5'-UTR", "3'-UTR", "UTR",
}
CDS_FEATURES = {"CDS"}
UTR5_FEATURES = {"five_prime_utr", "5'-UTR"}
UTR3_FEATURES = {"three_prime_utr", "3'-UTR"}

COLLAPSE_6 = {
    "CDS": [CDS_F, CDS_R],
    "5UTR": [UTR5_F, UTR5_R],
    "3UTR": [UTR3_F, UTR3_R],
    "Donor": [DF, DR],
    "Intron": [IF, IR],
    "Acceptor": [AF, AR],
}
COLLAPSE_4_BENDLIKE = {
    "Exon": [CDS_F, UTR5_F, UTR3_F, CDS_R, UTR5_R, UTR3_R],
    "Intron": [IF, IR],
    "Donor": [DF, DR],
    "Acceptor": [AF, AR],
}


def decode_labels(s: str) -> np.ndarray:
    return np.fromiter((LABEL_CHARS.index(c) for c in s), dtype=np.int16)


def read_truth(path: Path) -> Dict[str, Dict[str, object]]:
    truth = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            labels = decode_labels(rec["labels"])
            if len(labels) != len(rec["sequence"]):
                raise ValueError(f"{rec['id']}: label/sequence length mismatch")
            truth[rec["id"]] = {
                "labels": labels,
                "length": len(labels),
                "meta": {k: rec.get(k) for k in [
                    "id", "split", "chrom", "start0", "end0",
                    "strand", "gene_id", "transcript_id", "gene_name",
                ]},
            }
    return truth


def parse_attrs(s: str) -> Dict[str, str]:
    attrs = {}
    for key, val in re.findall(r'(\S+)\s+"([^"]+)"', s):
        attrs[key] = val
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, val = part.split("=", 1)
            attrs[key.strip()] = val.strip()
    return attrs


def tx_id_for(feature: str, attrs: Dict[str, str], seqid: str, start: int, end: int, strand: str) -> str:
    for key in ("transcript_id", "transcriptId", "Parent", "ID"):
        val = attrs.get(key)
        if not val:
            continue
        if feature == "gene" and key == "ID":
            continue
        return val.split(",")[0]
    gene = attrs.get("gene_id") or attrs.get("geneId") or attrs.get("gene") or "gene"
    return f"{seqid}:{gene}:{strand}:{start}-{end}"


def parse_gff(path: Path):
    tx = defaultdict(lambda: {
        "seqid": None,
        "strand": "+",
        "exons": [],
        "cds": [],
        "utr5": [],
        "utr3": [],
    })
    feature_counts = Counter()
    with open(path) as fh:
        for raw in fh:
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            seqid, source, feature, start_s, end_s, score, strand, frame, attrs_s = fields
            feature_counts[feature] += 1
            wanted = feature in EXON_FEATURES or feature in CDS_FEATURES
            if not wanted:
                continue
            start, end = int(start_s) - 1, int(end_s)
            attrs = parse_attrs(attrs_s)
            tid = tx_id_for(feature, attrs, seqid, start, end, strand)
            rec = tx[tid]
            rec["seqid"] = seqid
            rec["strand"] = strand if strand in "+-" else "+"
            if feature in CDS_FEATURES:
                rec["cds"].append((start, end))
                rec["exons"].append((start, end))
            elif feature in UTR5_FEATURES:
                rec["utr5"].append((start, end))
                rec["exons"].append((start, end))
            elif feature in UTR3_FEATURES:
                rec["utr3"].append((start, end))
                rec["exons"].append((start, end))
            else:
                rec["exons"].append((start, end))
    return tx, feature_counts


def merge_intervals(xs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ys = sorted((max(0, s), max(0, e)) for s, e in xs if e > s)
    out: List[List[int]] = []
    for s, e in ys:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(s, e) for s, e in out]


def clip_interval(s: int, e: int, n: int) -> Tuple[int, int]:
    return max(0, min(n, s)), max(0, min(n, e))


def add_op(ops: List[Tuple[int, int, int]], s: int, e: int, lbl: int, n: int) -> None:
    s, e = clip_interval(s, e, n)
    if e > s:
        ops.append((s, e, lbl))


def paint_one(length: int, transcripts: List[Dict[str, object]]) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int16)
    intron_ops: List[Tuple[int, int, int]] = []
    exon_ops: List[Tuple[int, int, int]] = []
    splice_ops: List[Tuple[int, int, int]] = []

    for tr in transcripts:
        exons = merge_intervals(tr["exons"])
        if not exons:
            continue
        cds = merge_intervals(tr["cds"])
        utr5 = merge_intervals(tr["utr5"])
        utr3 = merge_intervals(tr["utr3"])
        strand = tr["strand"]
        if strand == "+":
            cds_lab, utr5_lab, utr3_lab = CDS_F, UTR5_F, UTR3_F
            donor_lab, intron_lab, acceptor_lab = DF, IF, AF
        else:
            cds_lab, utr5_lab, utr3_lab = CDS_R, UTR5_R, UTR3_R
            donor_lab, intron_lab, acceptor_lab = DR, IR, AR

        tx_start, tx_end = exons[0][0], exons[-1][1]
        add_op(intron_ops, tx_start, tx_end, intron_lab, length)

        if cds:
            cds_min = min(s for s, _ in cds)
            cds_max = max(e for _, e in cds)
            for es, ee in exons:
                left_lab = utr5_lab if strand == "+" else utr3_lab
                right_lab = utr3_lab if strand == "+" else utr5_lab
                add_op(exon_ops, es, min(ee, cds_min), left_lab, length)
                add_op(exon_ops, max(es, cds_min), min(ee, cds_max), cds_lab, length)
                add_op(exon_ops, max(es, cds_max), ee, right_lab, length)
            for s, e in cds:
                add_op(exon_ops, s, e, cds_lab, length)
        else:
            for s, e in exons:
                add_op(exon_ops, s, e, cds_lab, length)

        for s, e in utr5:
            add_op(exon_ops, s, e, utr5_lab, length)
        for s, e in utr3:
            add_op(exon_ops, s, e, utr3_lab, length)

        for (s1, e1), (s2, e2) in zip(exons, exons[1:]):
            intron_start, intron_end = e1, s2
            if intron_end - intron_start < 4:
                continue
            if strand == "+":
                donor = (intron_start, intron_start + 2, donor_lab)
                acceptor = (intron_end - 2, intron_end, acceptor_lab)
            else:
                donor = (intron_end - 2, intron_end, donor_lab)
                acceptor = (intron_start, intron_start + 2, acceptor_lab)
            add_op(splice_ops, *donor, length)
            add_op(splice_ops, *acceptor, length)

    for s, e, lbl in intron_ops + exon_ops + splice_ops:
        labels[s:e] = lbl
    return labels


def per_class_mcc(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, float]:
    out = {}
    for i, name in enumerate(names):
        t = (y_true == i).astype(np.int8)
        p = (y_pred == i).astype(np.int8)
        if t.sum() == 0 and p.sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(matthews_corrcoef(t, p))
    return out


def collapse(y: np.ndarray, mapping: Dict[str, List[int]]) -> Tuple[np.ndarray, List[str]]:
    names = list(mapping.keys())
    out = np.zeros_like(y, dtype=np.int16)
    for i, (_, classes) in enumerate(mapping.items(), start=1):
        out[np.isin(y, classes)] = i
    return out, ["background"] + names


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, object]:
    labels = list(range(len(names)))
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_mcc": per_class_mcc(y_true, y_pred, names),
        "precision_per_class": dict(zip(names, precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist())),
        "recall_per_class": dict(zip(names, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist())),
        "true_counts": dict(zip(names, np.bincount(y_true, minlength=len(names)).astype(int).tolist())),
        "pred_counts": dict(zip(names, np.bincount(y_pred, minlength=len(names)).astype(int).tolist())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-jsonl", required=True, type=Path)
    ap.add_argument("--augustus-gff", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    truth = read_truth(args.truth_jsonl)
    tx, feature_counts = parse_gff(args.augustus_gff)
    by_seq = defaultdict(list)
    for rec in tx.values():
        if rec["seqid"] in truth:
            by_seq[rec["seqid"]].append(rec)

    y_true_parts = []
    y_pred_parts = []
    per_seq = []
    for seqid, item in truth.items():
        y_true = item["labels"]
        y_pred = paint_one(item["length"], by_seq.get(seqid, []))
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)
        mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(np.r_[y_true, y_pred])) > 1 else 0.0
        per_seq.append({
            "id": seqid,
            "n_pred_transcripts": len(by_seq.get(seqid, [])),
            "mcc": mcc,
        })

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    y_true6, names6 = collapse(y_true, COLLAPSE_6)
    y_pred6, _ = collapse(y_pred, COLLAPSE_6)
    y_true4, names4 = collapse(y_true, COLLAPSE_4_BENDLIKE)
    y_pred4, _ = collapse(y_pred, COLLAPSE_4_BENDLIKE)

    out = {
        "truth_jsonl": str(args.truth_jsonl),
        "augustus_gff": str(args.augustus_gff),
        "n_sequences": len(truth),
        "n_labeled_bp": int(y_true.size),
        "n_sequences_with_prediction": int(sum(1 for x in per_seq if x["n_pred_transcripts"] > 0)),
        "augustus_feature_counts": dict(feature_counts),
        "overall_mcc_13class": float(matthews_corrcoef(y_true, y_pred)),
        "overall_mcc_6class_collapsed": float(matthews_corrcoef(y_true6, y_pred6)),
        "overall_mcc_4class_bendlike": float(matthews_corrcoef(y_true4, y_pred4)),
        "metrics_13class": metrics_for(y_true, y_pred, LABEL_NAMES),
        "metrics_6class_collapsed": metrics_for(y_true6, y_pred6, names6),
        "metrics_4class_bendlike": metrics_for(y_true4, y_pred4, names4),
        "per_sequence_mcc_mean": float(np.mean([x["mcc"] for x in per_seq])),
        "per_sequence_mcc_median": float(np.median([x["mcc"] for x in per_seq])),
        "per_sequence": per_seq,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps({
        "n_sequences": out["n_sequences"],
        "n_labeled_bp": out["n_labeled_bp"],
        "n_sequences_with_prediction": out["n_sequences_with_prediction"],
        "overall_mcc_13class": out["overall_mcc_13class"],
        "overall_mcc_6class_collapsed": out["overall_mcc_6class_collapsed"],
        "overall_mcc_4class_bendlike": out["overall_mcc_4class_bendlike"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
