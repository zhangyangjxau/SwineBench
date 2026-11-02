"""Build the 1:1 matched OMIA pathogenic / common-control dataset (same species and effect class)."""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "ref_seq",
    "alt_seq",
    "class",
    "chromosome",
    "ref",
    "pos",
    "alt",
    "species",
    "variant_effect",
]

TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
BASES = {"A", "C", "G", "T"}


def normalize_effect(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).lower().replace("_", " ").replace("-", " ")
    if "missense" in text:
        return "missense"
    if "splic" in text:
        return "splicing"
    if "stop gained" in text or "nonsense" in text:
        return "stop_gained"
    if "stop lost" in text or "stop retained" in text or "extension" in text:
        return "stop_lost_or_extension"
    if "start lost" in text or "initiator" in text:
        return "start_lost"
    if "synonymous" in text or "silent" in text:
        return "synonymous"
    if "frameshift" in text:
        return "frameshift"
    if "utr" in text or "untranslated" in text:
        return "utr"
    if "intron" in text:
        return "intronic"
    if "regulatory" in text or "promoter" in text or "enhancer" in text:
        return "regulatory"
    if "intergenic" in text:
        return "intergenic"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown"


def normalize_species(value: object) -> str:
    return str(value).strip().lower()


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text


def normalize_allele(value: object) -> str:
    return str(value).strip().upper()


def numeric_position(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def sequence_gc(seq: object) -> float:
    text = str(seq).upper()
    a = text.count("A")
    c = text.count("C")
    g = text.count("G")
    t = text.count("T")
    denom = a + c + g + t
    if denom == 0:
        return float("nan")
    return (g + c) / denom


def allele_change(ref: object, alt: object) -> str:
    return f"{normalize_allele(ref)}>{normalize_allele(alt)}"


def change_class(ref: object, alt: object) -> str:
    ref_base = normalize_allele(ref)
    alt_base = normalize_allele(alt)
    if len(ref_base) != 1 or len(alt_base) != 1:
        return "non_snv"
    if ref_base not in BASES or alt_base not in BASES:
        return "non_acgt"
    if ref_base == alt_base:
        return "same"
    if (ref_base, alt_base) in TRANSITIONS:
        return "transition"
    return "transversion"


def add_common_features(df: pd.DataFrame, row_id_name: str) -> pd.DataFrame:
    df = df.copy()
    df[row_id_name] = [f"{row_id_name.upper()}{i + 1:06d}" for i in range(len(df))]
    df["source_row_index"] = df.index
    df["species_norm"] = df["Species Name"].map(normalize_species)
    df["effect_norm"] = df["Variant Effect"].map(normalize_effect)
    df["chr_norm"] = df["Chr"].map(normalize_chromosome)
    df["pos_norm"] = numeric_position(df["Position"])
    df["ref_norm"] = df["Ref Allele"].map(normalize_allele)
    df["alt_norm"] = df["Alt Allele"].map(normalize_allele)
    df["is_snv"] = df["ref_norm"].str.len().eq(1) & df["alt_norm"].str.len().eq(1)
    df["allele_change"] = [allele_change(r, a) for r, a in zip(df["Ref Allele"], df["Alt Allele"])]
    df["change_class"] = [change_class(r, a) for r, a in zip(df["Ref Allele"], df["Alt Allele"])]
    df["ref_seq_len"] = df["Ref_Sequence_Window"].astype(str).str.len()
    df["alt_seq_len"] = df["Alt_Sequence_Window"].astype(str).str.len()
    df["gc_content"] = df["Ref_Sequence_Window"].map(sequence_gc)
    df["locus_key"] = (
        df["species_norm"].astype(str)
        + "|"
        + df["chr_norm"].astype(str)
        + "|"
        + df["pos_norm"].astype(str)
    )
    return df


def match_score(pos_row: pd.Series, ctrl_row: pd.Series) -> float:
    same_chr_penalty = 0.0 if pos_row["chr_norm"] == ctrl_row["chr_norm"] else 1.0
    change_class_penalty = 0.0 if pos_row["change_class"] == ctrl_row["change_class"] else 1.0
    exact_change_penalty = 0.0 if pos_row["allele_change"] == ctrl_row["allele_change"] else 1.0
    gc_diff = abs(float(pos_row["gc_content"]) - float(ctrl_row["gc_content"]))
    if math.isnan(gc_diff):
        gc_diff = 1.0
    if same_chr_penalty == 0 and not pd.isna(pos_row["pos_norm"]) and not pd.isna(ctrl_row["pos_norm"]):
        distance_component = min(abs(int(pos_row["pos_norm"]) - int(ctrl_row["pos_norm"])) / 100_000_000, 1.0)
    else:
        distance_component = 1.0
    length_component = min(abs(int(pos_row["ref_seq_len"]) - int(ctrl_row["ref_seq_len"])) / 10_000, 1.0)
    return (
        same_chr_penalty * 1000.0
        + change_class_penalty * 100.0
        + exact_change_penalty * 10.0
        + gc_diff
        + distance_component * 0.01
        + length_component * 0.001
    )


def pick_control(pos_row: pd.Series, candidates: pd.DataFrame, unused_indexes: Set[int]) -> Tuple[Optional[int], dict]:
    available = candidates.loc[candidates.index.intersection(unused_indexes)]
    if available.empty:
        return None, {}

    same_chr = available["chr_norm"].eq(pos_row["chr_norm"])
    same_change_class = available["change_class"].eq(pos_row["change_class"])
    exact_change = available["allele_change"].eq(pos_row["allele_change"])
    gc_diff = (available["gc_content"] - float(pos_row["gc_content"])).abs().fillna(1.0)

    pos_value = pos_row["pos_norm"]
    if not pd.isna(pos_value):
        distance = (available["pos_norm"].astype("float64") - float(pos_value)).abs()
        distance = distance.where(same_chr, np.inf)
    else:
        distance = pd.Series(np.inf, index=available.index)

    ranking = pd.DataFrame(
        {
            "not_same_chr": (~same_chr).astype(int),
            "not_same_change_class": (~same_change_class).astype(int),
            "not_exact_change": (~exact_change).astype(int),
            "gc_diff": gc_diff,
            "distance": distance.fillna(np.inf),
            "source_row_index": available["source_row_index"].astype(int),
        },
        index=available.index,
    ).sort_values(
        [
            "not_same_chr",
            "not_same_change_class",
            "not_exact_change",
            "gc_diff",
            "distance",
            "source_row_index",
        ],
        kind="mergesort",
    )
    selected_idx = int(ranking.index[0])
    selected = available.loc[selected_idx]
    metrics = {
        "same_chromosome": bool(selected["chr_norm"] == pos_row["chr_norm"]),
        "same_change_class": bool(selected["change_class"] == pos_row["change_class"]),
        "exact_allele_change": bool(selected["allele_change"] == pos_row["allele_change"]),
        "position_distance_bp": (
            int(abs(int(pos_row["pos_norm"]) - int(selected["pos_norm"])))
            if selected["chr_norm"] == pos_row["chr_norm"]
            and not pd.isna(pos_row["pos_norm"])
            and not pd.isna(selected["pos_norm"])
            else None
        ),
        "gc_abs_diff": float(abs(float(pos_row["gc_content"]) - float(selected["gc_content"])))
        if not math.isnan(float(pos_row["gc_content"])) and not math.isnan(float(selected["gc_content"]))
        else None,
        "match_score": float(match_score(pos_row, selected)),
        "candidate_count_unused": int(len(available)),
    }
    return selected_idx, metrics


def make_standard_rows(
    selected_pos: pd.DataFrame,
    selected_ctrl: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    pair_by_pos = pairs.set_index("positive_id")
    pair_by_ctrl = pairs.set_index("control_id")

    pos_rows = pd.DataFrame(
        {
            "ref_seq": selected_pos["Ref_Sequence_Window"],
            "alt_seq": selected_pos["Alt_Sequence_Window"],
            "class": 1,
            "chromosome": selected_pos["Chr"],
            "ref": selected_pos["Ref Allele"],
            "pos": selected_pos["Position"],
            "alt": selected_pos["Alt Allele"],
            "species": selected_pos["Species Name"],
            "variant_effect": selected_pos["Variant Effect"],
            "pair_role": "pathogenic",
            "positive_id": selected_pos["positive_id"],
            "control_id": selected_pos["positive_id"].map(pair_by_pos["control_id"]),
            "match_id": selected_pos["positive_id"].map(pair_by_pos["match_id"]),
            "effect_norm": selected_pos["effect_norm"],
            "change_class": selected_pos["change_class"],
            "allele_change": selected_pos["allele_change"],
            "gc_content": selected_pos["gc_content"],
        }
    )
    ctrl_rows = pd.DataFrame(
        {
            "ref_seq": selected_ctrl["Ref_Sequence_Window"],
            "alt_seq": selected_ctrl["Alt_Sequence_Window"],
            "class": 0,
            "chromosome": selected_ctrl["Chr"],
            "ref": selected_ctrl["Ref Allele"],
            "pos": selected_ctrl["Position"],
            "alt": selected_ctrl["Alt Allele"],
            "species": selected_ctrl["Species Name"],
            "variant_effect": selected_ctrl["Variant Effect"],
            "pair_role": "common_control",
            "positive_id": selected_ctrl["control_id"].map(pair_by_ctrl["positive_id"]),
            "control_id": selected_ctrl["control_id"],
            "match_id": selected_ctrl["control_id"].map(pair_by_ctrl["match_id"]),
            "effect_norm": selected_ctrl["effect_norm"],
            "change_class": selected_ctrl["change_class"],
            "allele_change": selected_ctrl["allele_change"],
            "gc_content": selected_ctrl["gc_content"],
        }
    )
    merged = pd.concat([pos_rows, ctrl_rows], ignore_index=True)
    return merged[STANDARD_COLUMNS + [c for c in merged.columns if c not in STANDARD_COLUMNS]]


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 1:1 matched OMIA dataset.")
    parser.add_argument("--positive-csv", default="data/omia_snv_sequences.csv")
    parser.add_argument("--control-csv", default="data/control_snvs_type_matched_sequences.csv")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    positive_path = Path(args.positive_csv)
    control_path = Path(args.control_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_raw = pd.read_csv(positive_path)
    ctrl_raw = pd.read_csv(control_path)

    pos = pos_raw[pos_raw["Processing_Status"].eq("Success")].copy()
    ctrl = ctrl_raw[ctrl_raw["Processing_Status"].eq("Success")].copy()
    pos = add_common_features(pos, "positive_id")
    ctrl = add_common_features(ctrl, "control_id")

    positive_loci = set(pos["locus_key"])
    ctrl_before_overlap = len(ctrl)
    ctrl = ctrl[~ctrl["locus_key"].isin(positive_loci)].copy()
    ctrl_overlap_removed = ctrl_before_overlap - len(ctrl)
    ctrl_before_dedup = len(ctrl)
    ctrl = ctrl.drop_duplicates(["locus_key", "ref_norm", "alt_norm", "effect_norm"]).copy()
    ctrl_duplicate_removed = ctrl_before_dedup - len(ctrl)

    eligible_pos = pos[pos["is_snv"]].copy()
    ctrl = ctrl[ctrl["is_snv"]].copy()

    candidate_groups = {key: group.copy() for key, group in ctrl.groupby(["species_norm", "effect_norm"], sort=True)}
    candidate_counts = {key: len(group) for key, group in candidate_groups.items()}

    unavailable_rows = []
    matchable_indexes = []
    for idx, row in eligible_pos.iterrows():
        key = (row["species_norm"], row["effect_norm"])
        if key not in candidate_groups or candidate_groups[key].empty:
            reason = "no_exact_species_effect_control"
            if row["species_norm"] not in set(ctrl["species_norm"]):
                reason = "no_control_species"
            unavailable_rows.append({"positive_index": idx, "drop_reason": reason})
        else:
            matchable_indexes.append(idx)

    matchable_pos = eligible_pos.loc[matchable_indexes].copy()
    matchable_pos["same_chr_candidate_count"] = [
        int(candidate_groups[(row["species_norm"], row["effect_norm"])]["chr_norm"].eq(row["chr_norm"]).sum())
        for _, row in matchable_pos.iterrows()
    ]
    matchable_pos = matchable_pos.sort_values(
        ["species_norm", "effect_norm", "same_chr_candidate_count", "source_row_index"],
        kind="mergesort",
    )

    unused_by_group = {key: set(map(int, group.index.tolist())) for key, group in candidate_groups.items()}
    pairs = []
    exhausted_rows = []

    for _, pos_row in matchable_pos.iterrows():
        key = (pos_row["species_norm"], pos_row["effect_norm"])
        selected_idx, metrics = pick_control(pos_row, candidate_groups[key], unused_by_group[key])
        if selected_idx is None:
            exhausted_rows.append({"positive_index": int(pos_row.name), "drop_reason": "exact_controls_exhausted"})
            continue
        unused_by_group[key].remove(selected_idx)
        ctrl_row = ctrl.loc[selected_idx]
        match_id = f"MATCH{len(pairs) + 1:06d}"
        pair = {
            "match_id": match_id,
            "positive_id": pos_row["positive_id"],
            "control_id": ctrl_row["control_id"],
            "species": pos_row["Species Name"],
            "effect_norm": pos_row["effect_norm"],
            "positive_variant_effect": pos_row["Variant Effect"],
            "control_variant_effect": ctrl_row["Variant Effect"],
            "positive_chr": pos_row["Chr"],
            "positive_pos": int(pos_row["pos_norm"]) if not pd.isna(pos_row["pos_norm"]) else None,
            "control_chr": ctrl_row["Chr"],
            "control_pos": int(ctrl_row["pos_norm"]) if not pd.isna(ctrl_row["pos_norm"]) else None,
            "positive_change": pos_row["allele_change"],
            "control_change": ctrl_row["allele_change"],
            "positive_change_class": pos_row["change_class"],
            "control_change_class": ctrl_row["change_class"],
            "positive_gc": float(pos_row["gc_content"]),
            "control_gc": float(ctrl_row["gc_content"]),
            "candidate_count_initial": int(candidate_counts[key]),
        }
        pair.update(metrics)
        pairs.append(pair)

    pairs_df = pd.DataFrame(pairs)
    matched_positive_ids = set(pairs_df["positive_id"]) if not pairs_df.empty else set()
    matched_control_ids = set(pairs_df["control_id"]) if not pairs_df.empty else set()
    selected_pos = pos[pos["positive_id"].isin(matched_positive_ids)].copy()
    selected_ctrl = ctrl[ctrl["control_id"].isin(matched_control_ids)].copy()

    pair_order = pairs_df["match_id"].tolist()
    selected_pos["match_id"] = selected_pos["positive_id"].map(pairs_df.set_index("positive_id")["match_id"])
    selected_ctrl["match_id"] = selected_ctrl["control_id"].map(pairs_df.set_index("control_id")["match_id"])
    selected_pos["match_order"] = pd.Categorical(selected_pos["match_id"], pair_order, ordered=True)
    selected_ctrl["match_order"] = pd.Categorical(selected_ctrl["match_id"], pair_order, ordered=True)
    selected_pos = selected_pos.sort_values("match_order").drop(columns=["match_order"])
    selected_ctrl = selected_ctrl.sort_values("match_order").drop(columns=["match_order"])

    merged = make_standard_rows(selected_pos, selected_ctrl, pairs_df)

    dropped_positive_indexes = [item["positive_index"] for item in unavailable_rows + exhausted_rows]
    dropped = pos.loc[dropped_positive_indexes].copy() if dropped_positive_indexes else pos.iloc[0:0].copy()
    if not dropped.empty:
        reason_map = {item["positive_index"]: item["drop_reason"] for item in unavailable_rows + exhausted_rows}
        dropped["drop_reason"] = dropped.index.map(reason_map)

    control_sequence_out = output_dir / "control_snvs_matched_sequences.csv"
    control_raw_out = output_dir / "control_snvs_matched.csv"
    merged_out = output_dir / "omia_seqs_matched.csv"
    pairs_out = output_dir / "omia_matched_pairs.csv"
    dropped_out = output_dir / "omia_matched_dropped_positives.csv"
    report_out = output_dir / "omia_matched_report.json"

    selected_ctrl_with_pairs = selected_ctrl.merge(
        pairs_df[["match_id", "positive_id", "control_id"]],
        on=["match_id", "control_id"],
        how="left",
        suffixes=("", "_pair"),
    )
    sequence_cols = [c for c in selected_ctrl_with_pairs.columns if not c.endswith("_pair")]
    selected_ctrl_with_pairs[sequence_cols].to_csv(control_sequence_out, index=False)
    raw_drop = {"Ref_Sequence_Window", "Alt_Sequence_Window", "species_norm", "chr_norm", "pos_norm", "ref_norm", "alt_norm", "locus_key"}
    selected_ctrl_with_pairs[[c for c in sequence_cols if c not in raw_drop]].to_csv(control_raw_out, index=False)
    merged.to_csv(merged_out, index=False)
    pairs_df.to_csv(pairs_out, index=False)
    dropped.to_csv(dropped_out, index=False)

    report = {
        "positive_input_success": len(pos),
        "positive_input_snv_success": int(pos["is_snv"].sum()),
        "control_input_success": len(ctrl_raw[ctrl_raw["Processing_Status"].eq("Success")]),
        "control_after_positive_locus_exclusion": len(ctrl),
        "control_positive_locus_removed": ctrl_overlap_removed,
        "control_duplicate_removed": ctrl_duplicate_removed,
        "hard_constraints": [
            "SNV only",
            "successful sequence extraction",
            "same species",
            "same normalized consequence/effect category",
            "negative control locus not overlapping any OMIA positive locus in the same species",
            "1:1 matching without control reuse",
        ],
        "soft_ranking": [
            "same chromosome",
            "same transition/transversion class",
            "same ref>alt allele change",
            "minimum GC-content difference of the reference window",
            "minimum genomic distance when on the same chromosome",
        ],
        "matched_pairs": len(pairs_df),
        "merged_rows": len(merged),
        "class_counts": merged["class"].value_counts().sort_index().to_dict(),
        "dropped_positives": len(dropped),
        "dropped_by_reason": dropped["drop_reason"].value_counts().to_dict() if not dropped.empty else {},
        "matched_by_species": pairs_df["species"].value_counts().to_dict() if not pairs_df.empty else {},
        "matched_by_effect_norm": pairs_df["effect_norm"].value_counts().to_dict() if not pairs_df.empty else {},
        "matched_same_chromosome": int(pairs_df["same_chromosome"].sum()) if not pairs_df.empty else 0,
        "matched_same_change_class": int(pairs_df["same_change_class"].sum()) if not pairs_df.empty else 0,
        "matched_exact_allele_change": int(pairs_df["exact_allele_change"].sum()) if not pairs_df.empty else 0,
        "mean_gc_abs_diff": float(pairs_df["gc_abs_diff"].mean()) if not pairs_df.empty else None,
        "outputs": {
            "control_raw": str(control_raw_out),
            "control_sequences": str(control_sequence_out),
            "merged_sequences": str(merged_out),
            "pair_report": str(pairs_out),
            "dropped_positives": str(dropped_out),
            "report": str(report_out),
        },
    }
    report_out.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")

    print(json.dumps(json_ready(report), indent=2))


if __name__ == "__main__":
    main()
