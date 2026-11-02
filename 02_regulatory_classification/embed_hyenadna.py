#!/usr/bin/env python3
# Mean-pooled HyenaDNA (medium-450k) embeddings for the 21 regulatory-sequence tasks.
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATASETS = [
    "ATAC/GSM4799615_ATAC_Cortex_P348_Peaks",
    "ATAC/GSM4799623_ATAC_Liver_P348_Peaks",
    "ATAC/GSM4799631_ATAC_Muscle_P348_Peaks",
    "CTCF/GSM4799659_CTCF_Adipose_P348_Peaks",
    "CTCF/GSM4799666_CTCF_Cerebellum_P350_Peaks",
    "CTCF/GSM4799690_CTCF_Lung_P350_Peaks",
    "H3K27ac/GSM4799734_H3K27ac_Cortex_P348_Peaks",
    "H3K27ac/GSM4799746_H3K27ac_Liver_P348_Peaks",
    "H3K27ac/GSM4799758_H3K27ac_Muscle_P348_Peaks",
    "H3K27me1/GSM4800054_H3K4me1_Cortex_P348_Peaks",
    "H3K27me1/GSM4800066_H3K4me1_Liver_P348_Peaks",
    "H3K27me1/GSM4800078_H3K4me1_Muscle_P348_Peaks",
    "H3K27me3/GSM4799782_H3K27me3_Cortex_P348_Peaks",
    "H3K27me3/GSM4799794_H3K27me3_Liver_P348_Peaks",
    "H3K27me3/GSM4799806_H3K27me3_Muscle_P348_Peaks",
    "enhancer/Cortex_15_segments.enhancer",
    "enhancer/Liver_15_segments.enhancer",
    "enhancer/Muscle_15_segments.enhancer",
    "promoter/Cortex_15_segments.promoter",
    "promoter/Liver_15_segments.promoter",
    "promoter/Muscle_15_segments.promoter",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="embeddings/hyenadna")
    parser.add_argument("--model-name", default="LongSafari/hyenadna-medium-450k-seqlen-hf")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--task-stride", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def encode_batch(sequences, tokenizer, model, max_length, device):
    tok = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    pad_id = tokenizer.pad_token_id
    attention_mask = torch.ones_like(input_ids, dtype=torch.int32) if pad_id is None else (input_ids != pad_id).int()
    out = model(input_ids)
    hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
    return mean_pooling(hidden, attention_mask).detach().float().cpu().numpy()


def encode_sequences(sequences, tokenizer, model, max_length, batch_size, device):
    chunks = []
    for start in tqdm(range(0, len(sequences), batch_size), desc="batches", leave=False):
        chunks.append(encode_batch(sequences[start:start + batch_size], tokenizer, model, max_length, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return np.concatenate(chunks, axis=0)


def output_complete(path, expected_rows):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        n_rows = sum(1 for _ in open(path, "r")) - 1
    except OSError:
        return False
    return n_rows == expected_rows


def main():
    args = parse_args()
    data_root, output_root = Path(args.data_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = DATASETS[args.task_offset::args.task_stride]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[model] {args.model_name}; tasks {len(selected)} of {len(DATASETS)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    for task_i, dataset in enumerate(selected, 1):
        print(f"[{task_i}/{len(selected)}] {dataset}", flush=True)
        in_dir, out_dir = data_root / dataset, output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_dir.exists():
            print(f"  missing input dir: {in_dir}", flush=True)
            continue
        for split in ("train", "test"):
            in_file = in_dir / f"{split}.csv"
            out_file = out_dir / f"{split}_embed_hyenadna_450k.csv"
            if not in_file.exists():
                continue
            df = pd.read_csv(in_file)
            if args.max_rows:
                df = df.head(args.max_rows).copy()
            if "sequence" not in df.columns or "label" not in df.columns:
                raise ValueError(f"{in_file} must contain sequence and label columns")
            if out_file.exists() and not args.force and output_complete(out_file, len(df)):
                continue
            seqs = df["sequence"].astype(str).str.upper().tolist()
            labels = df["label"].astype(np.int8).to_numpy()
            emb = encode_sequences(seqs, tokenizer, model, args.max_length, args.batch_size, device)
            out_df = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(emb.shape[1])])
            out_df["target"] = labels
            tmp_file = out_file.with_suffix(".tmp.csv")
            out_df.to_csv(tmp_file, index=False)
            tmp_file.replace(out_file)
            print(f"  saved {out_file} shape={emb.shape}", flush=True)


if __name__ == "__main__":
    main()
