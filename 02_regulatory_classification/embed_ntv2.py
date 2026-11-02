#!/usr/bin/env python3
# Mean-pooled NT-v2 (500M multi-species) embeddings for the regulatory-sequence tasks.
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
SPLITS = ("train", "dev", "test")


def mean_pooling(hidden_states, attention_mask):
    hidden_states = hidden_states.float()
    mask = attention_mask.float().unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def embed_batch(sequences, tokenizer, model, device, max_length):
    valid_indices, valid_sequences = [], []
    for i, seq in enumerate(sequences):
        if isinstance(seq, str) and seq.strip() and seq.strip().upper() not in {"NA", "NAN", "NONE"}:
            valid_indices.append(i)
            valid_sequences.append(seq.strip().upper())
    if not valid_sequences:
        return None

    tok = tokenizer(valid_sequences, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    if hasattr(model, "esm"):
        outputs = model.esm(input_ids=input_ids, attention_mask=attention_mask,
                            encoder_attention_mask=attention_mask, output_hidden_states=False)
        token_embeddings = outputs.last_hidden_state
    else:
        outputs = model(input_ids, attention_mask=attention_mask,
                        encoder_attention_mask=attention_mask, output_hidden_states=True)
        token_embeddings = outputs["hidden_states"][-1]
    emb = mean_pooling(token_embeddings, attention_mask).cpu().numpy().astype(np.float32)

    result = np.full((len(sequences), emb.shape[1]), np.nan, dtype=np.float32)
    for j, original_i in enumerate(valid_indices):
        result[original_i] = emb[j]
    return result


def process_csv(input_csv, output_csv, tokenizer, model, device, batch_size, max_length, max_records):
    df = pd.read_csv(input_csv)
    if "sequence" not in df.columns:
        raise ValueError(f"{input_csv} does not contain a sequence column")
    if max_records:
        df = df.head(max_records).copy()
    sequences = df["sequence"].astype(str).tolist()
    labels = df["label"].to_numpy() if "label" in df.columns else None
    batches = math.ceil(len(sequences) / batch_size)

    parts = []
    for i in tqdm(range(batches), desc=f"  {input_csv.name}", leave=False):
        start, end = i * batch_size, min((i + 1) * batch_size, len(sequences))
        parts.append(embed_batch(sequences[start:end], tokenizer, model, device, max_length))

    first = next((x for x in parts if x is not None), None)
    if first is None:
        raise ValueError(f"No valid sequences in {input_csv}")
    dim = first.shape[1]
    stitched = []
    for i, emb in enumerate(parts):
        if emb is None:
            start, end = i * batch_size, min((i + 1) * batch_size, len(sequences))
            stitched.append(np.full((end - start, dim), np.nan, dtype=np.float32))
        else:
            stitched.append(emb)
    arr = np.vstack(stitched)

    out = pd.DataFrame(arr, columns=[f"emb_{i}" for i in range(arr.shape[1])])
    if labels is not None:
        out.insert(0, "target", labels)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} N={len(out)} dim={arr.shape[1]}", flush=True)


def discover_tasks(input_dir, dataset):
    if dataset:
        return [input_dir / dataset]
    return sorted(p.parent for p in input_dir.rglob("train.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data")
    parser.add_argument("--output_dir", default="embeddings/ntv2")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task_offset", type=int, default=0)
    parser.add_argument("--task_stride", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    tasks = discover_tasks(input_dir, args.dataset)
    tasks = [p for i, p in enumerate(tasks) if i % args.task_stride == args.task_offset]
    print(f"Model: {args.model_id}; tasks: {len(tasks)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_id, trust_remote_code=True)
    model.to(args.device).eval()

    for done, task in enumerate(tasks, 1):
        rel = task.relative_to(input_dir)
        print(f"[{done}/{len(tasks)}] {rel}", flush=True)
        for split in SPLITS:
            input_csv = task / f"{split}.csv"
            if not input_csv.exists():
                continue
            output_csv = output_dir / rel / f"{split}_embed_ntv2.csv"
            if output_csv.exists() and not args.force:
                continue
            process_csv(input_csv, output_csv, tokenizer, model, args.device,
                        args.batch_size, args.max_length, args.max_records)


if __name__ == "__main__":
    main()
