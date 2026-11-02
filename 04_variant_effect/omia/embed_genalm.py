#!/usr/bin/env python3
"""GENA-LM embeddings of OMIA ref/alt sequences (4,500-bp centre crop); writes alt-ref differences, distances and concatenated embeddings."""
import argparse
import gc
import importlib
import math
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "AIRI-Institute/gena-lm-bert-base-t2t-multi"
MAX_LENGTH = 512
BATCH_SIZE = 32
CENTER_CROP_BP = 4500


def center_crop(seq, target_bp=CENTER_CROP_BP):
    if len(seq) <= target_bp:
        return seq
    start = (len(seq) - target_bp) // 2
    return seq[start:start + target_bp]


def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def get_embeddings_batch(sequences, model, tokenizer, device, max_length, use_amp=True):
    """Mean-pooled embeddings for a batch; rows with missing sequences are NaN."""
    valid_indices, valid_sequences = [], []
    for i, seq in enumerate(sequences):
        if seq and isinstance(seq, str) and seq.strip() and seq.strip().upper() not in {"NA", "NAN", "NONE"}:
            valid_indices.append(i)
            valid_sequences.append(center_crop(seq.strip().upper()))
    if not valid_sequences:
        return None

    tok = tokenizer(valid_sequences, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    tok = {k: v.to(device) for k, v in tok.items()}
    if use_amp and str(device).startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**tok)
    else:
        outputs = model(**tok)
    seq_emb = mean_pooling(outputs.last_hidden_state, tok["attention_mask"]).float().cpu().numpy()

    result = np.full((len(sequences), seq_emb.shape[1]), np.nan, dtype=np.float32)
    for j, orig_i in enumerate(valid_indices):
        result[orig_i] = seq_emb[j]
    return result


def process_group(df_group, group_name, model, tokenizer, device, output_dir, batch_size, max_length, prefix='genalm'):
    ref_seqs = df_group['ref_seq'].astype(str).tolist()
    alt_seqs = df_group['alt_seq'].astype(str).tolist()
    n = len(ref_seqs)
    num_chunks = math.ceil(n / batch_size)

    all_ref_embs, all_alt_embs = [], []
    for i in range(num_chunks):
        start, end = i * batch_size, min((i + 1) * batch_size, n)
        all_ref_embs.append(get_embeddings_batch(ref_seqs[start:end], model, tokenizer, device, max_length))
        all_alt_embs.append(get_embeddings_batch(alt_seqs[start:end], model, tokenizer, device, max_length))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    H = next((e.shape[1] for e in all_ref_embs + all_alt_embs if e is not None), None)
    if H is None:
        print(f"No embeddings for {group_name}, skipping.")
        return

    def stitch(parts):
        result = []
        for idx, part in enumerate(parts):
            if part is None:
                s, e = idx * batch_size, min((idx + 1) * batch_size, n)
                result.append(np.full((e - s, H), np.nan, dtype=np.float32))
            else:
                result.append(part)
        return np.vstack(result)

    v_ref, v_alt = stitch(all_ref_embs), stitch(all_alt_embs)
    metadata = df_group[['chromosome', 'pos', 'ref', 'alt', 'species']].reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)

    differences = v_alt - v_ref
    pd.concat([pd.DataFrame(differences), metadata], axis=1).to_csv(
        f"{output_dir}/{prefix}_differences_{group_name}.csv", index=False, header=False)

    cols = [f'Ref_Emb_{i}' for i in range(H)] + [f'Alt_Emb_{i}' for i in range(H)]
    pd.concat([pd.DataFrame(np.hstack([v_ref, v_alt]), columns=cols), metadata], axis=1).to_csv(
        f"{output_dir}/{prefix}_concat_{group_name}.csv", index=False, header=True)

    l1 = np.nansum(np.abs(differences), axis=1)
    l2 = np.sqrt(np.nansum(differences ** 2, axis=1))
    norm1 = np.sqrt(np.nansum(v_ref ** 2, axis=1))
    norm2 = np.sqrt(np.nansum(v_alt ** 2, axis=1))
    dot = np.nansum(v_ref * v_alt, axis=1)
    cosine = np.divide(dot, norm1 * norm2, out=np.zeros_like(dot), where=(norm1 * norm2) != 0)
    for name, col, vals in [('L1', 'L1_distance', l1), ('L2', 'L2_distance', l2), ('cosine', 'cosine_similarity', cosine)]:
        m_df = metadata.copy()
        m_df.insert(0, col, vals)
        m_df.to_csv(f"{output_dir}/{prefix}_{name}_{group_name}.csv", index=False, header=True)
    print(f"Saved {prefix} {group_name} to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='GENA-LM embeddings for OMIA variants')
    parser.add_argument('--input', default='data/omia_seqs_matched.csv')
    parser.add_argument('--output-dir', default='embeddings/flank3000_total6000')
    parser.add_argument('--checkpoint', default=MODEL_ID)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    full_df = pd.read_csv(args.input)
    df_pathogenic = full_df[full_df['class'] == 1].copy()
    df_common = full_df[full_df['class'] == 0].copy()

    model_temp = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)
    gena_module_name = model_temp.__class__.__module__
    del model_temp
    BertModel = getattr(importlib.import_module(gena_module_name), 'BertModel')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = BertModel.from_pretrained(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for group_name, df_group in [('pathogenic', df_pathogenic), ('common', df_common)]:
        process_group(df_group, group_name, model, tokenizer, device, args.output_dir, args.batch_size, args.max_length)


if __name__ == "__main__":
    main()
