#!/usr/bin/env python3
# Mean-pooled Evo2 7B embeddings (blocks.20.mlp.l3) for the regulatory-sequence tasks.
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evo2.models import Evo2
from tqdm import tqdm

MODEL_NAME = 'evo2_7b'
EMBEDDING_LAYER_NAME = 'blocks.20.mlp.l3'
BATCH_SIZE = 8
DEVICE = 'cuda:0'


def get_embeddings_batch(sequences, model, tokenizer, device, layer_name):
    if not sequences:
        return None
    valid_indices, valid_sequences = [], []
    for i, seq in enumerate(sequences):
        if seq and isinstance(seq, str) and seq.strip() and seq.upper() != 'NA':
            valid_indices.append(i)
            valid_sequences.append(seq.upper())
    if not valid_sequences:
        return None

    try:
        batch_token_ids = [tokenizer.tokenize(seq) or [] for seq in valid_sequences]
        actual_lengths = [len(t) for t in batch_token_ids]
        if not any(actual_lengths):
            return None
        max_len = max(actual_lengths)
        pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        padded = [tokens + [pad_id] * (max_len - len(tokens)) for tokens in batch_token_ids]
        input_ids = torch.tensor(padded, dtype=torch.long).to(device)

        with torch.no_grad():
            _, embeddings_dict = model(input_ids, return_embeddings=True, layer_names=[layer_name])
        mean_pooled = torch.mean(embeddings_dict[layer_name], dim=1).to(torch.float32).cpu().numpy()

        result = np.full((len(sequences), mean_pooled.shape[1]), np.nan)
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = mean_pooled[i]
        return result
    except KeyError:
        print(f"Error: layer '{layer_name}' not found in model.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error during embedding extraction: {e}", file=sys.stderr)
        return None


def process_csv_file(input_csv, output_csv, model, tokenizer, device, layer_name, batch_size):
    print(f"Processing: {input_csv}")
    df = pd.read_csv(input_csv)
    if 'sequence' not in df.columns:
        print(f"  Error: 'sequence' column not found in {input_csv}", file=sys.stderr)
        return False
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else None

    all_embeddings = []
    for i in tqdm(range(math.ceil(len(sequences) / batch_size)), desc="  batches"):
        batch = sequences[i * batch_size:min((i + 1) * batch_size, len(sequences))]
        emb = get_embeddings_batch(batch, model, tokenizer, device, layer_name)
        all_embeddings.append(emb if emb is not None else np.full((len(batch), 1), np.nan))
    embeddings_array = np.vstack(all_embeddings)

    emb_columns = [f"emb_{i}" for i in range(embeddings_array.shape[1])]
    df_embeddings = pd.DataFrame(embeddings_array, columns=emb_columns)
    if labels is not None:
        df_embeddings['target'] = labels
        df_embeddings = df_embeddings[['target'] + emb_columns]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_embeddings.to_csv(output_csv, index=False)
    print(f"  saved {output_csv}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data')
    parser.add_argument('--output_dir', default='embeddings/evo2')
    parser.add_argument('--dataset', default=None, help='single dataset, e.g. promoter/Liver_15_segments.promoter')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--layer', default=EMBEDDING_LAYER_NAME)
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)

    model = Evo2(MODEL_NAME)
    tokenizer = model.tokenizer

    if args.dataset:
        datasets = [input_dir / args.dataset]
    else:
        datasets = sorted(p.parent for p in input_dir.rglob('train.csv'))
    print(f"Found {len(datasets)} datasets")

    for dataset_path in datasets:
        rel_path = dataset_path.relative_to(input_dir)
        for split in ['train', 'dev', 'test']:
            input_csv = dataset_path / f'{split}.csv'
            if input_csv.exists():
                process_csv_file(str(input_csv), str(output_dir / rel_path / f'{split}_embed_evo2.csv'),
                                 model, tokenizer, DEVICE, args.layer, args.batch_size)


if __name__ == "__main__":
    main()
