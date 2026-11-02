"""Evo2 embeddings of OMIA ref/alt sequences; writes alt-ref differences, distances and concatenated embeddings."""
import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from evo2.models import Evo2
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 8
EMBEDDING_LAYER_NAME = 'blocks.20.mlp.l3'
MODEL_NAME = 'evo2_7b'


def get_evo2_embedding_batch(seq_list, model, device, layer_name):
    """Tokenize, right-pad, run Evo2 and mean-pool the chosen layer; returns [batch, emb_dim] or None."""
    tokenizer = model.tokenizer
    batch_token_ids, actual_lengths = [], []
    for seq in seq_list:
        if not seq or not isinstance(seq, str) or seq.upper() == 'NA' or seq.strip() == '':
            batch_token_ids.append([])
            actual_lengths.append(0)
            continue
        tokens = tokenizer.tokenize(seq)
        batch_token_ids.append(tokens if tokens else [])
        actual_lengths.append(len(tokens) if tokens else 0)

    if not any(actual_lengths):
        return None
    max_len = max(actual_lengths)

    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    padded = [t + [pad_id] * (max_len - len(t)) for t in batch_token_ids]
    input_ids = torch.tensor(padded, dtype=torch.long).to(device)

    with torch.no_grad():
        _, embeddings_dict = model(input_ids, return_embeddings=True, layer_names=[layer_name])

    pooled = torch.mean(embeddings_dict[layer_name], dim=1).to(torch.float32)
    for i, length in enumerate(actual_lengths):
        if length == 0:
            pooled[i] = 0.0
    return pooled


def calculate_and_save_metrics(vectors1, vectors2, metadata_df, output_path, prefix, group_name):
    os.makedirs(output_path, exist_ok=True)
    metadata_df = metadata_df.reset_index(drop=True)

    v1 = vectors1.cpu().numpy()
    v2 = vectors2.cpu().numpy()
    differences = v2 - v1
    l1 = np.sum(np.abs(differences), axis=1)
    l2 = np.linalg.norm(differences, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    dot_product = np.sum(v1 * v2, axis=1)
    cosine_sim = np.divide(dot_product, norm1 * norm2, out=np.zeros_like(dot_product, dtype=float),
                           where=(norm1 * norm2) != 0)

    pd.concat([pd.DataFrame(differences), metadata_df], axis=1).to_csv(
        f"{output_path}/{prefix}_differences_{group_name}.csv", index=False, header=False)
    for name, col, values in [('L1', 'L1_distance', l1), ('L2', 'L2_distance', l2),
                              ('cosine', 'cosine_similarity', cosine_sim)]:
        out = metadata_df.copy()
        out.insert(0, col, values)
        out.to_csv(f"{output_path}/{prefix}_{name}_{group_name}.csv", index=False, header=True)

    emb_dim = v1.shape[1]
    cols = [f'Ref_Emb_{i}' for i in range(emb_dim)] + [f'Alt_Emb_{i}' for i in range(emb_dim)]
    pd.concat([pd.DataFrame(np.hstack([v1, v2]), columns=cols), metadata_df], axis=1).to_csv(
        f"{output_path}/{prefix}_concat_{group_name}.csv", index=False, header=True)
    print(f"Saved {prefix} {group_name} to {output_path}")


class SequenceDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        df["ref_seq"] = df["ref_seq"].astype(str)
        df["alt_seq"] = df["alt_seq"].astype(str)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


def collate_fn(batch_list):
    return {key: [item[key] for item in batch_list] for key in batch_list[0]}


def main():
    parser = argparse.ArgumentParser(description='Evo2 embeddings for OMIA variants')
    parser.add_argument('--input', default='data/omia_seqs_matched.csv')
    parser.add_argument('--output-dir', default='embeddings/flank3000_total6000')
    parser.add_argument('--model-name', default=MODEL_NAME)
    parser.add_argument('--layer-name', default=EMBEDDING_LAYER_NAME)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Evo2(args.model_name)

    full_df = pd.read_csv(args.input)
    groups = {'pathogenic': full_df[full_df['class'] == 1].copy(), 'common': full_df[full_df['class'] == 0].copy()}

    for group_name, df_group in groups.items():
        if df_group.empty:
            continue
        loader = DataLoader(SequenceDataset(df_group), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
        all_embeds_ref, all_embeds_alt = [], []
        with torch.no_grad():
            for batch in loader:
                embed_ref = get_evo2_embedding_batch(batch["ref_seq"], model, device, args.layer_name)
                embed_alt = get_evo2_embedding_batch(batch["alt_seq"], model, device, args.layer_name)
                if embed_ref is not None:
                    all_embeds_ref.append(embed_ref.cpu())
                if embed_alt is not None:
                    all_embeds_alt.append(embed_alt.cpu())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        if not all_embeds_ref:
            continue
        calculate_and_save_metrics(
            vectors1=torch.cat(all_embeds_ref, dim=0),
            vectors2=torch.cat(all_embeds_alt, dim=0),
            metadata_df=df_group[['chromosome', 'pos', 'ref', 'alt', 'species']],
            output_path=args.output_dir, prefix="evo2", group_name=group_name)


if __name__ == "__main__":
    main()
