"""DNABERT-2 embeddings of OMIA ref/alt sequences; writes alt-ref differences, distances and concatenated embeddings."""
import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MAX_LENGTH = 2048
BATCH_SIZE = 32


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask = attention_mask.unsqueeze(-1)
    return torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)


def get_dnabert_embedding(seq_list, tokenizer, model, max_length):
    tok = tokenizer(seq_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    tok = {k: v.to(model.device) for k, v in tok.items()}
    with torch.no_grad():
        hidden_states = model(tok["input_ids"], attention_mask=tok["attention_mask"])[0]
    return mean_pooling(hidden_states, tok["attention_mask"])


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
    parser = argparse.ArgumentParser(description='DNABERT-2 embeddings for OMIA variants')
    parser.add_argument('--input', default='data/omia_seqs_matched.csv')
    parser.add_argument('--output-dir', default='embeddings/flank3000_total6000')
    parser.add_argument('--checkpoint', default='zhihan1996/DNABERT-2-117M')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    full_df = pd.read_csv(args.input)
    groups = {'pathogenic': full_df[full_df['class'] == 1].copy(), 'common': full_df[full_df['class'] == 0].copy()}

    for group_name, df_group in groups.items():
        loader = DataLoader(SequenceDataset(df_group), batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
        all_embeds_ref, all_embeds_alt = [], []
        with torch.no_grad():
            for batch in loader:
                all_embeds_ref.append(get_dnabert_embedding(batch["ref_seq"], tokenizer, model, MAX_LENGTH).cpu())
                all_embeds_alt.append(get_dnabert_embedding(batch["alt_seq"], tokenizer, model, MAX_LENGTH).cpu())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        if not all_embeds_ref:
            continue
        calculate_and_save_metrics(
            vectors1=torch.cat(all_embeds_ref, dim=0),
            vectors2=torch.cat(all_embeds_alt, dim=0),
            metadata_df=df_group[['chromosome', 'pos', 'ref', 'alt', 'species']],
            output_path=args.output_dir, prefix="dnabert2", group_name=group_name)


if __name__ == "__main__":
    main()
