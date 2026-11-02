# Mean-pooled GENA-LM embeddings for the 21 regulatory-sequence tasks.
import gc
import importlib
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATA_ROOT = "data"
OUTPUT_ROOT = "embeddings/genalm"
CHECKPOINT = "AIRI-Institute/gena-lm-bert-base-t2t-multi"

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

MAX_LENGTH = 512
BATCH_SIZE = 32
POOLING = "mean"  # "cls" or "mean"


def mean_pooling(hidden_states, attention_mask):
    attention_mask = attention_mask.unsqueeze(-1).float()
    return torch.sum(hidden_states * attention_mask, dim=1) / torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)


def get_embeddings(seq_list, tokenizer, model, max_length, pooling):
    tok = tokenizer(seq_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    tok = {k: v.to(model.device) for k, v in tok.items()}
    with torch.no_grad():
        hidden_states = model(**tok).last_hidden_state
    if pooling == "cls":
        return hidden_states[:, 0, :]
    return mean_pooling(hidden_states, tok["attention_mask"])


def process_sequences(sequences, tokenizer, model, batch_size, max_length, pooling):
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        emb = get_embeddings(sequences[i:i + batch_size], tokenizer, model, max_length, pooling)
        all_embeddings.append(emb.cpu().numpy())
        del emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    # load the plain BertModel encoder (no MLM head) from the checkpoint's remote module
    model_temp = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True)
    gena_module_name = model_temp.__class__.__module__
    del model_temp
    BertModel = getattr(importlib.import_module(gena_module_name), 'BertModel')
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = BertModel.from_pretrained(CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    suffix = "_cls" if POOLING == "cls" else "_meanpool"
    for data_name in DATASETS:
        print(f"Processing: {data_name}")
        data_dir = f"{DATA_ROOT}/{data_name}"
        output_dir = f"{OUTPUT_ROOT}/{data_name}"
        if not os.path.exists(data_dir):
            print(f"  directory not found: {data_dir}")
            continue
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "test"]:
            input_file = f"{data_dir}/{split}.csv"
            output_file = f"{output_dir}/{split}_embed_genalm{suffix}.csv"
            if not os.path.exists(input_file):
                continue
            df = pd.read_csv(input_file)
            sequences = df['sequence'].astype(str).tolist()
            embeddings = process_sequences(sequences, tokenizer, model, BATCH_SIZE, MAX_LENGTH, POOLING)
            out_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
            out_df['target'] = df['label'].values
            out_df.to_csv(output_file, index=False)
            print(f"  saved {output_file}")
