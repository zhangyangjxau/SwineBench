# Mean-pooled GENERator-v2 (3B) embeddings for the 21 regulatory-sequence tasks.
import gc
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_ROOT = "data"
OUTPUT_ROOT = "embeddings/generator"
MODEL_ID = "GenerTeam/GENERator-v2-eukaryote-3b-base"

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

POOLING = "mean"  # "eos" or "mean"
BATCH_SIZE = 32


def truncate_to_multiple_of_6(sequence):
    remainder = len(sequence) % 6
    return sequence[:len(sequence) - remainder] if remainder else sequence


def get_embeddings(seq_list, tokenizer, model, max_length, pooling):
    # 6-mer tokenizer: truncate to a multiple of 6 and prepend the BOS token manually
    processed = [tokenizer.bos_token + truncate_to_multiple_of_6(seq) for seq in seq_list]
    tokenizer.padding_side = "right"
    tok = tokenizer(processed, add_special_tokens=False, padding=True, truncation=True,
                    max_length=max_length, return_tensors='pt')
    tok = {k: v.to(model.device) for k, v in tok.items()}
    with torch.inference_mode():
        outputs = model(**tok, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    attention_mask = tok["attention_mask"]
    if pooling == "eos":
        last = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last, :]
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(torch.float32)
    return torch.sum(hidden_states * mask, dim=1) / mask.sum(dim=1)


def process_sequences(sequences, tokenizer, model, batch_size, max_length, pooling):
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        emb = get_embeddings(sequences[i:i + batch_size], tokenizer, model, max_length, pooling)
        all_embeddings.append(emb.cpu())
        torch.cuda.empty_cache()
        gc.collect()
    return torch.cat(all_embeddings, dim=0).numpy()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    max_length = model.config.max_position_embeddings

    suffix = "_eos" if POOLING == "eos" else "_meanpool"
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
            output_file = f"{output_dir}/{split}_embed_generator{suffix}.csv"
            if not os.path.exists(input_file):
                continue
            df = pd.read_csv(input_file)
            sequences = df['sequence'].astype(str).tolist()
            embeddings = process_sequences(sequences, tokenizer, model, BATCH_SIZE, max_length, POOLING)
            out_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
            out_df['target'] = df['label'].values
            out_df.to_csv(output_file, index=False)
            print(f"  saved {output_file}")
