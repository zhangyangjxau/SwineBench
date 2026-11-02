"""GENERator embeddings (single BOS token) of OMIA ref/alt sequences; writes alt-ref differences."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "GenerTeam/GENERator-v2-eukaryote-3b-base"
MAX_LENGTH = 16384
BATCH_SIZE = 8


def trunc6(s):
    return s[len(s) % 6:]


@torch.inference_mode()
def embed(seqs, tok, model, bs):
    out = []
    for i in range(0, len(seqs), bs):
        proc = [tok.bos_token + trunc6(s) for s in seqs[i:i + bs]]
        enc = tok(proc, add_special_tokens=False, padding=True, truncation=True, max_length=MAX_LENGTH,
                  return_tensors="pt").to("cuda")
        h = model(**enc, output_hidden_states=True).hidden_states[-1].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser(description='GENERator embeddings for OMIA variants')
    parser.add_argument('--input', default='data/omia_seqs_matched.csv')
    parser.add_argument('--output-dir', default='embeddings/flank3000_total6000')
    parser.add_argument('--checkpoint', default=MODEL_ID)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    df = pd.read_csv(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    for cls, tag in [(1, "pathogenic"), (0, "common")]:
        sub = df[df["class"] == cls]
        d = embed(sub.alt_seq.tolist(), tok, model, args.batch_size) - embed(sub.ref_seq.tolist(), tok, model, args.batch_size)
        meta = sub[["chromosome", "pos", "ref", "alt", "species"]].reset_index(drop=True)
        outdf = pd.concat([pd.DataFrame(d), meta], axis=1)
        outdf.to_csv(f"{args.output_dir}/generator_differences_{tag}.csv", header=False, index=False)
        print(tag, outdf.shape, flush=True)


if __name__ == "__main__":
    main()
