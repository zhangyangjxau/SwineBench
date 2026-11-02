#!/usr/bin/env python3
"""Zero-shot variant scoring with HyenaDNA: distances between ref and alt sequence embeddings."""
import argparse
import gc
import gzip
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CONSEQUENCE_SEVERITY = {
    'stop_gained': 1, 'frameshift_variant': 2, 'stop_lost': 3, 'start_lost': 4,
    'splice_acceptor_variant': 5, 'splice_donor_variant': 6, 'inframe_insertion': 7,
    'inframe_deletion': 8, 'missense_variant': 9, 'protein_altering_variant': 10,
    'splice_region_variant': 11, 'splice_polypyrimidine_tract_variant': 12,
    'synonymous_variant': 13, '5_prime_UTR_variant': 14, '3_prime_UTR_variant': 15,
    'non_coding_transcript_exon_variant': 16, 'intron_variant': 17,
    'upstream_gene_variant': 18, 'downstream_gene_variant': 19,
    'regulatory_region_variant': 20, 'TF_binding_site_variant': 21, 'intergenic_variant': 22,
}
METRICS = ['cosine_similarity', 'dot_product', 'l1_distance', 'l2_distance']


@dataclass
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str
    variant_id: str
    consequence: str


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def parse_consequence_from_csq(csq_string: str, alt_allele: str) -> str:
    for annot in csq_string.split(','):
        fields = annot.split('|')
        if len(fields) < 2:
            continue
        if fields[0] and fields[0] != alt_allele:
            continue
        if fields[1]:
            return fields[1]
    return "unknown"


def parse_vcf(vcf_path: str) -> List[Variant]:
    variants = []
    open_func = gzip.open if vcf_path.endswith('.gz') else open
    with open_func(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
            chrom, pos = fields[0], int(fields[1])
            variant_id = fields[2] if fields[2] != '.' else f"{chrom}_{pos}"
            ref, alt = fields[3].upper(), fields[4].upper()
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue
            csq = re.search(r'CSQ=([^;]+)', fields[7])
            consequence = parse_consequence_from_csq(csq.group(1), alt) if csq else "unknown"
            variants.append(Variant(chrom, pos, ref, alt, variant_id, consequence))
    print(f"Parsed {len(variants)} SNVs")
    return variants


def extract_sequences(variant: Variant, genome: Fasta, window: int) -> Tuple[Optional[str], Optional[str]]:
    chrom = next((k for k in (variant.chrom, f"chr{variant.chrom}", f"Chr{variant.chrom}") if k in genome), None)
    if chrom is None:
        return None, None
    chrom_seq = genome[chrom]
    chrom_len = len(chrom_seq)
    pos0 = variant.pos - 1
    if pos0 < 0 or pos0 >= chrom_len or str(chrom_seq[pos0]).upper() != variant.ref:
        return None, None
    start = max(0, pos0 - window)
    end = min(chrom_len, pos0 + window + 1)
    if end - start < window * 2:
        return None, None
    ref_seq = str(chrom_seq[start:end]).upper()
    i = pos0 - start
    return ref_seq, ref_seq[:i] + variant.alt + ref_seq[i + 1:]


def pick_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    capability = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if capability >= 8 else torch.float16 if capability >= 6 else torch.float32


def load_hyenadna_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=pick_dtype(), device_map="auto")
    model.eval()
    return model, tokenizer


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.float()
    attention_mask = attention_mask.float().unsqueeze(-1)
    return torch.sum(hidden_states * attention_mask, dim=1) / torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)


def get_embeddings(sequences: List[str], tokenizer, model, batch_size: int, max_length: int) -> np.ndarray:
    loader = DataLoader(SequenceDataset(sequences), batch_size=batch_size, shuffle=False)
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    all_embeddings = []
    with torch.no_grad():
        for batch_seqs in tqdm(loader, desc="Embedding"):
            tok = tokenizer(list(batch_seqs), padding='max_length', truncation=True, max_length=max_length,
                            return_tensors='pt')
            input_ids = tok["input_ids"].to(model_device)
            attention_mask = (input_ids != tokenizer.pad_token_id).int()
            if model_dtype in (torch.bfloat16, torch.float16) and str(model_device) != "cpu":
                with torch.amp.autocast('cuda', dtype=model_dtype):
                    outputs = model(input_ids)
            else:
                outputs = model(input_ids)
            hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            all_embeddings.append(mean_pooling(hidden, attention_mask).detach().cpu())
            del input_ids, attention_mask, hidden, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0).numpy()


def calculate_all_metrics(ref_embeddings: np.ndarray, alt_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    ref = torch.from_numpy(ref_embeddings).float()
    alt = torch.from_numpy(alt_embeddings).float()
    return {
        'cosine_similarity': cosine_similarity(ref, alt, dim=1).numpy(),
        'dot_product': torch.sum(ref * alt, dim=1).numpy(),
        'l1_distance': torch.sum(torch.abs(ref - alt), dim=1).numpy(),
        'l2_distance': torch.sqrt(torch.sum((ref - alt) ** 2, dim=1)).numpy(),
    }


def sample_variants_by_consequence(variants: List[Variant], sample_per_class: int, seed: int) -> List[Variant]:
    random.seed(seed)
    by_conseq = defaultdict(list)
    for v in variants:
        by_conseq[v.consequence].append(v)
    sampled = []
    for var_list in by_conseq.values():
        sampled.extend(random.sample(var_list, sample_per_class) if len(var_list) >= sample_per_class else var_list)
    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="HyenaDNA zero-shot variant scoring")
    parser.add_argument("--input-vcf", required=True)
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument("--model", default="LongSafari/hyenadna-medium-450k-seqlen-hf")
    parser.add_argument("--window", type=int, default=2000, help="flank size in bp")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None, help="default: window*2+1")
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_length is None:
        args.max_length = args.window * 2 + 1

    variants = parse_vcf(args.input_vcf)
    if args.sample_per_class is not None:
        variants = sample_variants_by_consequence(variants, args.sample_per_class, args.seed)

    genome = Fasta(args.input_fasta, rebuild=False)
    ref_seqs, alt_seqs, valid_variants = [], [], []
    for variant in tqdm(variants, desc="Extracting sequences"):
        ref_seq, alt_seq = extract_sequences(variant, genome, args.window)
        if ref_seq is not None:
            ref_seqs.append(ref_seq)
            alt_seqs.append(alt_seq)
            valid_variants.append(variant)
    print(f"Valid variants: {len(valid_variants)}/{len(variants)}")
    if not valid_variants:
        raise SystemExit("No valid sequences; check reference genome and chromosome naming.")
    del genome
    gc.collect()

    model, tokenizer = load_hyenadna_model(args.model)
    ref_embeddings = get_embeddings(ref_seqs, tokenizer, model, args.batch_size, args.max_length)
    alt_embeddings = get_embeddings(alt_seqs, tokenizer, model, args.batch_size, args.max_length)
    metrics = calculate_all_metrics(ref_embeddings, alt_embeddings)

    df = pd.DataFrame([{
        'chrom': v.chrom, 'pos': v.pos, 'variant_id': v.variant_id, 'ref': v.ref, 'alt': v.alt,
        'consequence': v.consequence, 'severity_rank': CONSEQUENCE_SEVERITY.get(v.consequence, 100),
        **{m: metrics[m][i] for m in METRICS},
    } for i, v in enumerate(valid_variants)])
    df.to_csv(f"{args.output}_scores.tsv", sep='\t', index=False)

    stats = df.groupby('consequence').agg({
        'cosine_similarity': ['count', 'mean', 'std', 'median'],
        'dot_product': ['mean', 'std', 'median'],
        'l1_distance': ['mean', 'std', 'median'],
        'l2_distance': ['mean', 'std', 'median'],
    })
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats.to_csv(f"{args.output}_stats.tsv", sep='\t')
    print(f"Saved {args.output}_scores.tsv and {args.output}_stats.tsv")


if __name__ == "__main__":
    main()
