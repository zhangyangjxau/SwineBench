#!/usr/bin/env python3
"""Zero-shot variant scoring with GENA-LM: distances between ref and alt sequence embeddings."""
import argparse
import gc
import gzip
import importlib
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
    if 'N' in ref_seq:
        return None, None
    i = pos0 - start
    return ref_seq, ref_seq[:i] + variant.alt + ref_seq[i + 1:]


def load_genalm_model(model_path: str, device: str):
    model_temp = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    gena_module_name = model_temp.__class__.__module__
    del model_temp
    BertModel = getattr(importlib.import_module(gena_module_name), 'BertModel')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.to(device).eval()
    return model, tokenizer


def center_crop(seq: str, target_bp: int = 4500) -> str:
    if len(seq) <= target_bp:
        return seq
    start = (len(seq) - target_bp) // 2
    return seq[start:start + target_bp]


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask = attention_mask.unsqueeze(-1).float()
    return torch.sum(hidden_states * attention_mask, dim=1) / torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)


def get_embeddings(sequences: List[str], tokenizer, model, device: str, batch_size: int, max_length: int,
                   pooling: str) -> np.ndarray:
    loader = DataLoader(SequenceDataset(sequences), batch_size=batch_size, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for batch_seqs in tqdm(loader, desc="Embedding"):
            tok = tokenizer([center_crop(s) for s in batch_seqs], padding='max_length', truncation=True,
                            max_length=max_length, return_tensors='pt')
            tok = {k: v.to(device) for k, v in tok.items()}
            hidden_states = model(**tok).last_hidden_state
            if pooling == "cls":
                emb = hidden_states[:, 0, :]
            else:
                emb = mean_pooling(hidden_states, tok["attention_mask"])
            all_embeddings.append(emb.cpu())
            del tok, hidden_states, emb
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
    parser = argparse.ArgumentParser(description="GENA-LM zero-shot variant scoring")
    parser.add_argument("--input-vcf", required=True)
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument("--model", default="AIRI-Institute/gena-lm-bert-base-t2t-multi")
    parser.add_argument("--window", type=int, default=2000, help="flank size in bp")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

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

    model, tokenizer = load_genalm_model(args.model, device)
    ref_embeddings = get_embeddings(ref_seqs, tokenizer, model, device, args.batch_size, args.max_length, args.pooling)
    alt_embeddings = get_embeddings(alt_seqs, tokenizer, model, device, args.batch_size, args.max_length, args.pooling)
    metrics = calculate_all_metrics(ref_embeddings, alt_embeddings)

    df = pd.DataFrame([{
        'chrom': v.chrom, 'pos': v.pos, 'variant_id': v.variant_id, 'ref': v.ref, 'alt': v.alt,
        'consequence': v.consequence, 'severity_rank': CONSEQUENCE_SEVERITY.get(v.consequence, 100),
        **{m: metrics[m][i] for m in METRICS},
    } for i, v in enumerate(valid_variants)])
    suffix = "_cls" if args.pooling == "cls" else "_meanpool"
    df.to_csv(f"{args.output}{suffix}_scores.tsv", sep='\t', index=False)

    stats = df.groupby('consequence').agg({
        'cosine_similarity': ['count', 'mean', 'std', 'median'],
        'dot_product': ['mean', 'std', 'median'],
        'l1_distance': ['mean', 'std', 'median'],
        'l2_distance': ['mean', 'std', 'median'],
    })
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats.to_csv(f"{args.output}{suffix}_stats.tsv", sep='\t')
    print(f"Saved {args.output}{suffix}_scores.tsv and {args.output}{suffix}_stats.tsv")


if __name__ == "__main__":
    main()
