#!/usr/bin/env python3
"""Zero-shot variant scoring with Evo2: distances between ref and alt sequence embeddings."""
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
from tqdm import tqdm

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


def load_evo2_model(model_name: str):
    from evo2.models import Evo2
    model = Evo2(model_name)
    return model, model.tokenizer


def get_embeddings_batch(sequences: List[str], model, tokenizer, device: str, layer_name: str) -> Optional[np.ndarray]:
    valid_indices = [i for i, seq in enumerate(sequences) if seq and isinstance(seq, str) and seq.strip()]
    valid_sequences = [sequences[i].upper() for i in valid_indices]
    if not valid_sequences:
        return None
    try:
        batch_token_ids = [tokenizer.tokenize(seq) or [] for seq in valid_sequences]
        max_len = max(len(t) for t in batch_token_ids)
        if max_len == 0:
            return None
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
        raise
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        return None


def get_embeddings(sequences: List[str], model, tokenizer, device: str, batch_size: int, layer_name: str) -> np.ndarray:
    all_embeddings = []
    for start in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
        batch = sequences[start:start + batch_size]
        emb = get_embeddings_batch(batch, model, tokenizer, device, layer_name)
        all_embeddings.append(emb if emb is not None else np.full((len(batch), 1), np.nan))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.vstack(all_embeddings)


def calculate_all_metrics(ref_embeddings: np.ndarray, alt_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    valid = ~(np.isnan(ref_embeddings).any(axis=1) | np.isnan(alt_embeddings).any(axis=1))
    out = {m: np.full(len(ref_embeddings), np.nan) for m in METRICS}
    if valid.sum() > 0:
        ref = torch.from_numpy(ref_embeddings[valid]).float()
        alt = torch.from_numpy(alt_embeddings[valid]).float()
        out['cosine_similarity'][valid] = cosine_similarity(ref, alt, dim=1).numpy()
        out['dot_product'][valid] = torch.sum(ref * alt, dim=1).numpy()
        out['l1_distance'][valid] = torch.sum(torch.abs(ref - alt), dim=1).numpy()
        out['l2_distance'][valid] = torch.sqrt(torch.sum((ref - alt) ** 2, dim=1)).numpy()
    return out


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
    parser = argparse.ArgumentParser(description="Evo2 zero-shot variant scoring")
    parser.add_argument("--input-vcf", required=True)
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument("--model", default="evo2_7b")
    parser.add_argument("--layer", default="blocks.20.mlp.l3")
    parser.add_argument("--window", type=int, default=2000, help="flank size in bp")
    parser.add_argument("--batch-size", type=int, default=8)
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

    model, tokenizer = load_evo2_model(args.model)
    ref_embeddings = get_embeddings(ref_seqs, model, tokenizer, device, args.batch_size, args.layer)
    alt_embeddings = get_embeddings(alt_seqs, model, tokenizer, device, args.batch_size, args.layer)
    metrics = calculate_all_metrics(ref_embeddings, alt_embeddings)
    print(f"Valid scores: {(~np.isnan(metrics['cosine_similarity'])).sum()}/{len(valid_variants)}")

    df = pd.DataFrame([{
        'chrom': v.chrom, 'pos': v.pos, 'variant_id': v.variant_id, 'ref': v.ref, 'alt': v.alt,
        'consequence': v.consequence, 'severity_rank': CONSEQUENCE_SEVERITY.get(v.consequence, 100),
        **{m: metrics[m][i] for m in METRICS},
    } for i, v in enumerate(valid_variants)])
    df.to_csv(f"{args.output}_scores.tsv", sep='\t', index=False)

    stats = df.dropna(subset=['cosine_similarity']).groupby('consequence').agg({
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
