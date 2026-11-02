#!/usr/bin/env python3
"""Extract NT-v2 (500M multi-species) mean-pooled embeddings (TSS +/-3 kb, haplotypes averaged) for each gene and individual."""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

MODEL_PREFIX = "nt-v2-500m"
CFG = {
    "hf_name": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "use_automodel": False,
    "use_encoder_attn": True,
    "max_length": 2000,
}

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "data" / "inputs"


def get_tss(gene_row):
    if "tss" in gene_row and not pd.isna(gene_row.get("tss", np.nan)):
        return int(gene_row["tss"])
    return int(gene_row["start"]) if gene_row["strand"] == "+" else int(gene_row["end"])


def convert_sample_name(vcf_name):
    for prefix in ("Duroc_", "Landrace_", "LargeWhite_"):
        if vcf_name.startswith(prefix):
            return vcf_name[len(prefix):]
    return vcf_name


class FastSequenceExtractor:
    """Extract reference and phased haplotype sequences with pysam."""

    def __init__(self, ref_genome, vcf_file=None):
        self.ref = pysam.FastaFile(ref_genome)
        self.vcf = pysam.VariantFile(vcf_file) if vcf_file else None
        self.samples = list(self.vcf.header.samples) if self.vcf else []

    def close(self):
        self.ref.close()
        if self.vcf:
            self.vcf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_ref_seq(self, chrom, tss, window):
        start = max(0, int(tss) - int(window))
        end = int(tss) + int(window)
        try:
            seq = self.ref.fetch(str(chrom), start, end).upper()
            return seq if len(seq) >= 10 else "N" * (window * 2)
        except (ValueError, KeyError):
            return "N" * (window * 2)

    def get_all_haplotype_seqs(self, chrom, tss, window):
        start = max(0, int(tss) - int(window))
        end = int(tss) + int(window)
        ref_seq = self.get_ref_seq(chrom, tss, window)
        region_len = len(ref_seq)

        if not self.vcf or not self.samples:
            return {s: (ref_seq, ref_seq) for s in self.samples}

        variants = []
        try:
            for record in self.vcf.fetch(str(chrom), start, end):
                rel_pos = record.pos - 1 - start
                if rel_pos < 0 or rel_pos >= region_len:
                    continue
                ref_allele = record.ref
                alt_alleles = list(record.alts) if record.alts else []
                if not alt_alleles:
                    continue
                all_alleles = [ref_allele] + alt_alleles
                sample_gts = {}
                for sample in self.samples:
                    gt = record.samples[sample].get("GT", (None, None))
                    if gt is None or None in gt:
                        sample_gts[sample] = (0, 0)
                    else:
                        sample_gts[sample] = (gt[0], gt[1])
                variants.append({
                    "pos": rel_pos,
                    "ref": ref_allele,
                    "alleles": all_alleles,
                    "sample_gts": sample_gts,
                })
        except (ValueError, KeyError):
            return {s: (ref_seq, ref_seq) for s in self.samples}

        if not variants:
            return {s: (ref_seq, ref_seq) for s in self.samples}

        variants.sort(key=lambda v: v["pos"], reverse=True)
        result = {}
        for sample in self.samples:
            seq_hap1 = list(ref_seq)
            seq_hap2 = list(ref_seq)
            for var in variants:
                pos = var["pos"]
                ref_len = len(var["ref"])
                gt0, gt1 = var["sample_gts"].get(sample, (0, 0))
                alleles = var["alleles"]
                if gt0 > 0 and gt0 < len(alleles):
                    seq_hap1[pos:pos + ref_len] = list(alleles[gt0])
                if gt1 > 0 and gt1 < len(alleles):
                    seq_hap2[pos:pos + ref_len] = list(alleles[gt1])
            result[sample] = ("".join(seq_hap1), "".join(seq_hap2))
        return result


def load_model(device, checkpoint=None, max_length=None):
    cfg = CFG.copy()
    if max_length is not None:
        cfg["max_length"] = max_length
    model_path = checkpoint or cfg["hf_name"]
    local_only = checkpoint is not None

    print(f"Loading model: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
    if cfg["use_automodel"]:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
    model = model.to(device).eval()
    print(f"  max_length = {cfg['max_length']} tokens")
    return model, tokenizer, cfg


def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed_one_allele(sequences, model, tokenizer, cfg, device):
    tok = tokenizer(sequences, padding="max_length", truncation=True, max_length=cfg["max_length"], return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}

    if cfg["use_automodel"]:
        outputs = model(tok["input_ids"], attention_mask=tok["attention_mask"])
        hidden = outputs[0]
    else:
        kwargs = {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "output_hidden_states": True,
        }
        if cfg["use_encoder_attn"]:
            kwargs["encoder_attention_mask"] = tok["attention_mask"]
        outputs = model(**kwargs)
        hidden = outputs.hidden_states[-1]
    return mean_pooling(hidden, tok["attention_mask"])


def embed_diploid_batch(allele1_list, allele2_list, model, tokenizer, cfg, device, batch_size):
    all_embs = []
    for i in range(0, len(allele1_list), batch_size):
        a1 = allele1_list[i:i + batch_size]
        a2 = allele2_list[i:i + batch_size]
        with torch.no_grad():
            emb1 = embed_one_allele(a1, model, tokenizer, cfg, device)
            emb2 = embed_one_allele(a2, model, tokenizer, cfg, device)
            emb_avg = ((emb1 + emb2) / 2.0).cpu().numpy()
        all_embs.append(emb_avg)
        del emb1, emb2
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(all_embs)


def save_df(df, path, compress):
    if compress:
        path = path if path.endswith(".gz") else path + ".gz"
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)
    return path

def run_ref_embeddings(genes_df, extractor, model, tokenizer, cfg, device, args):
    print("\n=== NT-v2 Reference Embeddings ===")
    seqs, gene_ids = [], []
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="Ref sequences"):
        seqs.append(extractor.get_ref_seq(str(gene["chr"]), get_tss(gene), args.window))
        gene_ids.append(gene["gene_id"])

    embeddings = embed_diploid_batch(seqs, seqs, model, tokenizer, cfg, device, args.batch_size)
    emb_cols = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, "subject_id", "reference")
    df.insert(1, "gene_id", gene_ids)
    out_path = os.path.join(args.output_dir, f"{args.prefix}_ref_embeddings.csv")
    print(f"Saved: {save_df(df, out_path, args.gzip)} | shape={df.shape}")


def run_individual_embeddings(genes_df, extractor, model, tokenizer, cfg, device, args):
    print("\n=== NT-v2 Individual Embeddings ===")
    samples = extractor.samples
    print(f"{len(genes_df)} genes x {len(samples)} individuals")

    n_batches = (len(genes_df) + args.genes_per_batch - 1) // args.genes_per_batch
    for batch_idx in range(n_batches):
        start = batch_idx * args.genes_per_batch
        end = min(start + args.genes_per_batch, len(genes_df))
        batch_genes = genes_df.iloc[start:end]

        out_path = os.path.join(args.output_dir, f"{args.prefix}_ind_{batch_idx + 1}.csv")
        check_path = out_path + ".gz" if args.gzip else out_path
        if os.path.exists(check_path):
            print(f"Batch {batch_idx + 1}/{n_batches}: skipped exists")
            continue

        print(f"\nBatch {batch_idx + 1}/{n_batches}: genes {start + 1}-{end}")
        batch_records = []
        for _, gene in tqdm(batch_genes.iterrows(), total=len(batch_genes), desc=f"Batch {batch_idx + 1}"):
            gene_id = gene["gene_id"]
            hap_seqs = extractor.get_all_haplotype_seqs(str(gene["chr"]), get_tss(gene), args.window)

            allele1_list, allele2_list, subject_ids = [], [], []
            for vcf_sample in samples:
                a1, a2 = hap_seqs.get(vcf_sample, ("N" * args.window * 2, "N" * args.window * 2))
                allele1_list.append(a1)
                allele2_list.append(a2)
                subject_ids.append(convert_sample_name(vcf_sample))

            embeddings = embed_diploid_batch(allele1_list, allele2_list, model, tokenizer, cfg, device, args.batch_size)
            for i, subject_id in enumerate(subject_ids):
                record = {"subject_id": subject_id, "gene_id": gene_id}
                for j in range(embeddings.shape[1]):
                    record[f"embedding_{j}"] = embeddings[i, j]
                batch_records.append(record)

            del hap_seqs, allele1_list, allele2_list, embeddings
            gc.collect()

        df = pd.DataFrame(batch_records)
        print(f"Saved: {save_df(df, out_path, args.gzip)} | shape={df.shape}")
        del df, batch_records
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Local model checkpoint (overrides HuggingFace)")
    parser.add_argument("--prefix", default=MODEL_PREFIX)
    parser.add_argument("--mode", default="both", choices=["ref", "individual", "both"])
    parser.add_argument("--genes", default=str(INPUTS / "genes/genes_cis_eGenes.csv"))
    parser.add_argument("--ref", default=str(INPUTS / "reference/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"))
    parser.add_argument("--vcf", default=str(INPUTS / "variants/ALL_300.phased.renamed.vcf.gz"))
    parser.add_argument("--window", type=int, default=3000)
    parser.add_argument("--max-length", type=int, default=None, help="Override tokenizer max_length")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--genes-per-batch", type=int, default=50)
    parser.add_argument("--gene-start", type=int, default=None)
    parser.add_argument("--gene-end", type=int, default=None)
    parser.add_argument("--chr", default=None)
    parser.add_argument("--random-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(ROOT / "embeddings" / MODEL_PREFIX))
    parser.add_argument("--gzip", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    genes_df = pd.read_csv(args.genes)
    total = len(genes_df)
    if args.chr is not None:
        genes_df = genes_df[genes_df["chr"].astype(str) == str(args.chr)].reset_index(drop=True)
        print(f"Filtered to chr{args.chr}: {len(genes_df)}/{total}")
    if args.gene_start is not None or args.gene_end is not None:
        genes_df = genes_df.iloc[args.gene_start or 0:args.gene_end].reset_index(drop=True)
    if args.random_sample is not None and args.random_sample < len(genes_df):
        genes_df = genes_df.sample(n=args.random_sample, random_state=args.seed).reset_index(drop=True)
    print(f"Genes to process: {len(genes_df)}")

    device = torch.device(args.device)
    model, tokenizer, cfg = load_model(device, args.checkpoint, args.max_length)
    vcf_path = args.vcf if args.mode in ("individual", "both") else None
    with FastSequenceExtractor(args.ref, vcf_path) as extractor:
        if args.mode in ("ref", "both"):
            run_ref_embeddings(genes_df, extractor, model, tokenizer, cfg, device, args)
        if args.mode in ("individual", "both"):
            run_individual_embeddings(genes_df, extractor, model, tokenizer, cfg, device, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
