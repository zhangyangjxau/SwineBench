#!/usr/bin/env python3
"""Export TSS +/-3 kb reference and phased haplotype sequences (before model-specific cropping) as FASTA."""
import argparse
import csv
import gzip
from pathlib import Path

import pysam

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "data" / "inputs"
WINDOW = 3000


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ref", "individual", "both"], default="ref")
    parser.add_argument("--gene-start", type=int, default=0)
    parser.add_argument("--gene-end", type=int)
    parser.add_argument("--genes", default=str(INPUTS / "genes/genes_cis_eGenes.csv"))
    parser.add_argument("--ref", default=str(INPUTS / "reference/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"))
    parser.add_argument("--vcf", default=str(INPUTS / "variants/ALL_300.phased.renamed.vcf.gz"))
    parser.add_argument("--output", required=True, help="New empty output directory")
    args = parser.parse_args()

    out = Path(args.output).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("Output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    with open(args.genes) as f:
        genes = list(csv.DictReader(f))[args.gene_start:args.gene_end]
    vcf = args.vcf if args.mode != "ref" else None

    with FastSequenceExtractor(args.ref, vcf) as extractor:
        if args.mode in ("ref", "both"):
            with gzip.open(out / "reference_tss_6kb.fa.gz", "wt") as f:
                for gene in genes:
                    seq = extractor.get_ref_seq(gene["chr"], int(gene["tss"]), WINDOW)
                    f.write(">" + gene["gene_id"] + "|reference\n" + seq + "\n")
        if args.mode in ("individual", "both"):
            for index in range(0, len(genes), 50):
                with gzip.open(out / ("haplotypes_%03d.fa.gz" % (index // 50 + 1)), "wt") as f:
                    for gene in genes[index:index + 50]:
                        seqs = extractor.get_all_haplotype_seqs(gene["chr"], int(gene["tss"]), WINDOW)
                        for sample, (a1, a2) in seqs.items():
                            sid = convert_sample_name(sample)
                            f.write(">" + gene["gene_id"] + "|" + sid + "|hap1\n" + a1 + "\n")
                            f.write(">" + gene["gene_id"] + "|" + sid + "|hap2\n" + a2 + "\n")
                print("Exported genes", index + 1, min(index + 50, len(genes)), flush=True)
    print("DONE", len(genes), "genes", out)


if __name__ == "__main__":
    main()
