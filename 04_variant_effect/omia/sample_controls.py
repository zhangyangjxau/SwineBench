#!/usr/bin/env python3
"""Reservoir-sample control SNVs from Ensembl VCFs: genome-wide (blind) and per functional category (matched)."""
import argparse
import gzip
import random
import time
from pathlib import Path

import pandas as pd

CONSEQUENCE_TO_CATEGORY = {
    'missense_variant': 'missense',
    'stop_gained': 'nonsense (stop-gain)',
    'stop_lost': 'extension (stop-lost)',
    'splice_donor_variant': 'splicing',
    'splice_acceptor_variant': 'splicing',
    'splice_region_variant': 'splicing',
    'start_lost': 'start-lost',
    'regulatory_region_variant': 'regulatory',
    'TF_binding_site_variant': 'regulatory',
}

# VCF filename prefix -> OMIA species name and assembly (must match GENOME_NAME_TO_ACCESSION in extract_seqs.py)
VCF_SPECIES_MAP = {
    'bos_taurus': {'omia_name': 'taurine cattle', 'assembly': 'ARS-UCD1.2'},
    'canis_lupus_familiaris': {'omia_name': 'dog', 'assembly': 'ROS_Cfam_1.0'},
    'equus_caballus': {'omia_name': 'domestic horse', 'assembly': 'EquCab3.0'},
    'felis_catus': {'omia_name': 'domestic cat', 'assembly': 'Felis_catus_9.0'},
    'gallus_gallus': {'omia_name': 'chicken', 'assembly': 'GRCg6a'},
    'capra_hircus': {'omia_name': 'goat', 'assembly': 'ARS1.2'},
    'ovis_aries_rambouillet': {'omia_name': 'sheep', 'assembly': 'Oar_rambouillet_v1.0'},
    'sus_scrofa': {'omia_name': 'pig', 'assembly': 'Sscrofa11.1'},
}


def print_vcf_reference(vcf_path):
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('##reference'):
                print(f"  VCF reference: {line.strip()}")
            if not line.startswith('#'):
                break


def sample_from_vcf(vcf_path, blind_target, category_targets, seed):
    """Single-pass reservoir sampling for the blind set and every category set."""
    rng = random.Random(seed)
    print_vcf_reference(vcf_path)

    blind_reservoir, blind_count = [], 0
    cat_reservoirs = {cat: [] for cat in category_targets}
    cat_counts = {cat: 0 for cat in category_targets}
    total_lines = 0

    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            total_lines += 1
            fields = line.split('\t', 8)
            if len(fields) < 8:
                continue
            chrom, pos, ref, alt_field, info = fields[0], fields[1], fields[3], fields[4], fields[7]

            for alt in alt_field.split(','):
                if len(ref) != 1 or len(alt) != 1 or ref not in 'ACGT' or alt not in 'ACGT' or ref == alt:
                    continue

                # consequence from the VE= INFO field: csq|idx|feature_type|feature_id,...
                category = None
                ve_pos = info.find(';VE=')
                if ve_pos != -1:
                    val_start = ve_pos + 4
                elif info.startswith('VE='):
                    val_start, ve_pos = 3, 0
                if ve_pos != -1:
                    ve_end = info.find(';', val_start)
                    ve_str = info[val_start:ve_end] if ve_end != -1 else info[val_start:].rstrip('\n')
                    for annotation in ve_str.split(','):
                        cat = CONSEQUENCE_TO_CATEGORY.get(annotation.split('|')[0])
                        if cat is not None:
                            category = cat
                            break

                snv = (chrom, pos, ref, alt, category)

                blind_count += 1
                if blind_count <= blind_target:
                    blind_reservoir.append(snv)
                else:
                    j = rng.randint(0, blind_count - 1)
                    if j < blind_target:
                        blind_reservoir[j] = snv

                if category in cat_reservoirs:
                    cat_counts[category] += 1
                    target = category_targets[category]
                    if target == 0:
                        continue
                    if cat_counts[category] <= target:
                        cat_reservoirs[category].append(snv)
                    else:
                        j = rng.randint(0, cat_counts[category] - 1)
                        if j < target:
                            cat_reservoirs[category][j] = snv

    stats = {'total_lines': total_lines, 'total_snvs': blind_count, 'category_counts': cat_counts}
    return blind_reservoir, cat_reservoirs, stats


def main():
    parser = argparse.ArgumentParser(description='Sample control SNVs from Ensembl VCFs')
    parser.add_argument('omia_csv', help='OMIA variants with sequences (Processing_Status column)')
    parser.add_argument('vcf_dir', help='directory with <species>_incl_consequences.vcf.gz')
    parser.add_argument('--ratio', type=int, default=10, help='controls per OMIA variant')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-blind', default='data/control_snvs_type_blind.csv')
    parser.add_argument('--output-matched', default='data/control_snvs_type_matched.csv')
    args = parser.parse_args()

    omia_df = pd.read_csv(args.omia_csv)
    if 'Processing_Status' in omia_df.columns:
        omia_df = omia_df[omia_df['Processing_Status'] == 'Success']
    omia_df = omia_df[omia_df['Species Name'] != 'rabbit']
    print(f"OMIA variants (post-QC, no rabbit): {len(omia_df)}")

    species_totals = omia_df['Species Name'].value_counts().to_dict()
    species_cat_counts = omia_df.groupby(['Species Name', 'Variant Effect']).size()

    all_blind_records, all_matched_records = [], []
    vcf_dir = Path(args.vcf_dir)

    for vcf_key, sp_info in VCF_SPECIES_MAP.items():
        omia_name, assembly = sp_info['omia_name'], sp_info['assembly']
        if omia_name not in species_totals:
            print(f"Skipping {vcf_key}: no OMIA variants for '{omia_name}'")
            continue
        vcf_file = vcf_dir / f"{vcf_key}_incl_consequences.vcf.gz"
        if not vcf_file.is_file():
            print(f"Warning: VCF not found: {vcf_file}")
            continue

        print(f"\n{omia_name} ({vcf_key}, {assembly}): {vcf_file}")
        blind_target = species_totals[omia_name] * args.ratio
        category_targets = {}
        for cat in CONSEQUENCE_TO_CATEGORY.values():
            key = (omia_name, cat)
            if key in species_cat_counts.index:
                category_targets[cat] = int(species_cat_counts[key]) * args.ratio
        category_targets = dict(sorted(set(category_targets.items())))
        print(f"Blind target: {blind_target}; matched targets: {category_targets}")

        start_time = time.time()
        per_species_seed = args.seed + hash(vcf_key) % 10000
        blind_sample, matched_samples, stats = sample_from_vcf(
            str(vcf_file), blind_target, category_targets, seed=per_species_seed)
        print(f"  {stats['total_snvs']:,} SNVs in {stats['total_lines']:,} lines ({time.time() - start_time:.0f}s)")
        print(f"  Blind sampled: {len(blind_sample)} / {blind_target}")
        for cat, samples in matched_samples.items():
            print(f"  Matched '{cat}': {len(samples)} / {category_targets.get(cat, 0)} "
                  f"({stats['category_counts'].get(cat, 0):,} available)")

        for chrom, pos, ref_allele, alt_allele, category in blind_sample:
            all_blind_records.append({
                'Species Name': omia_name, 'Reference Sequence': assembly, 'Chr': chrom, 'Position': int(pos),
                'Ref Allele': ref_allele, 'Alt Allele': alt_allele,
                'Variant Effect': category if category else 'other',
                'VCF_Source': f"{vcf_key}_incl_consequences.vcf.gz",
            })
        for cat, samples in matched_samples.items():
            for chrom, pos, ref_allele, alt_allele, _ in samples:
                all_matched_records.append({
                    'Species Name': omia_name, 'Reference Sequence': assembly, 'Chr': chrom, 'Position': int(pos),
                    'Ref Allele': ref_allele, 'Alt Allele': alt_allele, 'Variant Effect': cat,
                    'VCF_Source': f"{vcf_key}_incl_consequences.vcf.gz",
                })

    if all_blind_records:
        blind_df = pd.DataFrame(all_blind_records)
        Path(args.output_blind).parent.mkdir(parents=True, exist_ok=True)
        blind_df.to_csv(args.output_blind, index=False)
        print(f"\nBlind control set: {len(blind_df)} variants -> {args.output_blind}")
    if all_matched_records:
        matched_df = pd.DataFrame(all_matched_records)
        Path(args.output_matched).parent.mkdir(parents=True, exist_ok=True)
        matched_df.to_csv(args.output_matched, index=False)
        print(f"Matched control set: {len(matched_df)} variants -> {args.output_matched}")
        print(pd.crosstab(matched_df['Variant Effect'], matched_df['Species Name'], margins=True).to_string())


if __name__ == '__main__':
    main()
