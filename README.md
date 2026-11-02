# SwineBench

## Setup

```bash
conda env create -f environment.yml && conda activate swinebench
```

DNABERT-2 needs `transformers==4.29`; Evo2 needs the [evo2](https://github.com/ArcInstitute/evo2) package. Model weights are downloaded from Hugging Face on first use.

## Usage

Each folder reads `data/` and writes `embeddings/` and `results/`; paths can be changed via command-line arguments.

```bash
# 01 gene structure
python build_dataset.py --fasta Sus_scrofa.fa --gtf Sus_scrofa.gtf --outdir data/bend13
python split_graphpart.py --data-dir data/bend13 --outdir data/bend13_graphpart80
bash run_all.sh                             # embed_<model>.py + train_probe.py for each model

# 02 regulatory classification
python embed_ntv2.py                        # likewise embed_<model>.py
python rf.py --model NT-v2 && python cnn.py --seeds 42 13 17 123 997

# 03 gene expression
python filter_egenes.py && python prepare_genes.py && python gene_table.py
python export_sequences.py --mode both --output data/sequences
python embed_ntv2.py --mode both --gzip     # likewise embed_<model>.py
python prepare_features.py ntv2 dnabert2 genalm hyenadna evo2 generator && python prepare_labels.py
python mlp.py --model ntv2 && python snp_baselines.py

# 04 variant effect
python vep/score_ntv2.py --input-vcf variants.vcf --input-fasta Sus_scrofa.fa --output results/scores/ntv2 --window 3000 --sample-per-class 980
python omia/extract_snvs.py omia.tsv && python omia/extract_seqs.py && python omia/sample_controls.py data/omia_snv_sequences.csv vcf_dir/ && python omia/extract_control_seqs.py
python omia/build_dataset.py && python omia/embed_ntv2.py && python omia/rf.py
```
