#!/usr/bin/env bash
# Precompute embeddings and train the CNN probe for all models on the GraphPart-80 split.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data/bend13_graphpart80}

DATA=$DATA EMB=embeddings/ntv2      OUT=results/ntv2      bash run_ntv2.sh
DATA=$DATA EMB=embeddings/dnabert2  OUT=results/dnabert2  bash run_dnabert2.sh
DATA=$DATA EMB=embeddings/genalm    OUT=results/genalm    bash run_genalm.sh
DATA=$DATA EMB=embeddings/hyenadna  OUT=results/hyenadna  MODEL_NAME=LongSafari/hyenadna-medium-450k-seqlen-hf bash run_hyenadna.sh
DATA=$DATA EMB=embeddings/generator OUT=results/generator bash run_generator.sh
DATA=$DATA EMB=embeddings/evo2      OUT=results/evo2      bash run_evo2.sh
