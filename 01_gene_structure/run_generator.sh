#!/usr/bin/env bash
# GENERator eukaryote 3B: precompute per-base embeddings (with BOS), then train the CNN probe.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data/bend13_graphpart80}
EMB=${EMB:-embeddings/generator}
OUT=${OUT:-results/generator}
MODEL_NAME=${MODEL_NAME:-GenerTeam/GENERator-v2-eukaryote-3b-base}
mkdir -p "$EMB" "$OUT"

if [[ ! -f "$EMB/train.h5" || ! -f "$EMB/valid.h5" || ! -f "$EMB/test.h5" ]]; then
  python embed_generator.py --data-dir "$DATA" --output-dir "$EMB" --model-name "$MODEL_NAME" \
    --splits train valid test --target-chunk-bp 12000 --max-tokens 0 --gpu-batch 1 --record-batch 4 \
    --dtype float32 --compression lzf --log-every 32 --prepend-bos
fi

python train_probe.py --embedding-dir "$EMB" --output-dir "$OUT" \
  --batch-size 32 --gradient-accumulation-steps 1 --lr 0.003 --weight-decay 0.01 \
  --hidden-size 64 --kernel-size 3 --seed 13 --num-workers 4 --amp --log-every 10 \
  --epochs 100 --patience 20
