#!/usr/bin/env bash
# DNABERT-2 117M: precompute per-base embeddings, then train the CNN probe.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data/bend13_graphpart80}
EMB=${EMB:-embeddings/dnabert2}
OUT=${OUT:-results/dnabert2}
MODEL_NAME=${MODEL_NAME:-zhihan1996/DNABERT-2-117M}
mkdir -p "$EMB" "$OUT"

if [[ ! -f "$EMB/train.h5" || ! -f "$EMB/valid.h5" || ! -f "$EMB/test.h5" ]]; then
  python embed_dnabert2.py --data-dir "$DATA" --output-dir "$EMB" --model-name "$MODEL_NAME" \
    --splits train valid test --target-chunk-bp 10000 --max-tokens 0 --gpu-batch 8 --record-batch 32 \
    --dtype float16 --compression lzf --log-every 64
fi

python train_probe.py --embedding-dir "$EMB" --output-dir "$OUT" \
  --batch-size 128 --gradient-accumulation-steps 1 --lr 0.003 --weight-decay 0.01 \
  --hidden-size 64 --kernel-size 3 --seed 13 --num-workers 4 --amp --log-every 10 \
  --epochs 100 --patience 20
