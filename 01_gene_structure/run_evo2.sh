#!/usr/bin/env bash
# Evo2 7B: precompute per-base embeddings (blocks.20.mlp.l3, projected to 1024 dims), then train the CNN probe.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data/bend13_graphpart80}
EMB=${EMB:-embeddings/evo2}
OUT=${OUT:-results/evo2}
MODEL_NAME=${MODEL_NAME:-evo2_7b}
LAYER_NAME=${LAYER_NAME:-blocks.20.mlp.l3}
PROJECT_DIM=${PROJECT_DIM:-1024}
TARGET_CHUNK_BP=${TARGET_CHUNK_BP:-14000}
LOCAL_PATH=${LOCAL_PATH:-}   # optional local Evo2 checkpoint
mkdir -p "$EMB" "$OUT"

if [[ ! -f "$EMB/train.h5" || ! -f "$EMB/valid.h5" || ! -f "$EMB/test.h5" ]]; then
  python embed_evo2.py --data-dir "$DATA" --output-dir "$EMB" --model-name "$MODEL_NAME" \
    --layer-name "$LAYER_NAME" --target-chunk-bp "$TARGET_CHUNK_BP" --gpu-batch 1 --record-batch 1 \
    --storage-dtype float16 --project-dim "$PROJECT_DIM" --compression lzf --log-every 16 \
    ${LOCAL_PATH:+--local-path "$LOCAL_PATH"}
fi

python train_probe.py --embedding-dir "$EMB" --output-dir "$OUT" \
  --batch-size 128 --gradient-accumulation-steps 1 --lr 0.003 --weight-decay 0.01 \
  --hidden-size 64 --kernel-size 3 --seed 13 --num-workers 4 --amp --log-every 10 \
  --epochs 100 --patience 20
