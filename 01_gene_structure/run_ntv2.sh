#!/usr/bin/env bash
# NT-v2 500M: precompute per-base embeddings, then train the CNN probe.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data/bend13_graphpart80}
EMB=${EMB:-embeddings/ntv2}
OUT=${OUT:-results/ntv2}
MODEL_NAME=${MODEL_NAME:-InstaDeepAI/nucleotide-transformer-v2-500m-multi-species}
mkdir -p "$EMB" "$OUT"

if [[ ! -f "$EMB/train.h5" || ! -f "$EMB/valid.h5" || ! -f "$EMB/test.h5" ]]; then
  python embed_ntv2.py --data-dir "$DATA" --output-dir "$EMB" --model-name "$MODEL_NAME" \
    --splits train valid test --max-chunk-bp 12000 --gpu-batch 12 --record-batch 64 \
    --dtype float16 --compression lzf --log-every 64
fi

python train_probe.py --embedding-dir "$EMB" --output-dir "$OUT" \
  --epochs 100 --batch-size 128 --gradient-accumulation-steps 1 --lr 0.003 --weight-decay 0.01 \
  --hidden-size 64 --kernel-size 3 --seed 13 --num-workers 4 --amp --log-every 10 --patience 20
