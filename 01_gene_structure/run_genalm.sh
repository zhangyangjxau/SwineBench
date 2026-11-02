#!/usr/bin/env bash
# GENA-LM (bert-base-t2t-multi): precompute per-base embeddings, then train the CNN probe.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data/bend13_graphpart80}
EMB=${EMB:-embeddings/genalm}
OUT=${OUT:-results/genalm}
MODEL_NAME=${MODEL_NAME:-AIRI-Institute/gena-lm-bert-base-t2t-multi}
mkdir -p "$EMB" "$OUT"

if [[ ! -f "$EMB/train.h5" || ! -f "$EMB/valid.h5" || ! -f "$EMB/test.h5" ]]; then
  python embed_genalm.py --data-dir "$DATA" --output-dir "$EMB" --model-name "$MODEL_NAME" \
    --splits train valid test --target-chunk-bp 1200 --max-tokens 0 --gpu-batch 32 --record-batch 16 \
    --forward-dtype float32 --storage-dtype float16 --compression lzf --log-every 64
fi

python train_probe.py --embedding-dir "$EMB" --output-dir "$OUT" \
  --batch-size 128 --gradient-accumulation-steps 1 --lr 0.003 --weight-decay 0.01 \
  --hidden-size 64 --kernel-size 3 --seed 13 --num-workers 4 --amp --log-every 10 \
  --epochs 100 --patience 20
