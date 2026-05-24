#!/usr/bin/env bash
# Run multiple RPS predictor experiments in sequence
# Usage: postdoc submit ./scripts/run_rps_batch.sh

set -e

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(simple_conv simple_conv_bigru simple_conv_attn_pool simple_conv_se_next)
fi

for model in "${MODELS[@]}"; do
    save_dir="results/rps_exp_${model}"
    echo "========================================"
    echo "Training: $model"
    echo "Save path: $save_dir"
    echo "========================================"
    python train_rps_predictor.py \
        --model "$model" \
        --epochs 200 \
        --patience 15 \
        --batch_size 16 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --grad_clip 5.0 \
        --save_path "$save_dir"
done

echo "All experiments complete!"
