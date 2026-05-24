#!/usr/bin/env bash
# Run two RPS predictor experiments in parallel on 2 GPUs
# Usage: postdoc submit ./scripts/run_rps_parallel.sh model1 model2

set -e

MODEL1="${1:-simple_conv}"
MODEL2="${2:-simple_conv_bigru}"

run_exp() {
    local model="$1"
    local gpu="$2"
    local save_dir="results/rps_exp_${model}"
    echo "[GPU $gpu] Training: $model -> $save_dir"
    CUDA_VISIBLE_DEVICES="$gpu" python train_rps_predictor.py \
        --model "$model" \
        --device "cuda:$gpu" \
        --epochs 200 \
        --patience 15 \
        --batch_size 16 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --grad_clip 5.0 \
        --save_path "$save_dir" \
        2>&1 | tee "results/rps_exp_${model}/log.txt"
}

mkdir -p "results/rps_exp_${MODEL1}" "results/rps_exp_${MODEL2}"

# Run both in background
run_exp "$MODEL1" 0 &
PID1=$!
run_exp "$MODEL2" 1 &
PID2=$!

# Wait for both to finish
wait $PID1
wait $PID2

echo "Both experiments complete!"
