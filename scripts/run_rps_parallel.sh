#!/usr/bin/env bash
# Run two RPS predictor experiments in parallel on 2 GPUs
# Usage: postdoc submit ./scripts/run_rps_parallel.sh model1 model2

MODEL1="${1:-simple_conv}"
MODEL2="${2:-simple_conv_bigru}"

run_exp() {
    local model="$1"
    local gpu="$2"
    local save_dir="results/rps_exp_${model}"
    echo "[GPU $gpu] Training: $model -> $save_dir"
    mkdir -p "$save_dir"
    CUDA_VISIBLE_DEVICES="$gpu" python train_rps_predictor.py \
        --model "$model" \
        --device "cuda:0" \
        --epochs 200 \
        --patience 15 \
        --batch_size 16 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --grad_clip 5.0 \
        --save_path "$save_dir" \
        2>&1 | tee "results/rps_exp_${model}/log.txt"
    echo "[GPU $gpu] Finished: $model"
}

mkdir -p "results/rps_exp_${MODEL1}" "results/rps_exp_${MODEL2}"

# Run both in background
run_exp "$MODEL1" 0 &
PID1=$!
run_exp "$MODEL2" 1 &
PID2=$!

# Wait for both to finish (don't exit on error of either)
wait $PID1
RET1=$?
wait $PID2
RET2=$?

if [ $RET1 -ne 0 ]; then
    echo "WARNING: $MODEL1 exited with code $RET1"
fi
if [ $RET2 -ne 0 ]; then
    echo "WARNING: $MODEL2 exited with code $RET2"
fi

echo "All experiments complete!"
