#!/usr/bin/env bash
# Launch RPS prediction training on DREGON-LM-V2
# Two models in parallel: SimpleConv (baseline) on GPU 0, BiGRU v2 on GPU 1
set -euo pipefail

cd /root/harmonic-noise-suppression

# Wait for dataset to be ready
echo "Waiting for DREGON-LM-V2 dataset..."
while [ ! -f datasets/DREGON-LM-V2/metadata.json ]; do
    sleep 10
done
# Give it a moment to finish writing
sleep 5
echo "Dataset ready!"

RESULTS_DIR="results/rps_predictor_v2"
mkdir -p "$RESULTS_DIR"

# Check for wandb key
WANDB_KEY="${WANDB_API_KEY:-}"
WANDB_ARG=""
if [ -n "$WANDB_KEY" ]; then
    WANDB_ARG="--wandb_key $WANDB_KEY"
fi

# ---- Launch SimpleConv baseline on GPU 0 ----
echo "=== Launching SimpleConv (baseline) on GPU 0 ==="
tmux new-session -d -s rps_simpleconv \
    "uv run python train_rps_predictor.py \
        --model simple_conv \
        --device cuda:0 \
        --data_root datasets/DREGON-LM-V2 \
        --save_path ${RESULTS_DIR}/simple_conv \
        --epochs 200 \
        --patience 15 \
        --batch_size 32 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        $WANDB_ARG \
        2>&1 | tee ${RESULTS_DIR}/simple_conv.log"

echo "SimpleConv launched in tmux window 'rps_simpleconv'"

# ---- Launch BiGRU v2 on GPU 1 ----
echo "=== Launching SimpleConvBiGRUV2 on GPU 1 ==="
tmux new-session -d -s rps_bigruv2 \
    "uv run python train_rps_predictor.py \
        --model simple_conv_bigru_v2 \
        --device cuda:1 \
        --data_root datasets/DREGON-LM-V2 \
        --save_path ${RESULTS_DIR}/simple_conv_bigru_v2 \
        --epochs 200 \
        --patience 15 \
        --batch_size 32 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        $WANDB_ARG \
        2>&1 | tee ${RESULTS_DIR}/simple_conv_bigru_v2.log"

echo "BiGRUv2 launched in tmux window 'rps_bigruv2'"
echo ""
echo "Monitor with:"
echo "  tmux attach -t rps_simpleconv"
echo "  tmux attach -t rps_bigruv2"
echo "  tmux ls"
