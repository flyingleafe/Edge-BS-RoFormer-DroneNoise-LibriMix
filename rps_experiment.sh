#!/usr/bin/env bash
# RPS Experiment: Compare DCUNet with and without RPS conditioning on DREGON-LibriMix
#
# Usage:
#   ./rps_experiment.sh create-dataset    # Create the full DREGON-LibriMix dataset
#   ./rps_experiment.sh train             # Run both training jobs in parallel on GPU 0 and 1
#   ./rps_experiment.sh train-rps         # Run only RPS-conditioned training
#   ./rps_experiment.sh train-baseline    # Run only baseline training
#   ./rps_experiment.sh eval              # Evaluate both models
#
# Environment variables:
#   GPU_RPS=0       # GPU for RPS model (default: 0)
#   GPU_BASELINE=1  # GPU for baseline model (default: 1)

set -e

# Configuration
DATASET_DIR="datasets/DREGON-LM"
RESULTS_RPS="results/dcunet_rps_dregon"
RESULTS_BASELINE="results/dcunet_baseline_dregon"
CONFIG_RPS="configs/7a_DCUNet_RPS_DREGON.yaml"
CONFIG_BASELINE="configs/7b_DCUNet_baseline_DREGON.yaml"

# Dataset parameters
NUM_TRAIN=6000
NUM_VALID=600
SAMPLE_DURATION=8.224  # Match chunk_size / sample_rate = 131584 / 16000

# GPU assignments (can be overridden via environment variables)
GPU_RPS=${GPU_RPS:-0}
GPU_BASELINE=${GPU_BASELINE:-1}

create_dataset() {
    echo "=========================================="
    echo "Creating DREGON-LibriMix dataset"
    echo "=========================================="
    echo "Train samples: $NUM_TRAIN"
    echo "Valid samples: $NUM_VALID"
    echo "Sample duration: $SAMPLE_DURATION seconds"
    echo ""

    python create_dregon_librimix.py \
        --speech_dir data/librispeech/LibriSpeech/train-clean-100 \
        --dregon_dir data/DREGON \
        --output_dir "$DATASET_DIR" \
        --num_train $NUM_TRAIN \
        --num_valid $NUM_VALID \
        --duration $SAMPLE_DURATION \
        --seed 42

    echo ""
    echo "Dataset created at: $DATASET_DIR"
}

train_rps() {
    echo "=========================================="
    echo "Training DCUNet with RPS conditioning"
    echo "GPU: $GPU_RPS"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_RPS python train.py \
        --model_type dcunet \
        --config_path "$CONFIG_RPS" \
        --results_path "$RESULTS_RPS" \
        --data_path "$DATASET_DIR/train" \
        --valid_path "$DATASET_DIR/valid"
}

train_baseline() {
    echo "=========================================="
    echo "Training DCUNet baseline (no RPS)"
    echo "GPU: $GPU_BASELINE"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_BASELINE python train.py \
        --model_type dcunet \
        --config_path "$CONFIG_BASELINE" \
        --results_path "$RESULTS_BASELINE" \
        --data_path "$DATASET_DIR/train" \
        --valid_path "$DATASET_DIR/valid"
}

train_parallel() {
    echo "=========================================="
    echo "Training both models in parallel"
    echo "RPS model on GPU $GPU_RPS"
    echo "Baseline model on GPU $GPU_BASELINE"
    echo "=========================================="

    # Start both training jobs in background
    train_rps &
    PID_RPS=$!

    train_baseline &
    PID_BASELINE=$!

    echo "Started RPS training (PID: $PID_RPS)"
    echo "Started baseline training (PID: $PID_BASELINE)"
    echo ""
    echo "Waiting for both jobs to complete..."

    # Wait for both jobs
    wait $PID_RPS
    RPS_EXIT=$?

    wait $PID_BASELINE
    BASELINE_EXIT=$?

    echo ""
    echo "=========================================="
    echo "Training complete"
    echo "RPS model exit code: $RPS_EXIT"
    echo "Baseline model exit code: $BASELINE_EXIT"
    echo "=========================================="
}

eval_models() {
    echo "=========================================="
    echo "Evaluating models"
    echo "=========================================="

    # Find best checkpoints
    RPS_CKPT=$(ls -t "$RESULTS_RPS"/model_*.ckpt 2>/dev/null | head -1)
    BASELINE_CKPT=$(ls -t "$RESULTS_BASELINE"/model_*.ckpt 2>/dev/null | head -1)

    if [ -z "$RPS_CKPT" ]; then
        echo "Warning: No RPS model checkpoint found in $RESULTS_RPS"
    else
        echo "Evaluating RPS model: $RPS_CKPT"
        CUDA_VISIBLE_DEVICES=$GPU_RPS python inference.py \
            --model_type dcunet \
            --config_path "$CONFIG_RPS" \
            --start_check_point "$RPS_CKPT" \
            --input_folder "$DATASET_DIR/valid" \
            --store_dir "$RESULTS_RPS/eval_output"
    fi

    if [ -z "$BASELINE_CKPT" ]; then
        echo "Warning: No baseline model checkpoint found in $RESULTS_BASELINE"
    else
        echo "Evaluating baseline model: $BASELINE_CKPT"
        CUDA_VISIBLE_DEVICES=$GPU_BASELINE python inference.py \
            --model_type dcunet \
            --config_path "$CONFIG_BASELINE" \
            --start_check_point "$BASELINE_CKPT" \
            --input_folder "$DATASET_DIR/valid" \
            --store_dir "$RESULTS_BASELINE/eval_output"
    fi
}

print_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  create-dataset    Create the full DREGON-LibriMix dataset"
    echo "  train             Run both training jobs in parallel"
    echo "  train-rps         Run only RPS-conditioned training"
    echo "  train-baseline    Run only baseline training"
    echo "  eval              Evaluate both models"
    echo ""
    echo "Environment variables:"
    echo "  GPU_RPS=0         GPU for RPS model (default: 0)"
    echo "  GPU_BASELINE=1    GPU for baseline model (default: 1)"
}

# Main
case "${1:-}" in
    create-dataset)
        create_dataset
        ;;
    train)
        train_parallel
        ;;
    train-rps)
        train_rps
        ;;
    train-baseline)
        train_baseline
        ;;
    eval)
        eval_models
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
