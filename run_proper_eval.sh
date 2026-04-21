#!/bin/bash
# Run proper evaluation for all models
# - Audio models: use final_valid.py for SI-SDR, PESQ, STOI
# - RPS-only models: use valid_rps_only for MSE, neg_mse, R²

set -e

REMOTE_DIR="/root/harmonic-noise-suppression"
PYTHON="$REMOTE_DIR/.venv/bin/python"
VALID_PATH="$REMOTE_DIR/datasets/DREGON-LM/valid"
RESULTS_DIR="$REMOTE_DIR/results"
EVAL_DIR="$RESULTS_DIR/evaluation"

cd "$REMOTE_DIR"

echo "=== Proper Model Evaluation ==="
echo ""

# =============================================================================
# 1. AUDIO MODELS (need audio metrics: SI-SDR, PESQ, STOI)
# =============================================================================

echo "--- Audio Models ---"

# DCUNet baseline (checkpoint in dcunet_baseline_dregon, config from configs/)
echo "Evaluating DCUNet baseline..."
if [ -f "$RESULTS_DIR/dcunet_baseline_dregon/best_model.ckpt" ]; then
    if [ ! -f "$RESULTS_DIR/dcunet_baseline_dregon/eval_done.txt" ]; then
        $PYTHON -c "
import sys
sys.path.insert(0, '$REMOTE_DIR')
from final_valid import check_validation
check_validation({
    'model_type': 'dcunet',
    'config_path': 'configs/7b_DCUNet_baseline_DREGON.yaml',
    'start_check_point': '$RESULTS_DIR/dcunet_baseline_dregon/best_model.ckpt',
    'valid_path': ['$VALID_PATH'],
    'store_dir': '$EVAL_DIR/dcunet_baseline_dregon',
    'device_ids': [0],
    'metrics': ['estoi', 'si_sdr', 'pesq'],
})
" 2>&1 | tail -30
        touch "$RESULTS_DIR/dcunet_baseline_dregon/eval_done.txt"
    else
        echo "  Already evaluated, skipping."
    fi
else
    echo "  Checkpoint not found, skipping."
fi

# DCUNet+RPS (checkpoint in dcunet_rps_dregon, config from configs/)
echo "Evaluating DCUNet+RPS..."
if [ -f "$RESULTS_DIR/dcunet_rps_dregon/best_model.ckpt" ]; then
    if [ ! -f "$RESULTS_DIR/dcunet_rps_dregon/eval_done.txt" ]; then
        $PYTHON -c "
import sys
sys.path.insert(0, '$REMOTE_DIR')
from final_valid import check_validation
check_validation({
    'model_type': 'dcunet',
    'config_path': 'configs/7a_DCUNet_RPS_DREGON.yaml',
    'start_check_point': '$RESULTS_DIR/dcunet_rps_dregon/best_model.ckpt',
    'valid_path': ['$VALID_PATH'],
    'store_dir': '$EVAL_DIR/dcunet_rps_dregon',
    'device_ids': [0],
    'metrics': ['estoi', 'si_sdr', 'pesq'],
})
" 2>&1 | tail -30
        touch "$RESULTS_DIR/dcunet_rps_dregon/eval_done.txt"
    else
        echo "  Already evaluated, skipping."
    fi
else
    echo "  Checkpoint not found, skipping."
fi

# =============================================================================
# 2. RPS-ONLY MODELS (need RPS metrics: MSE, neg_mse, R²)
# =============================================================================

echo ""
echo "--- RPS-Only Models ---"

# DCUNet RPS-Only (job 351b060fbb93)
echo "Evaluating DCUNet RPS-Only..."
if [ -f "$RESULTS_DIR/351b060fbb93/training/best_model.ckpt" ]; then
    if [ ! -f "$RESULTS_DIR/351b060fbb93/eval_done.txt" ]; then
        $PYTHON -c "
import sys
sys.path.insert(0, '$REMOTE_DIR')
from valid import valid_rps_only
from argparse import Namespace
import yaml

# Load config
with open('$RESULTS_DIR/351b060fbb93/config.yaml') as f:
    config = yaml.safe_load(f)

from ml_collections import ConfigDict
config = ConfigDict(config)

# Load model
from utils import get_model_from_config, load_start_checkpoint
model, cfg = get_model_from_config('dcunet', '$RESULTS_DIR/351b060fbb93/config.yaml')
args = Namespace(
    model_type='dcunet',
    start_check_point='$RESULTS_DIR/351b060fbb93/training/best_model.ckpt',
    lora_checkpoint='',
    valid_path=['$VALID_PATH'],
    store_dir='',
)
load_start_checkpoint(args, model, type_='valid')

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = valid_rps_only(model, args, config, device, verbose=True)
print('RPS metrics:', result)
" 2>&1 | tail -50
        touch "$RESULTS_DIR/351b060fbb93/eval_done.txt"
    else
        echo "  Already evaluated, skipping."
    fi
else
    echo "  Checkpoint not found, skipping."
fi

# DCCRN RPS-Only (job 2fb61da1190e)
echo "Evaluating DCCRN RPS-Only..."
if [ -f "$RESULTS_DIR/2fb61da1190e/training/best_model.ckpt" ]; then
    if [ ! -f "$RESULTS_DIR/2fb61da1190e/eval_done.txt" ]; then
        $PYTHON -c "
import sys
sys.path.insert(0, '$REMOTE_DIR')
from valid import valid_rps_only
from argparse import Namespace
import yaml

# Load config
with open('$RESULTS_DIR/2fb61da1190e/config.yaml') as f:
    config = yaml.safe_load(f)

from ml_collections import ConfigDict
config = ConfigDict(config)

# Load model
from utils import get_model_from_config, load_start_checkpoint
model, cfg = get_model_from_config('dccrn', '$RESULTS_DIR/2fb61da1190e/config.yaml')
args = Namespace(
    model_type='dccrn',
    start_check_point='$RESULTS_DIR/2fb61da1190e/training/best_model.ckpt',
    lora_checkpoint='',
    valid_path=['$VALID_PATH'],
    store_dir='',
)
load_start_checkpoint(args, model, type_='valid')

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = valid_rps_only(model, args, config, device, verbose=True)
print('RPS metrics:', result)
" 2>&1 | tail -50
        touch "$RESULTS_DIR/2fb61da1190e/eval_done.txt"
    else
        echo "  Already evaluated, skipping."
    fi
else
    echo "  Checkpoint not found, skipping."
fi

echo ""
echo "=== Evaluation Complete ==="
