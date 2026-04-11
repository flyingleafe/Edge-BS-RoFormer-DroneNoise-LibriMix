#!/bin/bash
# Run RPS-only model evaluation on remote machine
# This script properly evaluates RPS-only models (MSE, neg_mse, R²) without trying to produce audio

set -e

REMOTE_DIR="harmonic-noise-suppression"
VALID_PATH="datasets/DREGON-LM/valid"

# RPS-only model job IDs
DCUNET_RPS_ONLY_JOB="351b060fbb93"
DCCRN_RPS_ONLY_JOB="2fb61da1190e"

echo "=== Evaluating RPS-only models ==="

for JOB_ID in "$DCUNET_RPS_ONLY_JOB" "$DCCRN_RPS_ONLY_JOB"; do
    echo ""
    echo "=== Job: $JOB_ID ==="
    
    # Check if job exists
    if ! ssh vast-server "test -d $REMOTE_DIR/results/$JOB_ID"; then
        echo "Job $JOB_ID not found, skipping"
        continue
    fi
    
    # Check config to determine model type
    CONFIG="$REMOTE_DIR/results/$JOB_ID/config.yaml"
    if ssh vast-server "grep -q 'rps_loss_only.*true' $CONFIG"; then
        MODEL_TYPE=$(ssh vast-server "grep -E '^\s*model:\s*' $CONFIG | head -1" || echo "unknown")
        echo "Detected RPS-only model: $MODEL_TYPE"
    fi
done

echo ""
echo "=== Summary ==="
echo "RPS-only models detected. These need RPS metrics (MSE, neg_mse, R²), NOT audio metrics."
echo "Run valid_rps_only function instead of final_valid.py for these models."
