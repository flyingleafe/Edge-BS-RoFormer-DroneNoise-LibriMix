#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run_uni_gru_clip_array.sh offline|online}
GRAD_CLIP=${GRAD_CLIP:-0.5}
DATA=${DATA:-/gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels}
MIX_CONFIG=${MIX_CONFIG:-configs/online_mix_v4_michaels_train_no_room1_gpfs.yaml}
SAMPLES_PER_VALIDATION=${SAMPLES_PER_VALIDATION:-5000}

MODELS=(
  simple_conv_v2_uni_gru
  simple_conv_v2_uni_gru128
  simple_conv_v2_uni_gru128_norm
  simple_conv_v2_uni_gru128_norm_do03
  simple_conv_v2_uni_gru96_norm_do03
  simple_conv_v2_uni_gru96_norm_do02
  simple_conv_v2_uni_gru64_norm_do03
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#MODELS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID for ${#MODELS[@]} models" >&2
  exit 2
fi
MODEL=${MODELS[$TASK_ID]}
CLIP_SAFE=${GRAD_CLIP//./p}
BASE_ROOT=/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/uni_gru_clip${CLIP_SAFE}_200ep_p50

case "$MODE" in
  offline)
    ROOT=$BASE_ROOT/offline_fixed_train
    EXTRA_ARGS=()
    ;;
  online)
    ROOT=$BASE_ROOT/online_mix_aug50k
    EXTRA_ARGS=(--online_mix --mix_config "$MIX_CONFIG" --samples_per_validation "$SAMPLES_PER_VALIDATION")
    ;;
  *)
    echo "Unknown MODE=$MODE (expected offline or online)" >&2
    exit 2
    ;;
esac

SAVE=$ROOT/$MODEL
mkdir -p "$SAVE"

cat <<EOF
Unidirectional-GRU clipped rerun
mode=$MODE
array_task=$TASK_ID
model=$MODEL
grad_clip=$GRAD_CLIP
data=$DATA
save=$SAVE
mix_config=$MIX_CONFIG
samples_per_validation=$SAMPLES_PER_VALIDATION
epochs=200
patience=50
EOF

python train_rps_predictor.py \
  --model "$MODEL" \
  --device cuda:0 \
  --data_root "$DATA" \
  --save_path "$SAVE" \
  --epochs 200 \
  --patience 50 \
  --batch_size 32 --lr 1e-3 --weight_decay 1e-4 \
  --loss pit_mse \
  --grad_clip "$GRAD_CLIP" \
  --epoch-progress \
  "${EXTRA_ARGS[@]}"
