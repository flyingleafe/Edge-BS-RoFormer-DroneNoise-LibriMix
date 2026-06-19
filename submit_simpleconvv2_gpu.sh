#!/bin/bash
#SBATCH -J scv2_test
#SBATCH -o /gpfs/scratch/acw592/logs/%x.o%j
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 10:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRATCH=/gpfs/scratch/acw592
RESULTS_DIR="$SCRATCH/results/apocrita_simpleconvv2"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$SCRATCH/logs" "$RESULTS_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

# If Apocrita requires module setup for your venv, keep this.
# If your venv works without module load, this is still usually harmless.

source .venv/bin/activate

echo "Python: $(which python)"
python -V

echo "GPU check:"
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cudnn version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# SimpleConvV2 training test.
# Keep device as cuda:0. Slurm maps your allocated GPU to visible device 0.
python train_rps_predictor.py \
  --model simple_conv_v2 \
  --device cuda:0 \
  --data_root "$SCRATCH/datasets/DREGON-LM-V4" \
  --save_path "$RESULTS_DIR" \
  --epochs 20 \
  --patience 5 \
  --batch_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --epoch-progress

date
