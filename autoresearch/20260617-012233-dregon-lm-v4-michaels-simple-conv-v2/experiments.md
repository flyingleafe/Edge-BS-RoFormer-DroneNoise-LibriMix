# Autoresearch Experiments — 20260617-012233-dregon-lm-v4-michaels-simple-conv-v2

## Fixed context

- Dataset: `/gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels`
- Results root: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2`
- Baseline: `simple_conv_v2`
- Target metrics: PIT MSE (lower is better), R^2 (higher is better, 1.0 is max)
- Training budget: 50 epochs, patience 10, gpushort <= 1:00:00
- Extra training args: `--batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress`

## Experiment log

Record exact commands, job IDs, log paths, failures, fixes, and conclusions.

### E0 — Baseline simple_conv_v2

Status: completed

Submitted: 2026-06-17 01:23 BST

Job:

- Slurm job id: `12513837`
- Job name: `ar_012233_simplecv2`
- Log: `/gpfs/scratch/acw592/logs/ar_012233_simplecv2.o12513837`
- Save path: `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2`

Command:

```bash
python train_rps_predictor.py --model simple_conv_v2 --device cuda:0 --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels --save_path /gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2 --epochs 50 --patience 10 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 --loss pit_mse --epoch-progress
```

Notes:

- Initial Slurm status was `PENDING`; per autoresearch rule, no candidate jobs were submitted in this cycle after the baseline entered the queue.
- Added a compatibility parser alias in `train_rps_predictor.py` so the fixed autoresearch flag `--loss pit_mse` maps to the existing PIT-MSE training path (`args.pit_loss=True`).
- After baseline submission, proposed H1–H4 in `ideas.md`; no candidate implementation/submission yet because the first Slurm job initially queued as `PENDING`, triggering the workflow stop condition for new submissions.

Results:

- Slurm status: `COMPLETED` (elapsed `00:23:04`)
- Early stopping: epoch 31
- Best/final checkpoint evaluation: PIT MSE `7.8920`, RMSE `2.81`, Std MSE `42.1642`, frame MAE `2.08`, clip MAE `1.62`, R² `0.8183`
- Best epoch by validation PIT: epoch 21 (`Val PIT=7.8920`, `R²=0.8183`)
