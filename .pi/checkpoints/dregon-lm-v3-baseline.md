# Checkpoint: DREGON-LM-V3 Dataset + SimpleConv Baseline

**Date**: 2026-06-02

## What was done

### 1. DREGON-LM-V3 dataset created
- **Script**: `create_dregon_librimix_v3.py`
- **Structure**: 6000 train + 600 valid, 1-second samples at 16 kHz
- **Contents per sample**: `mixture.wav`, `vocals.wav`, `noise.wav`, `rps.npy` (4 motors × 32 STFT frames, float32)
- **SNR range**: [−30, 0] dB
- **Key fixes** applied vs the crashed first attempt:
  - Fixed `get_random_chunk` axis bug: command was `(N_samples, N_motors)` but code sliced as `(N_motors, N_samples)` — caused 18 MB rps.npy instead of 1 KB
  - Pre-cleaned motor commands via `clean_command_spikes` once per recording (not per sample)
  - On-demand LibriSpeech loading from 28.5K files (preserves diversity)
  - `resample_rps` outputs float32 (was float64)
- **Size**: 672 MB total
- **Location**: `datasets/DREGON-LM-V3/`

### 2. SimpleConv trained on DREGON-LM-V3
- **Script**: `train_rps_predictor.py` (standalone RPS training, NOT main train.py)
- **Command**:
  ```
  CUDA_VISIBLE_DEVICES=0 python train_rps_predictor.py \
      --model simple_conv \
      --data_root datasets/DREGON-LM-V3 \
      --no_pit_loss \
      --epochs 200 --patience 30 --batch_size 128
  ```
- **Loss**: Standard MSE (no PIT, no smoothness), LR=1e-3 (AdamW)
- **Result**: Converged in 40 epochs (early stopped at patience 30)
  - Val MSE: 227.0 (RMSE: 15.1 RPS), MAE/clip: 8.14 RPS (∼9% of RPS range 1–90)
  - Improvement over naive baseline: 81.9%
- **Checkpoint**: `results/rps_predictor_comparison/best_simple_conv.pt`
- **WandB**: https://wandb.ai/flyingleafe/rps-prediction/runs/ivbyimpe

### 3. Cross-evaluation: OLD vs NEW SimpleConv
- **Script**: `scripts/cross_eval_simpleconv.py`
- **OLD model**: `results/rps_exp_simple_conv/best_simple_conv.pt` (trained on original DREGON-LM, 8s samples)
- **NEW model**: `results/rps_predictor_comparison/best_simple_conv.pt` (trained on DREGON-LM-V3, 1s samples)
- **Results**:

| | DREGON-LM (orig) | DREGON-LM-V3 (new) |
|---|---|---|
| OLD | MSE 5.2 / MAE 0.67 | MSE 477.8 / MAE 12.1 |
| NEW | MSE 84.5 / MAE 3.65 | MSE 229.0 / MAE 8.14 |

- **Per-channel V3 (MSE)**: ch4=97 (easiest), ch6=93, ch3=394 (hardest). ~4× spread preserved across both models → intrinsic to microphone geometry.
- **Results JSONs**: `results/cross_eval_old.json`, `results/cross_eval_new.json`, `results/cross_eval_summary.json`

### 4. Config created
- `configs/14a_SimpleConv_DREGON_V3.yaml` — for `train.py --model_type rps_predictor` (alternative path, NOT used for actual training; `train_rps_predictor.py` is the canonical script)

## Key files changed/created

| File | Status | Purpose |
|------|--------|---------|
| `create_dregon_librimix_v3.py` | NEW (committed) | V3 dataset generator |
| `configs/14a_SimpleConv_DREGON_V3.yaml` | NEW (committed) | Config for main train.py path |
| `scripts/cross_eval_simpleconv.py` | NEW (not committed) | Per-channel cross-evaluation |
| `datasets/DREGON-LM-V3/` | gitignored | 6000+600 samples, 672 MB |
| `results/cross_eval_*.json` | gitignored | Evaluation results |
| `results/cross_eval_summary.json` | gitignored | Comparison summary |

## Gotchas / notes

- **R² is broken on V3**: 1-second clips have near-constant RPS → `SS_total ≈ 0` → R² → −∞. Trust MSE/MAE only.
- **V3 is harder than original LM**: Even the in-distribution NEW model gets 229 MSE vs OLD's 5 on LM. Likely because 1s clips have less temporal structure.
- **`train_rps_predictor.py` is the canonical RPS training script** — NOT `train.py --model_type rps_predictor`. The standalone script has better defaults and the full eval/monitoring suite.
- **Checkpoint format**: Raw `state_dict` (not wrapped in a dict with metadata keys). Load with `model.load_state_dict(torch.load(path))`.
- **Two GPUs on this machine**: RTX 4070 Ti × 2 (CUDA 0 and 1). Both idle as of this writing.
- **This machine IS the GPU server** — do NOT use `postdoc submit`. Run training directly with `CUDA_VISIBLE_DEVICES=N`.

## What's next (suggested)

- Train more RPS predictor variants (SimpleConvV2, BiGRU, etc.) on V3 for comparison
- Compare with DCUNet/DCCRN encoder + RPS head on V3
- Run RPS predictor on multi-channel V3 samples independently to measure microphone-conditional accuracy
- Integrate RPS prediction into the speech enhancement pipeline (auxiliary RPS loss)
- Generate model comparison plots using `generate-model-comparisons` skill
