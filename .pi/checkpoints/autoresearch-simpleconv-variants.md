# Autoresearch: SimpleConv Architecture Variants

**Branch:** `autoresearch-attempt` (commit `961c324`)  
**Goal:** Systematically explore novel modifications to the SimpleConv RPS predictor architecture to improve tracking accuracy without exploding parameter count.  
**Status:** ✅ **Complete** — all 9 variants + baseline trained and evaluated (2026-05-29/30, vast-server, ~10.5 GPU-hours).  
**Last touched:** 2026-05-30 01:12 UTC

---

## Quick Start (for a colleague picking this up)

```bash
# You are on vast-server, in /root/harmonic-noise-suppression
cd /root/harmonic-noise-suppression
git branch --show-current   # should be autoresearch-attempt

# All checkpoints and logs:
ls results/rps_exp_*/

# Evaluate any checkpoint on the validation set:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_rps_checkpoint.py \
    --model simple_conv_bigru \
    --checkpoint results/rps_exp_simple_conv_bigru/best_simple_conv_bigru.pt
```

---

## Final Leaderboard (uniform hyperparams, same data split)

All trained with: `--epochs 200 --patience 15 --batch_size 16 --lr 0.001 --weight_decay 0.0001 --grad_clip 5.0`

| # | Model | MSE↓ | RMSE | MAE↓ | R²↑ | Impr% | Params | Time | Checkpoint |
|---|-------|------|------|------|-----|-------|--------|------|------------|
| 1 | **simple_conv_v2** | 2.61 | 1.61 | 0.76 | **0.951** | 99.3% | 1.50M | 70m | `results/rps_exp_v2/best_simple_conv_v2.pt` |
| 2 | simple_conv_bigru_v2 | 2.67 | 1.64 | 0.78 | 0.948 | 99.3% | 1.44M | 60m | `results/rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt` ✅ |
| 3 | simple_conv_bigru | 2.74 | 1.66 | 0.80 | 0.945 | 99.2% | 663K | 63m | `results/rps_exp_simple_conv_bigru/best_simple_conv_bigru.pt` |
| 4 | simple_conv_tcn | 3.09 | 1.76 | 0.83 | 0.936 | 99.2% | 1.38M | ~55m | `results/rps_exp_tcn/best_simple_conv_tcn.pt` |
| 5 | simple_conv_magphase_bigru | 3.16 | 1.78 | 0.96 | 0.917 | 99.1% | 666K | 43m | `results/rps_exp_magphase_bigru/best_simple_conv_magphase_bigru.pt` |
| 6 | simple_conv_attn_pool | 4.87 | 2.21 | 1.25 | 0.860 | 98.7% | 563K | 58m | `results/rps_exp_attn_pool/best_simple_conv_attn_pool.pt` |
| 7 | simple_conv_wide | 5.04 | 2.24 | 1.32 | 0.847 | 98.6% | 3.94M | 91m | `results/rps_exp_wide/best_simple_conv_wide.pt` |
| 8 | simple_conv_multiscale | 5.15 | 2.27 | 1.31 | 0.840 | 98.6% | 1.36M | ~45m | `results/rps_exp_multiscale/best_simple_conv_multiscale.pt` |
| 9 | simple_conv (baseline) | 5.21 | 2.28 | 1.36 | 0.837 | 98.6% | 538K | 65m | `results/rps_exp_simple_conv/best_simple_conv.pt` |
| 10 | simple_conv_se_next | 7.30 | 2.70 | 1.86 | 0.688 | 98.0% | 1.41M | 64m | `results/rps_exp_se_next/best_simple_conv_se_next.pt` ✅ |

✅ All results verified via `scripts/eval_rps_checkpoint.py` (2026-05-30 01:15 UTC).

**Note**: `bigru_v2` and `se_next` training logs are missing (tee failed before dir creation), but checkpoints are intact and verified. These two also have **older** results from May 24–25 in `results/rps_exp_simple_conv_bigru_v2/` and `results/rps_exp_simple_conv_se_next/` (different random seed — do not mix with this sweep).

### All result directories

```
results/
├── rps_exp_attn_pool/          # My run (May 29) — log + checkpoint ✓
├── rps_exp_bigru_v2/           # My run (May 29) — checkpoint ✓, NO LOG ⚠️
├── rps_exp_magphase_bigru/     # My run (May 29) — log + checkpoint ✓
├── rps_exp_multiscale/         # My run (May 29) — log + checkpoint ✓
├── rps_exp_se_next/            # My run (May 29) — checkpoint ✓, NO LOG ⚠️
├── rps_exp_simple_conv/        # My run (May 29) — log + checkpoint ✓
├── rps_exp_simple_conv_bigru/  # My run (May 29) — log + checkpoint ✓
├── rps_exp_tcn/                # My run (May 29) — log + checkpoint ✓
├── rps_exp_v2/                 # My run (May 29) — log + checkpoint ✓
├── rps_exp_wide/               # My run (May 29) — log + checkpoint ✓
│
├── rps_exp_simple_conv_attn_pool/      # Older run (May 24) — DO NOT USE for this sweep
├── rps_exp_simple_conv_bigru_v2/       # Older run (May 24–25) — DO NOT USE
├── rps_exp_simple_conv_magphase_bigru/ # Older run (May 24–25) — DO NOT USE
└── rps_exp_simple_conv_se_next/        # Older run (May 24) — DO NOT USE
```

---

## Key Findings

### What works

1. **BiGRU temporal head is the dominant component.** Every top-5 model has it. Adding BiGRU alone jumps R² from 0.837 (baseline) to 0.945 — the single largest improvement.
2. **v2 wins marginally** — SE + frequency attention + BiGRU + deeper encoder achieves R²=0.951, but only +0.003 above bigru_v2 and +0.006 above plain bigru.
3. **Pareto-optimal: `simple_conv_bigru`** — 663K params, R²=0.945. 99.4% of v2's performance at 44% of the parameters.
4. **TCN is the best non-BiGRU architecture** — dilated convolutions (receptive field 31) give R²=0.936, competitive but ~0.015 behind BiGRU family.

### What doesn't

5. **SE-Next is actively harmful** — causes training instability (massive val MSE spikes), R²=0.688 which is 18% worse than baseline. Likely needs different optimizer/regularization.
6. **Bigger models on 6000 samples don't help** — wide (3.94M params, 7.3× baseline) barely beats baseline (0.847 vs 0.837). Overparameterized for this dataset size.
7. **Phase information adds complexity without clear gain** — magphase_bigru (3-channel input) at R²=0.917 is 0.028 behind plain bigru.
8. **Multi-scale fusion (FPN-style) is unstable** — oscillating validation loss, final R²=0.840 (only +0.003 over baseline).
9. **Attention pooling alone helps modestly** — +0.023 R² over baseline, but far behind BiGRU.

### Training dynamics observed

- **BiGRU models converge fast and smooth** — monotonic improvement, no spikes
- **TCN converges fast early but plateaus lower** — R² hit 0.91 by epoch 27 but only reached 0.94
- **SE-containing models (se_next, v2, multiscale) show instability** — occasional val MSE spikes. v2 eventually stabilized; se_next and multiscale did not
- **Wide model trains very slowly** — 91 min vs 63 min for bigru with worse results

---

## Architecture Details (for reference)

All in `models/rps_predictor.py`. Key utility blocks:

| Block | Purpose | Used by |
|-------|---------|---------|
| `BiGRUHead` | Bidirectional GRU over time frames | bigru, bigru_v2, v2, magphase_bigru |
| `TCNHead` | Dilated conv temporal head (rf=31) | tcn |
| `FrequencyAttentionPool` | Multi-head attention over frequency bins | attn_pool, v2 |
| `SqueezeExcitation2d` | Channel attention | se_next, v2, bigru_v2, wide |
| `ResidualConvBlock2d` | Conv+BN+LeakyReLU with skip + optional SE | All variants |
| `MultiScaleFusionHead` | FPN-style bottom-up feature fusion | multiscale |

Model architectures:

| Model | Encoder | Temporal head | Input channels | Other |
|-------|---------|---------------|----------------|-------|
| simple_conv | 4-block residual (64→128) | Global avg pool + MLP | 1 (log-mag) | — |
| simple_conv_bigru | 4-block residual (64→128) | BiGRU (2×128) | 1 | — |
| simple_conv_bigru_v2 | 6-block deep residual (128) | BiGRU (2×128) | 1 | SE after each block |
| simple_conv_v2 | 6-block deep residual (128) | BiGRU (2×128) | 1 | SE + FreqAttn pool |
| simple_conv_tcn | 4-block residual (64→128) | TCN (dilated, rf=31) | 1 | — |
| simple_conv_magphase_bigru | 4-block residual (64→128) | BiGRU (2×128) | 3 (mag+cos+sin) | — |
| simple_conv_attn_pool | 4-block residual (64→128) | FreqAttn pool + MLP | 1 | — |
| simple_conv_se_next | 6-block residual+SE (128→256) | Global avg pool + MLP | 1 | No temporal modeling |
| simple_conv_multiscale | 4-block encoder | MultiScaleFusion + MLP | 1 | FPN fusion |
| simple_conv_wide | Wider 4-block (128→256→512) | Global avg pool + MLP | 1 | Pure scaling |

---

## How To Run Things

### Run a new training (on vast-server, postdoc is broken)

```bash
# 1. Check GPU availability
nvidia-smi

# 2. Create tmux session and launch
mkdir -p results/rps_exp_<name>
tmux new-session -d -s exp-<name> -n <name> bash
sleep 1
tmux send-keys -t exp-<name>:<name> \
  "CUDA_VISIBLE_DEVICES=<0|1> .venv/bin/python train_rps_predictor.py \
   --model <model_name> \
   --epochs 200 --patience 15 --batch_size 16 \
   --lr 0.001 --weight_decay 0.0001 --grad_clip 5.0 \
   --save_path results/rps_exp_<dir> \
   2>&1 | tee results/rps_exp_<dir>/log.txt" Enter

# 3. ALSO enable pane logging (backup in case tee fails)
tmux pipe-pane -t exp-<name>:<name> -o "cat >> results/rps_exp_<dir>/log.txt"

# 4. Check progress
tmux capture-pane -t exp-<name>:<name> -p | tail -20
# or
tail -5 results/rps_exp_<dir>/log.txt

# 5. GPU monitoring
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv

# 6. Kill if needed
tmux kill-session -t exp-<name>
```

### Evaluate a trained checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_rps_checkpoint.py \
    --model <model_name> \
    --checkpoint results/rps_exp_<dir>/best_<model_name>.pt \
    --data_root datasets/DREGON-LM
```

Output: MSE, MAE/frame, MAE/clip, R², R² median.

### Run evaluation on new/high-SNR data

The `scripts/eval_rps_checkpoint.py` script evaluates on the **validation split** by default. To evaluate on custom data, you'd need to either:
1. Create a new dataset split and pass `--data_root`
2. Or use the dataset classes from `train_rps_predictor.py` (`DREGONRPSDataset`) directly

### Full model training (speech enhancement, not RPS prediction)

For the actual speech enhancement models that USE RPS predictions:
```bash
# See valid.py, final_valid.py, train.py for the enhancement pipeline
# RPS prediction is a sub-component; the enhancement models are in models/
```

---

## Gotchas

### ⚠️ Critical: tee fails if directory doesn't exist first
The training script prints to stdout BEFORE `os.makedirs(save_path)` runs. Since `tee` tries to open the log file immediately, it fails with "No such file or directory." **Always `mkdir -p` BEFORE launching.** Also enable `tmux pipe-pane` as backup.

### Python path
The tmux shell doesn't have `.venv/bin` in PATH. Use `.venv/bin/python` explicitly, NOT `python` or `python3`.

### Postdoc is broken
Don't use `postdoc submit`. The queue daemon (`postdoc-runner` in `postdoc-queue` tmux session) says "not enough free GPUs" even when GPUs are free. Run training directly via tmux.

### Duplicate results directories
There are OLD results under `results/rps_exp_simple_conv_*` from May 24–25 (before this session). These were trained with potentially different random seeds or data splits. The CURRENT sweep results are in `results/rps_exp_<name>/` (without the `simple_conv_` prefix). Don't mix results from both sets.

### Missing logs for bigru_v2 and se_next
The `tee` command failed early for these two runs. The checkpoints are saved and the final metrics were captured from tmux scrollback. To reproduce the exact numbers, re-evaluate:
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_rps_checkpoint.py \
    --model simple_conv_bigru_v2 \
    --checkpoint results/rps_exp_bigru_v2/best_simple_conv_bigru_v2.pt
```

### Git state
- On branch `autoresearch-attempt`
- Working tree has uncommitted changes to `.pi/checkpoints/autoresearch-simpleconv-variants.md`
- New untracked file: `.pi/experiment-queue.md`
- All training results are NOT committed — they're in `results/` which is presumably gitignored

---

## Dataset

- **Dataset**: DREGON-LM (drone noise + LibriMix speech)
- **Location**: `datasets/DREGON-LM/`
- **Splits**: `train/` (6000 samples), `valid/` (600 samples)
- **Format**: Each sample is a directory with `noisy.wav`, `clean.wav`, `rps.npy` (ground-truth RPS values per frame)
- **Metadata**: `datasets/DREGON-LM/metadata.json`
- **Extra eval samples**: `datasets/DREGON-LM/rps_eval_long_samples/`, `rps_eval_specific_samples/`, `rps_train_specific_samples/`

---

## Next Steps (priority order)

1. **Verify bigru_v2 and se_next** — run `scripts/eval_rps_checkpoint.py` on their checkpoints to get exact logged numbers
2. **Ablation on v2** — strip components to isolate what matters:
   - v2 without SE → does SE help or is it just the deeper encoder + BiGRU?
   - v2 without FreqAttn → does attention pooling matter?
   - bigru_v2 (already done) vs v2 → SE + attention vs just deeper encoder
3. **Smoothness loss sweep** — `--smoothness_weight 0.01 0.1 1.0` on `simple_conv_bigru`
4. **High-SNR evaluation** — test top-3 models on higher SNR ranges from `rps_eval_specific_samples/`
5. **Cross-validation** — single train/valid split used. Rankings might be noise.
6. **Paper writeup** — compare against literature RPS prediction methods (classical approaches in `classical_rps_predictors.py`)

---

## Tracking File

Runtime tracking (GPU assignments, queue, live progress): `.pi/experiment-queue.md`
