# Autoresearch: SimpleConv Architecture Variants

**Branch:** `autoresearch-attempt`  
**Goal:** Systematically explore novel modifications to the SimpleConv RPS predictor architecture to improve tracking accuracy without exploding parameter count.  
**Status:** Code complete, experiments ready to run, no training results yet for variants.  
**Last touched:** 2026-05-29 (session restored from git history after merge)  

## What was built (commits unique to this branch)

### 1. Core architecture variants — `models/rps_predictor.py`

Created 9 variants of SimpleConv, all tested and instantiable:

| Model | Key changes | Params | Relative |
|-------|-------------|--------|----------|
| `simple_conv` | Baseline (moved from `train_rps_predictor.py`) | 538K | 1.0× |
| `simple_conv_v2` | Deeper residual encoder (6 blocks) + SE + freq attention pool + BiGRU head | 1.50M | 2.8× |
| `simple_conv_wide` | Wider/deeper, no attention/GRU, simple scaling | 3.94M | 7.3× |
| `simple_conv_tcn` | TCN head with dilated convolutions (receptive field 31 frames) | 1.38M | 2.6× |
| `simple_conv_multiscale` | FPN-style multi-scale encoder feature fusion | 1.36M | 2.5× |
| `simple_conv_bigru` | Baseline encoder + BiGRU temporal head | 663K | 1.2× |
| `simple_conv_bigru_v2` | Deeper 6-block encoder (128 ch) + BiGRU | 1.44M | 2.7× |
| `simple_conv_magphase_bigru` | 3-channel input: log-mag + cos(phase) + sin(phase) + BiGRU | 666K | 1.2× |
| `simple_conv_attn_pool` | Baseline encoder + learned frequency attention pooling | 563K | 1.0× |
| `simple_conv_se_next` | Residual + SE blocks, deeper encoder, larger head | 1.41M | 2.6× |

Utility blocks introduced:
- `SqueezeExcitation2d` — channel attention
- `ResidualConvBlock2d` — conv + BN + LeakyReLU with optional skip + SE
- `FrequencyAttentionPool` — multi-head attention over frequency bins
- `TCNHead` — dilated conv temporal head
- `BiGRUHead` — bidirectional GRU temporal head
- `MultiScaleFusionHead` — bottom-up feature fusion (FPN-style)

### 2. Training improvements — `train_rps_predictor.py`

- **Temporal smoothness loss** (`--smoothness_weight`): second-order finite-difference regularization on RPS predictions to encourage smooth temporal trajectories.
- **Model registry** expanded to all 9 variants + 3 encoder baselines (DCUNet-enc, DCCRN-enc, DCCRN-lite).
- **Cleanup done in this session:** removed duplicate inline `SimpleConv` class from `train_rps_predictor.py` that was shadowing the imported `models.rps_predictor.SimpleConv`.

### 3. Experiment scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_rps_batch.sh` | Sequential training of a list of models (defaults: 4 simplest variants) |
| `scripts/run_rps_parallel.sh` | Parallel training of 2 models on 2 GPUs |
| `scripts/eval_rps_checkpoint.py` | Standalone validation-set evaluation of a saved checkpoint |
| `scripts/test_hello.sh` | Minimal postdoc connectivity test |

## What was NOT done yet

- **No variant has been trained.** The only existing RPS predictor results are for the baseline `simple_conv` (from before this branch).
- **No ablation study.** We don't know which component (SE, attention pool, BiGRU, TCN, etc.) contributes most.
- **Smoothness loss not tuned.** The `--smoothness_weight` flag exists but was never swept.
- **No high-SNR or OOD evaluation** for any variant.

## Known issues / cleanup notes

- `train_rps_predictor.py` still embeds `RPSPredictionHead`, `DCUNetEncRPS`, and `DCCRNEncRPS` inline. These could be moved to `models/` for consistency, but they work.
- `models/rps_predictor.py` has `MultiScaleFusionHead` which overlaps conceptually with `RPSPredictionHead`. Consider unifying later.
- `stft_time_frames` is defined in both files (harmless but duplicative).

## Next steps (in order of priority)

1. **Run a pilot comparison** of the 4 smallest variants against baseline:
   ```bash
   postdoc submit ./scripts/run_rps_parallel.sh simple_conv simple_conv_bigru
   postdoc submit ./scripts/run_rps_parallel.sh simple_conv_attn_pool simple_conv_se_next
   ```
   Or batch:
   ```bash
   postdoc submit ./scripts/run_rps_batch.sh simple_conv simple_conv_bigru simple_conv_attn_pool simple_conv_se_next
   ```

2. **If pilot shows promise**, scale to the larger variants (v2, wide, TCN, multiscale, BiGRU v2, magphase).

3. **Ablation**: if one variant wins, ablate its components to understand why.

4. **Smoothness sweep**: try `--smoothness_weight 0.01 0.1 1.0` on the best model.

## Resume

```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
git branch --show-current   # should be autoresearch-attempt

# Verify all models instantiate
python -c "from train_rps_predictor import get_model; [get_model(m) for m in ['simple_conv','simple_conv_v2','simple_conv_bigru','simple_conv_tcn','simple_conv_se_next']]"

# Launch first pilot
postdoc submit ./scripts/run_rps_batch.sh simple_conv simple_conv_bigru simple_conv_attn_pool simple_conv_se_next
```
