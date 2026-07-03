# Checkpoint: Online Generated-Noise Augmentation — Experiment Log & Colleague Handoff

**Date**: 2026-07-03 · **Branch**: `main` · **Status**: complete (11 Slurm jobs, 1 useful result)

## What we tried

Train RPS predictors on the online-mixing stream augmented with a **generated
noise source**: a trained `PositionalHarmonicNoiseGen` rendered live on the GPU
by a spawn-producer process and fed into the mixer alongside real DREGON +
Michael's recordings. The generated chunks carry synthetic (intermittent) RPS
trajectories that double as exact labels — unlimited-variety, perfectly-labelled
augmentation.

Full plumbing: `data_processing/generated_noise.py` (`GeneratedNoisePool`) +
shared-memory ring buffer + `OnlineMixIterableDataset.from_config`.

## Key result: generated-noise augmentation DEGRADES RPS prediction

| Model | Online-mix baseline | With generated noise | Δ PIT MSE |
|-------|-------------------|---------------------|-----------|
| `scv2_uni_gru128` | PIT MSE 7.33, R² 0.822 | PIT MSE **9.29**, R² 0.791 | **+27%** |
| `scv2_transformer` | PIT MSE 8.46, R² 0.808 | PIT MSE **10.63**, R² 0.762 | **+26%** |

All other attempts (simple_conv_v2 with augmentations, simple_conv_v2 without,
transformer dual-generated) failed with NaN divergence or OOM. Only the two
runs above produced usable checkpoints — and both are *worse* than their
no-generator baselines.

## Double-overfitting hypothesis

The degradation is not uniform across architectures:

- **Unidirectional GRU**: +27% PIT MSE degradation. Causal constraint acts as a
  regularizer — it *can't* look at the full sequence, so it's forced to read
  harmonics frame-by-frame. Still memorizes some generator artifacts, but less
  severely.

- **Transformer (global self-attention)**: +26% degradation *and* classic
  overfitting curve. Train loss drops monotonically (9.9→3.7) while validation
  collapses (PIT 10.6→43.6 after epoch 9). The transformer can simultaneously
  memorize (a) the limited set of real RPS trajectories via global attention
  across any time offset, and (b) spectral fingerprints specific to the
  generator's output. Once it exhausts real harmonic structure (~epoch 9), it
  pivots to exploiting generator artifacts — improving train loss while
  validation collapses.

**Train/val divergence (transformer, run 13971617):**

```
Epoch   Train MSE   Val PIT    R²
   1      264.7      34.1     0.24
   9        9.9      10.6     0.76  ← BEST (PIT 10.63)
  10        8.8      26.3     0.34  ← overfitting begins
  16        4.9      23.4     0.44
  24        3.7      43.6    -0.13  ← early stop
```

## All jobs attempted

| Job ID | Node | Model | Config | Result |
|--------|------|-------|--------|--------|
| 13719874 | rdg1 (H100) | uni_gru128 | gen_michael's + aug | PIT 9.29 ✅ |
| 13720893 | sbg2 (V100) | simple_conv_v2 | gen_michael's + aug | OOM |
| 13721162 | sbg2 (V100) | simple_conv_v2 (bs16) | gen_michael's + aug | OOM |
| 13721312 | sbg2 (V100) | simple_conv_v2 (bs12) | gen_michael's + aug | Dataloader crash |
| 13721809 | rdg12 (sae) | simple_conv_v2 (bs32) | gen_michael's + aug | NaN divergence |
| 13953000 | ddg2 (V100) | transformer (bs12) | gen_michael's + aug | PIT ~17 (cancelled) |
| 13961446 | ddg2 (V100) | transformer (bs12) | gen_michael's, NO aug | PIT ~18 (cancelled) |
| 13970210 | rdg1 (H100) | transformer (bs12) | dual-gen (mich+dreg) | GPU device error |
| 13970325 | rdg1 (H100) | transformer (bs12) | dual-gen (mich+dreg) | GPU device error |
| 13971617 | sbg3 (V100) | transformer (bs8) | gen_michael's, NO aug | PIT 10.63 ✅ |

**Node issues encountered**: rdg1 GPU 0 broken on 2026-07-03; ddg2 had
multiprocessing bugs with the generated-noise producer; OOM on sbg2 (other
jobs eating VRAM); sbg3 worked cleanly.

## Configuration variants

All configs are at `configs/online_mix_generated_augment_gpfs.yaml` (evolved
over the session). Final version:

- **Noise sources**: real DREGON (weight 1.0) + real Michael's FLY125 (weight 1.0) + generated Michael's (weight 0.5)
- **Speech**: LibriSpeech train-clean-100-readable
- **SNR**: uniform [-30, 0] dB
- **Augmentations**: removed (found to hurt further; the generated data + real data alone was enough diversity)
- **Generated noise params**: `drone: michaels`, checkpoint at `/gpfs/scratch/acw592/results/noise_gen_sweep/baseline/best_positional_harmonic_gen.pt`, `n_harmonics: 100`, `random_phase: true`, RPS `synthetic_intermittent, aggressiveness: 1.0`

Note: we attempted adding a second generated source for DREGON (`drone: dregon`)
but hit VRAM issues (two spawn-producer CUDA contexts + training model > 16GB).
Same model + checkpoint, just different embeddings — should be one producer in
the future (see "Next steps").

## Checkpoints

Only two usable checkpoints exist:

- `/gpfs/scratch/acw592/results/rps_gen_aug_uni_gru128/best_simple_conv_v2_uni_gru128.pt` — best PIT 9.29
- `/gpfs/scratch/acw592/results/rps_gen_aug_scv2_xfrm/best_simple_conv_v2_transformer.pt` — best PIT 10.63

## Why it doesn't work (analysis)

1. **Generator quality is the bottleneck.** The baseline noise-gen checkpoint
   (no smoothness) produces output with imperfect mid-frequency harmonics and
   a specific filter-envelope distribution. The RPS model latches onto these
   as shortcut features rather than learning universal harmonic structure.

2. **Limited RPS trajectory diversity.** Even with synthetic-intermittent
   trajectories, the statistical model (OU process + Poisson maneuvers) produces
   a narrow distribution. The transformer can memorize the "script" of which
   trajectory follows which, using global attention to skip the hard work of
   reading harmonics.

3. **No augmentation of RPS trajectories themselves.** The online mixer
   diversifies acoustic dressing (speech, SNR) but not the underlying RPS
   curves. Time-warping / speed perturbation of RPS sequences would likely help.

4. **Generator-trained models overfit to generator artifacts.** The same
   phenomenon that makes noise-gen smoothness useful (removing high-frequency
   amplitude flicker) may also make the output *less* like real recordings in
   ways the RPS model can exploit. A smoothness-trained noise-gen checkpoint
   (e.g. `harm=1e-1`) might produce output with fewer exploitable artifacts.

## Survival guide for a colleague picking this up

### Reproduce the baseline (what to beat)

```bash
# No-generator online-mix baseline (from June 19 sweep)
./sbatch.sh -J rps_baseline --partition=gpushort --time=1:00:00 -- \
  python train_rps_predictor.py \
    --model simple_conv_v2_transformer --device cuda:0 \
    --epochs 80 --patience 15 --batch_size 12 \
    --lr 1e-3 --weight_decay 1e-4 --grad_clip 0.5 \
    --loss pit_mse --epoch_progress \
    --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels \
    --online_mix \
    --mix_config configs/online_mix_v4_michaels_train_no_room1_gpfs.yaml \
    --samples_per_validation 5000 \
    --save_path /gpfs/scratch/acw592/results/rps_baseline_scv2_xfrm
```

### Reproduce the generated-noise run

```bash
./sbatch.sh -J rps_genaug --partition=gpushort --time=1:00:00 -- \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_rps_predictor.py \
    --model simple_conv_v2_transformer --device cuda:0 \
    --epochs 80 --patience 15 --batch_size 8 \
    --lr 1e-3 --weight_decay 1e-4 --grad_clip 0.5 \
    --loss pit_mse --epoch_progress \
    --data_root /gpfs/scratch/acw592/datasets/DREGON-LM-V4-michaels \
    --online_mix \
    --mix_config configs/online_mix_generated_augment_gpfs.yaml \
    --samples_per_validation 5000 \
    --save_path /gpfs/scratch/acw592/results/rps_gen_aug_scv2_xfrm
```

### Things to try (priority order)

1. **Smoothness-trained noise-gen checkpoint.** Replace the baseline checkpoint
   with `harm=1e-1` smoothness (`noise_gen_sweep/harm_1e-1/`). Smoother harmonic
   amplitudes = fewer artifacts for the RPS model to exploit.

2. **RPS trajectory augmentation.** Time-warp / speed-perturb the RPS curves
   (both real and synthetic) to break memorization. This is independent of
   acoustic augmentation and directly addresses the root cause.

3. **Reduce generated weight.** Try `weight: 0.25` or even `0.1` — a small
   sprinkle of synthetic variety without overwhelming the model.

4. **Single producer, multiple drones.** Modify `GeneratedNoisePool` to accept
   a list of drone types, using one CUDA context and one ring buffer. Then add
   both `michaels` and `dregon` generated sources without doubling VRAM.

5. **Bigger patience / LR schedule.** The unidirectional GRU showed NaN at epoch
   13 but recovered; with patience 50 and cosine LR it might fully converge.

6. **Compare generated vs real spectrograms** at the same RPS to identify
   specific generator artifacts the model could be exploiting.

## Code pointers

- Generated noise pool + producer: `data_processing/generated_noise.py`
- Online mixing config dispatch: `data_processing/online_mixing.py` (`build_noise_pool`)
- Config: `configs/online_mix_generated_augment_gpfs.yaml`
- Training script: `train_rps_predictor.py`
- Noise-gen checkpoint used: `/gpfs/scratch/acw592/results/noise_gen_sweep/baseline/best_positional_harmonic_gen.pt`
- Output checkpoints: `/gpfs/scratch/acw592/results/rps_gen_aug_*/`
- Slurm logs: `/gpfs/scratch/acw592/logs/rps_genaug.o*`
- WandB project: `flyingleafe/rps-prediction`
- Previous architecture sweep report: `writing/reports/2026-06-19_rps-arch-sweep-v4-michaels/report.typ`
- Previous noise-gen smoothness sweep: `.pi/checkpoints/noise-gen-smoothness-sweep-results.md`

## Gotchas

- **rdg1 GPU 0 is broken** (2026-07-03): use `--exclude=rdg1` or sae partition.
- **ddg2 has multiprocessing bugs** with the spawn producer: avoid if using
  generated noise sources.
- **sbg3 is reliable** for these experiments (V100 32GB, clean).
- **batch_size must be ≤8** on V100 with generated noise (producer shares GPU).
- **`--num_workers` is hardcoded to 4** in `train_rps_predictor.py` — do not
  pass it as a CLI argument.
- **DREGON geometry path is `/gpfs/scratch/acw592/data/DREGON`**, not
  `/gpfs/scratch/acw592/data` (relevant if adding generated DREGON source).
- **Checkpoints only save `best_*.pt`** — no last-epoch checkpoint, no
  optimizer state. If a run diverges after its best epoch, you can't resume.
