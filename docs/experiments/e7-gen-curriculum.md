# E7 — Generated-Noise Curriculum for RPS Prediction (sim-to-real)

**Status:** built, not yet run — **Date:** 2026-07-11

## Motivation

RPS predictors overfit the handful of *real* rotor-speed trajectories in the
training pool (DREGON + Michael's — a few dozen distinct profiles). E4 tried to
add variety by *augmenting* real training with generated noise and it **hurt**
(+27% PIT MSE — [[e4-pit-curves-wandb]], [noise-generation-augmentation.md](noise-generation-augmentation.md)):
the E3 generator's clean, telemetry-exact combs were off-distribution vs real
recordings. E5 (time-warp of real pairs) was the alternative that worked.

E7 revisits generated noise with two things E4 lacked:

1. **A much better generator** — the E6 per-drone adaptive-σ
   `PositionalHarmonicNoiseGen` (`e6_noisegen_jitter_latreg_perdrone`), with
   learned OU linewidth (σ ≈ 0.61–0.63), spectral-norm-regularised FiLM, and
   vicinal z-noise training so the embedding space interpolates cleanly
   ([[e6-perdrone-generators-done]]).
2. **Vicinal sampling instead of the two fixed real drones** — every training
   chunk draws a *novel* drone by sampling along the DREGON↔Michael's embedding
   segment, so the predictor sees a continuum of timbres and geometries rather
   than two points.

The **question this batch answers** is deliberately narrow and posed *without
augmentation first*: **can a predictor trained on generated data ALONE reach a
reasonable PIT MSE on the real validation set?** If yes, generated data is a
viable pretraining substrate; then Stage 2 tests whether it is a better *init*
than from-scratch.

## The generated source (vicinal interp)

`data_processing/generated_noise.py::GeneratedNoisePool` gained an `interp` mode
(policy `conf/online_mix/rps_generated_only_interp.yaml`). Per producer batch:

- **embedding:** `z(α) = (1−α)·z_dregon + α·z_michaels + N(0, 0.15·‖z1−z0‖)`,
  `α ~ U(0,1)` (α also drives the `rps_synthesis` `drone_profile` blend);
- **rotor positions:** linearly interpolated along the same α (both rigs are
  4-rotor; index-wise, no reordering — a blended layout, accepted as "simple");
- **mic array:** a rig (DREGON *or* Michael's, both 8-mic) chosen 50/50
  **independently** of α, each mic jittered by `N(0, 2 cm)` — sampling the
  *vicinity* of the real arrays (they differ substantially: DREGON centred at
  origin, Michael's offset to ≈(0.30, 0, 0.33));
- **RPS jitter:** the learned per-drone OU linewidth σ interpolated at α, forced
  ON at eval (`rps_jitter=True`), applied to the excitation before synthesis, so
  the audio carries realistic linewidth while the clean synthetic RPS trajectory
  stays the exact label.

The generator checkpoint is read as a flat `_CodebookConditionedNoiseGen`
state_dict rebuilt via `models.registry.build_noise_gen_model` (spectral-norm /
per-drone-σ aware). Producer/geometry/jitter verified end-to-end on CPU.

## The curriculum (two chained runs)

No bespoke scheduler — the curriculum is two independent `train.py` runs (jobs
are shell commands):

- **Stage 1** (`e7_gencurric_s1_*`): train on **generated-only** noise
  (`rps_generated_only_interp`), validate on the fixed **real**
  DREGON+Michael's split, **patience 5**. Its `best.ckpt` is the hand-off.
- **Stage 2** (`e7_gencurric_s2_*`): **real-only** fine-tune
  (`online_mix_v4_michaels_no_aug`, no augmentation), **warm-started** from
  Stage 1's checkpoint via the new top-level `checkpoint:` field
  (`training.loop._warm_start`: fresh optimizer/scheduler/early-stopping,
  `strict=False`), **patience 20**.

The `checkpoint:` warm-start seam accepts a local path or an `r2://` URI
(`resolve_checkpoint_uri` downloads + caches), so Stage 2 can run on a fresh
cloud box off Stage 1's uploaded checkpoint.

## Arms

Three predictor heads × {Stage 1 gen-only, Stage 2 real fine-tune}, all PIT MSE,
same fixed real validation split (`DREGON-LM-V4-michaels/valid`):

| Arch | Stage 1 (gen-only) | Stage 2 (real fine-tune) | Real-data reference |
|---|---|---|---|
| `simple_conv_v2_uni_gru128` | `e7_gencurric_s1_unigru128` | `e7_gencurric_s2_unigru128` | C10 online **7.33** |
| `simple_conv_v2` | `e7_gencurric_s1_scv2` | `e7_gencurric_s2_scv2` | E5 baseline 9.71 |
| `simple_conv_v2_transformer` | `e7_gencurric_s1_transformer` | `e7_gencurric_s2_transformer` | E5 warp best 8.85 |

## Running

Local (GPU box; generator producer needs `cuda:0`):

```bash
python train.py experiment=e7_gencurric_s1_unigru128   # stage 1
python train.py experiment=e7_gencurric_s2_unigru128   # stage 2 (after stage 1 uploads best.ckpt)
```

Cloud (colab/kaggle) — stream everything from R2:

```bash
omnirun submit --backend colab --gpu-type L4 --gpus 1 --time 3h --yes -- \
  python train.py experiment=e7_gencurric_s1_unigru128 \
    data.train.params.path=conf/online_mix/rps_generated_only_interp_dload.yaml \
    "data.valid.params.data_dir='dload:DREGON-LM-V4-michaels-valid'"
```

## Conclusion

_Pending run._
