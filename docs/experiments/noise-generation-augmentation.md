# Harmonic Noise Generator as an RPS-Training Augmentation Source

**Status:** done — **negative result** (generated-noise augmentation *degrades* RPS prediction) · **Dates:** 2026-06-26 – 2026-07-03

## Motivation

RPS-predictor training relies on a handful of real recordings (DREGON + Michael's
FLY-series) mixed online with LibriSpeech at ultra-low SNR. The bet — from the
2026-06-30 supervisor slides ("train RPS predictors with synthetic-trajectory
augmentation") — is that a **learned generative model of rotor harmonic noise**
(the inverse of RPS prediction: RPS + array geometry → multichannel noise) can
synthesize unlimited, perfectly-labelled training variety and thereby improve
RPS prediction, especially generalization to unseen trajectories/drones.

## Implementation (three prerequisite stages)

1. **Generator + online data pipeline (E2, 2026-06-26).** `PositionalHarmonicNoiseGen`
   (single-rotor harmonic + filtered-noise emitter; differentiable propagation to
   8 mics via 1/r attenuation + fractional delay), trained jointly on two drones
   with per-drone conditioning, streaming DREGON `in_flight_noise` + Michael's
   `FLY125`/`FLY124`.
2. **Swapped split + random phase + smoothness (E3, 2026-07-01).** The original
   DREGON split was backwards (trained on 1 room, validated on 5); corrected to
   train on room2 (5 recs) + FLY125, validate on room1 + FLY124. Added
   random-per-harmonic initial phase at train (zero phase at eval) and opt-in
   squared-2nd-difference smoothness penalties on harmonic-amplitude curves (time)
   and diffuse-noise filter shape (time+freq), targeting DREGON's weak
   mid-frequency harmonics.
3. **RPS-training augmentation wiring (E4, 2026-07-02).** A `kind: generated`
   source (`GeneratedNoisePool`): a spawned producer owns the CUDA context for a
   frozen noise-gen checkpoint and renders into a shared-memory ring buffer
   (seqlock) read lock-free by DataLoader workers; each chunk's synthetic RPS
   trajectory is its exact label. A/B config pair: baseline (real noise only) vs.
   treatment (+ one generated `michaels` source). Plumbing verified by
   `tests/data_processing/test_generated_noise.py` (42 pass).

## Results

### E3 — noise-gen smoothness sweep (2026-07-02, 9 runs)

1-D sweep over `harm_smooth_weight` and `noise_smooth_weight` (orthogonal
regularisers, swept independently). `harm_smooth_weight = 1e-1` is the nominal
winner (best spectral val **5.3506**), but the top three are within ~0.008 —
**noise-level**:

| harm_smooth | noise_smooth | best val |
|---|---|---|
| **1e-1** | 0 | **5.3506** |
| 0 (baseline) | 0 | 5.3554 |
| 0 | 10 | 5.3581 |

Harm smoothness peaks at 1e-1 (1e-2 does little; 1/10 over-smooth and hurt);
noise smoothness only helps at large weights (>1) and *actively hurts* at 1e-2
(5.60). **Smoothness does not meaningfully move the raw spectral val loss.**
Whether it yields qualitatively cleaner harmonic/noise *components* is an open
analysis (checkpoints at `.../noise_gen_sweep/{baseline,harm_*,noise_*}`; render
per-component via the emitter with `return_dict=True` →
`harm_amps`/`noise_amps`/`harm_noise`/`diff_noise`).

### E4 — generated-noise augmentation for RPS prediction (2026-07-03, 11 Slurm jobs, 2 usable)

Augmenting the RPS online-mix stream with a live generated `michaels` source
(weight 0.5, ~⅓ of noise batches; the **baseline** no-smoothness noise-gen
checkpoint) **degrades** RPS prediction versus the no-generator online-mix
baseline:

| Model | Online-mix baseline | + generated noise | Δ PIT MSE |
|---|---|---|---|
| `simple_conv_v2_uni_gru128` | 7.33 (R² 0.822) | **9.29** (R² 0.791) | **+27%** |
| `simple_conv_v2_transformer` | 8.46 (R² 0.808) | **10.63** (R² 0.762) | **+26%** |

Only 2 of ~11 jobs produced usable checkpoints; the rest died on NaN divergence,
OOM (V100 — the producer shares VRAM, so batch ≤ 8), or broken-GPU /
multiprocessing node issues. The **transformer** shows a textbook overfitting
curve (train MSE 9.9→3.7 while val PIT 10.6→**43.6** after epoch 9): global
attention lets it memorize both the narrow set of synthetic RPS trajectories and
the generator's spectral fingerprint, pivoting to generator artifacts once the
real harmonic structure is exhausted. The causal uni-GRU is regularised by its
frame-by-frame constraint and degrades less, but still lands below baseline.

**Why it fails (analysis):** (1) generator quality is the bottleneck — the
baseline checkpoint has imperfect mid-frequency harmonics and a fixed
filter-envelope the RPS model latches onto as a shortcut feature; (2)
RPS-trajectory diversity is narrow (OU + Poisson maneuvers); (3) the online mixer
diversifies only the acoustic dressing (speech, SNR), not the RPS curves
themselves.

## Conclusion

Generated-noise augmentation, as built, **hurts** RPS prediction — a negative
result driven by exploitable generator artifacts plus insufficient
RPS-trajectory diversity, not by the plumbing (verified end-to-end). Highest-value
next steps, in order: (1) swap in a **smoothness-trained** noise-gen checkpoint
(`harm=1e-1`) to reduce exploitable artifacts; (2) **augment the RPS trajectories
themselves** (time-warp / speed-perturb) — directly attacks the memorization root
cause, independent of acoustic augmentation; (3) drop the generated weight to
0.25/0.1; (4) single producer serving multiple drones (michaels + dregon) to lift
the VRAM ceiling. Investigation diagnostics: `scripts/diag_generated_noise*.py`,
`scripts/diag_online_mix_integrity.py`, `scripts/diag_tune_gen.py`; the cluster
online-mix policy used is `conf/online_mix/online_mix_generated_augment_gpfs.yaml`.
