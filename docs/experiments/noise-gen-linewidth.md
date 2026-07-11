# E6 — Harmonic Linewidth in the Noise Generator (jitter injection)

**Status:** done — **positive result** (RPS-jitter injection wins; latent
regularization free on top) · **Date:** 2026-07-11

## Motivation

The RPS-refinement audit (`rps-trajectory-refinement.md`) reframed the
mid-frequency harmonic washout in generator training: real rotor speeds carry
~0.6 rev/s of fast, zero-mean, label-invisible jitter, which broadens real
harmonic k by ±k·σ Hz, while a telemetry-conditioned oscillator bank renders
clean tones the loss then suppresses. Hypothesis: model the linewidth *inside
the generator*. The user's earlier `use_random_phases` (STFT phase scrambling
→ constant ~30 Hz linewidth, coherence destroyed) is the crude ablation.

## Arms (swapped DREGON+Michael's split, differ ONLY in model flag)

- `e6_noisegen_baseline` — clean oscillator bank.
- `e6_noisegen_randphase` — `use_random_phases: true`.
- `e6_noisegen_jitter` — OU perturbation of the conditioning fundamental
  inside the emitter (`rps_jitter_sigma=0.6`, `tau=0.016 s`; calibrated from
  the refinement-audit NPZs by `scripts/calibrate_rps_jitter.py`; train-on,
  eval-off with forward override) → coherent, frequency-proportional
  broadening across all harmonics of a rotor.
- `e6_noisegen_jitter_latreg` — jitter + relative z-noise (0.1×RMS, train-only)
  + spectral-normed FiLM (Lipschitz conditioning) for latent smoothness.

## Results

Comb-masked mean |Δlog-mag| (dB) along telemetry harmonic tracks vs real
recordings (notebook `noise_gen_real_vs_generated.ipynb`, four-way section):

| DREGON (in-flight) | k<10 | k10–25 | k25–40 | msSTFT |
|---|---|---|---|---|
| baseline | 9.00 | 7.31 | 7.82 | 5.67 |
| randphase | 8.29 | 7.33 | 6.91 | 5.49 |
| jitter | 7.26 | 6.34 | 6.53 | 4.96 |
| **jitter_latreg** | **7.04** | 6.40 | **6.36** | **4.85** |

FLY124 (out-of-train, near-idle throttle): four-way tie (~7.3/6.3/6.1/4.4) —
the DREGON-calibrated σ does not transfer to the M100 at idle; **per-drone
jitter calibration (or learnable σ per codebook entry) is the follow-up**.
val/mrstft (eval renders jitter OFF): baseline 9.744 ≈ jitter 9.702,
randphase 9.280, latreg 9.505.

**Winner: `jitter_latreg`** — coherent ∝k linewidth beats both clean tones and
constant-width scrambling by 0.7–2.0 dB across every DREGON band, and the
latent regularization costs nothing on real-data comb fidelity while buying a
smooth conditioning space for vicinal sampling/interpolation (the codebook
expansion prerequisite).

## Engineering findings (the round took a day instead of an afternoon)

- The generator forward was col2im/complex-exp bound: 2.9–4.5 s/it on
  T4/P100 ≈ CPU speed. Fixed (fold-free 50% OLA, `torch.polar`,
  FIR-vectorized OU): **T4 fwd 161 ms / fwd+bwd 464 ms**, full run < 55 min →
  gpushort is the right backend again. `epochs 60 / patience 8` (plateau by
  ~ep 12). `amp: false` required (complex cumprod not implemented for
  ComplexHalf).
- `frames:` dload streaming for noise-gen needed four fixes (resolve_source
  Path-wrapping, float64 tick overflow on epoch timestamps, geometry from
  published frames, telemetry stamp-bounds trimming) — commits
  1e0ad3f..62cee4f.
- wandb now logs `val/loss` and real-vs-generated audio pairs for
  noise_generation (was: target-only "mixture" fallback — looked like the
  model reproduced real audio).
- kaggle kernels cannot be cancelled via API (omnirun#14) — "cancelled" jobs
  silently hold both GPU slots.

## Checkpoints

All four: `omnirun-outputs/r2-artifacts/e6_noisegen_*_best.ckpt` (canonical:
R2 `artifacts/<experiment>/checkpoints/best.ckpt`). Load recipe in the
notebook's `load_variant` (composite state_dict; latreg needs
`film_spectral_norm=True` at build).

## Next

Per-drone jitter σ (learnable per codebook entry or calibrated); generated-
noise augmentation retry (E4 redo) with the jitter_latreg checkpoint +
curriculum separation; codebook expansion with external constant-RPS data.
