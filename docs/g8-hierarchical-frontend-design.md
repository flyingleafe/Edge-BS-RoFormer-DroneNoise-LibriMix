# G8 — Hierarchical front-end: multi-resolution pyramid + harmonic fusion + Fisher decode

Design doc for the next VK-parity front-end phase (criterion 2.3). Grounded
in the 2026-07-24 literature sweep (bibliography tag `g7-frontend`) and the
G1–G6 evidence chain (`docs/experiments/g1-vk-parity.md`).

## Why (the evidence so far)

The single-window STFT (n_fft 2048 @16 kHz: 7.8 Hz bins, 32 ms hop-frames)
is caught in the classic resolution conundrum, and it lands differently per
harmonic:

- At the **fundamental** (30–120 Hz), 7.8 Hz bins are catastrophically
  coarse (a whole rotor-speed range spans ~12 bins), but the *signal* there
  moves slowly in Hz (1 rev/s error = 1 Hz at k=1) — what's needed is fine
  FREQUENCY resolution; time can be coarse.
- At **high harmonics** (k≈15–25, 1–2 kHz), 7.8 Hz bins are fine in rev/s
  terms (0.5 rev/s error = ~10 Hz at k=20), but the comb moves *fast* in Hz
  (the same RPS wiggle is k× amplified) — what's needed is fine TIME
  resolution and phase stability; frequency bins can be coarse because IF
  provides sub-bin readings.

One window cannot serve both. Constant-Q allocates resolution exactly
backwards (fine at the fundamental, coarse at high k — G2a refuted, 3.32
protocol / 195.5 val). The IF channel (G2b) is the only arm that beat the
baseline (2.481 vs 2.62) — phase evidence works; the comb-ridge front-end
(G4a/b) died at val. Severe overfitting on ~2 drones of real data (val
doubles within 20 epochs of best, all arms) means **parameter-light
structural priors beat learned modules** — which is also the literature's
converged position (LEAF filters barely move from init; free waveform
convnets provably unstable; PESTO's structural equivariance substitutes for
data at <30k params).

## Design — three composable components

### C1 · Octave-banded multi-window STFT pyramid with per-band IF (~0 params)

3–4 parallel STFTs, each *used only in its band*, mirroring the
wavelet/MuReNN resolution allocation without CQT's high-k coarseness:

| band | n_fft | Δf | window | serves |
|---|---|---|---|---|
| 30–250 Hz | 8192 | 1.95 Hz | 512 ms | fundamentals + k≤2: fine Hz, "wide filters, stable phases" |
| 250–1000 Hz | 4096 | 3.9 Hz | 256 ms | mid harmonics |
| 1–2 kHz | 2048 | 7.8 Hz | 128 ms | k≈10–25: fine time, IF sub-bin |
| (2–4 kHz opt.) | 1024 | 15.6 Hz | 64 ms | headroom band |

Each band contributes log-magnitude + IF-deviation channels (the proven
G2b estimator per band), all resampled onto a **common log-frequency axis**
and the standard hop-512 time grid. Log-f is chosen deliberately: it makes
comb patterns *shift-equivariant* in f0 (a comb at 80 rev/s is a translate
of one at 60), which C2 exploits and which composes with 4-comb PIT.
Resampling is fixed interpolation (precomputed index/weight buffers, same
machinery as G4's gather — that part of G4 worked; its *features* were the
failure).

### C2 · Harmonic-aligned convolutional fusion (~0–50k params)

Deep-Salience-style harmonic stacking on the log-f axis: channels resampled
at offsets log(k) for k = 1..K so a 1×1 conv sees each candidate comb
coherently — or equivalently HarmoF0/HPPNet log-spaced dilated convolution.
This is the field's only *polyphony-proven, parameter-light* cross-harmonic
mechanism (HPPNet: 88-voice piano SOTA with a small model; our earlier
harmonic-stacking failure — Basic Pitch salience — was a *resolution*
failure of its CQT substrate, not of the alignment mechanism, which is
front-end-agnostic).

**Harmonic cross-attention is deliberately parked**: hFT-Transformer-style
frequency attention is the data-hungry version of the same idea (~5M
params) and sits on the wrong side of our overfitting constraint. Revisit
only if the distillation program multiplies the data pool.

### C3 · Fisher-weighted coarse-to-fine residual decode (novel; fixed/few params)

The residual architecture, i.e. VK's information logic as a differentiable
head — no prior art does this in a DNN (checked: CREPE/FCNF0/PENN/PESTO/
SwiftF0 all do exactly ONE coarse→fine step, bins → weighted-average
decode, single-f0):

1. **Base estimate** from the low band: coarse per-rotor f0 posterior
   (PENN-style bins over 30–120 rev/s + weighted decode). PIT attaches
   here, at the coarse stage.
2. **Residual updates** from higher harmonics: at harmonic k, the IF
   reading near k·f̂0 measures f0 with k× finer resolution but k-fold
   wrapping ambiguity; the coarse estimate *unwraps* it (selects the
   branch), and the update is combined with Fisher weights
   w_k ∝ k²·SNR_k — exactly the VK phase-slope fusion.
3. Optionally **unrolled 2–3 iterations** (algorithm-unrolling recipe:
   each iteration re-centers the harmonic readings on the refined
   estimate), and/or a KalmanNet-style learned temporal gain for the
   smoothness prior. **K2-kill constraint respected**: this is a
   refinement *layer* on top of the trunk's coarse posterior, never a
   replacement of the trunk (the joint neural-tracker update was the part
   that collapsed in K2).

Steps 2–3 have zero or near-zero trainable parameters — the structure IS
the prior. This imports the mechanism that makes VK win (high-k phase
slopes, Fisher-fused) into a causal, fast, trainable estimator, and is a
publishable architectural contribution on its own.

## Explicitly rejected (with reasons)

- **Learnable filterbanks** (LEAF/SincNet): evidence says filters don't
  move from init; the pyramid gives the allocation by construction.
- **Constant-Q anywhere in the path**: refuted twice (G2a; salience
  baselines) — resolution allocation is backwards for combs.
- **hFT-style attention**: parameter/data budget (above).
- **Scattering (pitch-spiral)**: kept as the fixed-feature fallback if the
  learned trunk keeps overfitting — its octave-coupling + chirp
  linearization is free; not the first arm.

## Rollout (each arm ~40 min kaggle + protocol eval)

- **G8a**: C1 alone on the existing trunk (in_ch = 2×bands via the
  frontend-aware first conv). Isolates the resolution fix. Gate: val <
  63.7 (g2_if) and protocol DREGON < 2.48.
- **G8b**: C1 + C2 (stacked-harmonic channels, small conv fusion). Gate:
  beat G8a.
- **G8c**: + C3 decode head replacing the plain regression head (trunk
  kept). Gate: beat G8b; target the VK bars (DREGON 0.68–0.74 with the
  telemetry-jitter label floor ~0.6 as the hard limit of what MSE-vs-
  telemetry can show).
- Training schedule: the winning augmentation schedule from G6/G7 readout.

## Risks

- Pyramid bookkeeping: per-band hop alignment to the hop-512 grid (the
  8192 window's effective time smear is 512 ms — acceptable at k≤2 where
  trajectories are slow, but must not leak into high bands).
- Log-f resample of the 8192 band can alias its 1.95 Hz structure if the
  log grid is too coarse at the bottom — grid density set by the low band.
- C3's unwrap needs the coarse estimate within ±f0/(2k_max) — bound k_max
  per iteration by the current posterior width (annealing, as in VK
  capture→refine).
