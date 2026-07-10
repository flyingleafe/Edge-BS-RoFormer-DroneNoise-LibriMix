# RPS Trajectory Refinement by Comb Alignment (Stages A–D)

**Status:** done — method built + validated; headline finding is a *reframe*
(labels were nearly unbiased; washout ≈ jitter linewidth) · **Date:** 2026-07-10

## Motivation

Hypothesis from the generator-improvement discussion: telemetry RPS label
error δ displaces harmonic k by k·δ Hz, so mid-frequency harmonics land off
their spectral peaks during noise-generator training and get amplitude-
suppressed (the observed mid-harmonic washout). If labels can be *refined*
against the audio (comb alignment), generator + predictor training improve,
and a half-trained predictor + refiner can bootstrap-annotate unlabeled data.

## Method (src/data_processing/rps_refinement.py)

Separable NLS: amplitudes/phases linear (VP-transform primitives,
`lstsq_VP_transform` + new fast `method="gelsy"`), trajectories = low-dim
spline corrections. Stages: A clock-offset scan; B windowed constant-δ grid
(capture); C joint spline gradient refinement on the multichannel comb
log-mag score; D coherent phase-slope refinement via narrowband harmonic
demodulation (Fisher weights k²|z|²). Confidence = on-comb vs de-tuned comb
contrast. Fit metric = joint block LSQ residual (same VP basis the generator
synthesizes in). 8 unit tests; scripts `rps_refinement_{validation,spcup,robustness}.py`.

## Results

- **DREGON natural experiment** (5 room1 recordings, command→measured,
  pooled): command err 0.633 / bias −0.057 (vs 0.25s-smoothed measured:
  0.484 / **+0.017**); stage B+C **0.848 / −0.440** (biased!); stage D
  0.638 / −0.078 (vs smoothed: **−0.004**); measured best LSQ everywhere.
- **B+C bias mechanism**: rotors fly as two tight pairs (~0.65 rev/s apart)
  → low/mid harmonics merge; low-band confounders outvote resolved high k
  under uniform averaging. High-k-only flips bias to +0.8 (twin capture).
  Magnitude ridges cannot arbitrate twin pairs; phase (stage D) can.
- **Irreducible jitter**: command's 0.63 unsigned error is zero-mean fast
  jitter; recovering it needs low k (modulation index), twin rejection needs
  high k — structural trap. **Washout reframe:** ±0.6 rev/s label-invisible
  jitter smears harmonic k=30 by ±18 Hz → generator renders clean tones the
  loss then suppresses. Fix belongs in the generator: per-harmonic
  linewidth/phase-diffusion ∝ k, not better labels.
- **SPCup blind annotation** (7 recs, 6 rigs): 5 lock (28–60 rev/s, one true
  two-comb resolution), 1 honest refusal (conf 0.018), 1 calibration-tone
  trap caught by LSQ residual. Bootstrap loop viable with three gates:
  confidence + rotor uniqueness + LSQ residual.
- **Envelope**: capture basin = coarse grid range exactly (±3/±6); noise
  floor ≈ 0 dB harmonic SNR (white/pink), +5 dB (speech-shaped); confidence
  gate 0.17 → 95% precision / 80% recall, blind to identity capture; cost
  0.2–1.4× realtime single-core.

## Conclusion

Refinement pays as clock aligner (offsets to 0.21 s found), verifier, and
blind annotator — not as a label polisher on DREGON/Michael's, whose
telemetry is already within the refiner's noise floor. Trust stage D over
B+C wherever they disagree. Report:
`writing/reports/2026-07-10_rps-refinement/` (10 pp, figures from
`results/rps_refinement/**`). Next for the generator thread: per-harmonic
linewidth parameter; comb-mask loss redesign remains valid but must use
peak-snapping and should expect pair-merged low-k peaks.
