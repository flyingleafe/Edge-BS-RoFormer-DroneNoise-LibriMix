# Per-rotor attribution of the VK broadband residual: REFUTED on both arrays

**Date**: 2026-08-12 · **Status**: DONE — verdict: supervise per-MIC floors.
Code: `src/experiments/residual_attribution/` + `scripts/residual_attribution.py`;
results + figures in `results/residual_attribution/` (gitignored; reproduce in
minutes on CPU, commands in the script docstring).

## Verdict

Per-rotor attribution of the broadband residual is NOT trustworthy on either
array. Amplitude-target training must supervise **per-microphone floors
D_m(f)** (fitted, stored in the result JSONs), not per-rotor PSDs.

1. Geometry is good enough: above 339 Hz (DREGON) / 630 Hz (Michael's) the
   four rotor steering signatures are near-orthogonal (VIF ≈ 1), and synthetic
   data through the repo's own `propagate` recovers known per-rotor PSDs to
   0.5 dB (≈20 dB dynamic range above 500 Hz).
2. The residual is almost all incoherent: the per-mic diagonal term carries
   90.0 % (DREGON) / 69.7 / 73.4 % (FLY124/125) of residual power.
3. The rotor hypothesis is not earned: a WRONG geometry of equal dimension
   (rotor square turned 45°, or random positions) fits the measured
   cross-spectra as well or better, while the same control separates cleanly
   on ideal data. At 1-2 kHz the 45°-rotated geometry beats the truth.

**Epistemics lesson (extends the wind-channel one):** non-degenerate shares
with tight bootstrap intervals (±0.01) can still be UNEARNED. Only a null
control on the geometry itself separates a real spatial fit from a flexible
parameterization absorbing structure. Run geometry nulls before believing any
array-attribution result.

## The number that binds the generator design

Measured per-mic residual energy spread on DREGON: **8.54 dB**. Four equal
rotors radiating 1/r into this array can produce **1.59 dB**. The generator's
propagated per-rotor noise branch therefore CANNOT reproduce the observed
per-mic pattern — an incoherent per-mic term (or per-mic calibration with
enough range) is mandatory in any noise-branch supervision.

Why the free-field four-rotor model fails (not separated): extended/wake
sources rather than hub point sources, reverberation (room1 indoors),
airframe scattering, and diaphragm flow noise (DREGON mics sit in the wake).

Caveats: one DREGON + two Michael's recordings; Michael's geometry is an
unvalidated ring model; residual defined by the 1 Hz-bandwidth VK solve
(leftover tonal energy shows as narrow coherence peaks).
