---
experiment: p7_labelsens_tach_presmooth
training_config: conf/experiment/p7_labelsens_tach_presmooth.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `p7_labelsens_tach_presmooth`

## Motivation

ARM C — the mitigation. Arm B's label through the tracking
campaign's 5 Hz detrended low pass (tracking.telemetry_refit.presmooth, reached via
data_processing.rps_corruption.presmooth_track). It cuts the label error from
0.106 to 0.037 rev/s while moving the TRUE track by only 0.017, so how much of arm
A this recovers is the mitigation's ceiling — and it is free, since it is a filter
on data the training loop already has.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Setup

Data `static_comb_gen` with `label_mode: tach_presmooth` — a frozen-profile analytic comb
(one rotor, one mic, 1 s, 80 harmonics) whose target audio is always rendered from
the TRUE trajectory, so the four arms differ in the conditioning and in nothing
else. Model `positional_harmonic_gen` (unconditioned, `n_harmonics: 100`), loss
`multiscale_stft`, metrics `noise_gen_spectral`, `amp: false`. Shared budget:
seed 1234, 40 epochs, patience 8, batch 32, Adam 1e-3, 8000 train / 512 valid.

Train with `python train.py experiment=p7_labelsens_tach_presmooth`; read the arms
together with `python scripts/gen_label_sensitivity_eval.py`.

## Conclusion

**The mitigation fails, and not because smoothing is wrong.** The filter does what it claims — label error 0.106 -> 0.037 rev/s, true track moved only 0.017 — but it removes the staircase, which was never the binding term, and leaves the constant bias untouched. C ~ B ~ S: `mrstft` 13.31 against 13.57 and 13.71, and C - B is sign-inconsistent across the two readouts. The fix that would work is a scale correction (x0.99458) applied before conditioning, not a low pass.
