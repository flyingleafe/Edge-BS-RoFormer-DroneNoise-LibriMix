---
experiment: p7_labelsens_scale
training_config: conf/experiment/p7_labelsens_scale.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `p7_labelsens_scale`

## Motivation

ARM S — the constant-bias control. The label is 0.99458 x truth (DREGON's
measured 0.542 % telemetry over-report, inverted) with no staircase. A constant
gain is an invertible reparameterization, so a conditioned generator should absorb
it and land on arm A. This arm is what makes arm B readable: if S matches A, B's
deficit is the staircase alone; if S degrades with k, the bias is not benign and B
carries two effects at once.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Setup

Data `static_comb_gen` with `label_mode: scale` — a frozen-profile analytic comb
(one rotor, one mic, 1 s, 80 harmonics) whose target audio is always rendered from
the TRUE trajectory, so the four arms differ in the conditioning and in nothing
else. Model `positional_harmonic_gen` (unconditioned, `n_harmonics: 100`), loss
`multiscale_stft`, metrics `noise_gen_spectral`, `amp: false`. Shared budget:
seed 1234, 40 epochs, patience 8, batch 32, Adam 1e-3, 8000 train / 512 valid.

Train with `python train.py experiment=p7_labelsens_scale`; read the arms
together with `python scripts/gen_label_sensitivity_eval.py`.

## Conclusion

**The constant bias is NOT benign.** Arm S loses -1.58/-2.08/-3.84/-8.61 dB across the k bands against arm A's flat line, and `mrstft` drops 30.96 -> 13.71. The model compensates the bias in *placement* (fitted frequency scale 1.000000, centroids within 0.01 % of k*f0) but cannot keep the lines sharp while doing so: at k=47 there is no line at 3760 Hz at all, only scattered maxima ~18 dB down, and a +-2 % band recovers just 0.3 dB. This arm carries the whole label-noise effect the batch set out to find.
