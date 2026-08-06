---
experiment: p7_labelsens_tach_pure
training_config: conf/experiment/p7_labelsens_tach_pure.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `p7_labelsens_tach_pure`

## Motivation

ARM B0 — the staircase alone. Arm B's label transform at unit scale: the
refresh-interval mean, the 0.269 rev/s quantization lattice and the 49.7 Hz
zero-order hold, with no 0.542 % constant bias. Label error 0.106 rev/s rms,
which displaces harmonic k by 0.106*k Hz — about one bin at k = 80 at the loss's
finest scale (n_fft 2048, 7.8 Hz bins).

The A/S/B/C round read the staircase only as `B - S`, which assumes the bias and
the staircase compose additively in the arm's response. Since arm S turned out to
be far from arm A, `B - S` is a difference of two badly damaged arms and it can
hide a staircase effect of its own size. This arm measures the staircase against
the undamaged control directly. The model-free pressure predicts 1.18 dB at
k50-80 against arm A's floor of 0.20 dB.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Setup

Data `static_comb_gen` with `label_mode: tach` and `label_scale: 1.0` — a
frozen-profile analytic comb (one rotor, one mic, 1 s, 80 harmonics) whose target
audio is always rendered from the TRUE trajectory, so the arms differ in the
conditioning and in nothing else. Model `positional_harmonic_gen` (unconditioned,
`n_harmonics: 100`), loss `multiscale_stft`, metrics `noise_gen_spectral`,
`amp: false`. Shared budget: seed 1234, 40 epochs, patience 8, batch 32, Adam
1e-3, 8000 train / 512 valid.

Train with `python train.py experiment=p7_labelsens_tach_pure`; read the arms
together with `python scripts/gen_label_sensitivity_eval.py --arms exact,scale,tach,tach_presmooth,tach_pure`.

## Conclusion

Pending — the job is in flight. The batch doc carries the five-arm table.
