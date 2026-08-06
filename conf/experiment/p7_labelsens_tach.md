---
experiment: p7_labelsens_tach
training_config: conf/experiment/p7_labelsens_tach.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `p7_labelsens_tach`

## Motivation

ARM B — the hypothesis. The label passes through the tachometer's whole
measurement model: 0.99458 x truth, then the refresh-interval mean, then the
0.269 rev/s quantization lattice, then the 49.7 Hz zero-order hold. Measured label
error 0.106 rev/s rms, which displaces harmonic k by 0.106*k Hz — sub-bin below
k ~ 30 at the loss's finest scale (n_fft 2048, 7.8 Hz bins), about one bin at
k = 80. The prediction is a per-k underfit that switches on around there.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Setup

Data `static_comb_gen` with `label_mode: tach` — a frozen-profile analytic comb
(one rotor, one mic, 1 s, 80 harmonics) whose target audio is always rendered from
the TRUE trajectory, so the four arms differ in the conditioning and in nothing
else. Model `positional_harmonic_gen` (unconditioned, `n_harmonics: 100`), loss
`multiscale_stft`, metrics `noise_gen_spectral`, `amp: false`. Shared budget:
seed 1234, 40 epochs, patience 8, batch 32, Adam 1e-3, 8000 train / 512 valid.

Train with `python train.py experiment=p7_labelsens_tach`; read the arms
together with `python scripts/gen_label_sensitivity_eval.py`.

## Conclusion

_Pending run._
