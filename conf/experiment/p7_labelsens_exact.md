---
experiment: p7_labelsens_exact
training_config: conf/experiment/p7_labelsens_exact.yaml
batch: docs/experiments/generator-label-sensitivity.md
---

# `p7_labelsens_exact`

## Motivation

ARM A — the control. Exact labels: the generator is told exactly the
trajectory that made its target. Everything that is not label noise — the MSSTFT
scale mix, the mean-over-bins reduction, the emitter's capacity — acts on this arm
identically to every other. If A itself underfits the high harmonics, loss design
is the cause and the other three arms cannot rescue the label-noise hypothesis.

Full batch context: [Generator label sensitivity](../../docs/experiments/generator-label-sensitivity.md).

## Setup

Data `static_comb_gen` with `label_mode: exact` — a frozen-profile analytic comb
(one rotor, one mic, 1 s, 80 harmonics) whose target audio is always rendered from
the TRUE trajectory, so the four arms differ in the conditioning and in nothing
else. Model `positional_harmonic_gen` (unconditioned, `n_harmonics: 100`), loss
`multiscale_stft`, metrics `noise_gen_spectral`, `amp: false`. Shared budget:
seed 1234, 40 epochs, patience 8, batch 32, Adam 1e-3, 8000 train / 512 valid.

Train with `python train.py experiment=p7_labelsens_exact`; read the arms
together with `python scripts/gen_label_sensitivity_eval.py`.

## Conclusion

_Pending run._
