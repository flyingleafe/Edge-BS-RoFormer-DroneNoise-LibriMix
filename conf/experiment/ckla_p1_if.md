---
experiment: ckla_p1_if
training_config: conf/experiment/ckla_p1_if.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p1_if`

## Motivation

P1 real-protocol arm of the CKLA campaign (`docs/ckla-design.md` §5): the
complex-Kalman-linear-attention temporal head trained on the same
e12_real_fullflight stream and protocol as every G-series arm, so its
vk_valid_comparison score lands directly on the criterion-2.3 ledger.
Hypothesis: the complex-OU filter-bank inductive bias (uncertainty-gated
integration + input-dependent rotation = closed-loop frequency tracking)
buys what generic attention could not — the structural residual between
the neural floor (2.481) and the VK bars (0.68–0.74 / 1.027).

## Setup

Exact clone of `g2_if_transformer` (data `e12_real_fullflight`, loss
`pit_mse`, patience 20, batch 16, lr 1e-3) with `model` →
`simple_conv_v2_ckla` (stft_mag_if front-end, ComplexKLA head). Gate only
after `ckla_p0_staticcomb` passes (design §4). Eval:
`scripts/rps_predictor_vk_eval.py`, pools dregon_cruise / fly124_cruise.

Train: `python train.py experiment=ckla_p1_if`.

## Conclusion

_Pending run._
