---
experiment: ckla_p0_staticcomb
training_config: conf/experiment/ckla_p0_staticcomb.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p0_staticcomb`

## Motivation

P0 gate of the CKLA campaign (`docs/ckla-design.md` §4): does the
complex-Kalman-linear-attention temporal head — an input-dependent-rotation
complex-OU filter bank run as a sequence mixer — track harmonic combs at
matched budget against the E8 transformer arm? Static-comb noise carries
zero amplitude→RPS information, so comb-frequency tracking is the only
available cue: the exact task the complex-OU inductive bias claims. An
architecture that cannot at least match the transformer here, on unlimited
on-distribution synthetic data, has no path to beating it on real data.

## Setup

Exact clone of `e8_staticcomb_s1_transformer` (data
`rps_static_comb_only`, loss `pit_mse`, metrics `rps`, patience 8, batch
16, lr 1e-3, `samples_per_validation=5000`) with `model` →
`simple_conv_v2_ckla_mag` (stft_mag front-end, isolating the head swap).
Cloud: override train policy to `rps_static_comb_only_dload.yaml`, valid
to dload. Compare train-dist PIT-MSE at common epochs vs the E8
transformer (wandb); fixed real valid = transfer read.

Train: `python train.py experiment=ckla_p0_staticcomb`.

## Conclusion

_Pending run._
