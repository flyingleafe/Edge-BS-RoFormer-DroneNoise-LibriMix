---
experiment: fkla_fs_v2
training_config: conf/experiment/fkla_fs_v2.yaml
batch: docs/experiments/ckla.md
---

# `fkla_fs_v2`

## Motivation

Cross-implementation plain-KLA control for the CKLA ladder. The in-repo
rotation ablation (`ckla_norot_fs_v2`) disables the complex rotation inside
*our own* CKLA scan; if a result there is an artifact of our implementation,
the ablation inherits it. This arm answers the same question through an
**independent codebase**: the flat exact-KLA layer from the kla-loglinear
repo (vendored at `src/models/fkla/`, commit 11e5a39 — flat readout, real OU
state, learned fold weight, no rotation by construction), dropped into the
exact `SimpleConvV2CKLA` wrapper (same `stft_mag_if` front-end, encoder,
pooling, head wiring, d_model 128 / 2 layers / n_state 16, and the same
p_init 1.0 gain fix). Agreement between `fkla_fs_v2` and `ckla_norot_fs_v2`
validates both implementations; disagreement localizes a bug.

## Setup

Exact clone of `ckla_pnoise_fs_v2` (uniform freq-scale v2 policy p=1.0,
α∈[0.7, 1.3], batch 128, samples_per_validation 40000) with
`model=simple_conv_v2_fkla`. Compare against `ckla_pnoise_fs_v2` (rotating)
and `ckla_norot_fs_v2` (in-repo no-rotation) on the A6 scale-response probe
and cruise pools.

Train: `python train.py experiment=fkla_fs_v2`.

## Conclusion

_Pending run._
