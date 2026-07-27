---
experiment: ckla_p1_pnoise_norot
training_config: conf/experiment/ckla_p1_pnoise_norot.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p1_pnoise_norot`

## Motivation

The goal-deciding rotation-attribution cell. Every rotation null so far
(synthetic 1 s eval+train, synthetic 4 s eval, real 1 s train: norot 83.9
vs base 85.2) was measured in the gain-collapsed accumulator regime
(activation analysis §A2: gain 1e-7..1e-4) where the state barely
integrates evidence — rotation structurally cannot pay off there. With
p_init 1.0 the filter has live gain and actually tracks (`ckla_p1_pnoise`
best 44.8 vs base 85.2). This arm removes the complex path in that
regime: the pnoise − pnoise_norot gap is the complex extension's true
contribution in a working filter. Null here + null everywhere else =
the definitive quantified refutation of the complex hypothesis; a gap
here = rotation matters exactly when the filter filters.

## Setup

Exact clone of `ckla_p1_pnoise` with `model` →
`simple_conv_v2_ckla_pnoise_norot` (registry `simple_conv_v2_ckla_norot`
+ p_init 1.0).

Train: `python train.py experiment=ckla_p1_pnoise_norot`.

## Conclusion

_Pending run._
