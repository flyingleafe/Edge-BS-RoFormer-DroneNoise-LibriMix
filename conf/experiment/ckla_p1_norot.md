---
experiment: ckla_p1_norot
training_config: conf/experiment/ckla_p1_norot.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p1_norot`

## Motivation

Rotation-off twin of `ckla_p1_if`, attributing the complex path's share of
the P1 real-protocol result (dregon_cruise 2.87 / fly124_cruise 1.39 —
the FLY124 score being the best neural cross-drone number to date). At 1 s
static-comb the rotation contributed exactly nothing (`ckla_p0_norot`
21.51 vs 21.70); real data has richer temporal structure (drift, maneuver
transients), so the null does not automatically transfer.

## Setup

Exact clone of `ckla_p1_if` with `model` → `simple_conv_v2_ckla_norot`
(real-KLA head, stft_mag_if front-end).

Train: `python train.py experiment=ckla_p1_norot`.

## Conclusion

_Pending run._
