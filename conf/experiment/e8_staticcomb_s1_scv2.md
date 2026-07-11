---
experiment: e8_staticcomb_s1_scv2
training_config: conf/experiment/e8_staticcomb_s1_scv2.yaml
batch: docs/experiments/e8-static-comb.md
---

# `e8_staticcomb_s1_scv2`

## Motivation

E8 sim-to-real probe with the analytic **static-comb** noise model (simple_conv_v2
head — the E4/E7 baseline arch). Amplitudes are static and RPS-independent and >=30% of each
rotor's harmonics clear the broadband floor, so the ONLY cue for RPS is the
comb's frequency spacing — forcing harmonic tracking. Train on static-comb only,
validate on the fixed **real** DREGON+Michael's split. Contrast with the E7
neural-generated arm (real val PIT MSE ~222, R2 -10.5).

## Setup

Data `rps_static_comb_only` (analytic static-comb train stream, fixed real
valid), model `simple_conv_v2`, loss `pit_mse`, metrics `rps`. Patience 8, batch 16,
`samples_per_validation=5000`. Cloud: override train policy to
`rps_static_comb_only_dload.yaml`, valid to `dload:DREGON-LM-V4-michaels-valid`.
See the [E8 batch doc](../../docs/experiments/e8-static-comb.md).

Train: `python train.py experiment=e8_staticcomb_s1_scv2`.

## Conclusion

_Pending run._
