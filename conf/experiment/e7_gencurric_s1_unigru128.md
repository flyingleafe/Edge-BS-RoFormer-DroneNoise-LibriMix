---
experiment: e7_gencurric_s1_unigru128
training_config: conf/experiment/e7_gencurric_s1_unigru128.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s1_unigru128`

## Motivation

Curriculum **Stage 1**, uni_gru128 head. Train the RPS predictor on
**generated noise only** — the E6 per-drone adaptive-σ generator sampled
vicinally along the DREGON↔Michael's embedding segment — and validate on the
fixed **real** DREGON+Michael's split. The narrow question: can gen-only
training reach a reasonable PIT MSE on real data? Reference is the C10 online
real-data winner for this arch (PIT MSE 7.33). No augmentation.

## Setup

Data `rps_generated_only_interp` (generated-only train stream, fixed real
valid), model `simple_conv_v2_uni_gru128`, loss `pit_mse`, metrics `rps`.
Patience 5 (curriculum hand-off); `best.ckpt` warm-starts
`e7_gencurric_s2_unigru128`. `samples_per_validation=5000`, batch 16.
Cloud: override the train policy to `rps_generated_only_interp_dload.yaml` and
valid to `dload:DREGON-LM-V4-michaels-valid`. See the
[E7 batch doc](../../docs/experiments/e7-gen-curriculum.md) for the full sampling
scheme.

Train: `python train.py experiment=e7_gencurric_s1_unigru128`.

## Conclusion

_Pending run._
