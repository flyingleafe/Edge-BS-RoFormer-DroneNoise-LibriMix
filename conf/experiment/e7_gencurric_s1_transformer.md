---
experiment: e7_gencurric_s1_transformer
training_config: conf/experiment/e7_gencurric_s1_transformer.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s1_transformer`

## Motivation

Curriculum **Stage 1**, `simple_conv_v2_transformer` head (the E5 time-warp
best-overall arch). Train on **generated noise only** (vicinal-interp E6
generator), validate on the fixed **real** DREGON+Michael's split. No
augmentation.

## Setup

Identical to [`e7_gencurric_s1_unigru128`](e7_gencurric_s1_unigru128.md) but
model `simple_conv_v2_transformer`. Patience 5; `best.ckpt` warm-starts
`e7_gencurric_s2_transformer`. See the
[E7 batch doc](../../docs/experiments/e7-gen-curriculum.md).

Train: `python train.py experiment=e7_gencurric_s1_transformer`.

## Conclusion

_Pending run._
