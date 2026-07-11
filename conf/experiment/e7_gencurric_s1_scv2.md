---
experiment: e7_gencurric_s1_scv2
training_config: conf/experiment/e7_gencurric_s1_scv2.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s1_scv2`

## Motivation

Curriculum **Stage 1**, plain `simple_conv_v2` head (E4 lineage). Train on
**generated noise only** (vicinal-interp E6 generator), validate on the fixed
**real** DREGON+Michael's split — the sim-to-real probe. No augmentation.

## Setup

Identical to [`e7_gencurric_s1_unigru128`](e7_gencurric_s1_unigru128.md) but
model `simple_conv_v2`. Patience 5; `best.ckpt` warm-starts
`e7_gencurric_s2_scv2`. See the
[E7 batch doc](../../docs/experiments/e7-gen-curriculum.md).

Train: `python train.py experiment=e7_gencurric_s1_scv2`.

## Conclusion

_Pending run._
