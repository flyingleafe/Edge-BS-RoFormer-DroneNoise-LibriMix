---
experiment: e7_gencurric_s2_scv2
training_config: conf/experiment/e7_gencurric_s2_scv2.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s2_scv2`

## Motivation

Curriculum **Stage 2**, `simple_conv_v2` head. **Real-only fine-tune**
warm-started from the Stage-1 gen-only checkpoint (`e7_gencurric_s1_scv2`).
No augmentation.

## Setup

Identical to [`e7_gencurric_s2_unigru128`](e7_gencurric_s2_unigru128.md) but
model `simple_conv_v2` and `checkpoint:
r2://ml-data/artifacts/e7_gencurric_s1_scv2/checkpoints/best.ckpt`. Patience 20.
Requires Stage 1 to have run. See the
[E7 batch doc](../../docs/experiments/e7-gen-curriculum.md).

Train: `python train.py experiment=e7_gencurric_s2_scv2`.

## Conclusion

_Pending run._
