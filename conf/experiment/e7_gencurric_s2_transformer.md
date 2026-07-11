---
experiment: e7_gencurric_s2_transformer
training_config: conf/experiment/e7_gencurric_s2_transformer.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s2_transformer`

## Motivation

Curriculum **Stage 2**, `simple_conv_v2_transformer` head. **Real-only
fine-tune** warm-started from the Stage-1 gen-only checkpoint
(`e7_gencurric_s1_transformer`). No augmentation.

## Setup

Identical to [`e7_gencurric_s2_unigru128`](e7_gencurric_s2_unigru128.md) but
model `simple_conv_v2_transformer` and `checkpoint:
r2://ml-data/artifacts/e7_gencurric_s1_transformer/checkpoints/best.ckpt`.
Patience 20. Requires Stage 1 to have run. See the
[E7 batch doc](../../docs/experiments/e7-gen-curriculum.md).

Train: `python train.py experiment=e7_gencurric_s2_transformer`.

## Conclusion

_Pending run._
