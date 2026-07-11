---
experiment: e7_gencurric_s2_unigru128
training_config: conf/experiment/e7_gencurric_s2_unigru128.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s2_unigru128`

## Motivation

Curriculum **Stage 2**, uni_gru128 head. **Real-only fine-tune** warm-started
from the Stage-1 gen-only checkpoint (`e7_gencurric_s1_unigru128`). Tests whether
gen-only pretraining is a better initialisation than from-scratch for the
real-data RPS task (baseline C10 online: PIT MSE 7.33). No augmentation.

## Setup

Data `online_mix_v4_michaels_no_aug` (real DREGON−room1 + Michael's FLY125 +
LibriSpeech), model `simple_conv_v2_uni_gru128`, loss `pit_mse`, metrics `rps`.
Warm-start via `checkpoint:
r2://ml-data/artifacts/e7_gencurric_s1_unigru128/checkpoints/best.ckpt`
(`training.loop._warm_start` → fresh optimizer/scheduler/early-stopping,
`strict=False`). Patience 20. **Requires Stage 1 to have run + uploaded its
checkpoint.** See the [E7 batch doc](../../docs/experiments/e7-gen-curriculum.md).

Train: `python train.py experiment=e7_gencurric_s2_unigru128`.

## Conclusion

_Pending run._
