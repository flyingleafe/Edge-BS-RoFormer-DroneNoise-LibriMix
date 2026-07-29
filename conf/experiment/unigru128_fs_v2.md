---
experiment: unigru128_fs_v2
training_config: conf/experiment/unigru128_fs_v2.yaml
batch: docs/experiments/beat-vk.md
---

# `unigru128_fs_v2`

## Motivation

Beat-VK scoreboard requires architectural diversity under the SAME (best)
training regime. uni_gru128 — the online-mix arch-sweep winner — only has
E12-era (pre-uniform-freq-scale) checkpoints, which are not
regime-comparable to the fs_v2 CKLA/KLA/transformer arms. This arm trains
it under the exact g2_if_freqscale_v2 recipe (model swap only).

## Setup

Clone of `g2_if_freqscale_v2` with `model: simple_conv_v2_uni_gru128`.
Scored on the fixed raw protocol (`beatvk-valid-raw`) by
`scripts/beatvk_eval.py` alongside the other scoreboard rows.

## Conclusion

(pending)
