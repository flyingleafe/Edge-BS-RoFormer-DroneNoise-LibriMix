---
experiment: ckla_p1_4s
training_config: conf/experiment/ckla_p1_4s.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p1_4s`

## Motivation

4 s-context arm of the CKLA real-protocol push. The 1 s `ckla_p1_if` run
landed dregon_cruise 2.87 (above the g2_if floor 2.481) but fly124_cruise
**1.39** (vs 2.33 — best neural cross-drone score to date). Longer native
context was refuted for the transformer (G1: no gain at 4 s/8 s), but that
refutation is architecture-specific: a recurrent uncertainty-gated filter
bank integrates evidence over its whole context by construction, and the
P0b capture analysis showed the CKLA head locking on 4 s clips. This arm
tests whether native 4 s context closes the dregon_cruise gap without
giving back the FLY124 win.

## Setup

`g1_transformer_4s` recipe (data `g1_real_fullflight_4s`, batch 8,
patience 20) with `model` → `simple_conv_v2_ckla` (stft_mag_if). Eval:
`scripts/rps_predictor_vk_eval.py` after registering the checkpoint.

Train: `python train.py experiment=ckla_p1_4s`.

## Conclusion

_Pending run._
