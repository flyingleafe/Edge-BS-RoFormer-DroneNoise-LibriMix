---
experiment: ebsrof_rps_freqscale
training_config: conf/experiment/ebsrof_rps_freqscale.yaml
batch: docs/experiments/ckla.md
---

# `ebsrof_rps_freqscale`

## Motivation

Freq-scale-regime cell of the Edge-BS-RoFormer-RPS comparison: identical
solo spacing-pressure augmentation as `ckla_p1_freqscale` and
`g2_if_freqscale`, so all three heads (CKLA / transformer / roformer) are
compared under identical training conditions. If rotary embeddings do help
harmonic-line tracking, the spacing-forcing regime is where it should show.

## Setup

`ebsrof_rps_e12` with train policy →
`conf/online_mix/e12_fullflight_freqscale_dload.yaml`.

Train: `python train.py experiment=ebsrof_rps_freqscale`.

## Conclusion

_Pending run._
