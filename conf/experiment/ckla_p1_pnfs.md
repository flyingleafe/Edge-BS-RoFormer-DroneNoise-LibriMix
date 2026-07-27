---
experiment: ckla_p1_pnfs
training_config: conf/experiment/ckla_p1_pnfs.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p1_pnfs`

## Motivation

Combination of the two mechanistic levers, each individually confirmed on
the full-envelope valid (base `ckla_p1_if` 85.2 → freqscale 63.0 → pnoise
**44.8**): p_init 1.0 restores within-clip tracking bandwidth (activation
analysis §A2 pathology) and the solo freq-scale augmentation forces
comb-spacing reading (§A6 pathology). The mechanisms are independent —
state dynamics vs training distribution — so their combination is the
natural candidate for the campaign's best arm: a filter that can track
drift, trained so spacing is the only reliable cue.

## Setup

`ckla_p1_if` recipe with `model` → `simple_conv_v2_ckla_pnoise` AND train
policy → `conf/online_mix/e12_fullflight_freqscale_dload.yaml`. Eval:
vk_eval cruise pools + the §A2 λ-gain and §A6 scale-response probes.

Train: `python train.py experiment=ckla_p1_pnfs`.

## Conclusion

_Pending run._
