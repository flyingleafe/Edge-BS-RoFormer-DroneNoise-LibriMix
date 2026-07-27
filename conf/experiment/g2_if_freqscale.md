---
experiment: g2_if_freqscale
training_config: conf/experiment/g2_if_freqscale.yaml
batch: docs/experiments/ckla.md
---

# `g2_if_freqscale`

## Motivation

The matched transformer control for the freq-scale training regime. The
CKLA freqscale arm (63.0) beat the CKLA base (85.2) but was compared
against a transformer (63.7) that never saw the augmentation — an
unmatched cell. This arm trains the champion transformer (g2_if recipe)
under the identical solo freq-scale policy, so (a) transformer+freqscale
vs CKLA+freqscale isolates the head under equal spacing pressure, and
(b) transformer+freqscale vs transformer tests whether the augmentation
alone moves the amplitude-anchored transformer (activation analysis §A6:
0.05% response to a 2% frequency shift).

## Setup

Exact clone of `g2_if_transformer` with train policy →
`conf/online_mix/e12_fullflight_freqscale_dload.yaml`. Eval: vk_eval +
the §A6 scale-response probe.

Train: `python train.py experiment=g2_if_freqscale`.

## Conclusion

_Pending run._
