---
experiment: g2_if_freqscale_v2
training_config: conf/experiment/g2_if_freqscale_v2.yaml
batch: docs/experiments/ckla.md
---

# `g2_if_freqscale_v2`

## Motivation

Freq-scale v2: the v1 policy (p=0.5, α∈[0.85, 1.18]) left ~63% of all
training chunks unscaled and capped shifts at ±18%; the trained models
measurably kept the amplitude anchor (scale response ~0.1% vs ideal 2%).
v2 scales EVERY post-warmup chunk over [0.7, 1.3], removing the unscaled
majority so the RPS prior stops being a winning strategy.

## Setup

Clone of the freqscale arm with the v2 policy. Post-training: the
scale-response probe is the success criterion, cruise pools second.

Train: `python train.py experiment=g2_if_freqscale_v2`.

## Conclusion

_Pending run._
