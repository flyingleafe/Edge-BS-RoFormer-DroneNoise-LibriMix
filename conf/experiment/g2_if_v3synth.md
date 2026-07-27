---
experiment: g2_if_v3synth
training_config: conf/experiment/g2_if_v3synth.yaml
batch: docs/experiments/ckla.md
---

# `g2_if_v3synth`

## Motivation

Synthesis-first v3: v2 (and G5/G6) used a hard two-stage schedule — 50k
unaugmented warmup chunks, then all augmentations at once. But freq_scale and
noise_time_warp are not corruption: they synthesize new correctly-labeled
(audio, RPS) pairs, so withholding them during warmup only narrows the data.
v3 keeps them at p=1.0 from sample 0 and ramps only the actual corruption
(the G6 five-transform pack, pre-mix) and the post-mix gain/polarity augs
linearly p 0 -> 0.7 over epochs 5 -> 25 (25000 -> 125000 chunks), replacing
the warmup cliff with a gradual schedule.

## Setup

Clone of the freq-scale v2 arm with the v3 synthesis-first policy
(`conf/online_mix/e12_fullflight_v3_synthfirst_dload.yaml`). Post-training:
the scale-response probe is the success criterion, cruise pools second.

Train: `python train.py experiment=g2_if_v3synth`.

## Conclusion

_Pending run._
