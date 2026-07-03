---
experiment: e4_no_aug_baseline
training_config: conf/experiment/e4_no_aug_baseline.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e4_no_aug_baseline`

## Motivation

Builds and applies a learned harmonic drone-noise generator as an online-mixing augmentation source.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

E4 — A/B baseline arm for conf/experiment/e4_generated_noise_augment.yaml: same real DREGON (minus room1) + Michael's FLY125 + LibriSpeech sources, WITHOUT the generated-noise augmentation source. Use this and e4_generated_noise_augment.yaml as a matched pair to isolate the augmentation's effect. See REPLICATION.md § E4.

Hydra wiring — data `online_mix_v4_michaels_no_aug` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=e4_no_aug_baseline`,
evaluate with `python eval.py experiment=e4_no_aug_baseline`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).
