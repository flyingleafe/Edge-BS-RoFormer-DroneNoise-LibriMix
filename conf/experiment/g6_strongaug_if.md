---
experiment: g6_strongaug_if
training_config: conf/experiment/g6_strongaug_if.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g6_strongaug_if`

## Motivation

Same strong-augmentation intervention as `g6_strongaug_transformer`, on the
current best protocol arm (`simple_conv_v2_transformer_if` — g2_if,
DREGON-cruise MAE 2.481, best val/mse 63.7). The weak mixture-level family is
particularly inert for THIS model: polarity is an exact no-op for the IF
front-end and gain is a log-magnitude offset, so overfit onset (~ep 8-18) was
essentially un-regularized. The augmented stage now applies one of six strong
noise-chunk transforms (probability 0.7; `noise_augmentations` — freq_scale
with exact label rescale, spectral_recolor, random_reverb, tooth_dropout,
spec_mask, floor_inject) while keeping the load-bearing 50k unaugmented
warmup (G5: removing it made val worse, 117.9 vs 63.7) and the time-warp.
Success gate: best val/mse < 63.7 with a later best epoch; then the
vk_valid_comparison protocol eval (`scripts/rps_predictor_vk_eval.py`) vs the
2.481 IF bar (VK bars 0.68-0.74 / 1.03).

## Conclusion

(pending)
