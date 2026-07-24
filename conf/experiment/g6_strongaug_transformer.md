---
experiment: g6_strongaug_transformer
training_config: conf/experiment/g6_strongaug_transformer.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g6_strongaug_transformer`

## Motivation

Every VK-parity arm overfits (train falls, val roughly doubles within ~20
epochs of best), and the current augmentation family is provably weak:
`random_polarity` is an exact no-op for magnitude and IF front-ends,
`random_gain` a log-magnitude offset — only `channel_drop` and the mild
+-12% time-warp change anything the model sees. G5 showed the two-stage
schedule itself is load-bearing (augs-from-sample-0 made best val WORSE,
117.9 vs 63.7), so G6 keeps the E12 schedule (50k unaugmented warmup, then
augs) and replaces the augmented stage's *content* with six strong
noise-chunk transforms (`data_processing/noise_augmentations.py`, applied to
the noise+RPS pair before mixing, probability 0.7): freq_scale (comb scale
with exact label rescale — genuinely new (audio, RPS) pairs),
spectral_recolor, random_reverb, tooth_dropout (label-aware harmonic
masking), spec_mask, floor_inject. Baseline-model arm; time-warp kept as-is.
Success gate: best val/mse below the E12-family baseline (65-79) with a
later best epoch, then the vk_valid_comparison protocol eval
(`scripts/rps_predictor_vk_eval.py`) vs E12-smoothed 2.62 and IF 2.481.

## Conclusion

(pending)
