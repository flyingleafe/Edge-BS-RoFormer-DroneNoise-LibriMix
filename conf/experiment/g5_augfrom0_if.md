---
experiment: g5_augfrom0_if
training_config: conf/experiment/g5_augfrom0_if.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g5_augfrom0_if`

## Motivation

Every VK-parity arm to date (g1/g2/g3/g4 + E12 baseline) shows severe
overfitting: train loss keeps falling while val roughly doubles within
~20 epochs of the best (e.g. g2_if val 63.7 -> 150.2, g1_8s 68.8 -> 140.4).
The E12 online-mix policy's two-stage curriculum keeps the first 50k
samples UNAUGMENTED — that is epochs 1-10 at samples_per_validation 5000 —
while best-val epochs land at 8-18, so the gain/polarity/channel-drop augs
and the noise time-warp barely engaged before early stopping in every
prior arm. This arm (simple_conv_v2_transformer_if) removes the warmup stage: augmentation +
time-warp active from sample 0. Success gate: best val/mse below the
warmup counterpart (baseline family 65-79 / IF 63.7) with a later best
epoch; then the vk_valid_comparison protocol eval vs the VK bars
(DREGON 0.68-0.74, FLY124 1.03).

## Conclusion

Refuted. Training lqrhvu8z: best val/mse 117.9 (mae 7.58) at epoch 5
vs the two-stage counterpart's 63.7 at ep 17; val exploded to 592 by ep 25.
Removing the warmup makes both the optimum and the overfit trajectory worse.
