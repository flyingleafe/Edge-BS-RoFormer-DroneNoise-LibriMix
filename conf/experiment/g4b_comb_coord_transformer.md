---
experiment: g4b_comb_coord_transformer
training_config: conf/experiment/g4b_comb_coord_transformer.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g4b_comb_coord_transformer`

## Motivation

G4a (`g4_comb_transformer`, wandb 2qnc8y8v) refuted the 3-channel comb
front-end at val level: best val/mse 576.5 / mae_frame 15.4 at epoch 4,
flat for 20 epochs — the model predicted a near-constant, never learning to
read WHERE the ridge is. Diagnosis: in f0-space the answer IS the position
along the 361-row axis, but the trunk's frequency pooling averages that
axis away; the spectrogram baseline never needed positional readout because
speed is encoded in translation-covariant texture there.

G4b is the minimal fix: a CoordConv-style 4th channel — each row's f0 value
in rev/s divided by 100, constant over time. With it,
``rps ≈ coord·100 + consensus`` at the comb-score argmax becomes a
near-linear readout for the head.

## Setup

Identical to `g4_comb_transformer` (which mirrors
`e12_real_fullflight_transformer` verbatim: E12 online-mix stream, 1 s
chunks, pit_mse, augs, patience 20) except the model config:

* `simple_conv_v2_transformer_comb` with `coord_channel: true` (explicit) —
  `comb_if` outputs 4 channels × 361 f0 rows; the frontend-aware first conv
  takes in_ch=4. `coord_channel: false` reproduces the G4a model for A/B.

## Evaluation

`python scripts/rps_predictor_vk_eval.py` after registering the checkpoint —
per-clip pooled PIT-MAE on the VK-comparison clips vs E12 (3.186 raw /
2.62 med), g2_if (3.082 raw / 2.481) and the blind-VK bars (DREGON cruise
0.68-0.74, FLY124 1.027). First gate: val mse/mae_frame must beat the G4a
predict-the-mean plateau (576.5 / 15.4) by a wide margin.

## Conclusion

Pending — training not yet run.
