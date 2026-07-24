---
experiment: g8a2_pyramid_dense_transformer
training_config: conf/experiment/g8a2_pyramid_dense_transformer.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g8a2_pyramid_dense_transformer`

## Motivation

G8a (`g8a_pyramid_transformer`, wandb 1m7nguxw) failed at val with a
violently unstable curve: best val/mse 142.5 (mae 6.67) at epoch 10, then
142→619→652→180→659. Diagnosis — channel sparsity, a design flaw of the
8-channel layout: the four bands PARTITION the 340 log-f rows, so at every
row exactly one band's (mag, IF) pair is nonzero and the other 6 channels
are exactly zero, with hard edges at the band boundaries. The first conv
therefore saw mostly-zero inputs whose support pattern, not content,
dominated its statistics.

G8a2 is the minimal fix: `collapse_bands: true` SUMS the masked per-band
tensors instead of concatenating — 2 dense channels (mag + IF over the full
log-f axis). Because the row masks partition the axis (coverage == 1,
asserted in tests), the sum is exact — no double counting; the per-band
resolution allocation (the actual C1 hypothesis) is unchanged.

## Setup

Identical to `g8a_pyramid_transformer` (which mirrors
`e12_real_fullflight_transformer` verbatim: E12 online-mix stream, weak-aug
default policy, 1 s chunks, pit_mse, patience 20) except the model config:

* `simple_conv_v2_transformer_pyramid` with `collapse_bands: true`
  (explicit) — `pyramid_if` outputs (B, 2, 340, T); in_ch=2 via the
  frontend-aware first conv. `collapse_bands: false` reproduces the dead
  G8a model.

## Evaluation

Gate 1 (val): stable curve and best val/mse below g2_if's 63.7 (G8a's
142.5 is the floor to beat trivially). Gate 2 (protocol):
`python scripts/rps_predictor_vk_eval.py` — pooled per-clip PIT-MAE;
DREGON below g2_if's 2.481, vs the blind-VK bars (0.68-0.74 / FLY124
1.027). If passed, proceed to G8b (+harmonic fusion) per the design doc.

## Conclusion

Pending — training not yet run.
