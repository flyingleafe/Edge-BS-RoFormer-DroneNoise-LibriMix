---
experiment: g1_transformer_8s
training_config: conf/experiment/g1_transformer_8s.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g1_transformer_8s`

## Motivation

See `g1_transformer_4s.md` and `docs/experiments/g1-vk-parity.md` — same
phase-B question (does longer native training context close the gap to the
blind-VK bar on the VK-comparison clips?), at the chunk length that exactly
matches the 8 s evaluation clips of the VK-comparison protocol, removing the
train/eval length mismatch entirely.

## Setup

Identical to `e12_real_fullflight_transformer` except:

* `duration_s: 8.0` (policy `conf/online_mix/g1_real_fullflight_8s_dload.yaml`)
* `batch_size: 4` (was 16) to fit 8x-longer chunks on a T4/P100 16 GB.

## Evaluation

`python scripts/rps_predictor_vk_eval.py` after registering the checkpoint —
per-clip pooled PIT-MAE on the VK-comparison clips, with and without
test-time smoothing arms.

## Conclusion

Pending — training not yet run.
