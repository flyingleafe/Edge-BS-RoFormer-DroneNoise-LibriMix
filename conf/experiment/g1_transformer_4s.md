---
experiment: g1_transformer_4s
training_config: conf/experiment/g1_transformer_4s.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g1_transformer_4s`

## Motivation

Phase A of the VK-parity push (campaign criterion 2.3) applied test-time
temporal aggregation (sliding-window stitching, moving-average and median
smoothing over 2-20 s) to the E12 real-full-flight checkpoints on the
VK-comparison clips (`scripts/rps_predictor_vk_eval.py`). Smoothing improved
the transformer's DREGON-cruise pooled MAE only from ~3.2 to ~2.7 rev/s —
far from the blind-VK bar of 0.68-0.74. The residual error is systematic
within a window, not zero-mean jitter, so averaging cannot remove it. The
obvious next lever is native temporal context: E12 trained on 1 s chunks
(`duration_s: 1.0` in the online-mix policy) but is evaluated on 8 s clips.

## Setup

Identical to `e12_real_fullflight_transformer` (online-mix DREGON
whole-envelope + FLY125 real noise, LibriSpeech speech, time-warp +
gain/polarity/channel-drop augmentations, valid =
`dload:DREGON-LM-V4-michaels-valid-full`) except:

* `duration_s: 4.0` (policy `conf/online_mix/g1_real_fullflight_4s_dload.yaml`)
* `batch_size: 8` (was 16) to fit 4x-longer chunks on a T4/P100 16 GB.

## Evaluation

`python scripts/rps_predictor_vk_eval.py` after registering the checkpoint —
per-clip pooled PIT-MAE on the VK-comparison clips, with and without
test-time smoothing arms.

## Conclusion

Pending — training not yet run.
