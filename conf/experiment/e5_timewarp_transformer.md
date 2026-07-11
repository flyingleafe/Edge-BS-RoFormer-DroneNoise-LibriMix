---
experiment: e5_timewarp_transformer
training_config: conf/experiment/e5_timewarp_transformer.yaml
batch: docs/experiments/e5-timewarp.md
---

# `e5_timewarp_transformer`

## Motivation

Tests time-varying time-warp augmentation of the noise+RPS pair for online-mixed RPS prediction: resample the noise recording at a slowly time-varying playback rate `alpha(t) = c + a*sin(2*pi*f*t + phi)` (total `|alpha - 1| <= 0.12`) and transform the label consistently as `r_tilde(t) = alpha(t) * r(tau(t))`, `tau(t) = integral_0^t alpha`. Pure interpolation resampling, applied before mixing with speech.

## Setup

E5 time-warp arm on `simple_conv_v2_transformer`. Data `online_mix_v4_michaels_timewarp` adds `noise_time_warp` (probability 0.5) to the second stage of the V4-michaels online-mix policy, alongside the existing mixture-level augmentations. All other settings mirror the no-warp baseline (`c10_arch_sweep_online` with `model=simple_conv_v2_transformer`) for a fair A/B.

Hydra wiring — data `online_mix_v4_michaels_timewarp` · model `simple_conv_v2_transformer` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=e5_timewarp_transformer`, evaluate with `python eval.py experiment=e5_timewarp_transformer`.

## Conclusion

**Big win: −26%, best model overall.** Best val PIT MSE **8.737** (RMSE 2.70,
ep80) vs same-framework baseline 11.759 (`e5_baseline_transformer`, ep79). The
Transformer baseline is the *worst* of the three heads despite the most capacity
(overfits the limited trajectory set); time-warp rescues it to the best result to
date. Both runs hit gpushort walltime at ep115, but best-vals (ep79–80) had
plateaued before. The legacy 8.46 is a different-pipeline number, not comparable.
See [E5 batch doc](../../docs/experiments/e5-timewarp.md).
