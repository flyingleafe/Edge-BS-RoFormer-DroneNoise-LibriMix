---
experiment: e5_timewarp_scv2
training_config: conf/experiment/e5_timewarp_scv2.yaml
batch: docs/experiments/e5-timewarp.md
---

# `e5_timewarp_scv2`

## Motivation

Tests time-varying time-warp augmentation of the noise+RPS pair for online-mixed RPS prediction: resample the noise recording at a slowly time-varying playback rate `alpha(t) = c + a*sin(2*pi*f*t + phi)` (total `|alpha - 1| <= 0.12`) and transform the label consistently as `r_tilde(t) = alpha(t) * r(tau(t))`, `tau(t) = integral_0^t alpha`. Pure interpolation resampling, applied before mixing with speech.

## Setup

E5 time-warp arm on the base `simple_conv_v2`. Data `online_mix_v4_michaels_timewarp` adds `noise_time_warp` (probability 0.5) to the second stage of the V4-michaels online-mix policy, alongside the existing mixture-level augmentations. All other settings mirror the no-warp baseline (`c10_arch_sweep_online` with `model=simple_conv_v2`) for a fair A/B.

Hydra wiring — data `online_mix_v4_michaels_timewarp` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=e5_timewarp_scv2`, evaluate with `python eval.py experiment=e5_timewarp_scv2`.

## Conclusion

**Win: −9%.** Best val PIT MSE **8.849** (ep31) vs baseline 9.707 (`e5_baseline_scv2`).
Run crashed at ep66 but the best-val was reached and plateaued well before, so
the number stands. See [E5 batch doc](../../docs/experiments/e5-timewarp.md).
