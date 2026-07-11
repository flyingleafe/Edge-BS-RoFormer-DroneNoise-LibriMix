---
experiment: e5_timewarp_uni_gru128
training_config: conf/experiment/e5_timewarp_uni_gru128.yaml
batch: docs/experiments/e5-timewarp.md
---

# `e5_timewarp_uni_gru128`

## Motivation

Tests time-varying time-warp augmentation of the noise+RPS pair for online-mixed RPS prediction: resample the noise recording at a slowly time-varying playback rate `alpha(t) = c + a*sin(2*pi*f*t + phi)` (total `|alpha - 1| <= 0.12`) and transform the label consistently as `r_tilde(t) = alpha(t) * r(tau(t))`, `tau(t) = integral_0^t alpha`. Pure interpolation resampling, applied before mixing with speech.

## Setup

E5 time-warp arm on the C10 winner `simple_conv_v2_uni_gru128`. Data `online_mix_v4_michaels_timewarp` adds `noise_time_warp` (probability 0.5) to the second stage of the V4-michaels online-mix policy, alongside the existing mixture-level augmentations. All other settings mirror the no-warp baseline `c10_uni_gru128_online` for a fair A/B.

Hydra wiring — data `online_mix_v4_michaels_timewarp` · model `simple_conv_v2_uni_gru128` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=e5_timewarp_uni_gru128`, evaluate with `python eval.py experiment=e5_timewarp_uni_gru128`.

## Conclusion

**Tie.** Best val PIT MSE **10.331** (ep24) vs same-framework baseline 10.454
(`e5_baseline_uni_gru128`, ep22) — −1%, within noise. The small causal GRU has
little capacity to overfit the trajectory set, so time-warp buys almost nothing
here (contrast scv2/transformer, which win). The legacy `c10_uni_gru128_online`
7.33 is a *different-pipeline* number and not comparable. See
[E5 batch doc](../../docs/experiments/e5-timewarp.md).
