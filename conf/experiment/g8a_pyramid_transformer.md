---
experiment: g8a_pyramid_transformer
training_config: conf/experiment/g8a_pyramid_transformer.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g8a_pyramid_transformer`

## Motivation

Phase G8a of the VK-parity push — C1 of the hierarchical front-end design
(docs/g8-hierarchical-frontend-design.md, grounded in the g7-frontend
literature sweep). The single-window STFT (2048 @16 kHz) is caught in the
classic resolution conundrum: fundamentals (30-120 Hz) need fine FREQUENCY
resolution (7.8 Hz bins are catastrophically coarse there) while high
harmonics (1-2 kHz) need fine TIME resolution with IF supplying sub-bin
frequency. One window cannot serve both; constant-Q allocates exactly
backwards (G2a refuted). The severe overfitting on ~2 drones of real data
means parameter-light structural priors beat learned modules — the pyramid
adds ZERO trainable parameters.

## Setup

Identical to `e12_real_fullflight_transformer` (E12 online-mix stream —
weak-aug default policy per the G6 verdict — 1 s chunks, time-warp +
gain/polarity/channel-drop augs, pit_mse, patience 20, valid =
`dload:DREGON-LM-V4-michaels-valid-full`) except the model:

* `simple_conv_v2_transformer_pyramid` — the unchanged trunk (in_ch=8) on
  `pyramid_if`: four parallel STFTs, n_fft 8192 (30-250 Hz, 1.95 Hz bins),
  4096 (250-1000), 2048 (1000-2000), 1024 (2000-4000), each at hop =
  n_fft/4; per-band log1p-mag + IF-deviation channels (the G2b estimator
  with each band's own n_fft/hop scaling); band-cropped and resampled by
  fixed interpolation onto a geometric log-f axis (30 Hz-4 kHz, 340 rows
  ≈48 bins/octave, density set by the 8192-band's bottom resolution) and
  the hop-512 time grid. Band channels are zero outside their own rows, so
  the 8192-band's 512 ms smear stays confined to its slow k≤2 rows.

## Evaluation

Gate 1 (val): best val/mse below g2_if's 63.7. Gate 2 (protocol):
`python scripts/rps_predictor_vk_eval.py` — pooled per-clip PIT-MAE on the
VK-comparison clips; DREGON below g2_if's 2.481 (raw 3.082), vs the
blind-VK bars (DREGON cruise 0.68-0.74, FLY124 1.027). If passed, proceed
to G8b (+harmonic-aligned fusion) per the design doc rollout.

## Conclusion

Pending — training not yet run.
