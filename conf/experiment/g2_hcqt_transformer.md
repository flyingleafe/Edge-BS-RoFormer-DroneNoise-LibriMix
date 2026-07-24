---
experiment: g2_hcqt_transformer
training_config: conf/experiment/g2_hcqt_transformer.yaml
batch: docs/experiments/g1-vk-parity.md
---

# `g2_hcqt_transformer`

## Motivation

Phase G2 of the VK-parity push (campaign criterion 2.3). Phases A
(test-time smoothing) and B (native 4 s / 8 s context) both failed to close
the neural predictor's systematic within-window error on DREGON cruise
(best ~2.6-2.9 rev/s pooled PIT-MAE vs the blind-VK bar 0.68-0.74). The
remaining hypothesis is the front-end: a log-magnitude STFT gives the trunk
neither harmonically aligned evidence aggregation nor sub-bin frequency
precision. This arm tests the *harmonic alignment* half: the HCQT stacks
log-spaced CQT copies at fmin*h, so all harmonics of a candidate f0 fall on
the same frequency bin across input channels — the comb structure VK
exploits explicitly.

## Setup

Identical to `e12_real_fullflight_transformer` (online-mix DREGON
whole-envelope + FLY125 real noise, LibriSpeech speech, 1 s chunks,
time-warp + gain/polarity/channel-drop augmentations, pit_mse, patience 20,
valid = `dload:DREGON-LM-V4-michaels-valid-full`) except the model:

* `simple_conv_v2_transformer_hcqt` — the same SimpleConvV2Transformer
  trunk on an HCQT front-end (nnAudio backend, native 16 kHz, fmin=32.7,
  6 octaves, 60 bins/octave, harmonics auto = [1,2,3] under Nyquist,
  mag + dphase → 6 input channels; hop 256, time-interpolated onto the
  [16000, 512] output grid).

## Evaluation

`python scripts/rps_predictor_vk_eval.py` after registering the checkpoint —
per-clip pooled PIT-MAE on the VK-comparison clips, with and without
test-time smoothing arms; compare against the E12 baseline (3.19 none /
2.62 med) and the blind-VK bars (DREGON cruise 0.68-0.74, FLY124 1.027).

## Conclusion

Pending — training not yet run.
