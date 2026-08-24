---
experiment: hb_sal_bp
training_config: conf/experiment/hb_sal_bp.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `hb_sal_bp`

## Motivation

The second salience-map row of the unified baseline evaluation (see the batch
doc for the full protocol). The June salience baselines
(`c7_multif0_salience`, `c8_basic_pitch_salience`) trained on fixed
`DREGON-LM-V4` mixtures and scored on `DREGON-LM-V4/valid`, thus their numbers
are not comparable with the current neural rows. This experiment retrains the
same model on the current regime.

Architecture: `basic_pitch_salience` — the Bittner contour branch at native
16 kHz, 264 contour bins, 36 bins per octave, `fmin` 27.5 Hz. Loss and metric:
BCE on an RPS-derived salience map (`salience_bce_basic_pitch`, `blur_bins` 2,
`pos_weight` auto). Monitor: `bce`, minimized. All four are C8's, unchanged.

Data: the `hb` regime (`conf/online_mix/hb_silence_dload.yaml`) — the fs_v2
real full-envelope pool plus a zero-labeled silence arm (16.7% of chunks) and
an SNR reference floor (`snr_ref_floor_rms` 0.02), with the freq-scale,
time-warp, gain and polarity augmentations. This is the stream every HB grid
cell trains on. Validation: the fixed full-envelope real split
`dload:DREGON-LM-V4-michaels-valid-full`.

Differences against C8, for analysis time:

- Training data: the `hb` online stream, not fixed `DREGON-LM-V4/train`
  mixtures. One epoch is 40000 frames (5000 chunks at 8 mics).
- Validation split: `dload:DREGON-LM-V4-michaels-valid-full`, not
  `datasets/DREGON-LM-V4/valid`. BCE values are therefore not comparable
  against the June C8 numbers — only the retrained rows compare.
- Chunk duration: 1.0 s from the online mixer. The V4 clips were longer, thus
  the contour time grid per sample is shorter here.
- Patience: 20, not 15 (the HB convention).

The salience target needs no dataset support: `losses.SalienceRPSBCELoss` and
`metrics.SalienceBCEMetric` both build it from the `rps` entry on every call,
thus the online stream feeds this model with no extra seam.

Train with `python train.py experiment=hb_sal_bp`.

## Conclusion

Pending.
