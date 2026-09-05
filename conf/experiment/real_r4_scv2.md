---
experiment: real_r4_scv2
training_config: conf/experiment/real_r4_scv2.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r4_scv2`

## Motivation

Rung 4 of the real-data ladder for the BiGRU head (`hb_scv2_mag_nogate`): DREGON room2 + FLY125,
all 8 microphones, the silence arm, speech at -30..0 dB, and the
label-transforming augmentations (frequency scaling, time warp) on top of
gain and polarity. The stream is `conf/online_mix/hb_m3s2_dload.yaml`: no
warm-up stage, no warm start.

This row is the with-speech twin of `r2hb_scv2_nomix` (one difference: the
speech-bearing stream in place of the nomix policy). It is also the row that
rungs 1-3 (`real_r1_scv2`, `real_r2_scv2`, `real_r3_scv2`) step up to, because
their policies are this stream with the label-transforming blocks removed.
The older R2 row of the same trunk trained on `hb_silence_dload.yaml`, which
adds a 50k-sample unaugmented warm-up stage. That row stays as a cross-check.

Train: `python train.py experiment=real_r4_scv2`.

## Conclusion

Pending.
