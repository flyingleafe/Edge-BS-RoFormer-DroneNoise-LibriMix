---
experiment: r2hb_scv2_nomix
training_config: conf/experiment/r2hb_scv2_nomix.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `r2hb_scv2_nomix`

## Motivation

The noise-only arm of the speech A/B on REAL data, at
the BiGRU head (`hb_scv2_mag_nogate`). This file is `hb_scv2_mag_nogate` with the
training stream replaced and nothing else changed.

The stream is `conf/online_mix/hb_m3s2_nomix_dload.yaml` — the honest-base
stage-2 pool with `policy.source_prob: 0.0`. At that setting the sample is the
noise chunk itself: no speech is added, no level is renormalized, and only the
post-mix `random_gain` / `random_polarity` scalar is applied. The speech pool
stays declared so the per-sample random draw order matches the with-speech
twin, and `snr_ref_floor_rms` becomes inert because it reaches the mixer only
through the speech scaling.

The validation split does NOT change: selection stays on the frozen REAL split
`dload:DREGON-LM-V4-michaels-valid-full`, which carries speech. That is
deliberate. The question is whether training speech helps a model that must
work under speech, so both arms must be selected and scored on the same
speech-bearing set; making the validation set speech-free too would answer a
different question and would also make the two arms incomparable to every other
row of the matrix.

Train: `python train.py experiment=r2hb_scv2_nomix`.

## Conclusion

Pending.
