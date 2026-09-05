---
experiment: r2hb_gru_nomix
training_config: conf/experiment/r2hb_gru_nomix.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `r2hb_gru_nomix`

## Motivation

The noise-only arm of the speech A/B on REAL data, at
the causal GRU head (`simple_conv_v2_uni_gru128`). This file is `r2hb_gru_nogate` with the
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

**Read the A/B against `real_r4_gru`, not against the source row.** The
source row's stream is `conf/online_mix/hb_silence_dload.yaml`, which is
`hb_m3s2_dload.yaml` PLUS a 50k-sample unaugmented warm-up stage (that is the
only difference between the two files). This row's stream is the nomix twin of
`hb_m3s2_dload.yaml`, so it differs from its source in TWO ways: no speech and
no warm-up. The exact with-speech twin would be `gru` on
`hb_m3s2_dload.yaml` with no warm start, which the matrix does not contain yet
for this trunk — `real_r4_sc` is the only rung-4 row of that shape, and it is
SimpleConv.

Train: `python train.py experiment=r2hb_gru_nomix`.

## Conclusion

Pending.
