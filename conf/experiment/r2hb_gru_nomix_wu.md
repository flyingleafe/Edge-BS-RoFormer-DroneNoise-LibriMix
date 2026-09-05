---
experiment: r2hb_gru_nomix_wu
training_config: conf/experiment/r2hb_gru_nomix_wu.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `r2hb_gru_nomix_wu`

## Motivation

The noise-only arm of the speech A/B on real data for the causal GRU head
(`simple_conv_v2_uni_gru128`), with the 50k-sample unaugmented warm-up stage of the R2 rows.
The stream is `conf/online_mix/hb_silence_nomix_dload.yaml`, which is
`hb_silence_dload.yaml` with `source_prob: 0.0` in both stages. This row is
the exact twin of the R2 row of the same trunk. It replaces `r2hb_gru_nomix`
(the warm-up-free pool) as the A/B arm: the warm-up-free schedule on its own
moved the with-speech scv2 row from 2.77 to 4.59 (`real_r4_scv2`), so the
two warm-up-free rows do not isolate the speech.

Train: `python train.py experiment=r2hb_gru_nomix_wu`.

## Conclusion

Pending.
