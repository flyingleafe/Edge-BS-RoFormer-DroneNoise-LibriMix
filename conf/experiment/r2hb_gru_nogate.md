---
experiment: r2hb_gru_nogate
training_config: conf/experiment/r2hb_gru_nogate.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# `r2hb_gru_nogate`

## Motivation

Regime cell R2 of the training-regime taxonomy — the final real-only honest
regime (`conf/online_mix/hb_silence_dload.yaml`: the fs_v2 real pool, the
zero-labeled silence arm at weight 0.4, and the SNR reference floor
`snr_ref_floor_rms: 0.02`) — at the fixed original architecture, the causal
unidirectional-GRU head (`simple_conv_v2_uni_gru128`). The HB grid already
covers this regime, but every HB arm also changes the model: it adds a voicing
gate and pins an explicit front-end. This control changes only the regime, so
the R2 row uses the same plain registry model that R1, R3, R4 and R5 use.
`hb_scv2_mag_nogate` is the matching control for the BiGRU baseline; this file
and `r2hb_tr_nogate` complete the trio. Validation stays the fixed
full-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`. Train:
`python train.py experiment=r2hb_gru_nogate`.

## Conclusion

Pending.
