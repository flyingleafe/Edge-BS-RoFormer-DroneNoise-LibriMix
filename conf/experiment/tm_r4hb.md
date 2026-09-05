---
experiment: tm_r4hb
training_config: conf/experiment/tm_r4hb.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `tm_r4hb`

## Motivation

Regime cell R4 — the comb-only curriculum, stage 2 — for the transformer
trunk at the MAGNITUDE front end. `r4hb_tr` with `params.name` swapped from
`simple_conv_v2_transformer_if` to `simple_conv_v2_transformer`, and the warm
start moved to the matching stage 1: `tm_comb_s1`. An IF-front-end checkpoint
has a different first layer and cannot be loaded into a magnitude trunk, so the
stage-1 swap is forced by the model swap and is not a second free variable.

Everything else is `r4hb_tr` verbatim: the fine-tune stream
`conf/online_mix/hb_m3s2_dload.yaml` (the R2 honest pool — silence arm at weight
0.4, `snr_ref_floor_rms: 0.02`, warm-up stage removed), the loss, the metrics,
the optimizer and the frozen real validation split
`dload:DREGON-LM-V4-michaels-valid-full`.

Submit only after `tm_comb_s1` finishes. Train:
`python train.py experiment=tm_r4hb`.

## Conclusion

Pending.
