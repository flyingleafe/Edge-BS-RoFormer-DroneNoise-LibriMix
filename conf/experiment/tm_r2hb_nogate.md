---
experiment: tm_r2hb_nogate
training_config: conf/experiment/tm_r2hb_nogate.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `tm_r2hb_nogate`

## Motivation

Regime cell R2 — real-only honest base — for the transformer trunk at the
MAGNITUDE front end. `r2hb_tr_nogate` with `params.name` swapped from
`simple_conv_v2_transformer_if` to `simple_conv_v2_transformer`; the stream
(`conf/online_mix/hb_silence_dload.yaml`), loss, metrics, optimizer, batch size,
epoch budget and frozen real validation split are that file verbatim.

The paper drops the instantaneous-frequency front end, so every transformer row
of the regime matrix is re-measured on the plain log-STFT trunk. There is no
warm start and one stage. Train: `python train.py experiment=tm_r2hb_nogate`.

## Conclusion

Pending.
