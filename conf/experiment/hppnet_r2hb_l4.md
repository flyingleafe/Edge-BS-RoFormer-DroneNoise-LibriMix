---
experiment: hppnet_r2hb_l4
training_config: conf/experiment/hppnet_r2hb_l4.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hppnet_r2hb_l4`

## Motivation

Regime cell R2 — real-only honest base — for the salience port
HPPNet (`hppnet_rps_l4`). This file is `hppnet_r4_l4` with two changes.

First, the warm start is removed: without a `checkpoint:` the row trains from
scratch on the real pool `conf/online_mix/hb_m3s2_dload.yaml`, which is what
makes it the R2 cell of the salience column instead of the stage 2 of a
comb-only curriculum. It also removes the dependency on a synthetic stage-1
job, so the row can be submitted straight away.

Second, the monitor becomes `rps_mae` instead of `bce`. Both are reported by
`conf/metrics/salience_layers_r150.yaml`, but only `rps_mae` is the quantity
the matrix compares rows on, and cross-entropy was measured to select badly on
these arms — `val/bce` rose while `val/rps_mae` fell, and the bce-selected
`hf0_r4_l4` checkpoint came out unusable.

Everything else — the loss, the metrics, the batch size of 16 frames, the epoch
budget and the frozen real validation split — is the source file verbatim.
Train: `python train.py experiment=hppnet_r2hb_l4`.

## Conclusion

Pending.
