---
experiment: hf0_r4_l4_v2
training_config: conf/experiment/hf0_r4_l4_v2.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hf0_r4_l4_v2`

## Motivation

`hf0_r4_l4` re-run with `optim.monitor: rps_mae` in place of `bce`. One line
changes; the warm start
(`r2://ml-data/artifacts/hf0_comb_l4/checkpoints/best.ckpt`), the stream
`conf/online_mix/hb_m3s2_dload.yaml`, the loss, the metrics, the batch size and
the frozen real validation split are the original file verbatim.

The original row selected on cross-entropy and kept a broken checkpoint.
Cross-entropy is unbounded and punishes a confident near-miss that a bounded
argmax error barely notices, so `val/bce` can rise while `val/rps_mae` falls —
which is what happened. `rps_mae` is also the one number the regime matrix
compares rows on. Train: `python train.py experiment=hf0_r4_l4_v2`.

## Conclusion

Pending.
