---
experiment: tm_comb_s1
training_config: conf/experiment/tm_comb_s1.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `tm_comb_s1`

## Motivation

Stage 1 of the R4 (comb-only curriculum) column for the transformer trunk,
grown at the MAGNITUDE front end. The as-run transformer rows all use
`simple_conv_v2_transformer_if`, the instantaneous-frequency variant; the paper
drops that front end, so the whole transformer column has to be rebuilt from a
magnitude stage 1. This row is `m3abl_comb_transformer_s1` with `params.name`
swapped and nothing else touched — same policy
(`conf/online_mix/m3abl_comb_s1_dload.yaml` through data `m3abl_comb_s1`), same
loss, metrics, optimizer, batch size and epoch budget.

It is a dependency of `tm_r4hb`, which warm-starts from its `best.ckpt`, so it
must run first. Train: `python train.py experiment=tm_comb_s1`.

## Conclusion

Pending.
