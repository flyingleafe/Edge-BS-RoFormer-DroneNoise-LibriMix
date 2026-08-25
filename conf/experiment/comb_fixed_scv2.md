---
experiment: comb_fixed_scv2
training_config: conf/experiment/comb_fixed_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `comb_fixed_scv2`

## Motivation

The control of the stochastic-transfer campaign. Arms A through E change two
things at once: the noise family (the stochastic model in place of the analytic
static comb) and a list of fixes the diagnostics asked for — level invariance,
a scattered speed prior, a realistic per-rotor spread, a room, a coloration.
This row carries the fixes on the OLD family, so a transfer result can be
attributed.

`conf/online_mix/comb_fixed_dload.yaml` is `stoch_s1e_dload.yaml` with
`kind: stochastic` replaced by `kind: static_comb`. Its unfixed ancestor is
`m3abl_comb_scv2_s1`, which reaches 336.8 validation PIT-MSE on the frozen
split, so the pair reads as the fixes' contribution and the pair with
`stoch_s1e_scv2` reads as the family's.

Data `comb_fixed`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=comb_fixed_scv2`.

## Conclusion

PENDING — the run has not finished.
