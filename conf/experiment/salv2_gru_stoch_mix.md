---
experiment: salv2_gru_stoch_mix
training_config: conf/experiment/salv2_gru_stoch_mix.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `salv2_gru_stoch_mix`

## Motivation

One cell of the SALV2 synthetic grid, extended to a second trunk:
the causal unidirectional-GRU head (`simple_conv_v2_uni_gru128`) on the STOCHASTIC rotor-noise family,
WITH speech mixed in at -30 to 0 dB SNR.

This file is `salv2_scv2_stoch_mix` with the model swapped and nothing else changed
— same data config, loss, metrics, optimizer, batch size, epoch size and
monitor. The published grid measured only SimpleConvV2 among the regressors;
the paper regime matrix reads the synthetic column and the real column on the
same three trunks, so the two remaining trunks are filled in here.

The monitor is `mae_frame` (PIT per-frame absolute error in rev/s), not the
training objective, exactly as in the source row. Train:
`python train.py experiment=salv2_gru_stoch_mix`.

## Conclusion

Pending.

## Recipe note (2026-09-05)

This cell runs with `amp: false` and `grad_clip: 0.5` (the E5 uni-GRU recipe). Under the defaults the causal GRU dropped non-finite batches from epoch 1 on the stochastic stream, and the first run died at epoch 22 with NaN predictions in the Hungarian matcher. The comb cells of this trunk and every transformer cell did not need it.
