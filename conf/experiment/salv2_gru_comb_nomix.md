---
experiment: salv2_gru_comb_nomix
training_config: conf/experiment/salv2_gru_comb_nomix.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `salv2_gru_comb_nomix`

## Motivation

One cell of the SALV2 synthetic grid, extended to a second trunk:
the causal unidirectional-GRU head (`simple_conv_v2_uni_gru128`) on the analytic STATIC COMB,
WITHOUT speech.

This file is `salv2_scv2_comb_nomix` with the model swapped and nothing else changed
— same data config, loss, metrics, optimizer, batch size, epoch size and
monitor. The published grid measured only SimpleConvV2 among the regressors;
the paper regime matrix reads the synthetic column and the real column on the
same three trunks, so the two remaining trunks are filled in here.

The monitor is `mae_frame` (PIT per-frame absolute error in rev/s), not the
training objective, exactly as in the source row. Train:
`python train.py experiment=salv2_gru_comb_nomix`.

## Conclusion

Pending.
