---
experiment: ladder_r0_deep
training_config: conf/experiment/ladder_r0_deep.yaml
batch: docs/experiments/synthetic-solvability-limits.md
---

# ladder_r0_deep — curriculum rung 0 on the deep temporal head

## Motivation

Rung 0 (`gamma_slope_hz` 0, sharp lines) trained with
`simple_conv_v2_transformer_deep` instead of the reference trunk, so the rung
measures what the data allows rather than what the reference trunk allows.

Depth was the only scaling axis that helped on the static comb — see
`docs/experiments/synthetic-solvability-limits.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Limits of solving synthetic data](../../docs/experiments/synthetic-solvability-limits.md).
