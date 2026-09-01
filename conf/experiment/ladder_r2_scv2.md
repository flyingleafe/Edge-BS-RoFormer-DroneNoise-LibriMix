---
experiment: ladder_r2_scv2
training_config: conf/experiment/ladder_r2_scv2.yaml
batch: docs/experiments/synthetic-solvability-limits.md
---

# ladder_r2_scv2 — curriculum rung 2 of 4

## Motivation

`gamma_slope_hz` [0.02, 0.15], warm-started from rung 1 (`ladder_r1_scv2`). Trains the 1.5M SimpleConvV2 trunk on
`conf/online_mix/ladder_r2_dload.yaml` and validates on half the frozen real
split, half synthetic from the same rung.

The ladder axis is line width. The variance knobs are not the difficulty axis,
and peak-to-bulk contrast is not a difficulty measure — see
`docs/experiments/synthetic-solvability-limits.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Limits of solving synthetic data](../../docs/experiments/synthetic-solvability-limits.md).
