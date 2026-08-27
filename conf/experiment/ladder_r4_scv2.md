# ladder_r4_scv2 — curriculum rung 4 of 4

`gamma_slope_hz` [0.05, 0.8], warm-started from rung 3 (`ladder_r3_scv2`). Trains the 1.5M SimpleConvV2 trunk on
`conf/online_mix/ladder_r4_dload.yaml` and validates on half the frozen real
split, half synthetic from the same rung.

The ladder axis is line width. The variance knobs are not the difficulty axis,
and peak-to-bulk contrast is not a difficulty measure — see
`docs/experiments/synthetic-solvability-limits.md`.
