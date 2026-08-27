# ladder_r3_scv2 — curriculum rung 3 of 4

Realized comb contrast 9.6 dB, warm-started from rung 2 (`ladder_r2_scv2`). Trains the 1.5M SimpleConvV2 trunk
on `conf/online_mix/ladder_r3_dload.yaml` and validates on half the frozen
real split and half synthetic drawn from the same rung.

The ladder axes are `rolloff_p` and `gamma_slope_hz`. The variance knobs are not
the difficulty axis — see `docs/experiments/synthetic-solvability-limits.md`.
