# ladder_r0_scv2 — curriculum rung 0 of 4

Realized comb contrast 12.2 dB, from scratch. Trains the 1.5M SimpleConvV2 trunk
on `conf/online_mix/ladder_r0_dload.yaml` and validates on half the frozen
real split and half synthetic drawn from the same rung.

The ladder axes are `rolloff_p` and `gamma_slope_hz`. The variance knobs are not
the difficulty axis — see `docs/experiments/synthetic-solvability-limits.md`.
