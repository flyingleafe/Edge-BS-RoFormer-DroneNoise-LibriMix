# Cross-drone RPS generalization — adding Michael's FLY125 to training

**Status:** done | **Dates:** 2026-06 | **Depends on:** [simpleconv-rps-architecture-search.md](simpleconv-rps-architecture-search.md), [channel-generalization-pit-loss.md](channel-generalization-pit-loss.md)

## Motivation

RPS-prediction models trained only on the DREGON quadrotor did not transfer to
a *different* drone: evaluated on Michael's FLY124 recordings, a DREGON-only
`simple_conv_v2` generalised poorly. If telemetry-free RPS estimation is to be
a cross-domain building block (constraint C4/C3 in `GOALS.md`), the model must
work across airframes. This experiment tests the simplest fix — put a second
drone (Michael's **FLY125**) into the training mix and measure transfer to the
held-out **FLY124**.

## Experiments

- `c11_dregon_fly125_retrain` — retrain `simple_conv_v2` on DREGON `in_flight_noise`
  (minus room1) **+ Michael's FLY125**, 8-channel source with channel 0 selected,
  PIT loss; validate on the fixed DREGON-LM-V4-michaels split and evaluate
  cross-drone on FLY124. (Run `955yy1wv`; the `fly125_simpleconvv2_eval` script
  evaluates both the DREGON-only and the DREGON+FLY125 checkpoints on both
  datasets.)

## Results

Adding FLY125 to training closed the cross-drone gap almost entirely: on
FLY124, PIT RMSE dropped **7.96 → 1.63** and R² rose **0.52 → 0.96**. The cost
was a small regression on the in-domain DREGON-LM-V4 validation set, attributed
to an early-stop at epoch ~20. (FLY124 slices must be filtered to stable
in-flight frames — per-frame mean RPS > 45 — since idle/landing segments inflate
the error; see the note in [simpleconv-rps-architecture-search.md](simpleconv-rps-architecture-search.md).)

## Conclusion

A single additional airframe in the training mix is enough to make the
SimpleConvV2 RPS predictor generalise across drones (FLY124 R² 0.96),
supporting the "RPS as a cross-domain latent" thesis candidate (C5). The
in-domain regression suggests the retrain was under-trained (early stop) rather
than a genuine trade-off — a longer schedule is the obvious next step. This
motivates extending the training pool toward the non-drone rotating-machinery
target (MIMII, C3) next.
