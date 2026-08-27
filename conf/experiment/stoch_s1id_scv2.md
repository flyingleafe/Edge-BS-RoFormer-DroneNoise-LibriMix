---
experiment: stoch_s1id_scv2
training_config: conf/experiment/stoch_s1id_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1id_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

One change from arm X: the flight phase ranges, so the stream contains
sustained low-speed holds.

The remaining gap is concentrated in the ramp cell, and that cell is mostly a
held low-speed comb rather than a sweep — Michael's ramp frames have a speed
derivative with mean 4.25 but median 0.23 rev/s per second. Every arm to this
point cut the warm-up idle roughly fourfold against the generator's calibrated
ranges, so a sustained low-speed hold is nearly absent from the stream.

Arm SI applies the same change to arm S; this arm applies it to arm X, so the
idle is tested on both of the two families that own a regime.

NOTE, found after the run: this arm's phase overrides were written one level
above where the loader reads them (`rps.phases`), so they never applied and the
run used `FlightPhaseRanges`' defaults — `warmup_s (3.0, 25.0)`,
`idle_frac (0.38, 0.52)`. Those defaults are themselves a long idle, so the arm
did test a long warm-up, but not the written values. Arms BE and BES omit the
block entirely to reproduce exactly what this arm ran.

Data `stoch_s1id`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1id_scv2`.

## Conclusion

PENDING — the run has not finished.
