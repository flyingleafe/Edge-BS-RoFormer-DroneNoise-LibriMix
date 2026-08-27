---
experiment: stoch_s1rp_scv2
training_config: conf/experiment/stoch_s1rp_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1rp_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

One change from arm X: `rps.mode_scales`, which sets how far the four rotors
separate from one another.

A quadrotor's four rotors do not wander independently. Roll, pitch and yaw each
move a diagonal PAIR against the other pair, so the four speeds fall into two
groups rather than four free trajectories. The synthetic generator drives the
three control modes at equal strength, which spreads the four rotors more evenly
than any real airframe does. `mode_scales` reweights the modes so the pair
structure survives into the rendered speeds.

This is a geometry-level fix, in contrast to the trajectory-level fixes (arm ID's
idle, arm Z's level law). Running it separately is what tells the two classes
apart.

Data `stoch_s1rp`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1rp_scv2`.

## Conclusion

PENDING — the run has not finished.
