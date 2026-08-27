---
experiment: stoch_s1si_scv2
training_config: conf/experiment/stoch_s1si_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1si_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

One change from arm S: the flight phase ranges.

Arm S owns the Michael's ramp cell, which is the larger half of the remaining
in-domain gap, and that cell is mostly a HELD low-speed comb rather than a
sweep. Michael's carries 1071 of the split's 1253 ramp frames, and their speed
derivative has mean 4.25 but median 0.23 rev/s per second — half of them are a
stationary hold at low RPM, the warm-up idle.

Every arm to this point overrode the phases to `warmup_s: [0.5, 6.0]`,
shortening the idle roughly fourfold against ranges the generator's own
docstring calls calibrated to these recordings. The resulting stream's ramp
frames are neither rig: median 6.84, against Michael's 0.23 and DREGON's 24.72.
Restoring a long idle gives median 1.47 with a p90 of 16.55 — about half the
ramp frames near-steady, quick sweeps still present.

The comparison row is `stoch_s1s_scv2`.

Data `stoch_s1si`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1si_scv2`.

## Conclusion

PENDING — the run has not finished.
