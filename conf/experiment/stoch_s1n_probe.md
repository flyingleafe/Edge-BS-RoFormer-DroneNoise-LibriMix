---
experiment: stoch_s1n_probe
training_config: conf/experiment/stoch_s1n_probe.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1n_probe`

## Motivation

`stoch_s1g_scv2`'s data with the voicing gate on the head. That arm is the
campaign's best synthetic-only model by all-regime mean absolute error (8.08),
and its worst cell is the stopped rotors: 20.27 rev/s, against 2.87 for the
real-trained target.

The failure is the one `GatedProjection` was written for. Fitting arm G's
predictions against the truth on real frames gives `pred = 0.42 * truth + 36.6`
on the ramps and a mean of 26.6 rev/s where the truth is zero — an uncertain
regression head under a squared-error loss answers with the conditional mean,
and the conditional mean of a stream whose rotors are usually turning is a
hover. The gate makes rotor-off a decision instead: the head emits
`speed * sigmoid(gate)`, so an exact zero is reachable by classification while
the speed branch stays a free regression.

The same swap on the real-data regime went the other way — `hb_scv2_mag`
reaches 39.7 against `hb_scv2_mag_nogate`'s 22.1 — which is why the honest-base
campaign kept the plain head. That model's stopped-rotor cell is already 3.36,
so it has no hedging to remove and the gate only costs it. This arm tests
whether the gate earns its place where the hedging is real.

Data `stoch_s1g`, model `hb_scv2_mag` (the `simple_conv_v2` trunk on the
log-magnitude front end, `voicing_gate: true`), loss `pit_mse`, metrics `rps`,
batch 128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1n_gate`.

## Conclusion

PENDING — the run has not finished.


A 55-minute triage copy of its parent experiment on the short GPU partition,
identical except for the run name. It exists so the probe checkpoint can be
loaded by name and scored per regime; the parent carries the motivation.
