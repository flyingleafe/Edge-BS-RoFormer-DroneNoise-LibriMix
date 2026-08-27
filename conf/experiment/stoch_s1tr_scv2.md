---
experiment: stoch_s1tr_scv2
training_config: conf/experiment/stoch_s1tr_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1tr_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen validation
split. One change from arm BE: the four rotors no longer idle in unison.

The trajectory model gates the differential modes (roll, pitch, yaw) to cruise,
on the reasoning that an aircraft holds near-zero attitude control on the
ground. The consequence was never checked: the built stream's ramp frames have
a rotor spread of 0.00 rev/s, with 90.4% of them inside 2 rev/s. Every
synthetic ramp frame the models have ever trained on shows four identical
speeds.

One of the two rigs disagrees, and it is the rig that holds the whole remaining
gap:

| source | rotor spread, median | frames inside 2 rev/s |
|---|---|---|
| Michael's ramp | 9.67 | 3.7% |
| DREGON ramp | 0.03 | 83.0% |
| arm ID stream ramp | 0.00 | 90.4% |

Michael's four motors idle at visibly different speeds. On a real aircraft the
cause is not attitude control but per-motor variation — ESC calibration, motor
and propeller differences, an uneven load — and none of that switches off on the
ground. A model trained only on unison idles learns that a ramp frame means four
equal speeds, which is a configuration Michael's ramp never presents.

This also fits the measured error. Arm ID over-predicts Michael's ramp by +10.02
rev/s on average, over-predicting on 64.6% of frames, while its DREGON ramp is
unbiased at -0.11 — the bias sits on exactly the rig whose idle the stream
cannot represent.

Michael's relative spread is nearly the same in both regimes (9.67/36 at ramp,
17.32/78 at cruise), so ONE per-rotor speed ratio, constant over a clip,
reproduces both. Drawn per clip from zero upward, a clip can be Michael's-like
or DREGON-like. At `[0.0, 0.15]` the stream lands between the two rigs on both
regimes: ramp median 5.76 against 0.03 and 9.67, cruise median 14.96 against
11.77 and 17.32.

Data `stoch_s1tr`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1tr_scv2`.

## Conclusion

PENDING — the run has not finished.
