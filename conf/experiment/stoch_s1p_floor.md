---
experiment: stoch_s1p_floor
training_config: conf/experiment/stoch_s1p_floor.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1p_floor`

## Motivation

The composite of everything the campaign has measured to work, aimed at
per-regime parity with the real-trained target (zero 2.87, low 3.48,
flight 2.49).

Arm G is the best synthetic-only model so far by all-regime mean absolute error
— 8.08, with zero 20.27, low 16.20, flight 4.50. Two changes are added to it,
each for a measured reason, and one change is deliberately left out.

**In: the recording floor.** Arm L's per-regime score separates the two things
it changed. Its stopped-rotor cell is 12.85 against arm G's 20.27, a fall of
37%, which is `floor_static_rel` — a real stopped-rotor clip is not silent, it
carries room tone at about a sixth of a cruise clip's level, and driving it to
digital zero teaches a model that a sixth of cruise means a turning rotor.

**Out: the voicing gate.** It was in this arm and is not any more. The gate is
the designed answer to arm G's hedging — the head emits `speed * sigmoid(gate)`,
so an exact zero is reachable by classification — but `stoch_s1n_gate` is
testing it separately and its first epochs diverge (1462 then 3426, against arm
G's 364 at the same epoch), which matches the gate costing the real-data regime
17 points of validation error. This arm therefore carries only what is measured
to work, and the gate stays on its own row until it earns one.

**Out: the decade of rotor speed.** Arm L also widened the speed range to
[0.25, 2.5] of hover, and that is the half of arm L that failed: cruise went
from 4.50 to 13.14 and the ramps did not move. A model asked to find a comb
anywhere between 20 and 200 rev/s spends capacity on a range the evaluation
never visits. The speed range stays where arm G had it.

Data `stoch_s1p`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`,
batch 128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1p_floor`.

## Conclusion

PENDING — the run has not finished.
