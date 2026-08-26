---
experiment: stoch_s1v_ground
training_config: conf/experiment/stoch_s1v_ground.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1v_ground`

## Motivation

Both synthetic families, with zero-labelled windows that are both plentiful and
drawn from the drone families themselves.

**What arm T measured, and why it failed.** `scripts/zero_probe.py` showed the
stochastic models reading their own combless floor as a 39 to 46 rev/s rotor
while reading the silence pool correctly, so they had learned "stopped" as the
silence pool's texture rather than as the absence of a comb. Arm T removed the
silence pool to force the real cue, and its stopped-rotor cell got **worse** —
20.27 to 37.61.

The reason is coverage, and it had already been measured: the stochastic pool's
own ground phases supply 3.3% of frames and the silence pool supplied the other
16%. Removing it cut zero coverage sixfold, which costs more than the texture
shortcut ever did. The probe's mechanism is real; deleting the pool is the wrong
fix for it.

**What this arm does instead.** It keeps the coverage and changes where it comes
from. `pre_ground_s` and `post_ground_s` rise from half a second to three, up to
twenty, so every cached flight of BOTH families spends long spells with its
rotors stopped and the drone families themselves produce plentiful zero windows.
The silence pool stays at half its former weight, because a real recording does
contain room tone that no rotor model generates.

Measured on the stream: zero 8.9%, low 27.2%, flight 63.9%, against the frozen
split's 12.7 / 13.5 / 73.8.

It also keeps what arm S proved — both families in one stream gives the best ramp
cell the campaign has produced, 8.94 rev/s against a previous best of 16.20.

Data `stoch_s1v`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1v_ground`.

## Conclusion

PENDING — the run has not finished.
