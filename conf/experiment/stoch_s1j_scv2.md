---
experiment: stoch_s1j_scv2
training_config: conf/experiment/stoch_s1j_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1j_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen validation
split, on the convolutional trunk, with no real noise anywhere in the stream.
The noise pool is the stochastic rotor-noise family
(`conf/online_mix/stoch_s1j_dload.yaml`).

Arm J keeps everything arm H bought and repairs what it broke. Arm H reads real
cruise audio at 2.60 rev/s against the real-trained model's 2.49 — the first
synthetic-only model here to match real training at cruise, against 6.00 for the
best earlier synthetic family. Its stopped-rotor cell is 27.98 and its ramp cell
26.77, and it already carries the level fix, so the level is not the cause.

The cause is the warm-up idle band. Arms F and H widened it to [0.05, 0.65] of
hover to cover the 10 to 30 rev/s a real ramp passes through, which works by
making the drone idle down there — and a rotor at 5 rev/s is 45 dB below cruise,
which is silence carrying a nonzero label. 2.9% of arm H's frames sit under
8 rev/s against 0.4% here. A real drone idles at 0.38 to 0.52 of hover and
passes through the low band on its ramps instead, so the idle band returns to
[0.28, 0.55] and the low-speed coverage comes from a longer spin-up. Measured
against the split's low-regime frames: tenth percentile 12.8 rev/s against the
real 10.1, mean rate of change 7.79 rev/s per second against 7.16, ninetieth
percentile 21.4 against 24.9.

Data `stoch_s1j`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1j_scv2`.

## Conclusion

PENDING — the run has not finished.
