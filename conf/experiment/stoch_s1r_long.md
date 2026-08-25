---
experiment: stoch_s1r_long
training_config: conf/experiment/stoch_s1r_long.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1r_long`

## Motivation

Arm P's pool, trained at the length it is scored at.

Every arm of this campaign trains on one-second chunks — about 31 short-time
Fourier frames — and is scored on eight-second clips, about 250 frames. The
temporal head is recurrent, so the length is not a detail. Scoring
`stoch_s1g_scv2`'s own checkpoint at three slice lengths, with no retraining, on
channel 0 of the frozen split:

| scored in | all-MAE | zero | low | flight |
|---|---|---|---|---|
| 8 s, how it is evaluated | 9.22 | 26.55 | 18.56 | 4.53 |
| 2 s | 9.04 | 21.46 | 10.99 | 6.56 |
| 1 s, how it was trained | 9.98 | 19.18 | **8.30** | 8.70 |

The ramp cell more than halves and the stopped-rotor cell improves, from a
change in nothing but the length of the window the model answers over. Cruise
moves the other way, because a long window genuinely helps a steady speed.

So the two failing cells are partly a train-and-evaluate mismatch and not only a
noise model. A recurrent state fitted on 31 frames does not carry 250; on a ramp
the rest of the clip drags the answer, while at cruise the same averaging is
what makes the answer good. Slicing the evaluation would trade one cell for
another. Training at the evaluation length should not have to.

`samples_per_validation` falls from 40000 to 5000 in step with the eightfold
longer chunk, so an epoch still covers the same amount of audio.

Data `stoch_s1r` (arm P's pool at `duration_s: 8.0`), model `simple_conv_v2`,
loss `pit_mse`, metrics `rps`, batch 128, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1r_long`.

## Conclusion

PENDING — the run has not finished.
