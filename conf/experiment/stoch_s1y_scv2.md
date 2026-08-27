---
experiment: stoch_s1y_scv2
training_config: conf/experiment/stoch_s1y_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1y_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

Arm H is the campaign's cruise winner and cannot read a slow rotor. This arm
changes one thing about it: the width of the per-window reference level.

Arm H redraws that level log-uniformly over 21.9 dB, so two windows at the same
rotor speed can arrive 21.9 dB apart. A real flight couples level to speed far
more tightly than that. The intent was to keep `level_mode: flight`'s speed law
as the dominant term while leaving some scatter around it, at 9.5 dB.

An earlier version of this note claimed the synthetic stream deletes the
level-to-speed correlation outright, measured at +0.09 against real audio's
+0.55. That was a 39-chunk sample and it did not survive: at 491 chunks arm H
reads +0.25, inside the range real recordings span. The narrowing is therefore a
plausible move against a real difference in SCATTER, not a repair of a missing
cue.

Data `stoch_s1y`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1y_scv2`.

## Conclusion

PENDING — the run has not finished.
