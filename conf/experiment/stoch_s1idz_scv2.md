---
experiment: stoch_s1idz_scv2
training_config: conf/experiment/stoch_s1idz_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1idz_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

Arm ID's warm-up idle and arm Z's fixed level law together: a trajectory-level
fix and a level-domain fix, both of which helped on their own.

As with arm IDRP, the combination is the test of whether the two are
independent. Unlike arm IDRP, the two fixes here act on different domains
entirely — when the rotors are, and how loud they are — so if any pair should
add, it is this one.

Data `stoch_s1idz`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1idz_scv2`.

## Conclusion

PENDING — the run has not finished.
