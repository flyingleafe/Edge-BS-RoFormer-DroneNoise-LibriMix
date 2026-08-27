---
experiment: stoch_s1idrp_scv2
training_config: conf/experiment/stoch_s1idrp_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1idrp_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

Arm ID's warm-up idle and arm RP's rotor-pair structure together: the two
measured mismatches that each helped on their own.

The point of the combination is to find out whether the two fixes address the
same deficiency or different ones. Two fixes that repair one underlying problem
do not add; two that repair different problems should.

Data `stoch_s1idrp`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1idrp_scv2`.

## Conclusion

PENDING — the run has not finished.
