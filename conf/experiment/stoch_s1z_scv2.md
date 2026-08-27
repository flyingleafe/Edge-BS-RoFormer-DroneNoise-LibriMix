---
experiment: stoch_s1z_scv2
training_config: conf/experiment/stoch_s1z_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1z_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

The far end of arm Y. Arm H redraws a reference level per window over 21.9 dB
and arm Y narrows that to 9.5 dB; this arm removes the draw entirely, leaving
`level_mode: flight`'s speed law as the only thing that sets a window's
loudness.

If level scatter is what keeps a synthetic-only model from reading slow rotors,
this arm is where that shows most clearly. If instead the scatter is the
augmentation that makes the model level-invariant, this arm is where it breaks —
which is the point of running both ends rather than one.

Data `stoch_s1z`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1z_scv2`.

## Conclusion

PENDING — the run has not finished.
