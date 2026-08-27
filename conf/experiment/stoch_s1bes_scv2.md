---
experiment: stoch_s1bes_scv2
training_config: conf/experiment/stoch_s1bes_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1bes_scv2`

## Motivation

Synthetic-only training whose target is transfer to the real frozen
validation split, on the bidirectional GRU trunk. No real noise appears anywhere
in the stream.

Arm BE plus a sparser comb.

Our engines put power in EVERY harmonic order from 1 to K. Real rotors do not:
blade-passing structure and interference between the blades leave orders weak or
missing, so a real comb is sparser and more irregular than a synthetic one. A
model trained only on complete combs can rely on every order being present,
which is another regularity that does not survive contact with a real
recording.

`harm_dropout_p: [0.0, 0.35]` sends a random share of orders to the floor,
redrawn per clip so no particular order is ever special.

The comparison row is `stoch_s1be_scv2` — the same stream with complete
combs.

Data `stoch_s1bes`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch 128
frames, `samples_per_validation=40000`, validation on the fixed FULL-envelope
real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1bes_scv2`.

## Conclusion

PENDING — the run has not finished.
