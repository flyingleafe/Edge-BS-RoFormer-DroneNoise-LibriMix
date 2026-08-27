---
experiment: stoch_s1be2_scv2
training_config: conf/experiment/stoch_s1be2_scv2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1be2_scv2`

## Motivation

One change from arm BES: the band taper narrows from 0.30 to 0.10.

Arms BE and BES removed the comb's speed-linear top edge, and it worked where
it was aimed — 100% of ramp frames carried the artifact, and arm ID's Michael's
ramp went from 20.71 to 14.72 (BE) and 10.83 (BES). Cruise paid for it:
Michael's cruise went 3.71 to 5.24 and 7.70, and neither arm set a new best in
any cell.

The cause is the taper WIDTH, not the idea. At 0.30 the fade begins at 5600 Hz,
so a cruise clip at 78 rev/s has every harmonic above k of about 72 out of 100
attenuated — and high-order lines are what carry cruise precision. The artifact
itself cannot have been helping on real validation, because real audio does not
contain it; so the cruise loss is the taper removing real signal, not the fix
removing a shortcut.

At 0.10 the fade begins at 7200 Hz. A cruise clip keeps its harmonics out to k
of about 90 of 100, while a slow frame still has its comb reach the band edge
(the raised order cap covers down to 20 rev/s) and fade smoothly rather than
stopping dead at a frequency proportional to its own speed.

The comparison row is `stoch_s1bes_scv2` — same stream, same sparse comb, and
the taper width is the only difference.

Data `stoch_s1be2`, model `simple_conv_v2`, loss `pit_mse`, metrics `rps`, batch
128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1be2_scv2`.

## Conclusion

PENDING — the run has not finished.
