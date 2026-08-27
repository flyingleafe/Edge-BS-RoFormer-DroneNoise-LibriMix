---
experiment: stoch_s1id_combif
training_config: conf/experiment/stoch_s1id_combif.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1id_combif`

## Motivation

A front-end arm aimed at the one cell that holds the remaining gap. The stream
is arm ID's, unchanged, so this row differs from `stoch_s1id_scv2` by the front
end (and, for the hires arm, the frequency aggregation) alone.

Both ramp cells are the binding constraint — DREGON 8.79 and Michael's 8.14
against a 2.67 target, while three of the six rig-by-regime cells already meet
it. Two measurements say the ramp cell may be a front-end limit rather than a
data limit.

First, resolvability. At `n_fft=2048` a Hann window's main lobe is 31.25 Hz
wide, and the spacing between adjacent harmonics is the rotor speed itself. So
below 31 rev/s adjacent harmonics are not separable at any frequency, and the
ramp regime is defined as 1 to 45 rev/s. Across most of that cell the model
never sees a comb, only a smeared envelope.

Second, representability. The `comb_if` front end puts f0 on its own axis —
one row per candidate — which is the right shape for this task, but it searches
only 30 to 120 rev/s. Most ramp frames fall below its lowest candidate and
cannot be represented at all. `comb_if_ramp` starts the search at 5 rev/s,
giving 461 rows instead of 361.

This arm changes the front end only, keeping the standard pooled trunk. It is
the control: it says whether letting the front end REPRESENT a slow rotor is
worth anything on its own, before any change to how the trunk reads it.

`comb_if`'s own docstring predicts it will not be enough — "in f0-space the
answer IS the position along the row axis, but the trunk's frequency pooling
averages that axis away". That pooling is exactly permutation invariant over
frequency, verified to 1e-7, and `coord_channel` exists to work around it.
Costs about twice the baseline's forward time.

Data `stoch_s1id`, model `simple_conv_v2_combif`, loss `pit_mse`, metrics `rps`, batch 128
frames, validation on `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1id_combif`.

## Conclusion

PENDING — the run has not finished.
