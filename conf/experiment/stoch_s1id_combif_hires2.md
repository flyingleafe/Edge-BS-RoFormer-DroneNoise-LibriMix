---
experiment: stoch_s1id_combif_hires2
training_config: conf/experiment/stoch_s1id_combif_hires2.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1id_combif_hires2`

## Motivation

Identical to `stoch_s1id_combif_hires`, renamed only because a cancelled first
attempt left a non-empty `results/` directory that the training loop refuses to
write into. No checkpoint survived that attempt, so nothing is lost.

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

This arm pairs the ramp-capable front end with a trunk that keeps its axis. The
pairing is not an arbitrary combination of two winners: in f0-space the answer
is a POSITION along the row axis, so a front end that puts f0 on an axis and a
trunk that averages that axis away cancel each other out. `simple_conv_v2_freqhires`
stops striding frequency in the last two blocks and replaces the pooling with a
linear map over the flattened (channel, frequency) pairs, so 29 rows reach the
head rather than 8.

Read against `stoch_s1id_combif`: same front end, and the only difference is
whether the trunk can see where along the f0 axis the evidence sits.

Data `stoch_s1id`, model `simple_conv_v2_freqhires_combif_hires`, loss `pit_mse`, metrics `rps`, batch 128
frames, validation on `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1id_combif_hires2`.

## Conclusion

PENDING — the run has not finished.
