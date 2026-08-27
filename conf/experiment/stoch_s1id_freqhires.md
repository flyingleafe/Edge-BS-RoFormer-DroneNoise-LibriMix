---
experiment: stoch_s1id_freqhires
training_config: conf/experiment/stoch_s1id_freqhires.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1id_freqhires`

## Motivation

An architecture arm, not a data arm. The stream is arm ID's, unchanged, so the
comparison row `stoch_s1id_scv2` differs from this one by architecture alone.

`SimpleConvV2` aggregates frequency with a `FrequencyAttentionPool` that ends in
`out.mean(dim=1)` over the frequency axis, and its attention over that axis
carries no positional encoding. The pool and everything after it are therefore
EXACTLY permutation invariant over frequency: shuffle the 17 surviving bands and
the predicted RPS does not move, verified to 1e-7. The encoder is convolutional
and so weight-shared over frequency, which leaves absolute frequency position
with essentially no route into the head.

That is a strange property for a task whose answer IS an absolute frequency. It
is also not the only loss on the way. The six encoder blocks each stride
frequency by two, taking the axis from 1025 bins to 17, from 7.8 Hz/bin to
470.6 Hz/bin. Rotor speeds of 20 to 90 rev/s put the comb's spacing at 2.6 to
11.5 bins at the front end and 0.3 to 1.4 bins by the third block — below the
frequency axis' own Nyquist. The spacing that carries the answer is aliased away
early, and what survives does so encoded in channels rather than in position.

Every architecture arm run so far has changed the temporal head or the front
end. None has touched the frequency aggregation.

This arm adds the resolution to arm FreqCat's aggregation. The last two encoder
blocks stop striding frequency, so 65 bins reach the head instead of 17 — 123
Hz per bin rather than 471. The comb's spacing is still below that axis'
Nyquist, so this does not restore the spacing; what it restores is a
four-times-finer picture of WHERE the energy sits, which is the part the pooling
threw away.

Costs 2.1 M parameters over the baseline and about 28% more forward time, so it
is the expensive end of the three. Read against `stoch_s1id_freqcat`: same
aggregation, four times the resolution.

Data `stoch_s1id`, model `simple_conv_v2_freqhires`, loss `pit_mse`, metrics `rps`,
batch 128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1id_freqhires`.

## Conclusion

PENDING — the run has not finished.
