---
experiment: stoch_s1id_freqpos
training_config: conf/experiment/stoch_s1id_freqpos.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `stoch_s1id_freqpos`

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

This arm is the control that separates the two losses. It adds a learned
positional embedding over the frequency axis and changes nothing else: same
shapes, same pool, 2 k more parameters, no extra arithmetic. If position
blindness alone is what costs accuracy, this arm recovers it. If the arm moves
nothing, the loss is the resolution rather than the position, and the next two
arms are the ones that matter.

Data `stoch_s1id`, model `simple_conv_v2_freqpos`, loss `pit_mse`, metrics `rps`,
batch 128 frames, `samples_per_validation=40000`, validation on the fixed
FULL-envelope real split `dload:DREGON-LM-V4-michaels-valid-full`.
Train: `python train.py experiment=stoch_s1id_freqpos`.

## Conclusion

PENDING — the run has not finished.
