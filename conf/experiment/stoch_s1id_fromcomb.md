---
experiment: stoch_s1id_fromcomb
training_config: conf/experiment/stoch_s1id_fromcomb.yaml
batch: docs/experiments/stochastic-transfer.md
---

# stoch_s1id_fromcomb — stage 1.5 of the three-stage curriculum

## Motivation

Arm ID's stochastic stream, warm-started from the comb stage-1 checkpoint.
One line differs from `stoch_s1id_scv2`. This run exists to produce the
checkpoint that `r9hb_scv2` fine-tunes from.

## The evidence that argues for a third stage

| regime | arrangement | result |
|---|---|---|
| R4 | comb stage 1 → real stage 2 | **2.67** all-MAE (the record) |
| R6 | stochastic stage 1 → real stage 2 | 3.04 |
| R5 | one stage, synthetic mixed into real | 147.6 val/mse (R4: 17.59) |
| R7 | R5 with the stochastic family in the slot | 92.17 val/mse |

Two clean readings come out of these. **Staging works**: every strong result is
a pure synthetic stage followed by a pure real stage. **Mixing does not**: both
one-stage arms are five to eight times worse than the staged ones on the
monitored metric, and R7 improves on R5 by 38% while staying nowhere near.

And R6 says the choice *between* synthetic families barely matters at stage 1 —
a family that is 12% better synthetic-only produced a 14% worse mixed result.
So the lever is not which family, and not how to blend them into the real data.

## What is left

Keep every stage pure and add one: comb → stochastic → real. The model gets the
coverage of both synthetic families before it ever sees real data, and never
sees the two mixed together. That is the one arrangement the three results
above do not rule out.

Reference rows: `stoch_s1id_scv2` at 7.40 all-MAE (this stream from scratch)
and `m3abl_comb_scv2_s1` (the checkpoint being warm-started from). If stage 1.5
lands below 7.40, the two families' coverage composes and R9 is worth the GPU
hour; if it lands above, comb pre-training actively interferes with the
stochastic stream and R9 should not be run.

Policy: `conf/online_mix/stoch_s1id_dload.yaml`.
Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Synthetic-only transfer with the stochastic noise family](../../docs/experiments/stochastic-transfer.md).
