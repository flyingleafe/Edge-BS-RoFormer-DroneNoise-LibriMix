---
experiment: r8hb_scv2
training_config: conf/experiment/r8hb_scv2.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# r8hb_scv2 — the record holder, with the stochastic family in its fine-tune pool

## Motivation

## Where this sits

| arm | stage-1 checkpoint | stage-2 pool | all-MAE |
|---|---|---|---|
| `r4hb_scv2` | comb | real + silence | **2.67** |
| `r6hb_scv2` | **stochastic** | real + silence | 3.04 |
| `r7hb_scv2` | none (one stage) | real + silence + **stochastic** + comb | running |
| `r8hb_scv2` | comb | real + silence + **stochastic** | this arm |

R8 differs from the record holder in exactly one line: the stage-2 stream.

## Why R6's result points here

R6 gave the fine-tune a much better starting point — `stoch_s1id_scv2` reads
7.40 all-MAE on the frozen split before seeing any real data, against the comb
checkpoint's 8.30, and wins three of six cells outright — and came out WORSE
after the fine-tune (3.04 against 2.67). Only one cell improved:

| cell | target | R6 |
|---|---|---|
| DREGON zero | **2.24** | 2.45 |
| DREGON ramp | 7.21 | **7.13** |
| DREGON cruise | **2.98** | 3.38 |
| Michael's zero | **4.48** | 6.32 |
| Michael's ramp | **2.85** | 3.01 |
| Michael's cruise | **1.55** | 1.83 |

So synthetic-only score does not predict post-fine-tune score, and the
stochastic family's value is not as an initialization the real data overwrites.
The natural inference is that it has to stay in the pool to be worth anything —
which is what R7 and R8 test, R7 without a curriculum and R8 on top of the
configuration that actually holds the record.

## The pool

Real 58.8% : silence 11.8% : stochastic 29.4% (weights 2.0 : 0.4 : 1.0). Two
things are deliberately NOT copied from arm ID's policy:

- its `static_comb` source, because the comb is what the stage-1 checkpoint was
  already trained on — including it would confound initialization with pool.
- its `silence` arm at weight 0.2, because this pool carries the honest silence
  arm at 0.4 and both would put 0.6 of silence in and change the R2 regime this
  whole family holds fixed.

Stream check: PASS (determinism 4/4 bit-identical).
Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Unified baseline evaluation on the frozen validation split](../../docs/experiments/unified-baseline-eval.md).
