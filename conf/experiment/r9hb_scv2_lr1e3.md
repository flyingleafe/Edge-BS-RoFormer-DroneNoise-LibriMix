---
experiment: r9hb_scv2_lr1e3
training_config: conf/experiment/r9hb_scv2_lr1e3.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# r9hb_scv2 — the three-stage curriculum, and a withdrawn gate

## Motivation

Comb → stochastic → real. This is the arm that uses the stochastic family in a
configuration the evidence now supports, and it exists because a gate I wrote
earlier turned out to rest on a premise that later measurement destroyed.

## The gate, and why it is withdrawn

Stage 1.5 (`stoch_s1id_fromcomb`) was gated on its synthetic-only score:
below 7.40 → build R9; above → do not. It scored 12.21, and R9 was not built.

The gate assumed synthetic-only score predicts initialization quality. The
learning-rate ladder measured the opposite:

| checkpoint | synthetic-only all-MAE | as an init (best val PIT-MSE) |
|---|---|---|
| comb (`m3abl_comb_scv2_s1`) | 8.30 | **15.37** |
| stochastic (`stoch_s1id_scv2`) | **7.40** | 23.78 |

The better synthetic-only model is the worse initialization, at every rate
tested, and at lr 3e-4 the two differ by 11.74×. Under that finding stage 1.5's
12.21 is not evidence against it — the quantity the gate measured does not
measure what the gate claimed.

The gate is withdrawn explicitly, on the record. It was sound when written; it
is not sound now.

## Two rates, because the interaction is large

Stage 1.5's checkpoint is comb-*derived* (comb's optimum is 3e-4) but
stochastic-*trained* (stochastic's optimum is 1e-3). Which optimum it inherits
is the unknown, so both are run: `r9hb_scv2_lr1e3` and `r9hb_scv2_lr3e4`.

## What the outcomes mean

- Below the 2.60–2.67 cluster: the stochastic family does contribute in the
  mixed regime, and it needed to sit between the comb and the real data rather
  than replace either.
- Inside that cluster: it contributes nothing measurable but costs nothing.
- Above it: comb pre-training genuinely does interfere, the original gate
  reached the right answer for the wrong reason, and the stochastic family has
  no place in this recipe.

Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Unified baseline evaluation on the frozen validation split](../../docs/experiments/unified-baseline-eval.md).
