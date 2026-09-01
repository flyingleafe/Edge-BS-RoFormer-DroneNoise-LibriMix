---
experiment: r6a_lr3e4
training_config: conf/experiment/r6a_lr3e4.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# r6a_lr3e4 — learning-rate ladder on the mixed curriculum

## Motivation

See the header of `conf/experiment/r6a_lr3e4.yaml` for the arm's role.

## The paradox this set attacks

| arm | stage-1 checkpoint | synthetic-only all-MAE | mixed all-MAE |
|---|---|---|---|
| `r4hb_scv2` | comb | 8.30 | **2.67** |
| `r6hb_scv2` | stochastic | **7.40** | 3.04 |

The better starting point produced the worse finish. Every other explanation
this campaign tested has failed, and one mechanism predicts exactly this shape:
lr 1e-3 is large enough to destroy the initialization in the first epochs, and
a BETTER initialization has strictly more to lose.

## The three arms

| arm | init | lr |
|---|---|---|
| `r6a_lr3e4` | stochastic | 3e-4 |
| `r6b_lr1e4` | stochastic | 1e-4 |
| `r4a_lr3e4` | comb (the record holder) | 3e-4 |

`r4a_lr3e4` is what makes the result attributable. Lowering the rate could
simply help every arm in this family, in which case the stage-1 choice still
does not matter and the campaign's central negative result stands. The
mechanism is confirmed only if the lower rate helps the stochastic init MORE
than it helps the comb init.

Everything else — the pure real stage-2 stream (`hb_m3s2_dload.yaml`), the
trunk, the loss, the patience, the validation split — is unchanged from
`r4hb_scv2`, so within each pair exactly one line differs.

Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Unified baseline evaluation on the frozen validation split](../../docs/experiments/unified-baseline-eval.md).
