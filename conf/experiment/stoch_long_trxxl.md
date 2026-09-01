---
experiment: stoch_long_trxxl
training_config: conf/experiment/stoch_long_trxxl.yaml
batch: docs/experiments/stochastic-transfer.md
---

# stoch_long_trxxl — trained to convergence, selected on synthetic

## Motivation

## The two things every earlier arm got wrong

**Selection.** Validation ran on the real frozen split, so `monitor: mse` kept
the best-on-real epoch. On `stoch_s1id_scv2` that checkpoint scores 8.63
all-MAE on held-out synthetic while the run's own final weights score 3.70 —
early stopping was throwing away the fit as it appeared.

**Duration.** Nothing converged. `trbig`'s best is epoch 1 of 11; `trxl`
reached 20 epochs, `trxxl` 34, the BiGRU's best was epoch 5. Patience 20 on a
real-split metric that was actively getting worse halted every run early.

This arm fixes both: a finite synthetic validation set at a held-out seed, and
patience 200 / epochs 2000.

## What it tests

Whether the stochastic family is hard to fit because it is genuinely hard, or
because no model was ever trained on it properly. And, with the `trxxl`
sibling, whether capacity helps once training is allowed to finish — the
earlier ladder said no, but every point on it was cut off.

| reference | synthetic all-MAE |
|---|---|
| `stoch_s1id_scv2` `last` (1.5M BiGRU, cut off) | 3.70 |
| `stoch_s1id_trxl` `last` (10.4M, cut off) | 6.62 |

Both are floors from truncated runs, not converged results.

Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Synthetic-only transfer with the stochastic noise family](../../docs/experiments/stochastic-transfer.md).
