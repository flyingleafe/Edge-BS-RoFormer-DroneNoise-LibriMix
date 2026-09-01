---
experiment: r6c_lr3e3
training_config: conf/experiment/r6c_lr3e3.yaml
batch: docs/experiments/unified-baseline-eval.md
---

# r6c_lr3e3 — overwrite the stochastic initialization harder

## Motivation

The other end of the ladder from `r6a_lr3e4`, and a prediction rather than a
sweep point.

| init | lr 1e-3 | lr 3e-4 |
|---|---|---|
| comb (`r4hb_scv2` / `r4a_lr3e4`) | 17.59 | **15.37** |
| stochastic (`r6hb_scv2` / `r6a_lr3e4`) | **23.78** | 180.39 |

A lower rate preserves more of the initialization. It helps the comb start and
destroys the stochastic one — the two differ by 1.35× at 1e-3 and 11.74× at
3e-4. So the stochastic checkpoint is the worse place to begin a real
fine-tune, despite being the better model on the frozen split: it is strong
where real data is not needed (cruise) and empty where real data carries the
information (the zero cells, which collapse by an order of magnitude when the
fine-tune is prevented from overwriting it).

If preservation is the problem, more overwriting is the fix, and this
checkpoint's ladder should run the opposite way to the comb's.

**Falsifiable.** If 3e-3 is also worse than 1e-3, the stochastic checkpoint is
a poor initialization at every rate, no schedule rescues it, and the
stage-1-family question is closed.

Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Unified baseline evaluation on the frozen validation split](../../docs/experiments/unified-baseline-eval.md).
