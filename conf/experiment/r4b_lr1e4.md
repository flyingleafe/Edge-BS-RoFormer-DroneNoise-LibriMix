# r4b_lr1e4 — the third point on the winning ladder

`r4a_lr3e4` took the record by lowering the fine-tune rate on the comb
initialization. That ladder has two points; this is the third.

| arm | lr | val PIT-MSE | all-MAE |
|---|---|---|---|
| `r4hb_scv2` | 1e-3 | 17.59 | 2.67 |
| `r4a_lr3e4` | 3e-4 | **15.37** | **2.61** |
| `r4b_lr1e4` | 1e-4 | this arm | |

Two outcomes, both useful. It extends the record, in which case the rate should
be pushed further. Or it locates the floor — and if it also stops early on the
unchanged `patience: 20`, as the stochastic init's low-rate arms did (20 and 30
epochs against 85), then the schedule rather than the rate is what to change,
and the ladder's later points have been measuring patience rather than
learning rate.

Batch doc: `docs/experiments/stochastic-transfer.md`.
