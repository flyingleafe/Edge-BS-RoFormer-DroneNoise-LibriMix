# stoch_s1id_trxl — scale the whole trunk, not just the head

## Why this arm exists

The synthetic stream is hard to fit. Under matched conditions
(`scripts/synth_regime_eval.py`, 8 s, no augmentation, 30,120 frames) the
best synthetic-only model scores **8.63** on held-out draws from its own
training distribution and **7.40** on the real split — so there is no
sim-to-real gap and the limit is how well the family can be fitted at all.
That makes capacity the natural suspect.

## Two flaws in the first capacity ladder, both fixed here

**1. It only moved the head.** `tr` / `trmed` / `trbig` went 1.48M ->
2.21M -> 6.24M by growing the temporal head alone; the encoder stayed at 128
channels in every one. The trunk that actually reads the spectrogram never
changed. These arms widen it: 256 and 384 channels, with the frequency pool and
head following.

**2. It was scored on the wrong thing.** Those arms were only ever measured on
the REAL split, and — worse — `conf/data/stoch_s1id.yaml` validates on the
real split, so `monitor: mse` selects each checkpoint by REAL performance.
`trbig`'s best checkpoint is from epoch 1 of 11. A high-capacity model that is
steadily fitting the synthetic data better would have exactly that signature
and would be discarded by real-split selection. The earlier conclusion
"capacity is not the axis" was therefore measured on real-selected checkpoints
scored on real data, and tested neither half of the capacity question.

Both `best` and `last` checkpoints of every arm are scored on synthetic here.

## The ladder

| arm | encoder width | head | params |
|---|---|---|---|
| `stoch_s1id_scv2` (BiGRU) | 128 | — | 1.50M |
| `stoch_s1id_tr` | 128 | 2 x 64 | 1.48M |
| `stoch_s1id_trbig` | 128 | 6 x 256 | 6.24M |
| `stoch_s1id_trxl` | **256** | 6 x 256 | 10.40M |
| `stoch_s1id_trxxl` | **384** | 8 x 512 | 38.19M |

The number to beat is **8.63 all-MAE on held-out synthetic**. If capacity is
the binding constraint, that falls as width rises.

Batch doc: `docs/experiments/stochastic-transfer.md`.
