---
experiment: comb_floor_base
training_config: conf/experiment/comb_floor_base.yaml
batch: docs/experiments/e8-static-comb.md
---

# comb_floor_base — stock 2 layers x d_model 64, 4 heads (head 0.141M)

## Motivation

One of three arms that train the transformer to saturation on the analytic
static comb, validated on the same distribution.

| arm | head shape | head params |
|---|---|---|
| `comb_floor_base` | 2 layers x 64, 4 heads | 0.141M |
| `comb_floor_wide` | 2 layers x **128**, 8 heads | 0.479M |
| `comb_floor_deep` | **4** layers x 64, 4 heads | 0.241M |

Width and depth are varied along their own axes, not at matched parameter
count — doubling d_model is quadratic, so `wide` is 3.4x the head and
`deep` is 1.7x. That asymmetry is inherent to the comparison and is reported
with the result.

## Why the static comb

It is the easiest harmonic task available: one amplitude profile held fixed per
clip, comb spacing the only cue. A model that cannot approach zero here is not
limited by the data. The reference is **1.29 all-MAE**, what
`m3abl_comb_scv2_s1`'s last checkpoint reaches — good, and not near zero.

## What each outcome means

- **No floor** — the error keeps falling with epochs. Every synthetic number
  this campaign produced was about training length.
- **A floor that scaling moves** — capacity binds. *Which* scaling moves it
  separates representation per time step (wide) from sequential processing
  (deep).
- **A floor that scaling does not move** — the limit is upstream of the
  sequence model: the STFT front end and the encoder's frequency pooling, which
  collapses 1025 bins to 17 (7.8 -> 470.6 Hz per bin) and is exactly
  permutation-invariant over frequency.

Batch doc: `docs/experiments/stochastic-transfer.md`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [E8 — Static-Comb Noise Model (force harmonic tracking)](../../docs/experiments/e8-static-comb.md).
