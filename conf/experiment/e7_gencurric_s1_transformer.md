---
experiment: e7_gencurric_s1_transformer
training_config: conf/experiment/e7_gencurric_s1_transformer.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s1_transformer`

## Motivation

Curriculum **Stage 1**, `simple_conv_v2_transformer` head (the E5 time-warp
best-overall arch). Train on **generated noise only** (vicinal-interp E6
generator), validate on the fixed **real** DREGON+Michael's split. No
augmentation.

## Setup

Identical to [`e7_gencurric_s1_unigru128`](e7_gencurric_s1_unigru128.md) but
model `simple_conv_v2_transformer`. Patience 5; `best.ckpt` warm-starts
`e7_gencurric_s2_transformer`. See the
[E7 batch doc](../../docs/experiments/e7-gen-curriculum.md).

Train: `python train.py experiment=e7_gencurric_s1_transformer`.

## Conclusion

Ran 2026-07-11. Best val PIT-MSE 225.3 (R² −10.1) — on the **contaminated** valid (`min_motor_rps=30`); never rescored on the clean split. Verdict revised by the valid-set fix: see the conclusion of [e7-gen-curriculum.md](../../docs/experiments/e7-gen-curriculum.md).

*(Backfilled 2026-08-20.)*
