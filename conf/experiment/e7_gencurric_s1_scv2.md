---
experiment: e7_gencurric_s1_scv2
training_config: conf/experiment/e7_gencurric_s1_scv2.yaml
batch: docs/experiments/e7-gen-curriculum.md
---

# `e7_gencurric_s1_scv2`

## Motivation

Curriculum **Stage 1**, plain `simple_conv_v2` head (E4 lineage). Train on
**generated noise only** (vicinal-interp E6 generator), validate on the fixed
**real** DREGON+Michael's split — the sim-to-real probe. No augmentation.

## Setup

Identical to [`e7_gencurric_s1_unigru128`](e7_gencurric_s1_unigru128.md) but
model `simple_conv_v2`. Patience 5; `best.ckpt` warm-starts
`e7_gencurric_s2_scv2`. See the
[E7 batch doc](../../docs/experiments/e7-gen-curriculum.md).

Train: `python train.py experiment=e7_gencurric_s1_scv2`.

## Conclusion

Ran 2026-07-11. Best val PIT-MSE 222.8 (R² −10.6) — on the **contaminated** valid (`min_motor_rps=30`); never rescored on the clean split. Verdict revised by the valid-set fix: see the conclusion of [e7-gen-curriculum.md](../../docs/experiments/e7-gen-curriculum.md).

*(Backfilled 2026-08-20.)*
