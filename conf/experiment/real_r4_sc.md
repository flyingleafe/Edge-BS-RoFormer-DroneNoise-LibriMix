---
experiment: real_r4_sc
training_config: conf/experiment/real_r4_sc.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r4_sc`

## Motivation

Rung 4 of the reality ladder — the full honest-base pool
`conf/online_mix/hb_m3s2_dload.yaml` (two rigs, 8 microphones, plus
`freq_scale` and the time warp) — on SimpleConv (`simple_conv`), the small
June trunk.

The other three trunks already have a row on this pool, so their R3-to-R4 step
is readable. SimpleConv does not, and its ladder would otherwise stop at rung
3. This row is optional in the sense that it adds no new axis; it only
completes the column.

This file is `hb_scv2_mag_nogate` with the model swapped to `simple_conv`, the
stream set to the R4 pool and the monitor changed to `mae_frame`.

Train: `python train.py experiment=real_r4_sc`.

## Conclusion

Pending.
