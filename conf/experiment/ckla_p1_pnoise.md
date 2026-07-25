---
experiment: ckla_p1_pnoise
training_config: conf/experiment/ckla_p1_pnoise.yaml
batch: docs/experiments/ckla.md
---

# `ckla_p1_pnoise`

## Motivation

Mechanistic lever derived from the activation analysis
(`docs/experiments/ckla-activation-analysis.md` §A2): with the KLA-default
p_init 0.01 the discretised process noise is ~1e-6, so state precision λ
accumulates for the whole clip (1e3–1e5, never saturating in 8 s) and the
Kalman gain φ/λ collapses to 1e-7..1e-4 — the head degenerates into
clip-scale accumulators with no within-clip tracking bandwidth. That is
the direct suspect for losing to the transformer on drifting DREGON cruise
(2.87 vs 2.481) while winning on near-constant-RPS FLY124 (1.39 vs 2.33).
p_init 1.0 bounds λ* ≈ 1/p̄ ≈ 100, keeping the gain alive (~1e-2). If
FLY124 degrades while DREGON improves, that confirms the
accumulator-prior mechanism either way.

## Setup

Exact clone of `ckla_p1_if` with `model` → `simple_conv_v2_ckla_pnoise`
(p_init 1.0; everything else identical). Post-training: vk_eval + rerun
the λ/gain trajectories (activation analysis §A2) to verify the gain
actually stays alive.

Train: `python train.py experiment=ckla_p1_pnoise`.

## Conclusion

_Pending run._
