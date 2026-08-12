---
experiment: gen_m2_refined_perrotor
training_config: conf/experiment/gen_m2_refined_perrotor.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `gen_m2_refined_perrotor`

## Motivation

`gen_m1_refined` + per-rotor sub-embeddings (z_r = z_drone + dz_r, shared
across drones). The perrotor-dynamics campaign exonerated the deltas: they
neither impede comb formation nor damage rendering, and with comb-aware
checkpoint selection the superset wins. With two drones AND per-rotor deltas
the codebook has its full intended structure. Per-epoch checkpoints ON.

## Conclusion

Pending.
