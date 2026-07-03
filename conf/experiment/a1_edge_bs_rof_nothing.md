---
experiment: a1_edge_bs_rof_nothing
training_config: conf/experiment/a1_edge_bs_rof_nothing.yaml
batch: docs/experiments/paper1-edge-bs-roformer-dn-lm.md
---

# `a1_edge_bs_rof_nothing`

## Motivation

Reproduces the Paper-1 ultra-low-SNR UAV speech-enhancement benchmark on DN-LM; this config is one rung of the ablation ladder / baseline comparison.

Full batch context: [Paper 1 — Edge-BS-RoFormer on DN-LM](../../docs/experiments/paper1-edge-bs-roformer-dn-lm.md).

## Setup

Edge-BS-RoFormer ablation floor (no flash-attn, no RoPE) on DN-LM (Paper 1). See REPLICATION.md § A1.

Hydra wiring — data `dn_lm` · model `a1_edge_bs_rof_nothing` · loss `masked_mse` · metrics `separation_basic`. Train with `python train.py experiment=a1_edge_bs_rof_nothing`,
evaluate with `python eval.py experiment=a1_edge_bs_rof_nothing`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Paper 1 — Edge-BS-RoFormer on DN-LM](../../docs/experiments/paper1-edge-bs-roformer-dn-lm.md).
