---
experiment: a1_edge_bs_rof_fa_rope64
training_config: conf/experiment/a1_edge_bs_rof_fa_rope64.yaml
batch: docs/experiments/paper1-edge-bs-roformer-dn-lm.md
---

# `a1_edge_bs_rof_fa_rope64`

## Motivation

Reproduces the Paper-1 ultra-low-SNR UAV speech-enhancement benchmark on DN-LM; this config is one rung of the ablation ladder / baseline comparison.

Full batch context: [Paper 1 — Edge-BS-RoFormer on DN-LM](../../docs/experiments/paper1-edge-bs-roformer-dn-lm.md).

## Setup

Proposed Edge-BS-RoFormer (Paper 1 headline model): + RoPE head dim 64, on DN-LM. See REPLICATION.md § A1.

Hydra wiring — data `dn_lm` · model `a1_edge_bs_rof_fa_rope64` · loss `masked_mse` · metrics `separation_basic`. Train with `python train.py experiment=a1_edge_bs_rof_fa_rope64`,
evaluate with `python eval.py experiment=a1_edge_bs_rof_fa_rope64`.

## Conclusion

The Paper-1 headline model (RoPE head dim 64): leads the DCUNet/DPTNet/HTDemucs baselines on SI-SDR/PESQ at -15 dB while staying edge-deployable (Jetson RTF ~0.325, 8.5 MB).
