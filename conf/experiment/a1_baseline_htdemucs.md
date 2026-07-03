---
experiment: a1_baseline_htdemucs
training_config: conf/experiment/a1_baseline_htdemucs.yaml
batch: docs/experiments/paper1-edge-bs-roformer-dn-lm.md
---

# `a1_baseline_htdemucs`

## Motivation

Reproduces the Paper-1 ultra-low-SNR UAV speech-enhancement benchmark on DN-LM; this config is one rung of the ablation ladder / baseline comparison.

Full batch context: [Paper 1 — Edge-BS-RoFormer on DN-LM](../../docs/experiments/paper1-edge-bs-roformer-dn-lm.md).

## Setup

HTDemucs baseline (Paper 1 comparison point), on DN-LM. See REPLICATION.md § A1.

Hydra wiring — data `dn_lm` · model `a1_baseline_htdemucs` · loss `masked_mse` · metrics `separation_basic`. Train with `python train.py experiment=a1_baseline_htdemucs`,
evaluate with `python eval.py experiment=a1_baseline_htdemucs`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Paper 1 — Edge-BS-RoFormer on DN-LM](../../docs/experiments/paper1-edge-bs-roformer-dn-lm.md).
