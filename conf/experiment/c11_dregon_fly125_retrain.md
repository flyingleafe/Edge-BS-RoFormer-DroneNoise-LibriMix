---
experiment: c11_dregon_fly125_retrain
training_config: conf/experiment/c11_dregon_fly125_retrain.yaml
batch: docs/experiments/cross-drone-generalization-fly125.md
---

# `c11_dregon_fly125_retrain`

## Motivation

Tests cross-drone RPS transfer by adding Michael's FLY125 to training and evaluating on the held-out FLY124.

Full batch context: [Cross-drone RPS generalization (FLY125)](../../docs/experiments/cross-drone-generalization-fly125.md).

## Setup

C11 step 2 — retrain simple_conv_v2 (8ch source, channel:0 selected) from scratch on DREGON + Michael's FLY125, closing the FLY124 cross-drone gap (PIT RMSE 7.96 -> 1.63 Hz, R² median 0.52 -> 0.96 vs. the DREGON-only checkpoint). In-domain DREGON-LM-V4 regresses slightly (1.62 -> 2.77 Hz), attributed to an early-stop artifact (best-by-combined-val at epoch 20), not a capacity tradeoff — see REPLICATION.md § C11. Composes identically to conf/experiment/c10_arch_sweep_offline.yaml's default (same dataset, same default model) but kept as a separately-named experiment for its own wandb run / results dir, matching the historical record. C11 step 1 (zero-shot cross-drone eval of the DREGON-only checkpoint on FLY124, no retraining) is an eval.py-only exercise with no dedicated conf/data/fly124_eval.yaml yet — see REPLICATION.md "needs follow-up wiring" for what's missing.

Hydra wiring — data `dregon_lm_v4_michaels` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c11_dregon_fly125_retrain`,
evaluate with `python eval.py experiment=c11_dregon_fly125_retrain`.

## Conclusion

Closed the cross-drone gap: FLY124 PIT RMSE **7.96 -> 1.63**, R^2 **0.52 -> 0.96** (small in-domain V4 regression from an epoch-20 early stop).

## Replication note (2026-07-07)

Directionally replicated via omnirun on apocrita-long + dload data (W&B `nn17a6u0`, commit bf0ebc6): post-hoc `eval.py` on `dload:DREGON-LM-V4-valid` gave PIT RMSE 3.18 / R² 0.65 (channel 0) vs. the reference 2.77 — the in-domain regression vs. the 1.62 V4 baseline is confirmed. Caveats for future replicators:

- **This yaml is not a faithful hyperparameter record of the historical run**: it sets `epochs: 50` / `batch_size: 32`, while the historical W&B run used epochs=200, batch_size=16 (early-stopped around epoch 20).
- The reference 2.77 came from a post-hoc eval script that is not in the repo — its provenance is the report, not a re-runnable command.
- The historical run validated per-mic-flattened (legacy loop); the replica evaluated channel 0 only, and the ch0-trained replica does not generalize across mics (flattened eval RMSE 9.38).

The michaels valid-split crash this replication exposed is fixed (`flatten_channels`, commit ffba378). See REPLICATION.md § C11.
