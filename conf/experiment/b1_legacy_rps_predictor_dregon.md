---
experiment: b1_legacy_rps_predictor_dregon
training_config: conf/experiment/b1_legacy_rps_predictor_dregon.yaml
batch: docs/experiments/rps-conditioned-se-dregon.md
---

# `b1_legacy_rps_predictor_dregon`

## Motivation

Tests whether conditioning an enhancement backbone on oracle rotor speeds (RPS) beats the telemetry-blind baseline on DREGON-LM — the Paper-2 telemetry-given upper bound.

Full batch context: [RPS-conditioned speech enhancement on DREGON-LM](../../docs/experiments/rps-conditioned-se-dregon.md).

## Setup

Standalone RPS predictor via the legacy models.registry.LEGACY_MODEL_BUILDERS rps_predictor model_type (configs/11c) on DREGON-LM.

Hydra wiring — data `dregon_lm_v1` · model `b1_legacy_rps_predictor_dregon` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=b1_legacy_rps_predictor_dregon`,
evaluate with `python eval.py experiment=b1_legacy_rps_predictor_dregon`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [RPS-conditioned speech enhancement on DREGON-LM](../../docs/experiments/rps-conditioned-se-dregon.md).
