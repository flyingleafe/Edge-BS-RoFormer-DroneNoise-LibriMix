---
experiment: b1_dccrn_rps_predrps_dregon
training_config: conf/experiment/b1_dccrn_rps_predrps_dregon.yaml
batch: docs/experiments/rps-conditioned-se-dregon.md
---

# `b1_dccrn_rps_predrps_dregon`

## Motivation

Tests whether conditioning an enhancement backbone on oracle rotor speeds (RPS) beats the telemetry-blind baseline on DREGON-LM — the Paper-2 telemetry-given upper bound.

Full batch context: [RPS-conditioned speech enhancement on DREGON-LM](../../docs/experiments/rps-conditioned-se-dregon.md).

## Setup

DCCRN + encoder RPS conditioning + auxiliary RPS-prediction head (rps_aux_weight=2.0) on DREGON-LM.

Hydra wiring — data `dregon_lm_v1` · model `b1_dccrn_rps_predrps_dregon` · loss `masked_mse_plus_pit_rps_w2` · metrics `separation_plus_rps`. Train with `python train.py experiment=b1_dccrn_rps_predrps_dregon`,
evaluate with `python eval.py experiment=b1_dccrn_rps_predrps_dregon`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [RPS-conditioned speech enhancement on DREGON-LM](../../docs/experiments/rps-conditioned-se-dregon.md).
