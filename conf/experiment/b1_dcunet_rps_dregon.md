---
experiment: b1_dcunet_rps_dregon
training_config: conf/experiment/b1_dcunet_rps_dregon.yaml
batch: docs/experiments/rps-conditioned-se-dregon.md
---

# `b1_dcunet_rps_dregon`

## Motivation

Tests whether conditioning an enhancement backbone on oracle rotor speeds (RPS) beats the telemetry-blind baseline on DREGON-LM — the Paper-2 telemetry-given upper bound.

Full batch context: [RPS-conditioned speech enhancement on DREGON-LM](../../docs/experiments/rps-conditioned-se-dregon.md).

## Setup

DCUNet + encoder bottleneck RPS fusion (5-layer variant) on DREGON-LM.

Hydra wiring — data `dregon_lm_v1` · model `b1_dcunet_rps_dregon` · loss `masked_mse` · metrics `separation_basic`. Train with `python train.py experiment=b1_dcunet_rps_dregon`,
evaluate with `python eval.py experiment=b1_dcunet_rps_dregon`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [RPS-conditioned speech enhancement on DREGON-LM](../../docs/experiments/rps-conditioned-se-dregon.md).
