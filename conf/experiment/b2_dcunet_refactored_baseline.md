---
experiment: b2_dcunet_refactored_baseline
training_config: conf/experiment/b2_dcunet_refactored_baseline.yaml
batch: docs/experiments/refactored-decoder-rps-fusion.md
---

# `b2_dcunet_refactored_baseline`

## Motivation

Tests decoder-side RPS fusion and an auxiliary RPS-prediction head (multi-task, telemetry-at-train-only) on the refactored DCUNet/DCCRN.

Full batch context: [Refactored DCUNet/DCCRN — decoder-side RPS fusion + aux head](../../docs/experiments/refactored-decoder-rps-fusion.md).

## Setup

DCUNetRefactored baseline, no RPS, on DREGON-LM (docs/dcunet-refactored.md) — reuses conf/model/dcunet.yaml (configs/12a).

Hydra wiring — data `dregon_lm_v1` · model `dcunet` · loss `masked_mse` · metrics `separation_basic`. Train with `python train.py experiment=b2_dcunet_refactored_baseline`,
evaluate with `python eval.py experiment=b2_dcunet_refactored_baseline`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Refactored DCUNet/DCCRN — decoder-side RPS fusion + aux head](../../docs/experiments/refactored-decoder-rps-fusion.md).
