---
experiment: b2_dcunet_refactored_predrps_hierarchical
training_config: conf/experiment/b2_dcunet_refactored_predrps_hierarchical.yaml
batch: docs/experiments/refactored-decoder-rps-fusion.md
---

# `b2_dcunet_refactored_predrps_hierarchical`

## Motivation

Tests decoder-side RPS fusion and an auxiliary RPS-prediction head (multi-task, telemetry-at-train-only) on the refactored DCUNet/DCCRN.

Full batch context: [Refactored DCUNet/DCCRN — decoder-side RPS fusion + aux head](../../docs/experiments/refactored-decoder-rps-fusion.md).

## Setup

DCUNetRefactored, decoder hierarchical RPS fusion + auxiliary RPS-prediction head (rps_aux_weight=0.1), on DREGON-LM.

Hydra wiring — data `dregon_lm_v1` · model `b2_dcunet_refactored_predrps_hierarchical` · loss `masked_mse_plus_pit_rps_w0p1` · metrics `separation_plus_rps`. Train with `python train.py experiment=b2_dcunet_refactored_predrps_hierarchical`,
evaluate with `python eval.py experiment=b2_dcunet_refactored_predrps_hierarchical`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Refactored DCUNet/DCCRN — decoder-side RPS fusion + aux head](../../docs/experiments/refactored-decoder-rps-fusion.md).
