---
experiment: b2_dcunet_refactored_decoder_hierarchical
training_config: conf/experiment/b2_dcunet_refactored_decoder_hierarchical.yaml
batch: docs/experiments/refactored-decoder-rps-fusion.md
---

# `b2_dcunet_refactored_decoder_hierarchical`

## Motivation

Tests decoder-side RPS fusion and an auxiliary RPS-prediction head (multi-task, telemetry-at-train-only) on the refactored DCUNet/DCCRN.

Full batch context: [Refactored DCUNet/DCCRN — decoder-side RPS fusion + aux head](../../docs/experiments/refactored-decoder-rps-fusion.md).

## Setup

DCUNetRefactored, decoder-side hierarchical RPS fusion, on DREGON-LM.

Hydra wiring — data `dregon_lm_v1` · model `b2_dcunet_refactored_decoder_hierarchical` · loss `masked_mse` · metrics `separation_basic`. Train with `python train.py experiment=b2_dcunet_refactored_decoder_hierarchical`,
evaluate with `python eval.py experiment=b2_dcunet_refactored_decoder_hierarchical`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Refactored DCUNet/DCCRN — decoder-side RPS fusion + aux head](../../docs/experiments/refactored-decoder-rps-fusion.md).
