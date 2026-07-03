---
experiment: c9_simple_conv_v2_8ch
training_config: conf/experiment/c9_simple_conv_v2_8ch.yaml
batch: docs/experiments/channel-generalization-pit-loss.md
---

# `c9_simple_conv_v2_8ch`

## Motivation

Probes whether RPS predictors generalize across the 8 DREGON microphone channels, and the PIT-loss / 8-channel-training fix.

Full batch context: [Channel-generalization failure and the PIT-loss fix](../../docs/experiments/channel-generalization-pit-loss.md).

## Setup

C9 — 8-channel retrain / channel-generalization finding. Reproduces the legacy channel-as-extra-batch-item scheme (data_processing/AGENTS.md § "Multichannel Training & Evaluation Wiring") via conf/data/dregon_lm_v4_8ch_flat.yaml (flatten_channels=true), instead of the mono-only `channel: 0` used by rps_simple_conv_v2_v4 / the C10/C11 family. Historical command (approximate — the legacy _flatten_channels-based DREGONRPSDataset run had no dedicated `train_rps_predictor.py` flag beyond the model already returning a `(B*C, ...)`-shaped batch; see REPLICATION.md § C9): train_rps_predictor.py --model simple_conv_v2 --data_root datasets/DREGON-LM-V4 --epochs 200 --patience 30 --batch_size 16

Hydra wiring — data `dregon_lm_v4_8ch_flat` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c9_simple_conv_v2_8ch`,
evaluate with `python eval.py experiment=c9_simple_conv_v2_8ch`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Channel-generalization failure and the PIT-loss fix](../../docs/experiments/channel-generalization-pit-loss.md).
