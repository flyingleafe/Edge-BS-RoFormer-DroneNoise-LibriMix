---
experiment: c5_simpleconv_dregon_v3
training_config: conf/experiment/c5_simpleconv_dregon_v3.yaml
batch: docs/experiments/dregon-lm-v2-v3-baseline.md
---

# `c5_simpleconv_dregon_v3`

## Motivation

Part of the DREGON-LM V2/V3 dataset-evolution and baseline-RPS-training study (since superseded by V4).

Full batch context: [DREGON-LM V2/V3 dataset evolution + baseline RPS training](../../docs/experiments/dregon-lm-v2-v3-baseline.md).

## Setup

C5 — DREGON-LM-V3 baseline: `simple_conv` on 1 s clips. Historical training used plain (non-PIT) MSE (`--no_pit_loss`); this encoding uses `pit_mse` instead — a deliberate, documented deviation (PIT is the current best practice per the C9 channel-generalization finding), see REPLICATION.md § C5. R² is degenerate on this dataset (near-constant RPS per 1 s clip); trust MSE/MAE. Historical command: train_rps_predictor.py --model simple_conv --data_root datasets/DREGON-LM-V3 --no_pit_loss --epochs 200 --patience 30 --batch_size 128.

Hydra wiring — data `dregon_lm_v3` · model `simple_conv` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c5_simpleconv_dregon_v3`,
evaluate with `python eval.py experiment=c5_simpleconv_dregon_v3`.

## Conclusion

V3 `simple_conv` baseline: val MSE ~227.0; the dataset-flaw finding that motivated V4.
