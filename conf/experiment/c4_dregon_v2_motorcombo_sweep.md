---
experiment: c4_dregon_v2_motorcombo_sweep
training_config: conf/experiment/c4_dregon_v2_motorcombo_sweep.yaml
batch: docs/experiments/dregon-lm-v2-v3-baseline.md
---

# `c4_dregon_v2_motorcombo_sweep`

## Motivation

Part of the DREGON-LM V2/V3 dataset-evolution and baseline-RPS-training study (since superseded by V4).

Full batch context: [DREGON-LM V2/V3 dataset evolution + baseline RPS training](../../docs/experiments/dregon-lm-v2-v3-baseline.md).

## Setup

C4 — DREGON-LM-V2 motor-combo-fraction sweep. `simple_conv_bigru_v2` + 2.5% combo fraction is the best result found (PIT MSE 56.70 vs 71.1 @20%, 65.9 @5%, 117.3 @0% — collapses without a PIT anchor). Override `data` to reproduce another combo-fraction point: python train.py experiment=c4_dregon_v2_motorcombo_sweep data=dregon_lm_v2_0pct python train.py experiment=c4_dregon_v2_motorcombo_sweep data=dregon_lm_v2_5pct python train.py experiment=c4_dregon_v2_motorcombo_sweep data=dregon_lm_v2_20pct Historical command: train_rps_predictor.py --model simple_conv_bigru_v2 --data_root datasets/DREGON-LM-V2-2.5pct --epochs 500 --patience 30 --batch_size 96 --lr 1e-3. See REPLICATION.md § C1/C4.

Hydra wiring — data `dregon_lm_v2_2p5pct` · model `simple_conv_bigru_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=c4_dregon_v2_motorcombo_sweep`,
evaluate with `python eval.py experiment=c4_dregon_v2_motorcombo_sweep`.

## Conclusion

Found the **2.5%** motor-combo-fraction sweet spot (PIT MSE **56.70**) — U-shaped in combo fraction.
