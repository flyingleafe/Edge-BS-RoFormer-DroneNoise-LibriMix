---
experiment: c8_basic_pitch_salience
training_config: conf/experiment/c8_basic_pitch_salience.yaml
batch: docs/experiments/salience-map-rps-tracking.md
---

# `c8_basic_pitch_salience`

## Motivation

Evaluates multi-pitch salience-map models (multif0 / basic-pitch) as an alternative to direct RPS regression.

Full batch context: [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).

## Setup

C8 — basic_pitch_salience (Bittner contour branch) salience-map RPS baseline on DREGON-LM-V4, 8-channel-as-extra-batch-item (see conf/data/dregon_lm_v4_8ch_flat.yaml and conf/experiment/c7_multif0_salience.yaml for the shared rationale). `optim.monitor: bce` tracks BCE-on-salience on the validation split. Historical command: train_rps_predictor.py --model basic_pitch_salience --data_root datasets/DREGON-LM-V4 --device cuda:0 --epochs 200 --patience 15.

Hydra wiring — data `dregon_lm_v4_8ch_flat` · model `basic_pitch_salience` · loss `salience_bce_basic_pitch` · metrics `salience_bce_basic_pitch`. Train with `python train.py experiment=c8_basic_pitch_salience`,
evaluate with `python eval.py experiment=c8_basic_pitch_salience`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).
