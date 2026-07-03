---
experiment: c7_multif0_salience
training_config: conf/experiment/c7_multif0_salience.yaml
batch: docs/experiments/salience-map-rps-tracking.md
---

# `c7_multif0_salience`

## Motivation

Evaluates multi-pitch salience-map models (multif0 / basic-pitch) as an alternative to direct RPS regression.

Full batch context: [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).

## Setup

C7 — multif0_salience (LateDeep CNN over HCQT) salience-map RPS baseline on DREGON-LM-V4, 8-channel-as-extra-batch-item (see conf/data/dregon_lm_v4_8ch_flat.yaml — matches .pi/checkpoints/salience-baselines-dregon-v4-report.md: "30 clips x 8 channels (channels flattened into the batch via _flatten_channels)"). `optim.monitor: bce` tracks BCE-on-salience on the validation split (see conf/metrics/salience_bce_multif0.yaml / src/metrics/salience.py) — there is no RPS-space metric for salience models yet (predict_rps()'s Hungarian tracking is not wired into MetricSuite, see REPLICATION.md § C7/C8). Historical command: train_rps_predictor.py --model multif0_salience --data_root datasets/DREGON-LM-V4 --device cuda:0 --epochs 200 --patience 15.

Hydra wiring — data `dregon_lm_v4_8ch_flat` · model `multif0_salience` · loss `salience_bce_multif0` · metrics `salience_bce_multif0`. Train with `python train.py experiment=c7_multif0_salience`,
evaluate with `python eval.py experiment=c7_multif0_salience`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).
