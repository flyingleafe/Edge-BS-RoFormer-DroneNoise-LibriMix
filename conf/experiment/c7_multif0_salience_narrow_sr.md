---
experiment: c7_multif0_salience_narrow_sr
training_config: conf/experiment/c7_multif0_salience_narrow_sr.yaml
batch: docs/experiments/salience-map-rps-tracking.md
---

# `c7_multif0_salience_narrow_sr`

## Motivation

Evaluates multi-pitch salience-map models (multif0 / basic-pitch) as an alternative to direct RPS regression.

Full batch context: [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).

## Setup

C7 narrow-band + super-resolution variant — see .pi/checkpoints/salience-narrow-superres-experiment.md. Historical command: train_rps_predictor.py --model multif0_salience --data_root datasets/DREGON-LM-V4 --device cuda:0 --epochs 200 --patience 15 --hcqt_fmin 55 --hcqt_n_octaves 1 --hcqt_over_sample 10 --hcqt_harmonics 1 2 3 4 --superres_out --out_fmin 55 --out_fmax 110 --out_bins 360 --salience_blur_bins 2.

Hydra wiring — data `dregon_lm_v4_8ch_flat` · model `multif0_salience_narrow_sr` · loss `salience_bce_narrow_sr` · metrics `salience_bce_narrow_sr`. Train with `python train.py experiment=c7_multif0_salience_narrow_sr`,
evaluate with `python eval.py experiment=c7_multif0_salience_narrow_sr`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).
