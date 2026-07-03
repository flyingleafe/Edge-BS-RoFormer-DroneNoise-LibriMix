---
experiment: c8_basic_pitch_salience_narrow_sr
training_config: conf/experiment/c8_basic_pitch_salience_narrow_sr.yaml
batch: docs/experiments/salience-map-rps-tracking.md
---

# `c8_basic_pitch_salience_narrow_sr`

## Motivation

Evaluates multi-pitch salience-map models (multif0 / basic-pitch) as an alternative to direct RPS regression.

Full batch context: [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).

## Setup

C8 narrow-band + super-resolution variant — see .pi/checkpoints/salience-narrow-superres-experiment.md. Historical command: train_rps_predictor.py --model basic_pitch_salience --data_root datasets/DREGON-LM-V4 --device cuda:0 --epochs 200 --patience 15 --bp_fmin 55 --bp_bins_per_semitone 4 --bp_n_contour_semitones 12 --superres_out --out_fmin 55 --out_fmax 110 --out_bins 360 --salience_blur_bins 2.

Hydra wiring — data `dregon_lm_v4_8ch_flat` · model `basic_pitch_salience_narrow_sr` · loss `salience_bce_narrow_sr` · metrics `salience_bce_narrow_sr`. Train with `python train.py experiment=c8_basic_pitch_salience_narrow_sr`,
evaluate with `python eval.py experiment=c8_basic_pitch_salience_narrow_sr`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Salience-map RPS tracking](../../docs/experiments/salience-map-rps-tracking.md).
