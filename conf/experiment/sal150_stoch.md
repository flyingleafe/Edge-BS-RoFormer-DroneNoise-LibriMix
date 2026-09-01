---
experiment: sal150_stoch
training_config: conf/experiment/sal150_stoch.yaml
batch: docs/experiments/salience-map-rps-tracking.md
---

# `sal150_stoch`

## Motivation

multif0_salience on the stochastic rotor-noise curriculum, WIDE 0-150 rev/s output grid.

THE QUESTION. Every salience row in this project trains on real recordings.
None has ever been trained on a synthetic curriculum, thus "the salience
family is weak" and "the salience family was never given a task it can fit"
have never been separated. This row gives it the cleanest version of the
task: an unlimited synthetic stream, and a grid wide enough that no label is
quantized onto a clamp.

WHAT MOVES relative to conf/experiment/hb_sal_multif0.yaml (the real-data
row, flight MAE 4.01):
1. The stream, from the honest real base to conf/online_mix/stoch_s1_dload.yaml.
2. The output grid, from the standard log grid (32.7 Hz, 6 octaves) to a
linear 0-150 rev/s grid at 0.150 Hz/bin
(conf/model/multif0_salience_w150.yaml and its mirrored loss/metrics).
3. Validation, from the frozen real split to a held-out draw of the SAME
synthetic family (conf/data/sal_stoch_synthval.yaml) — the row asks
whether the architecture can learn the task, not whether it transfers.
The trunk, the optimizer, the monitor, the batch size and the clip length are
hb_sal_multif0's, unchanged.

samples_per_validation is 16000 frames (2000 chunks/epoch at 8 mics) against
hb_sal_multif0's 40000. The stream is infinite either way, so this only sets
how often the model is scored and checkpointed; the smaller epoch buys more
early-stopping resolution inside a bounded job.

Full batch context: [Salience-Map Multi-F0 Tracking for RPS Prediction](../../docs/experiments/salience-map-rps-tracking.md).

## Setup

Hydra wiring — data `sal_stoch_synthval` · model `multif0_salience_w150` · loss `salience_bce_w150` · metrics `salience_bce_w150`. Train with `python train.py experiment=sal150_stoch`.

## Conclusion

This arm's outcome is recorded in the batch write-up: [Salience-Map Multi-F0 Tracking for RPS Prediction](../../docs/experiments/salience-map-rps-tracking.md).
