---
experiment: e2_noise_gen_dregon_michaels
training_config: conf/experiment/e2_noise_gen_dregon_michaels.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e2_noise_gen_dregon_michaels`

## Motivation

Builds and applies a learned harmonic drone-noise generator as an online-mixing augmentation source.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

E2 — positional multi-observer noise-gen (PositionalHarmonicNoiseGen), jointly on DREGON in_flight_noise + Michael's FLY-series, per-drone conditioned (cond_dim=16). Historical command (.pi/checkpoints/noise-generation-online-dregon-michaels.md): train_noise_generation.py --online_config conf/online_mix/noise_gen_online_dregon_michaels.yaml --cond_dim 16 --device cuda:0 --epochs 200 --patience 20 --batch_size 32 --duration_s 1.0 --n_harmonics 100 --samples_per_epoch 6000 --num_valid 256 --num_workers 8 New-framework deviation (see REPLICATION.md § E2/E3): conf/data/noise_rps_dregon_michaels.yaml wraps the offline chunkable NoiseRPSDataset (single mic, time-holdout split) instead of the historical *online* per-frame-geometry streaming slicer (all 8 mics jointly) — documented, not a byte-for-byte reproduction.

Hydra wiring — data `noise_rps_dregon_michaels` · model `positional_harmonic_gen_conditioned` · loss `multiscale_stft` · metrics `noise_gen_spectral`. Train with `python train.py experiment=e2_noise_gen_dregon_michaels`,
evaluate with `python eval.py experiment=e2_noise_gen_dregon_michaels`.

## Conclusion

This run's outcome is reported comparatively in the batch write-up — see [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).
