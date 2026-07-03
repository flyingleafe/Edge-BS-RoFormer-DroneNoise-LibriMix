---
experiment: e3_noise_gen_swapped_smoothness
training_config: conf/experiment/e3_noise_gen_swapped_smoothness.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e3_noise_gen_swapped_smoothness`

## Motivation

Builds and applies a learned harmonic drone-noise generator as an online-mixing augmentation source.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

E3 — E2 + swapped (corrected) DREGON split + Stage-2 smoothness regularisers (random-phase training is automatic — HarmonicNoiseGenNew always draws random per-harmonic phases while model.training is True, no flag needed). Historical command (.pi/checkpoints/noise-gen-swapped-smoothness-random-phase.md): train_noise_generation.py --online_config conf/online_mix/noise_gen_online_dregon_michaels_swapped.yaml --cond_dim 16 --device cuda:0 --epochs 200 --patience 20 --batch_size 32 --duration_s 1.0 --n_harmonics 100 --samples_per_epoch 6000 --num_valid 256 --num_workers 8 --harm_smooth_weight 1e-2 --noise_smooth_weight 1e-2 `model.task_params.return_dict: true` requests harm_amps/noise_amps as extra pred entries (tasks.codecs.NoiseGenerationCodec) for conf/loss/multiscale_stft_smoothness.yaml's smoothness terms. Same offline-dataset deviation as E2 (see conf/experiment/e2_noise_gen_dregon_michaels.yaml) plus conf/data/noise_rps_dregon_michaels_swapped.yaml's own swapped-split caveat (REPLICATION.md § E2/E3).

Hydra wiring — data `noise_rps_dregon_michaels_swapped` · model `positional_harmonic_gen_conditioned` · loss `multiscale_stft_smoothness` · metrics `noise_gen_spectral`. Train with `python train.py experiment=e3_noise_gen_swapped_smoothness`,
evaluate with `python eval.py experiment=e3_noise_gen_swapped_smoothness`.

## Conclusion

9-run smoothness sweep completed (2026-07-02): `harm_smooth_weight = 1e-1` is the
nominal winner (best spectral val **5.3506**) but sits within ~0.008 of the
no-smoothing baseline (5.3554) — **noise-level**. Harm smoothness peaks at 1e-1
(1/10 over-smooth and hurt); noise smoothness only helps at weights > 1 and hurts
at 1e-2. Smoothness does **not** meaningfully improve the raw spectral val loss;
whether it yields qualitatively cleaner harmonic/noise components is an open
per-component analysis. Sweep table + checkpoint map: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).
