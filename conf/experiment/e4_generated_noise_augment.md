---
experiment: e4_generated_noise_augment
training_config: conf/experiment/e4_generated_noise_augment.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e4_generated_noise_augment`

## Motivation

Builds and applies a learned harmonic drone-noise generator as an online-mixing augmentation source.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

E4 — RPS training (simple_conv_v2) augmented with a live GENERATED noise source (frozen, pretrained PositionalHarmonicNoiseGen rendered by a background producer process — data_processing.generated_noise.GeneratedNoisePool), ~1/3 of noise batches synthetic. See conf/data/online_mix_generated_augment.yaml for the hard prerequisite (noise-gen checkpoint must exist) and REPLICATION.md § E4 for status (code done + CPU-smoke verified; no GPU run recorded yet — reproducing this config runs the *intended* experiment, not a verified result). A/B baseline arm (same sources, no synthetic augmentation): conf/experiment/e4_no_aug_baseline.yaml. Historical command: train_rps_predictor.py --model simple_conv_v2 --data_root datasets/DREGON-LM-V4-michaels --online_mix --mix_config conf/online_mix/online_mix_generated_augment_example.yaml --samples_per_validation 5000 --pit_loss --epochs 200 --patience 50 --batch_size 16 --num_workers 6.

Hydra wiring — data `online_mix_generated_augment` · model `simple_conv_v2` · loss `pit_mse` · metrics `rps`. Train with `python train.py experiment=e4_generated_noise_augment`,
evaluate with `python eval.py experiment=e4_generated_noise_augment`.

## Conclusion

**Negative result.** Adding the live generated `michaels` source (weight 0.5,
~⅓ of noise batches; baseline no-smoothness noise-gen checkpoint) *degrades* RPS
prediction vs. the no-generator baseline: `uni_gru128` PIT MSE 7.33 → **9.29**
(+27%), `transformer` 8.46 → **10.63** (+26%). Only 2 of ~11 Slurm jobs produced
usable checkpoints (rest NaN/OOM/GPU-node failures); the transformer overfits to
generator artifacts (train MSE 9.9→3.7 while val PIT 10.6→43.6). Root cause and
next steps (smoothness-trained checkpoint, RPS-trajectory augmentation, lower
generated weight) in the batch write-up: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).
