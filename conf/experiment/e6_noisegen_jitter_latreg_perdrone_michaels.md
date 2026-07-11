---
experiment: e6_noisegen_jitter_latreg_perdrone_michaels
training_config: conf/experiment/e6_noisegen_jitter_latreg_perdrone_michaels.yaml
batch: docs/experiments/noise-gen-linewidth.md
---

# `e6_noisegen_jitter_latreg_perdrone_michaels`

## Motivation

Inspecting the trained two-drone generator
([e6_noisegen_jitter_latreg_perdrone](e6_noisegen_jitter_latreg_perdrone.md))
showed that **mid/high harmonics (>~2 kHz) sit below the broadband-residual floor
on both drones** — only sub-2 kHz harmonic peaks exceed the broadband magnitude.
One candidate cause is the **two-drone codebook sharing**: a single emitter is
FiLM-adapted to both DREGON and Michael's from a 2-entry codebook, and that shared
capacity (plus the vicinal z-noise / spectral-norm regularisers) might be pushing
the model to represent harmonic detail as broadband rather than fitting each
drone's comb.

This arm removes that variable: **train on Michael's only**, a single codebook
embedding (`drone_names=[michaels]`), everything else identical (learnable
per-drone σ, `z_noise_std=0.1`, spectral-norm FiLM, same schedule). If the
mid/high harmonics are *just as weak* with one drone, multi-drone adaptation is
exonerated and the culprit is the generator/loss (harmonic-amplitude rolloff, the
free-field point-source propagation missing body scattering, or the msSTFT loss
tolerating a broadband fit). If they improve markedly, the codebook sharing was
contributing.

Full batch context: [Harmonic linewidth in the noise generator](../../docs/experiments/noise-gen-linewidth.md).

## Setup

Identical to
[e6_noisegen_jitter_latreg_perdrone](e6_noisegen_jitter_latreg_perdrone.md)
except `/data: noise_rps_michaels_only` (dregon_dir=null → Michael's FLY124/FLY125
only, same swapped time-split) and `model.params.drone_names=[michaels]` (one
code + one learnable σ). Schedule unchanged (epochs 60, patience 8, batch 32,
`amp: false`, monitor mrstft max).

Hydra wiring — data `noise_rps_michaels_only` · model
`positional_harmonic_gen_cond_jitter_latreg_perdrone` (drone_names=[michaels]) ·
loss `multiscale_stft` · metrics `noise_gen_spectral`. Stream on cluster/cloud
backends with `data.{train,valid}.params.michaels_dir=frames:michaels-frames`
(dregon stays null). Train with
`python train.py experiment=e6_noisegen_jitter_latreg_perdrone_michaels`.

## Conclusion

_Pending run._
