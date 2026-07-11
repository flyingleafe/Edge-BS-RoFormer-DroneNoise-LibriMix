---
experiment: e6_noisegen_randphase
training_config: conf/experiment/e6_noisegen_randphase.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e6_noisegen_randphase`

## Motivation

Random-phase arm of the 3-arm generator-augmentation ablation (see
[e6_noisegen_baseline](e6_noisegen_baseline.md)). Tests whether STFT-phase
scrambling of the harmonic bank (`HarmonicNoiseGenNew` `use_random_phases=true`)
helps the generator avoid the mid-harmonic washout.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

Identical to [e6_noisegen_baseline](e6_noisegen_baseline.md) except
`/model: positional_harmonic_gen_cond_randphase` (`use_random_phases: true`).

Hydra wiring — data `noise_rps_dregon_michaels_swapped` · model
`positional_harmonic_gen_cond_randphase` · loss `multiscale_stft` · metrics
`noise_gen_spectral`. Train with `python train.py experiment=e6_noisegen_randphase`,
evaluate with `python eval.py experiment=e6_noisegen_randphase`.

## Conclusion

_Pending run._
