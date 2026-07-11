---
experiment: e6_noisegen_baseline
training_config: conf/experiment/e6_noisegen_baseline.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e6_noisegen_baseline`

## Motivation

Reference arm of a 3-arm generator-augmentation ablation (baseline /
random-phase / RPS-jitter) that isolates whether STFT-phase scrambling or
stochastic RPS-jitter injection reduce the mid-harmonic "washout" — the loss
suppressing mid-k harmonics because the clean rendered tones cannot match the
real, jitter-broadened harmonics.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

Same data/split/conditioning as E3 (swapped/corrected DREGON split, DREGON
in_flight + Michael's FLY-series, per-drone conditioned `cond_dim=16`) but with
the plain multi-scale STFT loss (no Stage-2 smoothness regularisers). The three
E6 arms differ **only** in the `/model` flag, so any spectral-val difference is
attributable to the generator augmentation alone. This baseline arm has no extra
augmentation (only HarmonicNoiseGenNew's always-on per-harmonic initial-phase
randomisation).

Hydra wiring — data `noise_rps_dregon_michaels_swapped` · model
`positional_harmonic_gen_conditioned` · loss `multiscale_stft` · metrics
`noise_gen_spectral`. Train with `python train.py experiment=e6_noisegen_baseline`,
evaluate with `python eval.py experiment=e6_noisegen_baseline`.

## Conclusion

_Pending run._
