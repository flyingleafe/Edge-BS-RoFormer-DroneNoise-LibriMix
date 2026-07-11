---
experiment: e6_noisegen_jitter
training_config: conf/experiment/e6_noisegen_jitter.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e6_noisegen_jitter`

## Motivation

RPS-jitter arm of the 3-arm generator-augmentation ablation (see
[e6_noisegen_baseline](e6_noisegen_baseline.md)). The RPS-refinement audit showed
real rotor speeds carry fast zero-mean jitter (~0.6–0.8 rev/s std on DREGON) that
telemetry-conditioned generation cannot know; the real harmonic k is broadened by
±k·σ Hz while the generator renders clean tones, so the loss suppresses mid-k
harmonics (the "washout"). Injecting a matched Ornstein-Uhlenbeck perturbation
into the conditioning RPS *inside* the emitter gives every harmonic of a rotor a
coherent, frequency-proportional, correctly-shaped broadening.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

Identical to [e6_noisegen_baseline](e6_noisegen_baseline.md) except
`/model: positional_harmonic_gen_cond_jitter` (OU injection `rps_jitter_sigma=0.6`
rev/s, `rps_jitter_tau=0.016` s). Parameters calibrated by
`scripts/calibrate_rps_jitter.py` on the 5 DREGON refined-RPS validation
recordings (`results/rps_refinement/jitter_calibration.json`): pooled σ≈0.82 rev/s
(an upper bound — measured telemetry includes sensor noise, so de-rated to 0.6)
and pooled τ≈16 ms (grid-limited by the 32 ms frame grid).

Hydra wiring — data `noise_rps_dregon_michaels_swapped` · model
`positional_harmonic_gen_cond_jitter` · loss `multiscale_stft` · metrics
`noise_gen_spectral`. Train with `python train.py experiment=e6_noisegen_jitter`,
evaluate with `python eval.py experiment=e6_noisegen_jitter`.

## Conclusion

_Pending run._
