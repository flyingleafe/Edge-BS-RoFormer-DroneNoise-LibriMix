---
experiment: e6_noisegen_jitter_latreg
training_config: conf/experiment/e6_noisegen_jitter_latreg.yaml
batch: docs/experiments/noise-generation-augmentation.md
---

# `e6_noisegen_jitter_latreg`

## Motivation

The drone codebook has only 2 embeddings (dregon/michaels), so the decoder is
constrained at exactly 2 points in z-space — nothing regularises its behaviour
*around* each code, which later vicinal sampling / interpolation between drones
would rely on. This arm adds two opt-in latent-space regularisers on top of the
RPS-jitter arm ([e6_noisegen_jitter](e6_noisegen_jitter.md)):

1. **Vicinal conditioning noise** (`z_noise_std=0.1`): during training, each
   sample's code z is perturbed by ε ~ N(0, (0.1·RMS(z))²·I) before the
   emitter's FiLM (off at eval). The scale is *relative* to the code's own RMS
   because `DroneCodebook` codes start tiny (init_std=0.01) and grow freely —
   an absolute std would first dominate then vanish.
2. **Spectral norm on the FiLM generator** (`film_spectral_norm=true`): bounds
   the Lipschitz constant of the z → (γ, β) conditioning path.

Checkpoint caveat: spectral norm changes state-dict keys — this model is
new-training-only, not weight-compatible with the plain jitter arm.

Full batch context: [Learned noise-generation augmentation](../../docs/experiments/noise-generation-augmentation.md).

## Setup

Identical to [e6_noisegen_jitter](e6_noisegen_jitter.md) except
`/model: positional_harmonic_gen_cond_jitter_latreg`. Built on the jitter arm
by default; if the 3-arm E6 ablation crowns randphase instead, rebase this
config on that model before training.

Hydra wiring — data `noise_rps_dregon_michaels_swapped` · model
`positional_harmonic_gen_cond_jitter_latreg` · loss `multiscale_stft` · metrics
`noise_gen_spectral`. Train with `python train.py experiment=e6_noisegen_jitter_latreg`,
evaluate with `python eval.py experiment=e6_noisegen_jitter_latreg`.

## Conclusion

_Pending run._
