---
experiment: e6_noisegen_jitter_latreg_perdrone
training_config: conf/experiment/e6_noisegen_jitter_latreg_perdrone.yaml
batch: docs/experiments/noise-gen-linewidth.md
---

# `e6_noisegen_jitter_latreg_perdrone`

## Motivation

The E6 winner ([e6_noisegen_jitter_latreg](e6_noisegen_jitter_latreg.md)) used a
single **global** OU jitter linewidth `sigma=0.6 rev/s`, calibrated on DREGON. It
won every DREGON comb-masked band by 0.7–2.0 dB — but was a **four-way tie** on
FLY124 (out-of-train DJI M100 near idle). A DREGON-calibrated linewidth does not
transfer to a different airframe/throttle regime: the physical jitter amplitude
differs per drone.

This arm promotes `sigma` to a **learnable per-drone** parameter
(`learn_rps_jitter_sigma=true`): one softplus-positive scalar per codebook entry
(dregon/michaels), initialised from `rps_jitter_sigma=0.6` and fit jointly with
the codes and the decoder. `_apply_rps_jitter` factors sigma out of the
gradient-free OU innovation path, so sigma gradients flow; the per-drone raws
live on the codebook wrapper (name-keyed, few-shot-adaptable), and `rps_jitter_tau`
stays shared. Hypothesis: each drone learns its own harmonic linewidth and the
FLY124 tie breaks.

Checkpoint caveat: adds new state-dict keys (`log_jitter_sigma.*`) on top of the
spectral-norm rename — new-training-only, not weight-compatible with the
fixed-sigma arms.

Full batch context: [Harmonic linewidth in the noise generator](../../docs/experiments/noise-gen-linewidth.md).

## Setup

Identical to [e6_noisegen_jitter_latreg](e6_noisegen_jitter_latreg.md) except
`/model: positional_harmonic_gen_cond_jitter_latreg_perdrone`, which flips the
single new `learn_rps_jitter_sigma` knob. Same swapped DREGON+Michael's split,
same z-noise + spectral-norm FiLM latent regularisation, same schedule
(epochs 60, patience 8, batch 32, `amp: false`, monitor mrstft max).

Hydra wiring — data `noise_rps_dregon_michaels_swapped` · model
`positional_harmonic_gen_cond_jitter_latreg_perdrone` · loss `multiscale_stft` ·
metrics `noise_gen_spectral`. Train with
`python train.py experiment=e6_noisegen_jitter_latreg_perdrone` (stream via
`data.{train,valid}.params.{dregon_dir=frames:DREGON-frames,michaels_dir=frames:michaels-frames}`
on cluster backends), evaluate with
`python eval.py experiment=e6_noisegen_jitter_latreg_perdrone`.

## Conclusion

_Pending run._
