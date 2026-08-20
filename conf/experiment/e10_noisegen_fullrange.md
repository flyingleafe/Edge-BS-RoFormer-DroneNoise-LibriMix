---
experiment: e10_noisegen_fullrange
training_config: conf/experiment/e10_noisegen_fullrange.yaml
batch: docs/experiments/e10-full-flight.md
---

# `e10_noisegen_fullrange`

## Motivation

Retrain the per-drone `PositionalHarmonicNoiseGen` (the E6 winner arch) on the
**full RPS range** — including warm-up / takeoff / landing / rotors-off — so it
learns that low/zero RPS means (near-)silence. The E6 generator was trained with
proportional-in-time sampling, which saw low/zero RPS <1% of the time, so driving
it with full-flight RPS trajectories (which visit zero) would produce garbage at
zero. The new `noise_rps_dregon_michaels_fullrange` data uses `balance_rps` to
flatten each recording's RPS histogram (mean-RPS<40 coverage ~15%→50%). Dependency
for the E10 full-flight predictor trainings.

## Setup

Same as `e6_noisegen_jitter_latreg_perdrone` (model
`positional_harmonic_gen_cond_jitter_latreg_perdrone`, loss `multiscale_stft`,
metrics `noise_gen_spectral`, 60 ep, patience 8, batch 32, grad_clip 1.0, amp
off) except `/data: noise_rps_dregon_michaels_fullrange`. Cloud: override
`data.{train,valid}.params.{dregon_dir=frames:DREGON-frames,michaels_dir=frames:michaels-frames}`.

Train: `python train.py experiment=e10_noisegen_fullrange`.

## Conclusion

Ran 2026-07-12. The full-range generator retrain fed E10/E11; superseded by the silence-gated `e11_noisegen_silence`. See [e10-full-flight.md](../../docs/experiments/e10-full-flight.md).

*(Backfilled 2026-08-20.)*
