---
experiment: hb_hgckla_ref
training_config: conf/experiment/hb_hgckla_ref.yaml
batch: docs/experiments/rps-trajectory-refinement.md
---

# `hb_hgckla_ref`

## Motivation

`ckla_refiner` learns the pull that classical refinement cannot do, but it
learns it through a conv trunk: the frequency pool collapses the spectral axis
before the recurrence sees it, so the measurement the recurrence gets is
independent of its own state. `pi_kalman` works the other way — it reads the
spectrogram *at the harmonic positions its current estimate predicts*. This
arm is the smallest architecture that makes that true for a neural model
(`docs/pikalman-ckla-design.md`): each cell gathers the complex STFT at
`k * f_r(t)`, measures the per-harmonic innovation phasors, fuses them with
the WP18 weight law, and smooths the rate error with one CKLA scan. Three
cells = three `pi_kalman` outer iterations, on the classical coarse-to-fine
harmonic caps 10 / 25 / 40.

This is stage A of the design, the cheap and decisive one: it answers gates
G1 (synthetic capture range and precision) and G2 (protocol parity with
`--pred telem --refine pi_kalman` at more than 10x the speed).

## Setup

Model `hg_ckla_refiner` (`models/hg_ckla.py`, 221k parameters — a refinement
stack, not a trunk): `forward(audio, cond)`, output
`cond + 5 tanh(sum_j df_j / 5)` in rev/s, output rotor order == conditioning
order. The head of every cell starts at near-zero weight with a positive gain
bias, so an untrained model already applies 0.73 of the classical
linear-physics correction per cell; on a clean synthetic comb it pulls a
1.5 rev/s conditioning error to 0.04 rev/s before any training.

Data: the `ckla_refiner` corruption seam (`conf/data/ckla_refiner.yaml` — a
corrupted copy of the RPS target as `rps_cond`, train corruption seeded per
chunk id x channel, validation corruption fixed per sample) on the
honest-base R2 policy `conf/online_mix/hb_silence_dload.yaml`. Loss
`conf/loss/mse_cond.yaml` (plain non-PIT MSE). Training knobs as
`ckla_refiner`: batch 128 frames, spv 40000, grad_clip 1.0, AdamW 1e-3 /
wd 1e-4.

Train: `python train.py experiment=hb_hgckla_ref`.

## Conclusion

_Pending run._
