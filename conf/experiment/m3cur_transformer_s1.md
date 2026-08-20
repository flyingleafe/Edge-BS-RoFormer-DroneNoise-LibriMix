---
experiment: m3cur_transformer_s1
training_config: conf/experiment/m3cur_transformer_s1.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3cur_transformer_s1`

## Motivation

Stage 1 of the M3 two-stage RPS-predictor curriculum for the IF front-end transformer (`simple_conv_v2_transformer_if`): pre-train
on SYNTHETIC noise only, then (stage 2) fine-tune on real recordings. The
generated half is the refined-label per-rotor generator
`gen_m3_refined_all_perrotor` (ep30, mrstft 6.7391) — the first generator whose
emitter code is per-rotor (`z_r = z_drone + delta_z_r`), so the four rotors of a
drone can differ in timbre instead of sharing one code. Two new vicinal knobs
widen that axis at sampling time: `perrotor_noise: 0.5` redraws the per-rotor
deltas each producer batch, and `rotor_jitter_std: 0.2` jitters the interpolated
rotor POSITIONS, so the stream is a continuum of airframes rather than a
continuum between two fixed rigs. The question this stage answers: does a
generator good enough to place lines ON the comb (the refined-labels result)
give a predictor a better starting point than real data alone?

## Setup

Data `m3cur_s1` (`conf/online_mix/m3cur_s1_dload.yaml`): 50% generated + 50%
static comb, both on `rps.kind: full_flight` (ground -> warm-up -> takeoff ->
cruise -> landing -> ground, so silence at zero RPS), LibriSpeech
train-clean-100 at SNR U[-30, 0] dB. ONE policy stage, no warm-up: `freq_scale`
alpha in [0.7, 1.3] on EVERY chunk, post-mix gain/polarity on 50%, noise
time-warp on 50%, all from sample 1 — a deliberate override of the G5 warm-up
convention (the synthetic distribution is not the one we want fitted, so
settling on it first would train the amplitude anchor stage 2 must undo).
Model `simple_conv_v2_transformer_if`, loss `pit_mse`, metrics `rps`, batch 128 frames,
`samples_per_validation=40000`. Validation is the fixed FULL-envelope real split
`dload:DREGON-LM-V4-michaels-valid-full`, so the sim->real gap is read directly
off the validation curve.

Train: `python train.py experiment=m3cur_transformer_s1`.

## Conclusion

Pending.
