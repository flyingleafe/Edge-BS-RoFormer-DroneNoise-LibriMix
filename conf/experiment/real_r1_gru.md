---
experiment: real_r1_gru
training_config: conf/experiment/real_r1_gru.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r1_gru`

## Motivation

Rung 1 of the reality ladder — ONE rig (DREGON), ONE microphone — on
the causal unidirectional-GRU head (`simple_conv_v2_uni_gru128`).

The reality ladder asks how much REAL data a rotor-speed predictor needs. Four
rungs add one real ingredient each, so a gain has one cause:

- R1 (`conf/online_mix/real_r1_dload.yaml`) — DREGON only, microphone 0 only.
- R2 (`real_r2_dload.yaml`) — DREGON only, all 8 microphones. R1 to R2 is what the array buys.
- R3 (`real_r3_dload.yaml`) — DREGON plus Michael's FLY125. R2 to R3 is what a second rig buys.
- R4 (`hb_m3s2_dload.yaml`) — R3 plus `freq_scale` and the time warp. R3 to R4 is what augmentation buys.

Every rung keeps the honest-base ingredients: the zero-labeled silence arm at
16.7 % of chunks, `snr_ref_floor_rms: 0.02`, LibriSpeech at -30 to 0 dB SNR and
the post-mix gain/polarity block. `noise_augmentations` and `noise_time_warp`
are absent from R1 to R3 on purpose: both resample the noise and MULTIPLY the
labels, so they manufacture rotor speeds the rig never flew, which a "how much
real data" measurement must not be handed for free.

This file is `r2hb_gru_nogate` with the training stream replaced by
`conf/online_mix/real_r1_dload.yaml` and the monitor changed to `mae_frame`, the
PIT per-frame absolute error in rev/s that every row of the regime matrix is
compared on. Everything else is that file verbatim.

Measured stream properties (real_r1_dload.yaml, `scripts/check_stream.py --flatten
--samples-per-epoch 40000 --probes 32`, RESULT PASS): DREGON only, microphone 0 only. `scripts/check_stream.py` measures the stream
at 1 audio channel per chunk and C = 1 training frame per chunk, with
`noise_augmentations` and `noise_time_warp` structurally absent and the
post-mix gain/polarity block firing at 0.41-0.56 against the configured 0.5,
label-diff 0/32 — no augmentation on this rung transforms a label.

One consequence of C = 1: `samples_per_validation` counts FRAMES, so this rung
draws 8x as many distinct real windows per epoch as the 8-microphone rungs
while taking the same number of gradient steps. The step budget is the control
that is held; the wall-clock cost per epoch is higher.

Train: `python train.py experiment=real_r1_gru`.

## Conclusion

Pending.
