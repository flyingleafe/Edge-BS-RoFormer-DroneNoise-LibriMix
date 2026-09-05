---
experiment: real_r2_hppnet
training_config: conf/experiment/real_r2_hppnet.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r2_hppnet`

## Motivation

Rung 2 of the reality ladder — ONE rig (DREGON), ALL 8 microphones — on the salience
port HPPNet (`hppnet_rps_l4`).

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

This file is `hppnet_r2hb_l4` with the training stream replaced by
`conf/online_mix/real_r2_dload.yaml` and nothing else changed; like its source it trains
from scratch and selects on `rps_mae`, the per-rotor layer peak read in rev/s.

Measured stream properties (real_r2_dload.yaml, `scripts/check_stream.py --flatten
--samples-per-epoch 40000 --probes 32`, RESULT PASS): DREGON only, the whole 8-microphone ring. `scripts/check_stream.py` measures
8 audio channels per chunk and C = 8 training frames per chunk, with
`noise_augmentations` and `noise_time_warp` structurally absent and the
post-mix block at 0.38-0.53 against 0.5, label-diff 0/32.

Train: `python train.py experiment=real_r2_hppnet`.

## Conclusion

Pending.
