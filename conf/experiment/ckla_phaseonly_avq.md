---
experiment: ckla_phaseonly_avq
training_config: conf/experiment/ckla_phaseonly_avq.yaml
batch: docs/experiments/ckla.md
---

# `ckla_phaseonly_avq`

## Motivation

Beat-VK R2 arm: does adding a *third real drone* to the training noise pool
improve the phase-only CKLA predictor? The AVQ audio-visual quadrotor's 5
pure ego-noise recordings (mono 16 kHz, ~705 s) carry no rotor telemetry, so
they were pseudo-labeled by the blind-VK annotator
(`scripts/vk_pseudolabel.py` @ fa5053fc — windowed blind_seed{K,R} → VK
CAPTURE/REFINE → PIT-stitch, honest refusal at conf < 0.02) and published as
`AVQ-egonoise-vkrps` (7 contiguous accepted segments, ~617 s; refused spans
dropped, segments >= 10 s). If pseudo-labels from the classical solver can
train the neural model on new drones, the data bottleneck (2 labeled drones)
opens without telemetry.

## Setup

Clone of `ckla_phaseonly_fs_v2` (phase-only readout, p_init 1.0, rotation on,
uniform freq-scale v2 p=1.0 α∈[0.7, 1.3], grad_clip=1.0 — phase_only's atan2
readout needs it) pointing at `conf/online_mix/beatvk_avq_dload.yaml`: the
fs_v2 policy + `AVQ-egonoise-vkrps` as a `kind: frames` source at `weight:
0.5` — its own sub-pool vs the merged DREGON+michaels pool (weight 2.0), so
0.25 relative / 20% of noise chunks (AVQ is mono, so ~3% of training *frames*
under flatten_channels; DREGON+michaels stay dominant). Labels are
cruise-only (66–117 rev/s). Validation unchanged (FLY124 full-envelope).
Compare against `ckla_phaseonly_fs_v2` on the V4 valid + the VK-eval cruise
pools.

Train: `python train.py experiment=ckla_phaseonly_avq`.

## Conclusion

_Pending run._
