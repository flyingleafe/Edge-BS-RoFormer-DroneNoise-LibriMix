---
experiment: ckla_phaseonly_8s
training_config: conf/experiment/ckla_phaseonly_8s.yaml
batch: docs/experiments/ckla.md
---

# `ckla_phaseonly_8s`

## Motivation

Longer-window arm of the phase-only CKLA readout. The fs_v2 arms train on
1 s chunks (32 STFT frames) — short enough that the recurrent state barely
settles and slow RPS drift is invisible within a sample. VK, by contrast,
integrates over 20 s windows. If the phase-only readout's tracked
instantaneous frequency benefits from longer integration, 4 s training
chunks (126 STFT frames) should close part of the gap at identical
per-step compute/memory.

## Setup

Clone of `ckla_phaseonly_fs_v2` with two changes, everything else preserved
(grad_clip=1.0 — phase_only's atan2 readout needs it):

- **Chunk length**: `conf/online_mix/e12_fullflight_freqscale_v2_8s_dload.yaml`
  — byte-identical to the fs_v2 policy except `duration_s: 4.0`.
- **batch_size 128 → 32 frames** (4 chunks × 8 mics): 4x fewer frames, each
  4x longer, so per-step memory stays equal to fs_v2.

`samples_per_validation` stays 40000 frames = 5000 chunks/epoch — the stage
boundary (50000 chunks) is still effective epoch 10; an epoch now covers 4x
the audio-time. Validation unchanged (FLY124 full-envelope 8 s clips).
Compare against `ckla_phaseonly_fs_v2`.

Train: `python train.py experiment=ckla_phaseonly_8s`.

## Conclusion

_Pending run._
