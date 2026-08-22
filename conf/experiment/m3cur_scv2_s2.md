---
experiment: m3cur_scv2_s2
training_config: conf/experiment/m3cur_scv2_s2.yaml
batch: docs/experiments/generator-refined-labels.md
---

# `m3cur_scv2_s2`

## Motivation

Stage 2 of the M3 curriculum for the plain BiGRU baseline (`simple_conv_v2`): fine-tune on REAL data,
warm-started from `m3cur_scv2_s1`'s `best.ckpt`. Measures what the synthetic
pre-training was worth — the same real recipe trained from scratch is the
`g2_*_freqscale_v2` / e12 family, so the pair is a controlled read of the
curriculum against a cold start.

## Setup

Data `m3cur_s2` (`conf/online_mix/m3cur_s2_dload.yaml`): the fs_v2 real sources
verbatim — DREGON `in_flight_noise` minus `free-flight_nosource_room1` and
Michael's FLY125, both `min_motor_rps: 0.0` (whole envelope), plus LibriSpeech
train-clean-100. The ONE change from fs_v2 is the policy: its 50k unaugmented
warm-up stage is removed, so `freq_scale` (p=1.0, alpha in [0.7, 1.3]), the
post-mix gain/polarity block (p=0.5) and the noise time-warp (p=0.5) all fire
from sample 1 — the model arrives converged, so a warm-up would only re-teach
the RPS prior stage 1 paid to break. Model `simple_conv_v2`, loss `pit_mse`, metrics
`rps`, warm start
`checkpoint: r2://ml-data/artifacts/m3cur_scv2_s1/checkpoints/best.ckpt`
(loaded `strict=False` before training), renewed patience 20, batch 128 frames,
`samples_per_validation=40000`. Same fixed FULL-envelope real validation split
as stage 1.

Train: `python train.py experiment=m3cur_scv2_s2` (after stage 1 uploads
best.ckpt).

## Conclusion

Ran 2026-08-21. Best val/mse **28.4** / mae_frame 3.16 vs control scv2_fs_v2 52.5/3.98 (−46% MSE — best paper-architecture number on this valid to date). Regime probe: the gain is almost entirely the stopped-rotor regime (silence MAE 11.8 → 4.8 rev/s); flight +10% MSE on a small base.

*(Backfilled 2026-08-22; regime probe results/m3cur_regime_probe/.)*
