# G1 — VK-parity training arms (longer chunks)

Campaign criterion 2.3: bring an audio-only neural RPS predictor to parity
with the best blind VK tracker on the SAME evaluation clips
(`results/vk_eval/vk_valid_comparison.csv` protocol; blind-VK bars: DREGON
free-flight cruise pooled ~0.68-0.74 rev/s, FLY124 cruise 3.24).

## Phase A result (test-time smoothing, no training)

`scripts/rps_predictor_vk_eval.py` evaluated the E12 real-full-flight
checkpoints (+ C11 DREGON+FLY125 scv2) with sliding-window stitching and
2-20 s moving-average / running-median aggregation, single-mic (protocol)
and 8-mic-averaged inputs. Outcome: smoothing helps but saturates well short
of the bar on DREGON cruise (see `results/rps_predictor_vk_eval/`); the
neural error is systematic within a window, not zero-mean jitter. FLY124
cruise is already below the blind-VK 3.24 bar without any smoothing.

## Phase B hypothesis

E12 trained on 1 s chunks (`duration_s: 1.0`) but the protocol evaluates 8 s
clips; VK integrates over the whole trajectory. Give the model native
context: same recipe, `duration_s` 4/8, batch size scaled down to fit a
T4/P100 16 GB.

## Arms

| experiment | chunk | batch | policy |
|---|---|---|---|
| `g1_transformer_4s` | 4 s | 8 | `conf/online_mix/g1_real_fullflight_4s_dload.yaml` |
| `g1_transformer_8s` | 8 s | 4 | `conf/online_mix/g1_real_fullflight_8s_dload.yaml` |

Everything else matches `e12_real_fullflight_transformer` (whole-envelope
DREGON + FLY125 online mix, time-warp + gain/polarity/channel-drop augs,
valid `dload:DREGON-LM-V4-michaels-valid-full`, patience 20).

## Success criterion

Pooled per-clip PIT-MAE on the VK-comparison DREGON cruise clips
(phase-A eval script, `none` and best smoothing arm) at or below ~1.1 rev/s
(1.5x the blind-VK bar); FLY124 cruise must stay below 3.24.
