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

## Phase B result (2026-07-24) — hypothesis REFUTED

Trainings (kaggle P100, wandb `9pf3rpoh` 4s / `4bwjujj7` 8s; first
submission pair died because an unanchored `data` pattern in
`.git/info/exclude` had silently kept `conf/data/g1_*.yaml` out of the
repo — fixed in b4a3eee): 4s early-stopped ep 36 (best ep 16), 8s ep 28
(best ep 8). Eval: `python-267f52`,
`omnirun-outputs/python-267f52/results/rps_predictor_vk_eval/`.

Best pooled per-clip PIT-MAE (rev/s) across all input/aggregation arms:

| model | best arm | DREGON cruise | FLY124 cruise |
|---|---|---|---|
| E12 baseline (1 s) | none | 3.186 | 1.766 |
| E12 + phase-A smoothing | med | 2.62 | 1.55 |
| g1 4 s best.ckpt | chmean/stitchmed | 3.043 | 2.094 |
| **g1 8 s best.ckpt** | chmean/med2 | **2.872** | 1.898 |
| g1 4 s / 8 s last.ckpt | — | 3.554 / 7.071 | (overfit) |
| VK telemetry-init | — | 0.729 | 0.283 |
| VK blind (guarded, §7.5) | — | 0.68–0.74 | **1.027** |

Native 4/8 s context does not close the gap: the best phase-B number
(2.87) is *worse* than phase-A smoothing of the 1 s model (2.62), and
`last` checkpoints degrade sharply (context length trades against
optimization stability at fixed compute). The systematic within-window
error is therefore NOT a context-length artifact. FLY124 note: the
stage-guarded blind VK (1.027) now beats every neural variant there too —
both halves of the parity criterion are open.

**Conclusion:** criterion 2.3 fails via the context lever. Remaining
parity ideas (from the campaign plan, unexplored): comb-structured
front-end, VK-distilled training targets, VK-annotated unlabeled data —
each a larger build than a config change; park for a deliberate decision.

## Phase G2 (front-end arms)

Phases A and B established that the within-window error is systematic and
NOT a context-length or aggregation artifact. G2 tests the remaining
campaign-plan lever: the **magnitude-STFT front-end is the bottleneck**.
Two independent deficits, one arm each:

- the trunk has no **harmonically aligned evidence aggregation** — VK
  integrates a full comb of harmonics per rotor, while a plain spectrogram
  makes the CNN learn harmonic geometry from scratch;
- the trunk has no **sub-bin frequency precision** — one bin at
  n_fft=2048/16 kHz is 7.8 Hz, so magnitude alone cannot resolve the
  ~0.7 rev/s scale of the VK bar.

| experiment | model key | front-end |
|---|---|---|
| `g2_hcqt_transformer` | `simple_conv_v2_transformer_hcqt` | HCQT (nnAudio, 16 kHz native, fmin 32.7, 6 oct x 60 bins/oct, harmonics [1,2,3], mag+dphase = 6 ch, hop 256 time-interpolated onto the hop-512 grid) |
| `g2_if_transformer` | `simple_conv_v2_transformer_if` | `stft_mag_if` (log-mag + instantaneous-frequency deviation in fractional bins, 2 ch, same 2048/512 grid) |

Everything else mirrors `e12_real_fullflight_transformer` exactly (same
`e12_real_fullflight` online-mix stream, 1 s chunks, pit_mse, augs,
patience 20) to isolate the front-end effect. Success criterion unchanged
(§ above): pooled per-clip PIT-MAE on the VK-comparison DREGON cruise clips
at or below ~1.1 rev/s, FLY124 cruise below the guarded blind-VK 1.027.

Status: built (front-ends + models + configs committed); trainings not yet
submitted.
