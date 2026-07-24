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

## Phase G2 result (2026-07-24) — HCQT refuted; IF marginal new best

Protocol eval `python-e1e448` (all smoothing arms), best cells:

| model | raw DREGON | best DREGON | best FLY124 |
|---|---|---|---|
| E12 baseline | 3.186 | 2.62 | 1.55 |
| g2_hcqt_best | 4.203 | 3.323 | 2.317 |
| g2_if_best | 3.082 | **2.481** (chmean/med20) | 2.325 |
| VK bars | — | 0.68–0.74 | 1.027 |

G2a (HCQT) is refuted on every pool — consistent with the constant-Q
resolution argument (bins ~18 Hz at k=20 where 0.7 rev/s = 14 Hz) and the
prior salience-baseline failures; val MSE 195.5 had already signalled it.
G2b (IF) is the first arm to beat the baseline at all (DREGON 2.62→2.481,
raw 3.186→3.082) — the phase evidence is directionally right — but the
effect is marginal against a 3.4× gap, and FLY124 regresses (1.55→2.33).
The front-end alone, at this capacity and data scale, does not close
parity. Next lever (G4): harmonic aggregation on the LINEAR grid (comb
matched-filter stacking over an f0 grid, the trainable analogue of the VK
whitened scan) combined with the IF channel — attacks the aggregation
ingredient without CQT smearing.

## Phase G4 (comb matched-filter front-end)

Hypothesis, from the G2 evidence split: the trunk needs harmonic
aggregation like VK's whitened scan, but on the LINEAR frequency grid
(constant-Q smears sub-rev/s structure — G2a refuted), composed with the
IF machinery that already helped (G2b). The `comb_if` front-end computes,
per candidate f0 (30..120 rev/s, step 0.25 → 361 rows) and frame:

1. **comb score** — mean whitened log-mag over teeth k·f0 ≤ 1200 Hz
   (whitening = running median over frequency, 150 Hz window — the
   `vk_blind_seeding.whitened_logmag` recipe; interpolated tooth gather =
   `_tooth_values`, precomputed as index/weight buffers, one fused gather
   per forward);
2. **frequency consensus** — per-tooth IF deviation → rev/s (IF·Δf/k),
   Fisher magnitude·k²-weighted mean, clamped ±2 rev/s;
3. **occupancy** — fraction of teeth above the frame's spectrum median
   (the stage-guard tooth statistic).

| experiment | model key | front-end |
|---|---|---|
| `g4_comb_transformer` | `simple_conv_v2_transformer_comb` | `comb_if` (3 ch × 361 f0 rows, hop-512 grid) |

Everything else mirrors `e12_real_fullflight_transformer` exactly (same
`e12_real_fullflight` stream, 1 s chunks, pit_mse, augs, patience 20). The
trunk operates in f0-space where each rotor is a ridge — the network's job
reduces from "learn harmonic geometry" to "track ridges and resolve the
4-rotor assignment". Success criterion unchanged.

Status: built (front-end + model + configs committed); training not yet
submitted. Front-end sanity (unit-tested): a synthetic 4-rotor comb at
[45, 62.5, 80, 105] rev/s yields comb-score NMS peaks at exactly those
rows; a comb offset +0.2 rev/s from a grid row reads +0.206 in the
consensus channel; occupancy 0.95 at the true row vs 0.33 off-comb.

### G4a result (2026-07-24) — as-built REFUTED at val; diagnosis: position readout

Training 2qnc8y8v: best val/mse 576.5 / mae_frame 15.4 at epoch 4, then no
improvement for 20 epochs (early-stop ep 24). 15.4 rev/s mae ≈ predicting a
near-constant — the model never learned to read the ridge position. Probable
cause: in f0-space the answer IS the position along the row axis, but the
trunk's freq-pool averages that axis away; the spectrogram baseline encodes
speed in translation-covariant texture instead and never needs positional
readout. Fix under test (G4b): a 4th coordinate channel (each row's f0 in
rev/s, normalized) — with it, rps ≈ coord + consensus at the comb-score
argmax becomes a near-linear readout. Epochs are also ~9x slower than the
family (gather cost at train shapes); acceptable for a verdict run.

### G4b (coordinate channel) — hypothesis

Minimal delta on `comb_if`: a CoordConv-style 4th channel = each row's f0
in rev/s / 100, constant over time (`coord_channel: true`, now the
front-end default; `false` reproduces G4a for A/B). The G4a failure mode —
frequency pooling averaging away the row axis that carries the answer —
disappears if position is an explicit feature: ``rps ≈ coord·100 +
consensus`` at the comb-score argmax is a near-linear readout for the
head. Experiment `g4b_comb_coord_transformer` (model config
`simple_conv_v2_transformer_comb_coord`), everything else identical to
G4a. First gate: val must beat the 576.5 mse / 15.4 mae_frame
predict-the-mean plateau by a wide margin before protocol eval.

### G4b result (2026-07-24) — coord channel does NOT rescue it; G4 family refuted

Training 5f4yvaiz: best val/mse 803.2 / mae_frame 14.5 (ep 10), noisy-flat
from epoch 1 — no better than G4a (576.5/15.4). The position-readout
diagnosis was insufficient: the ridge-space representation on this trunk
fails to train regardless. A design mismatch noted post-hoc: the f0 grid
(30–120 rev/s) cannot represent the E12 full-flight stream's sub-30
segments (idle/warmup, ~a third of the envelope) — the front-end emits no
evidence there — though this alone cannot explain a 10x val gap vs the
baseline (65). Verdict: the comb-matched-filter front-end family (G4a,
G4b) is refuted as-built at this capacity/recipe.

## Criterion 2.3 ledger (2026-07-24)

Measured levers, best DREGON-cruise protocol MAE: smoothing 2.62 · context
4s/8s 2.87 (worse) · HCQT 3.32 (worse) · IF channel **2.481** (best) ·
comb ridge ±coord — dead at val · GP-noise aug 3.13 (worse). VK bars
0.68–0.74 / 1.03. Every cheap lever (test-time, recipe, front-end,
augmentation) is now measured; the residual 3.4x gap is structural at this
model scale and data volume. Remaining levers are programs, not configs:
VK-distilled labels / VK-annotated unlabeled data (semi-supervised scale),
or a VK-hybrid inference design. Parity is NOT achieved.
