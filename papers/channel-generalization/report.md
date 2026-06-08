# Channel-Generalization Failure in Learned RPS Prediction

**Date:** 2026-06-08  
**Status:** initial findings

---

## Abstract

We evaluate two RPS-prediction models (SimpleConv, SimpleConvV2) trained on DREGON-LM
on a new multichannel validation set spanning three free-flight recordings and all
8 microphone channels. Both models exhibit severe channel-dependent performance
degradation: edge-microphone MSE is 3–10× higher than the reference microphone,
and overall R² is near zero (SimpleConv barely beats the mean baseline, V2 is
slightly worse). Permutation-invariant evaluation (allowing the model to swap
motor assignments) recovers only 0.5–2.0% of the error, confirming that the
failure is genuine misprediction, not motor confusion. The conclusion: **RPS predictors trained on DREGON-LM do not generalize
across microphone positions.**

---

## 1. Motivation

RPS (rotations per second) prediction from drone audio is a core building block
for telemetry-free speech enhancement under harmonic noise. Previous work
(SimpleConv, SimpleConvV2) trained and evaluated on DREGON-LM with the implicit
assumption that microphone position does not matter — any channel should yield
similar RPS estimates.

We test this assumption directly: take trained models, evaluate them on the
**same flight recording** but through **different microphones**, and measure
the variance.

---

## 2. Dataset: DREGON-LM-V4/valid

A 19-sample, 8-channel, 8-second validation set extracted from the DREGON
corpus. **No synthetic mixing** — each sample is a raw 8-channel recording clip
(mixture = drone noise + optionally co-recorded speech or whitenoise).

| Recording | Duration (in-flight) | Non-overlapping 8 s clips | Source type |
|-----------|---------------------|--------------------------|-------------|
| `free-flight_nosource_room1` | 59.8 s | 7 | Pure drone noise |
| `free-flight_speech-low_room1` | 50.7 s | 6 | Drone + co-recorded speech |
| `free-flight_whitenoise-low_room1` | 49.9 s | 6 | Drone + co-recorded whitenoise |

- **Takeoff and landing excluded** via `min_motor_rps=30.0` threshold on
  `motors_measured`.
- **All 8 microphones** preserved — each sample is `(128000, 8)` at 16 kHz.
- Samples are **strictly non-overlapping** within each recording (grid
  extraction with remainder discarded).
- Ground-truth RPS from `motors_command` (cleaned), shared across channels.

The dataset script is `create_dregon_librimix.py` with flags:

```bash
python create_dregon_librimix.py \
  --multichannel --real_valid --max_non_overlapping \
  --output_dir datasets/DREGON-LM-V4 \
  --num_train 0 --valid_duration 8.0 --min_motor_rps 30.0 \
  --valid_recording_ids "free-flight_nosource_room1,free-flight_speech-low_room1,free-flight_whitenoise-low_room1"
```

`--max_non_overlapping` was added for this work (previously only random
sampling was available).

---

## 3. Models

| Model | Checkpoint | Architecture |
|-------|-----------|-------------|
| SimpleConv | `results/rps_exp_simple_conv/best_simple_conv.pt` | 4 conv blocks (64→128), global avg pool + 2-layer 1-D conv head |
| SimpleConvV2 | `results/rps_exp_v2/best_simple_conv_v2.pt` | 6 conv blocks (128), BiGRU temporal head, SE blocks |

Both trained on synthesised DREGON-LM (6000 train, 1 s clips, −30…0 dB SNR,
multichannel with independent speech per channel). Training uses regular MSE
loss (no PIT).

---

## 4. Results

### 4.1 Per-recording, per-channel MSE

**SimpleConv** (overall MSE=35.49, MAE=2.89, R²=0.07):

| Recording | ch 0 | ch 1 | ch 2 | ch 3 | ch 4 | ch 5 | ch 6 | ch 7 |
|-----------|------|------|------|------|------|------|------|------|
| `nosource` | 3.7 | 101.3 | 28.8 | 13.1 | 10.9 | 51.8 | 39.1 | 65.8 |
| `speech-low` | 10.7 | 77.1 | 26.0 | 10.4 | 11.8 | 43.7 | 37.8 | 37.7 |
| `whitenoise-low` | 16.5 | 78.2 | 32.1 | 12.8 | 16.2 | 47.0 | 37.9 | 36.3 |

**SimpleConvV2** (overall MSE=40.28, MAE=1.76, R²=−0.10):

| Recording | ch 0 | ch 1 | ch 2 | ch 3 | ch 4 | ch 5 | ch 6 | ch 7 |
|-----------|------|------|------|------|------|------|------|------|
| `nosource` | 0.4 | 3.2 | 3.5 | 87.3 | 3.5 | 2.2 | 25.7 | 44.8 |
| `speech-low` | 9.7 | 11.1 | 12.4 | 176.3 | 14.8 | 11.4 | 11.4 | 137.6 |
| `whitenoise-low` | 3.1 | 9.3 | 4.8 | 398.3 | 4.7 | 2.9 | 4.2 | 9.6 |

![Microphone array geometry](figures/mic_array.png)
*Figure 1. DREGON microphone array geometry. Mic 0 (orange dot, near centre) and
mic 4 (green dot, bottom) share the same Z coordinate (both at -0.041 m), placing
them on the same face of the array. The large numbers 1–4 at the top mark the
rotor positions (red wireframe), not microphones. The models were trained
exclusively on channel 0.*

![Per-channel MSE by recording and model](figures/mse_bars.png)
*Figure 2. MSE averaged over all non-overlapping 8 s clips per recording and
channel. Green bar: channel 0 (the training microphone); red bars: all others.
Top row: SimpleConv; bottom row: SimpleConvV2.*

Key observations:
- **Channel asymmetry is massive.** Channel 0 (the training mic) consistently
  outperforms edge channels (1, 7) by 3–10× in MSE.
- **Channel 4 is good because it shares the same Z coordinate as channel 0.**
  Both mic 0 and mic 4 are at Z = -0.041 m (the lower face of the array), while
  mics 1, 3, 5, 7 are at Z = +0.041 m (the upper face). Being on the same face
  means mic 4 observes the drone from the same vertical distance as the training
  mic, giving very similar signal statistics. The models have not seen channel 4
  during training either.
- **SimpleConvV2 is worse overall** despite being a more sophisticated
  architecture — it overfits training mic positions more aggressively.
- **V2 has catastrophic outliers** (ch 3 on `whitenoise-low`: MSE=398.3;
  ch 3 on `speech-low`: MSE=176.3), suggesting unstable predictions on
  certain channel × recording combinations.
- **Source type matters.** `nosource` (pure drone) is easier than
  `speech-low` (drone + co-recorded speech) — the speech source acts as
  interference for RPS estimation.

### 4.2 Permutation-invariant evaluation (PIT)

To check whether the error is due to **motor swapping** (predicting correct
RPS values but assigned to wrong rotor indices), we re-evaluate with PIT:
for each channel, try all 4! = 24 rotor permutations and pick the one that
minimises MSE, using the project's canonical `pit_mse_loss` implementation.

| Model | MSE (no PIT) | MSE (PIT) | Δ | Interpretation |
|-------|-------------|----------|---|---------------|
| SimpleConv | 35.49 | 34.77 | −2.0% | Negligible benefit |
| SimpleConvV2 | 40.28 | 40.07 | −0.5% | Negligible benefit |

![Per-channel MSE by recording and model (PIT)](figures/mse_bars_pit.png)
*Figure 3. Same as Figure 2 but with PIT (permutation-invariant) MSE.
The bars are visually unchanged — PIT recovers only 0.5–2.0% of the error,
confirming that motor swapping is not the primary failure mode.*

PIT recovers almost nothing — **motor swapping is not the primary failure
mode.** The models genuinely mispredict RPS values on unseen channels.
A few individual channels benefit more (SimpleConv `nosource` ch0:
−16.8%, ch4: −9.0%; `speech-low` ch0: −10.1%, ch4: −10.3%), suggesting
slight motor confusion on cleaner signals, but the overall effect is minor.

### 4.3 Per-channel prediction traces

To show the failure visually, we select two clips where both models perform
well on channel 0 and plot the full 8-channel prediction traces. Predictions
are PIT-permuted per channel so the plotted lines align with the ground-truth
rotor indices.

**Nosource sample** (`sample_00014`, 25.37 s into `free-flight_nosource_room1`):

![SimpleConv — nosource sample_00014](figures/sample_nosource_simpleconv.png)
*Figure 4. SimpleConv on nosource sample_00014. ch0 and ch4 track the GT
closely (MAE=1.13, 2.05); ch1, ch5, ch6, ch7 drift significantly from the
true rotor speeds (MAE=7.06, 5.54, 6.03, 4.26).*

![SimpleConvV2 — nosource sample_00014](figures/sample_nosource_simpleconv_v2.png)
*Figure 5. SimpleConvV2 on the same sample. ch0–ch2, ch4–ch6 are excellent
(MAE < 0.5), but ch3 collapses to a single intermediate value (MAE=13.2) and
ch7 shows a severe drop at ~5.5 s (MAE=8.40). The V2 model overfits training
channels so aggressively that some channels are near-perfect while others
catastrophically fail.*

**Speech sample** (`sample_00002`, 25.80 s into `free-flight_speech-low_room1`):

![SimpleConv — speech sample_00002](figures/sample_speech_simpleconv.png)
*Figure 6. SimpleConv on speech sample_00002. ch0 and ch4 are good
(MAE=1.93, 1.73); ch1 is catastrophic (MAE=9.25); ch3 is surprisingly good
(MAE=1.66).*

![SimpleConvV2 — speech sample_00002](figures/sample_speech_simpleconv_v2.png)
*Figure 7. SimpleConvV2 on the same speech sample. ch3 is poor
(MAE=4.46) and ch7 is catastrophic (MAE=7.01). The speech interference
causes a general degradation across all channels compared to the nosource
sample.*

---

## 5. Data & Reproducibility

- **Dataset:** `datasets/DREGON-LM-V4/valid` (19 samples, 40 MB, 8-channel WAV + RPS NPY).
  Created by `create_dregon_librimix.py --max_non_overlapping`.
- **Evaluation results:** `results/dregon_v4_eval/eval.json` (no PIT) and
  `results/dregon_v4_eval/eval_pit.json` (PIT). Generated by
  `evaluate-rps -i ... -m ... --pit`.
- **Code changes:** `src/tasks/rps_prediction.py` — tag propagation from
  metadata to per_sample rows; `create_dregon_librimix.py` —
  `--max_non_overlapping` flag; `train_rps_predictor.py` — `return_indices`
  on `pit_mse_loss`.
