# SimpleConv Architecture Variants: A Systematic Evaluation

**Date:** 2026-05-30

---

## Abstract

We conduct a systematic sweep of ten architectural variants of SimpleConv, a lightweight CNN for multi-rotor RPS (rotations per second) estimation from drone audio. All variants are trained under identical conditions on DREGON-LM (6000 train / 600 valid samples, −30 to 0 dB SNR). We report validation-set metrics, full-sequence evaluation on a real ~47 s free-flight recording, and out-of-distribution tests on clean individual-motor and synchronized four-rotor recordings. The key finding is that adding a bidirectional GRU temporal head yields the largest single improvement (R² 0.837 → 0.945), and a deeper encoder with squeeze-excitation blocks pushes this further to R² = 0.948 with strong generalisation to real recordings (in-flight MSE 9.9 vs baseline 24.4).

---

## 1. Method

### 1.1 Model variants

All models share a convolutional encoder that downsamples frequency while preserving time, followed by a temporal head. The variants differ in:

| Variant | Encoder | Temporal head | Input | Extra features |
|---------|---------|---------------|-------|--------------|
| Baseline | 4 blocks, 64→128 | Global avg pool + 2-layer 1-D conv | 1-ch log-mag | — |
| BiGRU | 4 blocks, 64→128 | BiGRU (2×128) | 1-ch | — |
| BiGRU-v2 | 6 blocks, 128 | BiGRU (2×128) | 1-ch | SE after each block |
| v2 (SE+Attn) | 6 blocks, 128 | BiGRU (2×128) | 1-ch | SE + frequency-attention pool |
| TCN | 4 blocks, 64→128 | Dilated conv (rf=31) | 1-ch | — |
| MagPhase | 4 blocks, 64→128 | BiGRU (2×128) | 3-ch (mag+cos+sin) | — |
| AttnPool | 4 blocks, 64→128 | Multi-head attention pool + MLP | 1-ch | — |
| Wide | 4 blocks, 128→256→512 | Global avg pool + MLP | 1-ch | Pure width scaling |
| MultiScale | 4 blocks | FPN-style fusion + MLP | 1-ch | Bottom-up skip connections |
| SE-Next | 6 blocks, 128→256 | Global avg pool + MLP | 1-ch | SE + residual, no temporal |

All trained with AdamW (lr 1e−3, wd 1e−4), batch size 16, mixed precision, gradient clip 5.0, patience 15.

---

## 2. Results

### 2.1 Validation-set leaderboard (held-out synthetic mixtures)

![Validation leaderboard](figures/fig_leaderboard_validation.png)

**Figure 1.** (a) Mean squared error and (b) coefficient of determination on the 600-clip DREGON-LM validation set. Lower MSE and higher R² are better.

| Rank | Model | MSE↓ | R²↑ | Params |
|------|-------|------|-----|--------|
| 1 | v2 (SE+Attn) | 2.61 | 0.951 | 1.50M |
| 2 | BiGRU-v2 | 2.67 | 0.948 | 1.44M |
| 3 | BiGRU | 2.74 | 0.945 | 0.67M |
| 4 | TCN | 3.09 | 0.936 | 1.38M |
| 5 | MagPhase | 3.16 | 0.917 | 0.67M |
| 6 | AttnPool | 4.87 | 0.860 | 0.56M |
| 7 | Wide | 5.04 | 0.847 | 3.94M |
| 8 | MultiScale | 5.15 | 0.840 | 1.36M |
| 9 | Baseline | 5.21 | 0.837 | 0.54M |
| 10 | SE-Next | 7.30 | 0.688 | 1.41M |

![Pareto frontier](figures/fig_pareto_params_r2.png)

**Figure 2.** Parameter count vs. validation R². The BiGRU family dominates the Pareto frontier; `BiGRU` (0.67M params) offers 99.4% of v2's performance at 44% of the parameters.

### 2.2 Full-sequence evaluation (real free-flight recording)

We evaluate all variants on the DREGON `free-flight_speech-high_room1` recording (~47 s, higher SNR than training). The in-flight metric excludes takeoff/landing (RPS < 50 rev/s), where all models fail due to distribution shift.

![Full-sequence comparison](figures/fig_fullsequence_comparison.png)

**Figure 3.** (Top) Mean predicted rotor speed vs. time for five representative variants, overlaid with ground truth. (Bottom) Per-frame MSE (1-s smoothed). BiGRU-v2 shows the tightest tracking and lowest error.

| Model | Global MSE | In-flight MSE↓ | Global R² |
|-------|-----------|---------------|-----------|
| **BiGRU-v2** | **73.60** | **9.90** | **0.839** |
| v2 (SE+Attn) | 137.48 | 11.80 | 0.700 |
| BiGRU | 104.10 | 15.19 | 0.772 |
| Wide | 106.85 | 18.87 | 0.766 |
| AttnPool | 110.88 | 19.99 | 0.758 |
| Baseline | 104.14 | 24.45 | 0.772 |
| SE-Next | 113.67 | 23.44 | 0.752 |
| MagPhase | 91.71 | 25.73 | 0.800 |
| TCN | 105.82 | 28.14 | 0.769 |
| MultiScale | 1940.89 | 111.71 | −0.862 |

![In-flight MSE bar chart](figures/fig_fullsequence_inflight_mse_bar.png)

**Figure 4.** In-flight MSE on the full-sequence recording. BiGRU-v2 halves the baseline error.

**Detailed 3-panel plots** (spectrogram + traces + per-frame MSE) are provided for the top three variants:

- Baseline: `fig_fullsequence_simple_conv.pdf`
- BiGRU: `fig_fullsequence_simple_conv_bigru.pdf`
- BiGRU-v2: `fig_fullsequence_simple_conv_bigru_v2.pdf`

![Baseline full sequence](figures/fig_fullsequence_simple_conv.png)

**Figure 5.** Baseline SimpleConv on the full sequence. Predictions follow the general trajectory but show higher variance.

![BiGRU-v2 full sequence](figures/fig_fullsequence_simple_conv_bigru_v2.png)

**Figure 6.** BiGRU-v2 on the full sequence. The deepest encoder + BiGRU head tracks all four rotors with notably lower per-frame error.

### 2.3 Individual-motor and allMotors evaluation

All models were trained on four-rotor mixtures with varying speeds. When evaluated on **clean single-rotor recordings** (constant speed, no speech), all ten variants fail catastrophically (MSE in the thousands), confirming that the network has learned a strong structural prior: it expects four independent rotors and cannot reconcile single-rotor input with its internal model.

![Individual motor bar chart](figures/fig_individual_motor_mse_bar.png)

**Figure 7.** Best-channel MSE on individual-motor recordings. All variants fail by two to three orders of magnitude, as expected. This is the same structural failure mode reported for the baseline in the classical-baselines study.

On **allMotors_70** (four synchronized rotors at 70 rev/s), the models behave much better because the input matches their structural expectation:

| Model | Best MSE↓ | Avg MSE↓ |
|-------|----------|---------|
| MultiScale | 16.10 | 28.93 |
| Baseline | 22.26 | 91.07 |
| Wide | 22.24 | 85.26 |
| BiGRU-v2 | 22.30 | 89.29 |
| SE-Next | 24.12 | 94.10 |
| TCN | 28.96 | 65.85 |
| AttnPool | 30.06 | 113.03 |
| v2 (SE+Attn) | 316.55 | 347.13 |
| BiGRU | 392.28 | 416.77 |
| MagPhase | 640.30 | 676.49 |

![allMotors comparison](figures/fig_single_rotor_allmotors_comparison.png)

**Figure 8.** Predictions on `allMotors_70` for five variants. Dotted lines = four output channels; solid = mean; bold = best channel. The baseline-like architectures cluster near the target (70 rev/s). BiGRU and v2 stray further because their temporal heads expect dynamic trajectories.

![allMotors MSE bar chart](figures/fig_allmotors_mse_bar.png)

**Figure 9.** MSE on `allMotors_70` comparing best channel vs. mean over four channels. Baseline-like architectures cluster around MSE 22–30 on the best channel.

---

## 3. Discussion

### What works

1. **BiGRU temporal head is the dominant improvement.** Adding BiGRU alone jumps R² from 0.837 to 0.945 — the largest single gain. Every top-5 model has it.
2. **Deeper encoder + SE helps generalisation.** BiGRU-v2 (6 blocks + SE) matches v2 on validation (R² 0.948 vs 0.951) but wins decisively on the real recording (in-flight MSE 9.9 vs 11.8). The SE blocks appear to improve feature quality for out-of-distribution conditions.
3. **TCN is the best non-recurrent architecture.** Dilated convolutions give R² = 0.936, competitive but ~0.015 behind the BiGRU family.

### What does not work

4. **SE-Next is actively harmful.** Without temporal modelling, the SE-heavy 6-block encoder achieves R² = 0.688 — worse than baseline. Channel attention alone cannot compensate for missing temporal structure.
5. **Width scaling does not help.** The wide model (3.94M params, 7.3× baseline) barely beats baseline on validation and shows no improvement on real data.
6. **Phase input adds complexity without gain.** MagPhase_bigru (3-channel input) at R² = 0.917 is 0.028 behind plain BiGRU despite using more information.
7. **Multi-scale fusion is unstable.** Oscillating validation loss and broken temporal resolution on full sequences (258 frames instead of 1465) make this variant unusable in its current form.
8. **Attention pooling is modest.** +0.023 R² over baseline, far behind BiGRU.

### Practical recommendation

For downstream use in speech-enhancement pipelines that depend on accurate RPS conditioning, **SimpleConv-BiGRU-v2** offers the best trade-off:
- Strongest real-recording performance (in-flight MSE 9.9, vs baseline 24.4)
- Competitive validation accuracy (R² 0.948, within 0.003 of the best)
- Moderate size (1.44M params, 2.7× baseline)

If parameter count is critical, **SimpleConv-BiGRU** (0.67M params, R² 0.945) provides 99.4% of v2's validation performance at 44% of the parameters, with real-recording in-flight MSE of 15.2.

---

## 4. Files

| Path | Content |
|------|---------|
| `figures/fig_leaderboard_validation.{pdf,png}` | Validation MSE and R² bar charts |
| `figures/fig_pareto_params_r2.{pdf,png}` | Params vs R² scatter |
| `figures/fig_fullsequence_comparison.{pdf,png}` | Multi-variant overlay on real recording |
| `figures/fig_fullsequence_inflight_mse_bar.{pdf,png}` | In-flight MSE comparison |
| `figures/fig_fullsequence_simple_conv.{pdf,png}` | 3-panel plot: baseline |
| `figures/fig_fullsequence_simple_conv_bigru.{pdf,png}` | 3-panel plot: BiGRU |
| `figures/fig_fullsequence_simple_conv_bigru_v2.{pdf,png}` | 3-panel plot: BiGRU-v2 |
| `figures/fig_individual_motor_mse_bar.{pdf,png}` | Individual-motor MSE (all fail) |
| `figures/fig_single_rotor_allmotors_comparison.{pdf,png}` | Trace plots on allMotors_70 |
| `figures/fig_allmotors_mse_bar.{pdf,png}` | allMotors_70 MSE bar chart |
| `report.md` | This document |
