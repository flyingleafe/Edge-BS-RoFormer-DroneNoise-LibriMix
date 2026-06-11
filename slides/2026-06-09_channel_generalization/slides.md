---
theme: default
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  Channel Generalization in RPS Prediction
  Harmonic Noise Suppression Project
drawings:
  persist: false
transition: slide-left
title: Channel Generalization in RPS Prediction
mdc: true
---

# Channel Generalization in RPS Prediction

## Do ch0-trained models generalize across microphone positions?

**Dmitrii Mukhutdinov** — 2026-06-09

---

# SimpleConv

<img src="/simpleconv_tikz.png" style="max-height: 35vh; width: auto; display: block; margin: 0 auto;">

- **5** 2-D conv blocks
- **0.54 M** parameters
- Global average pooling + 1-D conv head
- Trained on **ch0 only** (DREGON-LM, 6000 samples)

---

# SimpleConvV2

<img src="/simpleconv_v2_tikz.png" style="max-height: 35vh; width: auto; display: block; margin: 0 auto;">

- **6** residual blocks + SE
- **1.50 M** parameters
- Frequency attention pooling + BiGRU head
- Trained on **ch0 only** (same data)

---

# The Question

<div class="grid grid-cols-2 gap-6">
<div>

![Mic array](/mic_array.png)

</div>
<div class="text-left">

- 8 microphones, 4 rotors
- Mic 0 (orange) = training channel
- Mic 4 (green) = same Z face as mic 0
- Mics 1,3,5,7 = opposite face

<br/>

**Do ch0-trained models work on all 8 channels?**

</div>
</div>

---

# Dataset: DREGON-LM-V4 / valid

| Recording | Clips | Source |
|-----------|------:|--------|
| `nosource` | 7 | Pure drone |
| `speech-low` | 6 | Drone + speech |
| `whitenoise-low` | 6 | Drone + noise |

- **19** non-overlapping 8 s clips
- **8 channels** per clip
- **No synthetic mixing** — raw recordings
- Early takeoff/landing excluded (`RPS > 30`)

---

# ch0-only Results: MSE

<img src="/mse_bars.png" style="max-height: 55vh; width: auto; display: block; margin: 0 auto;">

- Green = ch0 (training), red = all others
- **3–10×** MSE degradation on edge mics
- SimpleConvV2 is **worse** than SimpleConv

---

# ch0-only: SimpleConv

<img src="/slide_ch0only_sc.png" style="max-height: 50vh; width: auto; display: block; margin: 0 auto;">

- ch0: MAE = 1.13
- ch1: MAE = 7.06

---

# ch0-only: SimpleConvV2

<img src="/slide_ch0only_v2.png" style="max-height: 50vh; width: auto; display: block; margin: 0 auto;">

- ch0: MAE = 0.37
- ch3: MAE = 13.20 — **catastrophic collapse**

---

# PIT on ch0-only

| Model | MSE | PIT MSE | Δ |
|-------|----:|--------:|--:|
| SimpleConv | 35.49 | 34.77 | −2.0% |
| SimpleConvV2 | 40.28 | 40.07 | −0.5% |

---

# PIT on ch0 only

<img src="/mse_bars_pit.png" style="max-height: 55vh; width: auto; display: block; margin: 0 auto;">

---

# 8ch Training

```
Training batch: (B, C, T) → (B·C, T)

Model still sees 1 channel per prediction.
Difference: training batch contains all 8 mic positions.

We also use PIT (Permutation-invariant training) - selecting the best-fitting 
indexing of rotor predictions for loss propagation
```

| Model | Checkpoint |
|-------|------------|
| SimpleConv 8ch | `rps_8ch_v4_simple_conv` |
| SimpleConvV2 8ch | `rps_8ch_v4_simple_conv_v2` |

---

# 8ch Results: Non-PI evaluation

<img src="/mse_bars_8ch_v4.png" style="max-height: 55vh; width: auto; display: block; margin: 0 auto;">

- SimpleConv: uniform ~25–35 MSE per channel, **R² = 0.57**
- SimpleConvV2: uniform ~55–68 MSE per channel, **R² = −0.78**

---

# 8ch Results: PI evaluatoin

| Model | Not PI | PI | Δ |
|-------|-------:|----:|--:|
| SimpleConv | 29.70 | 28.37 | −4.5% |
| SimpleConvV2 | 61.39 | **3.30** | **−94.6%** |

---

# 8ch Results: PI evaluation

<img src="/mse_bars_8ch_v4_pit.png" style="max-height: 55vh; width: auto; display: block; margin: 0 auto;">

- SimpleConvV2: PIT MSE **3.30**, R² = **0.94**

It managed to learn well across all 8 channels!

---

# Why PIT is the Right Metric

<div class="grid grid-cols-2 gap-6">
<div class="text-left">

**The model gets the speeds right.**

- No-PIT MSE = 61.39
- PIT MSE = 3.30

But swaps rotor indices; however, this is okay.

Which rotor is loudest depends on mic position.
There is no acoustic signature that labels rotors
independently of where the mic is.

</div>
<div class="text-left">

Also, we do not really care about order of the rotors. We just need the model to say "here is 4 rotors and these are their RPS traces".

</div>
</div>

--- 


# Results on the sample with higly varying RPS

<img src="/slide_dynamic_8ch_sc.png" style="max-height: 65vh; width: auto; display: block; margin: 0 auto;">

---

# Results on the sample with higly varying RPS

<img src="/slide_dynamic_8ch_v2.png" style="max-height: 65vh; width: auto; display: block; margin: 0 auto;">

---

# Conclusion

<div class="grid grid-cols-2 gap-6">
<div class="text-left">

**ch0-trained models:**
- Easy to get good performance on one channel only
- This would not generalize

**8ch-trained models:**
- Simple models (SimpleConv) already cannot generalize over several mic positions for a single drone
- More complex model however can do that

</div>
<div class="text-left">

Next steps:

- Finish re-alignment of Michael's data, finally
- Include it in the training set
- Finish re-implementations of multi-pitch tracking models from literature - test them similarly on 8ch training task

</div>
</div>
