---
theme: default
background: https://images.unsplash.com/photo-1473968512647-3e447244af8f?w=1920
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  RPS Prediction & Refactored Models Presentation
  April 14, 2026
drawings:
  persist: false
transition: slide-left
title: RPS Prediction Study
mdc: true
---

# RPS Prediction for Drone Speech Enhancement

## April 14, 2026

**Harmonic Noise Suppression Project**

---

# Why This Experiment

- Multi-task training (`denoising + RPS prediction`) was much worse than baseline.
- Question: is RPS prediction itself broken for `DCUNet` / `DCCRN` encoders?
- So we isolated pure RPS prediction and compared against `SimpleConv`.

---

# What I Have Been Doing

- RPS prediction with `DCUNet` / `DCCRN` encoders vs `SimpleConv` baseline
- Goal: identify architectural limitations affecting motor-speed prediction
- In parallel: debug and restart multi-task experiments (`WIP`)

---

# RPS Predictor Architectures

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 10px;">
  <div style="border: 1px solid #c8c87a; border-radius: 8px; padding: 10px;">
    <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">SimpleConv</div>
    <div style="font-size: 0.8em; line-height: 1.35;">
      Log-Mag STFT<br/>
      ↓<br/>
      Real Conv Blocks<br/>
      ↓<br/>
      FPN Head<br/>
      ↓<br/>
      4-Rotor RPS
    </div>
  </div>
  <div style="border: 1px solid #c8c87a; border-radius: 8px; padding: 10px;">
    <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">DCUNet Encoder + RPS Head</div>
    <div style="font-size: 0.8em; line-height: 1.35;">
      Complex STFT<br/>
      ↓<br/>
      DCUNet Encoder<br/>
      ↓<br/>
      FPN Head<br/>
      ↓<br/>
      4-Rotor RPS
    </div>
  </div>
  <div style="border: 1px solid #c8c87a; border-radius: 8px; padding: 10px;">
    <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">DCCRN Encoder + RPS Head</div>
    <div style="font-size: 0.8em; line-height: 1.35;">
      Complex STFT<br/>
      ↓<br/>
      DCCRN Encoder<br/>
      ↓<br/>
      FPN Head<br/>
      ↓<br/>
      4-Rotor RPS
    </div>
  </div>
</div>

---

<div style="font-size: 80%;">

# Metrics (5 Validation Samples)

<img src="./assets/rps_comparison/summary_metrics.png" style="max-height: 20vh; width: auto; margin: 0 auto; display: block;" />

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|-------|--------|-------|------|
| **SimpleConv** | **1.63** | **1.16** | **0.82** |
| DCUNet | 2.62 | 1.69 | 0.56 |
| DCCRN | 2.46 | 1.54 | 0.61 |

</div>

---

<div style="font-size: 80%;">
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start;">
    <div>
      <div style="font-size: 0.85em; margin-bottom: 6px;">sample_00000</div>
      <img src="./assets/rps_comparison/sample_00000_plot.png" style="max-height: 40vh; width: auto; margin: 0 auto; display: block;" />
    </div>
    <div>
      <div style="font-size: 0.85em; margin-bottom: 6px;">sample_00149</div>
      <img src="./assets/rps_comparison/sample_00149_plot.png" style="max-height: 40vh; width: auto; margin: 0 auto; display: block;" />
    </div>
  </div>
</div>

---

<div style="font-size: 80%;">
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start;">
    <div>
      <div style="font-size: 0.85em; margin-bottom: 6px;">sample_00449</div>
      <img src="./assets/rps_comparison/sample_00449_plot.png" style="max-height: 40vh; width: auto; margin: 0 auto; display: block;" />
    </div>
    <div>
      <div style="font-size: 0.85em; margin-bottom: 6px;">sample_00599</div>
      <img src="./assets/rps_comparison/sample_00599_plot.png" style="max-height: 40vh; width: auto; margin: 0 auto; display: block;" />
    </div>
  </div>
</div>

---

# Main Result

- `DCUNet` / `DCCRN` encoders with attached RPS heads are worse than `SimpleConv`
- This supports the hypothesis that these encoder setups are not ideal for RPS prediction

---

# Critical Issues Found in Initial Multi-Task Setup

- Ground-truth RPS was injected at the **encoder** while the encoder also predicted RPS
- This made the auxiliary prediction objective partially meaningless
- RPS output was not normalized
- RPS loss dominated encoder updates and destabilized training dynamics

---

# Why We Restarted Multi-Task Experiments

- The RPS predictor comparison gave a real signal: architecture matters
- But the first multi-task setup had severe confounders
- We fixed those issues and relaunched experiments (`WIP`)
- Detailed multi-task results will follow later

---

# Current Training Dynamics (Restarted Multi-Task)

<div style="font-size: 80%;">
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start;">
    <img src="/@fs/home/flyingleafe/.cursor/projects/home-flyingleafe-Research-PhD-projects-harmonic-noise-suppression/assets/image-34e12723-7a58-480e-835a-1bfce252c71e.png" style="max-height: 42vh; width: auto; margin: 0 auto; display: block;" />
    <img src="/@fs/home/flyingleafe/.cursor/projects/home-flyingleafe-Research-PhD-projects-harmonic-noise-suppression/assets/image-3b697345-700c-4d0f-bc24-f7632684f77d.png" style="max-height: 42vh; width: auto; margin: 0 auto; display: block;" />
  </div>
  <div style="margin-top: 10px; text-align: left;">
    - `DCCRN (RPS+denoising, debugged)` appears on track to at least match the DREGON baseline.<br/>
    - `DCUNet (RPS+denoising, debugged)` still appears to underperform and likely needs further architectural/debugging work.
  </div>
</div>

---

# Next Steps

- Progress has been bad last month: I still had issues to finish up on the job I'm leaving
- But we have 1.5 months more for more focused research due to interruption extension
- Experimental directions to tackle before 18 May (arrival to London, end of extended interruption):
  - Still, diffusion models
  - Transformer architectures replacing convolutional architectures
  - Achieve good results separately on RPS prediction and on noise generation; then try to combine obtained RPS predictor and noise generator into the denoising model

Main issue is lack of continuous worktime spent on research still. There has not been a full day which I could entirely spend thinking about the research yet. Expectation of good progress in such a regime was incorrect. I spend too much time debugging experiment issues due to using AI for quick implementation, negating speed-ups from AI use.
