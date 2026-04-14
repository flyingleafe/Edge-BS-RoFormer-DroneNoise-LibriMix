---
theme: default
background: https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1920
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  Edge-BS-RoFormer Progress Update
drawings:
  persist: false
transition: slide-left
title: Edge-BS-RoFormer Progress Update
mdc: true
---

<style>
/* Safe area: keep content inside slide bounds */
.slidev-layout { 
  padding: 1.5rem 2.5rem !important; 
  overflow: hidden !important;
}
/* Prevent text overflow */
.slidev-layout p, .slidev-layout li, .slidev-layout td {
  overflow-wrap: break-word;
  word-wrap: break-word;
}
</style>

# Edge-BS-RoFormer Progress Update

## UAV Speech Enhancement Research

<div class="pt-12">
  <span class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Weekly Progress Report
  </span>
</div>

---

# What Was Done This Week

<div class="grid grid-cols-2 gap-6 px-2">
<div class="min-w-0 overflow-hidden">

## Edge-BS-RoFormer Fixes

- Investigated and fixed several training issues
- Experimented with different hyperparameter configurations
- **Result**: Fixes did not lead to significant improvement

</div>
<div class="min-w-0 overflow-hidden">

## Diffusion Buffer Implementation

- Reimplemented **Diffusion Buffer** model from scratch
- Based on: *"Diffusion Buffer: Online Diffusion-based Speech Enhancement"*
- Applied to the DroneNoise-LibriMix dataset
- Currently debugging training issues

</div>
</div>

---

# Diffusion Buffer: Key Idea

<div class="grid grid-cols-2 gap-8 mt-4 text-base">

<div class="min-w-0 overflow-hidden">

**Problem**: Standard diffusion models are **too slow**
- Multiple score model calls per frame required
- RTF >> 1 → real-time processing impossible

**Diffusion Buffer solution**:
- Align diffusion time-steps with physical time
- Introduce a **buffer of B frames**
- Frames closer to present → more noise
- Frames further in past → progressively denoised
- **Only one score model call** per input frame!

</div>
<div class="min-w-0 overflow-hidden">

**Trade-off**: Buffer size B controls latency vs quality
- Latency = hop_size × B
- Achievable: **320-960ms** latency

**Key insight**: By aligning diffusion steps with time, we amortize the cost of denoising across multiple frames

</div>
</div>

---
layout: center
---

# Diffusion Buffer: Concept Diagram

```mermaid {scale: 0.55}
flowchart TB
    subgraph Stream["Noisy Audio Stream"]
        direction LR
        P1["Past"] --> P2["..."] --> R["Current"]
    end

    subgraph DB["Diffusion Buffer (B frames)"]
        direction LR
        F1["Frame 1<br/>🟢 Clean"]
        F2["Frame 2"]
        F3["..."]
        FB["Frame B<br/>🔴 Noisy"]
        F1 --> F2 --> F3 --> FB
    end

    subgraph Out["Output"]
        O["Enhanced"]
    end

    R -->|"Add noise"| FB
    F1 -->|"Pop"| O

    style F1 fill:#90EE90,stroke:#333
    style FB fill:#FFB6C1,stroke:#333
    style R fill:#FFD700,stroke:#333
    style O fill:#87CEEB,stroke:#333
```

---

# Diffusion Buffer: Training & Inference

<div class="grid grid-cols-2 gap-8 px-2">
<div class="min-w-0 overflow-hidden text-base">

## Training Process

1. Sample clean/noisy pair from dataset
2. Pad with K-1 leading zeros (init)
3. Randomly crop K=128 frames (~2s)
4. Sample ascending time-steps t⃗
5. Compute perturbed input V^t⃗
6. Optimize denoising score matching loss

</div>
<div class="min-w-0 overflow-hidden text-base">

## Online Inference

1. Initialize empty buffer V^t⃗
2. For each new frame R in stream:
   - Pop first frame → output
   - Append R + σ_tB·Z to buffer end
   - Run **one reverse step** for all frames
3. Output delay = hop_size × B

</div>
</div>

<div class="mt-4 text-center text-yellow-500 text-base">

**Key advantage**: Score model called only **once** per hop → real-time processing

</div>

---
layout: center
---

# Diffusion Buffer: Architecture Flow

```mermaid {scale: 0.6}
flowchart LR
    subgraph Input["Input"]
        A[Noisy<br/>16kHz] --> B[STFT]
        B --> C[Compress]
    end

    subgraph Buffer["Diffusion Buffer"]
        C --> D[Buffer B]
        D --> E[+Noise]
        E --> F[NCSN++]
        F --> G[Reverse]
        G --> D
    end

    subgraph Output["Output"]
        D --> H[Pop]
        H --> I[ISTFT]
        I --> J[Enhanced]
    end

    style F fill:#FFD700,stroke:#333
```

---
layout: default
class: compact-table
---

# BBED SDE Configuration

<style>
.compact-table table { margin: 0.3rem 0; font-size: 0.9rem; }
.compact-table th, .compact-table td { padding: 0.2rem 0.5rem !important; }
</style>

<div class="text-sm px-2">

<div class="grid grid-cols-3 gap-6">
<div class="min-w-0 overflow-hidden">

**SDE Parameters**

| Param | Value |
|-------|-------|
| Type | BBED |
| c | 0.08 |
| k | 2.6 |
| T | 0.8 |
| t_eps | 0.03 |

**Audio**: SR=16kHz, Win=510, Hop=256

</div>
<div class="min-w-0 overflow-hidden">

**NCSN++ Network**

| Param | Orig | Red |
|-------|------|-----|
| Ch | 128 | 96 |
| Blocks | 6 | 4 |
| Res | 2 | 1 |
| Params | 65M | 18M |

**Training**: Adam, LR=1e-4, Batch=32

</div>
<div class="min-w-0 overflow-hidden">

**Key Design Choices**

- Reduced network → faster inference
- Single score call per frame
- Buffer size B controls latency
- BBED SDE: fewer steps needed

**Latency**: B=20→320ms, B=60→960ms

</div>
</div>

</div>

---

# Training Issue: Metrics Degrading

<div class="text-center mb-1 text-sm">

**Problem**: Validation metrics decrease during training

</div>

<div class="flex justify-center items-center">
  <img src="/training_metrics.png" class="max-h-[35vh] w-auto rounded shadow object-contain" alt="Training metrics" />
</div>

<div class="grid grid-cols-3 gap-3 mt-2 text-center px-2">
<div class="bg-green-900 bg-opacity-30 p-2 rounded min-w-0 text-xs overflow-hidden">

**Train Loss** ↓ Decreasing normally

</div>
<div class="bg-red-900 bg-opacity-30 p-2 rounded min-w-0 text-xs overflow-hidden">

**SI-SDR** ↓ Going down (bad!)

</div>
<div class="bg-red-900 bg-opacity-30 p-2 rounded min-w-0 text-xs overflow-hidden">

**SDR** ↓ Also degrading

</div>
</div>

<div class="mt-2 text-center text-yellow-400 text-xs px-4">

**Investigation**: Loss↓ but metrics↓ → overfitting or loss-metric mismatch?

</div>

---

# Model Comparison: Metrics

<div class="flex justify-center items-center py-1">
  <img src="/comparison/metrics_comparison.png" class="max-h-[60vh] max-w-[95%] rounded shadow object-contain" alt="Metrics comparison" />
</div>

<div class="text-center text-sm mt-2 px-4">

**SI-SDR**, **STOI**, **PESQ** across SNR levels (-30 to -5 dB) for all models

</div>

---

# Audio Sample Comparison: Spectrograms

<div class="flex justify-center items-center py-1">
  <img src="/audio/sample_00033_comparison.png" class="max-h-[60vh] max-w-[95%] rounded shadow object-contain" alt="Spectrogram comparison" />
</div>

<div class="text-center text-sm mt-2 px-4">

Waveform and spectrogram comparison for **sample_00033** across all models

</div>

---

# Audio Sample Comparison: Listen

<div class="grid grid-cols-3 gap-4 mt-3 px-2">
<div class="text-center min-w-0 overflow-hidden">

**Edge-BS-RoFormer**

<audio controls class="w-full mt-1">
  <source src="/audio/Edge-BS-RoFormer_00033.wav" type="audio/wav">
</audio>

</div>
<div class="text-center min-w-0 overflow-hidden">

**DCUNet**

<audio controls class="w-full mt-1">
  <source src="/audio/DCUNet_00033.wav" type="audio/wav">
</audio>

</div>
<div class="text-center min-w-0 overflow-hidden">

**DPTNet**

<audio controls class="w-full mt-1">
  <source src="/audio/DPTNet_00033.wav" type="audio/wav">
</audio>

</div>
</div>

<div class="grid grid-cols-2 gap-6 mt-4 mx-auto px-4" style="max-width: 80%">
<div class="text-center min-w-0 overflow-hidden">

**HTDemucs**

<audio controls class="w-full mt-1">
  <source src="/audio/HTDemucs_00033.wav" type="audio/wav">
</audio>

</div>
<div class="text-center min-w-0 overflow-hidden">

**Diffusion-Buffer**

<audio controls class="w-full mt-1">
  <source src="/audio/Diffusion-Buffer-BBED_00033.wav" type="audio/wav">
</audio>

</div>
</div>

---

# Next Steps

<div class="grid grid-cols-2 gap-8 mt-6 px-4 text-lg">
<div class="min-w-0 overflow-hidden">

## Diffusion Buffer

- *Results promising even with broken train loop!*
- Debug the training issue
- Check loss-metric alignment
- Verify data preprocessing pipeline
- Compare with paper's training curves
- Try different buffer sizes (B=5, 10, 30, 60)
- Scale the model size!

</div>
<div class="min-w-0 overflow-hidden">

## Edge-BS-RoFormer

- Explore alternative training strategies
- Consider ensemble approaches
- Investigate combining with diffusion?

</div>
</div>

---
layout: end
---

# Questions?
