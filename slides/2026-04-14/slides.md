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

# Motivation: Why Predict Rotor Speed?

## Rotor Per-Second (RPS) Conditioning

```mermaid
flowchart LR
    A[Noisy Speech<br/>with Drone Noise] --> B[Multi-rotor UAV<br/>4 Rotors]
    B --> C[Rotor Speed<br/>≈ 929 Hz]
    C --> D[Harmonic<br/>Noise Pattern]
    D --> E[Speech<br/>Enhancement]
    
    F[RPS Signal] --> |Conditions| E
```

**Key insight**: Drone noise harmonics depend directly on rotor speed. Providing RPS as a conditioning signal helps models better separate speech from noise.

---

# RPS Prediction Models Evaluated

## Three Architectures Compared

```mermaid
graph TD
    subgraph SimpleConv
        A1[Log-Mag<br/>Spectrogram] --> B1[Real Conv2D<br/>45 ch]
        B1 --> C1[Blocks<br/>128-256-512]
        C1 --> D1[FPN Head<br/>4 RPS]
    end
    
    subgraph DCUNet
        A2[Complex<br/>Spectrogram] --> B2[ComplexConv<br/>Encoder]
        B2 --> C2[Complex<br/>UNet Blocks]
        C2 --> D2[FPN Head<br/>4 RPS]
    end
    
    subgraph DCCRN
        A3[Complex<br/>Spectrogram] --> B3[ComplexConv<br/>+ LSTM]
        B3 --> C3[DCCRN<br/>Blocks]
        C3 --> D3[FPN Head<br/>4 RPS]
    end
```

---

# Evaluation Results: Metrics Comparison

## 5 Random Validation Samples from DREGON-LM

![Summary Metrics](assets/rps_comparison/summary_metrics.png)

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|-------|--------|-------|------|
| **simple_conv** | **1.52** | **1.17** | **0.84** |
| dcunet | 2.20 | 1.75 | 0.67 |
| dccrn | 2.16 | 1.69 | 0.68 |

**Finding**: SimpleConv baseline outperforms complex encoder models!

---

# Evaluation Results: Time Series Comparison

## RPS Predictions vs Ground Truth

![RPS Time Series](assets/rps_comparison/rps_timeseries.png)

**Observation**: All models capture the general trend, but SimpleConv has less variance in predictions.

---

# Sample-by-Sample Performance

## Individual Sample Metrics

<div class="grid grid-cols-3 gap-4">

<div>

### Sample 00114
| Model | MAE |
|-------|-----|
| SimpleConv | 1.04 |
| DCUNet | 1.38 |
| DCCRN | 1.40 |

</div>

<div>

### Sample 00025
| Model | MAE |
|-------|-----|
| SimpleConv | 1.32 |
| DCUNet | 2.11 |
| DCCRN | 2.19 |

</div>

<div>

### Sample 00250
| Model | MAE |
|-------|-----|
| SimpleConv | 0.92 |
| DCUNet | 1.38 |
| DCCRN | 1.15 |

</div>

</div>

---

# Why Does SimpleConv Outperform?

## Analysis

```mermaid
flowchart LR
    A[Complex Encoders] --> B[Multi-task Dilution<br/>Audio Enhancement<br/>+ RPS Prediction]
    A --> C[Overfitting to<br/>Speech Enhancement]
    A --> D[Complex Convolutions<br/>Harder to Optimize]
    
    E[SimpleConv] --> F[Single-task Focus<br/>RPS Prediction Only]
    E --> G[Real-valued Conv<br/>Easier Optimization]
```

**Hypotheses**:
1. Complex models are optimized for speech enhancement, not RPS prediction
2. Multi-task learning dilutes RPS prediction performance
3. Real-valued convolutions may be easier to optimize for this task

---

# New Architecture: Refactored Models

## Decoder-Only RPS Injection

```mermaid
flowchart TB
    subgraph Encoder
        A[Input Audio] --> B[Complex Conv<br/>Encoder]
        B --> C[Multi-scale<br/>Features]
        C --> D[Shared<br/>Encoder]
    end
    
    subgraph Decoder
        E[RPS<br/>Input] --> F[RotorEncoder<br/>4→64 dim]
        F --> G[Decoder<br/>+ RPS Fusion]
        C --> G
        G --> H[Enhanced<br/>Audio Output]
    end
    
    subgraph PredictionHead
        C --> I[FPN-style<br/>RPS Head]
        I --> J[Predicted<br/>RPS]
    end
```

**Key change**: RPS conditioning is now **decoder-side only**, not encoder-side.

---

# Decoder RPS Fusion Strategies

## Bottleneck vs Hierarchical

```mermaid
flowchart LR
    subgraph Bottleneck
        A[Encoder Output] --> B[RPS Features]
        B --> C[Inject at<br/>Bottleneck]
        C --> D[Full Decoder]
    end
    
    subgraph Hierarchical
        E[Encoder Output] --> F[RPS Features]
        F --> G[Inject at<br/>Level 1]
        F --> H[Inject at<br/>Level 2]
        F --> I[Inject at<br/>Level 3]
        G --> J[Decoder]
        H --> J
        I --> J
    end
```

- **Bottleneck**: Single injection point at start of decoder
- **Hierarchical**: Distributed injection at multiple decoder levels

---

# Multi-Task Learning Setup

## Auxiliary RPS Prediction with FPN Head

```mermaid
flowchart TD
    A[Input Audio] --> B[Encoder]
    B --> C[Multi-scale Features]
    C --> D[Decoder<br/>+ RPS]
    D --> E[Enhanced Audio]
    
    C --> F[FPN Head]
    F --> G[Predicted RPS]
    
    G --> H[Loss = L_audio + λ × L_rps]
    H --> I[Total Loss]
```

**Configuration**:
- `predict_rps: true`
- `rps_aux_weight: 0.1`
- λ = 0.1 (auxiliary loss weight)

---

# Current Training Experiments

## Running on Vast-Server GPUs

| Job ID | Model | Strategy | GPU | Status |
|--------|-------|----------|-----|--------|
| 3b21a556e127 | DCUNetRefactored | Bottleneck | 0 | Training |
| 2d3a2c6bcaa5 | DCUNetRefactored | Hierarchical | 1 | Training |
| 8d03823364d9 | DCCRNRefactored | Bottleneck | - | Queued |
| 25922d874184 | DCCRNRefactored | Hierarchical | - | Queued |

**Dataset**: DREGON-LM (DREGON + LibriMix)
- 6000 training samples, 8.224s each
- Real motor telemetry from DREGON dataset

---

# Experiment Configuration

## Config Files Created

```
configs/
├── 13a_DCUNetRefactored_PredRPS_bottleneck.yaml
├── 13b_DCUNetRefactored_PredRPS_hierarchical.yaml
├── 13c_DCCRNRefactored_PredRPS_bottleneck.yaml
└── 13d_DCCRNRefactored_PredRPS_hierarchical.yaml
```

**Key settings**:
- `use_rps: true` (decoder-side only)
- `predict_rps: true` (auxiliary RPS prediction)
- `rps_aux_weight: 0.1`
- `dcunet_rps_fusion: bottleneck` / `hierarchical`

---

# Visualizations: Sample Plots

## Spectrogram + RPS Time Series

<div class="grid grid-cols-2 gap-4">

<div>

### Sample 00114

![Sample 00114](assets/rps_comparison/sample_00114_plot.png)

</div>

<div>

### Sample 00025

![Sample 00025](assets/rps_comparison/sample_00025_plot.png)

</div>

</div>

---

# Visualizations: More Samples

## Samples 00250 and 00281

<div class="grid grid-cols-2 gap-4">

<div>

### Sample 00250

![Sample 00250](assets/rps_comparison/sample_00250_plot.png)

</div>

<div>

### Sample 00281

![Sample 00281](assets/rps_comparison/sample_00281_plot.png)

</div>

</div>

---

# Next Steps

## Planned Work

1. **Wait for training completion** (24h max duration)
2. **Evaluate refactored models** on DREGON-LM validation set
3. **Compare with baselines**:
   - DCUNet baseline (no RPS)
   - DCUNet with encoder-side RPS (old approach)
4. **Analyze RPS prediction accuracy** from auxiliary head
5. **Final evaluation** with metrics: SI-SDR, PESQ, STOI, ESTOI

---

# Summary

## Key Takeaways

✅ **RPS prediction evaluation completed** for 3 models on 5 samples

✅ **SimpleConv outperforms complex models** in RPS prediction:
- RMSE: 1.52 vs 2.20/2.16
- R²: 0.84 vs 0.67/0.68

✅ **Refactored DCUNet/DCCRN** with decoder-only RPS injection

✅ **4 training experiments submitted** via postdoc

⏳ **Training in progress** — results expected soon

---

# Thank You

## Questions?

**Project**: Harmonic Noise Suppression  
**Date**: April 14, 2026  
**Contact**: flyingleafe

