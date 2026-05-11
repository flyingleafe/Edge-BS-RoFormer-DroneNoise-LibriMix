---
theme: default
background: https://source.unsplash.com/1920x1080/?technology,dark
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  RPS Predictor High-SNR Analysis
drawings:
  persist: false
transition: slide-left
title: RPS Predictor Performance on High-SNR Samples
mdc: true
---

# Harmonic Noise Suppression

## Progress Update — 05 May 2026


---

# Topics

1. A bit of experiments since last presentatoin
2. What is the overall plan until write-up?

---

# First, a bit of evaluations

Answering the question - if RPS predictors perform well on high-SNR samples (with pronounced speeds)

---

# Research Question

<div class="grid grid-cols-2 gap-8">

<div>

## How do RPS predictors perform on **high-SNR** samples?

<br>

- Current evaluation focuses on **low-SNR** synthetic mixtures (-30 to 0 dB)
- Real-world scenarios include **high-SNR** recordings where speech is strong
- Question: *Does RPS conditioning help or hurt when speech dominates?*

<br>

**Hypothesis**: RPS predictors may struggle when speech masks rotor harmonics

</div>

<div>

## Spectrogram Comparison

<img src="./assets/spectrogram_comparison.png" class="w-full rounded" />

<div class="text-sm mt-2">
<span class="text-red-500 font-bold">Left:</span> Low-SNR — drone harmonics clearly visible below 1 kHz<br>
<span class="text-blue-500 font-bold">Right:</span> High-SNR — speech energy masks rotor harmonics
</div>

</div>

</div>

---

# Answer: Moderate Degradation at High-SNR

<div class="grid grid-cols-2 gap-8">

<div>

## Key Finding (Normalized Audio)

<br>

After correcting for audio level differences:

| Model | Low-SNR MSE | High-SNR MSE | Ratio |
|-------|-------------|--------------|-------|
| **SimpleConv** | **6.8** | **7.9** | **1.2×** |
| DCUNet-RPS | 3.1 | 16.4 | 5.3× |
| DCCRN-RPS | 2.6 | 10.0 | 3.8× |

<br>

**SimpleConv is SNR-robust** — only 1.2× degradation at high SNR.

**Encoder-based models** (DCUNet, DCCRN) degrade more significantly.

</div>

<div>

## MSE Comparison

<img src="./assets/mse_comparison_v2.png" class="w-full rounded" />

<div class="text-sm mt-2 text-gray-500">
Purple bars include one outlier sample (t≈38s) with drastically different acoustic conditions.
</div>

</div>

</div>

---

# Per-Sample Analysis

<img src="./assets/mse_per_sample_v3.png" class="w-full rounded" />

<br>

**Observation**: Most high-SNR samples cluster near the low-SNR average. SimpleConv (left) shows the tightest distribution around its low-SNR baseline, confirming its robustness.

---

# Outlier Investigation

<div class="grid grid-cols-[2fr_1fr] gap-4 h-full">

<div class="flex flex-col items-center">

<img src="./assets/outlier_full.png" class="w-full rounded" />

</div>

<div class="flex flex-col justify-center gap-4 text-sm">

<div>

**Ordinary** (t=16.2s): MSE = 4.7 — predictions track GT well

</div>

<audio controls class="w-full">
  <source src="./assets/ordinary_audio.wav" type="audio/wav">
</audio>

<div class="mt-2">

**Outlier** (t=38.6s): MSE = 276 — GT drops to 0 at t≈8s (drone landing), but model predicts steady ~80 Hz

</div>

<audio controls class="w-full">
  <source src="./assets/outlier_audio.wav" type="audio/wav">
</audio>

<div class="text-xs text-gray-500 mt-2">

**Implication**: RPS prediction reliability varies with recording conditions, not just SNR. Models need awareness of flight state.

</div>

</div>

</div>


---


# Now, the elephant in the room

Bad news:
- I did not manage to make proper progress before my return to the UK
- Turns out, it is hard to do meaningful science while working full-time and traveling around at the same time!

Good news:
- I guess my write-up transfer deadline is right around Christmas now (1.5 months later)

How much time is left and what can realistically be done in this time?

---

# What the Supervisor Actually Said

<div class="text-center">

<div class="text-2xl font-bold text-red-500 mb-4">

"Stop making plans. Start getting publishable results."

</div>

</div>

<div class="grid grid-cols-2 gap-8">

<div>

### The Feedback

- Annoyed that months have been spent on **planning** with **no paper**
- Current priority: **beat SOTA on some domain ASAP**
- Timeline: **1.5–2 months** if truly focused
- **Submit a paper first.** Then think about MIMII, thesis framing, or transfer.

</div>

<div>

### What This Means

- MIMII / cross-domain (C3) is **secondary**
- The 3-bet portfolio is **distraction** — pick one
- "Plan D" benchmark study is **not a paper**
- We need **one result that beats a number**, not a theory

</div>

</div>

---

# The One Thing That Matters

## What is our current best shot at beating SOTA?

<div class="grid grid-cols-2 gap-8">

<div>

### What we have now

- **DREGON-LM** dataset (real drone recordings + mixed speech)
- **RPS-conditioned SE models** (DCUNet-RPS, DCCRN-RPS, SimpleConv-RPS)
- **Oracle RPS** gives improvement over blind baselines
- **Paper 1** published on DN-LM (band-split RoPE Transformer)

### The gap

- Current RPS models are **not yet beating SOTA** on DREGON-LM
- Pseudo-RPS / multi-task are **unproven**
- We have **no submission-ready figure**

</div>

<div class="flex flex-col justify-center">

<div class="p-4 bg-red-100 rounded-lg border-2 border-red-500">
<div class="font-bold text-red-800 text-lg">Brutal truth</div>
<div class="text-red-700 mt-2">
The "current approach" the supervisor refers to is likely <strong>RPS-conditioned SE</strong> (Paper 2 direction). We need to make it <strong>actually work</strong> and <strong>beat a number</strong> on DREGON-LM. That is the domain. Not MIMII. Not theory.
</div>
</div>

<div class="text-sm text-gray-500 mt-4">
Novelty claim: "RPS conditioning improves SE under harmonic noise" — but only if the number is better.
</div>

</div>

</div>

---

# The 6-Week Sprint (May – mid-June)

<div class="grid grid-cols-2 gap-8">

<div>

### Goal: One result that beats SOTA on DREGON-LM

**Week 1–2 (Now – May 19)**
- Audit current best model on DREGON-LM: what is the SI-SDR?
- Run full eval against blind SOTA baseline (TF-GridNet? DPTNet?)
- Identify the gap: where exactly do we lose?

**Week 3–4 (May 19 – June 2)**
- Fix the gap. Options:
  - Better RPS fusion (concat vs FiLM vs cross-attn)
  - Stronger backbone (replace DCCRN/DCUNet with TF-GridNet + RPS)
  - Training tricks: longer training, better augmentation, loss tuning
- Run 2–3 promising variants

**Week 5–6 (June 2 – June 16)**
- Evaluate winner at −30, −20, −10, 0 dB
- Ablations: RPS vs no RPS, oracle vs pseudo
- Generate paper-ready figures (SI-SDR, STOI, PESQ curves)

</div>

<div class="flex flex-col justify-center gap-6">

<div class="p-4 bg-yellow-100 rounded-lg">
<div class="font-bold text-yellow-800">Constraint</div>
<div class="text-sm text-yellow-700 mt-1">
Only one direction at a time. No parallel bets. If Week 2 shows the gap is >3 dB SI-SDR, we need a bigger architecture change. If gap is <1 dB, we need training scale.
</div>
</div>

<div class="p-4 bg-blue-100 rounded-lg">
<div class="font-bold text-blue-800">Kill criterion for this sprint</div>
<div class="text-sm text-blue-700 mt-1">
If by Week 4 we have not beaten the blind SOTA by ≥0.5 dB SI-SDR at any SNR → escalate to supervisor immediately. Do not spend Week 5–6 polishing a loss.
</div>
</div>

<div class="text-center text-4xl font-bold text-red-500">
6 weeks
</div>
<div class="text-center text-sm text-gray-500">
Not 30. Not 8. 6.
</div>

</div>

</div>

---

# What Happens After the Sprint

<div class="grid grid-cols-2 gap-8">

<div>

### If we have a win (Week 7–10, mid-June – mid-July)

- Write paper draft (1 week with AI)
- Run cross-dataset validation on DN-LM (1 week)
- Internal review + supervisor sign-off
- **Submit to Interspeech / TASLP / similar** (check deadlines)

### If we do not have a win

- **Escalate to supervisor immediately** — do not hide
- Options: pivot architecture, add data, or accept Plan D
- Supervisor explicitly said 1.5–2 months — that means **8 weeks max** before a hard conversation

</div>

<div class="flex flex-col justify-center">

```mermaid
graph TD
  A[Week 6: Result?] -->|SI-SDR ↑ >0.5dB| B[Write paper]
  A -->|No improvement| C[Escalate to supervisor]
  B --> D[Submit by July]
  C --> E[Hard conversation:<br>what is missing?]
  E --> F[Architecture pivot?]
  E --> G[Accept MPhil?]
```

<div class="text-sm text-gray-500 mt-4">
The supervisor's 1.5–2 month estimate includes the possibility that it does not work. That conversation must happen by early July, not December.
</div>

</div>

</div>

---

# Immediate Next Actions (This Week)

1. **Today**: Run full eval of best current RPS model vs best blind baseline on DREGON-LM. Get the exact numbers.
2. **By May 9**: Know the gap. Is it 0.5 dB or 5 dB? This determines Week 3–4 strategy.
3. **By May 12**: Have 2 training runs queued: (a) stronger backbone + RPS, (b) longer training of current best.
4. **By May 19**: Know which direction (architecture vs scale) moves the needle.

<br>

<div class="text-center text-xl font-bold text-red-500">

No more planning slides. Only results slides from now on.

</div>
