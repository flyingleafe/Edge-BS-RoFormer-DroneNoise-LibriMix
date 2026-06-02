---
theme: default
background: https://source.unsplash.com/1920x1080/?technology
title: RPS Prediction Progress
info: |
  Three reports: classical baselines, architecture sweep, DREGON-LM-V2 cross-eval.
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
mdc: true
---

# RPS Prediction from Drone Audio
## Three experiments, one story

Dmitrii Mukhutdinov — June 2026`

---

# Comparison with classical Methods

PYIN, cepstral analysis, HPS, matched-filter bank, NMF

<div class="mt-4">
  <img src="./assets/classical_vs_neural.png" class="h-80 mx-auto rounded shadow" />
</div>

<v-click>

**Result:** All classical methods fail. SimpleConv (blue) tracks ground truth; the rest are noise.

</v-click>

---

# Second Attempt: Architecture Sweep

10 SimpleConv variants. Same data, same training.

<div class="grid grid-cols-2 gap-4 mt-4">
  <div>
    <img src="./assets/leaderboard.png" class="rounded shadow" />
  </div>
  <div>
    <img src="./assets/pareto.png" class="rounded shadow" />
  </div>
</div>

<v-click>

**Finding:** BiGRU temporal head is the single most important change. BiGRU-v2 dominates the Pareto frontier.

Overall, using better architectures is obviously fruitful.

</v-click>

---

# What the Improvement Looks Like

One random DREGON-LM valid sample (8 s, −15 dB SNR)

<div class="mt-2">
  <img src="./assets/sample_comparison.png" class="h-90 mx-auto rounded shadow" />
</div>

<v-click>

**Top:** noisy mixture spectrogram. **Middle:** SimpleConv baseline. **Bottom:** BiGRU-v2.

</v-click>

---

# Third Attempt: Build a Harder Dataset

DREGON-LM-V1 was too easy:
- Train/validation overlap (same recordings, even though different chunks of those)
- 1-second clips (≈32 STFT frames)
- Only one microphone channel used -- **maybe models only learned one microphone position**

DREGON-LM-V2 is harder:
- Zero recording overlap
- 3-second clips (≈94 frames)
- All 8 microphone channels
- Also, **added 20% of synthetic constant drone noise** obtained via adding individual motor recordings from DREGON
  - The idea was to push the model more to pitch tracking away from pattern-matching.

---

# Evaluating on Old Validation

<div class="grid grid-cols-2 gap-4 mt-4 h-full">
  <div>
    <img src="./assets/sample_comparison.png" class="rounded shadow h-full object-contain" />
  </div>
  <div>
    <img src="./assets/sample_comparison_v3.png" class="rounded shadow h-full object-contain" />
  </div>
</div>

---

# Evaluating on New Validation

<div class="grid grid-cols-2 gap-4 mt-4 h-full">
  <div>
    <img src="./assets/sample_comparison_v2_old.png" class="rounded shadow h-full object-contain" />
  </div>
  <div>
    <img src="./assets/sample_comparison_v2_v3.png" class="rounded shadow h-full object-contain" />
  </div>
</div>

---

# Cross-Evaluation Shock

Every model tested on both validation sets

<div class="mt-4">
  <img src="./assets/cross_eval.png" class="h-72 mx-auto rounded shadow" />
</div>

<v-click>

Old models: stellar on V1, collapse on V2. V3 models: worse on V1, robust on V2.

</v-click>

---

# Degradation Factors

<div class="mt-4">
  <img src="./assets/degradation.png" class="h-80 mx-auto rounded shadow" />
</div>

<v-click>

Old checkpoints: **63–123×** degradation. V3 checkpoints: **2.2–4.7×**.

</v-click>

---

# Does the Model Know Which Rotor Is Which?

PIT-MSE vs standard MSE on V2 valid

<div class="mt-4">
  <img src="./assets/pit_gap.png" class="h-80 mx-auto rounded shadow" />
</div>

<v-click>

SimpleConv: 38% gap (no temporal head = no stable ordering). BiGRU-v2: 4% gap.

</v-click>

---

# Conclusion

Expanding the dataset (more samples, more channels, synthetic motor combos) **did not yield generalization** - models start failing to train.

The V2 validation "gains" seem to be from predicting constant speeds better — a shortcut learned from 20% synthetic steady-state clips, which otherwise seem to collapse the training rather than regularizing it.

Today: will remove constant steady-state clips from training / validation sets and re-run; the goal is to observe generalization across channels on free-flight recordings.
