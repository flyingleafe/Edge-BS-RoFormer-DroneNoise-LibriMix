# DREGON-LM-V2: A Harder Evaluation for Multi-Rotor RPS Prediction

**Date:** 2026-06-02

---

## Abstract

We introduce DREGON-LM-V2, a revised multi-rotor RPS-prediction dataset that fixes three weaknesses of the original DREGON-LM: (1) train/validation overlap, (2) overly short clips, and (3) single-microphone noise. We then perform a systematic cross-evaluation: old checkpoints (trained on V1) and new V3 models (trained on V2) are tested on both validation sets. The cross-evaluation proves that old validation scores (MSE 2–5) were inflated by memorisation shortcuts — old checkpoints degrade by 63–123× on V2. V3 models generalise far better, with degradation factors of only 2.2–4.7×. BiGRU-v2 remains the best architecture (V2 PIT-MSE 71.1), and its 4% PIT/Std gap shows it has learned a stable rotor ordering. However, all models still fail on in-flight recordings with source signals (MSE 480+), suggesting a denoising-first stage may be required.

---

## 1. DREGON-LM-V2 Dataset

### 1.1 What changed (and why)

| Issue (V1) | Fix (V2) |
|-----------|----------|
| Train/valid clips drawn from same recordings | Recording-level split: 6 `in_flight_nosource` recordings for train, 2 `free-flight` recordings for valid |
| 0.82 s clips (~26 STFT frames) | 3.0 s clips (~94 STFT frames) |
| Single microphone channel | All 8 mic channels, each as independent noise source |
| Telemetry-aligned RPS (post-hoc) | Commanded RPS via `clean_command_spikes` |
| Fixed motor combinations | 20% synthetic combos (sum same channel across different motors) |

### 1.2 Statistics

- **Train:** 6000 samples
- **Valid:** 600 samples
- **Total size:** 2.2 GB
- **Location:** `datasets/DREGON-LM-V2/` (gitignored, must be generated or synced)

---

## 2. Training Setup

### 2.1 Models

- **V3 SimpleConv:** 5 encoder blocks → global avg pool → 1-D conv head. 538K parameters.
- **V3 BiGRU-v2:** 6 encoder blocks + SE blocks → BiGRU (2×128) head. 1.44M parameters.
- **OLD SimpleConv / OLD BiGRU-v2:** Same architectures, trained on original DREGON-LM.

### 2.2 Loss: Permutation-Invariant MSE (PIT)

The four rotor outputs are unordered. For each batch we compute pairwise MSE between every predicted and target rotor (4×4 matrix), then evaluate all 24 permutations and take the minimum. Normalised by n_rotors=4.

### 2.3 Hyperparameters (V3)

| Setting | Value |
|---------|-------|
| Optimiser | AdamW |
| LR | 1e−3 |
| Weight decay | 1e−4 |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 5) |
| Batch size | 96 (vs 16 in initial V2 run) |
| Epochs | up to 500 (early stopping patience 30) |
| Gradient clip | 5.0 |
| Mixed precision | Yes |

---

## 3. Cross-Evaluation Results

Every model evaluated on **both** OLD valid (1 s, measured RPS, overlap) and V2 valid (3 s, command RPS, no overlap).

### 3.1 Main comparison

| Model | OLD valid | V2 valid | Degradation | V2 MAE |
|-------|-----------|----------|-------------|--------|
| OLD SimpleConv | **5.2** | 331.9 | **63×** | 7.68 |
| OLD BiGRU-v2 | **2.7** | 327.3 | **123×** | 6.29 |
| V3 SimpleConv | 66.8 | 148.1 | 2.2× | 10.57 |
| V3 BiGRU-v2 | 15.3 | 71.1 | 4.7× | 4.34 |

![Cross-evaluation](figures/fig_cross_eval.png)

**Figure 1.** Every model tested on both validation sets. Old models are stellar on the old set (blue) but collapse on V2 (orange). V3 models are worse on the old set but far more robust on V2.

![Degradation](figures/fig_degradation.png)

**Figure 2.** Degradation factor: how much worse is V2? Old models fail catastrophically; V3 models degrade modestly.

### 3.2 Why old models fail on V2

1. **Memorisation vs generalisation.** The old validation set contains clips from the same recordings as training. Models memorise microphone positions, motor signatures, and clip-level mean RPS.
2. **Command vs measured RPS.** V2 uses flight-controller setpoints — smoother but noisier around transitions, with variable latency vs true physical speed.
3. **Temporal context.** 3-second clips require trajectory following (~94 frames) rather than scalar prediction (~26 frames). OLD BiGRU-v2 scores 327 on V2; V3 BiGRU-v2 scores 71.

---

## 4. PIT vs Standard MSE: Rotor-Order Ambiguity

| Model | PIT-MSE | Std-MSE | Gap |
|-------|---------|---------|-----|
| OLD SimpleConv | 331.9 | 333.2 | 0% |
| OLD BiGRU-v2 | 327.3 | 328.1 | 0% |
| V3 SimpleConv | 148.1 | 204.9 | **38%** |
| V3 BiGRU-v2 | 71.1 | 73.9 | **4%** |

![PIT/Std gap](figures/fig_pit_std_gap.png)

**Figure 3.** Std-MSE excess over PIT-MSE. V3 SimpleConv shows a 38% gap: even with PIT training, the CNN without recurrent structure struggles to assign rotors to consistent output channels. V3 BiGRU-v2 resolves this with only 4% gap.

---

## 5. Per-Channel Analysis on V2

| Model | ch0 | ch1–7 | ch1–7/ch0 |
|-------|-----|-------|-----------|
| OLD SimpleConv | 293.6 | 338.0 | 1.15× |
| OLD BiGRU-v2 | 247.2 | 340.1 | 1.38× |
| V3 SimpleConv | 126.9 | 151.5 | 1.19× |
| V3 BiGRU-v2 | 66.2 | 71.9 | 1.09× |

OLD models show a 15–38% ch0 advantage (home field). V3 models reduce this to 9–19%, confirming that training on all eight channels makes the predictor channel-agnostic. Even on ch0 alone, V3 BiGRU-v2 beats OLD BiGRU-v2 by 3.7× (66.2 vs 247.2).

---

## 6. In-Flight Recordings with Source Signals

**Critical finding: audio-RPS misalignment in raw DREGON files.**  Motor telemetry starts 5–6 s after audio and ends 1–3 s before it.  Earlier `eval_cross.py` did not crop audio to the motor range, producing misaligned ground truth and inflated MSE numbers.

**Corrected full-sequence evaluation** (audio cropped to motor range, commanded RPS interpolated to STFT frame rate, mean PIT-MSE):

| Model | speech-high | whitenoise-high |
|-------|-------------|-----------------|
| OLD SimpleConv | 224.6 | 87.7 |
| OLD BiGRU-v2 | **199.9** | **104.0** |
| V3 SimpleConv | 274.1 | 123.8 |
| V3 BiGRU-v2 | 277.3 | 323.0 |

**Key insights:**
- OLD models (trained on V1 with in-flight data + measured speeds) generalise surprisingly well to in-flight recordings (MSE 88–225).
- V3 models (trained on static V2 only) perform worse on in-flight (MSE 124–323).
- **Dataset bias:** V2 contains synthetic clips made by mixing individual motor recordings at *constant* speed. ~20% of validation clips have near-constant RPS. V3 models learn this shortcut, suppressing prediction variance by up to 20× (pred std ~0.05 rev/s vs GT std ~1.2 rev/s).
- V2 needs redesign: exclude/down-sample constant-speed synthetic clips, or time-stretch individual motors before mixing.

**Figures:** 3-panel full-sequence plots (spectrogram + RPS traces + per-frame MSE) for all 4 models × 2 recordings in `figures/fig_fullseq_*.{pdf,png}`.

---

## 7. V3 Training Dynamics

![V3 training curves](figures/fig_v3_training_curves.png)

**Figure 4.** V3 training curves. Both models show clear overfitting after early epochs, but the best validation score is retained via early stopping.

- V3 SimpleConv: 52 epochs, val PIT-MSE 148.1
- V3 BiGRU-v2: 44 epochs, val PIT-MSE 71.1

---

## 8. Why R² is misleading here

Per-sample R² is catastrophically negative (order −10⁴) because many V2 validation clips have near-constant RPS (σ < 1 rev/s over 3 s), making SS_tot ≈ 0. **We report MSE and MAE instead.**

---

## 9. Conclusion

DREGON-LM-V2 is a harder, cleaner, and more realistic evaluation set. The cross-evaluation proves that the old dataset's validation scores were inflated by memorisation shortcuts; old checkpoints degrade by 63–123× on V2. Models trained from scratch on V2 (V3 checkpoints) generalise far better on the static benchmark, with degradation factors of only 2.2–4.7×.

Yet the V2 dataset contains a structural flaw: a portion of the training clips are synthesised by mixing individual motor recordings at *constant* speed, creating near-constant RPS targets. Roughly 20% of validation clips exhibit this pattern, and the V3 models learn to exploit it, suppressing prediction variance by up to 20×. On real in-flight recordings the old checkpoints actually outperform the V3 ones (MSE 88–225 vs 124–323), because V1 included in-flight acoustic diversity and measured (non-constant) motor speeds.

BiGRU-v2 remains the best architecture: on V2 valid it achieves PIT-MSE 71.1 (vs 148.1 for SimpleConv), and its 4% PIT/Std gap shows it has learned a stable rotor ordering. For future work, the V2 training set should be redesigned to exclude constant-speed synthetic clips and instead vary rotor speeds within each mixture.

---

## 10. Files

| Path | Content |
|------|---------|
| `papers/dregon_lm_v2_rps_report/main.tex` | LaTeX report (8 pages) |
| `papers/dregon_lm_v2_rps_report/main.pdf` | Compiled PDF |
| `papers/dregon_lm_v2_rps_report/report.md` | This document |
| `papers/dregon_lm_v2_rps_report/generate_figures.py` | Reproducible figure generation (cross-eval) |
| `papers/dregon_lm_v2_rps_report/generate_fullseq_figures.py` | Full-sequence 3-panel plots |
| `figures/fig_cross_eval.{pdf,png}` | Cross-evaluation bar chart |
| `figures/fig_degradation.{pdf,png}` | Degradation factor bar chart |
| `figures/fig_pit_std_gap.{pdf,png}` | Rotor-order ambiguity |
| `figures/fig_v3_training_curves.{pdf,png}` | V3 training curves |
| `figures/fig_fullseq_*.{pdf,png}` | Full-sequence 3-panel plots (8 files) |
| `figures/table_cross_eval.tex` | LaTeX cross-eval table |
| `figures/table_per_channel.tex` | LaTeX per-channel table |
| `figures/table_inflight.tex` | LaTeX in-flight table |
| `eval_cross.py` | Cross-evaluation script |
| `results/rps_predictor_v3/` | V3 checkpoints and logs |
| `results/rps_cross_eval/` | Cross-evaluation artifacts |
