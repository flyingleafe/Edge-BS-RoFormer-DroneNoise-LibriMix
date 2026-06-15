# Experiment: narrow-band input + fine super-resolution salience output

**Goal / hypothesis:** Vindicate the salience-map RPS method by fixing the failure
mode we observed — the 4 rotor trajectories **collapse to their mean** because the
salience grid can't separate near-unison rotors. Concentrate the representation in
the rotor band and emit salience on a much **finer output grid** than the input
CQT can physically resolve (frequency super-resolution), so the tracker can split
rotors that sit a fraction of a Hz apart.

**Status:** training (GPU) as of 2026-06-15.
**Branch:** `salience-map-experiments` (pushed to origin). Code commit `9b40519`.
**Builds on:** [[salience-baselines-dregon-v4-report.md]] (the v4 baselines + eval/plot tooling).

---

## Why (data motivation — DREGON-LM-V4, all valid + 1000 train)

- **Band:** rotor fundamentals are tight — p1–p99 = **69–89 Hz**, mean 80.5±5.1,
  hard ceiling 90 Hz. Steady state is barely **one octave**; the old grids
  (multif0 32.7→2068 Hz, basic_pitch 27.5→4350 Hz) waste almost all bins.
- **Clustering (the bug):** instantaneous min gap between two rotors — median
  **0.88 Hz**, p25 0.37 Hz; **55% of frames have two rotors <1 Hz apart**, 32%
  <0.5 Hz, 18% <0.25 Hz. Old bin spacing ≈0.9 Hz @80 Hz → rotors share a bin →
  tracker merges → "collapse to the mean."
- **Temporal:** RPS is quasi-stationary (50% of modulation power <0.5 Hz, 90%
  <3.4 Hz; ~1 Hz movement per 0.5–1 s) — so a longer analysis window does NOT
  smear the rotor much.
- **The wall:** **train clips are 1 s** (valid is 8 s). Δf·Δt≳1 ⇒ a single CQT
  frame can't resolve <~1 Hz from 1 s. So *input* resolution is capped; the bet
  is that a learned **super-resolution output head** (peak shape + phase + the
  harmonic channels h·Δf) recovers finer localization than the input bins.

---

## What the experiment changes (design)

Decouple the salience **output** grid from the CQT **input** grid:

- **Input (narrow):** multif0 HCQT `fmin=55, n_octaves=1, over_sample=10,
  harmonics=[1,2,3,4]` (120 log bins, 55→110 Hz). basic_pitch contour CQT
  `bp_fmin=55, bins_per_semitone=4, n_contour_semitones=12` (48 log bins).
- **Output (fine, linear):** `FreqSuperResHead` resamples (fixed log→linear
  interp matrix) to a **linear 55–110 Hz grid, 360 bins (~0.153 Hz/bin)**, then a
  `(5,1)`-conv stack learns to sharpen. Trained end-to-end with BCE on a blurred
  target built on this grid; Hungarian tracker reads the same grid (max-jump auto-
  scaled to ~1.5 Hz). Helpers (`rps_to_salience`/`salience_to_rps_segmented`)
  gained an explicit `freqs=` arg; `linear_freq_grid()` added.

Key files: `src/models/salience_rps.py` (`FreqSuperResHead`, `out_freqs`,
`output_freqs()`), `src/models/multif0/utils.py`, `src/models/basic_pitch/{cqt,model}.py`
(narrow contour params), `train_rps_predictor.py` (CLI flags + `salience_cfg`).

---

## Run config (GPU)

```bash
# multif0
python train_rps_predictor.py --model multif0_salience \
  --data_root datasets/DREGON-LM-V4 --device cuda:0 --epochs 200 --patience 15 \
  --hcqt_fmin 55 --hcqt_n_octaves 1 --hcqt_over_sample 10 --hcqt_harmonics 1 2 3 4 \
  --superres_out --out_fmin 55 --out_fmax 110 --out_bins 360 --salience_blur_bins 2 \
  --save_path results/rps_baselines_v4/multif0_salience_narrow_sr

# basic_pitch
python train_rps_predictor.py --model basic_pitch_salience \
  --data_root datasets/DREGON-LM-V4 --device cuda:0 --epochs 200 --patience 15 \
  --bp_fmin 55 --bp_bins_per_semitone 4 --bp_n_contour_semitones 12 \
  --superres_out --out_fmin 55 --out_fmax 110 --out_bins 360 --salience_blur_bins 2 \
  --save_path results/rps_baselines_v4/basic_pitch_narrow_sr
```

Smoke-verified on CPU (2 epochs, 40-sample subset): both train end-to-end, BCE
decreases (multif0 0.659→0.554, basic_pitch 0.737→0.664), Hungarian eval runs.

---

## What to measure (for the report)

- **Headline:** RMSE/MAE/R² vs the baselines in
  `results/dregon_v4_eval/salience_baselines_final_valid.json`
  (multif0 RMSE 6.30 / R² 0.19; basic_pitch RMSE 23.2 / R² −16.2). Does
  narrow+super-res beat them — esp. does multif0 separate rotors that previously
  merged?
- **The real question:** do the 4 predicted trajectories stop collapsing to the
  mean? Plot per-sample with the `"salience"` renderer (it now reads the fine
  output grid) — look for 4 distinct tracked lines where GT rotors are 0.3–1 Hz
  apart.
- **Honesty check (critical):** super-res sharpness is a learned prior — verify it
  is *accurate, not just confident*. Measure sub-bin RPS error + calibration
  against GT (GT is exact in DREGON-LM). Guard against hallucinated precision.
- Floor: ~18% of frames have rotors <0.25 Hz apart — unresolvable in 8 s; expect
  residual merges there (segmented-PIT tolerates them).

## Caveats / gotchas
- `--hcqt_harmonics 1 2 3 4` is **required** (auto-derivation overshoots to ~57
  harmonics at fmin=55, n_octaves=1).
- 1 s train clips: nnAudio warns (low-octave CQT kernel > clip) — the time-
  bandwidth tightness. Longer training clips are the orthogonal lever if results
  disappoint.
- basic_pitch carries ~9k unused note/onset params (dead weight from the port);
  only the contour branch + head run. Harmless; prunable later.
- Per-epoch Hungarian eval is the CPU bottleneck (~8 min on full valid).
