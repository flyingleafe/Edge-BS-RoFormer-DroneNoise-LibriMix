# Salience Map Tracking & Segment-Based PIT

**Goal:** Reliable RPS trajectory extraction from salience maps (multi-F0 predictions), with segment-based PIT metric for round-trip evaluation and model comparison.

**Status:** in-progress
**Last touched:** 2026-06-12
**Resume on:** local (worktree `multi-pitch-baselines`)

## Done
- Hungarian (optimal linear-sum-assignment) tracking in `src/models/multif0/utils.py::_hungarian_tracking()`
  - Replaces greedy nearest-neighbor; uses `scipy.optimize.linear_sum_assignment`
  - Max-jump rejection (`max_jump_bins=3`) suppresses adjacent-bin jitter
  - Merge detection: frames where `n_peaks < n_active_rotors` → identity-ambiguous
- `salience_to_rps_segmented()` — public API returning `(rps, merge_mask)`
- `segmented_pit_mse()` — PIT-MSE loss that splits timeline at merge points, finds best perm per segment
- `roundtrip_error_segmented()` — GT → salience → tracking → RPS metric
- `_segment_boundaries()` — converts merge frame mask to segment intervals
- `_extract_peaks_per_frame()` — peak extraction for both binary (GT) and soft (model) salience
- `_detect_adjacent_merges()` — optional post-hoc adjacent-bin merge detection (`merge_mode="adjacent"`)
- HCQT frontend (`src/models/frontends/hcqt.py`):
  - Default `sr=16000`, `n_octaves=6` → auto 3 harmonics `[1,2,3]`
  - `input_sr` independent of CQT `sr` (auto-resampling)
  - Auto-derive max harmonics via `sr / 2 / (fmin * 2^n_octaves)`
- MultiF0RPSPredictor (`src/models/multif0/rps_predictor.py`): reads actual `frontend.harmonics` instead of hardcoding 5
- All 6 existing tests pass (`test_multif0.py`, `test_multif0_rps.py`)

## GT Round-Trip Results (DREGON-LM-V4 valid, 16 samples ≥ 33 Hz)

| Tracker | Per-frame PIT | Global PIT |
|---------|--------------|------------|
| Greedy  | 2.78 Hz      | 6.83 Hz    |
| **Hungarian** | **1.19 Hz** | **1.67 Hz** |

3 samples excluded (rotors at 30.7–31.3 Hz, below fmin=32.7).
Irreducible pure-quantization floor: **0.27 Hz** (per-frame PIT directly on salience bins, no tracking).

## DONE — BCE-salience baselines wired into train/eval (branch `salience-rps-baselines`)
Implemented on a fresh branch off `main` (NOT this stale worktree — `main` already
contains both the salience-tracking utils and the basic_pitch port).

- **New module `src/models/salience_rps.py`**: `SalienceRPSPredictor` base (flag
  `outputs_salience=True`, `grid_params`/`num_grid_frames`/`salience_target`/`predict_rps`)
  + `LateDeepSalience` (HCQT 16 kHz, 3 harmonics, 360-bin grid) + `BasicPitchSalience`
  (contour branch, 264-bin grid, native 16 kHz). Registry keys `multif0_salience`,
  `basic_pitch_salience`.
- **forward → (B, n_bins, T) logits**; **predict_rps → (B,4,T_stft)** via
  `sigmoid → salience_to_rps_segmented → F.interpolate to STFT grid` (nan_to_num).
- **utils.py**: `cqt_freq_grid` + `rps_to_salience` + `salience_to_rps_segmented` now
  take explicit `n_bins`/`bins_per_octave` (for the 36-bin/oct basic-pitch grid);
  `rps_to_salience` gained `blur_bins` (triangular freq-axis smoothing for BCE).
- **LateDeep.forward(return_logits=True)** (pre-sigmoid, checkpoint-compatible);
  **basic_pitch `CQTFrontEnd(sr=...)`** parameterized + `BasicPitch.contour_logits()`.
- **train_rps_predictor.py**: dataset precomputes/caches salience targets (no GPU fork
  into workers); train loop branches BCEWithLogitsLoss (auto `pos_weight`) vs pit_mse;
  eval loop branches `predict_rps` (tracking, fp32) then the **unchanged** global-PIT
  metrics; CLI `--salience_blur_bins/--bce_pos_weight/--track_threshold`.
- **Tests**: 6 existing multif0 tests pass; new `test_salience_rps.py` (round-trip on
  both grids, blur, shapes+BCE backward) passes. 1-epoch CPU smoke trains verified
  end-to-end for `multif0_salience`, `basic_pitch_salience`, and a `simple_conv`
  regression (2-tuple path intact).

### Decisions (locked with user)
- "LateFusion" = LateDeep. Paradigm = BCE-salience + tracking eval (soft-centroid
  `multif0_rps` kept untouched as a separate baseline). Headline metrics = existing
  global-PIT `evaluate()` (segmented PIT stays a side diagnostic only).
- **Native 16 kHz everywhere** for comparability with SimpleConv; only the (deferred)
  zero-shot *pretrained* basic-pitch would need 16k→22.05k resampling.

## Pending (next)
1. **Lower fmin** to 16.35 Hz (C0) so hovering-flight rotors (28–32 Hz) are captured
   (HCQT default + utils defaults; re-test round-trip on all 19 V4 valid samples).
2. **Real GPU training run** (multif0_salience, basic_pitch_salience) vs SimpleConv
   baselines on full DREGON-LM; tune `--track_threshold` / `--salience_blur_bins`.
3. **Deferred**: zero-shot pretrained basic-pitch (`from_pretrained` + 16k→22.05k
   resample + freeze; `pretrained=True` currently raises NotImplementedError).
4. **CLI `--frontend`** flag for SimpleConv* (pending #5) — not done (out of scope here).

## State
- Working tree: **dirty** — `src/models/multif0/utils.py`, `src/models/frontends/hcqt.py`, `src/models/multif0/rps_predictor.py` modified.
- No uncommitted binary artifacts.
- Files not in git: `.pi/checkpoints/salience-map-tracking.md`

## Decisions (do not relitigate)
- **Hungarian > greedy**: Greedy creates unnecessary identity swaps when bins are close; Hungarian avoids this via global cost minimization.
- **Merge = n_peaks < n_active**: Fewer peaks than tracked rotors means at least two rotors are indistinguishable — genuine identity ambiguity. Adjacent-bin merges optional (`merge_mode="adjacent"`) but add too many merge points for V4 data.
- **Max-jump = 3 bins**: Rejects assignments where rotor "teleports" > 3 bins between consecutive frames. Uses CQT bin index, not Hz distance, since bins are log-spaced.
- **fmin=32.7 retained** for now — 3 V4 valid samples have rotors below this. Lower to 16.35 Hz as next step.
- **Segment-based PIT, not per-frame**: Per-frame PIT produces discontinuous rotor identities at merge points. Segment-based PIT enforces consistent permutation within each segment while allowing swaps at genuine ambiguity points.

## Open questions
- For model predictions (soft salience), what's a good `threshold` for peak detection? 0.5? Adaptive?
- Should `MultiF0RPSPredictor` output raw logits (for BCE) or be a separate salience-prediction model? Current output is RPS via MLP.
- Do we need a separate `salience_to_rps_segmented` call in the eval loop, or integrate segment-based PIT into the existing `pit_mse_loss`?

## Resume
```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression/.worktrees/multi-pitch-baselines

# 1. Lower fmin to 16.35 Hz
#    In src/models/frontends/hcqt.py: change default fmin=32.7 → 16.35
#    In src/models/multif0/utils.py: change all cqt_freq_grid(fmin=32.7) → fmin=16.35
#    Re-run round-trip test on all 19 samples

# 2. Run existing tests to confirm nothing broken
uv run python -m pytest test_multif0.py test_multif0_rps.py -v

# 3. Verify Hungarian tracking with fmin=16.35
uv run python -c "
import torch, numpy as np, glob, os, soundfile as sf, json, itertools
import models.multif0.utils as mu
# ... (round-trip eval loop from conversation)
"

# 4. Integrate BCE loss — see train_rps_predictor.py lines ~350-450
#    Key insight: GT RPS → rps_to_salience() → binary targets.
#    Model outputs salience logits → BCEWithLogitsLoss.
#    At inference: salience logits → sigmoid → salience_to_rps_segmented().
```
