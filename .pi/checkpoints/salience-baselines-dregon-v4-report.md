# Salience-map RPS baselines — DREGON-LM-V4 final validation + report assets

**Goal:** One place to find every model, artifact, metric, and plotting helper
needed to write up the salience-map RPS baselines (`multif0_salience`,
`basic_pitch`) on DREGON-LM-V4 for the report.

**Status:** done (eval + viz tooling); ready for report write-up
**Last touched:** 2026-06-15
**Branch / worktree:** `typst-reports-slides-2` (worktree at
`.claude/worktrees/typst-reports-slides-2`)
**Related:** `.pi/checkpoints/salience-map-tracking.md` (how the BCE-salience
baselines were built + GT round-trip analysis).

---

## Models & checkpoints

All under `results/rps_baselines_v4/` (bare `state_dict`s — `torch.load(...,
map_location=...)` then `load_state_dict(..., strict=True)`). They are **only in
the main checkout**, not git worktrees (gitignored). Construction config matters
— it is NOT in the checkpoint, so use exactly:

| Key (report name) | Checkpoint | How to construct | Grid |
|---|---|---|---|
| `multif0_salience` (LateDeep, **headline**) | `multif0_salience/best_multif0_salience.pt` | `LateDeepSalience(n_fft=2048, hop_length=512, num_rotors=4, fmin=32.7)` | C1, **3 harmonics**, 360 bins, 60 bins/oct |
| `basic_pitch` (Bittner contour) | `basic_pitch/best_basic_pitch_salience.pt` | `BasicPitchSalience(n_fft=2048, hop_length=512, num_rotors=4)` | A0, 264 bins, 36 bins/oct |
| `multif0_salience_fastest` (stacked HCQT + fused) | `multif0_salience_fastest/best_multif0_salience.pt` | `LateDeepSalience(..., fmin=27.5, stacked=True, fused_branches=True)` | A0, **4 harmonics**, 360 bins |

**Gotchas (will bite you otherwise):**
- `multif0_salience` was trained at **`fmin=32.7` (C1) → 3 harmonics**, NOT the
  current code default `fmin=27.5` (→ 4 harmonics). Loading with the default
  raises a `size mismatch`. The `_fastest` variant *was* trained at `fmin=27.5`.
- `_fastest` uses the **stacked** single-CQT front-end (`stacked=True`) + fused
  branches (`fused_branches=True`) — a lossy approximation of the per-harmonic
  HCQT, so it does **not** cross-load with the non-stacked checkpoints.
- Other variants exist (`multif0_salience_faster` = fused branches only) but were
  not part of this report eval; `wandb_run_id.txt` sits next to each ckpt.

The salience model classes live in `src/models/salience_rps.py`
(`SalienceRPSPredictor` base, `LateDeepSalience`, `BasicPitchSalience`,
`outputs_salience=True`). RPS↔salience helpers: `src/models/multif0/utils.py`
(`cqt_freq_grid`, `rps_to_salience`, `salience_to_rps_segmented`).

---

## Final validation metrics — `DREGON-LM-V4/valid`

30 clips × 8 channels (channels flattened into the batch via `_flatten_channels`).
PIT-aware eval (`train_rps_predictor.evaluate(..., pit_eval=True)`),
`track_threshold=0.3`. Lower MAE/RMSE = better; R² = within-clip temporal tracking.

| Model | RMSE (Hz) | MAE frame (Hz) | MAE clip (Hz) | R² | R² median |
|---|---|---|---|---|---|
| `multif0_salience` | 6.30 | 3.40 | 2.99 | 0.19 | — |
| `multif0_salience_fastest` | 6.42 | 3.58 | 3.28 | 0.11 | — |
| `basic_pitch` | 23.24 | 16.19 | 15.90 | −16.21 | — |

**Takeaways for the write-up:** LateDeep (`multif0_salience`) tracks the rotor
fundamental tightly (single bright salience band). `basic_pitch` produces a
diffuse salience and mistracks (often locking onto the wrong octave/harmonic) →
large error and negative R². `_fastest` (stacked-HCQT approximation) costs almost
nothing in accuracy — RMSE 6.42 vs 6.30 Hz — so the ~2× faster front-end is
essentially free for tracking quality.

Raw numbers (authoritative): `results/dregon_v4_eval/salience_baselines_final_valid.json`
(keys: `results`, `checkpoints`, `model_configs`, `dataset`, `track_threshold`).

---

## Artifacts

| Artifact | Path |
|---|---|
| Eval metrics (JSON) | `results/dregon_v4_eval/salience_baselines_final_valid.json` |
| Interactive notebook | `notebooks/salience_baselines_dregon_v4.ipynb` (executed, figures embedded) |
| SimpleConv family eval on same set (context) | `results/dregon_v4_eval/eval.json`, `eval_pit.json` |

---

## Plotting code (reuse this — do not re-roll)

Everything dispatches through the generic `TimeFrame` machinery
(`utils.plots.timeframe`).

1. **`"salience"` renderer** — `src/utils/plots/timeframe/renderers.py`:
   - `make_salience_series(salience, *, freqs, frame_sr, t_start=0, rps_pred=None,
     title=None)` → a `UniformSeries` tagged for the renderer.
   - `render_salience(series, context)` — log-freq salience heatmap with GT RPS
     (dotted) and tracked-RPS prediction (solid) overlaid. Registered as
     `"salience"`; style knobs: `salience_vmax` (`1.0` or `"auto"`),
     `salience_colorbar`.

2. **High-level per-sample figure** —
   `src/utils/plots/rps_prediction/salience_comparison.py`:
   - `plot_salience_comparison(sample, models, *, channel=0, device="cpu",
     fmax=4000, track_threshold=0.3, show_rps_row=True, salience_vmax=...)`
     → spectrogram + one salience row per model + an RPS-vs-GT row.
   - Inference helpers (plotting-independent, reusable): `model_salience_series`,
     `model_rps_prediction`, `build_salience_frame`, `select_channel`.
   - `models` is `{display_name: loaded SalienceRPSPredictor}`.
   - `sample` is a `TimeFrame` — load one with
     `utils.plots.rps_prediction.sample_comparison._load_sample(path)`.

Minimal example:
```python
from utils.plots.rps_prediction.sample_comparison import _load_sample
from utils.plots.rps_prediction.salience_comparison import plot_salience_comparison
fig = plot_salience_comparison(_load_sample("datasets/DREGON-LM-V4/valid/sample_00026"),
                               models, channel=0, salience_vmax="auto")
```

---

## Reproduce

```bash
# from the MAIN checkout (datasets/results live here, not in worktrees)
cd <main-checkout>
# exercise this worktree's plotting code by putting its src first on the path:
PYTHONPATH=<worktree>/src python ...      # or run the notebook (its bootstrap cell does this)
```

- **Metrics:** open the notebook → section 3 (loads the cached JSON; set
  `RECOMPUTE=True` to re-run `evaluate()` — slow, Hungarian tracking on CPU,
  ~8 min/model for the non-stacked HCQT).
- **Figures:** notebook sections 4–6 (auto-picks 3 interesting clips by per-model
  error / disagreement; `SAMPLE_IDS` and `CHANNEL` are editable).

**Worktree data gotcha:** `datasets/`, `results/`, `.venv` are gitignored, so a
`git worktree` checkout does NOT contain them — only the main checkout does. The
editable install is a plain `.pth` adding `<main>/src`; prepend `<worktree>/src`
to `sys.path` to use worktree code. The notebook's bootstrap cell handles both
(prepends worktree `src`, `chdir`s to the checkout that has `datasets/`).

## Open / next
- `_faster` variant (fused branches, non-stacked) not yet evaluated on V4 — add
  with `LateDeepSalience(..., fmin=32.7, fused_branches=True)` if the report needs
  the full speed-variant sweep.
- 8-channel metrics here average over all channels; a per-channel or channel-0
  breakdown can be produced from the same loader if the report wants it.
