# src/plots — all plotting (import name `plots`)

Moved from `src/utils/plots` (imports `utils.plots.X` → `plots.X`) and
ported to `tdseries` during the 2026-07 refactor. The `make-plot` CLI entry
point lives in `cli.py`; the PLOT_TYPES registry survives.

## Architecture

- `timeframe/` — the layout engine. `plot_timeframe(frame, tracks=[...])`
  renders a stack of time-aligned subplots from a `td.Frame`; each track is
  a frame-entry **name**, a raw `td.Series`, or a **`PlotTrack`**.
- **`PlotTrack`** (`timeframe/registry.py`): `series` + optional explicit
  `renderer` key + `hints` dict (`title`, `freqs`, `rps_pred`,
  `freq_max_hz`, `kind`). This replaces the old series-level `plot.*` tags —
  plot hints do NOT live in the data model. `make_spectrogram_series` /
  `make_salience_series` return PlotTracks (`.series` for the data).
- Renderer dispatch: explicit `renderer` first, else by time-index type —
  GridIndex 1-D / (mic|channel, time) → `audio`, other 2-D grid →
  `waveform` (overlaid lines; heatmaps need an explicit renderer/hint),
  StampIndex → `rps`, SpanIndex → `spans`.
- `rps_prediction/` — high-level comparison plots (sample, salience, slide,
  full-sequence). RPS plots PIT-align predictions to GT via
  `tasks.rps_prediction.align_rps_to_gt` before drawing.

## Gotchas

- `make_salience_series(frame_sr=...)` requires an **exact rational** rate
  (`Fraction(16000, 512)` / `(16000, 512)`) — float `31.25` is rejected.
- `build_salience_frame` is gone → `build_salience_tracks(...)` returns a
  dict of PlotTracks/Series (PlotTracks cannot nest inside a `td.Frame`).
- `ROTOR_COLORS` lives in `timeframe/renderers.py` — import it, don't
  redefine it in report scripts.
