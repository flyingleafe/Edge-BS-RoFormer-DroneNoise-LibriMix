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
- `comb_page.py` + `comb_widget.py` — the **comb explorer**, the one
  non-matplotlib product here: a self-contained interactive HTML page
  (spectrogram + STFT/synchrosqueezed toggle + per-rotor harmonic combs +
  demodulated per-harmonic strips). `comb_page` holds the payload builder and
  the single HTML/JS template; `comb_widget.comb_explorer(frame, t0=, dur=)`
  is the notebook entry point over a plain `td.Frame` (auto-discovers the
  audio entry and every rotor-speed track, ships them all as selectable
  carriers and channels). The file-writing CLI over the same core is
  `scripts/displacement/comb_explorer.py`; the JS is verified by
  `scripts/displacement/verify_page.js`. Docs:
  `scripts/displacement/README.md`. It lives here, not in `data_processing`,
  because it is a figure: `plots` depends on nothing in `data_processing`.

## Gotchas

- `make_salience_series(frame_sr=...)` requires an **exact rational** rate
  (`Fraction(16000, 512)` / `(16000, 512)`) — float `31.25` is rejected.
- `build_salience_frame` is gone → `build_salience_tracks(...)` returns a
  dict of PlotTracks/Series (PlotTracks cannot nest inside a `td.Frame`).
- `ROTOR_COLORS` lives in `timeframe/renderers.py` — import it, don't
  redefine it in report scripts.
