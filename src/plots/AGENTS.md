# src/plots — all plotting (import name `plots`)

Moved from `src/utils/plots` (imports `utils.plots.X` → `plots.X`) and
ported to `tdseries` during the 2026-07 refactor. The `make-plot` CLI entry
point lives in `cli.py`; the PLOT_TYPES registry survives.

## Front door: `plots.dwym`

`plots.dwym(obj, **hints) -> DwymResult` is the one call for 90% of
figures. `obj` is a `td.Frame`, a list of Frames, or a `{label: Frame}`
dict. Frame-level dispatch on the coerced entry names:

| `result.route` | Trigger → figure |
|---|---|
| `se` | >= 2 of `mixture`/`target`/`enhanced` → spectrogram rows, shared time axis |
| `salience` | a `salience` entry → salience heatmap (+ spectrogram, + GT `rps` row) |
| `noise_gen` | `audio`+`generated` in one frame, or a dict of two bare-audio frames → real-vs-generated spectrogram grid |
| `rps` | `audio` + `rps` (and/or `rps_pred`, PIT-aligned to GT) → spectrogram + RPS rows |
| `audio` | only temporal entry is `audio` → spectrogram + waveform |
| `timeframe` | anything else → `plot_timeframe`'s per-track dispatch |

Multi-frame input renders one aligned figure with a row block per label
(route `multi:<r>`) when all frames dispatch the same; heterogeneous dicts
get one figure per frame (`mixed`). Hints: `renderer=<route>` forces a
path; `<canonical>="entry"` remaps entry names (see coercion below);
`fmax`/`freqs` shape the spectrogram/salience tracks; everything else
flows into `plot_timeframe` and the `PlotTrack.hints` channel.

`DwymResult` holds `.figures` + `.audio` (`{entry: (mono, sr)}`) + `.route`.
In IPython it displays figures AND one `IPython.display.Audio` player per
audio entry (`_ipython_display_`; figures are closed after building so the
inline backend does not double-render). Outside notebooks: `.figure`,
`.figures`, `.save(path)`.

## Coercion (`plots/coerce.py`)

`coerce_frame(frame, **overrides)` canonicalizes raw dataset frames before
dispatch (dwym calls it). Alias tables → canonical names:

- `rps` ← `motors_command`, `motors_measured` (DREGON raw, in
  `PUBLISHED_RPS_KEYS` order), `motor_rps`, `rotor_rps`, `rotor_speed`,
  `motor_speed` (Michael's RAW RPM block — last on purpose; the calibrated
  track on Michael's frames is already named `rps`)
- `audio` ← `waveform`, `wav`, `mix`; plus a sole-waveform fallback (one
  uniform >= 4 kHz series with mic/channel+time dims, only when no
  `audio`/`mixture` exists)
- `rps_pred` ← `pred_rps`, `predicted_rps`, `rps_prediction`
- `salience` ← `salience_map`; `generated` ← `generated_audio`, `gen_audio`

Every applied mapping emits one `warnings.warn` line naming it; explicit
overrides (`coerce_frame(frame, rps="motor_speed")` or the same kwargs on
`dwym`) are silent and win (a clobbered canonical entry moves to
`<name>_orig`). Unknown extra timeseries are KEPT as additional tracks.

## Exploration (`plots/explore.py`)

Notebook data-exploration primitives (Phase 6). All four accept a dload
dataset **name**, a map-style dataset, or any iterable of `td.Frame`s:

- `explore.datasets(sizes=False)` — the `dload.lock` catalog as a pandas
  DataFrame (`sizes=True` fetches manifests: network + credentials).
- `explore.meta_table(x, fields=None, limit=32)` — sample `meta` as a
  DataFrame, plus computed `entries` / `duration_s` columns.
- `explore.grid(x, n=12, seed=None, scan=None, fmax=..., audio=...)` —
  reservoir-sampled spectrogram thumbnails with meta captions; returns a
  `DwymResult` (`route="grid"`). A dataset name gets a seeded streaming
  shuffle for shard spread.
- `explore.pick(x, index_or_query)` — one sample by index, meta-id
  substring, or predicate; returned **coerced** (`coerce_frame`), ready for
  `dwym` / `zoo.FrameModel`.

`explore` imports `data_processing` (streams, frames) — allowed: `plots`
sits above `data_processing` in the layer contract. Only the thumbnail grid
layout is new rendering code; everything else delegates.

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
- `audio.py` — shared audio-`Series` helpers (`to_mono`, `sample_rate_of`,
  `first_channel`); also used by `training.val_logging`.
- `se.py` — SE recipes: `extract_se_triple(pred, target)` (the
  mixture/target/output audio triple `val_logging` wraps in `wandb.Audio`)
  and `plot_se_comparison` / `se_comparison_tracks` (spectrogram rows).
- `noise_gen.py` — noise-gen recipes: `extract_noise_gen_pair(pred, target)`
  (real/generated pair for `val_logging`) and
  `plot_noise_gen_comparison` / `noise_gen_comparison_tracks`.
- `rps_prediction/` — high-level comparison plots (sample, salience,
  full-sequence, summary/per-SNR/curves; `slide_comparison` was deleted as
  dead code). RPS plots PIT-align predictions to GT via
  `tasks.rps_prediction.align_rps_to_gt` before drawing.

## Gotchas

- `make_salience_series(frame_sr=...)` requires an **exact rational** rate
  (`Fraction(16000, 512)` / `(16000, 512)`) — float `31.25` is rejected.
- `build_salience_frame` is gone → `build_salience_tracks(...)` returns a
  dict of PlotTracks/Series (PlotTracks cannot nest inside a `td.Frame`).
- `ROTOR_COLORS` lives in `timeframe/renderers.py` — import it, don't
  redefine it in report scripts (`rps_prediction/*` now import it from
  there; the two local copies are gone).
- One spectrogram implementation: `make_spectrogram_series` + the
  `"audio_spectrogram"` renderer. Do not hand-roll `torch.stft` in plot
  code (`sample_comparison._plot_spectrogram` now routes through it).
- Layers: `plots` must not import `training`, `zoo`, or `scripts`.
  `training.val_logging` imports `plots` (not the reverse).
- `plots.dwym` is lazy (`plots/__init__.__getattr__`) and the function
  shadows the submodule attribute after first use — import the module
  internals explicitly via `from plots.dwym import ...` when needed.
