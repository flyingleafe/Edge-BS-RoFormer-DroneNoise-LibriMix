# src/utils/data/ — Aligned time-series containers (fixed-point)

> Full API reference: [`API.md`](./API.md). Read it before constructing or
> manipulating any time‑series container.

A small library for working with audio (and other regular-rate signals)
together with co‑recorded telemetry (RPS, IMU, VAD segments, ...). The
existing ad‑hoc records (`data_processing.dregon.DREGONRecord`,
`data_processing.michaels.MichaelsRecord`) are the prototype that this
abstraction generalises.

## Public surface

```python
from utils.data import (
    TICKS_PER_SECOND,  # int: resolution constant (1e9 = nanoseconds)
    TimeSeries,        # abstract base
    UniformSeries,     # regular sample-rate (audio, video, ...)
    EventSeries,       # sorted point-event timestamps + values
    SegmentSeries,     # half-open interval series (VAD, labels, ...)
    TimeFrame,         # dict-keyed container of aligned tracks
    DomainError,       # slice outside declared interval
    IncompatibleSeriesError,  # mismatched seam / dtype / sr / keys
)
```

All four container types are **frozen dataclasses** with a uniform algebra:

| Method / property | Meaning |
|-------------------|---------|
| `.t_start` / `.t_end` / `.duration` | float seconds (ergonomics) |
| `.t_start_ticks` / `.t_end_ticks` / `.duration_ticks` | exact int64 tick counts |
| `slice(t_a, t_b)` | restrict to `[t_a, t_b)` — accepts float seconds or int ticks |
| `concat(other)` | glue at seam; auto‑aligns `other`'s domain |
| `shift(t_delta)` | O(1); accepts float seconds or int ticks |
| `equal(other)` | exact structural equality (no tolerance) |
| `__len__`, `__getitem__` | row‑wise indexing; returns float seconds |

## Time model

All time is stored as **int64 tick counts** at a fixed `TICKS_PER_SECOND =
1_000_000_000` (nanoseconds). Integer add/subtract/compare are exact and
magnitude‑independent — no floating‑point tolerances, no scaled `atol`.

The **relative‑to‑container** principle is enforced exactly:
- Each series stores data timestamps **relative to its own `t_start_ticks`**.
- Within a `TimeFrame`, each track's `t_start_ticks` is stored **relative to
  the frame's `t_start_ticks`** (the single absolute anchor).
- `shift` and frame re‑basing are exact int additions — no precision loss for
  any delta.

The public API returns **float seconds** (`.t_start`, `.t_end`,
`__getitem__`) for back‑compat with audio loaders and legacy `*Record` types.
For exact round‑trips, use the `*_ticks` accessors. `slice` and constructors
accept either form.

## The one approximation: audio sub‑sample phase

Sample edges lie on a grid of spacing `1/sr`, which is irrational in ticks
(44.1 kHz → 22675.736… ns). The resolution: we never store per‑sample edge
times — only the anchor, sample count, and `sr`. The sub‑sample offset is
stored as a **small relative float** `phase` ∈ [−1, 0] (sample units). Being
bounded by one sample, a float64 stores it to ≈ 1e‑21 s — effectively exact.

For **integer `sr`** (all real audio rates), the cut‑time → sample‑index
mapping uses exact Python‑int `divmod(Δ·sr, TICKS_PER_SECOND)`. The sub‑sample
remainder is carried into the child series' `phase`, so `slice(a,b) ⊕
slice(b,c) == slice(a,c)` is sample‑exact. Non‑integer `sr` uses a float index
with a fixed `1e‑6`‑sample epsilon.

## Design notes (preserved from the float era)

1. **Sample‑index arithmetic > absolute‑time arithmetic.** Still true: the
   grid offset for concat is computed in sample units, not ticks. The math is
   now exact for integer `sr`.

2. **Sub‑sample cuts.** A sample's cell is one bin wide (`1/sr`). When the
   user cuts inside a cell, the sample appears in both halves and is de‑duped
   on concat via the integer offset.

3. **No tolerances.** All `_floats.py` helpers (`tclose`, `t_atol_at`,
   `grid_atol`) are gone. Time equality is `==` on ints. The only surviving
   epsilon is `INDEX_EPSILON = 1e‑6` in `uniform.py`, used only on the
   non‑integer‑`sr` fallback.

4. **Segment identity on concat.** Each segment carries a random 62‑bit `id`.
   Splitting a segment across a cut emits two rows sharing the same `id`;
   concat merges back any pair at the seam with matching `ids`. Now exact
   (`==` on ints at the seam).

## File map

| File | Role |
|------|------|
| `base.py` | abstract `TimeSeries`, `DomainError`, `IncompatibleSeriesError` |
| `_ticks.py` | `TICKS_PER_SECOND`, scalar/array seconds↔ticks conversion |
| `uniform.py` | `UniformSeries` — sample‑grid model with sub‑sample `phase` |
| `event.py` | `EventSeries` — sorted point timestamps + optional values |
| `segment.py` | `SegmentSeries` — half‑open intervals; split‑and‑rejoin via `ids` |
| `frame.py` | `TimeFrame` — dict‑keyed container; column ops + time ops |
| `_floats.py` | **legacy** — kept for reference but unused by the library |

## Tests

`tests/utils/data/` contains Hypothesis property tests:

* `test_uniform.py` — sub‑sample cut invariants, multi‑channel, concat
  alignment, shift round‑trip.  Cut points drawn as exact int ticks.
* `test_event.py` — slice‑concat identity, half‑open routing, shift O(1).
  Assertions are exact (`==`), no `allclose`.
* `test_segment.py` — straddling‑segment split‑and‑rejoin via `ids`,
  unrelated‑segments‑at‑seam not merged.
* `test_frame.py` — column ops (`select`/`drop`/`merge`), domain mismatch
  rejection, slice/concat identity composed across all three track types.

Strategies draw int64 tick anchors (both small and Unix‑magnitude ~1.6e18)
and construct series via `from_ticks` / `from_events` with int arguments to
guarantee exactness.
