# `utils.data` — aligned time-series containers (API reference)

A small library for audio together with co‑recorded telemetry (RPS, IMU, VAD).
Four frozen container types with a uniform algebra: `slice`, `concat`, `shift`,
`equal`.

## The tick model

All time is stored as **int64 tick counts**, where

```python
TICKS_PER_SECOND = 1_000_000_000   # nanoseconds
```

Integer add/subtract/compare are exact and magnitude‑independent — no floating‑point
cancellation, no scaled tolerances. Every boundary check is `==` on ints.

The *public* properties `.t_start`, `.t_end`, `.duration`, and `__getitem__` return
**float seconds** for ergonomics (the project's ecosystem uses float seconds: audio
loaders, legacy `*Record` types). For exact round‑trips, use the `*_ticks` accessors.

| Seconds API | Ticks API (exact) |
|-------------|-------------------|
| `.t_start` | `.t_start_ticks` |
| `.t_end` | `.t_end_ticks` |
| `.duration` | `.duration_ticks` |
| — | `from_ticks(…)` / `slice(t_ticks, t_ticks)` |

Constructors and `slice` accept either form: `float` → quantized once via
`round(f * TICKS_PER_SECOND)`; `int` → used directly.

---

## `UniformSeries` — regular‑rate signal (audio, video)

### Stored representation

| Field | Type | Meaning |
|-------|------|---------|
| `samples` | `np.ndarray` | shape `(N, …)`, axis 0 is time |
| `sr` | `float` | sample rate in Hz |
| `t_start` | `int` | absolute anchor (int64 ticks) |
| `phase` | `float` | offset of sample 0's left edge from `t_start`, in **sample units**, ∈ [−1, 0] |

No per‑sample edge times are stored. Every edge is derived from `(t_start, phase, sr, N)`.
The `phase` float is bounded by one sample period and therefore precision‑safe (≈ 1e‑21 s error).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `t_start` | `float` | seconds |
| `t_end` | `float` | `t_start + duration` in ticks, then to seconds |
| `duration` | `float` | `N / sr` |
| `t_start_ticks` | `int` | exact int64 |
| `t_end_ticks` | `int` | `round(t_start_ticks + N * TPS / sr)` |
| `duration_ticks` | `int` | `round(N * TPS / sr)` |
| `t_first_edge` | `float` | absolute seconds of `samples[0]` left edge |
| `t_first_edge_ticks` | `int` | `t_start_ticks + round(phase * TPS / sr)` |
| `n_samples` | `int` | `N` |
| `channel_shape` | `tuple` | `samples.shape[1:]` |

### Constructors

```python
UniformSeries.from_samples(samples, sr, *, t_start=0.0) → UniformSeries
UniformSeries.from_ticks(samples, sr, *, t_start=0, phase=0.0) → UniformSeries
```

`from_samples`: `phase = 0`, `t_start` quantized if float.

### Methods

```python
us.slice(t_a, t_b)     → UniformSeries   # t_a/t_b: float | int
us.shift(t_delta)       → UniformSeries   # O(1); delta: float | int
us.concat(other)        → UniformSeries   # glues other at us.t_end; auto-aligns
us.equal(other)         → bool            # exact
us.sample_times()       → np.ndarray      # float seconds, each sample edge
us.sample_times_ticks() → np.ndarray      # int64 ticks, each sample edge
us.time_to_index(t)     → int             # sample cell containing time t
```

### Precision of the grid

For integer `sr` (all real audio rates), the cut‑time→sample‑index mapping uses
exact `divmod(Δ·sr, TICKS_PER_SECOND)` — no float, no epsilon. A sub‑sample cut
carries the exact integer remainder into the child's `phase`, so
`slice(a,b) ⊕ slice(b,c) == slice(a,c)` is sample‑exact regardless of recording
length or anchor magnitude.

For non‑integer `sr`, a float index with a fixed `1e‑6`‑sample epsilon is used
(practical bound: ~28 hours at 44.1 kHz before the epsilon is reached; property‑tested
at extreme anchors).

---

## `EventSeries` — point‑event time series (RPS, IMU)

### Stored representation

| Field | Type | Meaning |
|-------|------|---------|
| `timestamps` | `np.ndarray` | int64 ticks, **relative to `t_start`** |
| `values` | `np.ndarray \| None` | shape `(…, M)` payload — **event axis is ALWAYS the LAST axis** (`M` = number of events) |
| `t_start` | `int` | absolute anchor (int64 ticks) |
| `dur` | `int` | declared duration (int64 ticks) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `t_start` / `t_end` / `duration` | `float` | seconds |
| `t_start_ticks` / `t_end_ticks` / `duration_ticks` | `int` | exact |
| `timestamps` | `np.ndarray` | relative int64 ticks |
| `abs_timestamps` | `np.ndarray` | absolute float seconds |
| `abs_timestamps_ticks` | `np.ndarray` | absolute int64 ticks |

`__getitem__[i]` returns `(abs_time_seconds, value)` — back‑compat.

### Constructors

```python
EventSeries.from_events(timestamps, values=None, *, t_start=None, t_end=None) → EventSeries
EventSeries.from_ticks(timestamps, values=None, *, t_start, dur)             → EventSeries
```

`from_events`: timestamps as float → once‑quantized to ticks; `t_start` inferred from
first event if `None`, `dur` from `t_end − t_start`.

### Methods

```python
es.slice(t_a, t_b)  → EventSeries   # searchsorted on relative int64; exact boundary logic
es.shift(t_delta)    → EventSeries   # t_start += delta_ticks; O(1)
es.concat(other)     → EventSeries   # other glued at es.t_end; exact int add
es.equal(other)      → bool          # exact timestamps ==, dur ==, t_start ==
```

---

## `SegmentSeries` — half‑open interval series (VAD, labels)

### Stored representation

| Field | Type | Meaning |
|-------|------|---------|
| `starts` | `np.ndarray` | int64 ticks, **relative to `t_start`** |
| `ends` | `np.ndarray` | int64 ticks, **relative to `t_start`** |
| `values` | `np.ndarray \| None` | shape `(…, M)` payload — **segment axis is ALWAYS the LAST axis** (`M` = number of segments); `ids` stay 1‑D `(M,)` |
| `ids` | `np.ndarray` | int64 identity tags (62‑bit random, or explicit) |
| `t_start` | `int` | absolute anchor |
| `dur` | `int` | declared duration |

Segments carry identity tags: splitting a segment across a cut emits two rows with
the same `id`; concat merges them back.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `t_start` / `t_end` / `duration` | `float` | seconds |
| `t_start_ticks` / `t_end_ticks` / `duration_ticks` | `int` | exact |
| `starts` / `ends` | `np.ndarray` | relative int64 ticks |
| `abs_starts` / `abs_ends` | `np.ndarray` | absolute float seconds |
| `abs_starts_ticks` / `abs_ends_ticks` | `np.ndarray` | absolute int64 ticks |

`__getitem__[i]` returns `(abs_start, abs_end, [value,] id)` — back‑compat.

### Constructors

```python
SegmentSeries.from_segments(starts, ends, values=None, ids=None, *,
                            t_start=None, t_end=None) → SegmentSeries
SegmentSeries.from_ticks(starts, ends, values=None, ids=None, *,
                         t_start, dur) → SegmentSeries
```

Ids auto‑generated if `None` (62‑bit random, collision negligible at VAD scales ≤ 10⁴).

### Methods

```python
ss.slice(t_a, t_b)  → SegmentSeries   # exact int clip & rebase; id‑preserving split
ss.shift(t_delta)    → SegmentSeries   # O(1), only t_start moves
ss.concat(other)     → SegmentSeries   # id‑aware seam merge; exact int
ss.equal(other)      → bool            # exact int arrays; ids match
```

---

## `TimeFrame` — column‑keyed container

### Stored representation

| Field | Type | Meaning |
|-------|------|---------|
| `tracks` | `dict[str, TimeSeries]` | series stored **frame‑relative** (t_start relative to frame) |
| `t_start` | `int` | absolute anchor (int64 ticks) |
| `dur` | `int` | declared duration |
| `tags` | `Mapping[str, Hashable]` | scalar metadata (recording_id, split, …) — preserved by all ops, equality-checked on concat/merge |
| `global_data` | `Mapping[str, Any]` | pytree of numpy arrays for non‑temporal metadata (mic_positions, rotor_positions, …) — same merge semantics as tags |

The raw `tracks` dict is internal; all public accessors re‑base to absolute time.

### Properties

| Property | Type |
|----------|------|
| `t_start` / `t_end` / `duration` | `float` (seconds) |
| `t_start_ticks` / `t_end_ticks` / `duration_ticks` | `int` |

### Constructors

```python
TimeFrame.from_tracks(tracks, *, t_start=None, t_end=None,
                      tags=None, global_data=None) → TimeFrame
```

Hull (`t_start`/`t_end`) inferred from track domains if not given. Constructor
re‑bases tracks to frame‑relative via `shift(−t_start)`. `tags` and `global_data`
are optional mappings carried through all ops.

### Dict‑like accessors (return absolute series)

```python
tf[key]           → TimeSeries
tf.keys()         → iterable[str]
tf.values()       → iterable[TimeSeries]    # each re‑based to absolute
tf.items()        → iterable[(str, TimeSeries)]
"key" in tf       → bool
len(tf)           → int
```

### Column ops

```python
tf.select(["audio", "rps"])         → TimeFrame   # subset by key
tf.drop(["vad"])                     → TimeFrame   # remove keys
tf.with_track("imu", series)        → TimeFrame   # add, expands hull
tf.merge(other, overwrite=False)     → TimeFrame   # column‑wise union
```

### Time ops

```python
tf.shift(delta)       → TimeFrame   # O(1); delta: float | int
tf.slice(t_a, t_b)    → TimeFrame   # slice every track; t_a/t_b: float | int
tf.concat(other)       → TimeFrame   # glue along time; auto‑aligns other's domain
tf + other             → TimeFrame   # alias for concat
tf.equal(other)        → bool        # exact comparison in absolute time
```

---

## Invariants

For every series `x` and any `t_start ≤ a ≤ b ≤ c ≤ t_end`:

```
x.slice(a, b).concat(x.slice(b, c)).equal(x.slice(a, c))
```

This holds across arbitrary cut points — including sub‑sample cuts in `UniformSeries`
— because:
- `EventSeries`: exact `searchsorted` on int64, half‑open routing exact.
- `SegmentSeries`: id‑preserving split/merge with exact int bounds.
- `UniformSeries`: exact `divmod` carries sub‑sample remainder into child `phase`;
  concat detects grid overlap via integer sample offset.

For `TimeFrame`:
```
tf.slice(a, b).concat(tf.slice(b, c)).equal(tf.slice(a, c))
```

holds with the same guarantees composed across all track types.

## What's *not* here (out of scope)

- Resampling / interpolation onto another grid.
- Lazy / disk‑backed storage (all samples in memory).
- Value‑merging across two `EventSeries` with the same timestamps.
- `datetime64[ns]` interop (deferred; add thin converters later).
