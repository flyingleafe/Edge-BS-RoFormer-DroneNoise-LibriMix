# Plan: fixed-point (int64 tick) time for `utils.data`

## Why

`float64` **seconds** at Unix magnitude (~1.6e9) has `ulp ≈ 2.4e-7 s`. That is
coarser than an audio sample at 44.1 kHz (2.3e-5 s) only by ~100×, so every
boundary check fights cancellation noise. The current code pays for this with
magnitude-scaled tolerances (`_floats.py`: `tclose`, `t_atol_at`, `grid_atol`),
relative storage, and *still* hits corner-case failures (e.g. an event rounded
onto `t_end`, a sliced `dur` re-based to ~0 drifting past its grid edge).

**Fixed-point integer time removes the problem at the root instead of patching
it.** `int64` addition / subtraction / comparison are *exact* and
*magnitude-independent*. There is no ulp, no cancellation, no tolerance.
Combined with the established **relative-to-container** principle, we get an
exact, hierarchical time model and we **delete most of `_floats.py`**.

This plan keeps the relative principle (nothing is abandoned): the integers are
just a better numeric substrate for it.

## Representation

- **Unit:** an `int64` count of *ticks*, where `TICKS_PER_SECOND` is a single
  module constant. **Settled: `1_000_000_000` (nanoseconds).** Microsecond
  (`1e6`) is the documented "more than enough" floor, but ns is chosen because
  (a) it is numpy-native (`datetime64[ns]`/`timedelta64[ns]`), and (b) it keeps
  *integer* sub-tick headroom even for high-rate audio: at 192 kHz one sample =
  5208 ns (vs only 5 µs), so ns leaves room to resolve sub-sample phase, where
  µs would be coarse. The constant is the only thing to change to revisit this.
- **Range (int64):** at ns, ±9.2e18 ticks ≈ ±292 years from tick 0. Unix epoch
  "now" is 1.6e18 ns — comfortably inside; valid to ≈ year 2262. At µs the
  range is ±292 000 years. Either covers every realistic recording.
- **Absolute time** = `int64` ticks from an arbitrary zero (epoch). **Duration
  / relative offset** = `int64` ticks. Arrays of timestamps are plain
  `np.int64` arrays (vectorized, exact).
- **Integer sample rates are exact.** Essentially all audio rates are integer
  Hz (8000, 16000, 44100, 48000, 96000, 192000). For integer `sr` the
  time↔sample-index mapping is done in **exact Python-int arithmetic**
  (`Δticks · sr // TICKS_PER_SECOND` for the index, `(Δticks · sr) %
  TICKS_PER_SECOND` for the sub-sample remainder) — no float, no epsilon,
  no overflow (Python ints are unbounded; these are scalar per-op). Non-integer
  `sr` falls back to a float index with a fixed `1e-6`-sample epsilon.
- **Rejected alternatives:**
  - `Decimal` — exact but *not* numpy-vectorizable, slow, object arrays. Only
    viable for scalars; we need arrays. ✗
  - `fractions.Fraction` — would make `1/sr` exact, but is even slower and
    non-vectorizable, and we do **not** need exact `1/sr` (see "audio grid"). ✗
  - `np.datetime64[ns]` / `timedelta64[ns]` — viable and self-documenting, but
    introduces dtype edge cases (NaT, unit coercion). **Use plain `int64`** +
    one conversion helper; revisit datetime64 only if interop demands it.

## The relative principle (unchanged, now exact)

- `TimeFrame.t_start` is the single **absolute** anchor (int64 ticks).
- Each track's `t_start` is stored **relative to the frame** (int64 ticks).
- Within a series, all data timestamps **and `dur`** are stored **relative to
  the series `t_start`** (int64 ticks).
- `shift` and frame re-basing are exact int64 additions — no precision loss for
  *any* delta. This is exactly the win the float version could not deliver.

## Per-type storage

| Type | Stored |
|------|--------|
| `EventSeries` | `t_start` (int64 ticks); `timestamps` (int64 ticks, relative); `dur` (int64 ticks) |
| `SegmentSeries` | `t_start`; `starts`, `ends` (int64 ticks, relative); `dur`; `ids` (unchanged) |
| `UniformSeries` | `t_start` (int64 ticks); `dur` (int64 ticks); `phase` (small float, see below); `samples`, `sr` (Hz, unchanged) |
| `TimeFrame` | `t_start` (int64 ticks, absolute anchor); `dur` (int64 ticks); tracks stored frame-relative |

## The one off-grid quantity: the audio sub-sample phase

Sample edges lie on a grid of spacing `1/sr`, which is irrational in ticks
(44.1 kHz → 22675.736… ns), so a sample-0 edge time **cannot** be an exact int.
The resolution: we never store per-sample edge times — only an anchor, the
integer sample count `N`, and `sr` — and we keep the sub-sample offset as a
**small relative float**.

- **`phase`** = offset of `samples[0]`'s left edge from `t_start`, in **sample
  units**, `∈ [-1, 0]`. Because it is *relative* and bounded by one sample, a
  float64 stores it to ≈ 1e-16 of a sample (≈ 1e-21 s) — effectively exact. The
  Unix-magnitude precision problem never touches it; only the (exact int64)
  anchor carries the magnitude.
- **Sample-index mapping** (cut time → index). With `Δ = t_cut_ticks −
  t_start_ticks` (exact int64) and integer `sr`:
  - `q, r = divmod(Δ * sr, TICKS_PER_SECOND)`  (exact Python ints)
  - real-valued index `k = q + r/TICKS_PER_SECOND − phase`; `ka = floor`,
    `kb = ceil` are then decided **exactly** from `(q, r, phase)` without
    lossy float `k` (compare `r/TPS` to `−phase` as the rational `r` vs
    `−phase·TPS`). For **non-integer** `sr`, fall back to float `k` with a
    `1e-6`-sample epsilon on `floor`/`ceil`.
  - the **new** sub-sample phase after a cut is recovered exactly from `r`
    (integer remainder) and stored back as the small float.
- `concat` duplicate-sample detection uses the **integer** sample offset
  (`round(k_offset)` with a phase check) — exact at the index level.
- Incoming cut points off the tick grid are **quantized to the nearest tick**
  once, on construction; the same quantized value is reused everywhere, so
  slice/concat identity at the sample level is preserved.

**Invariant to re-verify:** `slice(a,b) ⊕ slice(b,c) == slice(a,c)` now rests on
(1) tick-exact anchors, (2) integer sample indices, and (3) the precision-safe
relative `phase`. For integer `sr` this is fully exact; for the float fallback
the `1e-6`-sample epsilon must stay `< 0.5` sample — document and test the bound
at extreme `sr` and anchor magnitudes.

## Public API (settled)

Storage is **int64 ticks**; seconds are a convenience layer, *never* the
exact path.

- **Float seconds for ergonomics, ticks for exactness.** `t_start` / `t_end` /
  `duration` and `__getitem__` return **float seconds** (matches the project's
  float-seconds ecosystem: audio loaders, the legacy `*Record` types). This is
  display/interop only and is allowed to be lossy at Unix magnitude.
- **Exact accessors:** `t_start_ticks`, `t_end_ticks`, `duration_ticks`,
  `abs_timestamps_ticks`, `abs_starts_ticks`, etc. Reading a boundary as ticks
  and feeding it back is **exact**.
- **`slice` / `concat` / constructors accept either** int ticks (exact) **or**
  float seconds (quantized once via `round(seconds * TICKS_PER_SECOND)`). They
  operate internally on ticks.
- **The round-trip rule:** `x.slice(x.t_start, x.t_end) == x` is only
  guaranteed exact via the **ticks** accessors
  (`x.slice(x.t_start_ticks, x.t_end_ticks)`); the seconds path is exact for
  small magnitudes and within ≤ 0.5 tick otherwise (boundary requests are
  clamped into `[t_start_ticks, t_end_ticks]`, so they never fall *outside*).
- `from_seconds(...)` / `to_seconds(...)` and `from_ticks(...)` helpers.
- **Property tests use the ticks accessors** so identity is asserted exactly
  (`==` on int64 + `np.array_equal` on samples/values), not `allclose`.

## What we delete / simplify

- `_floats.py`: drop `tclose`, `t_atol_at`, `grid_atol`. Time equality becomes
  exact `==` on int64. The only surviving tolerance is the `1e-6`-sample
  **index-space** epsilon, used **only** on the non-integer-`sr` fallback path
  (integer `sr` is exact and needs none).
- Every magnitude-scaled `atol` in `slice` / `concat` / `equal` /
  `__post_init__` → exact integer comparison.
- The absolute/relative "heuristic" in `EventSeries.__post_init__` disappears:
  the constructor stores relative ticks; `from_*` converts.

## Phasing

1. Add `TICKS_PER_SECOND`, `to_ticks`/`from_ticks` (scalar + array), and the
   index epsilon. Keep `_floats.py` temporarily.
2. `EventSeries` → int64 relative ticks; exact `equal`. Update strategies to
   draw int64 tick anchors (now safe at huge magnitude); assert **exact**.
3. `SegmentSeries` → int64 relative ticks.
4. `UniformSeries` → `t_start`/`dur` to int64 ticks, sub-sample `phase` as the
   small relative float; cut→index via exact `divmod(Δ·sr, TPS)` for integer
   `sr` (float + epsilon fallback otherwise). Re-verify sub-sample identity.
5. `TimeFrame` → ticks anchor + relative tracks; re-basing is exact int add.
6. Remove the now-dead `_floats.py` helpers; replace tolerant compares with `==`.
7. Update `AGENTS.md`, module docstrings, and tests (strategies draw large
   int64 anchors; property tests assert exact equality, not `allclose`).

## Relationship to the in-progress float refactor

The current branch already moved every type to **relative storage** and made
`shift` O(1) — that structural work carries over unchanged. The fixed-point
switch *replaces* the precision-motivated parts (tolerances, `dur` re-derivation
gymnastics) with exact integers. Net effect: less code, no tolerances, and the
relative principle holds exactly.

## Settled decisions

1. **Resolution: nanoseconds** (`TICKS_PER_SECOND = 1_000_000_000`), a single
   module constant. µs is the documented floor; ns wins on numpy-native unit
   and sub-sample headroom at high audio rates, at no range cost for our data.
2. **Public time type: float seconds for ergonomics, int64 ticks under the
   hood, with explicit `*_ticks` accessors for exactness.** Exact round-trips
   and property tests go through ticks; seconds is interop/display only. Slice
   and constructors accept both (seconds quantized once via `round`).
3. **Sample-index mapping is exact for integer `sr`** via Python-int
   `divmod(Δ·sr, TICKS_PER_SECOND)`; the sub-sample **`phase`** is a small
   relative float (precision-safe, ≤ 1 sample). Non-integer `sr` uses a float
   index with a **`1e-6`-sample** epsilon; a property test must stress this at
   `sr = 192000`, durations of hours, and anchors out to ≈ year 2200, asserting
   exact sample counts and slice/concat identity.
4. **`datetime64[ns]` interop: deferred.** The core is plain `int64` + helpers
   (avoids `NaT` / unit-coercion friction). Add thin `to_datetime64()` /
   `from_datetime64()` converters later, only if a caller needs them.
