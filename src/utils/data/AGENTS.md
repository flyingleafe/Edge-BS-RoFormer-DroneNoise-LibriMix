# src/utils/data/ — Aligned time-series containers

A small library for working with audio (and other regular-rate signals)
together with co-recorded telemetry (RPS, IMU, VAD segments, ...). The
existing ad-hoc records (`data_processing.dregon.DREGONRecord`,
`data_processing.michaels.MichaelsRecord`) are the prototype that this
abstraction generalises.

## Public surface

```python
from utils.data import (
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

| Method | Meaning |
|--------|---------|
| `t_start`, `t_end` | declared half-open interval `[t_start, t_end)` in seconds |
| `duration` | `t_end - t_start` |
| `slice(t_a, t_b)` | restrict to `[t_a, t_b)` |
| `concat(other)` | glue at seam, requires `self.t_end ≈ other.t_start` |
| `equal(other)` | structural equality, modulo float tolerance on times |
| `__len__`, `__getitem__` | row-wise indexing into the underlying samples/events/segments |

## The one invariant

For every series `x` and any `t_start ≤ a ≤ b ≤ c ≤ t_end`:

```
x.slice(a, b).concat(x.slice(b, c)).equal(x.slice(a, c))
```

This holds across **arbitrary cut points** — including sub-sample cuts in
uniform signals — because each concrete subclass carries enough bookkeeping
to losslessly recover the original on re-concat. See the module docstring of
each file for the storage model.

## File map

| File | Role |
|------|------|
| `base.py` | abstract `TimeSeries`, `DomainError`, `IncompatibleSeriesError` |
| `uniform.py` | `UniformSeries` — sample-grid model with sub-sample cuts (the subtle one) |
| `event.py` | `EventSeries` — sorted point timestamps + optional values |
| `segment.py` | `SegmentSeries` — half-open intervals; split-and-rejoin via origin `ids` |
| `frame.py` | `TimeFrame` — dict-keyed container; column ops (`select`, `drop`, `merge`) + time ops (`slice`, `concat`) |
| `_floats.py` | time-tolerance helpers: `tclose`, `t_atol_at`, `grid_atol` |

## Design notes (the things you'd otherwise re-derive)

1. **Sub-sample cuts of uniform signals.** A sample's cell is one bin wide
   (`1/sr`). When the user cuts at time `t_cut` that falls inside sample
   `k`'s cell, the sample is *shared* between both halves — the left half
   keeps samples `0..k`, the right keeps samples `k..N`. Concat detects this
   via the integer sample-offset between the two `t_first_edge`s and drops
   the duplicate. This is the only model that gives strict slice/concat
   identity at sub-sample resolution.

2. **Sample-index arithmetic > absolute-time arithmetic.** When checking grid
   alignment between two `UniformSeries`, we compute the offset in *sample
   units* (`(t1 - t0) * sr`) and round to the nearest integer. This cancels
   the Unix-timestamp magnitude (~1.6e9) cleanly; an absolute subtraction
   accumulates ulp-noise that exceeds any reasonable sub-sample tolerance at
   audio sample rates.

3. **Time-tolerance scales with magnitude.** `t_atol_at(t)` returns
   `1e-9 + 8·ulp(|t|)`. Unix-magnitude anchors have `ulp ≈ 2e-7`, so a
   constant `atol=1e-9` would falsely reject anything that round-tripped
   through float arithmetic. All boundary checks use this helper.

4. **Boundary slack is *exact*-equality-gated.** When a slice request lands
   exactly on the declared `t_start` / `t_end` (`==` on floats — not
   `tclose`), the slice accepts events/segments lying within atol of that
   boundary (consistent with `__post_init__`). Interior cuts stay strictly
   half-open. Using `tclose` here was a bug: it conflated "interior cut one
   ulp away" with "boundary cut", which broke `slice(a,b)+slice(b,c) ==
   slice(a,c)` when `b` happened to be one ulp from the domain edge.

5. **Segment identity on concat.** Each segment carries a random 62-bit
   `id`. Splitting a segment across a cut emits two rows sharing the same
   id; concat merges back any pair at the seam with matching ids. Without
   identity tags, "abutting segments with equal payload" would be silently
   merged on concat — usually not what the user means.

## Status / known gaps

* **Resampling, interpolation onto another grid, value-merging across two
  EventSeries with the same timestamps** — out of scope here; build them on
  top.
* **Lazy / disk-backed storage** — out of scope. All samples live in memory.
* **Mass-construction performance.** `SegmentSeries` calls `secrets.randbits`
  per segment to assign ids; fine for typical drone-VAD scales (≤ 10⁴ segs)
  but linear in `M`. Pass explicit `ids` for bulk construction.

## Tests

`tests/utils/data/` contains Hypothesis property tests:

* `test_uniform.py` — sub-sample cut invariants, multi-channel, boundary-exact
  cuts, rejection of mismatched rates / seams.
* `test_event.py` — slice-concat identity, half-open routing at the seam,
  value-shape rejection.
* `test_segment.py` — straddling-segment split-and-rejoin via `ids`,
  unrelated-segments-at-seam not merged.
* `test_frame.py` — column ops (`select`/`drop`/`merge`), domain mismatch
  rejection, slice/concat identity composed across all three track types.

Property tests draw both small-magnitude and Unix-magnitude time anchors —
the latter exposes float-precision corner cases.
