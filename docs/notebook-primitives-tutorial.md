# Notebook primitives — build `speech_enhancement.ipynb` by hand

This tutorial is the Phase 6 ergonomics probe (`docs/refactor-2026-08-plan.md`
§4, Phase 6). The `speech_enhancement.ipynb` notebook stays unwritten on
purpose: you build it by hand from the steps below and report every friction
point. The fixes come from that list, not from more scaffolding.

Ground rules for the notebook itself:

- One markdown cell before each step, one or two sentences.
- Code cells of approximately 5 lines. If a step does not fit, that is a
  friction point — record it.
- The reference style is `notebooks/rps_tracking.ipynb` and
  `notebooks/noise_generation.ipynb`.

Every snippet below is tested syntax against the current tree. The model
snippets ran end to end with random weights; only the checkpoint download and
the dataset stream need R2 credentials.

## 0 · Setup

First cell — path bootstrap plus the two front doors:

```python
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import tdseries as td

from plots import dwym, explore
```

Second cell — the credentials guard. Data cells stream from R2, so fail
early with a clear message:

```python
import os

import data_processing.streams  # noqa: F401 — loads the .env credentials

if not os.environ.get("AWS_ACCESS_KEY_ID"):
    raise RuntimeError(
        "No R2 credentials. Fill .env at the repo root "
        "(see docs/data-and-artifacts.md), then rerun."
    )
```

## 1 · List datasets

```python
explore.datasets()
```

One row per `dload.lock` pin. `explore.datasets(sizes=True)` adds sample
counts and byte sizes (fetches manifests — network). The SE valid sets are
`SE-valid-drone`, `SE-valid-harmonic`, and `SE-valid-avq-survey`.

## 2 · Pull an SE valid sample

```python
sample = explore.pick("SE-valid-drone", 0)
sample
```

`pick` accepts an index, a substring of `meta.recording_id` / `meta.id`, or a
predicate `fn(frame) -> bool`. The result is coerced
(`data_processing.canonical.coerce_frame`), so entry names are already canonical.

What an SE valid frame contains:

- `mixture` — the noisy input, mono `(time,)` at 16 kHz, 2 s;
- `target` — the clean speech at the exact mixed gain;
- `meta` — scalars such as the SNR and provenance keys.

Inspect the metadata across samples as a table:

```python
explore.meta_table("SE-valid-drone", limit=8)
```

Thumbnail a spread of the set (SNR grid −30…0 dB, 50 clips per step):

```python
explore.grid("SE-valid-drone", n=8, seed=0, fmax=4000)
```

## 3 · List SE checkpoints

```python
import zoo

[row["experiment"] for row in zoo.checkpoints(task="speech_enhancement")]
```

Rows carry `experiment`, `task`, `files` / `sizes` / `mtimes`, `has_config`,
and `metrics` (present after an `eval.py` run + `zoo.refresh()`). The cache
is `<repo>/.checkpoints-cache.json`; `zoo.refresh(full=True)` re-lists R2.
Known F1 baselines: `f1_mpsenet_a`, `f1_tfgridnet_a`, `f1_dcunet_a`,
`f1_htdemucs_a`, `f1_edge_bs_rof_a` (and the `_b` seeds).

## 4 · Load and run a model

```python
fm = zoo.load("f1_mpsenet_a")
pred = fm(sample)
pred["enhanced"].shape
```

`zoo.load` composes `experiment=<name>` against `conf/`, builds the model +
codec, resolves `best.ckpt` (local `results/<name>/` first, else R2), and
returns a `FrameModel`: `td.Frame` in, `td.Frame` out, no tensors in sight.
`zoo.load(name, ckpt="ep12_....ckpt", device="cuda")` selects a different
checkpoint or device. For SE tasks `pred` holds one entry: `enhanced`.

## 5 · Visualize and listen

Merge the prediction into the sample, then let `dwym` dispatch:

```python
both = sample.with_entry("enhanced", pred["enhanced"])
dwym(both)
```

Two or more of `mixture` / `target` / `enhanced` route to the SE comparison:
aligned spectrogram rows plus one audio player per entry. `fmax=4000` trims
the displayed band.

## 6 · Compare several models

```python
frames = {}
for name in ("f1_mpsenet_a", "f1_tfgridnet_a", "f1_dcunet_a"):
    frames[name] = sample.with_entry("enhanced", zoo.load(name)(sample)["enhanced"])
dwym(frames)
```

A `{label: Frame}` dict of same-route frames renders one aligned figure with
a row block per label (`route == "multi:se"`), and audio players keyed
`label/entry`.

## 7 · Compute metrics on the pair

The metric classes read `pred["enhanced"]` against `target["target"]`:

```python
from metrics import ESTOIMetric, MetricSuite, PESQMetric, SISDRMetric

suite = MetricSuite({"si_sdr": SISDRMetric(), "estoi": ESTOIMetric(), "pesq": PESQMetric()})
suite.evaluate_one(pred, sample)
```

For many samples, `suite.evaluate(pairs, group_by="snr_db")` returns a
`SuiteResult` with per-group aggregation. For the full evaluation protocol
use `eval.py` — it is the same suite with dataset wiring and R2 upload.

## 8 · Escape hatches

When `dwym` guesses wrong or you need more control:

- **Entry remaps**: `dwym(frame, rps="motor_speed")` — canonical-name hints
  go to `data_processing.canonical.coerce_frame` and silence its warning.
- **Force a route**: `dwym(frame, renderer="timeframe")` — routes are
  `se` / `salience` / `noise_gen` / `rps` / `audio` / `timeframe`.
- **Track-level control**: `plots.timeframe.plot_timeframe(frame,
  tracks=[...])` with explicit `PlotTrack`s; converters in
  `plots.timeframe.renderers` (`make_spectrogram_series`, ...); task figures
  in `plots.se` / `plots.noise_gen` / `plots.rps_prediction`.
- **Codec internals**: `fm.model`, `fm.codec`, `fm.task` are public.
  `fm.codec.to_inputs(batch)` / `call_model` / `to_frame(outputs, batch)` is
  the full seam when you need raw tensors; `data_processing.collate
  .frame_collate` / `slice_sample` convert between samples and batches.
- **Raw streams**: `data_processing.streams` (`DloadFrameDataset`,
  `iter_published_frames`, `dload:` URIs) when `explore` is too thin.

## 9 · Report every friction point

While you build the notebook, record each of these the moment it occurs:

- a step that needs more than ~5 lines in one cell;
- an import you had to hunt for;
- a wrong `dwym` guess, a missing hint, or a bad default;
- a slow cell (note the duration and the data it pulled);
- an unclear error, or an error where a message should have been;
- any point where you opened library source to continue.

Put the list in the closing markdown cell of the notebook and report it back.
The ergonomics fixes of Phase 7 come from this list.
