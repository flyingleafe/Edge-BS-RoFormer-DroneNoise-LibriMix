"""Notebook data-exploration primitives: list, tabulate, thumbnail, pick.

Four calls cover the usual "what is in this dataset?" loop:

- :func:`datasets` — the dload catalog (``dload.lock`` pins) as a DataFrame.
- :func:`meta_table` — sample metadata rendered as a DataFrame.
- :func:`grid` — a sampled grid of spectrogram thumbnails with captions.
- :func:`pick` — one sample, coerced, ready for ``plots.dwym`` /
  ``zoo.FrameModel``.

Every function accepts the same dataset forms:

- a dload dataset **name** (``"DREGON-frames"`` — streamed through
  ``data_processing.streams.DloadFrameDataset``, so both ``tdframe-v1`` and
  DREGON-LM sample-dir layouts decode; needs R2 credentials in ``.env``);
- any **map-style dataset** (``__len__`` + ``__getitem__`` of ``td.Frame``);
- any **iterable of frames** (list, generator, ``DloadFrameDataset``).

Heavy lifting stays where it already lives: streaming/decoding in
``data_processing.streams``, figure assembly in the ``plots.timeframe``
renderers, and entry-name coercion in ``plots.coerce``. Only the thumbnail
grid layout is new here.
"""

from __future__ import annotations

import itertools
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tdseries as td

from plots.audio import first_channel
from plots.coerce import CANONICAL_ENTRIES, _audio_candidates, coerce_frame
from plots.dwym import DwymResult, _in_ipython

__all__ = ["datasets", "meta_table", "grid", "pick"]

#: Entry names probed (in order) for the thumbnail waveform of :func:`grid`.
_GRID_AUDIO_ENTRIES = ("audio", "mixture", "target", "enhanced", "generated")

#: Meta keys probed (in order) for thumbnail captions.
_CAPTION_KEYS = ("recording_id", "id", "split", "drone", "category", "snr_db", "snr")

_CAPTION_MAX_CHARS = 48
_CELL_MAX_CHARS = 80


# ---------------------------------------------------------------------------
# dataset catalog


def datasets(*, sizes: bool = False) -> pd.DataFrame:
    """The dload dataset catalog as a DataFrame (one row per ``dload.lock`` pin).

    Columns: ``name``, ``version`` (12-char prefix). With ``sizes=True`` each
    pinned manifest is fetched (network + credentials) and ``samples`` /
    ``size`` columns are added; a dataset whose manifest cannot be fetched
    gets null values instead of failing the whole table.
    """
    from data_processing.streams import REPO_ROOT

    lock = tomllib.loads((REPO_ROOT / "dload.lock").read_text())
    pins: dict[str, str] = {str(k): str(v) for k, v in dict(lock.get("datasets", {})).items()}
    rows: list[dict[str, Any]] = [{"name": n, "version": v[:12]} for n, v in sorted(pins.items())]
    if sizes:
        import dload

        from data_processing.streams import open_repository

        repo = open_repository()
        for row in rows:
            try:
                manifest = repo.manifest(row["name"], pins[row["name"]])
                row["samples"] = int(manifest.num_samples)
                row["size"] = dload.format_size(int(manifest.total_bytes))
            except Exception:
                row["samples"] = None
                row["size"] = None
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# frame iteration over the accepted dataset forms


def _iter_frames(obj: Any, *, shuffle_seed: int | None = None) -> Iterator[td.Frame]:
    """Yield ``td.Frame``s from any accepted dataset form (module docstring).

    ``shuffle_seed`` only applies to dataset *names* (a seeded streaming
    shuffle, for spread without a full download); other forms keep their own
    order.
    """
    if isinstance(obj, str):
        from data_processing.streams import DloadFrameDataset

        shuffle: bool | int = False if shuffle_seed is None else int(shuffle_seed)
        yield from DloadFrameDataset(obj, shuffle=shuffle)
        return
    if isinstance(obj, td.Frame):
        yield obj
        return
    if isinstance(obj, Mapping):
        raise TypeError("explore: got a Mapping — pass a dataset name, dataset, or frame iterable")
    if hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
        for i in range(len(obj)):
            yield obj[i]
        return
    if isinstance(obj, Iterable):
        yield from obj
        return
    raise TypeError(
        f"explore: cannot iterate {type(obj).__name__} — pass a dload dataset name, "
        "a map-style dataset, or an iterable of td.Frame"
    )


def _split_remaps(hints: dict[str, Any]) -> dict[str, str]:
    """Pop the canonical-entry string remaps (``rps="motor_speed"``) out of hints."""
    return {
        k: hints.pop(k) for k in list(hints) if k in CANONICAL_ENTRIES and isinstance(hints[k], str)
    }


def _coerce(frame: td.Frame, remaps: Mapping[str, str]) -> td.Frame:
    present = {k: v for k, v in remaps.items() if v in dict(frame.items())}
    return coerce_frame(frame, **present)


# ---------------------------------------------------------------------------
# meta_table


def _cell(value: Any) -> Any:
    """One DataFrame cell: scalars pass through, anything else gets a short repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    text = repr(value)
    return text if len(text) <= _CELL_MAX_CHARS else text[: _CELL_MAX_CHARS - 1] + "…"


def _temporal_names(frame: td.Frame) -> list[str]:
    return [k for k, v in frame.items() if isinstance(v, td.Series) and v.has_time]


def _meta_row(frame: td.Frame, fields: Sequence[str] | None) -> dict[str, Any]:
    from data_processing.frames import meta_dict

    meta = meta_dict(frame)
    if fields is not None:
        row = {k: _cell(meta.get(k)) for k in fields}
    else:
        row = {k: _cell(v) for k, v in meta.items()}
    row["entries"] = ", ".join(k for k, _ in frame.items() if k != "meta")
    if _temporal_names(frame):
        row["duration_s"] = round(float(frame.duration), 3)
    return row


def meta_table(
    frames_or_dataset: Any,
    fields: Sequence[str] | None = None,
    limit: int = 32,
) -> pd.DataFrame:
    """Sample metadata as a DataFrame — one row per sample, up to ``limit``.

    Columns are the (flattened, scalar-repr'd) ``meta`` keys, restricted to
    ``fields`` when given, plus two computed columns: ``entries`` (the frame's
    non-meta entry names) and ``duration_s`` (for frames with temporal
    entries).
    """
    rows = [
        _meta_row(frame, fields)
        for frame in itertools.islice(_iter_frames(frames_or_dataset), int(limit))
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# grid


def _grid_audio_series(frame: td.Frame) -> td.Series | None:
    """The entry to thumbnail: a canonical audio entry, else a sole waveform."""
    for name in _GRID_AUDIO_ENTRIES:
        if name in frame and isinstance(frame[name], td.Series):
            return frame[name]
    candidates = _audio_candidates(dict(frame.items()))
    if len(candidates) == 1:
        series = frame[candidates[0]]
        assert isinstance(series, td.Series)
        return series
    return None


def _caption(frame: td.Frame) -> str:
    from data_processing.frames import meta_dict

    meta = meta_dict(frame)
    parts = [f"{meta[k]}" for k in _CAPTION_KEYS if meta.get(k) is not None]
    text = " · ".join(parts) or ", ".join(k for k, _ in frame.items() if k != "meta")
    return text if len(text) <= _CAPTION_MAX_CHARS else text[: _CAPTION_MAX_CHARS - 1] + "…"


def _reservoir(stream: Iterator[td.Frame], n: int, rng: np.random.Generator) -> list[td.Frame]:
    kept: list[td.Frame] = []
    for i, frame in enumerate(stream):
        if len(kept) < n:
            kept.append(frame)
        else:
            j = int(rng.integers(0, i + 1))
            if j < n:
                kept[j] = frame
    return kept


def grid(
    dataset_or_frames: Any,
    n: int = 12,
    *,
    seed: int | None = None,
    scan: int | None = None,
    **dwym_hints: Any,
) -> DwymResult:
    """A sampled grid of compact spectrogram thumbnails with meta captions.

    ``n`` samples are reservoir-sampled from the first ``scan`` frames of the
    stream (default ``max(4 * n, 48)``; a dataset *name* additionally gets a
    seeded streaming shuffle so the scan spans shards). Hints: ``fmax``
    limits the thumbnail bandwidth; canonical-entry string remaps
    (``audio="waveform"``) pass to :func:`plots.coerce.coerce_frame`.

    Returns a :class:`plots.dwym.DwymResult` (``route="grid"``) — it displays
    itself in IPython and offers ``.figure`` / ``.save()`` outside.
    """
    remaps = _split_remaps(dwym_hints)
    fmax = dwym_hints.pop("fmax", None)
    if dwym_hints:
        raise TypeError(
            f"grid() got unsupported hints {sorted(dwym_hints)}; "
            "supported: fmax=<Hz> and canonical entry remaps (audio=..., rps=...)"
        )
    if n <= 0:
        raise ValueError(f"grid() needs n >= 1, got {n}")

    from plots.timeframe.renderers import make_spectrogram_series

    cap = int(scan) if scan is not None else max(4 * n, 48)
    rng = np.random.default_rng(seed)
    stream = itertools.islice(_iter_frames(dataset_or_frames, shuffle_seed=seed), cap)
    frames = [_coerce(f, remaps) for f in _reservoir(stream, int(n), rng)]
    if not frames:
        raise ValueError("grid() got no frames")

    ncols = min(4, len(frames))
    nrows = -(-len(frames) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False, constrained_layout=True
    )
    for ax, frame in zip(axes.flat, frames):
        series = _grid_audio_series(frame)
        if series is None:
            ax.text(0.5, 0.5, "no audio entry", ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            ax.set_title(_caption(frame), fontsize=8)
            continue
        track = make_spectrogram_series(first_channel(series), fmax=fmax)
        spec = np.asarray(track.series.data)
        extent = (
            float(track.series.t_start),
            float(track.series.t_end),
            0.0,
            float(track.hints["freq_max_hz"]),
        )
        ax.imshow(spec, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax.set_title(_caption(frame), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(frames) :]:
        ax.set_axis_off()

    result = DwymResult(figures=[fig], audio={}, route="grid")
    if _in_ipython():
        plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# pick


def _matches(frame: td.Frame, query: str) -> bool:
    from data_processing.frames import get_meta

    for key in ("recording_id", "id"):
        value = get_meta(frame, key)
        if value is not None and query in str(value):
            return True
    return False


def pick(
    dataset: Any,
    index_or_query: int | str | Callable[[td.Frame], bool] = 0,
    **dwym_hints: str,
) -> td.Frame:
    """One sample, coerced (:func:`plots.coerce.coerce_frame`) and ready for
    ``plots.dwym`` / ``zoo.FrameModel``.

    ``index_or_query`` selects the sample:

    - ``int`` — the n-th sample (direct ``dataset[i]`` on map-style datasets,
      n-th of the stream otherwise; negative indices need a map-style
      dataset);
    - ``str`` — the first sample whose ``meta.recording_id`` / ``meta.id``
      contains the string (a stream scans — and downloads — until it hits);
    - callable — the first sample where ``fn(frame)`` is true.

    ``dwym_hints`` are canonical-entry remaps (``rps="motor_speed"``) applied
    silently at coercion.
    """
    remaps = _split_remaps(dict(dwym_hints))
    if set(dwym_hints) - set(remaps):
        raise TypeError(
            f"pick() got unsupported hints {sorted(set(dwym_hints) - set(remaps))}; "
            "supported: canonical entry remaps (audio=..., rps=...)"
        )

    if isinstance(index_or_query, (int, np.integer)):
        i = int(index_or_query)
        if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            return _coerce(dataset[i], remaps)
        if i < 0:
            raise IndexError("negative indices need a map-style (len + getitem) dataset")
        seen = 0
        for frame in _iter_frames(dataset):
            if seen == i:
                return _coerce(frame, remaps)
            seen += 1
        raise IndexError(f"index {i} is beyond the stream (exhausted after {seen} samples)")

    predicate: Callable[[td.Frame], bool]
    if isinstance(index_or_query, str):
        query = str(index_or_query)

        def _by_id(frame: td.Frame) -> bool:
            return _matches(frame, query)

        predicate = _by_id
    elif callable(index_or_query):
        predicate = index_or_query
    else:
        raise TypeError(
            f"index_or_query must be an int, str, or callable, got {type(index_or_query).__name__}"
        )

    scanned = 0
    for frame in _iter_frames(dataset):
        scanned += 1
        if predicate(frame):
            return _coerce(frame, remaps)
    raise ValueError(f"pick(): no sample matched {index_or_query!r} ({scanned} samples scanned)")
