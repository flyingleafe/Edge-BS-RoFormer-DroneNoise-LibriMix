"""dload ↔ tdseries bridge: stream datasets from object storage as ``td.Frame``s.

`dload <https://github.com/flyingleafe/dload>`_ stores a dataset as versioned
shards of ``(key, {field: bytes})`` samples and streams them through lazy,
composable ``Pipeline``s. This module is the project's only seam between that
world and the ``tdseries``-Frame data model everything downstream speaks
(``data_processing.collate.frame_collate``, the task codecs, the training
loop). It provides, in order:

- :func:`open_repository` — the process-wide ``dload.Repository`` anchored at
  the repo root (``dload.toml`` / ``dload.lock`` live there, not in the cwd);
- **decoders** (all module-level, hence picklable into DataLoader workers):
  bytes → Series/Frame for wav/flac and ``.npy`` fields, and the per-sample
  :func:`decode_dregon_lm` mirroring exactly what
  ``frame_datasets.DregonLMFrameDataset`` yields from a local folder;
- a **generic Frame codec** (:func:`frame_to_sample` /
  :func:`sample_to_frame`, layout ``"tdframe-v1"``): lossless round-trip of an
  arbitrary ``td.Frame`` — any number of series, each with its exact
  grid/stamp/span time index and dtype, plus nested ``"meta"`` scalars — so
  rich recordings (audio + motor RPS + IMU + telemetry) can be published
  as-is;
- thin **Frame combinators** over pipelines (:func:`to_frames`,
  :func:`frame_windows`, :func:`mix_frames`, :func:`resample_frames`);
- :class:`DloadFrameDataset` — the torch ``IterableDataset`` / Hydra
  ``_target_`` (see ``conf/data/dregon_lm_v4_stream.yaml``);
- **materialization** back to files (:func:`ensure_local`) and the
  ``dload:NAME[/subpath]`` URI scheme (:func:`resolve_source`) that lets any
  existing ``data_dir``/``root`` config value point at a dload dataset.

A composed pipeline, end to end::

    from functools import partial
    import data_processing.streams as streams
    from data_processing.collate import frame_collate

    repo = streams.open_repository()
    speech = streams.to_frames(
        repo.dataset("librispeech-clean").samples().shuffle(4096, seed=0).repeat(),
        streams.decode_tdframe,
    )
    noise = streams.to_frames(
        repo.dataset("DREGON-noise-frames").samples().shuffle(4096, seed=1).repeat(),
        streams.decode_tdframe,
    )
    batches = (
        streams.mix_frames(
            streams.frame_windows(speech, win_s=1.0),
            streams.frame_windows(noise, win_s=1.0),
            snr_db=(-30.0, 0.0),
            seed=2,
        )
        .batch(16, collate=frame_collate)
    )

Everything that crosses into DataLoader workers (decoders, combinator bodies)
is a module-level function or a ``functools.partial`` over one — never a
lambda/closure — per dload's pickling contract.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

import dload
import numpy as np
import soundfile as sf
import tdseries as td
import torch
import torch.nn.functional as F
from dload.torch import as_iterable_dataset
from torch.utils.data import IterableDataset

from data_processing.frames import audio_series, get_meta, rps_series, with_meta

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a project dependency.
    load_dotenv = None

__all__ = [
    "open_repository",
    "local_repository",
    "stretch_rps_to_frames",
    "audio_series_from_bytes",
    "rps_series_from_bytes",
    "decode_dregon_lm",
    "decode_tdframe",
    "is_data_sample",
    "frame_to_sample",
    "sample_to_frame",
    "iter_published_frames",
    "TDFRAME_LAYOUT",
    "to_frames",
    "frame_windows",
    "mix_frames",
    "resample_frames",
    "DloadFrameDataset",
    "ensure_local",
    "resolve_source",
]

#: ``(key, {field: bytes})`` — re-exported for signatures below.
Sample = tuple[str, dict[str, bytes]]

REPO_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    # Credentials (AWS_*) and DLOAD_* cache settings live in the project .env;
    # shell-provided variables win (override=False).
    load_dotenv(REPO_ROOT / ".env", override=False)

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_SAMPLE_RATE = 16000

_repository: dload.Repository | None = None


def open_repository() -> dload.Repository:
    """The process-cached ``dload.Repository``, anchored at the repo root.

    Config resolution starts from ``REPO_ROOT`` (so the checked-in
    ``dload.toml`` is found regardless of the cwd — Hydra chdirs, notebooks,
    Slurm jobs), and version pinning reads/writes the repo-root
    ``dload.lock``, again independent of cwd. Credentials come from the
    standard AWS env chain (the project ``.env`` is loaded at import time).
    """
    global _repository
    if _repository is None:
        config = dload.Config.load(cwd=REPO_ROOT)
        repo = dload.Repository.open(config)
        repo.lock_path = REPO_ROOT / "dload.lock"
        _repository = repo
    return _repository


def local_repository(root: str | Path) -> dload.Repository:
    """A ``dload.Repository`` backed by a plain local directory instead of R2.

    Same pack/manifest format as the shared remote, just written under ``root``
    (``root/remote`` + ``root/cache``), so a dataset built here can later be
    published unchanged. Used for datasets that are built and consumed on one
    machine without being published — e.g. a replication's fixed SE valid set
    (``streams.local_repository``, consumed by
    :class:`data_processing.frame_datasets.SEValidFrameDataset` with
    ``local_root``). Relative roots resolve against the repo root, so the
    location is cwd-independent (Hydra chdirs).
    """
    from dload.cache import ShardCache
    from dload.remote import LocalRemote

    base = Path(root)
    if not base.is_absolute():
        base = REPO_ROOT / base
    (base / "remote").mkdir(parents=True, exist_ok=True)
    return dload.Repository(LocalRemote(base / "remote"), ShardCache(base / "cache", None))


# ─── Decoders: raw sample bytes → tdseries ─────────────────────────────────────
#
# All decoders are module-level (picklable) and pure — safe inside
# `Pipeline.map` in forked or spawned DataLoader workers.


def stretch_rps_to_frames(rps_raw: np.ndarray, n_frames: int) -> np.ndarray:
    """Shape-stretch a ``(rotor, M)`` RPS array onto ``n_frames`` STFT frames.

    Endpoint-to-endpoint linear resample via ``F.interpolate`` — byte-for-byte
    the resampling ``DregonLMFrameDataset`` (and the legacy
    ``train_rps_predictor.py::DREGONRPSDataset`` before it) has always used;
    see that class's docstring for the alignment caveat this inherits.
    """
    return (
        F.interpolate(
            torch.from_numpy(np.ascontiguousarray(rps_raw)).unsqueeze(0),
            size=int(n_frames),
            mode="linear",
            align_corners=False,
        )
        .squeeze(0)
        .numpy()
    )


def audio_series_from_bytes(
    data: bytes,
    *,
    sample_rate: int | None = None,
    channel: int | None = None,
) -> td.Series:
    """Decode wav/flac bytes to the canonical audio Series.

    Mono files (or ``channel=<int>`` selection) become a ``(time,)`` Series,
    multichannel ones ``(mic, time)`` — identical to what
    ``DregonLMFrameDataset`` builds from a ``mixture.wav`` on disk.
    ``sample_rate``, when given, is asserted against the encoded rate (the
    project convention: datasets are stored at the training rate; resampling
    is an explicit pipeline stage, see :func:`resample_frames`).
    """
    raw, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)  # (T, C)
    if sample_rate is not None and int(sr) != int(sample_rate):
        raise ValueError(f"audio sr={sr} != expected sample_rate={sample_rate}")
    audio = np.ascontiguousarray(raw.T)  # (C, T)
    if channel is not None:
        audio = audio[channel : channel + 1]
    return audio_series(audio, int(sr))


def rps_series_from_bytes(
    data: bytes,
    *,
    n_frames: int,
    sample_rate: int,
    hop_length: int,
) -> td.Series:
    """``rps.npy`` bytes → ``(rotor, time)`` Series on the exact STFT grid."""
    rps_raw = dload.codecs.npy_from(data).astype(np.float32)
    rps = stretch_rps_to_frames(rps_raw, n_frames)
    return rps_series(rps, sample_rate=sample_rate, hop_length=hop_length)


def is_data_sample(sample: Sample) -> bool:
    """Filter predicate dropping bookkeeping samples (keys starting ``_``,
    e.g. a dataset-level ``_meta`` sample)."""
    return not sample[0].startswith("_")


def decode_dregon_lm(
    sample: Sample,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channel: int | None = None,
) -> td.Frame:
    """Decode one DREGON-LM-style dload sample into the training Frame.

    Sample convention (how the processed splits are published): key
    ``sample_NNNNN``; fields named by the original file *stem* — ``mixture``
    (wav bytes), ``rps`` (npy bytes), optionally ``meta``/``metadata`` (json,
    merged into the Frame's ``"meta"``). The result is structurally identical
    to ``DregonLMFrameDataset.__getitem__``: ``mixture`` ``(time,)`` /
    ``(mic, time)``, ``rps`` ``(rotor, time)`` on the ``sample_rate /
    hop_length`` frame grid, nested invariant ``meta`` (with at least
    ``recording_id``), so ``frame_collate`` and the task codecs work
    unchanged. ``n_fft`` is accepted for config parity with
    ``DregonLMFrameDataset`` (the frame grid depends only on ``hop_length``).
    """
    del n_fft  # config parity only; the frame grid depends on hop_length
    key, fields = sample
    if "mixture" not in fields or "rps" not in fields:
        raise KeyError(
            f"sample {key!r} has fields {sorted(fields)}; expected the DREGON-LM "
            "convention ('mixture' wav bytes + 'rps' npy bytes)"
        )
    audio = audio_series_from_bytes(fields["mixture"], sample_rate=sample_rate, channel=channel)
    n_frames = audio.dim_size("time") // int(hop_length) + 1
    rps = rps_series_from_bytes(
        fields["rps"], n_frames=n_frames, sample_rate=sample_rate, hop_length=hop_length
    )
    meta: dict[str, Any] = {}
    for meta_field in ("meta", "metadata"):
        if meta_field in fields:
            decoded = dload.codecs.json_from(fields[meta_field])
            if isinstance(decoded, dict):
                meta.update(decoded)
    meta.setdefault("recording_id", key)
    return td.Frame({"mixture": audio, "rps": rps, "meta": td.Frame(meta)})


# ─── Generic Frame codec (layout "tdframe-v1") ─────────────────────────────────
#
# Lossless (de)serialization of an arbitrary td.Frame into dload sample
# fields, for publishing *rich* recordings (multichannel audio + motor RPS +
# IMU + telemetry columns, each with its own exact time index) rather than
# fixed file conventions. tdseries has no (de)serialization of its own; its
# indexes are plain dataclasses over exact integer ticks
# (GridIndex(sr_num, size, sr_den, t_start_ticks, dur_ticks, phase),
# StampIndex(stamps, t_start_ticks, dur_ticks), SpanIndex(starts, ends, ids,
# t_start_ticks, dur_ticks)), so encoding those fields verbatim IS the exact
# round-trip.
#
# Layout: one `.npy` field per Series (field name = entry path, nested frames
# joined with "/"), `#`-suffixed auxiliary `.npy` fields for stamp/span index
# arrays, and one `_frame` JSON field holding the structural descriptor
# (dims, index parameters, nested-frame scalars). Publishers mark the dataset
# with `meta={"layout": "tdframe-v1"}` so consumers (DloadFrameDataset)
# dispatch to `sample_to_frame` automatically.

TDFRAME_LAYOUT = "tdframe-v1"
LAYOUT_META_KEY = "layout"
_FRAME_FIELD = "_frame"


def _encode_index(idx: Any, path: str, dim: str, fields: dict[str, bytes]) -> dict[str, Any]:
    if isinstance(idx, td.GridIndex):
        rate = idx.rate
        return {
            "type": "grid",
            "sr_num": int(rate.numerator),
            "sr_den": int(rate.denominator),
            "size": int(idx.n),
            "t_start_ticks": int(idx.t_start_ticks),
            "dur_ticks": int(idx.dur_ticks),
            "phase": float(idx.phase),
        }
    if isinstance(idx, td.StampIndex):
        fields[f"{path}#{dim}.stamps"] = dload.codecs.npy_bytes(
            np.asarray(idx.stamps, dtype=np.int64)
        )
        return {
            "type": "stamps",
            "t_start_ticks": int(idx.t_start_ticks),
            "dur_ticks": int(idx.dur_ticks),
        }
    if isinstance(idx, td.SpanIndex):
        fields[f"{path}#{dim}.starts"] = dload.codecs.npy_bytes(
            np.asarray(idx.starts, dtype=np.int64)
        )
        fields[f"{path}#{dim}.ends"] = dload.codecs.npy_bytes(np.asarray(idx.ends, dtype=np.int64))
        fields[f"{path}#{dim}.ids"] = dload.codecs.npy_bytes(np.asarray(idx.ids))
        return {
            "type": "spans",
            "t_start_ticks": int(idx.t_start_ticks),
            "dur_ticks": int(idx.dur_ticks),
        }
    if isinstance(idx, td.LabelIndex):
        return {"type": "label", "labels": list(idx.labels)}
    if isinstance(idx, td.RangeIndex):
        return {"type": "range", "size": int(idx.n)}
    raise TypeError(f"cannot serialize index type {type(idx).__name__} (entry {path!r}/{dim!r})")


def _decode_index(desc: Mapping[str, Any], path: str, dim: str, fields: Mapping[str, bytes]) -> Any:
    kind = desc["type"]
    if kind == "grid":
        return td.GridIndex(
            int(desc["sr_num"]),
            int(desc["size"]),
            int(desc["sr_den"]),
            int(desc["t_start_ticks"]),
            int(desc["dur_ticks"]),
            float(desc["phase"]),
        )
    if kind == "stamps":
        return td.StampIndex(
            dload.codecs.npy_from(fields[f"{path}#{dim}.stamps"]),
            int(desc["t_start_ticks"]),
            int(desc["dur_ticks"]),
        )
    if kind == "spans":
        return td.SpanIndex(
            dload.codecs.npy_from(fields[f"{path}#{dim}.starts"]),
            dload.codecs.npy_from(fields[f"{path}#{dim}.ends"]),
            dload.codecs.npy_from(fields[f"{path}#{dim}.ids"]),
            int(desc["t_start_ticks"]),
            int(desc["dur_ticks"]),
        )
    if kind == "label":
        return td.LabelIndex(tuple(desc["labels"]))
    if kind == "range":
        return td.RangeIndex(int(desc["size"]))
    raise ValueError(f"unknown index descriptor type {kind!r} (entry {path!r}/{dim!r})")


def _encode_series(series: td.Series, path: str, fields: dict[str, bytes]) -> dict[str, Any]:
    time_axis = series.time_axis
    time_dim = series.dims[time_axis] if time_axis is not None else None
    indexes: dict[str, Any] = {}
    for dim in series.dims:
        if dim is None:
            continue
        idx = series.tindex if dim == time_dim else series.dim_index(dim)
        if isinstance(idx, td.RangeIndex):
            continue  # the constructor default — nothing to record
        indexes[dim] = _encode_index(idx, path, dim, fields)
    fields[path] = dload.codecs.npy_bytes(np.asarray(series.data))
    return {"kind": "series", "dims": list(series.dims), "indexes": indexes}


def _decode_series(desc: Mapping[str, Any], path: str, fields: Mapping[str, bytes]) -> td.Series:
    data = dload.codecs.npy_from(fields[path])
    dims = tuple(None if d is None else str(d) for d in desc["dims"])
    indexes = {
        str(dim): _decode_index(idesc, path, str(dim), fields)
        for dim, idesc in desc.get("indexes", {}).items()
    }
    return td.Series(data, dims, indexes)


def _json_value(value: Any, path: str) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    try:
        json.dumps(value)
    except TypeError as exc:
        raise TypeError(
            f"frame entry {path!r} is a plain value of non-JSON-serializable type "
            f"{type(value).__name__}; wrap arrays in td.wrap(...) and scalars in "
            "native Python types"
        ) from exc
    return value


def _frame_has_time(frame: td.Frame) -> bool:
    for key in frame:
        entry = frame[key]
        if isinstance(entry, td.Series) and entry.has_time:
            return True
        if isinstance(entry, td.Frame) and _frame_has_time(entry):
            return True
    return False


def _encode_frame(frame: td.Frame, prefix: str, fields: dict[str, bytes]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for key in frame:
        if "/" in key or "#" in key or key == _FRAME_FIELD:
            raise ValueError(
                f"frame entry name {key!r} is not serializable (reserved characters "
                f"'/', '#', and the {_FRAME_FIELD!r} field name)"
            )
        path = f"{prefix}{key}"
        entry = frame[key]
        if isinstance(entry, td.Series):
            entries[key] = _encode_series(entry, path, fields)
        elif isinstance(entry, td.Frame):
            entries[key] = _encode_frame(entry, f"{path}/", fields)
        else:
            entries[key] = {"kind": "value", "value": _json_value(entry, path)}
    desc: dict[str, Any] = {"kind": "frame", "entries": entries}
    if _frame_has_time(frame):
        desc["t_start_ticks"] = int(frame.t_start_ticks)
        desc["t_end_ticks"] = int(frame.t_end_ticks)
    return desc


def _decode_frame(desc: Mapping[str, Any], prefix: str, fields: Mapping[str, bytes]) -> td.Frame:
    entries: dict[str, Any] = {}
    for key, edesc in desc["entries"].items():
        path = f"{prefix}{key}"
        kind = edesc["kind"]
        if kind == "series":
            entries[key] = _decode_series(edesc, path, fields)
        elif kind == "frame":
            entries[key] = _decode_frame(edesc, f"{path}/", fields)
        elif kind == "value":
            entries[key] = edesc["value"]
        else:
            raise ValueError(f"unknown frame entry descriptor kind {kind!r} at {path!r}")
    t_start = desc.get("t_start_ticks")
    t_end = desc.get("t_end_ticks")
    return td.Frame(
        entries,
        t_start=int(t_start) if t_start is not None else None,
        t_end=int(t_end) if t_end is not None else None,
    )


def frame_to_sample(frame: td.Frame) -> dict[str, bytes]:
    """Serialize an arbitrary ``td.Frame`` into dload sample fields (lossless).

    Every ``td.Series`` entry — any dims, dtype, and grid/stamp/span/label
    index — becomes one ``.npy`` field (plus ``#``-suffixed index-array
    fields for stamp/span indexes); nested Frames (``"meta"`` included)
    recurse with ``/``-joined field paths; plain scalar entries are inlined
    into the single ``_frame`` JSON descriptor field. Exact inverse:
    :func:`sample_to_frame`. Publisher idiom::

        repo.commit(
            "DREGON-noise-frames",
            ((rec_id, frame_to_sample(tf)) for rec_id, tf in recordings),
            meta={"layout": TDFRAME_LAYOUT},
        )

    The ``meta={"layout": ...}`` marker is what makes
    :class:`DloadFrameDataset` auto-select :func:`decode_tdframe`.
    """
    fields: dict[str, bytes] = {}
    desc = _encode_frame(frame, "", fields)
    fields[_FRAME_FIELD] = dload.codecs.json_bytes({"format": TDFRAME_LAYOUT, "frame": desc})
    return fields


def sample_to_frame(fields: Mapping[str, bytes]) -> td.Frame:
    """Rebuild the exact ``td.Frame`` serialized by :func:`frame_to_sample`."""
    if _FRAME_FIELD not in fields:
        raise ValueError(
            f"sample has no {_FRAME_FIELD!r} descriptor field — not a "
            f"{TDFRAME_LAYOUT} sample (fields: {sorted(fields)})"
        )
    header = dload.codecs.json_from(fields[_FRAME_FIELD])
    if header.get("format") != TDFRAME_LAYOUT:
        raise ValueError(f"unsupported frame layout {header.get('format')!r}")
    return _decode_frame(header["frame"], "", fields)


def decode_tdframe(sample: Sample) -> td.Frame:
    """Per-sample decoder for ``tdframe-v1`` datasets: :func:`sample_to_frame`
    plus a ``meta.recording_id`` default (the dload sample key) so downstream
    grouping/eval code always finds one."""
    key, fields = sample
    frame = sample_to_frame(fields)
    if get_meta(frame, "recording_id") is None:
        frame = with_meta(frame, recording_id=key)
    return frame


def iter_published_frames(
    dataset: str,
    version: str | None = None,
    *,
    splits: Iterable[str] | None = None,
    repo: dload.Repository | None = None,
) -> Iterator[td.Frame]:
    """Stream a published ``tdframe-v1`` dataset as decoded ``td.Frame``s.

    The plain-Python counterpart of :class:`DloadFrameDataset` for pool-style
    consumers (``online_mixing.TimeFrameNoisePool``,
    ``noise_rps_dataset.build_noise_rps_datasets``) that want a lazy iterator
    of full recordings rather than a torch dataset. ``splits`` filters on each
    frame's ``meta.split`` (frames without the key are dropped when a filter
    is given). Exactly one decoded frame is alive per iteration — callers
    should subset what they keep before pulling the next one. ``repo`` overrides
    the shared R2 repository (e.g. :func:`local_repository` for an unpublished,
    locally-built dataset).
    """
    ds = (repo or open_repository()).dataset(str(dataset), version)
    manifest_meta = ds.manifest.meta if isinstance(ds.manifest.meta, dict) else {}
    if manifest_meta.get(LAYOUT_META_KEY) != TDFRAME_LAYOUT:
        raise ValueError(
            f"dataset {dataset!r} is not published with the {TDFRAME_LAYOUT} layout "
            f"(manifest meta: {manifest_meta!r})"
        )
    wanted = {str(s) for s in splits} if splits is not None else None
    for sample in ds.samples():
        if not is_data_sample(sample):
            continue
        frame = decode_tdframe(sample)
        if wanted is not None and str(get_meta(frame, "split", "")) not in wanted:
            continue
        yield frame


# ─── Frame combinators over dload Pipelines ────────────────────────────────────


def to_frames(pipe: dload.Pipeline, decoder: Callable[[Sample], td.Frame]) -> dload.Pipeline:
    """Raw-sample Pipeline → per-sample ``td.Frame`` Pipeline.

    Drops bookkeeping samples (:func:`is_data_sample`) and maps ``decoder``
    (a module-level function or ``partial`` — e.g.
    ``partial(decode_dregon_lm, channel=0)`` or :func:`decode_tdframe`)::

        pipe = to_frames(
            repo.dataset("DREGON-LM-V4-train").samples().shuffle(4096, seed=0),
            partial(decode_dregon_lm, sample_rate=16000, channel=0),
        ).batch(16, collate=frame_collate)
    """
    return pipe.filter(is_data_sample).map(decoder)


def _iter_frame_windows(win_s: float, hop_s: float, frame: td.Frame) -> Iterator[td.Frame]:
    t0, t1 = frame.t_start, frame.t_end
    i = 0
    while True:
        start = t0 + i * hop_s
        if start + win_s > t1 + 1e-9:
            return
        # map_data(np.copy): a tdseries slice is a *view* of the recording's
        # arrays — anything buffering windows downstream (shuffle, batch)
        # would pin the whole base recording in memory otherwise.
        yield frame.time[start : start + win_s].map_data(np.copy)
        i += 1


def frame_windows(pipe: dload.Pipeline, win_s: float, hop_s: float | None = None) -> dload.Pipeline:
    """Chop each Frame into fixed-length windows (``flat_map``, aligned slices).

    Every window is a full Frame slice — audio, RPS/telemetry tracks, and the
    invariant ``meta`` stay aligned, exactly like ``frame.time[a:b]``. Windows
    start at ``t_start + i * hop_s`` (``hop_s`` defaults to ``win_s``:
    non-overlapping) and a trailing remainder shorter than ``win_s`` is
    dropped, so downstream ``frame_collate`` sees equal-shape samples as long
    as ``win_s`` is an integer number of samples at each track's rate. Window
    data is copied out of the parent recording (safe to buffer)::

        one_sec = frame_windows(to_frames(pipe, decode_tdframe), win_s=1.0)
    """
    hop = float(win_s) if hop_s is None else float(hop_s)
    if hop <= 0 or float(win_s) <= 0:
        raise ValueError(f"frame_windows: win_s and hop_s must be > 0, got {win_s}/{hop}")
    return pipe.flat_map(partial(_iter_frame_windows, float(win_s), hop))


def _audio_ct(series: td.Series) -> np.ndarray:
    data = np.asarray(series.data, dtype=np.float32)
    return data[None, :] if data.ndim == 1 else data


def _sample_snr_db(
    snr_db: float | tuple[float, float] | Callable[[np.random.Generator], float],
    rng: np.random.Generator,
) -> float:
    if callable(snr_db):
        return float(snr_db(rng))
    if isinstance(snr_db, (int, float)):
        return float(snr_db)
    low, high = snr_db
    return float(rng.uniform(float(low), float(high)))


def _mix_frame_pair(
    snr_db: Any, seed: int, entry: str, speech_frame: td.Frame, noise_frame: td.Frame
) -> td.Frame:
    # Lazy import: online_mixing imports this module for the repository; the
    # SNR-mixing math is shared in the other direction only at call time.
    from data_processing.mixing import mix_at_source_to_noise_snr as _mix_sn

    speech = _audio_ct(speech_frame[entry])
    noise_series = noise_frame[entry]
    noise = _audio_ct(noise_series)
    if speech.shape[0] == 1 and noise.shape[0] > 1:
        speech = np.tile(speech, (noise.shape[0], 1))
    if speech.shape != noise.shape:
        raise ValueError(
            f"mix_frames: speech {speech.shape} and noise {noise.shape} are not "
            "mixable — window both streams to the same length/channels first"
        )
    rng = np.random.default_rng(
        dload.seeded(
            get_meta(speech_frame, "recording_id", ""),
            get_meta(noise_frame, "recording_id", ""),
            speech_frame.t_start_ticks,
            noise_frame.t_start_ticks,
            seed,
            "snr",
        )
    )
    snr = _sample_snr_db(snr_db, rng)
    mixture = _mix_sn(speech, noise, snr)
    if len(noise_series.dims) == 1:
        mixture = mixture[0]
    mixed = td.Series(mixture, noise_series.dims, {"time": noise_series.tindex})
    return with_meta(
        noise_frame.with_entry(entry, mixed),
        input_snr=float(snr),
        speech_id=get_meta(speech_frame, "recording_id", ""),
    )


def mix_frames(
    speech_pipe: dload.Pipeline,
    noise_pipe: dload.Pipeline,
    snr_db: float | tuple[float, float] | Callable[[np.random.Generator], float] = (-30.0, 0.0),
    *,
    seed: int = 0,
    entry: str = "mixture",
) -> dload.Pipeline:
    """Pair a speech stream with an aligned-noise stream at sampled SNR.

    ``dload.zip_with`` over two Frame pipelines: for each pair, scale the
    speech ``entry`` track to a sampled source-to-noise SNR (the project's
    standard ``mix_at_source_to_noise_snr`` math, i.e. noise level is the
    reference) and add it onto the noise Frame — which therefore *keeps* its
    aligned RPS/telemetry tracks and gains ``meta.input_snr`` /
    ``meta.speech_id``. ``snr_db`` may be a constant, a ``(low, high)``
    uniform range, or a picklable ``fn(rng) -> float``. Mono speech
    broadcasts across multichannel noise; anything else must already be
    shape-matched (use :func:`frame_windows` on both streams first)::

        train = mix_frames(speech_1s, noise_1s, snr_db=(-30.0, 0.0), seed=7)

    The SNR draw is a pure function of (both recording ids, both window
    starts, ``seed``) — reproducible per pair, worker-independent.
    """
    return dload.zip_with(
        partial(_mix_frame_pair, snr_db, int(seed), str(entry)), speech_pipe, noise_pipe
    )


def _resample_frame(sample_rate: int, entries: tuple[str, ...], frame: td.Frame) -> td.Frame:
    out = frame
    for name in entries:
        if name not in frame:
            continue
        series = frame[name]
        if not (isinstance(series, td.Series) and series.has_time):
            continue
        if not isinstance(series.tindex, td.GridIndex):
            raise ValueError(f"resample_frames: entry {name!r} is not uniformly sampled")
        if int(series.tindex.sr) == int(sample_rate) and series.tindex.sr == sample_rate:
            continue
        out = out.with_entry(name, series.resample(sample_rate))
    return out


def resample_frames(
    pipe: dload.Pipeline, sample_rate: int, *, entries: tuple[str, ...] = ("mixture", "audio")
) -> dload.Pipeline:
    """Resample the named uniformly-sampled tracks of every Frame to
    ``sample_rate`` (``tdseries`` linear resampling — fine for feature
    pipelines; decode at the target rate or resample offline when full audio
    fidelity matters). Non-grid tracks (stamped telemetry) and entries not
    present are left untouched::

        pipe_16k = resample_frames(to_frames(pipe, decode_tdframe), 16000)
    """
    return pipe.map(partial(_resample_frame, int(sample_rate), tuple(entries)))


# ─── Torch dataset (Hydra `_target_` for conf/data) ────────────────────────────


class DloadFrameDataset(IterableDataset):
    """Stream a dload-hosted dataset as per-sample ``td.Frame``s.

    The object-storage counterpart of ``DregonLMFrameDataset`` — a torch
    ``IterableDataset`` usable directly as a Hydra ``_target_`` in
    ``conf/data/*.yaml`` (see ``conf/data/dregon_lm_v4_stream.yaml``), wired
    through ``dload.torch.as_iterable_dataset`` so DataLoader workers shard
    shards exactly and ``set_epoch`` reshuffles deterministically.

    Decoding dispatches on the dataset's manifest ``meta``: datasets
    published with the generic Frame codec (``meta.layout == "tdframe-v1"``)
    decode via :func:`sample_to_frame`; anything else uses the DREGON-LM
    file-stem convention (:func:`decode_dregon_lm` with this dataset's
    ``n_fft``/``hop_length``/``sample_rate``/``channel``). Pass ``decoder``
    (any picklable ``fn(sample) -> td.Frame``) to override both.

    Parameters mirror the sibling folder datasets plus stream knobs:

    - ``dataset`` / ``version``: dload dataset name and optional version
      (default: repo-root ``dload.lock`` pin, else ``refs/latest``);
    - ``shuffle``: ``False`` (ordered), ``True`` (shuffled, fresh seed per
      construction), or an ``int`` seed for reproducible order;
    - ``shuffle_buffer`` / ``prefetch``: dload streaming-shuffle buffer and
      shard prefetch window;
    - ``take``: cap the stream at N samples (smoke tests / tiny valid sets);
    - ``repeat``: loop the dataset forever — required for the ``train:`` slot
      of the unified loop, which pulls ``samples_per_validation``-sized
      chunks from one persistent iterator (like the online-mix stream). Two
      caveats: the shard *visit order* repeats per cycle within one iterator
      (only ``set_epoch`` re-derives it; the streaming shuffle buffer keeps
      mixing across cycles), and a DataLoader worker whose shard stripe is
      empty (``num_workers`` > shard count — small datasets pack into few
      ~128 MiB shards) ends its stream instead of looping, so the remaining
      workers carry the stream.
    """

    def __init__(
        self,
        dataset: str,
        *,
        version: str | None = None,
        decoder: Callable[[Sample], td.Frame] | None = None,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channel: int | None = None,
        shuffle: bool | int = False,
        shuffle_buffer: int = 4096,
        prefetch: int = 3,
        take: int | None = None,
        repeat: bool = False,
    ) -> None:
        self.dataset_name = str(dataset)
        ds = open_repository().dataset(self.dataset_name, version)
        self.version = ds.version
        self.n_samples = len(ds)
        manifest_meta = ds.manifest.meta if isinstance(ds.manifest.meta, dict) else {}
        if decoder is None:
            if manifest_meta.get(LAYOUT_META_KEY) == TDFRAME_LAYOUT:
                decoder = decode_tdframe
            else:
                decoder = partial(
                    decode_dregon_lm,
                    n_fft=int(n_fft),
                    hop_length=int(hop_length),
                    sample_rate=int(sample_rate),
                    channel=channel,
                )
        self.decoder = decoder

        pipe = ds.samples(prefetch=int(prefetch))
        if shuffle is not False:
            seed = None if shuffle is True else int(shuffle)
            pipe = pipe.shuffle(int(shuffle_buffer), seed=seed)
        pipe = to_frames(pipe, decoder)
        if take is not None:
            pipe = pipe.take(int(take))
        self._repeat = bool(repeat)
        self._inner = as_iterable_dataset(pipe)

    def set_epoch(self, epoch: int) -> None:
        """Deterministic per-epoch reshuffle (see ``dload.torch``)."""
        self._inner.set_epoch(int(epoch))

    def __iter__(self) -> Iterator[td.Frame]:
        if not self._repeat:
            yield from self._inner
            return
        while True:
            empty = True
            for frame in self._inner:
                empty = False
                yield frame
            if empty:
                # This DataLoader worker's shard stripe is empty (more workers
                # than the dataset has shards). Ending the stream lets the
                # DataLoader retire the worker; looping would spin forever and
                # deadlock the loader waiting on this worker's first batch.
                return


# ─── Materialization + the `dload:` URI scheme ─────────────────────────────────

DLOAD_URI_PREFIX = "dload:"
#: Field values that mean "the file had no extension" when reconstructing
#: raw (CLI-committed) datasets, where field name == original extension.
_EXTENSIONLESS_FIELDS = frozenset({"data", ""})


def _field_relpath(key: str, field: str, fields_map: Mapping[str, str]) -> PurePosixPath:
    rel = PurePosixPath(key)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"refusing to materialize suspicious sample key {key!r}")
    if field in fields_map:
        # Manifest-declared mapping (processed datasets: key = sample dir,
        # field = file stem). The value is either the full original filename
        # ("mixture.wav" — what publish scripts record) or a bare extension
        # ("wav"): materialize as <key>/<original filename>.
        mapped = str(fields_map[field])
        if "." in mapped:
            return rel / mapped
        name = field if mapped in _EXTENSIONLESS_FIELDS else f"{field}.{mapped}"
        return rel / name
    # Raw datasets committed via the dload CLI: key = relpath minus
    # extension, field = the extension itself: <key>.<field>.
    if field in _EXTENSIONLESS_FIELDS:
        return rel
    return rel.with_name(f"{rel.name}.{field}")


def ensure_local(name: str, version: str | None = None) -> Path:
    """Materialize a dload dataset back into a plain directory tree.

    Streams every sample once and writes each field as a file, inverting the
    two publishing conventions (see :func:`_field_relpath`): the manifest
    ``meta["fields"]`` stem→extension mapping when present, else the raw-CLI
    field-name-is-the-extension rule. Destination is
    ``<dload cache dir>/materialized/<name>/<version[:12]>/`` — version-
    addressed and idempotent (a ``.complete`` marker short-circuits repeat
    calls); returns that path. This is the workhorse behind
    :func:`resolve_source`, letting path-based loaders consume dload
    datasets without knowing about shards.
    """
    repo = open_repository()
    manifest = repo.manifest(name, version)
    dest = Path(repo.cache.root) / "materialized" / name / manifest.version[:12]
    marker = dest / ".complete"
    if marker.exists():
        return dest
    meta = manifest.meta if isinstance(manifest.meta, dict) else {}
    fields_map: Mapping[str, str] = meta.get("fields", {}) or {}
    meta_sample = meta.get("meta_sample", {}) if isinstance(meta.get("meta_sample"), dict) else {}
    meta_key = meta_sample.get("key", "_meta")
    meta_fields: Mapping[str, str] = meta_sample.get("fields", {}) or {}
    dest.mkdir(parents=True, exist_ok=True)
    for key, fields in dload.Dataset(repo, manifest).samples():
        if key.startswith("_"):
            if key != meta_key or not meta_fields:
                continue  # bookkeeping samples with no declared file mapping
            # Dataset/split-root metadata files (e.g. metadata.json) — restore
            # them at the tree root so path-based loaders find them.
            for field, data in fields.items():
                name_in_tree = str(meta_fields.get(field, f"{field}.json"))
                (dest / name_in_tree).write_bytes(data)
            continue
        for field, data in fields.items():
            path = dest / _field_relpath(key, field, fields_map)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    marker.touch()
    return dest


def resolve_source(path_or_uri: str | Path) -> Path:
    """Resolve a data-source location that may be a plain path or a dload URI.

    - plain strings/Paths pass through unchanged (relative stays relative —
      exactly the pre-dload behaviour of every ``data_dir``/``root`` config
      knob this is wired into);
    - ``dload:NAME`` → :func:`ensure_local`'s materialized tree for NAME;
    - ``dload:NAME/sub/path`` → that tree's ``sub/path`` subdirectory/file;
    - ``dload:NAME@VERSION[/sub]`` pins a version (prefix allowed).
    """
    if isinstance(path_or_uri, Path):
        return path_or_uri
    text = str(path_or_uri)
    if not text.startswith(DLOAD_URI_PREFIX):
        return Path(text)
    spec = text[len(DLOAD_URI_PREFIX) :].lstrip("/")
    name, _, subpath = spec.partition("/")
    name, _, version = name.partition("@")
    if not name:
        raise ValueError(f"invalid dload URI {text!r}: expected dload:NAME[@VERSION][/subpath]")
    root = ensure_local(name, version or None)
    return root / subpath if subpath else root
