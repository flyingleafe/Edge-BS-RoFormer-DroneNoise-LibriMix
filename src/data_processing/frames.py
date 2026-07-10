"""Shared ``tdseries.Frame`` helpers for the dataset adapters in this package.

Ports the old ``TimeFrame.tags`` / ``TimeFrame.global_data`` conventions to
``tdseries``: per-recording scalar metadata lives in a nested *invariant*
``td.Frame`` under the entry name ``"meta"`` (it survives time slicing
untouched); geometry arrays live under ``"mic_pos"`` / ``"rotor_pos"`` as
``td.wrap``-ed Series sharing the ``"mic"`` / ``"rotor"`` dim with the audio
and rotor-speed tracks. All call sites in this package go through the helpers
below instead of ad-hoc ``frame["meta"]`` chains.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import tdseries as td


def audio_series(audio: np.ndarray, sample_rate: int) -> td.Series:
    """``(C, T)`` -> mono ``(time,)`` Series (``C == 1``) or ``(mic, time)``.

    The one canonical audio-array-to-Series convention shared by every
    dataset adapter (``frame_datasets``) and the dload bridge (``streams``);
    matches ``tasks.task``'s ``n_channels=None`` vs ``n_channels=C`` split.
    """
    if audio.shape[0] == 1:
        return td.uniform(audio[0], sample_rate, dims=("time",), t_start=0.0)
    return td.uniform(audio, sample_rate, dims=("mic", "time"), t_start=0.0)


def rps_series(rps: np.ndarray, *, sample_rate: int, hop_length: int) -> td.Series:
    """``(rotor, n_frames)`` array -> Series on the exact ``sr/hop`` STFT grid."""
    n_frames = rps.shape[-1]
    idx = td.GridIndex.create((sample_rate, hop_length), n_frames, t_start=0)
    return td.Series(rps, ("rotor", "time"), {"time": idx})


def resample_audio_series(series: td.Series, sample_rate: int) -> td.Series:
    """Resample a uniformly-sampled audio Series to ``sample_rate``.

    The same audio-fidelity resampling the folder loaders apply
    (``dregon.load_timeframe(target_sr=...)``,
    ``michaels.load_michaels_timeframes(sr=...)``): ``librosa.resample`` with
    ``res_type="soxr_hq"`` along the last (time) axis — **not** the linear
    ``tdseries`` resample, which is feature-grade only. Dims and the absolute
    ``t_start`` are preserved; a no-op when the rate already matches.
    """
    idx = series.tindex
    if not isinstance(idx, td.GridIndex):
        raise ValueError("resample_audio_series requires a uniformly-sampled (GridIndex) Series")
    if idx.sr == sample_rate:
        return series
    data = np.asarray(series.data, dtype=np.float32)
    resampled = librosa.resample(
        data, orig_sr=float(idx.sr), target_sr=int(sample_rate), axis=-1, res_type="soxr_hq"
    )
    return td.uniform(resampled, int(sample_rate), dims=series.dims, t_start=series.t_start)


def get_meta(frame: td.Frame, key: str, default: Any = None) -> Any:
    """Look up one scalar metadata key from ``frame["meta"]``."""
    if "meta" not in frame:
        return default
    meta = frame["meta"]
    if key not in meta:
        return default
    return meta[key]


def meta_dict(frame: td.Frame) -> dict[str, Any]:
    """Materialise ``frame["meta"]`` as a plain dict (``{}`` if absent)."""
    if "meta" not in frame:
        return {}
    meta = frame["meta"]
    return {k: meta[k] for k in meta}


def with_meta(frame: td.Frame, **updates: Any) -> td.Frame:
    """Return a new Frame with its ``"meta"`` entries merged with *updates*."""
    merged = meta_dict(frame)
    merged.update(updates)
    return frame.with_entry("meta", td.Frame(dict(merged)))


def make_recording_frame(
    tracks: dict[str, td.Series],
    *,
    meta: dict[str, Any],
    mic_pos: np.ndarray | None = None,
    rotor_pos: np.ndarray | None = None,
) -> td.Frame:
    """Build a standard recording Frame: tracks + ``"meta"`` (+ geometry).

    ``tracks`` are the temporal entries (``"audio"``, motor/RPS tracks, IMU,
    ...). ``mic_pos`` / ``rotor_pos``, when given, are wrapped as invariant
    Series sharing the ``"mic"`` / ``"rotor"`` dim with any track that uses
    those dim names (e.g. ``dims=("mic", "time")`` audio).
    """
    entries: dict[str, Any] = dict(tracks)
    if mic_pos is not None:
        entries["mic_pos"] = td.wrap(mic_pos, dims=("mic", None))
    if rotor_pos is not None:
        entries["rotor_pos"] = td.wrap(rotor_pos, dims=("rotor", None))
    entries["meta"] = td.Frame(dict(meta))
    return td.Frame(entries)
