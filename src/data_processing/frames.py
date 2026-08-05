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


#: Rotor-track entry names recognised in published recording frames, in
#: preference order: the generic ``rps`` (michaels-frames), then DREGON-frames'
#: ``motors_measured`` (the real tachometer), then ``motors_command`` (the
#: commanded track, *already cleaned* at publish time).
#:
#: ``motors_measured`` is preferred because it is what the rotors actually did:
#: it is the track the beat-VK evaluation protocol pins as ground truth
#: (``scripts/beatvk_eval.py``), and it shows the real spindown that the
#: command track's trailing logging freeze hides. Only the 5 DREGON
#: ``free-flight_*_room1`` recordings carry it; everything else falls through
#: to ``motors_command``. The two tracks share one timestamp vector and agree
#: to 0.04 % on average (``docs/experiments/dregon-telemetry-forensics.md`` § 1), so
#: the swap is a consistency fix, not a change of regime.
PUBLISHED_RPS_KEYS = ("rps", "motors_measured", "motors_command")


def adapt_recording_frame(frame: td.Frame, *, sample_rate: int) -> td.Frame | None:
    """Rich published recording -> the minimal (audio + rps) noise-source Frame.

    Published frames datasets (``DREGON-frames`` / ``michaels-frames`` / any
    :mod:`data_processing.sources` builder output) carry their fixes baked in
    — DREGON's ``motors_command`` is already ``clean_command_spikes``-cleaned
    and michaels' ``rps`` is already aligned. The rotor track is therefore
    stored under the generic ``rps`` name, which ``mixing.resolve_motor_tracks``
    treats as needing **no** cleaning, so no fix logic is ever re-applied at
    load time. Everything else (IMU, raw telemetry, geometry, per-sample
    clocks) is dropped so pools keep only what they slice; audio is
    soxr-resampled to ``sample_rate``. Returns ``None`` for frames without
    audio or a rotor track (e.g. clean-source recordings).
    """
    if "audio" not in frame:
        return None
    rps_key = next((k for k in PUBLISHED_RPS_KEYS if k in frame), None)
    if rps_key is None:
        return None
    entries: dict[str, Any] = {
        "audio": resample_audio_series(frame["audio"], int(sample_rate)),
        "rps": frame[rps_key],
    }
    if "meta" in frame:
        entries["meta"] = frame["meta"]
    return td.Frame(entries)


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
