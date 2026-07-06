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

import numpy as np
import tdseries as td


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
