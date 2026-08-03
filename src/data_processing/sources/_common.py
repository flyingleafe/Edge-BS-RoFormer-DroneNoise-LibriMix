"""Shared frame-building helpers for source-dataset builders.

Every builder in this package turns a raw download into a stream of rich
recording ``td.Frame``s (layout ``tdframe-v1``): an ``audio`` Series whose
params are read *from the files* (never hardcoded), documented geometry
(``mic_pos``/``source_pos`` where known), and a nested ``meta`` Frame with the
project's per-sample schema — ``system`` / ``observation`` / ``operating`` /
``label`` groups plus the raw relpath, so an unexpected path token degrades to
a preserved-but-unparsed field rather than a crash.

Torch-free: only numpy / soundfile / scipy / pandas (lazy) — so registry
integrity and synthetic build round-trips run on the small box.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.frames import audio_series

#: Manifest ``meta["layout"]`` value every frames dataset in this package uses.
LAYOUT = "tdframe-v1"


def clean_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values, coerce numpy scalars to native Python (JSON-safe)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[str(k)] = v
    return out


def meta_frame(
    recording_id: str,
    dataset: str,
    *,
    system: dict | None = None,
    observation: dict | None = None,
    operating: dict | None = None,
    label: dict | None = None,
    extra: dict | None = None,
) -> td.Frame:
    """Per-sample metadata as a nested invariant ``td.Frame``."""
    entries: dict[str, Any] = {"recording_id": str(recording_id), "dataset": str(dataset)}
    for name, group in (
        ("system", system),
        ("observation", observation),
        ("operating", operating),
        ("label", label),
    ):
        if group:
            cleaned = clean_dict(group)
            if cleaned:
                entries[name] = td.Frame(cleaned)
    if extra:
        entries.update(clean_dict(extra))
    return td.Frame(entries)


def audio_frame(
    audio_ct: np.ndarray,
    sample_rate: int,
    meta: td.Frame,
    *,
    mic_pos: np.ndarray | None = None,
    source_pos: np.ndarray | None = None,
) -> td.Frame:
    """``(C, T)`` audio + geometry + ``meta`` → a recording Frame."""
    entries: dict[str, Any] = {
        "audio": audio_series(np.ascontiguousarray(audio_ct), int(sample_rate))
    }
    if mic_pos is not None:
        entries["mic_pos"] = td.wrap(np.asarray(mic_pos, dtype=np.float64), dims=("mic", None))
    if source_pos is not None:
        entries["source_pos"] = td.wrap(
            np.asarray(source_pos, dtype=np.float64), dims=("source", None)
        )
    entries["meta"] = meta
    return td.Frame(entries)


def mic_ring(n: int, radius: float) -> np.ndarray:
    """``(n, 3)`` positions of ``n`` mics evenly on a circle (xy-plane, z=0)."""
    ang = 2.0 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], axis=1)


def safe_key(text: str) -> str:
    """dload sample key: filesystem-neutral, never leading ``_`` (reserved)."""
    key = re.sub(r"[^0-9A-Za-z._-]+", "_", text.replace("/", "__")).strip("_")
    return key or "sample"


def iter_audio_files(root: Path, suffixes: tuple[str, ...] = (".wav", ".flac")) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Decode wav/flac → ``((C, T) float32, sr)`` (params from the file)."""
    import soundfile as sf

    raw, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    return np.ascontiguousarray(raw.T), int(sr)


def find_csv(root: Path, name: str) -> Path | None:
    for p in root.rglob(name):
        return p
    # case-insensitive fallback
    for p in root.rglob("*.csv"):
        if p.name.lower() == name.lower():
            return p
    return None
