"""HuggingFace parquet drone sources: DroneAudioSet + drone-detection-samples.

Both HF datasets are parquet-only (no raw wav tree): drone-detection embeds
audio as encoded bytes + an int label; DroneAudioSet embeds a decoded
(channel, time) array + sampling_rate + file_path. The builders read the
snapshotted parquet shards row-batched (bulk download + local reads are far
faster and more reliable than per-row fsspec range requests over throttled
unauthenticated HF). The per-row decoders are pure (testable without network).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.sources._common import audio_frame, meta_frame, safe_key


def _row_audio_from_bytes(audio_struct: dict) -> tuple[np.ndarray, int]:
    """HF ``Audio`` struct ``{bytes, path}`` → ``((C, T) float32, sr)``."""
    import io

    import soundfile as sf

    raw, sr = sf.read(io.BytesIO(audio_struct["bytes"]), dtype="float32", always_2d=True)
    return np.ascontiguousarray(raw.T), int(sr)


def _row_audio_from_array(audio_struct: dict) -> tuple[np.ndarray, int]:
    """HF decoded ``Audio`` struct ``{array, sampling_rate}`` → ``((C, T), sr)``.

    Orients so the smaller axis is channels (channels ≤ time for real audio)."""
    arr = np.asarray(audio_struct["array"], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return np.ascontiguousarray(arr), int(audio_struct["sampling_rate"])


def _decode_drone_detection_row(idx: int, row: dict) -> tuple[str, td.Frame]:
    audio, sr = _row_audio_from_bytes(row["audio"])
    label = row.get("label")
    cls = None if label is None else ("drone" if int(label) == 1 else "no_drone")
    path = row["audio"].get("path") or f"row_{idx}"
    rid = f"{idx:06d}_{Path(str(path)).stem}"
    meta = meta_frame(
        rid,
        "drone-detection-samples",
        system={"category": "drone"},
        observation={"type": "unknown", "source_motion": "unknown", "relative_trajectory": "none"},
        label={"class": cls, "raw_label": None if label is None else int(label)},
        extra={"raw_path": str(path)},
    )
    return safe_key(rid), audio_frame(audio, sr, meta)


def _droneaudioset_frame(
    idx: int, audio: np.ndarray, sr: int, file_path: str | None, data_type: str | None
) -> tuple[str, td.Frame]:
    """Build a DroneAudioSet frame from already-decoded audio + row fields."""
    fp = str(file_path or f"row_{idx}")
    low = fp.lower()
    subsets = ("drone-with-source", "drone-only", "source-only", "ground-truth")
    subset = next((s for s in subsets if s in low), data_type)
    dtok = re.search(r"drone\d", low)
    dist_m = None
    mdist = re.search(r"mic-?dist-?(\d+)\s*cm", low)
    if mdist:
        dist_m = int(mdist.group(1)) / 100.0
    throttle = next((t for t in ("low", "high") if f"throttle-{t}" in low), None)
    rid = f"{idx:06d}_{fp}"
    meta = meta_frame(
        rid,
        "DroneAudioSet",
        system={"category": "drone", "drone_token": None if dtok is None else dtok.group(0)},
        observation={
            "type": "rig_mounted_static",
            "source_motion": "static",
            "mic_to_source_m": dist_m,
            "relative_trajectory": "none",
        },
        operating={"throttle": throttle},
        label={"subset": subset, "data_type": data_type},
        extra={"raw_path": fp},
    )
    return safe_key(rid), audio_frame(audio, sr, meta)


def _decode_droneaudioset_row(idx: int, row: dict) -> tuple[str, td.Frame]:
    audio, sr = _row_audio_from_array(row["audio"])
    fp = row.get("file_path") or row["audio"].get("path")
    return _droneaudioset_frame(idx, audio, sr, fp, row.get("data_type"))


def _arrow_list_scalar_to_ct(scalar: Any) -> np.ndarray:
    """A parquet ``list<list<double>>`` audio scalar → ``(C, T) float32``.

    Reads the flat child buffer once and reshapes — no per-element/per-row-dim
    Python iteration. Works for either orientation; the axis that comes out
    larger is time, so channels end up first."""
    inner = scalar.values  # ListArray: n_outer inner lists (all equal length)
    n_outer = len(inner)
    # .flatten() respects this row's offsets (unlike .values, which returns the
    # whole column's child buffer); it's a C-level concat, no Python iteration.
    flat = np.asarray(inner.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    if n_outer == 0 or flat.size == 0:
        return np.zeros((1, 0), dtype=np.float32)
    audio = flat.reshape(n_outer, flat.size // n_outer)
    if audio.shape[0] > audio.shape[1]:
        audio = audio.T  # make channels (the smaller axis) first
    return np.ascontiguousarray(audio)


def _iter_local_parquet(
    raw_dir: Path, decode: Callable[[int, dict], tuple[str, td.Frame]], batch_size: int
) -> Iterator[tuple[str, td.Frame]]:
    """``decode(idx, row)`` over every snapshotted parquet shard under ``raw_dir``."""
    import pyarrow.parquet as pq

    idx = 0
    for path in sorted(Path(raw_dir).rglob("*.parquet")):
        for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield decode(idx, row)
                idx += 1


def build_drone_detection(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """geronimobasso drone-audio-detection-samples (HF parquet): mono 16 kHz
    clips, binary drone/no-drone. Reads snapshotted parquet under ``raw_dir``."""
    yield from _iter_local_parquet(raw_dir, _decode_drone_detection_row, batch_size=16)


def build_droneaudioset(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """DroneAudioSet (HF parquet): rig-mounted static drone; subset + throttle +
    mic distance from ``file_path``; multichannel arrays.

    The audio arrays are huge, so the ``audio.array`` struct field is read via
    arrow buffers (``_arrow_list_scalar_to_ct``) rather than ``to_pylist`` —
    only the small ``file_path``/``sampling_rate``/``data_type`` columns are
    materialized to Python."""
    import pyarrow.parquet as pq

    idx = 0
    for path in sorted(Path(raw_dir).rglob("*.parquet")):
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=8):
            names = set(batch.schema.names)
            audio_col = batch.column("audio")
            arrays = audio_col.field("array")
            srs = audio_col.field("sampling_rate").to_pylist()
            n = batch.num_rows
            fpaths = batch.column("file_path").to_pylist() if "file_path" in names else [None] * n
            dtypes = batch.column("data_type").to_pylist() if "data_type" in names else [None] * n
            for i in range(n):
                audio = _arrow_list_scalar_to_ct(arrays[i])
                yield _droneaudioset_frame(idx, audio, int(srs[i]), fpaths[i], dtypes[i])
                idx += 1


DRONEAUDIOSET_PROVENANCE = {
    "source_url": "https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet",
    "doi": "10.48550/arXiv.2510.15383",
    "license": "MIT",
    "citation": "DroneAudioSet (arXiv:2510.15383).",
    "collection_method": "rig-mounted static quadcopters; clean drone-only + source-only stems + real mixtures; SNR -57..-2.5 dB",
    "equipment": "two 8-ch MEMS circular arrays (above/below) + central mic (17 ch), distances 25/50 cm",
    "observation_type": "rig_mounted_static",
    "channels": "varies (verify per file)",
    "description": "Drone speech-enhancement dataset: 2 quads, 2 throttles, 3 rooms; drone-only/source-only/mixed/ground-truth subsets.",
}

DRONE_DETECTION_PROVENANCE = {
    "source_url": "https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples",
    "license": "MIT",
    "citation": "geronimobasso/drone-audio-detection-samples (HuggingFace).",
    "collection_method": "aggregated open-licensed drone/no-drone detection clips",
    "observation_type": "unknown",
    "sample_rate": 16000,
    "channels": 1,
    "description": "180k mono 16 kHz clips, binary drone/no-drone; provenance mixed (attribution may flow through).",
}
