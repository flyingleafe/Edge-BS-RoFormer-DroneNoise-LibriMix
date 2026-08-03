"""HUSTmotor source (multi-modal motor fault dataset, acoustic channel)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tdseries as td

from data_processing.sources._common import audio_frame, meta_frame, safe_key


def _read_hust_txt(path: Path) -> tuple[np.ndarray, np.ndarray | None, int] | None:
    """Parse a HUSTmotor ``.txt`` (text header + tab-separated
    ``time, X, Y, Z, Sound``). Returns ``(acoustic (1,N), vibration (3,N)|None,
    sr)`` — the acoustic ``Sound`` channel is the last column."""
    data_start: int | None = None
    with open(path) as fh:
        for i, line in enumerate(fh):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    [float(x) for x in parts]
                    data_start = i
                    break
                except ValueError:
                    continue
    if data_start is None:
        return None
    arr = np.loadtxt(str(path), skiprows=data_start, delimiter="\t")
    if arr.ndim == 1:
        arr = arr[:, None]
    time = arr[:, 0]
    sr = 25600
    if len(time) > 2:
        dt = float(np.median(np.diff(time)))
        if dt > 0:
            sr = int(round(1.0 / dt))
    acoustic = arr[:, -1][None, :].astype(np.float32)  # "Sound" = last data column
    vibration = arr[:, 1:-1].T.astype(np.float32) if arr.shape[1] > 2 else None  # X, Y, Z
    return acoustic, vibration, sr


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """25.6 kHz instrument ``.txt`` (6 health × 4 speeds). Columns are
    ``time, X, Y, Z, Sound`` — ``audio`` is the acoustic ``Sound`` channel; the
    3-axis vibration is kept as a separate ``vibration`` track."""
    health_map = {
        "H": "healthy",
        "BF": "bearing_fault",
        "BOW": "bowed_rotor",
        "BROKEN": "broken_rotor_bars",
        "MISAL": "misalignment",
        "UNBAL": "voltage_unbalance",
    }
    for txt in sorted(raw_dir.rglob("*.txt")):
        parsed = _read_hust_txt(txt)
        if parsed is None:
            continue
        acoustic, vibration, sr = parsed
        rid = txt.stem
        toks = re.split(r"[_\-]", rid)
        health = next((health_map[t.upper()] for t in toks if t.upper() in health_map), None)
        speed = next((t for t in toks if re.fullmatch(r"\d+\s*[Hh][Zz]", t)), None)
        meta = meta_frame(
            rid,
            "HUSTmotor",
            system={"category": "motor", "health": health, "testbed": "spectraquest_mfs"},
            observation={
                "type": "fixed_mic_bench",
                "source_motion": "static",
                "relative_trajectory": "none",
            },
            operating={"speed": speed, "mic_channel": "Sound"},
            label={"health": health},
            extra={"raw_relpath": str(txt.relative_to(raw_dir)), "vibration_channels": "X,Y,Z"},
        )
        frame = audio_frame(acoustic, sr, meta)
        if vibration is not None:
            frame = frame.with_entry("vibration", td.uniform(vibration, sr, dims=("axis", "time")))
        yield safe_key(rid), frame


PROVENANCE = {
    "source_url": "https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset",
    "license": "unspecified (research use; contact author)",
    "citation": "Zhao, HUSTmotor multi-modal dataset.",
    "collection_method": "SpectraQuest mechanical fault simulator; synchronized vibration + acoustic",
    "equipment": "accelerometer + microphone, 25.6 kHz",
    "observation_type": "fixed_mic_bench",
    "sample_rate": 25600,
    "channels": "varies (channel roles unconfirmed)",
    "description": "6 health states × 4 speeds (5/10/20/30 Hz) as numeric .txt. NOTE: unlicensed; channel roles unconfirmed.",
}
