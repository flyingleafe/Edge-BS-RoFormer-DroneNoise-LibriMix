"""HornBase source (vehicle horns — tonal, not rotating-source)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import tdseries as td

from data_processing.sources._common import (
    audio_frame,
    iter_audio_files,
    meta_frame,
    read_audio_file,
    safe_key,
)


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Stereo 44.1 kHz 1 s clips. Label parsed best-effort from the filename
    ('not'/'no' → not-horn)."""
    for wav in iter_audio_files(raw_dir):
        rel = wav.relative_to(raw_dir)
        low = wav.name.lower()
        cls = None
        if "horn" in low:
            cls = "not_horn" if re.search(r"\b(not|no)\b|nothorn|non", low) else "horn"
        audio, sr = read_audio_file(wav)
        rid = str(rel)
        meta = meta_frame(
            rid,
            "HornBase",
            system={"category": "vehicle_horn"},
            observation={
                "type": "ground",
                "source_motion": "moving",
                "relative_trajectory": "none",
            },
            label={"class": cls},
            extra={"raw_relpath": rid},
        )
        yield safe_key(rid), audio_frame(audio, sr, meta)


PROVENANCE = {
    "source_url": "https://data.mendeley.com/datasets/y5stjsnp8s/2",
    "doi": "10.17632/y5stjsnp8s.2",
    "license": "CC BY 4.0",
    "citation": "HornBase — A Car Horns Dataset (Data in Brief).",
    "collection_method": "two-smartphone recording of vehicle horns in traffic scenarios",
    "observation_type": "ground",
    "sample_rate": 44100,
    "channels": 2,
    "description": "1,080 stereo 44.1 kHz 1 s clips, horn/not-horn. NOTE: horns are tonal, not rotating-source.",
}
