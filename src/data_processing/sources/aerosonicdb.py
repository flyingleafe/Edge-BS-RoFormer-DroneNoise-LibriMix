"""AeroSonicDB-YPAD0523 source (aircraft flyover + rich metadata)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import tdseries as td

from data_processing.sources._common import (
    audio_frame,
    find_csv,
    iter_audio_files,
    meta_frame,
    read_audio_file,
    safe_key,
)


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Mono 22.05 kHz aircraft flyover, driven by ``sample_meta.csv``
    (+ ``aircraft_meta.csv`` by ``hex_id``)."""
    import pandas as pd

    smeta = find_csv(raw_dir, "sample_meta.csv")
    if smeta is None:
        raise FileNotFoundError("AeroSonicDB: sample_meta.csv not found in raw dir")
    sdf = pd.read_csv(smeta)
    sdf.columns = [str(c).lower() for c in sdf.columns]
    ameta = find_csv(raw_dir, "aircraft_meta.csv")
    adf = None
    if ameta is not None:
        adf = pd.read_csv(ameta)
        adf.columns = [str(c).lower() for c in adf.columns]
        if "hex_id" in adf.columns:
            adf = adf.set_index("hex_id")
    files = {p.name: p for p in iter_audio_files(raw_dir)}
    for _, row in sdf.iterrows():
        fn = str(row.get("filename", "")).strip()
        wav = files.get(fn) or files.get(fn if fn.endswith(".wav") else f"{fn}.wav")
        if wav is None:
            continue
        r = {k: row.get(k) for k in sdf.columns}
        hex_id = r.get("hex_id")
        if adf is not None and hex_id in adf.index:
            for k, v in adf.loc[hex_id].to_dict().items():
                r.setdefault(k, v)
        audio, sr = read_audio_file(wav)
        rid = Path(fn).stem
        meta = meta_frame(
            rid,
            "AeroSonicDB",
            system={
                "category": "aircraft",
                "make": r.get("manu"),
                "model": r.get("model"),
                "engine_type": r.get("engtype"),
                "engine_count": r.get("engnum"),
                "prop_model": r.get("propmodel"),
                "type_designator": r.get("typedesig"),
                "hex_id": None if hex_id is None else str(hex_id),
            },
            observation={
                "type": "ground_flyover",
                "source_motion": "moving",
                "relative_trajectory": "scalar_altitude",
                "mic": r.get("mic"),
                "location": r.get("location"),
            },
            operating={"altitude_ft": r.get("altitude"), "duration_s": r.get("duration")},
            label={"class": r.get("class"), "subclass": r.get("subclass")},
            extra={"raw_relpath": str(wav.relative_to(raw_dir)), "fold": r.get("fold")},
        )
        yield safe_key(rid), audio_frame(audio, sr, meta)


PROVENANCE = {
    "source_url": "https://zenodo.org/records/8371595",
    "doi": "10.5281/zenodo.8371595",
    "license": "CC BY-NC 4.0",
    "citation": "Downes et al., AeroSonicDB (YPAD-0523).",
    "collection_method": "ADS-B-triggered ground recordings of low-altitude aircraft flyover",
    "equipment": "ground microphone (Shure SM58 / Samson Go Mic)",
    "observation_type": "ground_flyover",
    "sample_rate": 22050,
    "channels": 1,
    "description": "Labelled aircraft flyover audio + rich aircraft/engine/prop metadata; only scalar altitude per event.",
}
