"""MIMII + MIMII-DG sources (industrial machines, DCASE)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tdseries as td

from data_processing.sources._common import (
    audio_frame,
    iter_audio_files,
    meta_frame,
    mic_ring,
    read_audio_file,
    safe_key,
)

_MACHINES = ("fan", "pump", "slider", "valve", "gearbox", "bearing")


def build_mimii(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """MIMII: ``<snr>_dB_<machine>/<machine>/id_<NN>/{normal,abnormal}/*.wav``,
    8-ch 16 kHz. SNR/machine/unit/condition from the path; nominal 8-mic ring."""
    for wav in iter_audio_files(raw_dir):
        rel = wav.relative_to(raw_dir)
        snr: int | None = None
        machine: str | None = None
        unit: str | None = None
        condition: str | None = None
        for tok in rel.parts:
            m = re.match(r"(-?\d+)_dB_([A-Za-z]+)", tok)
            if m:
                snr = int(m.group(1))
                machine = machine or m.group(2).lower()
            if tok.lower() in _MACHINES:
                machine = machine or tok.lower()
            if re.fullmatch(r"id_?\d+", tok, re.IGNORECASE):
                unit = tok.lower()
            if tok.lower() in ("normal", "abnormal", "anomaly"):
                condition = "abnormal" if tok.lower() != "normal" else "normal"
        rid = "_".join(
            x
            for x in (machine, None if snr is None else f"snr{snr}", unit, condition, wav.stem)
            if x
        )
        audio, sr = read_audio_file(wav)
        meta = meta_frame(
            rid,
            "MIMII",
            system={"category": "industrial_machine", "machine_type": machine, "unit_id": unit},
            observation={
                "type": "fixed_array_bench",
                "source_motion": "static",
                "mic_to_source_m": 0.5,
                "array": "circular_8ch_nominal",
                "relative_trajectory": "none",
            },
            operating={"snr_db": snr, "background": "factory_noise"},
            label={"class": machine, "normal_vs_anomaly": condition},
            extra={"raw_relpath": str(rel)},
        )
        frame = audio_frame(
            audio,
            sr,
            meta,
            mic_pos=mic_ring(audio.shape[0], 0.05),
            source_pos=np.array([[0.5, 0.0, 0.0]]),
        )
        yield safe_key(rid), frame


def _parse_attr_pairs(stem: str) -> dict[str, str]:
    """MIMII-DG filename attribute pairs like ``vel_1200`` / ``f-n_A`` → dict."""
    out: dict[str, str] = {}
    for tok in stem.split("_"):
        m = re.fullmatch(r"([A-Za-z-]+)-([\w.]+)", tok)  # e.g. f-n_... handled loosely
        if m:
            out[m.group(1)] = m.group(2)
    return out


def build_mimii_dg(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """MIMII-DG: mono 16 kHz; ``<machine>/.../section_NN_<domain>_<split>_<label>_<idx>_<attrs>.wav``."""
    for wav in iter_audio_files(raw_dir):
        rel = wav.relative_to(raw_dir)
        machine = next((t.lower() for t in rel.parts if t.lower() in _MACHINES), None)
        stem = wav.stem
        toks = stem.split("_")
        section = next((f"{a}_{b}" for a, b in zip(toks, toks[1:]) if a == "section"), None)
        domain = next((t for t in toks if t in ("source", "target")), None)
        split = next((t for t in toks if t in ("train", "test")), None)
        condition = next((t for t in toks if t in ("normal", "anomaly")), None)
        audio, sr = read_audio_file(wav)
        rid = f"{machine}_{stem}" if machine else stem
        meta = meta_frame(
            rid,
            "MIMII-DG",
            system={"category": "industrial_machine", "machine_type": machine},
            observation={
                "type": "fixed_mic_bench",
                "source_motion": "static",
                "relative_trajectory": "none",
            },
            operating={
                "section": section,
                "domain": domain,
                "split": split,
                **_parse_attr_pairs(stem),
            },
            label={"class": machine, "normal_vs_anomaly": condition},
            extra={"raw_relpath": str(rel)},
        )
        yield safe_key(rid), audio_frame(audio, sr, meta)


MIMII_PROVENANCE = {
    "source_url": "https://zenodo.org/records/3384388",
    "doi": "10.5281/zenodo.3384388",
    "license": "CC BY-SA 4.0",
    "citation": "Purohit et al., MIMII Dataset (DCASE 2019).",
    "collection_method": "clean machine sound mixed with real factory background at 6/0/-6 dB SNR",
    "equipment": "circular 8-mic array (TAMAGO-03), machine ~0.5 m away",
    "observation_type": "fixed_array_bench",
    "sample_rate": 16000,
    "channels": 8,
    "description": "Industrial machines (fan/pump/slider/valve) normal + anomalous, 8-ch 16 kHz, 10 s; 3 SNR tiers.",
    "geometry_note": "mic_pos is a NOMINAL 8-mic ring (r=0.05 m); source_pos nominal 0.5 m on x.",
}

MIMII_DG_PROVENANCE = {
    "source_url": "https://zenodo.org/records/6529888",
    "doi": "10.5281/zenodo.6529888",
    "license": "CC BY-NC-SA 4.0",
    "citation": "Dohi et al., MIMII DG (DCASE 2022 Task 2 dev set).",
    "collection_method": "domain-generalization machine sounds; source/target sections",
    "equipment": "single-channel machine recording",
    "observation_type": "fixed_mic_bench",
    "sample_rate": 16000,
    "channels": 1,
    "description": "fan/gearbox/bearing/slider/valve mono 16 kHz; domain-shift sections; attrs (vel_/volt_) in filename.",
}
