"""Registry of external harmonic-noise datasets → streamable ``tdframe-v1``.

Each entry pairs a pinned :class:`DownloadSpec` (how to fetch, + provenance /
license) with a ``builder(raw_dir) -> Iterator[(key, td.Frame)]`` that turns the
raw files into rich recording Frames: an ``audio`` Series (params read *from the
files*, never hardcoded), documented geometry (``mic_pos``/``source_pos`` where
known), and a nested ``meta`` Frame with the project's per-sample schema —
``system`` / ``observation`` / ``operating`` / ``label`` groups plus the raw
relpath, so an unexpected path token degrades to a preserved-but-unparsed field
rather than a crash.

Publish with :mod:`scripts.publish_external_datasets` (the ``tdframe-v1`` codec
in :mod:`data_processing.streams`); harmonicity is measured separately
(:mod:`data_processing.harmonicity`) in the analysis stage. Design +
per-dataset notes: ``docs/external-datasets-plan.md``.

Torch-free: only numpy / soundfile / scipy / pandas (lazy) — so registry
integrity and a synthetic build round-trip run on the small box.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.frames import audio_series

__all__ = [
    "DownloadSpec",
    "ExternalDataset",
    "EXTERNAL_SPECS",
    "get",
    "list_names",
    "dataset_meta",
    "download_dataset",
]

LAYOUT = "tdframe-v1"

Builder = Callable[[Path], Iterator[tuple[str, td.Frame]]]


@dataclass(frozen=True)
class DownloadSpec:
    """How to fetch a dataset's raw files (pinned)."""

    kind: str  # "zenodo" | "mendeley" | "hf" | "gdrive"
    params: dict[str, Any]
    extract: bool = False  # unzip every *.zip after fetch (into <stem>/)


@dataclass(frozen=True)
class ExternalDataset:
    name: str
    download: DownloadSpec
    builder: Builder
    provenance: dict[str, Any]
    modality: str = "audio"
    fields: tuple[str, ...] = ("audio",)
    streaming: bool = False  # builder streams from the hub; no local raw dir


# ─── Frame-building helpers ────────────────────────────────────────────────────


def _clean(d: dict[str, Any]) -> dict[str, Any]:
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


def _meta_frame(
    recording_id: str,
    dataset: str,
    *,
    system: dict | None = None,
    observation: dict | None = None,
    operating: dict | None = None,
    label: dict | None = None,
    extra: dict | None = None,
) -> td.Frame:
    """Per-sample metadata as a nested invariant ``td.Frame`` (see module doc)."""
    entries: dict[str, Any] = {"recording_id": str(recording_id), "dataset": str(dataset)}
    for name, group in (
        ("system", system),
        ("observation", observation),
        ("operating", operating),
        ("label", label),
    ):
        if group:
            cleaned = _clean(group)
            if cleaned:
                entries[name] = td.Frame(cleaned)
    if extra:
        entries.update(_clean(extra))
    return td.Frame(entries)


def _audio_frame(
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


def _mic_ring(n: int, radius: float) -> np.ndarray:
    """``(n, 3)`` positions of ``n`` mics evenly on a circle (xy-plane, z=0)."""
    ang = 2.0 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], axis=1)


def _safe_key(text: str) -> str:
    """dload sample key: filesystem-neutral, never leading ``_`` (reserved)."""
    key = re.sub(r"[^0-9A-Za-z._-]+", "_", text.replace("/", "__")).strip("_")
    return key or "sample"


def _iter_audio_files(root: Path, suffixes: tuple[str, ...] = (".wav", ".flac")) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def _read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Decode wav/flac → ``((C, T) float32, sr)`` (params from the file)."""
    import soundfile as sf

    raw, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    return np.ascontiguousarray(raw.T), int(sr)


def _find_csv(root: Path, name: str) -> Path | None:
    for p in root.rglob(name):
        return p
    # case-insensitive fallback
    for p in root.rglob("*.csv"):
        if p.name.lower() == name.lower():
            return p
    return None


# ─── Per-dataset builders ──────────────────────────────────────────────────────

_MIMII_MACHINES = ("fan", "pump", "slider", "valve", "gearbox", "bearing")


def build_mimii(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """MIMII: ``<snr>_dB_<machine>/<machine>/id_<NN>/{normal,abnormal}/*.wav``,
    8-ch 16 kHz. SNR/machine/unit/condition from the path; nominal 8-mic ring."""
    for wav in _iter_audio_files(raw_dir):
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
            if tok.lower() in _MIMII_MACHINES:
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
        audio, sr = _read_audio_file(wav)
        meta = _meta_frame(
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
        frame = _audio_frame(
            audio,
            sr,
            meta,
            mic_pos=_mic_ring(audio.shape[0], 0.05),
            source_pos=np.array([[0.5, 0.0, 0.0]]),
        )
        yield _safe_key(rid), frame


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
    for wav in _iter_audio_files(raw_dir):
        rel = wav.relative_to(raw_dir)
        machine = next((t.lower() for t in rel.parts if t.lower() in _MIMII_MACHINES), None)
        stem = wav.stem
        toks = stem.split("_")
        section = next((f"{a}_{b}" for a, b in zip(toks, toks[1:]) if a == "section"), None)
        domain = next((t for t in toks if t in ("source", "target")), None)
        split = next((t for t in toks if t in ("train", "test")), None)
        condition = next((t for t in toks if t in ("normal", "anomaly")), None)
        audio, sr = _read_audio_file(wav)
        rid = f"{machine}_{stem}" if machine else stem
        meta = _meta_frame(
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
        yield _safe_key(rid), _audio_frame(audio, sr, meta)


def build_aerosonicdb(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """AeroSonicDB-YPAD0523: mono 22.05 kHz aircraft flyover, driven by
    ``sample_meta.csv`` (+ ``aircraft_meta.csv`` by ``hex_id``)."""
    import pandas as pd

    smeta = _find_csv(raw_dir, "sample_meta.csv")
    if smeta is None:
        raise FileNotFoundError("AeroSonicDB: sample_meta.csv not found in raw dir")
    sdf = pd.read_csv(smeta)
    sdf.columns = [str(c).lower() for c in sdf.columns]
    ameta = _find_csv(raw_dir, "aircraft_meta.csv")
    adf = None
    if ameta is not None:
        adf = pd.read_csv(ameta)
        adf.columns = [str(c).lower() for c in adf.columns]
        if "hex_id" in adf.columns:
            adf = adf.set_index("hex_id")
    files = {p.name: p for p in _iter_audio_files(raw_dir)}
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
        audio, sr = _read_audio_file(wav)
        rid = Path(fn).stem
        meta = _meta_frame(
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
        yield _safe_key(rid), _audio_frame(audio, sr, meta)


# -- HuggingFace parquet datasets (streamed from the hub, no local raw dir) --
#
# Both HF datasets are parquet-only (no raw wav tree): drone-detection embeds
# audio as encoded bytes + an int label; DroneAudioSet embeds a decoded
# (channel, time) array + sampling_rate + file_path. We stream the parquet
# shards row-batched straight from the hub (low memory, nothing on scratch) and
# decode each row. The per-row decoders are pure (testable without network).


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
    meta = _meta_frame(
        rid,
        "drone-detection-samples",
        system={"category": "drone"},
        observation={"type": "unknown", "source_motion": "unknown", "relative_trajectory": "none"},
        label={"class": cls, "raw_label": None if label is None else int(label)},
        extra={"raw_path": str(path)},
    )
    return _safe_key(rid), _audio_frame(audio, sr, meta)


def _decode_droneaudioset_row(idx: int, row: dict) -> tuple[str, td.Frame]:
    audio, sr = _row_audio_from_array(row["audio"])
    fp = str(row.get("file_path") or row["audio"].get("path") or f"row_{idx}")
    low = fp.lower()
    subsets = ("drone-with-source", "drone-only", "source-only", "ground-truth")
    subset = next((s for s in subsets if s in low), row.get("data_type"))
    dtok = re.search(r"drone\d", low)
    dist_m = None
    mdist = re.search(r"mic-?dist-?(\d+)\s*cm", low)
    if mdist:
        dist_m = int(mdist.group(1)) / 100.0
    throttle = next((t for t in ("low", "high") if f"throttle-{t}" in low), None)
    rid = f"{idx:06d}_{fp}"
    meta = _meta_frame(
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
        label={"subset": subset, "data_type": row.get("data_type")},
        extra={"raw_path": fp},
    )
    return _safe_key(rid), _audio_frame(audio, sr, meta)


def _iter_hf_parquet(
    repo_id: str, pattern: str, decode: Callable[[int, dict], tuple[str, td.Frame]], batch_size: int
) -> Iterator[tuple[str, td.Frame]]:
    """Stream ``decode(idx, row)`` over every parquet shard of a hub dataset."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{repo_id}"
    idx = 0
    for path in sorted(fs.glob(f"{base}/{pattern}")):
        with fs.open(path) as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    yield decode(idx, row)
                    idx += 1


def build_drone_detection(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """geronimobasso drone-audio-detection-samples (HF parquet, streamed): mono
    16 kHz clips, binary drone/no-drone. ``raw_dir`` unused (hub-streamed)."""
    del raw_dir
    yield from _iter_hf_parquet(
        "geronimobasso/drone-audio-detection-samples",
        "data/*.parquet",
        _decode_drone_detection_row,
        batch_size=16,
    )


def build_droneaudioset(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """DroneAudioSet (HF parquet, streamed): rig-mounted static drone; subset +
    throttle + mic distance from ``file_path``; multichannel arrays. Batch size 1
    — rows carry large multichannel arrays. ``raw_dir`` unused (hub-streamed)."""
    del raw_dir
    yield from _iter_hf_parquet(
        "ahlab-drone-project/DroneAudioSet", "*/*.parquet", _decode_droneaudioset_row, batch_size=1
    )


def build_hornbase(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """HornBase: stereo 44.1 kHz 1 s clips; tonal (not rotating). Label parsed
    best-effort from the filename ('not'/'no' → not-horn)."""
    for wav in _iter_audio_files(raw_dir):
        rel = wav.relative_to(raw_dir)
        low = wav.name.lower()
        cls = None
        if "horn" in low:
            cls = "not_horn" if re.search(r"\b(not|no)\b|nothorn|non", low) else "horn"
        audio, sr = _read_audio_file(wav)
        rid = str(rel)
        meta = _meta_frame(
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
        yield _safe_key(rid), _audio_frame(audio, sr, meta)


def build_kaist_acoustic(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """KAIST rotating machine (acoustic.zip): mic sound-pressure in ``.mat``,
    ~51.2 kHz, 0 Nm. Each file has one ``Signal`` struct with ``y_values.values``
    (the pressure vector) and ``x_values.increment`` (sample period). Filename
    ``<load>Nm_<fault>[_<severity>]``."""
    from scipy.io import loadmat

    for mat in sorted(raw_dir.rglob("*.mat")):
        d = loadmat(str(mat), squeeze_me=True, struct_as_record=False)
        sig = d.get("Signal")
        if sig is None or not hasattr(sig, "y_values"):
            continue
        values = np.asarray(sig.y_values.values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            continue
        sr = 51200
        inc = getattr(getattr(sig, "x_values", None), "increment", None)
        if inc is not None and float(inc) > 0:
            sr = int(round(1.0 / float(inc)))
        rid = mat.stem
        toks = rid.split("_")
        load_tok = next((t for t in toks if t.lower().endswith("nm")), None)
        meta = _meta_frame(
            rid,
            "KAIST-rotating-acoustic",
            system={"category": "industrial_machine", "machine": "rotating_machine_testbed"},
            observation={
                "type": "fixed_mic_bench",
                "source_motion": "static",
                "relative_trajectory": "none",
            },
            operating={"load": load_tok, "rpm": 3010, "sensor": "PCB378B02"},
            label={
                "fault": toks[1] if len(toks) > 1 else None,
                "severity": toks[2] if len(toks) > 2 else None,
            },
            extra={"raw_relpath": str(mat.relative_to(raw_dir))},
        )
        yield _safe_key(rid), _audio_frame(values[None, :], sr, meta)


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


def build_hustmotor(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """HUSTmotor: 25.6 kHz instrument ``.txt`` (6 health × 4 speeds). Columns are
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
        meta = _meta_frame(
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
        frame = _audio_frame(acoustic, sr, meta)
        if vibration is not None:
            frame = frame.with_entry("vibration", td.uniform(vibration, sr, dims=("axis", "time")))
        yield _safe_key(rid), frame


# ─── Registry ──────────────────────────────────────────────────────────────────

EXTERNAL_SPECS: dict[str, ExternalDataset] = {
    "MIMII": ExternalDataset(
        name="MIMII",
        download=DownloadSpec("zenodo", {"record_id": 3384388}, extract=True),
        builder=build_mimii,
        provenance={
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
        },
    ),
    "MIMII-DG": ExternalDataset(
        name="MIMII-DG",
        download=DownloadSpec("zenodo", {"record_id": 6529888}, extract=True),
        builder=build_mimii_dg,
        provenance={
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
        },
    ),
    "AeroSonicDB": ExternalDataset(
        name="AeroSonicDB",
        download=DownloadSpec(
            "zenodo",
            {"record_id": 8371595, "files": ["audio.zip", "sample_meta.csv", "aircraft_meta.csv"]},
            extract=True,
        ),
        builder=build_aerosonicdb,
        provenance={
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
        },
    ),
    "DroneAudioSet": ExternalDataset(
        name="DroneAudioSet",
        download=DownloadSpec("hf", {"repo_id": "ahlab-drone-project/DroneAudioSet"}),
        builder=build_droneaudioset,
        provenance={
            "source_url": "https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet",
            "doi": "10.48550/arXiv.2510.15383",
            "license": "MIT",
            "citation": "DroneAudioSet (arXiv:2510.15383).",
            "collection_method": "rig-mounted static quadcopters; clean drone-only + source-only stems + real mixtures; SNR -57..-2.5 dB",
            "equipment": "two 8-ch MEMS circular arrays (above/below) + central mic (17 ch), distances 25/50 cm",
            "observation_type": "rig_mounted_static",
            "channels": "varies (verify per file)",
            "description": "Drone speech-enhancement dataset: 2 quads, 2 throttles, 3 rooms; drone-only/source-only/mixed/ground-truth subsets.",
        },
        streaming=True,
    ),
    "drone-detection-samples": ExternalDataset(
        name="drone-detection-samples",
        download=DownloadSpec("hf", {"repo_id": "geronimobasso/drone-audio-detection-samples"}),
        builder=build_drone_detection,
        provenance={
            "source_url": "https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples",
            "license": "MIT",
            "citation": "geronimobasso/drone-audio-detection-samples (HuggingFace).",
            "collection_method": "aggregated open-licensed drone/no-drone detection clips",
            "observation_type": "unknown",
            "sample_rate": 16000,
            "channels": 1,
            "description": "180k mono 16 kHz clips, binary drone/no-drone; provenance mixed (attribution may flow through).",
        },
        streaming=True,
    ),
    "HornBase": ExternalDataset(
        name="HornBase",
        download=DownloadSpec("mendeley", {"dataset_id": "y5stjsnp8s", "version": 2}, extract=True),
        builder=build_hornbase,
        provenance={
            "source_url": "https://data.mendeley.com/datasets/y5stjsnp8s/2",
            "doi": "10.17632/y5stjsnp8s.2",
            "license": "CC BY 4.0",
            "citation": "HornBase — A Car Horns Dataset (Data in Brief).",
            "collection_method": "two-smartphone recording of vehicle horns in traffic scenarios",
            "observation_type": "ground",
            "sample_rate": 44100,
            "channels": 2,
            "description": "1,080 stereo 44.1 kHz 1 s clips, horn/not-horn. NOTE: horns are tonal, not rotating-source.",
        },
    ),
    "KAIST-rotating-acoustic": ExternalDataset(
        name="KAIST-rotating-acoustic",
        download=DownloadSpec(
            "mendeley",
            {"dataset_id": "ztmf3m7h5x", "version": 5, "files": ["acoustic.zip"]},
            extract=True,
        ),
        builder=build_kaist_acoustic,
        provenance={
            "source_url": "https://data.mendeley.com/datasets/ztmf3m7h5x/5",
            "doi": "10.17632/ztmf3m7h5x.5",
            "license": "CC BY 4.0",
            "citation": "Jung et al., Vibration/Acoustic/Temp/Current of Rotating Machine (Data in Brief 48:109049, 2023).",
            "collection_method": "rotating-machine testbed; acoustic mic only at 0 Nm load",
            "equipment": "PCB378B02 microphone, 51.2 kHz",
            "observation_type": "fixed_mic_bench",
            "sample_rate": 51200,
            "channels": 1,
            "description": "Acoustic subset only (acoustic.zip): sound-pressure .mat, 3010 RPM, fault/severity in filename. Vibration/current zips excluded (not audio).",
        },
    ),
    "HUSTmotor": ExternalDataset(
        name="HUSTmotor",
        download=DownloadSpec("gdrive", {"folder_id": "1XmahwIQ4o66FC3dpOaeTV-gqz2dd0XBw"}),
        builder=build_hustmotor,
        provenance={
            "source_url": "https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset",
            "license": "unspecified (research use; contact author)",
            "citation": "Zhao, HUSTmotor multi-modal dataset.",
            "collection_method": "SpectraQuest mechanical fault simulator; synchronized vibration + acoustic",
            "equipment": "accelerometer + microphone, 25.6 kHz",
            "observation_type": "fixed_mic_bench",
            "sample_rate": 25600,
            "channels": "varies (channel roles unconfirmed)",
            "description": "6 health states × 4 speeds (5/10/20/30 Hz) as numeric .txt. NOTE: unlicensed; channel roles unconfirmed.",
        },
    ),
}


def list_names() -> list[str]:
    return list(EXTERNAL_SPECS)


def get(name: str) -> ExternalDataset:
    if name not in EXTERNAL_SPECS:
        raise KeyError(f"unknown external dataset {name!r}; known: {list(EXTERNAL_SPECS)}")
    return EXTERNAL_SPECS[name]


def dataset_meta(name: str) -> dict[str, Any]:
    """Manifest ``meta`` for the published dataset: provenance + ``tdframe-v1``
    layout marker (so ``DloadFrameDataset`` auto-decodes)."""
    spec = get(name)
    from data_processing.streams import LAYOUT_META_KEY

    return {LAYOUT_META_KEY: LAYOUT, "modality": spec.modality, **spec.provenance}


def download_dataset(name: str, dest: Path) -> Path:
    """Fetch + (optionally) extract ``name``'s raw files into ``dest``.

    A no-op for ``streaming`` datasets (their builder reads from the hub)."""
    from data_processing import downloaders

    entry = get(name)
    if entry.streaming:
        print(f"{name}: hub-streamed at build time — nothing to pre-download")
        return Path(dest)
    spec = entry.download
    dest = Path(dest)
    if spec.kind == "zenodo":
        downloaders.zenodo_fetch(spec.params["record_id"], dest, files=spec.params.get("files"))
    elif spec.kind == "mendeley":
        downloaders.mendeley_fetch(
            spec.params["dataset_id"],
            dest,
            version=spec.params.get("version"),
            files=spec.params.get("files"),
        )
    elif spec.kind == "hf":
        downloaders.hf_fetch(
            spec.params["repo_id"], dest, allow_patterns=spec.params.get("allow_patterns")
        )
    elif spec.kind == "gdrive":
        downloaders.gdrive_fetch(spec.params["folder_id"], dest)
    else:
        raise ValueError(f"unknown download kind {spec.kind!r}")
    if spec.extract:
        # Extract each zip then delete it, so peak disk ≈ the extracted size
        # rather than zips + extracted (matters for MIMII's ~100 GB). A re-run
        # re-downloads any deleted zip (size-match skip only helps if kept), an
        # acceptable trade for one-shot publishing on quota'd scratch.
        for zip_path in sorted(dest.glob("*.zip")):
            marker = dest / zip_path.stem / ".extracted"
            if not marker.exists():
                downloaders.extract_zip(zip_path, dest / zip_path.stem)
                marker.touch()
            zip_path.unlink(missing_ok=True)
    return dest
