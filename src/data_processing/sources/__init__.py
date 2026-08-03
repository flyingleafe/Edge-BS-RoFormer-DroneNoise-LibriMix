"""The source-dataset registry — every external dataset, defined uniformly.

One entry per raw dataset the project consumes, whether it is a public
download (Zenodo/HF/Mendeley/GDrive/HTTP) or a project-local raw tree already
committed to dload (``DREGON``, ``recording_with_motor_speed``,
``librispeech``, ...). DREGON gets no preferential treatment: it is one entry
with a builder, exactly like MIMII or AVQ.

An entry pairs:

- **how to obtain the raw files** — a pinned :class:`DownloadSpec`, a custom
  ``fetcher(dest)``, or a ``raw_dataset`` dload pin (for project-local raws
  the raw tree is *already* the dload dataset; obtaining it is
  ``streams.resolve_source("dload:<raw_dataset>")``);
- **how to build recording Frames** — ``builder(raw_dir) ->
  Iterator[(key, td.Frame)]`` producing rich ``tdframe-v1`` samples (audio
  Series + documented geometry + nested ``meta``), or ``None`` for raw-only
  datasets consumed as plain files (``librispeech``).

Building/publishing the derived ``*-frames`` datasets is **not** done here —
that is a dload derivation declared in :mod:`data_processing.derivations`
(driver: ``scripts/derive.py``). This module stays torch-free (numpy /
soundfile / scipy / pandas lazy) so registry integrity and fingerprinting run
on any box.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tdseries as td

from data_processing.sources import (
    aerosonicdb,
    avq,
    dregon,
    droneaudio,
    hornbase,
    hustmotor,
    kaist,
    michaels,
    mimii,
    spcup19,
)
from data_processing.sources._common import LAYOUT

__all__ = [
    "DownloadSpec",
    "SourceDataset",
    "REGISTRY",
    "get",
    "list_names",
    "dataset_meta",
    "download_dataset",
    "raw_root",
    "iter_frames",
    "LAYOUT",
]

Builder = Callable[[Path], Iterator[tuple[str, td.Frame]]]
Fetcher = Callable[[Path], Path]


@dataclass(frozen=True)
class DownloadSpec:
    """How to fetch a dataset's raw files (pinned)."""

    kind: str  # "zenodo" | "mendeley" | "hf" | "gdrive" | "http"
    params: dict[str, Any]
    extract: bool = False  # unzip every *.zip after fetch (into <stem>/)


@dataclass(frozen=True)
class SourceDataset:
    """One external dataset, defined once."""

    name: str  # registry key; == published frames dataset name unless frames_dataset set
    provenance: dict[str, Any]
    download: DownloadSpec | None = None  # pinned external fetch
    fetcher: Fetcher | None = None  # custom raw fetch into a dest dir (overrides download)
    builder: Builder | None = None  # raw_dir -> (key, Frame) stream; None = raw-only
    raw_dataset: str | None = None  # dload dataset holding the raw tree (CLI convention)
    frames_dataset: str | None = None  # dload name of the derived frames dataset
    modality: str = "audio"

    @property
    def frames_name(self) -> str:
        """The dload dataset name of this source's derived frames dataset."""
        return self.frames_dataset or self.name


# ─── Registry ─────────────────────────────────────────────────────────────────
#
# Project-local raws (raw_dataset set, no download): the raw tree is committed
# to dload once (CLI convention); builders read it via the pin. External
# downloads (download set): the pinned DownloadSpec is the raw provenance.

REGISTRY: dict[str, SourceDataset] = {
    e.name: e
    for e in (
        # ── Project rotor rigs ────────────────────────────────────────────
        SourceDataset(
            name="DREGON",
            provenance=dregon.PROVENANCE,
            fetcher=dregon.download_dregon,
            builder=dregon.build,
            raw_dataset="DREGON",
            frames_dataset="DREGON-frames",
        ),
        SourceDataset(
            name="michaels",
            provenance=michaels.PROVENANCE,
            builder=michaels.build,
            raw_dataset="recording_with_motor_speed",
            frames_dataset="michaels-frames",
        ),
        # ── External harmonic-noise datasets ──────────────────────────────
        SourceDataset(
            name="MIMII",
            provenance=mimii.MIMII_PROVENANCE,
            download=DownloadSpec("zenodo", {"record_id": 3384388}, extract=True),
            builder=mimii.build_mimii,
        ),
        SourceDataset(
            name="MIMII-DG",
            provenance=mimii.MIMII_DG_PROVENANCE,
            download=DownloadSpec("zenodo", {"record_id": 6529888}, extract=True),
            builder=mimii.build_mimii_dg,
        ),
        SourceDataset(
            name="AeroSonicDB",
            provenance=aerosonicdb.PROVENANCE,
            download=DownloadSpec(
                "zenodo",
                {"record_id": 8371595, "files": ["audio.zip", "sample_meta.csv", "aircraft_meta.csv"]},
                extract=True,
            ),
            builder=aerosonicdb.build,
        ),
        SourceDataset(
            name="DroneAudioSet",
            provenance=droneaudio.DRONEAUDIOSET_PROVENANCE,
            download=DownloadSpec("hf", {"repo_id": "ahlab-drone-project/DroneAudioSet"}),
            builder=droneaudio.build_droneaudioset,
        ),
        SourceDataset(
            name="drone-detection-samples",
            provenance=droneaudio.DRONE_DETECTION_PROVENANCE,
            download=DownloadSpec("hf", {"repo_id": "geronimobasso/drone-audio-detection-samples"}),
            builder=droneaudio.build_drone_detection,
        ),
        SourceDataset(
            name="HornBase",
            provenance=hornbase.PROVENANCE,
            download=DownloadSpec("mendeley", {"dataset_id": "y5stjsnp8s", "version": 2}, extract=True),
            builder=hornbase.build,
        ),
        SourceDataset(
            name="KAIST-rotating-acoustic",
            provenance=kaist.PROVENANCE,
            download=DownloadSpec(
                "mendeley",
                {"dataset_id": "ztmf3m7h5x", "version": 5, "files": ["acoustic.zip"]},
                extract=True,
            ),
            builder=kaist.build,
        ),
        SourceDataset(
            name="HUSTmotor",
            provenance=hustmotor.PROVENANCE,
            download=DownloadSpec("gdrive", {"folder_id": "1XmahwIQ4o66FC3dpOaeTV-gqz2dd0XBw"}),
            builder=hustmotor.build,
        ),
        SourceDataset(
            name="SPCUP19-egonoise",
            provenance=spcup19.PROVENANCE,
            download=DownloadSpec("http", {"urls": spcup19.URLS}, extract=True),
            builder=spcup19.build,
        ),
        SourceDataset(
            name="AVQ",
            provenance=avq.PROVENANCE,
            download=DownloadSpec("http", {"urls": {"avq.zip": avq.URL}}, extract=True),
            builder=avq.build,
        ),
        # ── Raw-only dload datasets (consumed as files; no frames builder) ──
        SourceDataset(
            name="librispeech",
            provenance={
                "source_url": "https://www.openslr.org/12",
                "license": "CC BY 4.0",
                "citation": "Panayotov et al., LibriSpeech (ICASSP 2015).",
                "description": "train-clean-100 speech utterances (flac), consumed as files.",
            },
            raw_dataset="librispeech",
        ),
        SourceDataset(
            name="drone_audio",
            provenance={
                "source_url": "https://github.com/saraalemadi/DroneAudioDataset",
                "citation": "Al-Emadi et al., DroneAudioDataset.",
                "description": (
                    "Binary drone/no-drone audio; the DN-LM noise source is the "
                    "label-1 subtree Binary_Drone_Audio/yes_drone (the unknown/ "
                    "class mixes ESC-50 + white noise + silence negatives)."
                ),
            },
            raw_dataset="drone_audio",
        ),
        SourceDataset(
            name="new-drone-noises",
            provenance={
                "citation": "Michael's DJI flight logs + WAVs (project-local).",
                "description": (
                    "108 DJI recordings; only FLY124/FLY125 have manual alignment "
                    "constants (see the 'michaels' entry) — the rest are raw-only."
                ),
            },
            raw_dataset="new-drone-noises",
        ),
        SourceDataset(
            name="recording_with_motor_speed",
            provenance={
                "citation": "Michael's aligned DJI Matrice 100 recordings (project-local).",
                "description": "Raw tree behind the 'michaels' frames builder.",
            },
            raw_dataset="recording_with_motor_speed",
        ),
        SourceDataset(
            name="music",
            provenance={"description": "Music augmentation corpus (raw audio files)."},
            raw_dataset="music",
        ),
        SourceDataset(
            name="zenodo_drone_noises",
            provenance={
                "source_url": "https://zenodo.org",
                "description": "Zenodo drone-noise zip blobs (raw-only; not shard-streamable).",
            },
            raw_dataset="zenodo_drone_noises",
        ),
    )
}


def list_names() -> list[str]:
    return list(REGISTRY)


def get(name: str) -> SourceDataset:
    if name not in REGISTRY:
        raise KeyError(f"unknown source dataset {name!r}; known: {list(REGISTRY)}")
    return REGISTRY[name]


def dataset_meta(name: str) -> dict[str, Any]:
    """Manifest ``meta`` for the derived frames dataset: provenance + the
    ``tdframe-v1`` layout marker (so ``DloadFrameDataset`` auto-decodes)."""
    from data_processing.streams import LAYOUT_META_KEY

    entry = get(name)
    return {LAYOUT_META_KEY: LAYOUT, "modality": entry.modality, **entry.provenance}


def download_dataset(name: str, dest: Path) -> Path:
    """Fetch + (optionally) extract ``name``'s raw files into ``dest``.

    Dispatches on the entry's ``fetcher`` (custom) or ``download`` spec.
    Entries whose raw files live in a dload ``raw_dataset`` have nothing to
    fetch here — use :func:`raw_root`.
    """
    from data_processing import downloaders

    entry = get(name)
    dest = Path(dest)
    if entry.fetcher is not None:
        return Path(entry.fetcher(dest))
    spec = entry.download
    if spec is None:
        raise ValueError(
            f"{name!r} has no download spec (raw_dataset={entry.raw_dataset!r}); "
            "obtain the raw tree via dload (raw_root) instead"
        )
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
    elif spec.kind == "http":
        for fname, url in spec.params["urls"].items():
            print(f"  fetching {fname} ...", flush=True)
            downloaders.http_fetch(url, dest / fname)
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


def raw_root(name: str, *, raw_cache: Path | None = None) -> Path:
    """Resolve the local root of a source's raw files.

    - entries with a ``raw_dataset`` dload pin: the materialized dload tree
      (version-addressed, idempotent — ``streams.resolve_source``);
    - entries with a download spec / fetcher: downloaded into
      ``raw_cache / name`` (default ``.cache/source_raw/<name>`` under the repo
      root), idempotent (size-match skips / extraction markers).
    """
    entry = get(name)
    if raw_cache is None and entry.raw_dataset is not None:
        from data_processing.streams import resolve_source

        return resolve_source(f"dload:{entry.raw_dataset}")
    if raw_cache is None:
        from data_processing.streams import REPO_ROOT

        raw_cache = REPO_ROOT / ".cache" / "source_raw"
    return download_dataset(name, Path(raw_cache) / name)


def geometry(name: str) -> tuple[Any, Any]:
    """``(mic_pos, rotor_pos)`` for rigs with a fixed array geometry.

    Only the two project rigs have a single canonical geometry; datasets like
    AVQ carry per-recording ``mic_pos`` entries instead (read them off the
    frames). DREGON's geometry files live in its raw tree, resolved through
    the dload pin (``raw_root``).
    """
    if name == "DREGON":
        return dregon.get_geometry(raw_root("DREGON"))
    if name == "michaels":
        return michaels.get_geometry()
    raise KeyError(
        f"source {name!r} has no fixed registry geometry — read mic_pos/rotor_pos "
        "entries off its published frames instead"
    )


def iter_frames(name: str, *, raw_cache: Path | None = None) -> Iterator[tuple[str, td.Frame]]:
    """Stream a source's recording Frames: resolve raw files, run the builder."""
    entry = get(name)
    if entry.builder is None:
        raise ValueError(f"source {name!r} is raw-only (no frames builder)")
    root = raw_root(name, raw_cache=raw_cache)
    # Raw trees from dload materialize as <root>/<relpath>; some builders were
    # written against a *parent* root (e.g. michaels paths are relative to the
    # recording_with_motor_speed tree itself) — the entry's raw_dataset tree is
    # exactly that root.
    yield from entry.builder(root)


def iter_recording_frames(
    name: str,
    *,
    version: str | None = None,
    splits: list[str] | None = None,
    sample_rate: int | None = None,
) -> Iterator[td.Frame]:
    """Stream a source's *published* frames dataset (the dload-native way to
    load recordings — fixes baked in, no raw tree needed).

    ``splits`` filters on ``meta.split``; ``sample_rate`` soxr-resamples the
    ``audio`` entry (telemetry tracks are untouched).
    """
    from data_processing.frames import resample_audio_series
    from data_processing.streams import iter_published_frames

    entry = get(name)
    for frame in iter_published_frames(entry.frames_name, version, splits=splits):
        if sample_rate is not None and "audio" in frame:
            audio = frame["audio"]
            tindex = audio.tindex
            if isinstance(tindex, td.GridIndex) and int(tindex.sr) != int(sample_rate):
                frame = frame.with_entry("audio", resample_audio_series(audio, int(sample_rate)))
        yield frame
