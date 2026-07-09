"""Reproducible fetch helpers for external datasets.

Thin wrappers over mature tools (no bespoke sync layer): ``requests`` for
Zenodo/Mendeley/HTTP, ``huggingface_hub`` for HF, ``gdown`` for Google Drive.
Each populates a raw directory and is idempotent — a file whose size already
matches the remote is skipped, so re-running resumes cheaply. Heavy imports are
deferred into the functions so this module (and the registry that imports it)
loads without those deps present.

These run on the CPU cluster where the large downloads land on ``/gpfs/scratch``;
they are intentionally torch-free.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

__all__ = [
    "zenodo_fetch",
    "http_fetch",
    "mendeley_fetch",
    "hf_fetch",
    "gdrive_fetch",
    "extract_zip",
]

_CHUNK = 1 << 20  # 1 MiB


def _requests() -> Any:
    import requests  # deferred: transitively present, not a hard import-time dep

    return requests


def http_fetch(url: str, dest_path: Path, *, expected_size: int | None = None) -> Path:
    """Stream ``url`` to ``dest_path`` (skip if a same-size file already exists)."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        dest_path.exists()
        and expected_size is not None
        and dest_path.stat().st_size == expected_size
    ):
        return dest_path
    requests = _requests()
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
        resp.raise_for_status()
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest_path)
    return dest_path


def zenodo_fetch(record_id: str | int, dest: Path, *, files: list[str] | None = None) -> Path:
    """Download an *open* Zenodo record into ``dest``.

    Enumerates ``GET /api/records/<id>`` → ``files[]`` (``key`` + ``links.self``)
    and streams each (optionally filtered to ``files``). Returns ``dest``. Raises
    if the record exposes no files (access-gated records return ``files: []`` to
    anonymous requests — those need a granted token, handled elsewhere).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    requests = _requests()
    meta = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60).json()
    entries = meta.get("files", []) or []
    if not entries:
        raise RuntimeError(
            f"Zenodo record {record_id} exposes no files anonymously "
            "(likely access-gated — needs a granted token)"
        )
    wanted = set(files) if files else None
    for entry in entries:
        key = entry.get("key") or entry.get("filename")
        if wanted is not None and key not in wanted:
            continue
        url = entry.get("links", {}).get("self") or entry.get("links", {}).get("download")
        size = entry.get("size") or entry.get("filesize")
        http_fetch(url, dest / key, expected_size=int(size) if size else None)
    return dest


def mendeley_fetch(
    dataset_id: str, dest: Path, *, version: int | None = None, files: list[str] | None = None
) -> Path:
    """Download a Mendeley Data record via the *unauthenticated* public API.

    ``GET /public-api/datasets/<id>/files?folder_id=root&version=<N>`` →
    entries with ``content_details.download_url`` (a public S3 redirect that
    ``requests`` follows with no auth). ``version`` defaults to the latest.
    Raises if the file list comes back empty (some records return ``[]`` — those
    need the browser "Download All" flow).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    requests = _requests()
    if version is None:
        info = requests.get(
            f"https://data.mendeley.com/public-api/datasets/{dataset_id}",
            headers={"Accept": "application/json"},
            timeout=60,
        ).json()
        version = max(int(v["version"]) for v in info.get("versions", [{"version": 1}]))
    listing = requests.get(
        f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files",
        params={"folder_id": "root", "version": version},
        headers={"Accept": "application/json"},
        timeout=60,
    ).json()
    if not listing:
        raise RuntimeError(
            f"Mendeley dataset {dataset_id} v{version} returned an empty file list "
            "(not scriptable via the public API — use the browser download)"
        )
    wanted = set(files) if files else None
    for entry in listing:
        name = entry.get("filename") or entry.get("name")
        if wanted is not None and name not in wanted:
            continue
        details = entry.get("content_details", {})
        url = details.get("download_url")
        size = details.get("size")
        http_fetch(url, dest / name, expected_size=int(size) if size else None)
    return dest


def hf_fetch(
    repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None, max_workers: int = 4
) -> Path:
    """Snapshot a HuggingFace *dataset* repo into ``dest`` (raw files).

    Uses ``huggingface_hub.snapshot_download`` (resumable, dedup via the hub
    cache); ``allow_patterns`` restricts which paths are pulled, ``max_workers``
    bounds concurrency (kept low to stay within job memory limits). A private
    repo needs ``HF_TOKEN`` in the environment.
    """
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    return dest


def gdrive_fetch(folder_id: str, dest: Path) -> Path:
    """Download a public Google Drive *folder* into ``dest`` via ``gdown``."""
    import gdown

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(  # pyright: ignore[reportPrivateImportUsage]
        id=folder_id, output=str(dest), quiet=False, use_cookies=False
    )
    return dest


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract ``zip_path`` into ``dest`` (idempotent-ish: extracts every time,
    but ``ZipFile.extractall`` overwrites in place). Returns ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest
