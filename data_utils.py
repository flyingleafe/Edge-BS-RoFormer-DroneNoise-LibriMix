from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".WAV", ".FLAC", ".MP3", ".OGG")


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_dest)
        tmp_dest.rename(dest)
    except Exception:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise
    return dest


def unpack_zip(zip_path: Path, dest_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".unzipped"
    if marker.exists():
        return dest_dir
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.write_text("ok", encoding="utf-8")
    return dest_dir


def _has_audio_files(directory: Path) -> bool:
    for ext in AUDIO_EXTENSIONS:
        if any(directory.rglob(f"*{ext}")):
            return True
    return False


def find_split_dirs(root_dir: Path, split_names: Iterable[str] = ("train", "test")) -> Dict[str, Path]:
    candidates: Dict[str, List[Path]] = {split: [] for split in split_names}
    for split in split_names:
        direct = root_dir / split
        if direct.exists() and direct.is_dir():
            candidates[split].append(direct)
        for path in root_dir.rglob("*"):
            if path.is_dir() and split in path.name.lower():
                candidates[split].append(path)

    split_dirs: Dict[str, Path] = {}
    for split, paths in candidates.items():
        valid = [p for p in paths if _has_audio_files(p)]
        if valid:
            split_dirs[split] = valid[0]

    if not split_dirs:
        raise ValueError(f"Could not locate train/test splits under {root_dir}")
    return split_dirs


def collect_audio_files(root_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(root_dir.rglob(f"*{ext}"))
    return sorted(set(files))


def prepare_zenodo_drone_noises(
    data_dir: Path,
    *,
    url: str,
    archive_name: str = "all_drone_noises.zip",
    extracted_dirname: str = "zenodo_drone_noises",
) -> Tuple[Path, Dict[str, Path]]:
    zenodo_root = data_dir / extracted_dirname
    archive_path = zenodo_root / archive_name
    extract_dir = zenodo_root / "raw"

    download_file(url, archive_path)
    unpack_zip(archive_path, extract_dir)

    split_dirs = find_split_dirs(extract_dir)
    return extract_dir, split_dirs


def build_local_hf_dataset_from_splits(
    split_dirs: Dict[str, Path],
    output_dir: Path,
    *,
    sample_rate: int = 16000,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        return output_dir
    try:
        from datasets import Audio, Dataset, DatasetDict
    except ImportError as exc:
        raise ValueError(
            "datasets is required to build a local HF dataset. "
            "Install it with: uv add datasets"
        ) from exc

    dataset_dict = {}
    for split, directory in split_dirs.items():
        files = collect_audio_files(directory)
        dataset = Dataset.from_dict({"audio": [str(p) for p in files]})
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        dataset_dict[split] = dataset

    hf_dataset = DatasetDict(dataset_dict)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(output_dir)
    return output_dir


def download_hf_dataset_splits(
    dataset_name: str,
    *,
    splits: Iterable[str] = ("train", "test"),
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError(
            "datasets is required to download Hugging Face datasets. "
            "Install it with: uv add datasets"
        ) from exc

    # Download full dataset metadata and files (all splits)
    load_dataset(dataset_name)

    # Ensure specific splits are materialized in cache
    for split in splits:
        try:
            load_dataset(dataset_name, split=split)
        except Exception:
            continue
