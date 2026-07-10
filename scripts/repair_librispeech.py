"""Repair corrupt/missing flacs in a librispeech tree and republish to dload.

Background (2026-07-10): the R2-published ``librispeech`` dataset contains a
truncated flac (``train-clean-100/669/129061/669-129061-0001.flac``, exactly
160 KiB, mtime 2026-02-01 — an interrupted write that replaced the 2014
original) which killed every fresh ``AudioFileSourcePool`` packed-cache build
(local/colab/kaggle). The cluster only survived via the hand-made
``train-clean-100-readable`` symlink tree that omits the file. This script
fixes the root cause:

1. decode-verify every ``.flac`` under ``--source-root`` (parallel, full read);
2. download the original OpenSLR ``train-clean-100.tar.gz`` and extract
   verified replacements for every broken/missing file;
3. stage a clean publish tree (hardlinks; cluster-only ``*-readable`` trees
   excluded) with the repaired files swapped in;
4. ``dload commit librispeech --from <stage>`` (skippable via
   ``--skip-publish``).

After the job: run ``dload pin librispeech`` locally and commit ``dload.lock``.

Typical cluster invocation (CPU-only, from the repo root so ``dload.toml`` and
``.env`` R2 creds resolve):

    uv run python scripts/repair_librispeech.py \
        --source-root /gpfs/scratch/acw592/data/librispeech \
        --work-dir /gpfs/scratch/acw592/tmp/librispeech_repair
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

OPENSLR_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
# Cluster-only workaround trees that must never enter the published dataset.
EXCLUDE_DIR_SUFFIX = "-readable"
DATASET_NAME = "librispeech"


def _check_decodes(path: str) -> tuple[str, str | None]:
    try:
        sf.read(path)
    except Exception as exc:
        return path, f"{type(exc).__name__}: {exc}"
    return path, None


def scan_broken(root: Path, workers: int) -> list[Path]:
    flacs = [p for p in root.rglob("*.flac") if not p.is_symlink()]
    print(f"Scanning {len(flacs)} flac files under {root} with {workers} workers ...")
    broken: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (path, err) in enumerate(pool.map(_check_decodes, map(str, flacs), chunksize=64)):
            if err is not None:
                print(f"  BROKEN: {path}: {err}")
                broken.append(Path(path))
            if (i + 1) % 5000 == 0 or (i + 1) == len(flacs):
                print(f"  scanned {i + 1}/{len(flacs)}")
    return broken


def download_tar(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 6_000_000_000:
        print(f"Reusing downloaded archive at {dest}")
        return dest
    print(f"Downloading {OPENSLR_URL} -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    def _hook(blocks: int, block_size: int, _total: int) -> None:
        done = blocks * block_size
        if done % (500 * 1024 * 1024) < block_size:
            print(f"  downloaded ~{done / 1e9:.1f} GB")

    urllib.request.urlretrieve(OPENSLR_URL, tmp, reporthook=_hook)
    tmp.replace(dest)
    return dest


def extract_replacements(
    tar_path: Path, wanted_relpaths: set[str], out_dir: Path
) -> dict[str, Path]:
    """Extract ``LibriSpeech/...``-keyed members; returns relpath -> extracted file."""
    print(f"Extracting {len(wanted_relpaths)} members from {tar_path.name} ...")
    found: dict[str, Path] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.name not in wanted_relpaths or not member.isfile():
                continue
            target = out_dir / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            assert src is not None
            with open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            found[member.name] = target
            if len(found) == len(wanted_relpaths):
                break
    missing = wanted_relpaths - set(found)
    if missing:
        raise SystemExit(f"archive is missing {len(missing)} wanted members: {sorted(missing)[:5]}")
    for relpath, path in found.items():
        _, err = _check_decodes(str(path))
        if err is not None:
            raise SystemExit(f"replacement from archive does not decode: {relpath}: {err}")
    print("All replacements extracted and verified.")
    return found


def stage_tree(source_root: Path, stage_root: Path, replacements: dict[str, Path]) -> None:
    """Hardlink-copy source -> stage, excluding cluster-only trees; swap in repairs."""
    if stage_root.exists():
        shutil.rmtree(stage_root)
    print(f"Staging publish tree at {stage_root} (hardlinks) ...")
    n_linked = 0
    for dirpath, dirnames, filenames in os.walk(source_root):
        dirnames[:] = [d for d in dirnames if not d.endswith(EXCLUDE_DIR_SUFFIX)]
        rel_dir = Path(dirpath).relative_to(source_root)
        (stage_root / rel_dir).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = Path(dirpath) / name
            if src.is_symlink():
                continue
            os.link(src, stage_root / rel_dir / name)
            n_linked += 1
    print(f"  linked {n_linked} files")
    for relpath, fixed in replacements.items():
        target = stage_root / relpath
        if target.exists():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixed, target)
        print(f"  repaired {relpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True, help="existing librispeech root")
    ap.add_argument("--work-dir", type=Path, required=True, help="scratch dir (tar, stage tree)")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--skip-publish", action="store_true", help="repair + stage only")
    args = ap.parse_args()

    source_root: Path = args.source_root
    if not (source_root / "LibriSpeech").is_dir():
        raise SystemExit(f"{source_root} does not look like a librispeech root")

    broken = scan_broken(source_root, args.workers)
    if not broken:
        print("No broken flacs found — nothing to repair.")
        if args.skip_publish:
            return
    broken_rel = {str(p.relative_to(source_root)) for p in broken}

    replacements: dict[str, Path] = {}
    if broken_rel:
        tar_path = download_tar(args.work_dir / "train-clean-100.tar.gz")
        replacements = extract_replacements(tar_path, broken_rel, args.work_dir / "extracted")

    stage_root = args.work_dir / "stage" / source_root.name
    stage_tree(source_root, stage_root, replacements)

    still_broken = [rel for rel in broken_rel if _check_decodes(str(stage_root / rel))[1]]
    if still_broken:
        raise SystemExit(f"staged tree still has broken files: {still_broken}")
    print("Staged tree verified clean.")

    if args.skip_publish:
        print(f"--skip-publish: stage left at {stage_root}")
        return

    print(f"Publishing: dload commit {DATASET_NAME} --from {stage_root}")
    result = subprocess.run(
        ["dload", "commit", DATASET_NAME, "--from", str(stage_root)],
        check=False,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"dload commit failed with exit code {result.returncode}")
    print("Published. Now run `dload pin librispeech` locally and commit dload.lock.")


if __name__ == "__main__":
    main()
