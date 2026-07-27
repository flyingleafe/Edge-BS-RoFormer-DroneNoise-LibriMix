#!/usr/bin/env python3
"""Publish AVQ-raw — the byte-exact companion to the ``AVQ`` frames dataset.

The ``AVQ`` tdframe-v1 dataset (``external_datasets.build_avq``) carries the
usable, frame-serializable content: 8-ch audio + ``mic_pos`` + ``angle_vad`` +
per-session calibration/meta. This publishes EVERYTHING ELSE from the archive
byte-exact — the videos, the opaque ``cameraParams.mat`` MATLAB object, the
``.docx`` docs, and (for completeness) the raw ``mic_pos.mat`` / ``angle_vad.mat``
/ ``av_calibration.mat`` / ``readme.txt`` — as a plain raw dload dataset, so no
byte of the source is lost. The per-channel ``MONO-*.wav`` files are the only
thing skipped here (they are the audio in ``AVQ``).

    # after `publish_external_datasets download AVQ --dest <raw>`:
    python scripts/publish_avq_raw.py --raw <raw> [--pin]

Sample key = archive-relative path (slashes -> ``__``); field = file extension.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from data_processing import streams
from data_processing.external_datasets import _safe_key

Sample = tuple[str, dict[str, bytes]]

# per-channel audio lives in the AVQ frames dataset; everything else is preserved
_SKIP_STEM_PREFIX = "mono"
_AUDIO_EXT = {".wav"}


def _iter_raw(raw_dir: Path) -> Iterator[Sample]:
    root = Path(raw_dir)
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in _AUDIO_EXT and p.stem.lower().startswith(_SKIP_STEM_PREFIX):
            continue  # a MONO-NNN channel file -> in AVQ frames
        rel = p.relative_to(root)
        key = _safe_key(str(rel.with_suffix("")))
        ext = p.suffix.lower().lstrip(".") or "bin"
        yield key, {ext: p.read_bytes()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="extracted AVQ archive dir (contains S1/, S2/)")
    ap.add_argument("--pin", action="store_true")
    args = ap.parse_args()

    repo = streams.open_repository()
    samples = list(_iter_raw(Path(args.raw)))
    total_mb = sum(len(b) for _, f in samples for b in f.values()) / 1e6
    print(f"AVQ-raw: {len(samples)} files, {total_mb:.0f} MB", flush=True)
    manifest = repo.commit(
        "AVQ-raw",
        iter(samples),
        meta={
            "layout": "raw-files",
            "description": "Byte-exact companion to the AVQ frames dataset: videos, "
            "cameraParams.mat, .docx docs, and the raw mic_pos/angle_vad/av_calibration "
            "mats + readme. Excludes the per-channel MONO-*.wav (those are audio in AVQ).",
            "source_url": "https://webspace.eecs.qmul.ac.uk/lin.wang/download/avq.zip",
            "companion_frames_dataset": "AVQ",
        },
        recipe=Path(__file__).read_text(encoding="utf-8"),
        progress=print,
    )
    print(f"AVQ-raw@{manifest.version[:12]}: {manifest.num_samples} samples", flush=True)
    if args.pin:
        repo.pin("AVQ-raw", manifest.version)
        print(f"pinned AVQ-raw@{manifest.version[:12]}", flush=True)


if __name__ == "__main__":
    main()
