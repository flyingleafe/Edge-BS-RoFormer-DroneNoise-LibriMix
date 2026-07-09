#!/usr/bin/env python3
"""Download + publish external harmonic-noise datasets to dload (R2).

Registry-driven driver over ``data_processing.external_datasets``: each dataset
is fetched into a raw dir, streamed through its ``builder`` into ``td.Frame``s,
and committed with the generic ``tdframe-v1`` codec (so ``DloadFrameDataset``
auto-decodes it, exactly like ``DREGON-frames``). One frame is in memory at a
time — safe for the ~100 GB MIMII set.

Idempotent: dload shards are content-addressed, so re-running re-uploads only
new bytes. Typical flow on the CPU cluster (large downloads land on scratch)::

    python scripts/publish_external_datasets.py download MIMII --dest $SCRATCH/raw/MIMII
    python scripts/publish_external_datasets.py publish  MIMII --raw  $SCRATCH/raw/MIMII
    python scripts/publish_external_datasets.py pin       MIMII

or all at once::

    python scripts/publish_external_datasets.py run MIMII MIMII-DG --root $SCRATCH/raw

See ``docs/external-datasets-plan.md``. Harmonicity is measured separately
(analysis stage), not here.
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Iterator
from pathlib import Path

from data_processing import external_datasets as ext
from data_processing import streams

Sample = tuple[str, dict[str, bytes]]

REPO_ROOT = Path(__file__).resolve().parent.parent
_RECIPE = (REPO_ROOT / "src" / "data_processing" / "external_datasets.py").read_text(
    encoding="utf-8"
)


def _iter_samples(name: str, raw_dir: Path) -> Iterator[Sample]:
    """Stream ``(key, tdframe-v1 fields)`` for one dataset, one frame at a time."""
    builder = ext.get(name).builder
    seen: set[str] = set()
    for n, (key, frame) in enumerate(builder(raw_dir), start=1):
        if key in seen:
            raise ValueError(f"{name}: duplicate sample key {key!r}")
        seen.add(key)
        fields = streams.frame_to_sample(frame)
        del frame
        if n % 500 == 0:
            print(f"  {name}: built {n} samples...", flush=True)
        yield key, fields
        del fields
        gc.collect()


def cmd_list(_: argparse.Namespace) -> None:
    for name in ext.list_names():
        spec = ext.get(name)
        prov = spec.provenance
        print(
            f"{name:28s} {spec.download.kind:9s} {prov.get('license', '?'):16s} {prov.get('description', '')}"
        )


def cmd_download(args: argparse.Namespace) -> None:
    dest = Path(args.dest) if args.dest else REPO_ROOT / ".cache" / "external_raw" / args.name
    print(f"Downloading {args.name} → {dest} ...", flush=True)
    ext.download_dataset(args.name, dest)
    print(f"Done: {dest}")


def cmd_publish(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw) if args.raw else REPO_ROOT / ".cache" / "external_raw" / args.name
    if not raw_dir.exists():
        raise SystemExit(f"raw dir {raw_dir} does not exist — run `download {args.name}` first")
    repo = streams.open_repository()
    print(f"Publishing {args.name} from {raw_dir} ...", flush=True)
    manifest = repo.commit(
        args.name,
        _iter_samples(args.name, raw_dir),
        meta=ext.dataset_meta(args.name),
        recipe=_RECIPE,
        progress=print,
    )
    print(f"{args.name}@{manifest.version[:12]}: {manifest.num_samples} samples")
    if args.pin:
        repo.pin(args.name, manifest.version)
        print(f"pinned {args.name}@{manifest.version[:12]} in dload.lock")


def cmd_pin(args: argparse.Namespace) -> None:
    repo = streams.open_repository()
    manifest = repo.manifest(args.name)
    repo.pin(args.name, manifest.version)
    print(f"pinned {args.name}@{manifest.version[:12]} in dload.lock")


def cmd_run(args: argparse.Namespace) -> None:
    root = Path(args.root) if args.root else REPO_ROOT / ".cache" / "external_raw"
    for name in args.names:
        dest = root / name
        print(f"=== {name} ===", flush=True)
        ext.download_dataset(name, dest)
        repo = streams.open_repository()
        manifest = repo.commit(
            name,
            _iter_samples(name, dest),
            meta=ext.dataset_meta(name),
            recipe=_RECIPE,
            progress=print,
        )
        repo.pin(name, manifest.version)
        print(f"{name}@{manifest.version[:12]}: {manifest.num_samples} samples (pinned)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list the registry").set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="fetch raw files")
    p_dl.add_argument("name", choices=ext.list_names())
    p_dl.add_argument("--dest", default=None)
    p_dl.set_defaults(func=cmd_download)

    p_pub = sub.add_parser("publish", help="build frames + commit to R2")
    p_pub.add_argument("name", choices=ext.list_names())
    p_pub.add_argument("--raw", default=None)
    p_pub.add_argument("--pin", action="store_true", help="also pin in dload.lock")
    p_pub.set_defaults(func=cmd_publish)

    p_pin = sub.add_parser("pin", help="pin latest in dload.lock")
    p_pin.add_argument("name", choices=ext.list_names())
    p_pin.set_defaults(func=cmd_pin)

    p_run = sub.add_parser("run", help="download+publish+pin one or more datasets")
    p_run.add_argument("names", nargs="+", choices=ext.list_names())
    p_run.add_argument("--root", default=None, help="parent dir for raw downloads")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
