#!/usr/bin/env python3
"""Publish AVQ-egonoise-vkrps — AVQ ego-noise with blind-VK RPS pseudo-labels.

Joins the ``AVQ-egonoise`` audio (5 mono 16 kHz pure rotor ego-noise
recordings, ~705 s) with the per-recording blind-VK pseudo-label NPZs produced
by ``scripts/vk_pseudolabel.py`` (per-frame ``(4, N)`` RPS on the 0.032 s
project trajectory grid, NaN where the annotator refused) into a rich
``tdframe-v1`` dload dataset the online-mix noise pool can consume directly as
``kind: frames`` — the recordings become an additional REAL noise source WITH
rotor labels for RPS-predictor training (beat-VK campaign, R2 arm).

Layout (``scripts/publish_frame_datasets.py`` convention, mirrored from
``scripts/publish_avq_egonoise.py``): one sample per **contiguous accepted
segment** — recordings are split at NaN spans (a frame is accepted iff ALL 4
rotor labels are finite; refused spans are dropped) and only segments >=
``--min-seg-s`` (default 10 s) are kept. Each Frame carries:

- ``audio``: mono ``(time,)`` Series @ 16 kHz, timeline rebased to 0;
- ``rps``: ``(rotor, time)`` StampIndex Series on the 0.032 s grid (the same
  events convention as ``michaels-frames`` — ``_resolve_motor_tracks``'s
  no-cleaning path);
- ``meta``: the AVQ recording meta + provenance (annotator script/commit,
  ``refuse_conf``, per-segment mean stitched VK confidence, segment bounds in
  the source-recording timeline).

Run::

    python scripts/publish_avq_vkrps.py --dry-run     # segment table only
    python scripts/publish_avq_vkrps.py [--pin]       # publish (and pin)
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tdseries as td

from data_processing import streams
from data_processing.frames import audio_series, get_meta, with_meta

SOURCE_DATASET = "AVQ-egonoise"
DATASET_NAME = "AVQ-egonoise-vkrps"
SAMPLE_RATE = 16000
HOP = 512
FRAME_HOP_S = HOP / SAMPLE_RATE  # 0.032 s — the project-wide trajectory grid
N_ROTORS = 4
DEFAULT_LABELS_DIR = Path("omnirun-outputs/bash-e9e87c/results/vk_pseudolabel/AVQ-egonoise")
DEFAULT_MIN_SEG_S = 10.0
ANNOTATOR = "scripts/vk_pseudolabel.py"


def _annotator_commit() -> str:
    """Last commit touching the annotator script (the job's code provenance)."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", ANNOTATOR],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
        return out or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class Segment:
    """One contiguous all-rotors-accepted span of a pseudo-labeled recording."""

    rid: str
    index: int
    f0: int  # first frame index (inclusive) on the 0.032 s grid
    f1: int  # last frame index (exclusive)
    whole: bool  # the segment covers the entire recording
    rps: np.ndarray  # (4, f1 - f0) float32, finite
    vk_conf_per_rotor: np.ndarray  # (4,) mean stitched VK confidence in-span
    comb_conf_per_rotor: np.ndarray  # (4,) recording-level mean comb confidence
    refuse_conf: float

    @property
    def key(self) -> str:
        return self.rid if self.whole else f"{self.rid}_seg{self.index:02d}"

    @property
    def t0(self) -> float:
        return self.f0 * FRAME_HOP_S

    @property
    def t1(self) -> float:
        return self.f1 * FRAME_HOP_S

    @property
    def duration_s(self) -> float:
        return (self.f1 - self.f0) * FRAME_HOP_S


def _accepted_runs(good: np.ndarray) -> list[tuple[int, int]]:
    """``[f0, f1)`` index runs of consecutive True in a boolean mask."""
    edges = np.diff(np.concatenate([[0], good.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def load_segments(labels_dir: Path, min_seg_s: float) -> dict[str, list[Segment]]:
    """Per-recording accepted segments from the annotator NPZs."""
    npz_paths = sorted(labels_dir.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"no pseudo-label NPZs found in {labels_dir}")
    out: dict[str, list[Segment]] = {}
    for path in npz_paths:
        rid = path.stem
        z = np.load(path)
        ft = np.asarray(z["ft"], dtype=np.float64)
        rps = np.asarray(z["rps"], dtype=np.float64)
        if rps.shape != (N_ROTORS, ft.size):
            raise ValueError(f"{path.name}: rps shape {rps.shape} != (4, {ft.size})")
        if not np.allclose(np.diff(ft), FRAME_HOP_S):
            raise ValueError(f"{path.name}: ft is not on the {FRAME_HOP_S} s grid")
        confidence = np.asarray(z["confidence"], dtype=np.float64)  # (4, Nc)
        conf_times = np.asarray(z["conf_times"], dtype=np.float64)  # (Nc,)
        comb_conf = np.asarray(z["comb_conf"], dtype=np.float64)  # (4,)
        refuse_conf = float(z["refuse_conf"])

        # Accepted iff ALL 4 rotor labels are finite at the frame.
        good = np.asarray(~np.any(np.isnan(rps), axis=0))
        n_frames = ft.size
        segments: list[Segment] = []
        for f0, f1 in _accepted_runs(good):
            if (f1 - f0) * FRAME_HOP_S < min_seg_s:
                continue
            in_span = (conf_times >= ft[f0]) & (conf_times < f1 * FRAME_HOP_S)
            vk_conf = (
                confidence[:, in_span].mean(axis=1) if in_span.any() else np.full(N_ROTORS, np.nan)
            )
            segments.append(
                Segment(
                    rid=rid,
                    index=len(segments),
                    f0=int(f0),
                    f1=int(f1),
                    whole=(f0 == 0 and f1 == n_frames),
                    rps=np.ascontiguousarray(rps[:, f0:f1], dtype=np.float32),
                    vk_conf_per_rotor=vk_conf,
                    comb_conf_per_rotor=comb_conf,
                    refuse_conf=refuse_conf,
                )
            )
        out[rid] = segments
    return out


def print_segment_table(segments: dict[str, list[Segment]]) -> None:
    print(
        f"{'key':<18}{'span (s)':<18}{'dur (s)':<9}{'vk_conf (per rotor)':<30}"
        f"{'median rps (per rotor)'}"
    )
    total = 0.0
    for rid in sorted(segments):
        if not segments[rid]:
            print(f"{rid:<18}(no accepted segment >= min length)")
            continue
        for seg in segments[rid]:
            conf = " ".join(f"{c:.3f}" for c in seg.vk_conf_per_rotor)
            med = " ".join(f"{m:.1f}" for m in np.median(seg.rps, axis=1))
            print(
                f"{seg.key:<18}{f'{seg.t0:.1f}-{seg.t1:.1f}':<18}"
                f"{seg.duration_s:<9.1f}{conf:<30}{med}"
            )
            total += seg.duration_s
    n = sum(len(s) for s in segments.values())
    print(f"total: {n} segments, {total:.1f} s")


def _iter_samples(
    segments: dict[str, list[Segment]],
    source_version: str | None,
    annotator_commit: str,
    labels_dir: Path,
) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Stream ``AVQ-egonoise``, cut the accepted segments, emit labeled Frames."""
    ds = streams.DloadFrameDataset(SOURCE_DATASET, version=source_version)
    seen: set[str] = set()
    for frame in ds:
        rid = str(get_meta(frame, "recording_id", ""))
        segs = segments.get(rid)
        if not segs:
            continue
        audio = frame["audio"]
        assert isinstance(audio, td.Series)
        idx = audio.tindex
        if not isinstance(idx, td.GridIndex) or int(idx.sr) != SAMPLE_RATE:
            raise ValueError(f"{rid}: expected uniformly-sampled {SAMPLE_RATE} Hz audio")
        data = np.asarray(audio.data, dtype=np.float32)
        if data.ndim != 1:
            raise ValueError(f"{rid}: expected mono audio, got shape {data.shape}")
        n_total = data.shape[-1]
        for seg in segs:
            s0 = seg.f0 * HOP
            s1 = min(n_total, seg.f1 * HOP)
            if s0 >= n_total:
                raise ValueError(f"{seg.key}: segment start beyond audio ({s0} >= {n_total})")
            seg_audio = np.ascontiguousarray(data[s0:s1], dtype=np.float32)
            n_frames = seg.f1 - seg.f0
            stamps = np.arange(n_frames, dtype=np.float64) * FRAME_HOP_S
            rps_series = td.events(stamps, seg.rps, dims=("rotor", "time"))
            out = with_meta(
                td.Frame(
                    {
                        "audio": audio_series(seg_audio[None, :], SAMPLE_RATE),
                        "rps": rps_series,
                        "meta": frame["meta"],
                    }
                ),
                recording_id=seg.key,
                source_recording_id=rid,
                segment_index=seg.index,
                segment_start_s=round(seg.t0, 3),
                segment_end_s=round(seg.t1, 3),
                sample_rate=SAMPLE_RATE,
                n_channels=1,
                duration_s=round(seg_audio.shape[-1] / SAMPLE_RATE, 3),
                rps_grid_s=FRAME_HOP_S,
                rps_source="vk_blind_pseudolabel",
                annotator=ANNOTATOR,
                annotator_commit=annotator_commit,
                labels_dir=str(labels_dir),
                refuse_conf=seg.refuse_conf,
                vk_conf_mean=round(float(np.nanmean(seg.vk_conf_per_rotor)), 4),
                vk_conf_per_rotor=[round(float(c), 4) for c in seg.vk_conf_per_rotor],
                comb_conf_per_rotor=[round(float(c), 4) for c in seg.comb_conf_per_rotor],
                source_dataset=SOURCE_DATASET,
                source_version=str(ds.version),
                derived_note=(
                    f"Contiguous blind-VK-accepted segment of {SOURCE_DATASET}/{rid} "
                    f"({seg.t0:.1f}-{seg.t1:.1f} s); rps = stitched VK pseudo-label on "
                    f"the {FRAME_HOP_S} s grid, refused (NaN) spans dropped."
                ),
            )
            print(
                f"  {seg.key}: {seg_audio.shape[-1] / SAMPLE_RATE:.1f} s, "
                f"vk_conf {np.nanmean(seg.vk_conf_per_rotor):.3f}",
                flush=True,
            )
            yield seg.key, streams.frame_to_sample(out)
        seen.add(rid)
    missing = {rid for rid, segs in segments.items() if segs} - seen
    if missing:
        raise ValueError(f"{SOURCE_DATASET} did not yield recordings {sorted(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="directory with vk_pseudolabel.py per-recording NPZs + summary.csv",
    )
    ap.add_argument(
        "--min-seg-s",
        type=float,
        default=DEFAULT_MIN_SEG_S,
        help="drop accepted segments shorter than this (s)",
    )
    ap.add_argument(
        "--source-version", default=None, help="AVQ-egonoise version (default: dload.lock pin)"
    )
    ap.add_argument(
        "--annotator-commit",
        default=None,
        help=f"provenance commit of {ANNOTATOR} (default: git log -1 on the script)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the segment table and exit (no publish)"
    )
    ap.add_argument("--pin", action="store_true")
    args = ap.parse_args()

    segments = load_segments(args.labels_dir, args.min_seg_s)
    print(f"{DATASET_NAME}: segments from {args.labels_dir} (min {args.min_seg_s:g} s)")
    print_segment_table(segments)
    if args.dry_run:
        return

    annotator_commit = args.annotator_commit or _annotator_commit()
    n_segments = sum(len(s) for s in segments.values())
    total_s = sum(seg.duration_s for segs in segments.values() for seg in segs)
    refuse_conf = next(seg.refuse_conf for segs in segments.values() for seg in segs)
    repo = streams.open_repository()
    print(f"\n{DATASET_NAME}: streaming {SOURCE_DATASET} for {n_segments} segments", flush=True)
    manifest = repo.commit(
        DATASET_NAME,
        _iter_samples(segments, args.source_version, annotator_commit, args.labels_dir),
        meta={
            streams.LAYOUT_META_KEY: streams.TDFRAME_LAYOUT,
            "description": (
                "AVQ rotor ego-noise (mono 16 kHz) joined with blind-VK RPS pseudo-labels "
                f"(scripts/vk_pseudolabel.py @ {annotator_commit[:12]}): one Frame per "
                "contiguous accepted segment (recordings split at NaN/refused spans, "
                f"segments >= {args.min_seg_s:g} s kept; a frame is accepted iff all 4 "
                "rotor labels are finite). audio + rps (rotor, time) on the "
                f"{FRAME_HOP_S} s grid + provenance/confidence meta. The beat-VK R2 "
                "training source (kind: frames in the online-mix noise pool)."
            ),
            "source_dataset": SOURCE_DATASET,
            "labels_dir": str(args.labels_dir),
            "annotator": ANNOTATOR,
            "annotator_commit": annotator_commit,
            "refuse_conf": refuse_conf,
            "min_segment_s": float(args.min_seg_s),
            "sample_rate": SAMPLE_RATE,
            "n_channels": 1,
            "n_segments": n_segments,
            "total_duration_s": round(total_s, 1),
            "segments": [seg.key for segs in segments.values() for seg in segs],
            "source": "scripts/publish_avq_vkrps.py",
        },
        recipe=Path(__file__).read_text(encoding="utf-8"),
        progress=print,
    )
    print(f"{DATASET_NAME}@{manifest.version[:12]}: {manifest.num_samples} samples", flush=True)
    if args.pin:
        repo.pin(DATASET_NAME, manifest.version)
        print(f"pinned {DATASET_NAME}@{manifest.version[:12]}", flush=True)


if __name__ == "__main__":
    main()
