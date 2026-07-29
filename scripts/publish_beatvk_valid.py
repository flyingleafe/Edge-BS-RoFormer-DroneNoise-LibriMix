#!/usr/bin/env python3
"""Publish ``beatvk-valid-raw`` — the beat-VK campaign's FIXED raw validation set.

This file is the *dataset half* of the frozen beat-VK evaluation protocol
(the metric half is ``scripts/beatvk_eval.py``). Everything — neural RPS
predictors, blind VK trackers, hybrids — is re-scored on exactly this data.

Composition (frozen, user-decided): 4 recordings —
DREGON ``free-flight_nosource_room1``, ``free-flight_speech-low_room1``,
``free-flight_whitenoise-low_room1`` and Michael's ``FLY124``.

Sourcing / provenance: recordings are read FROM the pinned rich-frame
datasets ``DREGON-frames`` / ``michaels-frames`` (dload.lock pins recorded in
each sample's meta), never from local raw directories, so the published bytes
are a pure function of those pins.

Per-recording Frame (``tdframe-v1`` codec, one sample per recording):

- ``audio``: the source frame's audio UNTOUCHED — native sample rate
  (44.1 kHz), all 8 channels, original dtype. Time is re-anchored so the
  audio starts at t=0 (the one transformation applied): all window/manifest
  times below are seconds from audio start.
- ``rps_raw``: the RAW measured rotor telemetry on its native timestamps —
  DREGON ``motors_measured`` (~1 kHz), FLY124 the aligned ``rps`` track
  (~29 Hz). No smoothing, no gridding, no cleaning; stamps shifted by the
  same re-anchor offset.
- ``meta``: source dataset + version pin, recording id, and the WINDOW
  MANIFEST (see below), plus the full-recording audio/telemetry spans so
  future protocols can re-window WITHOUT changing the data.

Window manifest (the frozen eval protocol):

- *Eval span* = (audio span ∩ telemetry span) minus the leading and trailing
  maximal exact-constant telemetry runs. Those runs are logger-not-live
  artifacts, not flight: DREGON ``motors_measured`` opens/closes with ~1 s /
  ~0.3 s of exact zeros before the logger goes live, and FLY124 ends with a
  12.1 s frozen tail (constant [0, 0, -0.0167, 0]). Genuine ground/warmup
  telemetry fluctuates sample-to-sample and is therefore KEPT (this is the
  ``min_motor_rps=0`` decision: no in-flight thresholding — only frozen
  logger spans are excluded).
- *Windows*: contiguous non-overlapping 16 s windows tiling the eval span
  from its start; a trailing remainder < 16 s is dropped.
- *Regime tag* per window, from the mean of the raw telemetry linearly
  interpolated onto the window's 0.032 s frame grid (500 points — the same
  grid the scorer uses), averaged over the 4 rotors:
  mean < 5 rev/s → ``ground``, < 45 → ``warmup``, else ``cruise``.

Run, then pin::

    python scripts/publish_beatvk_valid.py [--dry-run]
    dload pin beatvk-valid-raw && git add dload.lock

Idempotent: shards are content-addressed; re-running re-uploads nothing that
already exists. The script source is stored as the version's recipe.
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import tdseries as td

from data_processing import streams

DATASET = "beatvk-valid-raw"

#: (source dataset, sample key, raw telemetry entry) per recording, in order.
RECORDINGS: tuple[tuple[str, str, str], ...] = (
    ("DREGON-frames", "free-flight_nosource_room1", "motors_measured"),
    ("DREGON-frames", "free-flight_speech-low_room1", "motors_measured"),
    ("DREGON-frames", "free-flight_whitenoise-low_room1", "motors_measured"),
    ("michaels-frames", "FLY124", "rps"),
)

WINDOW_S = 16.0
FRAME_S = 0.032  # the scorer's fixed evaluation grid (= 512 / 16000)
WINDOW_FRAMES = int(round(WINDOW_S / FRAME_S))  # 500
REGIME_GROUND_MAX = 5.0  # mean rps below -> "ground"
REGIME_WARMUP_MAX = 45.0  # mean rps below -> "warmup"; else "cruise"

Sample = tuple[str, dict[str, bytes]]


def trim_constant_runs(ts: np.ndarray, vals: np.ndarray) -> tuple[float, float, float, float]:
    """Trim leading/trailing exact-constant telemetry runs (logger not live).

    ``ts`` is ``(M,)`` stamps (seconds), ``vals`` ``(R, M)``. A run is a
    maximal block of consecutive samples whose full rotor vector is exactly
    identical. Returns ``(t_live_start, t_live_end, lead_s, trail_s)`` where
    the live span endpoints are the boundary samples of the constant blocks
    (the first/last sample at which the value stream is moving again).
    """
    same = np.all(vals[:, 1:] == vals[:, :-1], axis=0)  # (M-1,) consecutive equality
    lead = 0
    while lead < len(same) and same[lead]:
        lead += 1
    trail = 0
    while trail < len(same) and same[len(same) - 1 - trail]:
        trail += 1
    t0, t1 = float(ts[lead]), float(ts[len(ts) - 1 - trail])
    if t1 <= t0:
        raise ValueError("telemetry is entirely constant — no live span")
    return t0, t1, t0 - float(ts[0]), float(ts[-1]) - t1


def window_mean_rps(ts: np.ndarray, vals: np.ndarray, start: float) -> float:
    """Mean raw RPS over ``[start, start + WINDOW_S)`` on the 0.032 s grid.

    Per-rotor linear interpolation of the raw telemetry onto the window's
    ``WINDOW_FRAMES`` grid points, then a global mean — exactly the GT
    construction ``beatvk_eval.py`` scores against.
    """
    grid = start + np.arange(WINDOW_FRAMES) * FRAME_S
    per_rotor = [np.interp(grid, ts, vals[r]) for r in range(vals.shape[0])]
    return float(np.mean(per_rotor))


def regime_of(mean_rps: float) -> str:
    if mean_rps < REGIME_GROUND_MAX:
        return "ground"
    if mean_rps < REGIME_WARMUP_MAX:
        return "warmup"
    return "cruise"


def build_manifest(
    audio: td.Series, ts: np.ndarray, vals: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The window manifest for one recording (times in re-anchored seconds)."""
    live_t0, live_t1, lead_s, trail_s = trim_constant_runs(ts, vals)
    span_start = max(float(audio.t_start), live_t0)
    span_end = min(float(audio.t_end), live_t1)
    windows: list[dict[str, Any]] = []
    start = span_start
    while start + WINDOW_S <= span_end + 1e-9:
        mean_rps = window_mean_rps(ts, vals, start)
        windows.append(
            {
                "index": len(windows),
                "start_s": round(start, 6),
                "end_s": round(start + WINDOW_S, 6),
                "regime": regime_of(mean_rps),
                "mean_rps": round(mean_rps, 4),
            }
        )
        start += WINDOW_S
    spans = {
        "audio": [float(audio.t_start), float(audio.t_end)],
        "telemetry": [float(ts[0]), float(ts[-1])],
        "eval": [span_start, span_end],
        "telemetry_const_trim_s": [round(lead_s, 4), round(trail_s, 4)],
    }
    return spans, windows


def build_frame(source: str, key: str, rps_entry: str, src: td.Frame) -> td.Frame:
    """One source recording Frame -> the published beatvk-valid-raw Frame."""
    audio_src = src["audio"]
    rps_src = src[rps_entry]
    sr = audio_src.tindex.sr
    if float(sr) != int(sr):
        raise ValueError(f"{key}: non-integer audio sample rate {sr}")

    # Re-anchor time so the audio starts at t=0; telemetry stamps shift by the
    # same offset, so audio/telemetry alignment is exactly preserved. The
    # audio data array itself is byte-identical to the source.
    shift = float(audio_src.t_start)
    audio = td.uniform(np.asarray(audio_src.data), int(sr), dims=("mic", "time"), t_start=0.0)
    ts = cast(td.StampIndex, rps_src.tindex).abs_stamps.astype(np.float64) - shift
    vals = np.asarray(rps_src.data)
    rps_raw = td.events(ts, vals, dims=("rotor", "time"))

    spans, windows = build_manifest(audio, ts, vals)
    rate_hz = float(1.0 / np.median(np.diff(ts)))
    meta = {
        "recording_id": key,
        "source_dataset": source,
        "source_version": _SOURCE_VERSIONS[source],
        "source_rps_entry": rps_entry,
        "sample_rate": int(sr),
        "n_channels": int(audio.dim_size("mic")),
        "rps_rate_hz": round(rate_hz, 2),
        "time_anchor_offset_s": shift,
        "spans": spans,
        "window_s": WINDOW_S,
        "regime_thresholds": {"ground_max": REGIME_GROUND_MAX, "warmup_max": REGIME_WARMUP_MAX},
        "windows": windows,
    }
    return td.Frame({"audio": audio, "rps_raw": rps_raw, "meta": td.Frame(meta)})


_SOURCE_VERSIONS: dict[str, str] = {}


def iter_samples(dry_run: bool) -> Iterator[Sample]:
    repo = streams.open_repository()
    stats: list[str] = []
    for source in dict.fromkeys(src for src, _, _ in RECORDINGS):
        ds = repo.dataset(source)
        _SOURCE_VERSIONS[source] = ds.version
        wanted = {key: entry for src_name, key, entry in RECORDINGS if src_name == source}
        seen: set[str] = set()
        for key, fields in ds.samples():
            if key not in wanted:
                continue
            seen.add(key)
            frame = build_frame(source, key, wanted[key], streams.decode_tdframe((key, fields)))
            meta = frame["meta"]
            windows = meta["windows"]
            counts: dict[str, int] = {}
            for w in windows:
                counts[w["regime"]] = counts.get(w["regime"], 0) + 1
            dur = frame["audio"].t_end - frame["audio"].t_start
            span = meta["spans"]["eval"]
            stats.append(
                f"  {key} ({source}@{ds.version[:12]}): audio {dur:.1f} s, "
                f"eval span [{span[0]:.2f}, {span[1]:.2f}] s, "
                f"{len(windows)} windows ({counts})"
            )
            print(stats[-1], flush=True)
            if not dry_run:
                yield key, streams.frame_to_sample(frame)
            del frame
            gc.collect()
        missing = set(wanted) - seen
        if missing:
            raise KeyError(f"recordings {sorted(missing)} not found in {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="build frames and print stats without publishing"
    )
    args = parser.parse_args()

    if args.dry_run:
        for _ in iter_samples(dry_run=True):
            pass
        print("[dry-run] nothing published")
        return

    repo = streams.open_repository()
    manifest = repo.commit(
        DATASET,
        iter_samples(dry_run=False),
        meta={
            streams.LAYOUT_META_KEY: streams.TDFRAME_LAYOUT,
            "description": (
                "Beat-VK campaign FIXED raw validation set: 4 recordings (3 DREGON "
                "free-flight room1 + FLY124) as native-rate 8ch audio + RAW measured "
                "rotor telemetry (no smoothing/gridding) + a frozen 16 s window "
                "manifest with regime tags. Scored ONLY by scripts/beatvk_eval.py."
            ),
            "source": "pinned DREGON-frames / michaels-frames (versions in sample meta)",
            "protocol": {
                "window_s": WINDOW_S,
                "frame_s": FRAME_S,
                "regime_thresholds": {
                    "ground_max": REGIME_GROUND_MAX,
                    "warmup_max": REGIME_WARMUP_MAX,
                },
            },
        },
        recipe=Path(__file__).read_text(encoding="utf-8"),
        progress=print,
    )
    print(f"\n{DATASET}@{manifest.version[:12]}: {manifest.num_samples} samples")
    print(f"Now: dload pin {DATASET} && git add dload.lock")


if __name__ == "__main__":
    main()
