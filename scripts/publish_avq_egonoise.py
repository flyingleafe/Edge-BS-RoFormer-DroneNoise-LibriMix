#!/usr/bin/env python3
"""Publish AVQ-egonoise — the F2 noise pool, and nothing else.

The F2 survey replication (``docs/experiments/f2-survey-replication.md``) trains
and validates on exactly 5 of ``AVQ``'s 12 recordings — the pure rotor ego-noise
sequences ``S1_seq1``/``S1_seq2``/``S1_seq3``/``S2_seq1``/``S2_seq2`` (the only
ones without an ``angle_vad`` entry; the other 7 carry the speech source) — and
of those only **channel 0** at **16 kHz**, per the paper. Selecting that subset
out of ``AVQ`` at training time costs far more than it is worth:

- the dload manifest carries no per-shard key list, so an ``include_keys`` pool
  must OPEN (download) every one of AVQ's 11 shards, ~4 GiB of 8-ch 44.1 kHz
  audio + video-labeled sequences, to discover which hold the 5 wanted keys;
- AVQ's largest shard is a 352 MB, 42-part multipart upload, and s3transfer's
  ETag validation on some boto3 builds (Kaggle) rejects a multipart ETag
  outright (``S3DownloadFailedError ... did not match expected ETag``). The
  object is NOT corrupt — its sha256 matches its content-address digest — but
  the download fails there regardless.

So this publishes the wanted audio *as its own dataset*: 5 mono 16 kHz frames,
~705 s total, ~45 MB — one small, single-part shard, consumed by a plain
``kind: audio_pool`` entry with no ``include_keys`` and no ``channel``.

    python scripts/publish_avq_egonoise.py [--pin]

Sample key = the AVQ recording id (``S1_seq1``, ...); the per-recording ``meta``
is AVQ's, with ``sample_rate``/``n_channels``/``duration_s`` updated for the
derived mono 16 kHz audio and provenance fields added. Geometry (``mic_pos``) is
dropped: a single channel has no array.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tdseries as td

from data_processing import streams
from data_processing.frames import audio_series, get_meta, resample_audio_series, with_meta

SOURCE_DATASET = "AVQ"
DATASET_NAME = "AVQ-egonoise"
SAMPLE_RATE = 16000
CHANNEL = 0

# The 5 pure ego-noise sequences (same list as
# conf/online_mix/se_avq_survey.yaml's former include_keys and
# scripts/build_se_valid.py's AVQ_EGO_NOISE_KEYS).
EGO_NOISE_KEYS = ["S1_seq1", "S1_seq2", "S1_seq3", "S2_seq1", "S2_seq2"]


def _iter_egonoise(source_version: str | None = None) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Stream ``AVQ``, keep the 5 ego-noise recordings, emit mono 16 kHz frames."""
    ds = streams.DloadFrameDataset(SOURCE_DATASET, version=source_version)
    wanted = set(EGO_NOISE_KEYS)
    seen: set[str] = set()
    for frame in ds:
        rid = str(get_meta(frame, "recording_id", ""))
        if rid not in wanted:
            continue
        if "angle_vad" in frame:  # belt and braces: these 5 carry no DOA/VAD label
            raise ValueError(f"{rid} has an angle_vad entry — it is not a pure ego-noise sequence")
        audio = frame["audio"]  # (mic, time) @ 44.1 kHz
        assert isinstance(audio, td.Series)
        idx = audio.tindex
        if not isinstance(idx, td.GridIndex):
            raise ValueError(f"{rid}: audio is not uniformly sampled")
        mono = audio.data[CHANNEL] if audio.data.ndim == 2 else audio.data
        series = resample_audio_series(
            audio_series(np.ascontiguousarray(mono, dtype=np.float32)[None, :], int(idx.sr)),
            SAMPLE_RATE,
        )
        n = int(np.asarray(series.data).shape[-1])
        out = with_meta(
            td.Frame({"audio": series, "meta": frame["meta"]}),
            sample_rate=SAMPLE_RATE,
            n_channels=1,
            duration_s=round(n / SAMPLE_RATE, 3),
            source_dataset=SOURCE_DATASET,
            source_version=str(ds.version),
            source_channel=CHANNEL,
            derived_note=(
                f"Derived subset of {SOURCE_DATASET}: the 5 pure rotor ego-noise sequences, "
                f"channel {CHANNEL} only, soxr_hq-resampled 44100 -> {SAMPLE_RATE} Hz mono."
            ),
        )
        print(f"  {rid}: {n / SAMPLE_RATE:.1f} s mono @ {SAMPLE_RATE} Hz", flush=True)
        seen.add(rid)
        yield rid, streams.frame_to_sample(out)
    missing = wanted - seen
    if missing:
        raise ValueError(f"{SOURCE_DATASET} did not yield the ego-noise keys {sorted(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-version", default=None, help="AVQ version (default: dload.lock pin)")
    ap.add_argument("--pin", action="store_true")
    args = ap.parse_args()

    repo = streams.open_repository()
    print(f"{DATASET_NAME}: streaming {SOURCE_DATASET} for {len(EGO_NOISE_KEYS)} keys", flush=True)
    manifest = repo.commit(
        DATASET_NAME,
        _iter_egonoise(args.source_version),
        meta={
            streams.LAYOUT_META_KEY: streams.TDFRAME_LAYOUT,
            "description": (
                "F2 noise pool: the 5 pure rotor ego-noise sequences of the AVQ audio-visual "
                "quadrotor dataset (S1_seq1/2/3, S2_seq1/2 — the AVQ recordings without an "
                "angle_vad entry, i.e. without the speech source), channel 0 only, resampled "
                "to 16 kHz mono (~705 s total). A purpose-built derived subset of AVQ so the "
                "F2 online-mix stream needs no include_keys/channel filtering and never "
                "touches AVQ's 11 shards (~4 GiB) or its 352 MB multipart shard."
            ),
            "source_dataset": SOURCE_DATASET,
            "source_channel": CHANNEL,
            "sample_rate": SAMPLE_RATE,
            "n_channels": 1,
            "recordings": EGO_NOISE_KEYS,
            "source": "scripts/publish_avq_egonoise.py",
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
