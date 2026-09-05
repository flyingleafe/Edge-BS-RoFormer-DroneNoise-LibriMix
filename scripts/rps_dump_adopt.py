#!/usr/bin/env python3
"""Adopt the training-free baselines into a ``scripts/rps_dump.py`` dump.

The neural rows of the wrap-up paper are scored by ``scripts/rps_regime_table.py``
over ``results/rps_dump/real/<experiment>.npz``. The classical estimators, the
OT multi-pitch baseline and the blind two-stage tracker were each scored by
their own driver, so their numbers came from three different aggregations. This
CLI converts their stored per-unit outputs into the dump format, so ONE scoring
script reads every row of the leaderboard.

The dump format, per set directory:

    _gt.npz      rps (N, R, T) rev/s on the label grid, n_t (N,)
    _meta.json   one dict per sample: recording_id (``sample_000NN``), channel
    <name>.npz   pred (N, R, T) rev/s, n_t (N,), metric (N,)

``metric`` is ``metrics.rps.rps_mae_frame`` — the statistic behind
``RPSMetric("mae_frame")``, which is what ``rps_dump`` stores for a model that
emits ``rps_pred``. Samples come out in the order of ``_meta.json``: clip-major
(``sample_00000`` first) and microphone-minor (channel 0 to 7).

Sources
-------

``blind_ungated`` / ``blind_gated`` — ``results/blind_valid_row/vit2dsp``.
    ADOPTED. ``traj/<uid>.npz`` holds each window's ``(R, T)`` trajectory and
    its own time base, and ``raw/<uid>.json`` holds the acceptance readings.
    The two rows are the two refusal conventions of the campaign: ``ungated``
    keeps every window, ``gated`` decodes a window that fails gate g1 or g5 to
    0 rev/s. This CLI calls ``scripts/blind_valid_row.load_windows`` and
    ``predict`` unchanged, so the stitch (midpoint cut, refusal to zero) is the
    campaign's.

``classical_pyin`` / ``_cepstral`` / ``_hps`` / ``_matched_filter`` / ``_nmf``
    — ``results/classical_valid_eval/raw``. NOT adoptable as stored: a unit
    JSON holds per-regime sums of ``|err|`` and ``err**2`` only, and no
    prediction. ``--recompute`` runs the estimator again instead. The five
    estimators are training-free, deterministic and pure DSP, so a rerun is the
    same prediction, not a new one. Cost: about 1.5 CPU-s per unit.

``otmp`` — ``results/otmp_protocol/raw``. NOT adoptable as stored either: a
    unit JSON holds one ``mae`` and one ``mse`` per OT frame, and no pitch.
    ``--recompute`` runs ``experiments.otmp_baseline.estimate.estimate_clip``
    with the ``adapted`` preset, which is the preset the campaign ran. Cost,
    measured: about 370 CPU-s per unit at one BLAS thread, thus about 30
    CPU-hours for the 296-unit grid. This is a cluster job, not a laptop job.

A recompute writes one prediction JSON per unit under ``--cache`` through
``utils.gridrun``, so it resumes after a kill and one bad unit does not kill
the pool. The adoption step then reads that directory, which is why a
recomputed row and an adopted row go through the same code.

Grids
-----

The dump grid is the label grid of the set: 251 frames of the 2048/512 STFT at
16 kHz for the frozen real split. Each source meets it differently:

* The blind tracker carries a continuous trajectory, so it is EVALUATED at the
  dump's own frame times (``start_time + f * 512 / 16000`` of the clip). No
  resampling happens, and the result is the track ``blind_valid_row score``
  scores.
* The classical estimators already run on the 2048/512 grid, thus they need at
  most a crop.
* The OT baseline reports one estimate per 1 s frame, thus 8 frames per clip.
  Its ``(4, 8)`` estimate is resampled onto the 251 frames with
  ``experiments.rps_bench.resample_like_metric``, the resampling
  ``scripts/rps_regime_table.py`` itself applies when a prediction grid and a
  label grid differ.

Any source whose grid differs from the dump grid goes through
``resample_like_metric``, and the manifest records that with
``frames_resampled``.

Microphones
-----------

The blind tracker makes ONE trajectory per parent recording from all eight
microphones: ``blind_seed`` channel-averages its whitened spectrogram and its
second stage is a spatial joint Viterbi. Thus one prediction covers the eight
channel samples of a clip, and this CLI replicates it across them. The manifest
marks such a row with ``unit_kind`` ``8-channel``. The classical estimators and
the OT baseline are per-channel, thus their rows carry 296 distinct predictions.

Output
------

``<dump>/<name>.npz`` per row, plus the manifest ``<dump>/_adopted.json``:
name -> source path, unit kind, mode (``adopted`` / ``recomputed`` /
``not_adoptable``), whether frames were resampled, and the mean metric. The
manifest merges with an existing one, so a later run keeps the earlier rows.

Run::

    python scripts/rps_dump_adopt.py --names blind_ungated blind_gated
    python scripts/rps_dump_adopt.py --names classical_nmf --recompute --jobs 8
"""

from __future__ import annotations

import os
import sys


def _early_arg(name: str, default: str) -> str:
    """Read one ``--name value`` / ``--name=value`` argument before heavy imports."""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


# Cap the BLAS thread pools BEFORE numpy and torch. A --recompute runs one
# process per core, and ``utils.gridrun`` sets these variables too late for
# this module: numpy and torch are imported here first, and a forked worker
# inherits a pool that is already sized. An explicit environment wins.
_OMP = _early_arg("--omp", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _OMP)

import argparse  # noqa: E402
import json  # noqa: E402
from collections import OrderedDict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.rps_bench import resample_like_metric  # noqa: E402
from metrics.rps import rps_mae_frame  # noqa: E402
from utils.gridrun import Unit, run_grid  # noqa: E402

DATASET = "dload:DREGON-LM-V4-michaels-valid-full"
N_ROTORS = 4
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 16000

DEFAULT_DUMP = "results/rps_dump/real"
DEFAULT_CACHE = "results/rps_dump_adopt"
CLASSICAL_RAW = "results/classical_valid_eval/raw"
OTMP_RAW = "results/otmp_protocol/raw"
BLIND_ANNOT = "results/blind_valid_row/vit2dsp"

CLASSICAL_METHODS = ("pyin", "cepstral", "hps", "matched_filter", "nmf")
BLIND_GATES = "g1,g5"

#: every row this CLI knows how to write
NAMES = (
    *(f"classical_{m}" for m in CLASSICAL_METHODS),
    "otmp",
    "blind_ungated",
    "blind_gated",
)


class NotAdoptable(Exception):
    """The stored output of a source holds no per-frame prediction."""


# ─── The dump ─────────────────────────────────────────────────────────────────


def read_dump(dump: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[str]]:
    """``(gt, n_t, order, clip_ids)`` of one dumped set.

    ``order[i]`` is the ``(clip index, channel)`` of sample ``i``, read from
    ``_meta.json``. ``clip_ids`` holds the clip ids in first-seen order, which
    is the dataset order the sources index by.
    """
    z = np.load(dump / "_gt.npz")
    meta = json.loads((dump / "_meta.json").read_text())
    clips: OrderedDict[str, int] = OrderedDict()
    order = []
    for row in meta:
        key = str(row.get("recording_id", row.get("sample_id", "?")))
        order.append((clips.setdefault(key, len(clips)), int(row.get("channel", 0))))
    return z["rps"], z["n_t"], order, list(clips)


def assemble(
    per_unit: dict[tuple[int, int | None], np.ndarray],
    gt: np.ndarray,
    n_t: np.ndarray,
    order: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Per-unit predictions -> ``(pred, n_t, metric, frames_resampled)``.

    A unit keyed ``(clip, None)`` covers every microphone of that clip. A
    prediction whose frame count differs from the sample's label goes through
    ``resample_like_metric``. Every sample must be covered.
    """
    n_samples = len(order)
    width = int(n_t.max())
    pred = np.full((n_samples, N_ROTORS, width), np.nan, dtype=np.float32)
    metric = np.empty(n_samples, dtype=np.float64)
    resampled = False
    for i, (clip, channel) in enumerate(order):
        p = per_unit.get((clip, channel))
        if p is None:
            p = per_unit.get((clip, None))
        if p is None:
            raise SystemExit(f"no prediction for clip {clip} channel {channel}")
        target = int(n_t[i])
        if p.shape[-1] != target:
            p = resample_like_metric(np.asarray(p, dtype=np.float64), target)
            resampled = True
        p = np.asarray(p, dtype=np.float64)
        pred[i, :, :target] = p.astype(np.float32)
        metric[i] = rps_mae_frame(p, gt[i, :, :target].astype(np.float64))
    return pred, n_t.astype(np.int64), metric, resampled


# ─── Source: the blind two-stage tracker ──────────────────────────────────────


def clip_frames(order: list[tuple[int, int]], n_t: np.ndarray, n_clips: int) -> list[int]:
    """Frames per clip, from the first dump sample of each clip."""
    out = [0] * n_clips
    seen = set()
    for i, (clip, _channel) in enumerate(order):
        if clip not in seen:
            seen.add(clip)
            out[clip] = int(n_t[i])
    return out


def blind_units(
    annot: Path,
    *,
    gated: bool,
    clip_ids: list[str],
    frames: list[int],
) -> dict[tuple[int, int | None], np.ndarray]:
    """The stitched blind track of every clip, on the clips' own frame times.

    ``scripts/blind_valid_row`` supplies the stitch and the refusal convention.
    This function only picks the times at which to read the result: the frame
    times of the clip, which are ``start_time + f * 512 / 16000`` in the parent
    recording. One track covers all 8 microphones, thus the key channel is
    ``None``.
    """
    import blind_valid_row as bvr

    gates = {g.strip() for g in BLIND_GATES.split(",") if g.strip()}
    by_rec = bvr.load_windows(annot, gates)
    _root, clips = bvr.clip_index(DATASET)
    by_id = {str(c["id"]): c for c in clips}

    out: dict[tuple[int, int | None], np.ndarray] = {}
    for idx, cid in enumerate(clip_ids):
        clip = by_id.get(cid)
        if clip is None:
            raise SystemExit(f"{cid} is in the dump but not in {DATASET}")
        rid = bvr.parent_recording(str(clip["recording_id"]))
        t_abs = float(clip["start_time"]) + np.arange(frames[idx]) * (HOP_LENGTH / SAMPLE_RATE)
        out[(idx, None)] = bvr.predict(by_rec.get(rid, []), t_abs, gated=gated)
    return out


# ─── Sources that must be recomputed ──────────────────────────────────────────


_DATASETS: dict[int, Any] = {}


def _clip_audio(clip: int, channel: int) -> np.ndarray:
    """One clip of one microphone, as the source drivers read it.

    The dataset is built once per channel per worker process: a unit is under
    a second of work for the classical estimators, and a fresh dataset per unit
    would cost more than the estimator.
    """
    from data_processing.frame_datasets import DregonLMFrameDataset

    ds = _DATASETS.get(channel)
    if ds is None:
        ds = _DATASETS[channel] = DregonLMFrameDataset(
            data_dir=DATASET,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            sample_rate=SAMPLE_RATE,
            channel=channel,
        )
    return np.asarray(ds[clip]["mixture"].data, dtype=np.float32).reshape(-1)


def classical_worker(unit: Unit) -> dict[str, Any]:
    """One (method, clip, channel) classical prediction, on the STFT grid."""
    from experiments.classical_rps.predictors import CLASSICAL_TRACKERS

    method = str(unit.params["method"])
    clip = int(unit.params["clip"])
    channel = int(unit.params["channel"])
    audio = _clip_audio(clip, channel)
    pred = np.asarray(CLASSICAL_TRACKERS[method](audio), dtype=np.float64)
    return {"clip": clip, "channel": channel, "pred": pred.tolist()}


def otmp_worker(unit: Unit) -> dict[str, Any]:
    """One (clip, channel) OT multi-pitch prediction, on the OT frame grid."""
    from experiments.otmp_baseline.estimate import adapted_drone_config, estimate_clip

    clip = int(unit.params["clip"])
    channel = int(unit.params["channel"])
    cfg = adapted_drone_config()
    audio = _clip_audio(clip, channel).astype(np.float64)
    times, pitches = estimate_clip(audio, cfg.sample_rate, cfg)
    return {
        "clip": clip,
        "channel": channel,
        "times_s": np.asarray(times, dtype=float).tolist(),
        "pred": np.asarray(pitches, dtype=np.float64).tolist(),
    }


def read_pred_units(raw: Path) -> dict[tuple[int, int | None], np.ndarray]:
    """Every ``raw/<uid>.json`` that carries a ``pred`` array."""
    out: dict[tuple[int, int | None], np.ndarray] = {}
    for path in sorted(raw.glob("*.json")):
        row = json.loads(path.read_text())
        if "pred" not in row:
            raise NotAdoptable(
                f"{path} holds no per-frame prediction "
                f"(keys {sorted(row)}); rerun the estimator with --recompute"
            )
        out[(int(row["clip"]), int(row["channel"]))] = np.asarray(row["pred"], dtype=np.float64)
    if not out:
        raise NotAdoptable(f"no unit JSON under {raw}")
    return out


def recompute(name: str, cache: Path, n_clips: int, n_channels: int, jobs: int) -> Path:
    """Run the estimator of ``name`` over the grid, one prediction JSON per unit."""
    out_dir = cache / name
    if name == "otmp":
        worker, params = otmp_worker, {}
    else:
        method = name.removeprefix("classical_")
        worker, params = classical_worker, {"method": method}
    # Channel-major, so a pool worker keeps reading the same channel and reuses
    # the dataset it already built.
    units = [
        Unit(uid=f"clip{c:03d}_ch{ch}", params={**params, "clip": c, "channel": ch})
        for ch in range(n_channels)
        for c in range(n_clips)
    ]
    result = run_grid(units, worker, out_dir, jobs=jobs, summarize=lambda rows: {"n": len(rows)})
    if result.n_failed:
        raise SystemExit(f"{name}: {result.n_failed} unit(s) failed; see {out_dir}/raw/*.err")
    return out_dir / "raw"


# ─── Rows ─────────────────────────────────────────────────────────────────────


def build_row(
    name: str,
    args: argparse.Namespace,
    n_t: np.ndarray,
    order: list[tuple[int, int]],
    clip_ids: list[str],
) -> tuple[dict[tuple[int, int | None], np.ndarray], dict[str, Any]]:
    """The per-unit predictions of one row, plus its manifest fields."""
    n_channels = 1 + max(ch for _, ch in order)
    if name.startswith("blind_"):
        annot = Path(args.annot)
        units = blind_units(
            annot,
            gated=name == "blind_gated",
            clip_ids=clip_ids,
            frames=clip_frames(order, n_t, len(clip_ids)),
        )
        refusal = (
            f"a window that fails gate {BLIND_GATES} decodes to 0 rev/s"
            if name == "blind_gated"
            else "every window is kept"
        )
        return units, {
            "source": str(annot),
            "mode": "adopted",
            "unit_kind": "8-channel",
            "notes": (
                "one trajectory per parent recording, from all 8 microphones "
                "(channel-averaged seed, spatial joint Viterbi), replicated over the "
                "8 channel samples of every clip it covers. It is read at the dump's "
                "own frame times, thus nothing is resampled. "
                f"{refusal}, and a span no window covers decodes to 0 rev/s."
            ),
        }

    is_classical = name.startswith("classical_")
    stored = Path(CLASSICAL_RAW if is_classical else OTMP_RAW)
    if not args.recompute:
        pattern = f"{name.removeprefix('classical_')}__*.json" if is_classical else "*.json"
        kind = "per-regime sums of |err| and err**2" if is_classical else "per-OT-frame mae and mse"
        raise NotAdoptable(
            f"{stored} holds {kind} over {len(list(stored.glob(pattern)))} unit(s) "
            f"and no prediction; rerun the training-free estimator with --recompute"
        )
    raw = recompute(name, Path(args.cache), len(clip_ids), n_channels, args.jobs)
    units = read_pred_units(raw)
    return units, {
        "source": str(raw),
        "mode": "recomputed",
        "unit_kind": "per-channel",
        "notes": (
            f"{stored} holds no prediction, so the training-free estimator was run "
            "again. It is deterministic, thus the prediction is the campaign's."
        ),
    }


def write_row(dump: Path, name: str, pred: np.ndarray, n_t: np.ndarray, metric: np.ndarray) -> None:
    np.savez(dump / f"{name}.npz", pred=pred, n_t=n_t, metric=metric)


def load_manifest(dump: Path) -> dict[str, Any]:
    path = dump / "_adopted.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"dump": str(dump), "rows": {}}


def save_manifest(dump: Path, manifest: dict[str, Any]) -> None:
    manifest["dump"] = str(dump)
    manifest["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
    (dump / "_adopted.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dump", default=DEFAULT_DUMP, help="the dumped set to write into")
    ap.add_argument("--names", nargs="*", default=list(NAMES), help=f"rows: {', '.join(NAMES)}")
    ap.add_argument("--annot", default=BLIND_ANNOT, help="the blind tracker's annotation dir")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="where a --recompute keeps its units")
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="rerun the training-free estimator of a source that stored no prediction",
    )
    ap.add_argument("--jobs", type=int, default=8, help="processes for a --recompute")
    ap.add_argument(
        "--omp",
        default="1",
        help="BLAS threads per process (read before numpy; 1 keeps a --recompute pool honest)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite a row that already exists")
    ap.add_argument(
        "--recompute-only",
        action="store_true",
        help="run the estimator grid into --cache and stop, without a dump directory "
        "(for a remote CPU job; assemble the row later on a machine that has the dump)",
    )
    ap.add_argument("--n-clips", type=int, default=37, help="clips of the split (--recompute-only)")
    ap.add_argument("--n-channels", type=int, default=8, help="microphones (--recompute-only)")
    args = ap.parse_args()

    if args.recompute_only:
        for name in args.names:
            out = recompute(name, Path(args.cache), args.n_clips, args.n_channels, args.jobs)
            print(f"{name:24s} units in {out}", flush=True)
        return 0

    dump = Path(args.dump)
    gt, n_t, order, clip_ids = read_dump(dump)
    manifest = load_manifest(dump)
    unknown = [n for n in args.names if n not in NAMES]
    if unknown:
        raise SystemExit(f"unknown row(s): {', '.join(unknown)}; known: {', '.join(NAMES)}")

    for name in args.names:
        if (dump / f"{name}.npz").exists() and not args.force:
            print(f"{name:24s} exists, skipped (use --force)", flush=True)
            continue
        try:
            units, fields = build_row(name, args, n_t, order, clip_ids)
        except NotAdoptable as exc:
            manifest["rows"][name] = {
                "source": CLASSICAL_RAW if name.startswith("classical_") else OTMP_RAW,
                "mode": "not_adoptable",
                "reason": str(exc),
            }
            print(f"{name:24s} NOT ADOPTABLE: {exc}", flush=True)
            continue
        pred, lengths, metric, resampled = assemble(units, gt, n_t, order)
        write_row(dump, name, pred, lengths, metric)
        manifest["rows"][name] = {
            **fields,
            "frames_resampled": bool(resampled),
            "n_samples": int(pred.shape[0]),
            "n_units": len(units),
            "metric_mean": round(float(metric.mean()), 6),
        }
        print(
            f"{name:24s} {fields['mode']:12s} mae={metric.mean():8.3f} "
            f"median={np.median(metric):8.3f} resampled={resampled}",
            flush=True,
        )
    save_manifest(dump, manifest)
    print(f"wrote {dump / '_adopted.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
