#!/usr/bin/env python3
"""Beat-VK unified scorer — THE one metric for the campaign, on ``beatvk-valid-raw``.

Every candidate — neural RPS predictors, blind VK trackers, hybrids — is
scored by this script on the frozen dataset published by
``data_processing.derivations (beatvk_valid generator)``. This file is the metric half of the
protocol; numbers produced any other way are not comparable.

Protocol (frozen):

* Data: dload ``beatvk-valid-raw`` (default: the dload.lock pin) — 4
  recordings (3 DREGON free-flight room1 + FLY124), native-rate 8ch audio +
  RAW measured rotor telemetry + the per-recording 16 s window manifest
  (contiguous non-overlapping windows tiling the frozen eval span, each
  tagged ground/warmup/cruise). Windows and regime tags come FROM the
  manifest; this script never re-derives them.
* Ground truth: the raw telemetry (``rps_raw``) linearly interpolated onto
  the recording's fixed 0.032 s frame grid (grid points ``k * 0.032 s`` from
  recording start). No smoothing EVER touches the GT — the raw-telemetry
  jitter floor is part of the metric by design, so absolute numbers are
  lower-is-better comparable within this protocol only.
* Prediction: a per-recording trajectory on an arbitrary time grid
  (``ft`` seconds from recording start, ``rps`` (4, N) rev/s), linearly
  interpolated onto the same 0.032 s grid (edge-clamped outside ``ft``).
* Metric: per manifest window, ONE Hungarian rotor assignment
  (``tasks.rps_prediction.align_rps_to_gt``, MSE cost) between the gridded
  prediction and gridded GT restricted to the window; window MAE = mean
  absolute error over the aligned (rotor, frame) cells.
* Pooled outputs: ``dregon_cruise`` (mean window MAE over the cruise windows
  of the 3 DREGON recordings), ``fly124_cruise``, plus per-regime and
  per-recording tables. Pool means are unweighted over windows.
* Optional prediction-smoothing arms (``--arms none,med5``): ``med<sec>`` =
  running median over the gridded PREDICTION only (never the GT), applied to
  the full-recording trajectory before windowing. Default: ``none``.

Prediction sources (``--pred``):

* ``npz:<path-or-dir>`` — a directory of ``<recording_id>.npz`` files, each
  with ``ft`` (N,) and ``rps`` (4, N); or a single ``.npz`` with per-recording
  keys ``ft__<recording_id>`` / ``rps__<recording_id>``. Recordings absent
  from the source are skipped (with a warning) and their pools reported over
  the recordings present.
* ``model:<key>`` — a checkpoint from ``rps_predictor_vk_eval.MODELS``, run
  with that script's stitched chmean inference (all 8 mics, per-window
  rotor-alignment to mic 0, sliding 251-frame windows at 32-frame hop,
  overlap-aligned stitch, per-frame mean): the audio is soxr-resampled to the
  model's 16 kHz here in the scorer (the ONLY resample in the protocol — the
  dataset stays native), and the stitched trajectory lands directly on the
  0.032 s grid.

Run::

    python scripts/beatvk_eval.py --pred model:ckla_phaseonly_best --tag ckla_phaseonly
    python scripts/beatvk_eval.py --pred npz:results/vk_blind/traj --arms none,med5 --tag vk

Outputs: pooled table on stdout + ``results/beatvk_eval/<tag>/report.json``
and ``per_window.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
# Pin the repo's src/ ahead of site-packages (same rationale as
# scripts/rps_predictor_vk_eval.py), plus scripts/ itself so the model
# registry + stitch helpers of that script are importable.
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from tracking.protocols import (  # noqa: E402  (after the sys.path pin above)
    BEATVK,
    BEATVK_DREGON_RECS,
    BEATVK_FLY124_REC,
)

# Protocol constants come from the declarative spec (tracking.protocols.BEATVK);
# the module-level names stay — half the campaign scripts import them from here.
DATASET = BEATVK.dataset
SR = BEATVK.sr
HOP = BEATVK.hop_samples
FRAME_S = HOP / SR  # 0.032 s — the fixed evaluation grid
N_ROTORS = BEATVK.n_rotors
# Stitched-inference parameters for model: predictions (rps_predictor_vk_eval
# conventions: 251-frame = 8 s windows, 32-frame = 1.024 s hop).
STITCH_WIN_FRAMES = 251
STITCH_SLIDE_FRAMES = 32

DREGON_RECS = set(BEATVK_DREGON_RECS)
FLY124_REC = BEATVK_FLY124_REC


def load_recordings(
    version: str | None, wanted: set[str] | None, *, keep_audio: bool
) -> list[dict[str, Any]]:
    """Stream the frozen dataset; one dict per recording (manifest + GT)."""
    from data_processing import streams
    from data_processing.frames import meta_dict

    recs: list[dict[str, Any]] = []
    dataset_version = ""
    for frame in streams.iter_published_frames(DATASET, version):
        meta = meta_dict(frame)
        rid = str(meta["recording_id"])
        if wanted is not None and rid not in wanted:
            continue
        rps = frame["rps_raw"]
        recs.append(
            {
                "recording_id": rid,
                "ts": np.asarray(rps.tindex.abs_stamps, dtype=np.float64),
                "vals": np.asarray(rps.data, dtype=np.float64),
                "windows": meta["windows"],
                "audio": frame["audio"] if keep_audio else None,
            }
        )
        del frame
    if not recs:
        raise RuntimeError(f"no recordings loaded from {DATASET} (wanted={wanted})")
    # dataset version for the report: re-resolve cheaply via the manifest
    repo = streams.open_repository()
    dataset_version = repo.dataset(DATASET, version).version
    for r in recs:
        r["dataset_version"] = dataset_version
    return recs


# ─── Prediction sources ────────────────────────────────────────────────────────


def preds_from_npz(path: Path, rec_ids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``npz:`` source — see module docstring for the two accepted layouts."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if path.is_dir():
        for rid in rec_ids:
            f = path / f"{rid}.npz"
            if not f.is_file():
                continue
            with np.load(f) as z:
                out[rid] = (np.asarray(z["ft"], np.float64), np.asarray(z["rps"], np.float64))
    elif path.is_file():
        with np.load(path) as z:
            for rid in rec_ids:
                fk, rk = f"ft__{rid}", f"rps__{rid}"
                if fk in z and rk in z:
                    out[rid] = (np.asarray(z[fk], np.float64), np.asarray(z[rk], np.float64))
    else:
        raise FileNotFoundError(f"--pred npz path {path} does not exist")
    if not out:
        raise RuntimeError(
            f"no predictions found under {path} (expected <recording_id>.npz files "
            f"or ft__<id>/rps__<id> keys for {rec_ids})"
        )
    for rid, (ft, rps) in out.items():
        if ft.ndim != 1 or rps.shape != (N_ROTORS, len(ft)):
            raise ValueError(f"{rid}: expected ft (N,) + rps (4, N), got {ft.shape}/{rps.shape}")
    return out


def preds_from_model(
    model_key: str, recs: list[dict[str, Any]], device: str, batch: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``model:`` source — rps_predictor_vk_eval's stitched chmean inference."""
    import rps_predictor_vk_eval as vkev

    from data_processing.frames import resample_audio_series

    if model_key not in vkev.MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {sorted(vkev.MODELS)}")
    experiment, ckpt_uri, _ = vkev.MODELS[model_key]
    model = vkev.load_model(experiment, ckpt_uri, device)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rec in recs:
        t0 = time.time()
        # The protocol's ONLY resample: native audio -> the model's 16 kHz
        # (librosa soxr_hq, deterministic), all channels kept for chmean.
        audio16 = resample_audio_series(rec["audio"], SR)
        rec_audio = np.atleast_2d(np.asarray(audio16.data, dtype=np.float32))
        f_total = rec_audio.shape[-1] // HOP + 1
        starts = vkev.window_starts(f_total, STITCH_WIN_FRAMES, STITCH_SLIDE_FRAMES)
        preds = vkev.predict_windows(
            model, rec_audio, starts, "chmean", device, batch, STITCH_WIN_FRAMES
        )
        stack = vkev.stitch_stack(preds, starts, f_total, STITCH_WIN_FRAMES)
        traj = np.nanmean(stack, axis=0)  # (4, f_total), grid k * FRAME_S
        ft = np.arange(f_total, dtype=np.float64) * FRAME_S
        out[rec["recording_id"]] = (ft, traj.astype(np.float64))
        print(
            f"  [model] {rec['recording_id']}: {len(starts)} windows x "
            f"{rec_audio.shape[0]} mics in {time.time() - t0:.1f} s",
            flush=True,
        )
    return out


# ─── Scoring ───────────────────────────────────────────────────────────────────


def parse_arms(spec: str) -> list[str]:
    arms = [a.strip() for a in spec.split(",") if a.strip()]
    for arm in arms:
        if arm != "none" and not (arm.startswith("med") and _arm_span(arm) > 0):
            raise ValueError(f"unknown arm {arm!r} (expected 'none' or 'med<seconds>')")
    return arms


def _arm_span(arm: str) -> float:
    try:
        return float(arm[3:])
    except ValueError:
        return -1.0


def arm_trajectory(gridded: np.ndarray, arm: str) -> np.ndarray:
    """Apply a declared smoothing arm to the gridded PREDICTION (never GT)."""
    if arm == "none":
        return gridded
    import rps_predictor_vk_eval as vkev

    n = int(round(_arm_span(arm) / FRAME_S))
    return vkev.running_median(gridded, n + 1 if n % 2 == 0 else n)


def score_recording(
    rec: dict[str, Any], ft: np.ndarray, rps: np.ndarray, arms: list[str]
) -> list[dict[str, Any]]:
    """Per-window rows for one recording: one Hungarian + MAE per (window, arm)."""
    from tasks.rps_prediction import align_rps_to_gt

    ts, vals, windows = rec["ts"], rec["vals"], rec["windows"]
    n_frames = int(np.ceil(max(float(w["end_s"]) for w in windows) / FRAME_S)) + 1
    tg = np.arange(n_frames, dtype=np.float64) * FRAME_S
    pred = np.vstack([np.interp(tg, ft, rps[r]) for r in range(N_ROTORS)])
    gt = np.vstack([np.interp(tg, ts, vals[r]) for r in range(N_ROTORS)])

    rows: list[dict[str, Any]] = []
    for arm in arms:
        pred_arm = arm_trajectory(pred, arm)
        for w in windows:
            mask = (tg >= float(w["start_s"]) - 1e-6) & (tg < float(w["end_s"]) - 1e-6)
            aligned = align_rps_to_gt(pred_arm[:, mask], gt[:, mask])
            err = aligned - gt[:, mask]
            rows.append(
                {
                    "recording": rec["recording_id"],
                    "window": int(w["index"]),
                    "start_s": float(w["start_s"]),
                    "end_s": float(w["end_s"]),
                    "regime": str(w["regime"]),
                    "mean_rps": float(w["mean_rps"]),
                    "arm": arm,
                    "mae": float(np.mean(np.abs(err))),
                    "mse": float(np.mean(err**2)),
                }
            )
    return rows


def pool_rows(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    """Pooled tables: headline pools + per-regime + per-recording (per arm)."""

    def mean_mae(sel: list[dict[str, Any]]) -> float | None:
        return float(np.mean([r["mae"] for r in sel])) if sel else None

    pooled: dict[str, Any] = {}
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        regimes = sorted({r["regime"] for r in sub})
        recs = sorted({r["recording"] for r in sub})
        pooled[arm] = {
            "dregon_cruise": mean_mae(
                [r for r in sub if r["recording"] in DREGON_RECS and r["regime"] == "cruise"]
            ),
            "fly124_cruise": mean_mae(
                [r for r in sub if r["recording"] == FLY124_REC and r["regime"] == "cruise"]
            ),
            "per_regime": {
                reg: mean_mae([r for r in sub if r["regime"] == reg]) for reg in regimes
            },
            "per_recording": {
                rid: {
                    "all": mean_mae([r for r in sub if r["recording"] == rid]),
                    **{
                        reg: m
                        for reg in regimes
                        if (
                            m := mean_mae(
                                [r for r in sub if r["recording"] == rid and r["regime"] == reg]
                            )
                        )
                        is not None
                    },
                }
                for rid in recs
            },
        }
    return pooled


def format_table(pooled: dict[str, Any], arms: list[str]) -> str:
    lines = ["Beat-VK protocol — pooled window PIT-MAE (rev/s) vs RAW telemetry"]
    header = f"{'arm':<8}{'dregon_cruise':>15}{'fly124_cruise':>15}"
    regimes = sorted({reg for arm in arms for reg in pooled[arm]["per_regime"]})
    header += "".join(f"{'all_' + reg:>13}" for reg in regimes)
    lines += [header, "-" * len(header)]

    def cell(v: float | None, width: int) -> str:
        return f"{v:>{width}.3f}" if v is not None else f"{'—':>{width}}"

    for arm in arms:
        p = pooled[arm]
        row = f"{arm:<8}" + cell(p["dregon_cruise"], 15) + cell(p["fly124_cruise"], 15)
        row += "".join(cell(p["per_regime"].get(reg), 13) for reg in regimes)
        lines.append(row)
    lines.append("")
    lines.append(f"{'recording':<36}" + "".join(f"{arm:>12}" for arm in arms))
    recs = sorted(pooled[arms[0]]["per_recording"])
    for rid in recs:
        row = f"{rid:<36}"
        for arm in arms:
            row += cell(pooled[arm]["per_recording"].get(rid, {}).get("all"), 12)
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument(
        "--pred",
        required=True,
        help="prediction source: 'npz:<path-or-dir>' or 'model:<rps_predictor_vk_eval key>'",
    )
    ap.add_argument("--arms", default="none", help="comma list of 'none' / 'med<seconds>' arms")
    ap.add_argument("--tag", default=None, help="run name (default: derived from --pred)")
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--recordings", nargs="+", default=None, help="restrict to these recordings")
    ap.add_argument("--device", default=None, help="cuda|cpu for model: preds (default: auto)")
    ap.add_argument("--batch", type=int, default=16, help="inference batch for model: preds")
    ap.add_argument("--out", default="results/beatvk_eval", help="output root directory")
    args = ap.parse_args()

    kind, _, spec = args.pred.partition(":")
    if kind not in ("npz", "model") or not spec:
        raise SystemExit(f"--pred must be 'npz:<path>' or 'model:<key>', got {args.pred!r}")
    arms = parse_arms(args.arms)
    tag = args.tag or (spec if kind == "model" else Path(spec).stem)
    wanted = set(args.recordings) if args.recordings else None

    recs = load_recordings(args.dataset_version, wanted, keep_audio=kind == "model")
    rec_ids = [r["recording_id"] for r in recs]
    print(f"[beatvk_eval] {DATASET}@{recs[0]['dataset_version'][:12]}: {rec_ids}", flush=True)

    model_info: dict[str, Any] | None = None
    if kind == "model":
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        preds = preds_from_model(spec, recs, device, args.batch)
        model_info = {
            "key": spec,
            "device": device,
            "input_mode": "chmean",
            "stitch_win_frames": STITCH_WIN_FRAMES,
            "stitch_slide_frames": STITCH_SLIDE_FRAMES,
        }
    else:
        preds = preds_from_npz(Path(spec), rec_ids)
        missing = [rid for rid in rec_ids if rid not in preds]
        if missing:
            print(f"[beatvk_eval] WARNING: no predictions for {missing} — skipped", flush=True)

    rows: list[dict[str, Any]] = []
    for rec in recs:
        if rec["recording_id"] not in preds:
            continue
        ft, rps = preds[rec["recording_id"]]
        rows.extend(score_recording(rec, ft, rps, arms))
    if not rows:
        raise SystemExit("nothing scored (no windows / no matching predictions)")

    pooled = pool_rows(rows, arms)
    table = format_table(pooled, arms)
    print("\n" + table, flush=True)

    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": (
            "beatvk-valid-raw manifest windows (16 s); pred + RAW telemetry linearly "
            "interpolated to the 0.032 s recording grid; one Hungarian (align_rps_to_gt) "
            "per window; window MAE; pools = unweighted window means"
        ),
        "dataset": {"name": DATASET, "version": recs[0]["dataset_version"]},
        "pred": args.pred,
        "model": model_info,
        "arms": arms,
        "n_windows_scored": len({(r["recording"], r["window"]) for r in rows}),
        "recordings_scored": sorted({r["recording"] for r in rows}),
        "pooled": pooled,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "per_window.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[beatvk_eval] wrote {out_dir}/report.json, per_window.csv", flush=True)


if __name__ == "__main__":
    main()
