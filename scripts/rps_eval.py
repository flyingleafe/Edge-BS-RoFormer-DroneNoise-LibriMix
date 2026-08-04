#!/usr/bin/env python3
"""Generic RPS protocol evaluator: prediction source x refinement x protocol.

One CLI replaces the deleted per-question one-offs (rps_refiner_eval,
neural_seeded_vk, pi_kalman_protocol): pick a frozen protocol, a prediction
source, and optionally a refinement stage; get the protocol's pooled tables.

* ``--protocol`` — a :mod:`tracking.protocols` spec:
  ``beatvk`` (the frozen 16 s-window campaign protocol on ``beatvk-valid-raw``)
  or ``vk37`` (the 5-recording DREGON validation protocol, one 25 s
  mid-flight segment each).
* ``--pred`` — where the candidate trajectories come from:
  ``model:<key>`` (a checkpoint from ``rps_predictor_vk_eval.MODELS``, run
  with the frozen stitched-chmean inference), ``npz:<path-or-dir>``
  (``beatvk_eval`` NPZ layouts), or ``telem`` (the recorded telemetry init —
  raw measured for beatvk, the cleaned COMMAND labels for vk37 — the natural
  init for testing refinement stages).
* ``--refine`` — a :mod:`tracking.stages` adapter applied per window on top
  of the prediction: ``none`` | ``pi_kalman`` | ``vk`` (the validated
  ``vk_validation.MAIN_CFG`` refine-mode tracker) | ``warp``.
* ``--pools`` — restrict to the protocol's named pools
  (e.g. ``dregon_cruise,fly124_cruise``); default: every window.

Refinement runs as restartable per-window units on the
:mod:`utils.gridrun` harness (``--jobs``, ``--no-resume``); scoring reuses
``scripts/beatvk_eval.py`` (``score_recording`` / ``pool_rows`` /
``format_table``) — the one metric of the campaign, never duplicated.

Run::

    python scripts/rps_eval.py --protocol beatvk --pred model:ckla_phaseonly_best
    python scripts/rps_eval.py --protocol beatvk --pred telem --refine pi_kalman --jobs 8
    python scripts/rps_eval.py --protocol vk37 --pred telem --refine vk --jobs 5

Outputs: pooled table on stdout + ``<out>/<tag>/report.json`` and
``per_window.csv`` (+ ``raw/*.json`` refinement units when ``--refine``).
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (process-level parallelism instead) —
# the shared harness convention (utils.gridrun re-asserts it).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

import beatvk_eval as bve  # noqa: E402
from beatvk_eval import (  # noqa: E402
    FRAME_S,
    HOP,
    N_ROTORS,
    STITCH_SLIDE_FRAMES,
    STITCH_WIN_FRAMES,
    format_table,
    pool_rows,
    score_recording,
)

from tracking.protocols import (  # noqa: E402
    ProtocolSpec,
    WindowSpec,
    get_protocol,
    iter_windows,
    to_frame,
)
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args, unit_path  # noqa: E402

REFINERS = ("none", "pi_kalman", "vk", "warp")
DEFAULT_OUT = Path("results/rps_eval")


# ---------------------------------------------------------------------------
# window preps (audio + GT on the window frame grid, cached NPZ per window)


def vk37_prep_dir(out_root: Path) -> Path:
    return out_root / "vk37_prep"


def build_vk37_preps(out_root: Path, rids: list[str]) -> None:
    """Materialize the vk37 25 s segments (audio + command + measured) once.

    Serial on purpose: ``vk_validation.prepare_recording`` streams DREGON and
    estimates the clock offset — heavy, but one NPZ per recording, forever.
    """
    import vk_validation as vkv

    pdir = vk37_prep_dir(out_root)
    pdir.mkdir(parents=True, exist_ok=True)
    for rid in rids:
        p = pdir / f"{rid}.npz"
        if p.exists():
            continue
        tic = time.perf_counter()
        prep = vkv.prepare_recording(rid, dregon_dir="dload:DREGON")
        np.savez(
            p,
            audio=prep.audio.astype(np.float32),
            ft=prep.ft,
            r_init=prep.r_init,
            r_meas=prep.r_meas,
            r_meas_sm=prep.r_meas_sm,
            edge=prep.edge,
            seg_lo=np.float64(prep.seg_lo),
            seg_hi=np.float64(prep.seg_hi),
            tau=np.float64(prep.tau),
        )
        print(f"[prep] {rid}: segment cached ({time.perf_counter() - tic:.0f}s)", flush=True)


def load_vk37_prep(out_root: Path, rid: str) -> dict[str, Any]:
    with np.load(vk37_prep_dir(out_root) / f"{rid}.npz") as z:
        return {k: np.asarray(z[k]) for k in z.files}


def vk37_recs(out_root: Path, rids: list[str], edge_trim_s: float) -> list[dict[str, Any]]:
    """Score-ready rec dicts (the ``beatvk_eval.score_recording`` shape).

    GT = the measured telemetry on the segment frame grid; one window per
    recording covering the segment minus the edge trim (the
    ``vk_validation`` metric convention).
    """
    recs: list[dict[str, Any]] = []
    for rid in rids:
        prep = load_vk37_prep(out_root, rid)
        seg_lo, seg_hi = float(prep["seg_lo"]), float(prep["seg_hi"])
        recs.append(
            {
                "recording_id": rid,
                "dataset_version": "vk37-live-prep",
                "ts": seg_lo + prep["ft"],
                "vals": prep["r_meas"],
                "windows": [
                    {
                        "index": 0,
                        "start_s": seg_lo + edge_trim_s,
                        "end_s": seg_hi - edge_trim_s,
                        "regime": "cruise",
                        "mean_rps": float(np.mean(prep["r_meas"])),
                    }
                ],
            }
        )
    return recs


# ---------------------------------------------------------------------------
# prediction sources: {rid: (ft_abs, (4, N) rps)}

Preds = dict[str, tuple[np.ndarray, np.ndarray]]


def vk37_model_preds(
    model_key: str, out_root: Path, rids: list[str], device: str, batch: int
) -> Preds:
    """Stitched-chmean inference on the cached 16 kHz vk37 segments."""
    import rps_predictor_vk_eval as vkev

    if model_key not in vkev.MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {sorted(vkev.MODELS)}")
    experiment, ckpt_uri, _ = vkev.MODELS[model_key]
    model = vkev.load_model(experiment, ckpt_uri, device)
    out: Preds = {}
    for rid in rids:
        prep = load_vk37_prep(out_root, rid)
        audio32 = np.ascontiguousarray(prep["audio"], dtype=np.float32)
        f_total = audio32.shape[-1] // HOP + 1
        starts = vkev.window_starts(f_total, STITCH_WIN_FRAMES, STITCH_SLIDE_FRAMES)
        preds = vkev.predict_windows(
            model, audio32, starts, "chmean", device, batch, STITCH_WIN_FRAMES
        )
        stack = vkev.stitch_stack(preds, starts, f_total, STITCH_WIN_FRAMES)
        traj = np.nanmean(stack, axis=0)
        ft = float(prep["seg_lo"]) + np.arange(f_total, dtype=np.float64) * FRAME_S
        out[rid] = (ft, traj.astype(np.float64))
        print(f"  [model] {rid}: {len(starts)} stitched windows", flush=True)
    return out


def base_predictions(
    args: argparse.Namespace,
    protocol: ProtocolSpec,
    recs: list[dict[str, Any]],
    out_root: Path,
) -> tuple[Preds, dict[str, Any] | None]:
    """Resolve ``--pred`` into full-span trajectories per recording."""
    kind, _, spec = args.pred.partition(":")
    rec_ids = [r["recording_id"] for r in recs]
    model_info: dict[str, Any] | None = None

    if args.pred == "telem":
        if protocol.name == "vk37":
            preds: Preds = {}
            for rid in rec_ids:
                prep = load_vk37_prep(out_root, rid)
                preds[rid] = (float(prep["seg_lo"]) + prep["ft"], prep["r_init"])
        else:
            # beatvk: the raw measured telemetry itself (only meaningful as a
            # refinement init / jitter-floor probe — it IS the ground truth).
            preds = {r["recording_id"]: (r["ts"], r["vals"]) for r in recs}
        return preds, None

    if kind == "npz" and spec:
        return bve.preds_from_npz(Path(spec), rec_ids), None

    if kind == "model" and spec:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if protocol.name == "vk37":
            preds = vk37_model_preds(spec, out_root, rec_ids, device, args.batch)
        else:
            preds = bve.preds_from_model(spec, recs, device, args.batch)
        model_info = {
            "key": spec,
            "device": device,
            "input_mode": "chmean",
            "stitch_win_frames": STITCH_WIN_FRAMES,
            "stitch_slide_frames": STITCH_SLIDE_FRAMES,
        }
        return preds, model_info

    raise SystemExit(f"--pred must be 'model:<key>', 'npz:<path>' or 'telem', got {args.pred!r}")


# ---------------------------------------------------------------------------
# refinement units (utils.gridrun)


def _refine_stage(refine: str, sr: int):
    """The tracking.stages adapter for one ``--refine`` arm."""
    import tracking as trk

    if refine == "pi_kalman":
        return trk.pi_kalman_stage()
    if refine == "warp":
        return trk.warp_stage()
    if refine == "vk":
        from vk_validation import MAIN_CFG

        if abs(MAIN_CFG.fs - sr) > 1e-6:
            raise ValueError(f"vk refine config fs={MAIN_CFG.fs} != protocol sr {sr}")
        return trk.vk_stage(MAIN_CFG)
    raise KeyError(f"unknown refiner {refine!r}; known: {REFINERS}")


def refine_unit(unit: Unit) -> dict[str, Any]:
    """One (recording, window): interp the base pred onto the window grid,
    run the refinement stage on the window audio, return the refined slice."""
    p = dict(unit.params)
    spec = WindowSpec(**p["spec"])
    tic = time.perf_counter()

    if spec.protocol == "vk37":
        prep = load_vk37_prep(Path(p["out_root"]), spec.recording_id)
        audio, ft, start = prep["audio"], prep["ft"], float(prep["seg_lo"])
        rps_meas = prep["r_meas"]
    else:
        import beatvk_vk_arms as bva

        prepd, _regime = bva.load_prep(Path(p["prep_root"]), spec.recording_id, spec.index, 8)
        audio, ft, start = prepd.audio, prepd.ft, float(prepd.seg_lo)
        rps_meas = prepd.r_meas

    with np.load(p["preds_npz"]) as z:
        pft, prps = z[f"ft__{spec.recording_id}"], z[f"rps__{spec.recording_id}"]
    r0 = np.stack([np.interp(start + ft, pft, prps[i]) for i in range(N_ROTORS)])

    frame = to_frame(audio, p["sr"], spec, rps=r0, frame_times=ft, rps_meas=rps_meas)
    out = _refine_stage(p["refine"], p["sr"])(frame)
    import tracking as trk

    r_new, _times = trk.get_rps(out)
    log = list(out["meta"]["tracking"])
    return {
        "recording": spec.recording_id,
        "window": spec.index,
        "start_s": start,
        "refine": p["refine"],
        "wall_s": round(time.perf_counter() - tic, 1),
        "ft": [float(v) for v in ft],
        "rps": [[float(v) for v in row] for row in r_new],
        "stage_log": json.loads(json.dumps(log, default=str)),
    }


def refined_predictions(out_dir: Path, specs: list[WindowSpec]) -> Preds:
    """Assemble per-recording trajectories from the completed unit JSONs."""
    by_rid: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for spec in specs:
        up = unit_path(out_dir, f"{spec.recording_id}__w{spec.index:02d}")
        if not up.exists():
            raise RuntimeError(f"missing refinement unit {up} — did some units fail?")
        row = json.loads(up.read_text())
        ft = float(row["start_s"]) + np.asarray(row["ft"], dtype=np.float64)
        rps = np.asarray(row["rps"], dtype=np.float64)
        by_rid.setdefault(spec.recording_id, []).append((spec.index, ft, rps))
    out: Preds = {}
    for rid, parts in by_rid.items():
        parts.sort(key=lambda t: t[0])
        ft_all = np.concatenate([ft for _, ft, _ in parts])
        rps_all = np.concatenate([r for _, _, r in parts], axis=1)
        if not np.all(np.diff(ft_all) > 0):
            raise RuntimeError(f"{rid}: non-monotonic assembled ft")
        out[rid] = (ft_all, rps_all)
    return out


# ---------------------------------------------------------------------------


def select_specs(
    protocol: ProtocolSpec,
    specs: list[WindowSpec],
    pools: list[str] | None,
) -> list[WindowSpec]:
    if not pools:
        return specs
    unknown = [p for p in pools if p not in protocol.pools]
    if unknown:
        raise SystemExit(f"unknown pools {unknown}; known: {sorted(protocol.pools)}")
    return [s for s in specs if any(protocol.pools[p].contains(s) for p in pools)]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--protocol", required=True, choices=("beatvk", "vk37"))
    ap.add_argument(
        "--pred",
        required=True,
        help="'model:<rps_predictor_vk_eval key>', 'npz:<path-or-dir>' or 'telem'",
    )
    ap.add_argument("--refine", default="none", choices=REFINERS)
    ap.add_argument("--pools", default=None, help="comma list of protocol pool names")
    ap.add_argument("--recordings", nargs="+", default=None, help="restrict to these recordings")
    ap.add_argument("--tag", default=None, help="run name (default: protocol__pred__refine)")
    ap.add_argument("--dataset-version", default=None, help="beatvk dataset version override")
    ap.add_argument("--device", default=None, help="cuda|cpu for model: preds (default: auto)")
    ap.add_argument("--batch", type=int, default=16, help="inference batch for model: preds")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root directory")
    add_gridrun_args(ap)
    args = ap.parse_args()

    protocol = get_protocol(args.protocol)
    pools = [p for p in (args.pools or "").split(",") if p] or None
    wanted = set(args.recordings) if args.recordings else None
    kind, _, spec_str = args.pred.partition(":")
    pred_name = spec_str if kind in ("model", "npz") and spec_str else args.pred
    pred_name = Path(pred_name).stem if kind == "npz" else pred_name
    tag = args.tag or f"{protocol.name}__{pred_name}__{args.refine}"
    out_root = Path(args.out)
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Windows + ground truth (loaders live HERE, specs in tracking.protocols).
    if protocol.name == "vk37":
        rids = [r for r in protocol.recordings if wanted is None or r in wanted]
        build_vk37_preps(out_root, rids)
        recs = vk37_recs(out_root, rids, protocol.edge_trim_s)
        specs = list(iter_windows(protocol, recordings=wanted))
        prep_root = None
    else:
        import beatvk_vk_arms as bva

        prep_root = Path(bva.DEFAULT_OUT)
        manifest = bva.load_manifest(prep_root, wanted, args.dataset_version)
        specs = list(iter_windows(protocol, manifest["recordings"], recordings=wanted))
        recs = bve.load_recordings(args.dataset_version, wanted, keep_audio=kind == "model")
    specs = select_specs(protocol, specs, pools)
    if not specs:
        raise SystemExit("no windows selected (check --pools / --recordings)")
    print(f"[rps_eval] {protocol.name}: {len(specs)} windows, pred={args.pred}", flush=True)

    # 2. Base predictions.
    preds, model_info = base_predictions(args, protocol, recs, out_root)

    # 3. Optional per-window refinement (restartable gridrun units).
    if args.refine != "none":
        if protocol.name == "beatvk":
            assert prep_root is not None
            jobs_windows: dict[str, list[int]] = {}
            for s in specs:
                jobs_windows.setdefault(s.recording_id, []).append(s.index)
            import beatvk_vk_arms as bva

            bva.build_preps(prep_root, jobs_windows, args.dataset_version, "dload:DREGON")
        preds_npz = out_dir / "preds.npz"
        payload: dict[str, Any] = {}
        for rid, (ft, rps) in preds.items():
            payload[f"ft__{rid}"] = ft
            payload[f"rps__{rid}"] = rps
        np.savez(preds_npz, **payload)
        units = [
            Unit(
                uid=f"{s.recording_id}__w{s.index:02d}",
                params={
                    "spec": {
                        "protocol": s.protocol,
                        "recording_id": s.recording_id,
                        "index": s.index,
                        "start_s": s.start_s,
                        "end_s": s.end_s,
                        "regime": s.regime,
                        "mean_rps": s.mean_rps,
                    },
                    "refine": args.refine,
                    "sr": protocol.sr,
                    "out_root": str(out_root),
                    "prep_root": str(prep_root) if prep_root is not None else None,
                    "preds_npz": str(preds_npz),
                },
            )
            for s in specs
            if s.recording_id in preds
        ]
        result = gridrun_from_args(
            args,
            units,
            refine_unit,
            out_dir,
            summarize=lambda rows: {
                "n_windows": len(rows),
                "refine": args.refine,
                "wall_s_total": round(sum(r.get("wall_s", 0.0) for r in rows), 1),
            },
        )
        if result.n_failed:
            return result.exit_code
        preds = refined_predictions(out_dir, [s for s in specs if s.recording_id in preds])

    # 4. Score — beatvk_eval's frozen metric, restricted to the selected windows.
    sel: dict[str, set[int]] = {}
    for s in specs:
        sel.setdefault(s.recording_id, set()).add(s.index)
    rows: list[dict[str, Any]] = []
    for rec in recs:
        rid = rec["recording_id"]
        if rid not in preds or rid not in sel:
            continue
        sub = dict(rec)
        sub["windows"] = [w for w in rec["windows"] if int(w["index"]) in sel[rid]]
        if not sub["windows"]:
            continue
        ft, rps = preds[rid]
        rows.extend(score_recording(sub, ft, rps, ["none"]))
    if not rows:
        raise SystemExit("nothing scored (no windows / no matching predictions)")

    pooled = pool_rows(rows, ["none"])
    print("\n" + format_table(pooled, ["none"]), flush=True)

    report = {
        "protocol": protocol.name,
        "pred": args.pred,
        "refine": args.refine,
        "pools": pools,
        "model": model_info,
        "dataset": {"name": protocol.dataset, "version": recs[0].get("dataset_version")},
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
    print(f"\n[rps_eval] wrote {out_dir}/report.json, per_window.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
