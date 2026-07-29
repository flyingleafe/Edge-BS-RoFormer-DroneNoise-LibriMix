"""Neural-seeded coupled Vold-Kalman refinement: hybrid RPS-tracking pilot.

A neural RPS predictor (any checkpoint of the ``rps_predictor_vk_eval``
MODELS registry) provides the per-rotor seed trajectories; the existing
coupled VK tracker (``data_processing.vk_tracking.vk_track``) capture +
refine stages (configs imported from ``scripts/vk_blind_annotation.py``)
polish them. Scored on EXACTLY the protocol of
``scripts/rps_predictor_vk_eval.py`` — the 37 8-s clips of
``DREGON-LM-V4-michaels-valid-full``, per-clip PIT-aligned MAE (one
``tasks.rps_prediction.align_rps_to_gt`` Hungarian alignment per clip),
pooled as the mean of per-clip MAE over the dregon_cruise (n=18) /
dregon_gt30 (n=19) / fly124_cruise (n=9) regime groups — so results are
directly comparable to (a) the neural-only numbers of that script and
(b) the embedded telemetry-init VK reference per-clip MAEs
(``ref_mae_vk_rec``, the last CLIPS column).

Arms (per model x clip, seeded with the neural prediction ``r_init``):

* ``refine``    — single ``vk_track(audio, r_init, ft, REFINE_CFG)``.
* ``caprefine`` — ``vk_track(..., CAPTURE_CFG)`` (annealed grow schedule,
                  basin capture) then ``vk_track(capture.r_refined, ...,
                  REFINE_CFG)``.
* ``dp``        — the blind pipeline's spatial-DP ladder
                  (``vk_blind_annotation.vit2dsp_pipeline``: Viterbi
                  pair-mean c(t) -> spatial joint 2-rotor Viterbi with
                  per-rotor 1/d^2 mic mixes -> midband bw 6 -> refine,
                  ``stage_guard=True``) seeded with the neural trajectories
                  instead of the blind scan. In the blind pipeline the DP
                  stages — not ``vk_track`` — do the capture; ``vk_track``
                  is a polisher. Geometry weights need ``--dregon-dir``
                  (DREGON recordings) / the michaels loader (FLY124); the
                  track->physical-rotor map is PIT vs GT (experiment-level,
                  the validated ``run_vit2dsp`` methodology). NB vit2dsp
                  was designed for 16-25 s windows; 8 s clips are shorter
                  than its validated regime.

Run:
  PYTHONPATH=src python scripts/neural_seeded_vk.py            # full sweep
  ... --clips sample_00028,sample_00029 --models ckla_phaseonly_best \
      --arms refine                                            # smoke test
Remote (CPU is enough — vk_track is numpy/scipy):
  omnirun submit --backend apocrita-cpu --gpus 0 --time 4h -- \
    python scripts/neural_seeded_vk.py

Outputs (``results/neural_seeded_vk`` unless ``--out``): ``per_clip.csv``
(model, arm, clip, recording, regime, regime_mean_rps, mae_neural,
mae_refined, ref_mae_vk_rec, conf_mean, wall_s), pooled ``report.json``,
and a pooled-MAE table + VK wall-time/RTF summary on stdout.
"""

from __future__ import annotations

import os

# BLAS/FFT thread budget BEFORE numpy import: this script is serial (one
# vk_track at a time), so unlike the process-parallel vk_blind_* scripts a
# few threads help; the imported vk_blind_annotation setdefaults are no-ops.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
# Pin the repo's src/ ahead of site-packages (same rationale as
# scripts/rps_predictor_vk_eval.py: the editable install points at whatever
# checkout owns .venv, which on omnirun worktrees is NOT the job's checkout).
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_HERE))  # scripts/ — vk_eval + vk_blind_annotation
sys.path.insert(0, str(_ROOT / "src"))

import rps_predictor_vk_eval as vk_eval  # noqa: E402
from vk_blind_annotation import (  # noqa: E402
    CAPTURE_CFG,
    REFINE_CFG,
    pit_perm,
    vit2dsp_pipeline,
)
from vk_validation import Prepared, smooth_frames  # noqa: E402

from data_processing.vk_tracking import VKResult, vk_track  # noqa: E402

POOL_NAMES = ("dregon_cruise", "dregon_gt30", "fly124_cruise")
ARMS = ("refine", "caprefine", "dp")
CLIP_S = (vk_eval.CLIP_FRAMES - 1) * vk_eval.FRAME_S  # 8 s of audio per clip

_WEIGHTS_CACHE: dict[str, np.ndarray] = {}


def rotor_mic_weights(recording: str, dregon_dir: str) -> np.ndarray:
    """``(8 mics, 4 rotors)`` weights prop. to 1/d^2, normalized per rotor.

    Mirrors ``vk_blind_sweep.rotor_mic_weights``: DREGON geometry from
    ``data_processing.dregon.get_geometry`` (needs the DREGON data root),
    FLY124 from ``data_processing.michaels.get_geometry`` (no data needed).
    """
    key = "fly124" if recording == "michaels_FLY124" else "dregon"
    if key not in _WEIGHTS_CACHE:
        if key == "fly124":
            from data_processing.michaels import get_geometry

            mic, rot = get_geometry()
        else:
            from data_processing.dregon import get_geometry
            from data_processing.streams import resolve_source

            mic, rot = get_geometry(Path(resolve_source(dregon_dir)))
        d = np.linalg.norm(mic[:, None, :] - rot[None, :, :], axis=2)
        w = 1.0 / d**2
        _WEIGHTS_CACHE[key] = w / w.sum(axis=0, keepdims=True)
    return _WEIGHTS_CACHE[key]


def neural_predict(model, clip_audio: np.ndarray, mode: str, device: str, batch: int):
    """Per-clip neural seed ``(4, 251)`` — the vk_eval ``none``-arm forward path.

    ``ch0``: mic channel 0 only. ``chmean``: forward all C mics, permutation-
    align each mic's rotor rows to mic 0's (MSE Hungarian), average.
    """
    if mode == "ch0":
        wins = np.ascontiguousarray(clip_audio[:1])
        return vk_eval.batched_forward(model, wins, device, batch, vk_eval.CLIP_FRAMES)[0]
    per_ch = vk_eval.batched_forward(model, clip_audio, device, batch, vk_eval.CLIP_FRAMES)
    ref = per_ch[0]
    acc = ref.astype(np.float64).copy()
    for c in range(1, per_ch.shape[0]):
        acc += vk_eval.perm_align(per_ch[c].astype(np.float64), ref)
    return (acc / per_ch.shape[0]).astype(np.float32)


def run_arm(
    arm: str,
    audio64: np.ndarray,
    r_init: np.ndarray,
    ft: np.ndarray,
    gt: np.ndarray,
    info: tuple,
    dregon_dir: str,
) -> tuple[np.ndarray, VKResult, np.ndarray | None, float]:
    """Run one arm's stage ladder from the neural seed ``r_init``.

    Returns ``(final traj (4, F), final-stage VKResult, dp-stage traj | None,
    wall seconds)``. ``gt`` is used by the ``dp`` arm only, for the
    experiment-level track->physical-rotor map (PIT vs truth) and the
    ``Prepared`` truth fields — never to steer the tracking itself.
    """
    t0 = time.time()
    if arm == "refine":
        res = vk_track(audio64, r_init, ft, REFINE_CFG)
        out, dp = res.r_refined, None
    elif arm == "caprefine":
        cap = vk_track(audio64, r_init, ft, CAPTURE_CFG)
        res = vk_track(audio64, cap.r_refined, ft, REFINE_CFG)
        out, dp = res.r_refined, None
    elif arm == "dp":
        edge = (ft > 0.5) & (ft < ft[-1] - 0.5)
        prep = Prepared(
            rid=str(info[1]),
            tau=0.0,
            seg_lo=float(info[2]),
            seg_hi=float(info[2]) + CLIP_S,
            audio=audio64,
            ft=ft,
            r_init=r_init.copy(),
            r_meas=gt,
            r_meas_sm=smooth_frames(gt),
            edge=edge,
        )
        p = pit_perm(r_init, gt, edge)
        phys_map = np.empty(4, dtype=int)
        for truth_row, track_row in enumerate(list(p)):
            phys_map[track_row] = truth_row
        weights = rotor_mic_weights(str(info[1]), dregon_dir)
        stages, res, _extras, _ws, _wv = vit2dsp_pipeline(
            prep, r_init.copy(), weights, phys_map, stage_guard=True
        )
        sd = dict(stages)
        out, dp = sd["refine"], sd["vit2dsp"]
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return out, res, dp, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument(
        "--models",
        default="ckla_phaseonly_best",
        help=f"comma-separated keys of the vk_eval MODELS registry ({len(vk_eval.MODELS)} known)",
    )
    ap.add_argument("--data", default="dload:DREGON-LM-V4-michaels-valid-full")
    ap.add_argument("--mode", default="chmean", choices=["ch0", "chmean"])
    ap.add_argument("--arms", default="refine,caprefine", help=f"comma-separated of {ARMS}")
    ap.add_argument("--clips", default=None, help="comma-separated clip ids (smoke tests)")
    ap.add_argument("--out", default="results/neural_seeded_vk")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--dregon-dir",
        default="data/DREGON",
        help="DREGON data root for dp-arm mic/rotor geometry (path or dload:DREGON)",
    )
    args = ap.parse_args()

    models = args.models.split(",")
    unknown = [m for m in models if m not in vk_eval.MODELS]
    if unknown:
        ap.error(f"unknown models {unknown}; known: {sorted(vk_eval.MODELS)}")
    arms = args.arms.split(",")
    bad_arms = [a for a in arms if a not in ARMS]
    if bad_arms:
        ap.error(f"unknown arms {bad_arms}; known: {ARMS}")
    clip_ids = [c[0] for c in vk_eval.CLIPS]
    if args.clips:
        want = args.clips.split(",")
        missing = [c for c in want if c not in clip_ids]
        if missing:
            ap.error(f"unknown clips {missing}")
        clip_ids = [c for c in clip_ids if c in want]

    import torch

    from tasks.rps_prediction import align_rps_to_gt

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[neural_vk] src pin: {_ROOT / 'src'} | data: {args.data} | device: {device} | "
        f"mode: {args.mode} | clips: {len(clip_ids)}",
        flush=True,
    )

    audio, gt = vk_eval.load_clip_data(args.data)
    clip_info = {c[0]: c for c in vk_eval.CLIPS}
    ft = np.arange(vk_eval.CLIP_FRAMES) * vk_eval.FRAME_S

    rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    vk_wall_total = 0.0
    vk_audio_total = 0.0

    for model_name in models:
        experiment, ckpt_uri, _ = vk_eval.MODELS[model_name]
        print(f"\n=== {model_name} ({experiment}) ===", flush=True)
        model = vk_eval.load_model(experiment, ckpt_uri, device)

        # One neural forward per clip; both arms reuse the same seed.
        seeds: dict[str, np.ndarray] = {}
        mae_neural: dict[str, float] = {}
        for cid in clip_ids:
            r0 = neural_predict(model, audio[cid], args.mode, device, args.batch)
            seeds[cid] = r0
            g = gt[cid]
            mae_neural[cid] = float(np.mean(np.abs(align_rps_to_gt(r0, g) - g)))
        del model

        by_arm: dict[str, dict[str, float]] = {a: {} for a in arms}
        for arm in arms:
            for cid in clip_ids:
                info = clip_info[cid]
                audio64 = np.asarray(audio[cid], dtype=np.float64)
                err = ""
                mae_ref = float("nan")
                mae_dp = float("nan")
                conf_mean = float("nan")
                wall = float("nan")
                try:
                    g = np.asarray(gt[cid], dtype=np.float64)
                    out, res, dp, wall = run_arm(
                        arm, audio64, seeds[cid].astype(np.float64), ft, g, info, args.dregon_dir
                    )
                    mae_ref = float(np.mean(np.abs(align_rps_to_gt(out, g) - g)))
                    if dp is not None:
                        mae_dp = float(np.mean(np.abs(align_rps_to_gt(dp, g) - g)))
                    if res.confidence.size:
                        conf_mean = float(np.mean(res.confidence))
                    vk_wall_total += wall
                    vk_audio_total += CLIP_S
                except Exception as e:  # record and continue (per-clip isolation)
                    err = f"{type(e).__name__}: {e}"
                by_arm[arm][cid] = mae_ref
                rows.append(
                    {
                        "model": model_name,
                        "arm": arm,
                        "clip": cid,
                        "recording": info[1],
                        "regime": info[3],
                        "regime_mean_rps": info[4],
                        "mae_neural": mae_neural[cid],
                        "mae_refined": mae_ref,
                        "mae_dp_stage": mae_dp,
                        "ref_mae_vk_rec": info[7],
                        "conf_mean": conf_mean,
                        "wall_s": wall,
                        "error": err,
                    }
                )
                dp_txt = "" if np.isnan(mae_dp) else f"dp-stage {mae_dp:.3f} -> "
                print(
                    f"  [{arm}] {cid} ({info[3]}): neural {mae_neural[cid]:.3f} -> "
                    f"{dp_txt}refined {mae_ref:.3f} (vk_rec {info[7]:.3f}, "
                    f"conf {conf_mean:.3f}, {wall:.1f} s){' ERROR ' + err if err else ''}",
                    flush=True,
                )

        # Pooled numbers (identical pooling to vk_eval: mean per-clip MAE).
        pooled[model_name] = {}
        for arm in arms:
            pooled[model_name][arm] = {}
            for pool in POOL_NAMES:
                sel = [c for c in vk_eval.CLIPS if c[0] in by_arm[arm] and vk_eval.POOLS[pool](c)]
                if not sel:
                    continue
                pooled[model_name][arm][pool] = {
                    "neural": float(np.mean([mae_neural[c[0]] for c in sel])),
                    "refined": float(np.mean([by_arm[arm][c[0]] for c in sel])),
                    "vk_rec": float(np.mean([c[7] for c in sel])),
                    "n": float(len(sel)),
                }

    # ── report ──
    short = {"dregon_cruise": "drg_cru", "dregon_gt30": "drg_g30", "fly124_cruise": "fly_cru"}
    lines = [
        f"Pooled PIT-MAE (rev/s), mean of per-clip MAE — mode {args.mode}",
        "vk_rec = telemetry-init VK reference pooled from the same clips",
    ]
    hdr1 = f"{'':<24}{'':<11}" + "".join(
        f"{'---- ' + m + ' ----':>27}" for m in ("neural", "refined", "vk_rec")
    )
    hdr2 = f"{'model':<24}{'arm':<11}" + "".join(
        f"{short[p]:>9}" for _ in range(3) for p in POOL_NAMES
    )
    for model_name in models:
        lines += ["", hdr1, hdr2, "-" * len(hdr2)]
        for arm in arms:
            row = f"{model_name:<24}{arm:<11}"
            for metric in ("neural", "refined", "vk_rec"):
                for pool in POOL_NAMES:
                    cell = pooled.get(model_name, {}).get(arm, {}).get(pool, {})
                    row += f"{cell.get(metric, float('nan')):>9.3f}"
            lines.append(row)
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    rtf = vk_wall_total / vk_audio_total if vk_audio_total else float("nan")
    print(
        f"\n[neural_vk] VK stages total wall {vk_wall_total:.1f} s over "
        f"{vk_audio_total:.0f} s of audio (per arm-run) -> mean rtf {rtf:.2f}",
        flush=True,
    )

    report = {
        "protocol": "rps_predictor_vk_eval per-clip PIT-MAE (align_rps_to_gt once per clip)",
        "data": args.data,
        "mode": args.mode,
        "arms": arms,
        "clips": clip_ids,
        "capture_cfg": "vk_blind_annotation.CAPTURE_CFG",
        "refine_cfg": "vk_blind_annotation.REFINE_CFG",
        "pooled": pooled,
        "vk_wall_total_s": vk_wall_total,
        "vk_rtf": rtf,
        "reference_pooled": {
            "vk_telemetry_init": {"dregon_cruise": 0.7294, "fly124_cruise": 0.2825},
        },
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(table + "\n")
    print(f"[neural_vk] wrote {out_dir}/report.json, per_clip.csv, summary.txt", flush=True)


if __name__ == "__main__":
    main()
