#!/usr/bin/env python3
"""Re-anchored-neural arms on the frozen beat-VK protocol (``beatvk-valid-raw``).

Hypothesis: the CKLA phase-only neural predictors track per-rotor
FLUCTUATIONS well (bias-removed MAE ~1.0-1.2 rev/s) but place 1-2 rotors at
wrong absolute levels (anchor collapse), while the blind comb-scan seed
bases are 0.2-0.4-accurate constants. Combine, per manifest 16 s window::

    r_i(t) = base_i + (neural_{j(i)}(t) - mean_t neural_{j(i)})

with neural rows assigned to the sorted seed bases by RANK ORDER (both
sorted by their window mean, rank-to-rank). Windows where blind_fullrange's
coarse pass engaged (``coarse_mode == "coarse"``: the DREGON takeoff ramps,
the FLY124 warmup ramps and its w2 maneuver window) keep the
blind_fullrange FINAL trajectory unchanged — a constant-base re-anchor is
undefined on a ramp (adding the coarse shape to the detrended neural rows
algebraically cancels back to the constant-base formula, so there is no
distinct "coarse-anchored" variant); these fallbacks are flagged per window
in the report, with the would-have-been re-anchor MAE as a diagnostic.

Arms:

* ``reanchor_1s``     — bases + detrended ``ckla_phaseonly_best`` (1 s model)
  rows.
* ``reanchor_4s``     — bases + detrended ``ckla_phaseonly_4s_best`` rows.
* ``reanchor_4s_pik`` — reanchor_4s, then per-window ``pi_kalman_refine``
  joint refinement (``tracking.phase_increment_tracker``).

Inputs (all local; the VK ladder is never re-run):

* ``results/beatvk_vk_arms/runs/<rid>__wNN__blind_fullrange.npz`` — seed
  bases (post-octave-fix), coarse mode, final trajectories.
* ``results/beatvk_vk_arms/prep_cache/`` — 16 kHz window audio + telemetry.
* ``omnirun-outputs/bash-1e251e/results/beatvk_vk_arms/neural_cache/`` —
  the 1 s model's cached stitched chmean forwards; the 4 s model is
  forwarded here with the same machinery (``rps_predictor_vk_eval``
  sliding 251-frame windows, 32-frame hop, chmean) and cached.

Outputs: ``results/neural_reanchor/<arm>/<rid>.npz`` (the beatvk ``npz:``
convention — directly scorable by ``scripts/beatvk_eval.py --pred
npz:<dir> --arms none``), ``neural_cache/``, ``pik_cache/``,
``report.json``; pooled + per-window tables on stdout. Scoring reuses
``beatvk_eval.score_recording`` (the frozen scorer), so the printed numbers
match a subsequent ``beatvk_eval.py`` run window-for-window.

Run::

    .venv/bin/python scripts/neural_reanchor.py
    .venv/bin/python scripts/beatvk_eval.py \
        --pred npz:results/neural_reanchor/reanchor_4s --arms none
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import beatvk_eval  # noqa: E402  (scripts/ on path)

SR = beatvk_eval.SR
HOP = beatvk_eval.HOP
FRAME_S = beatvk_eval.FRAME_S
N_ROTORS = beatvk_eval.N_ROTORS
STITCH_WIN_FRAMES = beatvk_eval.STITCH_WIN_FRAMES
STITCH_SLIDE_FRAMES = beatvk_eval.STITCH_SLIDE_FRAMES

MODEL_1S = "ckla_phaseonly_best"
MODEL_4S = "ckla_phaseonly_4s_best"
ARM_MODELS: dict[str, str] = {
    "reanchor_1s": MODEL_1S,
    "reanchor_4s": MODEL_4S,
    "reanchor_4s_pik": MODEL_4S,
}
ALL_ARMS = tuple(ARM_MODELS)
REF_ARM = "blind_fullrange"

DEFAULT_VK_ARMS_DIR = Path("results/beatvk_vk_arms")
DEFAULT_NEURAL_1S_CACHE = Path("omnirun-outputs/bash-1e251e/results/beatvk_vk_arms/neural_cache")
DEFAULT_OUT = Path("results/neural_reanchor")


# ---------------------------------------------------------------------------
# inputs


def load_fullrange(runs_dir: Path, rid: str, widx: int) -> dict[str, Any]:
    """One ``blind_fullrange`` run NPZ -> the fields this experiment needs."""
    path = runs_dir / f"{rid}__w{widx:02d}__blind_fullrange.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing blind_fullrange run NPZ: {path}")
    with np.load(path) as z:
        return {
            "start_s": float(z["start_s"]),
            "end_s": float(z["end_s"]),
            "regime": str(z["regime"]),
            "ft": np.asarray(z["ft"], np.float64),
            "traj": np.asarray(z["traj"], np.float64),
            "seed_bases": np.sort(np.asarray(z["seed_bases"], np.float64)),
            "coarse_mode": str(z["coarse_mode"]) if "coarse_mode" in z else "none",
        }


def load_prep(prep_dir: Path, rid: str, widx: int, *, with_audio: bool) -> dict[str, Any]:
    """One prep-cache NPZ -> window audio (optional), frame grid, telemetry."""
    path = prep_dir / f"{rid}__w{widx:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing prep NPZ: {path}")
    with np.load(path) as z:
        out: dict[str, Any] = {
            "start_s": float(z["start_s"]),
            "ft": np.asarray(z["ft"], np.float64),
            "r_meas": np.asarray(z["r_meas"], np.float64),
        }
        if with_audio:
            out["audio"] = np.asarray(z["audio"], np.float32)
    return out


# ---------------------------------------------------------------------------
# neural stitched chmean forwards (per window, cached NPZ)


def neural_cache_path(cache_dir: Path, rid: str, widx: int, model_key: str) -> Path:
    return cache_dir / f"{rid}__w{widx:02d}__{model_key}.npz"


def load_neural_traj(dirs: list[Path], rid: str, widx: int, model_key: str) -> np.ndarray | None:
    for d in dirs:
        p = neural_cache_path(d, rid, widx, model_key)
        if p.exists():
            with np.load(p) as z:
                return np.asarray(z["traj"], np.float64)
    return None


def compute_neural_trajs(
    todo: list[tuple[str, int]],
    model_key: str,
    cache_dir: Path,
    prep_dir: Path,
    device: str | None,
    batch: int,
) -> None:
    """Forward ``model_key`` on each missing window (the beatvk_vk_arms
    ``compute_neural_seeds`` machinery: stitched sliding chmean windows,
    interp onto the window frame grid), cache to ``cache_dir``."""
    if not todo:
        return
    import rps_predictor_vk_eval as vkev
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    experiment, ckpt_uri, _ = vkev.MODELS[model_key]
    tic = time.perf_counter()
    model = vkev.load_model(experiment, ckpt_uri, dev)
    print(
        f"[neural] loaded {model_key} in {time.perf_counter() - tic:.0f}s ({dev})",
        flush=True,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for rid, widx in todo:
        prep = load_prep(prep_dir, rid, widx, with_audio=True)
        audio = np.ascontiguousarray(prep["audio"])  # (8, T) — chmean uses ALL mics
        f_total = audio.shape[-1] // HOP + 1
        if f_total < STITCH_WIN_FRAMES:
            raise ValueError(f"{rid} w{widx}: {f_total} frames < the 8 s model window")
        tic = time.perf_counter()
        starts = vkev.window_starts(f_total, STITCH_WIN_FRAMES, STITCH_SLIDE_FRAMES)
        preds = vkev.predict_windows(model, audio, starts, "chmean", dev, batch, STITCH_WIN_FRAMES)
        stack = vkev.stitch_stack(preds, starts, f_total, STITCH_WIN_FRAMES)
        traj_f = np.nanmean(stack, axis=0)  # (4, f_total) on the model frame grid
        times = np.arange(f_total) * FRAME_S
        ft = prep["ft"]
        traj = np.stack([np.interp(ft, times, traj_f[i]) for i in range(N_ROTORS)])
        wall = time.perf_counter() - tic
        np.savez(
            neural_cache_path(cache_dir, rid, widx, model_key),
            traj=traj,
            wall_s=np.float64(wall),
        )
        print(
            f"[neural | {rid} w{widx:02d} | {model_key}] medians "
            f"{np.round(np.median(traj, axis=1), 2)} ({wall:.0f}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# re-anchoring


def reanchor(bases_sorted: np.ndarray, neural: np.ndarray) -> np.ndarray:
    """``base_i + (neural_{j(i)} - mean neural_{j(i)})``, rank-to-rank.

    Neural rows sorted by window mean take the sorted bases in order; the
    output row order is the rank order (arbitrary — scoring is PIT).
    """
    order = np.argsort(neural.mean(axis=1))
    fluct = neural - neural.mean(axis=1, keepdims=True)
    return bases_sorted[:, None] + fluct[order]


def window_pit_mae(traj: np.ndarray, r_meas: np.ndarray) -> float:
    """Informational window-grid PIT-MAE vs raw telemetry (diagnostics only;
    arm scores come from the frozen scorer)."""
    from losses.pit import align_rps_to_gt

    return float(np.mean(np.abs(align_rps_to_gt(traj, r_meas) - r_meas)))


def assignment_check(
    bases_sorted: np.ndarray, neural: np.ndarray, r_meas: np.ndarray
) -> dict[str, Any]:
    """Rank-order assignment vs the PIT-optimal (via-GT) pairing.

    PIT-aligns the neural rows and the constant base rows to the telemetry
    independently; the induced base<->neural pairing is the PIT-optimal one.
    Reports whether it equals the rank-to-rank pairing.
    """
    from scipy.optimize import linear_sum_assignment

    def perm_to_gt(rows: np.ndarray) -> np.ndarray:
        cost = np.mean((rows[:, None, :] - r_meas[None, :, :]) ** 2, axis=-1)
        r, c = linear_sum_assignment(cost)
        gt_of = np.empty(N_ROTORS, dtype=int)
        gt_of[r] = c
        return gt_of  # gt rotor index for each row

    gt_of_neural = perm_to_gt(neural)
    base_rows = np.repeat(bases_sorted[:, None], neural.shape[1], axis=1)
    gt_of_base = perm_to_gt(base_rows)
    # PIT pairing: base k <-> neural row j sharing the same GT rotor.
    neural_of_gt = np.empty(N_ROTORS, dtype=int)
    neural_of_gt[gt_of_neural] = np.arange(N_ROTORS)
    pit_pairs = [int(neural_of_gt[gt_of_base[k]]) for k in range(N_ROTORS)]
    rank_pairs = [int(j) for j in np.argsort(neural.mean(axis=1))]
    return {
        "rank_pairs": rank_pairs,  # neural row taken by sorted base k
        "pit_pairs": pit_pairs,
        "mismatch": rank_pairs != pit_pairs,
    }


# ---------------------------------------------------------------------------
# pi_kalman refinement (per window, cached)


def pik_refine_window(
    out: Path,
    prep_dir: Path,
    rid: str,
    widx: int,
    r0: np.ndarray,
    *,
    pair_mode: str,
    n_iter: int,
    band_hz: float | tuple[float, ...],
) -> tuple[np.ndarray, float]:
    """``pi_kalman_refine`` on one window from init ``r0``; cached by the
    init's hash so a changed A2 trajectory invalidates the cache."""
    from tracking.phase_increment_tracker import pi_kalman_refine

    cache = out / "pik_cache" / f"{rid}__w{widx:02d}.npz"
    key = float(np.sum(r0))  # cheap init fingerprint
    if cache.exists():
        with np.load(cache) as z:
            if abs(float(z["init_sum"]) - key) < 1e-6:
                return np.asarray(z["r_hat"], np.float64), float(z["wall_s"])
    prep = load_prep(prep_dir, rid, widx, with_audio=True)
    audio = np.asarray(prep["audio"], np.float64)
    ft = prep["ft"]
    tic = time.perf_counter()
    r_hat, _ = pi_kalman_refine(
        audio, r0, ft, sr=SR, n_iter=n_iter, pair_mode=pair_mode, band_hz=band_hz
    )
    wall = time.perf_counter() - tic
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, r_hat=r_hat, wall_s=np.float64(wall), init_sum=np.float64(key))
    return r_hat, wall


# ---------------------------------------------------------------------------
# assembly + scoring


def assemble_arm(
    out: Path, arm: str, per_window: dict[str, list[tuple[float, np.ndarray, np.ndarray]]]
) -> None:
    """Write ``<out>/<arm>/<rid>.npz`` (absolute ft + concatenated rps)."""
    arm_dir = out / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    for rid, chunks in per_window.items():
        fts = [start + ft for start, ft, _ in chunks]
        ft_all = np.concatenate(fts)
        rps_all = np.concatenate([traj for _, _, traj in chunks], axis=1)
        if not np.all(np.diff(ft_all) > 0):
            raise RuntimeError(f"{arm}/{rid}: non-monotonic assembled ft")
        if not np.all(np.isfinite(rps_all)):
            raise RuntimeError(f"{arm}/{rid}: non-finite trajectory values")
        np.savez(arm_dir / f"{rid}.npz", ft=ft_all, rps=rps_all)


def score_npz_dir(recs: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Frozen-scorer rows for a beatvk npz: dir (arm field dropped)."""
    preds = beatvk_eval.preds_from_npz(path, [r["recording_id"] for r in recs])
    rows: list[dict[str, Any]] = []
    for rec in recs:
        rid = rec["recording_id"]
        if rid not in preds:
            continue
        ft, rps = preds[rid]
        rows.extend(beatvk_eval.score_recording(rec, ft, rps, ["none"]))
    return rows


def pool(rows: list[dict[str, Any]], predfn: Any) -> float | None:
    v = [r["mae"] for r in rows if predfn(r)]
    return float(np.mean(v)) if v else None


POOLS: list[tuple[str, Any]] = [
    (
        "dregon_cruise",
        lambda r: r["recording"] in beatvk_eval.DREGON_RECS and r["regime"] == "cruise",
    ),
    (
        "fly124_cruise",
        lambda r: r["recording"] == beatvk_eval.FLY124_REC and r["regime"] == "cruise",
    ),
    ("all_cruise", lambda r: r["regime"] == "cruise"),
    ("all_warmup", lambda r: r["regime"] == "warmup"),
    ("all_windows", lambda r: True),
]


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--arms", default=",".join(ALL_ARMS), help=f"comma subset of {ALL_ARMS}")
    ap.add_argument("--recordings", nargs="+", default=None, help="restrict to these recordings")
    ap.add_argument("--vk-arms-dir", default=str(DEFAULT_VK_ARMS_DIR))
    ap.add_argument("--neural-1s-cache", default=str(DEFAULT_NEURAL_1S_CACHE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--device", default=None, help="cuda|cpu for neural forwards (default: auto)")
    ap.add_argument("--batch", type=int, default=16, help="neural inference batch")
    ap.add_argument("--pair-mode", default="joint", choices=("gate", "joint"))
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument(
        "--band-hz",
        default="6",
        help="pi_kalman demod half-band (Hz): one float or a comma schedule",
    )
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; valid: {list(ALL_ARMS)}")
    vk_dir = Path(args.vk_arms_dir)
    runs_dir, prep_dir = vk_dir / "runs", vk_dir / "prep_cache"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_1s = Path(args.neural_1s_cache)
    bands = tuple(float(b) for b in str(args.band_hz).split(","))
    band_hz: float | tuple[float, ...] = bands[0] if len(bands) == 1 else bands

    wanted = set(args.recordings) if args.recordings else None
    recs = beatvk_eval.load_recordings(args.dataset_version, wanted, keep_audio=False)
    print(
        f"[neural_reanchor] {beatvk_eval.DATASET}@{recs[0]['dataset_version'][:12]}: "
        f"{[r['recording_id'] for r in recs]}",
        flush=True,
    )
    jobs = [(rec["recording_id"], int(w["index"])) for rec in recs for w in rec["windows"]]

    # Neural forwards: the 1 s model comes from the existing omnirun cache;
    # the 4 s model (and any missing 1 s window) is forwarded here.
    models_needed = sorted({ARM_MODELS[a] for a in arms})
    cache_dirs = {
        MODEL_1S: [cache_1s, out / "neural_cache"],
        MODEL_4S: [out / "neural_cache"],
    }
    for model_key in models_needed:
        todo = [
            (rid, widx)
            for rid, widx in jobs
            if load_neural_traj(cache_dirs[model_key], rid, widx, model_key) is None
        ]
        compute_neural_trajs(
            todo, model_key, out / "neural_cache", prep_dir, args.device, args.batch
        )

    # Per-window arm trajectories + diagnostics.
    per_arm: dict[str, dict[str, list[tuple[float, np.ndarray, np.ndarray]]]] = {
        arm: {} for arm in arms
    }
    diag_rows: list[dict[str, Any]] = []
    pik_wall_total = 0.0
    tic_all = time.perf_counter()
    for rid, widx in jobs:
        fr = load_fullrange(runs_dir, rid, widx)
        prep = load_prep(prep_dir, rid, widx, with_audio=False)
        if len(fr["ft"]) != len(prep["ft"]) or abs(fr["start_s"] - prep["start_s"]) > 1e-6:
            raise RuntimeError(f"{rid} w{widx}: fullrange/prep frame grids differ")
        is_ramp = fr["coarse_mode"] == "coarse"
        bases = fr["seed_bases"]
        drow: dict[str, Any] = {
            "recording": rid,
            "window": widx,
            "regime": fr["regime"],
            "coarse_mode": fr["coarse_mode"],
            "mode": "fallback" if is_ramp else "reanchor",
            "seed_bases": [round(float(b), 2) for b in bases],
        }
        for arm in arms:
            model_key = ARM_MODELS[arm]
            neural = load_neural_traj(cache_dirs[model_key], rid, widx, model_key)
            if neural is None:
                raise RuntimeError(f"{rid} w{widx}: no {model_key} forward cached")
            re_traj = reanchor(bases, neural)
            traj = fr["traj"].copy() if is_ramp else re_traj
            if arm == "reanchor_4s_pik":
                traj, wall = pik_refine_window(
                    out,
                    prep_dir,
                    rid,
                    widx,
                    traj,
                    pair_mode=args.pair_mode,
                    n_iter=args.n_iter,
                    band_hz=band_hz,
                )
                pik_wall_total += wall
                drow["pik_wall_s"] = round(wall, 1)
            per_arm[arm].setdefault(rid, []).append((fr["start_s"], fr["ft"], traj))
            # Info diagnostics (window grid, raw telemetry).
            tag = model_key.removesuffix("_best")
            if f"neural_raw_mae__{tag}" not in drow:
                drow[f"neural_raw_mae__{tag}"] = round(window_pit_mae(neural, prep["r_meas"]), 3)
                drow[f"reanchor_mae__{tag}"] = round(window_pit_mae(re_traj, prep["r_meas"]), 3)
                drow[f"assignment__{tag}"] = assignment_check(bases, neural, prep["r_meas"])
        diag_rows.append(drow)

    for arm in arms:
        assemble_arm(out, arm, per_arm[arm])

    # Frozen-protocol scoring: the arms + the blind_fullrange reference.
    score_dirs = {arm: out / arm for arm in arms}
    ref_dir = vk_dir / REF_ARM
    if ref_dir.is_dir():
        score_dirs[REF_ARM] = ref_dir
    scored = {name: score_npz_dir(recs, d) for name, d in score_dirs.items()}

    names = list(scored)
    key = lambda r: (r["recording"], r["window"])  # noqa: E731
    by_win = {name: {key(r): r for r in rows} for name, rows in scored.items()}
    print("\nPer-window PIT-MAE (rev/s), frozen beat-VK scoring:")
    header = f"{'recording':<36}{'w':>3} {'regime':<8}{'mode':<10}" + "".join(
        f"{n:>16}" for n in names
    )
    print(header)
    print("-" * len(header))
    for drow in diag_rows:
        k = (drow["recording"], drow["window"])
        cells = "".join(
            f"{by_win[n][k]['mae']:>16.3f}" if k in by_win[n] else f"{'—':>16}" for n in names
        )
        print(
            f"{drow['recording']:<36}{drow['window']:>3} {drow['regime']:<8}"
            f"{drow['mode']:<10}{cells}"
        )

    print("\nPooled window PIT-MAE (rev/s):")
    header = f"{'pool':<16}" + "".join(f"{n:>16}" for n in names)
    print(header)
    print("-" * len(header))
    pooled: dict[str, dict[str, float | None]] = {}
    for pname, predfn in POOLS:
        row = f"{pname:<16}"
        pooled[pname] = {}
        for n in names:
            v = pool(scored[n], predfn)
            pooled[pname][n] = v
            row += f"{v:>16.3f}" if v is not None else f"{'—':>16}"
        print(row)

    mismatches = [
        (d["recording"], d["window"], t)
        for d in diag_rows
        for t in (MODEL_1S.removesuffix("_best"), MODEL_4S.removesuffix("_best"))
        if d.get(f"assignment__{t}", {}).get("mismatch")
    ]
    print(
        f"\nAssignment sanity: {len(mismatches)} rank-vs-PIT mismatches"
        + (f" -> {mismatches}" if mismatches else ""),
        flush=True,
    )
    wall_all = time.perf_counter() - tic_all
    print(
        f"Runtime: assemble+score {wall_all:.0f}s"
        + (f" (pi_kalman {pik_wall_total:.0f}s total)" if pik_wall_total else "")
    )

    report = {
        "protocol": (
            "per blind_fullrange window: r_i(t) = sorted_base_i + rank-assigned "
            "detrended neural row (window-mean removed); coarse-engaged (ramp) "
            "windows keep the blind_fullrange final trajectory; scoring = "
            "beatvk_eval.score_recording on the frozen dataset"
        ),
        "dataset": {"name": beatvk_eval.DATASET, "version": recs[0]["dataset_version"]},
        "arms": {arm: {"model": ARM_MODELS[arm]} for arm in arms},
        "pik": {"pair_mode": args.pair_mode, "n_iter": args.n_iter, "band_hz": args.band_hz}
        if "reanchor_4s_pik" in arms
        else None,
        "pooled": pooled,
        "per_window": {
            name: [{k: r[k] for k in ("recording", "window", "regime", "mae", "mse")} for r in rows]
            for name, rows in scored.items()
        },
        "diagnostics": diag_rows,
        "runtime_s": {"total": round(wall_all, 1), "pi_kalman": round(pik_wall_total, 1)},
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[neural_reanchor] wrote {out}/report.json; arm dirs scorable via e.g.:")
    for arm in arms:
        print(f"  .venv/bin/python scripts/beatvk_eval.py --pred npz:{out / arm} --arms none")


if __name__ == "__main__":
    main()
