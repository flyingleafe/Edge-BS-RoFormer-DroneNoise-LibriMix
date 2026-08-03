#!/usr/bin/env python3
"""Single-rotor exact-DP probe: with zero search error, what does the objective buy?

The joint 4-rotor beam search loses to a trivial baseline, and `jb_probe --mode
cost` can only say whether the objective PREFERS the truth to the beam's own
output — it cannot say what an exact search would find.  For ONE rotor the
state space is a scalar speed grid, so exact banded Viterbi
(`data_processing.rotor_dp`) is tractable and there is zero search error by
construction: every failure measured here is the objective's.

Three arms per window, one fixed emission (the measured winner of the ceiling
sweep: n_fft 4096, k <= 16, quantile pooling q = 0.25, point-sampled teeth,
grid step 0.1):

``oracle_masked``
    The intermediate-goal test.  For each rotor, claim the OTHER three at
    their ground-truth trajectories and run the exact DP on the residual
    surface.  This isolates one rotor's per-frame evidence from the
    assignment problem: if the DP cannot follow a rotor even with its three
    siblings removed by an oracle, no 4-rotor search over this emission can.

``raw_dp``
    One DP on the unmasked surface — which rotor does the objective's global
    optimum lock onto, and how tightly?

``greedy_peel``
    The blind 4-rotor product: track, claim, repeat.  Scored with
    `rps_refine_lab.stage_metrics` on the GT frame grid, so `pooled_mae` is
    directly comparable to the frozen joint-beam numbers
    (6.64/5.24/21.66/8.73/8.34/0.44) and to fullrange_init (1.0-3.5).

Per-window restartable JSON units, mirroring `jb_probe.py`.

Cluster::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 8 --mem 24 --time 2h -- \\
        python scripts/sr_dp_probe.py --build-preps --jobs 6
    omnirun submit --backend uni-gpushort --gpus 1 --time 1h -- \\
        python scripts/sr_dp_probe.py --build-preps --device cuda --jobs 1
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import jb_probe  # noqa: E402
import numpy as np  # noqa: E402
from jb_probe import WINDOWS  # noqa: E402  (re-exported: the probe window set)

#: The spectrogram the tables are built on (not an EmissionCfg field).
#: Default; ``--n-fft 2048`` halves the analysis window for ramp windows,
#: where a comb moving ~8 rev/s/s smears its k >= 4 teeth across several bins
#: within a 256 ms window and the high-quantile pool goes blind.
N_FFT = 4096

#: One fixed emission — the measured winner of the jb_probe ceiling sweep —
#: with `step` 0.1 instead of the joint tracker's 0.5: exact DP is O(D * B)
#: per frame, so a 10x finer grid is affordable where a `len(grid)^4` beam
#: could not pay for it.
EMIS_KW: dict[str, Any] = {
    "k_max": 16,
    "pool": "quantile",
    "pool_q": 0.25,
    "b0_rps": 0.0,
    "step": 0.1,
}

#: Localisation tolerances for the oracle arm, rev/s.
TOLS = (0.25, 0.5, 1.0)

N_ROTORS = 4


def _snap_idx(gt_row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Grid indices of a GT trajectory; out-of-grid (and NaN) frames clamped."""
    v = np.nan_to_num(gt_row, nan=float(grid[0]))
    step = float(grid[1] - grid[0])
    return np.clip(np.round((v - grid[0]) / step), 0, len(grid) - 1).astype(np.int64)


def dp_unit(task: tuple[str, Path, str, float, float, int]) -> tuple[str, str]:
    """All three arms on one window; one restartable JSON unit."""
    window, results, device, s_rps, lambda_e, n_fft = task
    out = jb_probe.unit_path(results, "srdp", window)
    if out.exists():
        return window, "skip"
    tic = time.perf_counter()
    try:
        import rps_refine_lab as lab
        import torch

        from data_processing.joint_beam_tracker import EmissionCfg, comb_tables
        from data_processing.rotor_dp import LatticeCfg, greedy_peel, track_masked

        prep, meta = jb_probe.load_window(window)
        emis = EmissionCfg(**EMIS_KW)
        lat = LatticeCfg(s_rps=s_rps, lambda_e=lambda_e)
        lm, bin_hz, st = jb_probe.whitened_spec(prep.audio, n_fft)
        gt = jb_probe.gt_on(prep, st)  # (4, T)
        grid = np.asarray(emis.grid())

        lm_t = torch.as_tensor(lm, device=device, dtype=torch.float32)
        grid_t = torch.as_tensor(grid, device=device, dtype=torch.float32)
        t0 = time.perf_counter()
        tab = comb_tables(lm_t, bin_hz, emis, grid_t)
        wall: dict[str, float] = {"tables": time.perf_counter() - t0}

        rec: dict[str, Any] = {
            "window": window,
            "regime": meta.get("regime"),
            "n_spec_frames": int(len(st)),
            "audio_s": float(st[-1]),
            "emis_cfg": {"n_fft": n_fft, **EMIS_KW},
            "lat_cfg": {
                "s_rps": lat.s_rps,
                "huber_knee": lat.huber_knee,
                "max_step_rps": lat.max_step_rps,
                "lambda_e": lat.lambda_e,
            },
            "arms": {},
        }

        # --- arm 1: oracle_masked — one rotor's evidence, siblings removed
        t0 = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for r in range(N_ROTORS):
            others = [q for q in range(N_ROTORS) if q != r]
            claimed = torch.as_tensor(
                np.stack([_snap_idx(gt[q], grid) for q in others]), device=device
            )
            res = track_masked(tab, emis, lat, claimed, grid_t)
            fin = np.isfinite(gt[r]) & (gt[r] >= grid[0]) & (gt[r] <= grid[-1])
            err = np.abs(res["speeds"][fin] - gt[r][fin])
            row: dict[str, Any] = {
                "rotor": r,
                "n_frames_scored": int(fin.sum()),
                "mae": float(err.mean()) if fin.any() else None,
                "support_raw_mean": res["support_raw_mean"],
                "total_cost": res["total_cost"],
            }
            for tol in TOLS:
                row[f"hit_{tol}"] = float((err <= tol).mean()) if fin.any() else None
            rows.append(row)
        maes = [row["mae"] for row in rows if row["mae"] is not None]
        rec["arms"]["oracle_masked"] = {
            "per_rotor": rows,
            "mae_worst": float(max(maes)) if maes else None,
            "mae_mean": float(np.mean(maes)) if maes else None,
        }
        wall["oracle_masked"] = time.perf_counter() - t0

        # --- arm 2: raw_dp — where does the unmasked global optimum go?
        t0 = time.perf_counter()
        res = track_masked(tab, emis, lat, None, grid_t)
        mae_per: list[float | None] = []
        for r in range(N_ROTORS):
            fin = np.isfinite(gt[r])
            mae_per.append(
                float(np.abs(res["speeds"][fin] - gt[r][fin]).mean()) if fin.any() else None
            )
        scored = [(m, r) for r, m in enumerate(mae_per) if m is not None]
        best = min(scored) if scored else (None, None)
        rec["arms"]["raw_dp"] = {
            "mae_closest": best[0],
            "closest_rotor": best[1],
            "mae_per_rotor": mae_per,
            "support_raw_mean": res["support_raw_mean"],
            "total_cost": res["total_cost"],
        }
        wall["raw_dp"] = time.perf_counter() - t0

        # --- arm 3: greedy_peel — the blind 4-rotor product, scored like every
        # other stage so `pooled_mae` lands on the frozen scoreboard directly
        t0 = time.perf_counter()
        gp = greedy_peel(tab, emis, lat, n_rotors=N_ROTORS, grid=grid_t)
        on_ft = np.stack([np.interp(prep.ft, st, row) for row in gp["speeds"]])
        # per-frame minimum pairwise separation of the extracted tracks — the
        # collapse detector (the first run put all four within 0.4 rev/s)
        seps = [
            np.abs(gp["speeds"][i] - gp["speeds"][j])
            for i in range(N_ROTORS)
            for j in range(i + 1, N_ROTORS)
        ]
        rec["arms"]["greedy_peel"] = {
            **lab.stage_metrics(on_ft, prep),
            "supports": gp["supports"],
            "costs": gp["costs"],
            "speeds_mean": [float(v) for v in gp["speeds"].mean(axis=1)],
            "pair_sep_min_mean": float(np.min(np.stack(seps), axis=0).mean()),
        }
        wall["greedy_peel"] = time.perf_counter() - t0

        wall["total"] = time.perf_counter() - tic
        rec["wall_s"] = {k: round(v, 1) for k, v in wall.items()}
        jb_probe._write(out, lab.r3(rec))
        return window, "ok"
    except Exception:  # noqa: BLE001 - one bad unit must not kill the probe
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(traceback.format_exc())
        return window, "ERROR"


def summarise(results: Path) -> dict[str, Any]:
    rows = [json.loads(f.read_text()) for f in sorted((results / "raw").glob("srdp__*.json"))]
    table = []
    for r in rows:
        arms = r["arms"]
        table.append(
            {
                "window": r["window"],
                "oracle_mae_worst": arms["oracle_masked"]["mae_worst"],
                "oracle_mae_mean": arms["oracle_masked"]["mae_mean"],
                "oracle_hit_0.5_per_rotor": [
                    q["hit_0.5"] for q in arms["oracle_masked"]["per_rotor"]
                ],
                "raw_dp_mae_closest": arms["raw_dp"]["mae_closest"],
                "raw_dp_closest_rotor": arms["raw_dp"]["closest_rotor"],
                "greedy_pooled_mae": arms["greedy_peel"]["pooled_mae"],
                "greedy_supports": arms["greedy_peel"]["supports"],
                "wall_s": r["wall_s"],
            }
        )
    return {"n_windows": len(rows), "table": table}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", nargs="*", default=list(WINDOWS))
    ap.add_argument("--results", default="results/sr_dp_probe")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--s-rps", type=float, default=0.4, help="LatticeCfg.s_rps override")
    ap.add_argument("--lambda-e", type=float, default=3.0, help="LatticeCfg.lambda_e override")
    ap.add_argument("--n-fft", type=int, default=N_FFT, help="analysis FFT size")
    ap.add_argument(
        "--build-preps",
        action="store_true",
        help="materialise the beat-VK prep cache first.  REQUIRED on a cluster: "
        "`results/beatvk_vk_arms/prep_cache` is a gitignored local artefact, so a "
        "fresh worktree has no windows to score and every unit dies on "
        "FileNotFoundError — silently, because per-unit exceptions become .err files.",
    )
    args = ap.parse_args()

    if args.build_preps:
        import beatvk_rescore as brs
        from beatvk_vk_arms import DEFAULT_OUT

        brs.build_prep_cache(Path(DEFAULT_OUT), None, brs.resolve_dregon_dir())

    results = Path(args.results)
    tasks = [(w, results, args.device, args.s_rps, args.lambda_e, args.n_fft) for w in args.windows]
    if args.jobs <= 1:
        for t in tasks:
            print(dp_unit(t), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(dp_unit, t) for t in tasks]
            for f in as_completed(futs):
                print(f.result(), flush=True)

    summary = summarise(results)
    (results / "summary_srdp.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    n_err = len(list((results / "raw").glob("srdp__*.err")))
    if n_err:
        print(f"!! {n_err} unit(s) failed — see {results}/raw/*.err", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
