#!/usr/bin/env python3
"""Rescore trajectory hypotheses by the JOINT decomposition's own MAP objective.

The v3 decomposition is block-coordinate descent on one objective (see
:func:`tracking.map_objective`)::

    J = sum_{c,f,t} [ P/S + log S ]  +  phase priors  +  envelope prior

Everything in it is conditioned on a TRAJECTORY, and nothing in it is fitted to
one: the carrier is given, the harmonic set is pinned, and the cell set is the
same whatever the carrier does (``FVKConfig.vk_config`` opens the validity mask
for exactly this reason). So ``J`` at convergence is a likelihood-shaped score
OF the trajectory, and the decomposition is a MEASURE and not only a product.

This driver runs that measurement: for each frozen ``beatvk`` window and each
trajectory hypothesis it runs the shipped v3b alternation and records the
converged objective, term by term.

Hypotheses
----------
``telemetry``
    The prep window's own stored tachometer trajectory (``r_meas``).
``multistart`` / ``ours_full``
    The step-5 blind arms of ``scripts/fvk_arms.py``, read back from that
    campaign's unit JSONs (``<arms>/raw/s5__<window>__<arm>.json`` -> ``r_out``).
    They are BLIND — nothing in them saw the telemetry — which is what makes the
    comparison interesting: a blind trajectory that scores BELOW the telemetry
    is a trajectory the audio prefers.

Comparability is the whole point, so three things are pinned across the
hypotheses of one window: the audio, the microphone count, and the harmonic cap
``k_hi``, which is taken from the TELEMETRY trajectory and never from the
candidate. The objective is extensive, so read the per-cell column when
comparing windows and the raw column when comparing hypotheses of one window.

``--marginal`` ranks by the MARGINAL objective instead. The profiled ``J``
substitutes the envelopes' best value back and charges nothing for their
freedom, so a hypothesis whose bands cover more of the spectrum can win by
ABSORPTION alone — which is what a coverage fan does. The marginal objective
integrates the envelopes out exactly, ``J + 0.5 (log det M - log det' R)``, and
charges for it. Both columns are printed, and the two orders DISAGREEING is the
reading: it names the windows a fan wins by absorption.

``--h-aware`` ranks by the H-AWARE objective instead. The other half of the fan's
advantage is not absorption but the LINE FLANKS: no coherent envelope can carry
the ``0.6 k`` Hz flanks of a line (regime 3), so the profiled data term charges
every hypothesis for that flank energy alike and the true trajectory gains
nothing by sitting on it. The H-aware data term gives the noise model a
comb-shaped nuisance ``H = max(0, P~ - S)`` inside the hypothesis's OWN search
regions, so a trajectory whose regions sit on the humps stops paying for them
and a fan that opens regions on empty floor gains nothing. The profiled column
stays beside it, and where the two orders differ the difference is coverage the
profiled column did not charge.

Two levers sharpen that data term where it measured a null, and both are opt in.
``--adaptive-floor`` gives ``S`` one profiled scale per (channel, frame): the
floor is fitted per four-second block while DREGON's rotor wash is not
stationary over one, so a gust span pays a Whittle misfit no comb hypothesis
caused and the block floor plus its rent then move with however each
hypothesis's own solve spread that energy. ``--h-lorentzian`` constrains ``H``
to a non-negative Lorentzian mixture pinned at the hypothesis's OWN line
positions at the measured ``0.6 k`` Hz half width: the shape-free ``H`` explains
ANY excess inside a region, which discriminates nothing on the one window where
four dense combs make every hypothesis's regions blanket the band, and a wrong
comb's bumps land between its lines and fit near-zero amplitudes.

Run::

    python scripts/joint_rescore.py --smoke --out results/joint_rescore_smoke
    python scripts/joint_rescore.py --jobs 5 --out results/joint_rescore

On a cluster the step-5 results do not travel (``results/`` is gitignored), so
pack the trajectories first and hand the job the pack::

    python scripts/joint_rescore.py --pack hypotheses_step5.json
    # commit hypotheses_step5.json, then
    omnirun submit ... -- python scripts/joint_rescore.py --build-preps \
        --arms-dir hypotheses_step5.json --jobs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# The harness convention: process-level parallelism, one BLAS thread per worker
# (utils.gridrun re-asserts it, but the tracking stack reads its own thread knob
# at import time).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TRACKING_FFT_WORKERS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fvk_arms  # noqa: E402 — the campaign whose hypotheses this rescores

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

#: The protocol's sample rate and the campaign's modelling ceiling, both from
#: the drivers that produced the material this reads.
SR = fvk_arms.SR
F_MAX = 6000.0
#: The five step-5 windows: one cruise window from each DREGON recording plus
#: two FLY124 cruise windows. Declared once, in ``scripts/fvk_arms.py``.
DEFAULT_WINDOWS = fvk_arms.BLIND_REAL
#: ``telemetry`` is the prep window's own trajectory; every other name is a
#: step-5 arm read back from that campaign's unit JSONs.
TELEMETRY = "telemetry"
DEFAULT_HYPOTHESES = (TELEMETRY, "multistart", "ours_full")
#: The shipped v3b arm — the configuration ``scripts/vk_decompose.py`` runs and
#: the one the regression fixture pins. Nothing here retunes it.
DEFAULT_ITERS = 3
DEFAULT_LADDER = (3, 12, 80)
#: The terms pooled and tabulated, in the order a reader wants them.
TERMS = ("total", "data", "rent", "phase_priors", "envelope_prior")
#: The MARGINAL readout's own terms, present only under ``--marginal``.
MARGINAL_TERMS = ("total_marginal", "marginal_correction")
#: Which per-cell column ``--marginal`` ranks by. The profiled total is still
#: reported beside it, because the two disagreeing IS the reading.
MARGINAL_KEY = "total_marginal"
#: The H-AWARE readout's own terms, present only under ``--h-aware``.
H_TERMS = ("total_h", "data_h")
#: Which per-cell column ``--h-aware`` ranks by. It takes precedence over the
#: marginal key when both are asked for, and both other columns stay beside it.
H_KEY = "total_h"


# ---------------------------------------------------------------------------
# the hypotheses


def arm_path(arms_dir: str | Path, window: str, arm: str) -> Path:
    """Where ``scripts/fvk_arms.py`` left one step-5 arm's unit JSON."""
    return Path(arms_dir) / "raw" / f"s5__{window}__{arm}.json"


def read_arm(arms: str | Path, window: str, arm: str) -> list[list[float]]:
    """One arm's trajectory, from the campaign's ``raw/`` OR from a PACK file.

    The pack exists because ``results/`` is gitignored and does not travel to a
    cluster: :func:`pack_hypotheses` folds the handful of trajectories a rescore
    needs into one small JSON that can be shipped with the job (a 4 x 500 array
    of four-decimal rates is about 20 kB), and both spellings read the same way
    here so the worker never knows which one it got.
    """
    src = Path(arms)
    if src.is_file():
        packed = json.loads(src.read_text()).get("windows", {})
        if window not in packed or arm not in packed[window]:
            raise KeyError(f"{src}: no {arm!r} trajectory for {window!r} in the pack")
        return packed[window][arm]
    path = arm_path(src, window, arm)
    if not path.exists():
        raise FileNotFoundError(
            f"no step-5 result for {arm!r} on {window!r} at {path} — run scripts/fvk_arms.py "
            "--step 5 first, pull its results, or hand over a --pack file"
        )
    return json.loads(path.read_text())["r_out"]


def pack_hypotheses(
    arms: str | Path, windows: list[str], hyps: list[str], path: str | Path
) -> dict[str, Any]:
    """Fold the arms' trajectories for one rescore into ONE shippable JSON."""
    out = {
        "source": str(arms),
        "windows": {w: {h: read_arm(arms, w, h) for h in hyps if h != TELEMETRY} for w in windows},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out))
    return out


def load_hypothesis(hyp: str, window: str, r_meas: Any, arms: str | Path) -> Any:
    """``(R, N)`` trajectory of one hypothesis on the window's OWN frame grid.

    The arms wrote their trajectory rounded to four decimals, which is the
    hypothesis as the campaign recorded it — this reads it back rather than
    re-running the arm, so the rescore judges exactly the published trajectory.
    """
    import numpy as np

    if hyp == TELEMETRY:
        return np.asarray(r_meas, dtype=np.float64)
    r = np.asarray(read_arm(arms, window, hyp), dtype=np.float64)
    want = np.asarray(r_meas).shape
    if r.shape != want:
        raise ValueError(
            f"{hyp} on {window}: trajectory {r.shape} against the window's grid {want}"
        )
    return r


# ---------------------------------------------------------------------------
# the worker


def worker(unit: Unit) -> dict[str, Any]:
    """One (window, hypothesis): the v3b alternation, read out by its objective."""
    import numpy as np

    import tracking as trk
    from tracking.decompose import group_plan, solve_config
    from tracking.fitness_vk import k_cap, to_audio_grid
    from tracking.joint_decompose import JointConfig
    from tracking.protocols import load_prep_window

    p = dict(unit.params)
    window, hyp = str(p["window"]), str(p["hypothesis"])
    pdir = p.get("prep_dir")
    z = load_prep_window(window, Path(pdir) if pdir else None)
    ft, r_meas = np.asarray(z["ft"], dtype=np.float64), np.asarray(z["r"], dtype=np.float64)
    audio = np.ascontiguousarray(z["audio"][: int(p["mics"])], dtype=np.float64)
    sec = float(p.get("seconds") or 0.0)
    if sec > 0:
        n_a = min(audio.shape[-1], int(round(sec * SR)))
        keep = ft < (n_a / SR - 1e-9)
        audio, ft, r_meas = audio[:, :n_a], ft[keep], r_meas[:, keep]
    n_t, mics = int(audio.shape[-1]), int(audio.shape[0])

    cfg = solve_config(int(p["k_max"]), sr=SR, mics=mics, bw_rps=1.0, f_max=float(p["f_max"]))
    # The harmonic cap comes from the TELEMETRY, so every hypothesis of this
    # window is scored on the identical track set and the identical cells.
    k_hi = int(k_cap(cfg, r_meas))
    r = load_hypothesis(hyp, window, r_meas, p["arms_dir"])
    r_audio = to_audio_grid(r, ft, n_t, SR)

    plan = group_plan(r_audio, k_hi, cfg)
    budget = float(p.get("mem_budget_gb") or 0.0)
    if budget > 0 and float(plan["banded_gb"]) > budget:
        # Fail this unit with the arithmetic instead of letting the operating
        # system kill the pool; gridrun turns the exception into one .err file.
        raise MemoryError(
            f"the coupled group needs {plan['banded_gb']} GB (max group {plan['max_group']} "
            f"tracks, {plan['n_env']} envelope frames) against a --mem-budget-gb of {budget}"
        )

    jcfg = JointConfig(
        iters=int(p["iters"]),
        k_trust=tuple(int(v) for v in str(p["k_trust"]).split(",") if v.strip()),
        # The exact Gaussian envelope marginalization. Profiling the envelopes
        # pays no rent for their freedom, which is why absorption is free and
        # why a coverage fan can out-score the telemetry on the profiled total.
        marginal=bool(p.get("marginal", False)),
        # The H-AWARE data term. The coherent envelopes cannot carry the line
        # FLANKS, so the profiled data term charges every hypothesis for the
        # same flank energy; this lets a hypothesis EXPLAIN the humps its own
        # comb regions cover, and a fan that opens regions on empty floor gains
        # nothing by it.
        h_aware=bool(p.get("h_aware", False)),
        # One profiled floor scale per (channel, frame). The block floor is
        # constant over four seconds and DREGON's rotor wash is not, so without
        # it a gust span pays a Whittle misfit no hypothesis caused and the
        # block floor plus its rent move with how each solve spread that energy.
        adaptive_floor=bool(p.get("adaptive_floor", False)),
        # The H nuisance constrained to LORENTZIANS at the hypothesis's own
        # lines. The shape-free hump explains any excess a region covers, which
        # discriminates nothing where four dense combs make every hypothesis's
        # regions blanket the band.
        h_lorentzian=bool(p.get("h_lorentzian", False)),
    )
    tic = time.perf_counter()
    res = trk.joint_solve_window(audio, r_audio, cfg, k_hi=k_hi, mics=mics, jcfg=jcfg)
    wall = time.perf_counter() - tic

    last = dict(res.iterations[-1])
    obj = dict(last["objective"])
    n_cells = max(int(obj["n_cells"]), 1)
    terms = (
        TERMS
        + (MARGINAL_TERMS if bool(p.get("marginal", False)) else ())
        + (H_TERMS if bool(p.get("h_aware", False)) else ())
    )
    return {
        "uid": unit.uid,
        "window": window,
        "hypothesis": hyp,
        "regime": str(z["regime"]),
        "mics": mics,
        "n_frames": int(ft.size),
        "n_rotors": int(r.shape[0]),
        "duration_s": round(n_t / float(SR), 3),
        "k_max": int(p["k_max"]),
        "k_hi": k_hi,
        "n_tracks": int(len(res.env.k)),
        "n_env": int(res.env.x.shape[-1]),
        "iters": int(p["iters"]),
        "k_trust": str(p["k_trust"]),
        "marginal": bool(p.get("marginal", False)),
        "h_aware": bool(p.get("h_aware", False)),
        "adaptive_floor": bool(p.get("adaptive_floor", False)),
        "h_lorentzian": bool(p.get("h_lorentzian", False)),
        "mean_rev_s": round(float(r.mean()), 4),
        "rms_vs_telemetry": round(float(np.sqrt(np.mean((r - r_meas) ** 2))), 5),
        "wall_s": round(wall, 2),
        "objective": obj,
        "per_cell": {t: float(obj[t]) / n_cells for t in terms},
        # The energy shares of the same last solve — the reading the objective
        # is meant to REPLACE, kept beside it so the two can be compared.
        "residual_fraction": last.get("residual_fraction"),
        "track_fraction": last.get("track_fraction"),
        "flatness_whitened_mean": (last.get("flatness") or {}).get("flatness_whitened_mean"),
        "order_cell": last.get("order_cell"),
        "group_plan": plan,
    }


# ---------------------------------------------------------------------------
# units + summary


def build_units(args: argparse.Namespace) -> list[Unit]:
    windows = [w for w in str(args.windows).split(",") if w.strip()]
    hyps = [h for h in str(args.hypotheses).split(",") if h.strip()]
    common = {
        "k_max": int(args.k_max),
        "f_max": float(args.f_max),
        "mics": int(args.mics),
        "seconds": float(args.seconds),
        "iters": int(args.iters),
        "k_trust": str(args.k_trust),
        "arms_dir": str(args.arms_dir),
        "prep_dir": args.prep_dir or None,
        "mem_budget_gb": float(args.mem_budget_gb),
        "marginal": bool(getattr(args, "marginal", False)),
        "h_aware": bool(getattr(args, "h_aware", False)),
        "adaptive_floor": bool(getattr(args, "adaptive_floor", False)),
        "h_lorentzian": bool(getattr(args, "h_lorentzian", False)),
    }
    return [
        Unit(uid=f"{w}__{h}", params={**common, "window": w, "hypothesis": h})
        for w in windows
        for h in hyps
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The window x hypothesis table: every term, raw and per cell.

    Each window also carries its RANKING and the margin of the best hypothesis
    over the telemetry — a negative ``delta_vs_telemetry`` means the audio
    prefers that hypothesis to the tachometer. The ranking is by the per-cell
    PROFILED total, or by the per-cell MARGINAL total when every row carries one
    (``--marginal``), or by the per-cell H-AWARE total when every row carries
    that (``--h-aware``, which wins when both are asked for): profiling pays no
    rent for the envelopes' freedom and charges every hypothesis alike for the
    line flanks, so a hypothesis can win the profiled column by absorption or by
    coverage, and the columns DISAGREEING is the reading.
    """
    table: dict[str, Any] = {}
    marginal = bool(rows) and all(MARGINAL_KEY in (r.get("per_cell") or {}) for r in rows)
    h_aware = bool(rows) and all(H_KEY in (r.get("per_cell") or {}) for r in rows)
    terms = TERMS + (MARGINAL_TERMS if marginal else ()) + (H_TERMS if h_aware else ())
    for r in sorted(rows, key=lambda r: (str(r.get("window")), str(r.get("hypothesis")))):
        if "objective" not in r:
            continue
        cell = table.setdefault(str(r["window"]), {})
        cell[str(r["hypothesis"])] = {
            **{t: float(r["objective"][t]) for t in terms},
            **{f"{t}_per_cell": float(r["per_cell"][t]) for t in terms},
            "n_cells": int(r["objective"]["n_cells"]),
            **({"h_cells": int(r["objective"]["h_cells"])} if h_aware else {}),
            "k_hi": r.get("k_hi"),
            "mean_rev_s": r.get("mean_rev_s"),
            "rms_vs_telemetry": r.get("rms_vs_telemetry"),
            "residual_fraction": r.get("residual_fraction"),
            "wall_s": r.get("wall_s"),
        }
    key = "total_per_cell"
    if marginal:
        key = f"{MARGINAL_KEY}_per_cell"
    if h_aware:
        key = f"{H_KEY}_per_cell"
    for win, cell in table.items():
        order = sorted(cell, key=lambda h: cell[h][key])
        base = cell.get(TELEMETRY, {}).get(key)
        entry = {
            **cell,
            "_ranked_by": key,
            "_ranking": order,
            "_best": order[0] if order else None,
            "_delta_vs_telemetry": (
                None if base is None else {h: cell[h][key] - base for h in cell if h != TELEMETRY}
            ),
            # The cell counts MUST agree inside a window, or the totals are not
            # comparable and the ranking above is meaningless.
            "_cells_agree": len({cell[h]["n_cells"] for h in cell}) == 1,
        }
        if marginal or h_aware:
            # The PROFILED ranking beside the other one. Where they differ, the
            # difference is what the profiled column did not charge for —
            # absorption under ``--marginal``, coverage of the line flanks under
            # ``--h-aware``.
            entry["_ranking_profiled"] = sorted(cell, key=lambda h: cell[h]["total_per_cell"])
        if marginal and h_aware:
            entry["_ranking_marginal"] = sorted(
                cell, key=lambda h: cell[h][f"{MARGINAL_KEY}_per_cell"]
            )
        table[win] = entry
    return {
        "n_units": len(rows),
        "terms": list(terms),
        "marginal": marginal,
        "h_aware": h_aware,
        # The two levers, as metadata: both are properties of the RUN and not of
        # a row, so a summary that does not name them is not reproducible.
        "adaptive_floor": bool(rows) and all(bool(r.get("adaptive_floor")) for r in rows),
        "h_lorentzian": bool(rows) and all(bool(r.get("h_lorentzian")) for r in rows),
        "table": table,
    }


def print_table(summary: dict[str, Any]) -> None:
    """The summary as a fixed-width table on stdout — the thing a human reads."""
    marginal = bool(summary.get("marginal"))
    h_aware = bool(summary.get("h_aware"))
    head = f"\n{'window':<38} {'hypothesis':<11} {'J/cell':>14}"
    if marginal:
        head += f" {'Jmarg/cell':>14}"
    if h_aware:
        head += f" {'Jh/cell':>14}"
    print(head + f" {'data/cell':>10} {'rms':>7}")
    for win, cell in sorted(summary["table"].items()):
        for hyp in cell.get("_ranking", []):
            e = cell[hyp]
            flag = " *" if hyp == cell.get("_best") else "  "
            row = f"{win:<38} {hyp:<11} {e['total_per_cell']:>14.6f}"
            if marginal:
                row += f" {e[f'{MARGINAL_KEY}_per_cell']:>14.6f}"
            if h_aware:
                row += f" {e[f'{H_KEY}_per_cell']:>14.6f}"
            print(f"{row} {e['data_per_cell']:>10.4f} {e['rms_vs_telemetry']:>7.3f}{flag}")
        if (marginal or h_aware) and cell.get("_ranking") != cell.get("_ranking_profiled"):
            print(f"{win:<38} .. profiled order {cell['_ranking_profiled']} — absorption")
        if not cell.get("_cells_agree", True):
            print(f"{win:<38} !! cell counts disagree — the totals are NOT comparable")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="results/joint_rescore")
    ap.add_argument("--windows", default=",".join(DEFAULT_WINDOWS))
    ap.add_argument("--hypotheses", default=",".join(DEFAULT_HYPOTHESES))
    ap.add_argument(
        "--arms-dir",
        default="results/fvk_arms",
        help="where scripts/fvk_arms.py left step 5 — a campaign directory or a --pack file",
    )
    ap.add_argument(
        "--pack",
        default="",
        help="write the selected arms' trajectories to this JSON and exit (the cluster path: "
        "results/ does not travel with a job, a pack file does)",
    )
    ap.add_argument("--k-max", type=int, default=40, help="the F_VK campaign's harmonic cap")
    ap.add_argument("--f-max", type=float, default=F_MAX)
    ap.add_argument("--mics", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = the whole 16 s window")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--k-trust", default=",".join(str(v) for v in DEFAULT_LADDER))
    ap.add_argument("--prep-dir", default="", help="frozen beatvk prep cache")
    ap.add_argument("--build-preps", action="store_true", help="materialize the prep windows first")
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--mem-budget-gb", type=float, default=0.0, help="0 = no guard")
    ap.add_argument(
        "--marginal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "rank by the MARGINAL objective — the profiled one plus the exact Gaussian "
            "envelope marginalization 0.5 (log det M - log det' R). Profiling pays no rent "
            "for the envelopes' freedom, so absorption is free and a coverage fan can win "
            "the profiled column; this charges for it. Both columns are reported"
        ),
    )
    ap.add_argument(
        "--h-aware",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "rank by the H-AWARE objective — the data term with a comb-shaped nuisance "
            "H = max(0, P~ - S) inside the hypothesis's OWN search regions. The coherent "
            "envelopes cannot carry the line FLANKS, so the profiled data term charges every "
            "hypothesis for them alike and a coverage fan loses nothing by missing the humps; "
            "this lets a trajectory EXPLAIN the humps its regions cover. All columns reported"
        ),
    )
    ap.add_argument(
        "--adaptive-floor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "give the floor ONE profiled scale per (channel, frame). S is fitted per 4-second "
            "block and DREGON's rotor wash is not stationary over one, so a gust span pays a "
            "Whittle misfit no comb hypothesis caused — and the block floor plus its rent then "
            "move with however each hypothesis's solve spread that energy. The rent pays "
            "n_freq log gamma per frame, which is the Occam charge for invoking a loud one"
        ),
    )
    ap.add_argument(
        "--h-lorentzian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "constrain the H nuisance to a NON-NEGATIVE Lorentzian mixture at the hypothesis's "
            "OWN line positions, at the measured 0.6 k Hz half width. The shape-free hump "
            "explains any excess inside a region, which discriminates nothing where four dense "
            "combs make every hypothesis's regions blanket the band; a wrong comb's bumps land "
            "between its lines and fit near-zero amplitudes. Needs --h-aware"
        ),
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one window, k-max 12, 2 mics — the under-a-minute path",
    )
    ap.add_argument("--table", action="store_true", help="only re-read raw/ and print the table")
    add_gridrun_args(ap)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.smoke:
        args.windows = DEFAULT_WINDOWS[0]
        args.k_max, args.mics = 12, 2
    if args.pack:
        windows = [w for w in str(args.windows).split(",") if w.strip()]
        hyps = [h for h in str(args.hypotheses).split(",") if h.strip()]
        pack_hypotheses(args.arms_dir, windows, hyps, args.pack)
        print(f"[joint_rescore] {len(windows)} windows x {len(hyps) - 1} arms -> {args.pack}")
        return 0
    if args.table:
        rows = [json.loads(p.read_text()) for p in sorted((out_dir / "raw").glob("*.json"))]
        summary = summarize(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print_table(summary)
        return 0
    if args.build_preps:
        args.prep_dir = str(fvk_arms.build_preps(args))
    units = build_units(args)
    print(f"[joint_rescore] {len(units)} units -> {out_dir}", flush=True)
    result = gridrun_from_args(args, units, worker, out_dir, summarize=summarize)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        print_table(json.loads(summary_path.read_text()))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
