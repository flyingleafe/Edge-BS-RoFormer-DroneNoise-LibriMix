#!/usr/bin/env python3
"""F_VK on the frozen protocol windows: telemetry quality, and the L-BFGS oracle.

Two questions, one driver, both on ``tracking.fitness_vk`` and both over the
frozen ``beatvk`` prep windows (``tracking.protocols.resolve_prep_dir``). It is
a plain ``utils.gridrun`` harness: one unit = one window (x candidate), one JSON
under ``<out>/raw/``, restartable.

``--mode score`` — telemetry quality by F_VK
    Score three candidate trajectories per window with :func:`fvk_score`:

      telem            the window's raw telemetry (``r_meas``)
      fit:<arm>        the phase-6c fitted trajectory of ``--fit-dir`` (entry
                       ``r_fit``), matched to the window by name
      scale:<s>        telemetry times a constant rate scale, DREGON ONLY —
                       the known-bias arm (the 6d ridge profile's -0.683 %)

    Every candidate of a window is scored against the SAME pinned reference
    (the telemetry), so the harmonic cap and therefore the ``(channel, rotor,
    harmonic)`` cell set is identical across candidates — the fixed-degrees-of-
    freedom discipline of ``tracking.fitness_vk``. Reported per window and
    pooled by rig: R^2, residual, objective.

``--mode opt`` — the oracle sanity check
    :func:`optimize_trajectory` from the telemetry init under the default
    ``k_max`` annealing schedule, one unit per window. Reported per window:
    movement rms/max in rev/s, movement as a percent of the mean rate, the
    effective constant-scale component of the movement (one joint least-squares
    scale of refined onto init), F before/after, wall time. The refined
    trajectories are written to ``<out>/refined/<window>.npz``.

    The sanity verdicts this mode exists to state: FLY124's recalibrated labels
    must barely move, and DREGON's must move by a scale-like -0.3..-0.9 %,
    compatible with the 6d ridge profile -0.683 % [-0.877, -0.533]
    (``docs/experiments/telemetry-fitness.md`` § "The re-score").

``--mode report`` — re-aggregate ``<out>/raw/*.json`` into the tables and the
    figure without recomputing anything (the local step after ``omnirun pull``).

Run::

    # smoke: one window, both modes, a truncated schedule
    python scripts/fvk_telemetry.py --mode score --windows FLY124__w02
    python scripts/fvk_telemetry.py --mode opt --windows FLY124__w02 --schedule 5:3

    # step 2, local (needs --fit-dir, a gitignored artifact)
    python scripts/fvk_telemetry.py --mode score --jobs 4

    # step 3, cluster (builds its own prep cache from the pinned dataset)
    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 32 --time 4h \
      --env PYTHONPATH=src -- \
      python scripts/fvk_telemetry.py --mode opt --jobs 15 --build-preps

Outputs: ``<out>/summary.json``, ``<out>/tables/*.csv``, ``<out>/refined/*.npz``
and ``<out>/fig_oracle_sanity.{png,svg}``.
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
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # telemetry_fitness, for --build-preps

from tracking.protocols import resolve_prep_dir  # noqa: E402
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

OUT_DEFAULT = "results/fvk_telemetry"
#: The phase-6c fitted trajectories (arm ``main``) — one ``.npz`` per window.
FIT_DIR_DEFAULT = "results/telemetry_refit/campaign/traj/main"
#: The known-bias arm: the 6d ridge profile's constant scale, DREGON only.
DREGON_SCALE = 0.99317
#: The 6d one-parameter ridge scale profile on DREGON, in percent
#: (``docs/experiments/telemetry-fitness.md`` § "The re-score: the
#: one-parameter scale profile"). The band the step-3 movement must land in.
RIDGE_SCALE_PCT = -0.683
RIDGE_SCALE_CI_PCT = (-0.877, -0.533)

DREGON_WINDOWS = tuple(
    f"{rec}__w{i:02d}"
    for rec in (
        "free-flight_nosource_room1",
        "free-flight_speech-low_room1",
        "free-flight_whitenoise-low_room1",
    )
    for i in range(3)
)
FLY124_WINDOWS = tuple(f"FLY124__w{i:02d}" for i in range(6))
ALL_WINDOWS = DREGON_WINDOWS + FLY124_WINDOWS


def rig_of(key: str) -> str:
    return "fly124" if key.startswith("FLY124") else "dregon"


# ---------------------------------------------------------------------------
# window loading


def load_window(key: str) -> dict[str, Any]:
    """One prep window plus its edge mask.

    ``tracking.protocols.load_prep_window`` is THE reader of the frozen cache;
    it does not return the protocol edge mask, and the movement statistics of
    ``--mode opt`` are reported both on all frames and on the edge-trimmed ones
    (F_VK's own data weight tapers the window ends, so the refined trajectory is
    least constrained exactly where the metric must not be decided).
    """
    import numpy as np

    from tracking.protocols import load_prep_window

    win = load_prep_window(key)
    with np.load(resolve_prep_dir() / f"{key}.npz") as z:
        win["edge"] = np.asarray(z["edge"], dtype=bool)
    return win


def load_fit(key: str, fit_dir: str, ft: Any) -> Any:
    """The fitted trajectory of one window, checked onto the window's grid."""
    import numpy as np

    with np.load(Path(fit_dir) / f"{key}.npz") as z:
        r_fit = np.asarray(z["r_fit"], dtype=np.float64)
        ft_fit = np.asarray(z["ft"], dtype=np.float64)
    if ft_fit.shape != np.asarray(ft).shape or not np.allclose(ft_fit, ft):
        raise ValueError(f"{key}: fitted trajectory is on a different frame grid")
    return r_fit


def build_candidate(spec: str, key: str, win: dict[str, Any], fit_dir: str) -> Any:
    """Materialize one candidate trajectory: ``telem`` | ``fit:<arm>`` | ``scale:<s>``."""
    if spec == "telem":
        return win["r"]
    kind, _, rest = spec.partition(":")
    if kind == "scale":
        return win["r"] * float(rest)
    if kind == "fit":
        return load_fit(key, fit_dir, win["ft"])
    raise ValueError(f"unknown candidate spec {spec!r}")


def make_config(p: dict[str, Any]) -> Any:
    from tracking.fitness_vk import FVKConfig

    return FVKConfig(
        k_min=int(p["k_min"]),
        k_max=int(p["k_max"]),
        bw_rps=float(p["bw_rps"]),
        fs_env=float(p["fs_env"]),
        f_max=float(p["f_max"]),
    )


# ---------------------------------------------------------------------------
# the two workers


def _score_fields(s: dict[str, Any]) -> dict[str, Any]:
    """The scalar half of one :func:`fvk_score` result (the ``k_energy``
    profile is dropped: 40 numbers per unit that no table here reads)."""
    return {
        "residual": s["residual"],
        "data_term": s["data_term"],
        "prior_term": s["prior_term"],
        "r2": s["r2"],
        "objective": s["objective"],
        "energy": s["energy"],
        "k_hi": s["k_hi"],
        "n_cells": s["n_cells"],
        "n_channels": s["n_channels"],
        "n_tracks": s["n_tracks"],
    }


def score_worker(unit: Unit) -> dict[str, Any]:
    """One (window, candidate) unit of ``--mode score``."""
    import time

    import numpy as np

    from tracking.fitness_vk import fvk_score

    p = dict(unit.params)
    key, spec = str(p["key"]), str(p["candidate"])
    win = load_window(key)
    cand = build_candidate(spec, key, win, str(p["fit_dir"]))
    cfg = make_config(p)
    tic = time.perf_counter()
    # The REFERENCE is the telemetry for every candidate, so the harmonic cap
    # and the cell set are the window's and not the candidate's.
    s = fvk_score(win["audio"], 16000.0, cand, win["ft"], cfg, reference=win["r"])
    return {
        "mode": "score",
        "key": key,
        "rig": rig_of(key),
        "recording": key.split("__")[0],
        "regime": win["regime"],
        "candidate": spec,
        "mean_rev_s": float(np.mean(win["r"])),
        "wall_s": round(time.perf_counter() - tic, 2),
        **_score_fields(s),
    }


def _movement(refined: Any, init: Any, mask: Any = None) -> dict[str, Any]:
    """Movement of a refined trajectory away from its init.

    ``scale_pct`` is the effective CONSTANT-scale component: the single joint
    least-squares factor ``s = <refined, init> / <init, init>``, reported as
    ``(s - 1) * 100``. ``resid_rms`` is what the scale does not explain, which
    is the number that says whether a constant is the right model at all.
    """
    import numpy as np

    a = np.asarray(init, dtype=np.float64)
    b = np.asarray(refined, dtype=np.float64)
    if mask is not None:
        a, b = a[:, mask], b[:, mask]
    d = b - a
    s = float((b * a).sum() / max(float((a * a).sum()), 1e-30))
    per_rotor = [
        float((b[i] * a[i]).sum() / max(float((a[i] * a[i]).sum()), 1e-30))
        for i in range(a.shape[0])
    ]
    mean_rate = float(np.mean(np.abs(a)))
    return {
        "move_rms": float(np.sqrt((d**2).mean())),
        "move_max": float(np.abs(d).max()),
        "move_mean": float(d.mean()),
        "move_pct_of_rate": float(100.0 * np.sqrt((d**2).mean()) / max(mean_rate, 1e-30)),
        "scale_pct": 100.0 * (s - 1.0),
        "scale_pct_per_rotor": [round(100.0 * (v - 1.0), 4) for v in per_rotor],
        "resid_rms": float(np.sqrt(((b - s * a) ** 2).mean())),
        "mean_rev_s": mean_rate,
    }


def opt_worker(unit: Unit) -> dict[str, Any]:
    """One window of ``--mode opt``: L-BFGS on F_VK from the telemetry init."""
    import time

    import numpy as np

    from tracking.fitness_vk import FVKStage, fvk_score, optimize_trajectory

    p = dict(unit.params)
    key = str(p["key"])
    win = load_window(key)
    cfg = make_config(p)
    sched = p.get("schedule")
    schedule = None if not sched else tuple(FVKStage(int(k), 1.0, int(n)) for k, n in sched)

    before = fvk_score(win["audio"], 16000.0, win["r"], win["ft"], cfg, reference=win["r"])
    tic = time.perf_counter()
    r_ref, diag = optimize_trajectory(
        win["audio"],
        16000.0,
        win["r"],
        win["ft"],
        cfg,
        schedule=schedule,
        reference=win["r"],
    )
    wall = time.perf_counter() - tic
    after = fvk_score(win["audio"], 16000.0, r_ref, win["ft"], cfg, reference=win["r"])

    out_dir = Path(str(p["out"])) / "refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{key}.npz",
        allow_pickle=False,
        ft=win["ft"],
        r_init=win["r"],
        r_refined=r_ref,
        edge=win["edge"],
    )
    return {
        "mode": "opt",
        "key": key,
        "rig": rig_of(key),
        "recording": key.split("__")[0],
        "regime": win["regime"],
        "wall_s": round(wall, 2),
        "n_frames": int(win["ft"].size),
        "full": _movement(r_ref, win["r"]),
        "edge": _movement(r_ref, win["r"], win["edge"]),
        "before": _score_fields(before),
        "after": _score_fields(after),
        "d_objective": after["objective"] - before["objective"],
        "d_r2": after["r2"] - before["r2"],
        "diagnostics": diag,
    }


def worker(unit: Unit) -> dict[str, Any]:
    """Dispatch on the unit's mode (one grid may carry both)."""
    return score_worker(unit) if unit.params["mode"] == "score" else opt_worker(unit)


# ---------------------------------------------------------------------------
# report


def _mean(vals: list[Any]) -> float | None:
    import numpy as np

    v = np.asarray([x for x in vals if isinstance(x, (int, float))], dtype=np.float64)
    v = v[np.isfinite(v)]
    return round(float(v.mean()), 6) if v.size else None


def _pool_key(row: dict[str, Any]) -> str:
    """The pooling unit: rig, with FLY124 split by regime (its warmup windows
    are out of validity in the 6d re-score and must not silently join cruise)."""
    return row["rig"] if row["rig"] == "dregon" else f"fly124_{row['regime']}"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled step-2 table (by rig x candidate) and step-3 table (by rig)."""
    score_rows = [r for r in rows if r.get("mode") == "score"]
    opt_rows = [r for r in rows if r.get("mode") == "opt"]

    score_pooled: dict[str, Any] = {}
    for pool in sorted({_pool_key(r) for r in score_rows}):
        sel = [r for r in score_rows if _pool_key(r) == pool]
        for cand in sorted({r["candidate"] for r in sel}):
            got = [r for r in sel if r["candidate"] == cand]
            score_pooled[f"{pool}|{cand}"] = {
                "n_windows": len(got),
                "r2": _mean([r["r2"] for r in got]),
                "residual": _mean([r["residual"] for r in got]),
                "objective": _mean([r["objective"] for r in got]),
                "k_hi": _mean([r["k_hi"] for r in got]),
                "n_cells": _mean([r["n_cells"] for r in got]),
            }

    opt_pooled: dict[str, Any] = {}
    for pool in sorted({_pool_key(r) for r in opt_rows}):
        got = [r for r in opt_rows if _pool_key(r) == pool]
        opt_pooled[pool] = {
            "n_windows": len(got),
            "move_rms": _mean([r["edge"]["move_rms"] for r in got]),
            "move_max": _mean([r["edge"]["move_max"] for r in got]),
            "move_pct_of_rate": _mean([r["edge"]["move_pct_of_rate"] for r in got]),
            "scale_pct": _mean([r["edge"]["scale_pct"] for r in got]),
            "scale_pct_full": _mean([r["full"]["scale_pct"] for r in got]),
            "resid_rms": _mean([r["edge"]["resid_rms"] for r in got]),
            "objective_before": _mean([r["before"]["objective"] for r in got]),
            "objective_after": _mean([r["after"]["objective"] for r in got]),
            "r2_before": _mean([r["before"]["r2"] for r in got]),
            "r2_after": _mean([r["after"]["r2"] for r in got]),
            "wall_s": _mean([r["wall_s"] for r in got]),
        }
    return {
        "ridge_scale_pct_6d": {"point": RIDGE_SCALE_PCT, "ci": list(RIDGE_SCALE_CI_PCT)},
        "score_pooled": score_pooled,
        "opt_pooled": opt_pooled,
        "n_score_units": len(score_rows),
        "n_opt_units": len(opt_rows),
    }


def read_rows(out: Path) -> list[dict[str, Any]]:
    raw = out / "raw"
    return [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))] if raw.is_dir() else []


def write_tables(out: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """The four CSVs: per-window and pooled, one pair per mode."""
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, fields: list[str], recs: list[dict[str, Any]]) -> None:
        if not recs:
            return
        with open(tdir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(recs)
        print(f"[tables] {tdir / name} ({len(recs)} rows)")

    dump(
        "step2_per_window.csv",
        ["key", "rig", "regime", "candidate", "r2", "residual", "objective", "k_hi", "n_cells"],
        sorted(
            (r for r in rows if r.get("mode") == "score"),
            key=lambda r: (r["key"], r["candidate"]),
        ),
    )
    dump(
        "step2_pooled.csv",
        ["pool", "candidate", "n_windows", "r2", "residual", "objective", "k_hi", "n_cells"],
        [
            {"pool": tag.split("|")[0], "candidate": tag.split("|")[1], **vals}
            for tag, vals in summary.get("score_pooled", {}).items()
        ],
    )
    dump(
        "step3_per_window.csv",
        [
            "key",
            "rig",
            "regime",
            "mean_rev_s",
            "move_rms",
            "move_max",
            "move_pct_of_rate",
            "scale_pct",
            "scale_pct_full",
            "resid_rms",
            "objective_before",
            "objective_after",
            "r2_before",
            "r2_after",
            "wall_s",
        ],
        [
            {
                "key": r["key"],
                "rig": r["rig"],
                "regime": r["regime"],
                "mean_rev_s": round(r["edge"]["mean_rev_s"], 3),
                "move_rms": round(r["edge"]["move_rms"], 4),
                "move_max": round(r["edge"]["move_max"], 4),
                "move_pct_of_rate": round(r["edge"]["move_pct_of_rate"], 4),
                "scale_pct": round(r["edge"]["scale_pct"], 4),
                "scale_pct_full": round(r["full"]["scale_pct"], 4),
                "resid_rms": round(r["edge"]["resid_rms"], 4),
                "objective_before": round(r["before"]["objective"], 6),
                "objective_after": round(r["after"]["objective"], 6),
                "r2_before": round(r["before"]["r2"], 6),
                "r2_after": round(r["after"]["r2"], 6),
                "wall_s": r["wall_s"],
            }
            for r in sorted((r for r in rows if r.get("mode") == "opt"), key=lambda r: r["key"])
        ],
    )
    dump(
        "step3_pooled.csv",
        [
            "pool",
            "n_windows",
            "move_rms",
            "move_max",
            "move_pct_of_rate",
            "scale_pct",
            "resid_rms",
            "objective_before",
            "objective_after",
            "r2_before",
            "r2_after",
            "wall_s",
        ],
        [{"pool": pool, **vals} for pool, vals in summary.get("opt_pooled", {}).items()],
    )


def print_tables(summary: dict[str, Any]) -> None:
    sp = summary.get("score_pooled", {})
    if sp:
        print("\n=== step 2: F_VK by rig x candidate (pooled, higher R^2 / lower objective better)")
        print(
            f"{'pool':16s} {'candidate':16s} {'n':>3s} {'R^2':>9s} {'residual':>12s} {'objective':>11s}"
        )
        for tag in sorted(sp):
            pool, cand = tag.split("|")
            v = sp[tag]
            print(
                f"{pool:16s} {cand:16s} {v['n_windows']:3d} "
                f"{v['r2']:9.4f} {v['residual']:12.1f} {v['objective']:11.5f}"
            )
    op = summary.get("opt_pooled", {})
    if op:
        lo, hi = RIDGE_SCALE_CI_PCT
        print(
            f"\n=== step 3: L-BFGS from telemetry (6d ridge scale {RIDGE_SCALE_PCT} % [{lo}, {hi}])"
        )
        print(
            f"{'pool':16s} {'n':>3s} {'move rms':>9s} {'move max':>9s} {'move %':>8s} "
            f"{'scale %':>9s} {'resid rms':>9s} {'F before':>9s} {'F after':>9s} {'wall s':>8s}"
        )
        for pool, v in op.items():
            print(
                f"{pool:16s} {v['n_windows']:3d} {v['move_rms']:9.4f} {v['move_max']:9.4f} "
                f"{v['move_pct_of_rate']:8.3f} {v['scale_pct']:9.4f} {v['resid_rms']:9.4f} "
                f"{v['objective_before']:9.5f} {v['objective_after']:9.5f} {v['wall_s']:8.1f}"
            )


# ---------------------------------------------------------------------------
# the figure


def make_figure(out: Path, rows: list[dict[str, Any]]) -> None:
    """Slide 18: does the oracle move the right rig?

    Per-window constant-scale component of the L-BFGS movement, by rig, with
    the 6d ridge-profile band overlaid. A DREGON window whose label error is a
    scale must land in the band; a FLY124 window whose labels are already
    calibrated must land on zero.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    opt = sorted((r for r in rows if r.get("mode") == "opt"), key=lambda r: (r["rig"], r["key"]))
    if not opt:
        print("[figure] no opt units — skipped")
        return

    ink = "#222222"
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.edgecolor": "#888888",
            "axes.labelcolor": ink,
            "text.color": ink,
            "xtick.color": ink,
            "ytick.color": ink,
        }
    )
    color = {"dregon": "#1f77b4", "fly124": "#d62728"}
    label = {"dregon": "DREGON (raw telemetry)", "fly124": "FLY124 (recalibrated labels)"}

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    lo, hi = RIDGE_SCALE_CI_PCT
    ax.axhspan(lo, hi, color="#2ca02c", alpha=0.16, zorder=0)
    ax.axhline(RIDGE_SCALE_PCT, color="#2ca02c", ls="--", lw=1.6, zorder=1)
    ax.axhline(0.0, color="#888888", lw=1.0, zorder=1)

    x = np.arange(len(opt))
    seen: set[str] = set()
    for xi, r in zip(x, opt, strict=True):
        rig = r["rig"]
        warm = r["regime"] != "cruise"
        ax.plot(
            [xi],
            [r["edge"]["scale_pct"]],
            marker="s" if warm else "o",
            ms=9,
            mfc="none" if warm else color[rig],
            mec=color[rig],
            mew=2.0,
            ls="none",
            zorder=3,
            label=None if rig in seen else label[rig],
        )
        seen.add(rig)
    ax.plot(
        [],
        [],
        marker="s",
        ms=9,
        mfc="none",
        mec="#888888",
        mew=2.0,
        ls="none",
        label="warmup window",
    )

    ax.text(
        0.995,
        RIDGE_SCALE_PCT,
        f"6d ridge profile {RIDGE_SCALE_PCT:+.3f} % [{lo:+.3f}, {hi:+.3f}]",
        color="#2ca02c",
        fontsize=10,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["key"].replace("free-flight_", "").replace("_room1", "") for r in opt],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("constant-scale component of the\nL-BFGS movement (%)")
    ax.set_title("The oracle moves DREGON's labels by a scale and leaves FLY124's alone")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = out / f"fig_oracle_sanity.{ext}"
        fig.savefig(p, dpi=150)
        print(f"[figure] {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI


def parse_schedule(text: str) -> list[list[int]] | None:
    """``"5:3,10:5"`` -> ``[[5, 3], [10, 5]]`` (``""`` = the default schedule)."""
    if not text.strip():
        return None
    out = []
    for part in text.split(","):
        k, _, n = part.partition(":")
        out.append([int(k), int(n or 20)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", default="score", choices=("score", "opt", "both", "report"))
    ap.add_argument("--windows", default="", help="comma-separated keys (default: all 15)")
    ap.add_argument("--rig", default="all", choices=("dregon", "fly124", "all"))
    ap.add_argument(
        "--candidates",
        default="",
        help="score mode: comma-separated specs; default telem,fit:main(+scale on DREGON)",
    )
    ap.add_argument("--fit-dir", default=FIT_DIR_DEFAULT)
    ap.add_argument("--scale", type=float, default=DREGON_SCALE, help="the DREGON known-bias arm")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--bw-rps", type=float, default=1.0, help="THE basin knob (rev/s of capture)")
    ap.add_argument("--fs-env", type=float, default=100.0)
    ap.add_argument("--f-max", type=float, default=6000.0)
    ap.add_argument(
        "--schedule",
        default="",
        help="opt mode: 'k:iters,...' (default: fitness_vk.DEFAULT_SCHEDULE)",
    )
    ap.add_argument(
        "--build-preps",
        action="store_true",
        help="materialize the protocol windows first (REQUIRED on a cluster: the "
        "prep cache is a gitignored artifact, so a fresh worktree has no windows)",
    )
    ap.add_argument("--out", default=OUT_DEFAULT)
    add_gridrun_args(ap, jobs=4)
    args = ap.parse_args()

    out = Path(args.out)
    if args.mode == "report":
        rows = read_rows(out)
        summary = summarize(rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        write_tables(out, rows, summary)
        print_tables(summary)
        make_figure(out, rows)
        return

    keys = (
        [k.strip() for k in args.windows.split(",") if k.strip()]
        if args.windows
        else list(
            {"dregon": DREGON_WINDOWS, "fly124": FLY124_WINDOWS, "all": ALL_WINDOWS}[args.rig]
        )
    )
    for bad in [k for k in keys if k not in ALL_WINDOWS]:
        ap.error(f"unknown window {bad!r}; known: {', '.join(ALL_WINDOWS)}")
    if args.build_preps:
        from telemetry_fitness import build_preps

        build_preps(sorted(keys), resolve_prep_dir())

    common = {
        "k_min": args.k_min,
        "k_max": args.k_max,
        "bw_rps": args.bw_rps,
        "fs_env": args.fs_env,
        "f_max": args.f_max,
        "fit_dir": args.fit_dir,
        "out": str(out),
    }
    units: list[Unit] = []
    if args.mode in ("score", "both"):
        for key in sorted(keys):
            cands = (
                [c.strip() for c in args.candidates.split(",") if c.strip()]
                if args.candidates
                else ["telem", "fit:main"]
                + ([f"scale:{args.scale}"] if rig_of(key) == "dregon" else [])
            )
            units += [
                Unit(
                    f"score__{key}__{c.replace(':', '-')}",
                    {"mode": "score", "key": key, "candidate": c, **common},
                )
                for c in cands
            ]
    if args.mode in ("opt", "both"):
        sched = parse_schedule(args.schedule)
        units += [
            Unit(f"opt__{key}", {"mode": "opt", "key": key, "schedule": sched, **common})
            for key in sorted(keys)
        ]

    print(f"[fvk_telemetry] {len(units)} units -> {out}", flush=True)
    res = gridrun_from_args(args, units, worker, out, summarize=summarize)
    print_tables(res.summary)
    write_tables(out, read_rows(out), res.summary)
    raise SystemExit(res.exit_code)


if __name__ == "__main__":
    main()
