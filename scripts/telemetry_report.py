#!/usr/bin/env python3
"""Read the issue-17 campaign (phase 6c) and print the tables it must answer.

Input is what ``scripts/telemetry_campaign.sh`` writes — three ``gridrun``
output directories — and nothing else. Every number here is a re-aggregation of
those unit JSONs, so the report is reproducible from the pulled results without
touching audio.

The sections, in the order the issue asks for them:

  provenance  the window fingerprints (a rebuilt prep cache must be the SAME
              windows), the arms, coverage
  scale       the headline: the fitter's per-window scale, pooled per recording
              and per dataset, CI by resampling WINDOWS
  pp          the independent scale readout — the pulse-pair centre of the
              fitness harness is a rate error in rev/s, so ``100 pp_dr / rate``
              is a scale that comes with the harness's own mic/harmonic/block
              bootstrap, plus its off-comb and mismatch nulls
  profile     the ONE-PARAMETER family ``lp:5+scale:s``: the minimum of the
              pooled curve is a scale estimate at fixed degrees of freedom, and
              it is computed once per hold-out family, which is exactly the
              "harmonics vs channels vs blocks must agree" test
  controls    on / offcomb / mismatch / permute for every candidate
  ablation    what each of the six steps bought
  residual    systematic vs tachometer-signature parts, reported SEPARATELY
  identity    rotor order and the twin gaps, settled on the residual pairing

Usage:
  python scripts/telemetry_report.py --out results/telemetry_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DREGON = "dregon"
FLY = "fly124"
#: ``ridge`` (phase 6d) is the one component where MORE is better; the profile
#: and every table below flip for it via ``HIGHER_IS_BETTER``.
COMPONENTS = ("broadband", "phase_noise", "roughness", "ridge")
HIGHER_IS_BETTER = frozenset({"ridge"})


# ---------------------------------------------------------------------------
# loading


def load_units(out_dir: Path) -> list[dict[str, Any]]:
    """Every unit JSON of a gridrun output directory (errors are counted, not read)."""
    raw = out_dir / "raw"
    rows = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))]
    errs = sorted(raw.glob("*.err"))
    if errs:
        print(f"!! {out_dir}: {len(errs)} failed units, e.g. {errs[0].name}", file=sys.stderr)
    return rows


def dataset_of(row: dict[str, Any]) -> str:
    return FLY if str(row["key"]).startswith("FLY124") else DREGON


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(tuple(_field(r, k) for k in keys), []).append(r)
    return out


def short_candidate(spec: str) -> str:
    """``file:<out>/traj/<arm>/{key}.npz:r_fit`` -> ``fit:<arm>``; others unchanged."""
    return f"fit:{spec.split('/traj/')[1].split('/')[0]}" if "/traj/" in spec else spec


def _field(row: dict[str, Any], name: str) -> Any:
    if name == "dataset":
        return dataset_of(row)
    if name == "candidate":
        return short_candidate(str(row.get("candidate", "")))
    if name == "group":  # dregon | fly124-cruise | fly124-warmup
        ds = dataset_of(row)
        return ds if ds == DREGON else f"{ds}-{row.get('regime', '?')}"
    return row.get(name)


# ---------------------------------------------------------------------------
# uncertainty


def boot_ci(
    values: list[float], n_boot: int = 4000, seed: int = 0
) -> dict[str, float | int | None]:
    """Percentile CI of the mean, resampling the UNITS (windows) with replacement.

    The harness's own bootstrap resamples microphones, harmonics and time
    blocks INSIDE a window. This one resamples windows, which is the axis a
    per-window estimate such as the fitter's scale varies along. The two are
    reported side by side; neither replaces the other.
    """
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    if a.size == 1:
        return {"n": 1, "mean": round(float(a[0]), 5), "sd": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 5),
        "sd": round(float(a.std(ddof=1)), 5),
        "lo": round(float(np.percentile(draws, 2.5)), 5),
        "hi": round(float(np.percentile(draws, 97.5)), 5),
    }


def fmt_ci(ci: dict[str, Any], nd: int = 3) -> str:
    if ci.get("mean") is None:
        return "—"
    if ci.get("lo") is None:
        return f"{ci['mean']:.{nd}f}"
    return f"{ci['mean']:+.{nd}f} [{ci['lo']:+.{nd}f},{ci['hi']:+.{nd}f}]"


# ---------------------------------------------------------------------------
# section: scale from the fitter


def scale_section(refit: list[dict[str, Any]], arm: str = "main") -> dict[str, Any]:
    """Per-window ``global_pct`` pooled by group, plus the per-rotor readings."""
    out: dict[str, Any] = {"arm": arm, "per_window": {}, "pooled": {}, "per_recording": {}}
    sel = [r for r in refit if r.get("arm") == arm]
    for r in sorted(sel, key=lambda r: str(r["key"])):
        out["per_window"][r["key"]] = {
            "group": _field(r, "group"),
            "global_pct": r["scale"]["global_pct"],
            "per_rotor_pct": r["scale"]["per_rotor_pct"],
            "d_mean": r["scale"]["d_mean"],
            "d_rms": r["scale"]["d_rms"],
            "mean_rate": r["scale"]["mean_rate"],
            "stop_reason": r.get("stop_reason"),
            "k_ladder": r.get("k_ladder"),
        }
    for name, rows in group(sel, "group").items():
        out["pooled"][name[0]] = {
            "global": boot_ci([r["scale"]["global_pct"] for r in rows]),
            "per_rotor": boot_ci(
                [v for r in rows for v in r["scale"]["per_rotor_pct"] if v is not None]
            ),
            "by_rotor": [
                boot_ci([r["scale"]["per_rotor_pct"][i] for r in rows])
                for i in range(len(rows[0]["scale"]["per_rotor_pct"]))
            ],
        }
    for name, rows in group(sel, "recording").items():
        out["per_recording"][name[0]] = boot_ci([r["scale"]["global_pct"] for r in rows])
    return out


# ---------------------------------------------------------------------------
# section: the pulse-pair scale (the harness's own rate readout)


def pp_section(fit: list[dict[str, Any]], holdout: str = "none") -> dict[str, Any]:
    """``100 pp_dr / rate`` per (window, rotor), pooled, per candidate x control.

    ``pp_dr`` is the coherent pulse-pair centre of the demodulated envelope: the
    rate correction that would take the CANDIDATE onto the observed line, in
    rev/s (phase 6a verified it against an injected -0.40 rev/s). So it is a
    scale estimate that never sees the fitter, and the same unit carries the
    off-comb and mismatch nulls of that estimate.
    """
    out: dict[str, Any] = {}
    for (grp, cand, ctl), rows in group(fit, "group", "candidate", "control").items():
        per_window: list[float] = []
        per_unit: list[float] = []
        boot_lo, boot_hi = [], []
        for r in rows:
            sc = r["scores"].get(holdout)
            if not sc:
                continue
            rates = r["rotor_mean_rev_s"]
            vals = [
                100.0 * pr["pp_dr"] / rates[int(i)]
                for i, pr in sc["per_rotor"].items()
                if pr.get("pp_dr") is not None and pr["n_cells"] > 0 and rates[int(i)] > 5
            ]
            per_unit.extend(vals)
            if sc.get("pp_dr") is not None:
                per_window.append(100.0 * sc["pp_dr"] / float(np.mean(rates)))
            b = (r.get("bootstrap") or {}).get(holdout, {}).get("pp_dr")
            if b:
                per_rate = float(np.mean(rates))
                boot_lo.append(100.0 * b["lo"] / per_rate)
                boot_hi.append(100.0 * b["hi"] / per_rate)
        out[f"{grp}|{cand}|{ctl}"] = {
            "window_pct": boot_ci(per_window),
            "rotor_pct": boot_ci(per_unit),
            "within_window_ci_mean": [
                round(float(np.mean(boot_lo)), 4) if boot_lo else None,
                round(float(np.mean(boot_hi)), 4) if boot_hi else None,
            ],
        }
    return out


# ---------------------------------------------------------------------------
# section: the one-parameter scale profile


def _scale_of(spec: str) -> float | None:
    for part in spec.split("+"):
        if part.startswith("scale:"):
            return float(part.split(":", 1)[1])
    return None


def _argmin_parabola(s: np.ndarray, y: np.ndarray, depth: float = 0.35) -> tuple[float | None, str]:
    """Sub-grid minimum of a profile curve.

    A parabola through the three points around the discrete minimum is what one
    reaches for first, and it is too fragile here: the grid step is 0.05 % while
    the curve's own noise moves the discrete minimum by a step or two, so the
    three-point fit inherits that noise in full. Instead the parabola is fitted
    by least squares over every point in the BASIN — the points whose value is
    within ``depth`` of the way from the minimum to the maximum — which uses the
    whole shape of the curve and averages the noise down.
    """
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return None, "too few points"
    s, y = s[ok], y[ok]
    i = int(np.argmin(y))
    if i == 0 or i == s.size - 1:
        return float(s[i]), "at grid edge — unresolved"
    thr = y[i] + depth * (float(np.max(y)) - y[i])
    sel = y <= thr
    # keep the contiguous run containing the minimum, so a second basin
    # elsewhere (an alias ridge) cannot drag the fit
    lo = hi = i
    while lo > 0 and sel[lo - 1]:
        lo -= 1
    while hi < s.size - 1 and sel[hi + 1]:
        hi += 1
    if hi - lo + 1 < 3:
        lo, hi = max(i - 1, 0), min(i + 1, s.size - 1)
    a, b, _c = np.polyfit(s[lo : hi + 1], y[lo : hi + 1], 2)
    if a <= 0:
        return float(s[i]), "non-convex"
    smin = float(-b / (2.0 * a))
    if not (s[0] <= smin <= s[-1]):
        return float(s[i]), "vertex outside the grid"
    return smin, "ok"


def profile_section(rows: list[dict[str, Any]], component: str = "phase_noise") -> dict[str, Any]:
    """Pooled component vs the scale parameter, per group x control x hold-out.

    A component where more is better is profiled through its NEGATION, so the
    one extremum finder (and its basin logic, and its bootstrap) serves both
    directions. ``curves`` always reports the component's own sign.
    """
    sgn = -1.0 if component in HIGHER_IS_BETTER else 1.0
    out: dict[str, Any] = {"component": component, "curves": {}, "minima": {}}
    holdouts = sorted({h for r in rows for h in r.get("scores", {})})
    for (grp, ctl), got in group(rows, "group", "control").items():
        by_key: dict[str, list[dict[str, Any]]] = {}
        for r in got:
            by_key.setdefault(str(r["key"]), []).append(r)
        for ho in holdouts:
            # per window: the curve over s; then the mean curve over windows
            curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for key, rs in by_key.items():
                pts = sorted(
                    (_scale_of(str(r["candidate"])), (r["scores"].get(ho) or {}).get(component))
                    for r in rs
                )
                s = np.asarray([p[0] for p in pts], dtype=np.float64)
                y = sgn * np.asarray(
                    [np.nan if p[1] is None else p[1] for p in pts], dtype=np.float64
                )
                curves[key] = (s, y)
            if not curves:
                continue
            s0 = next(iter(curves.values()))[0]
            mat = np.vstack([y for _s, y in curves.values()])
            mean = np.nanmean(mat, axis=0)
            smin, note = _argmin_parabola(s0, mean)
            # CI over WINDOWS: resample the per-window curves
            rng = np.random.default_rng(0)
            draws = []
            n = mat.shape[0]
            for _ in range(2000 if n > 1 else 0):
                idx = rng.integers(0, n, size=n)
                sm, nt = _argmin_parabola(s0, np.nanmean(mat[idx], axis=0))
                if sm is not None and nt == "ok":
                    draws.append(100.0 * (sm - 1.0))
            tag = f"{grp}|{ctl}|{ho}"
            out["curves"][tag] = {
                "s": [round(float(v), 5) for v in s0],
                "mean": [None if not np.isfinite(v) else round(float(sgn * v), 6) for v in mean],
                "n_windows": int(mat.shape[0]),
            }
            out["minima"][tag] = {
                "scale_pct": round(100.0 * (smin - 1.0), 4) if smin is not None else None,
                "note": note,
                # Depth of the basin, absolute and relative. A component that
                # crosses zero (the ridge, in dB) has no meaningful relative
                # drop, so the absolute one is what the null is compared on.
                "depth": round(float(np.nanmax(mean) - np.nanmin(mean)), 5),
                "curvature_drop": (
                    round(
                        float(np.nanmax(mean) - np.nanmin(mean)) / float(np.nanmin(mean)),
                        4,
                    )
                    if np.isfinite(np.nanmin(mean)) and float(np.nanmin(mean)) > 1e-6
                    else None
                ),
                "ci": {
                    "lo": round(float(np.percentile(draws, 2.5)), 4) if len(draws) > 50 else None,
                    "hi": round(float(np.percentile(draws, 97.5)), 4) if len(draws) > 50 else None,
                    "n_draws": len(draws),
                },
                "per_window_pct": {
                    key: (
                        round(100.0 * (m - 1.0), 4)
                        if (m := _argmin_parabola(s, y)[0]) is not None
                        else None
                    )
                    for key, (s, y) in curves.items()
                },
            }
    return out


# ---------------------------------------------------------------------------
# section: controls, ablation, residual, identity


def controls_section(fit: list[dict[str, Any]], holdout: str = "none") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (grp, cand, ctl), rows in group(fit, "group", "candidate", "control").items():
        sc = [r["scores"][holdout] for r in rows if holdout in r.get("scores", {})]
        res = [r["residual"]["pooled"] for r in rows if (r.get("residual") or {}).get("pooled")]
        out[f"{grp}|{cand}|{ctl}"] = {
            "n": len(rows),
            **{c: _mean([x.get(c) for x in sc]) for c in COMPONENTS},
            "pp_dr": _mean([x.get("pp_dr") for x in sc]),
            "pp_abs": _mean([x.get("pp_abs") for x in sc]),
            "n_cells": _mean([x.get("n_cells") for x in sc]),
            "n_cells_ridge": _mean([x.get("n_cells_ridge") for x in sc]),
            "admit_frac": _mean([r["cells"]["admit_frac"] for r in rows]),
            "admit_frac_ridge": _mean([r["cells"].get("admit_frac_ridge") for r in rows]),
            # The share of the comb's line energy the CONDITIONING gate can
            # see. Phase 6c reported components read on 6.6 % of the cells; this
            # says what fraction of the thing being measured that was.
            "line_share_gated": _mean([r["cells"].get("line_share_gated") for r in rows]),
            "resid_d_rms": _mean([x.get("d_rms") for x in res]),
        }
    return out


def holdout_section(fit: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    """Every hold-out family's components, for the agreement check."""
    out: dict[str, Any] = {}
    for (grp, cand, ctl), rows in group(fit, "group", "candidate", "control").items():
        if ctl != "on" or cand not in candidates:
            continue
        hos = sorted({h for r in rows for h in r.get("scores", {})})
        out[f"{grp}|{cand}"] = {
            ho: {
                c: _mean([r["scores"][ho].get(c) for r in rows if ho in r["scores"]])
                for c in (*COMPONENTS, "pp_dr")
            }
            for ho in hos
        }
    return out


def ablation_section(refit: list[dict[str, Any]], fit: list[dict[str, Any]]) -> dict[str, Any]:
    """One row per (group, arm): what the arm's fit moved, and how well it fits."""
    out: dict[str, Any] = {}
    fit_by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in fit:
        spec = str(r["candidate"])
        if r["control"] != "on" or "/traj/" not in spec:
            continue
        fit_by_arm.setdefault(spec.split("/traj/")[1].split("/")[0], []).append(r)
    for (grp, arm), rows in group(refit, "group", "arm").items():
        f = [r for r in fit_by_arm.get(str(arm), []) if _field(r, "group") == grp]
        sc = [r["scores"]["none"] for r in f if "none" in r.get("scores", {})]
        out[f"{grp}|{arm}"] = {
            "n": len(rows),
            "global_pct": boot_ci([r["scale"]["global_pct"] for r in rows]),
            "d_rms": _mean([v for r in rows for v in r["scale"]["d_rms"]]),
            "order_kept": sum(bool(r["identity"]["order_kept"]) for r in rows),
            "gap_ratio_min": _min([v for r in rows for v in (r["identity"]["gap_ratio"] or [])]),
            "n_iters": _mean([r["n_iters"] for r in rows]),
            "converged": sum(bool(r["converged"]) for r in rows),
            **{c: _mean([x.get(c) for x in sc]) for c in COMPONENTS},
            "pp_dr": _mean([x.get("pp_dr") for x in sc]),
        }
    return out


def residual_section(refit: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "d_mean",
        "d_rms",
        "resid_rms",
        "scale_pct",
        "lag_s",
        "design_cond",
        "tach_bound_frac",
        "tach_flatness",
        "tach_line_ratio",
    )
    out: dict[str, Any] = {}
    for (grp, arm), rows in group(refit, "group", "arm").items():
        pooled = [r["residual"]["pooled"] for r in rows]
        out[f"{grp}|{arm}"] = {
            "n": len(rows),
            **{k: _mean([p.get(k) for p in pooled]) for k in keys},
            "f_tach_resolved": bool(pooled[0].get("f_tach_resolved")),
            "fs_frame_hz": pooled[0].get("fs_frame_hz"),
        }
    return out


def identity_section(refit: list[dict[str, Any]], fit: list[dict[str, Any]]) -> dict[str, Any]:
    """Order, gaps, and the permutation null of the residual pairing."""
    out: dict[str, Any] = {"per_window": {}, "permutation": {}}
    for r in refit:
        out["per_window"][f"{r['key']}|{r['arm']}"] = {
            "order_kept": r["identity"]["order_kept"],
            "gaps_raw": r["identity"]["gaps_raw"],
            "gaps_fit": r["identity"]["gaps_fit"],
            "gap_ratio": r["identity"]["gap_ratio"],
        }
    for (grp, cand, ctl), rows in group(fit, "group", "candidate", "control").items():
        res = [r["residual"]["pooled"] for r in rows if (r.get("residual") or {}).get("pooled")]
        out["permutation"].setdefault(f"{grp}|{cand}", {})[ctl] = _mean(
            [p.get("d_rms") for p in res]
        )
    return out


def _mean(vals: list[Any]) -> float | None:
    a = np.asarray([v for v in vals if isinstance(v, (int, float))], dtype=np.float64)
    a = a[np.isfinite(a)]
    return round(float(a.mean()), 6) if a.size else None


def _min(vals: list[Any]) -> float | None:
    a = np.asarray([v for v in vals if isinstance(v, (int, float))], dtype=np.float64)
    a = a[np.isfinite(a)]
    return round(float(a.min()), 4) if a.size else None


# ---------------------------------------------------------------------------
# printing


def show(report: dict[str, Any]) -> None:
    p = print
    p("\n" + "=" * 78)
    p("PROVENANCE")
    p("=" * 78)
    prov = report["provenance"]
    p(
        f"windows {prov['n_windows']}  arms {','.join(prov['arms'])}  "
        f"refit units {prov['n_refit']}  fitness units {prov['n_fit']}  "
        f"profile units {prov['n_profile']}"
    )
    for k, v in prov["prep_sha1"].items():
        p(f"  {k:40s} {v}")

    p("\n" + "=" * 78)
    p("A. SCALE — the fitter (arm=main), percent of rate; CI resamples WINDOWS")
    p("=" * 78)
    for grp, blk in report["scale"]["pooled"].items():
        p(f"{grp:16s} global {fmt_ci(blk['global'])}   per-rotor {fmt_ci(blk['per_rotor'])}")
        p(f"{'':16s} by rotor " + "  ".join(fmt_ci(b, 2) for b in blk["by_rotor"]))
    p("\nper recording:")
    for rec, ci in report["scale"]["per_recording"].items():
        p(f"  {rec:40s} {fmt_ci(ci)}")
    p("\nper window (global %, k ladder, stop):")
    for key, w in report["scale"]["per_window"].items():
        p(f"  {key:40s} {w['global_pct']:+8.3f}  {str(w['k_ladder']):28s} {w['stop_reason']}")

    p("\n" + "=" * 78)
    p("B. SCALE — the pulse-pair centre (independent of the fitter), percent")
    p("=" * 78)
    p(f"{'group|candidate|control':52s}{'per-window':>26s}{'per-rotor':>26s}")
    for tag, blk in sorted(report["pp"].items()):
        p(f"{tag:52s}{fmt_ci(blk['window_pct']):>26s}{fmt_ci(blk['rotor_pct']):>26s}")

    p("\n" + "=" * 78)
    p("C. SCALE — the one-parameter profile lp:5+scale:s (fixed DOF)")
    p("=" * 78)
    for tag, m in sorted(report["profile"].get("minima", {}).items()):
        ci = m["ci"]
        span = f"[{ci['lo']:+.3f},{ci['hi']:+.3f}]" if ci.get("lo") is not None else "—"
        val = f"{m['scale_pct']:+.3f}" if m["scale_pct"] is not None else "—"
        drop = m.get("curvature_drop")
        p(
            f"  {tag:46s} {val:>8s} {span:>18s}  depth {m.get('depth', float('nan')):.4f}"
            f"  drop {drop:.3f}  {m['note']}"
            if drop is not None
            else f"  {tag:46s} {val:>8s} {span:>18s}  "
            f"depth {m.get('depth', float('nan')):.4f}  {m['note']}"
        )
    for comp, blk in sorted(report.get("profile_all", {}).items()):
        p(f"\n  -- same profile read on {comp} --")
        for tag, m in sorted(blk.get("minima", {}).items()):
            if "|none" not in tag and "|on|none" not in tag:
                continue
            val = f"{m['scale_pct']:+.3f}" if m["scale_pct"] is not None else "—"
            drop = m.get("curvature_drop")
            p(
                f"  {tag:46s} {val:>8s}  depth {m.get('depth', float('nan')):.4f}"
                + (f"  drop {drop:.3f}" if drop is not None else "")
                + f"  {m['note']}"
            )

    p("\n" + "=" * 78)
    p("D. CONTROLS (hold-out=none)")
    p("=" * 78)
    head = f"{'group|candidate|control':58s}"
    head += "".join(f"{c:>12s}" for c in (*COMPONENTS, "pp_dr", "admit", "adm_ridge", "line_seen"))
    p(head)
    for tag, blk in sorted(report["controls"].items()):
        row = f"{tag:58s}"
        for c in (*COMPONENTS, "pp_dr", "admit_frac", "admit_frac_ridge", "line_share_gated"):
            v = blk.get(c)
            row += f"{v:12.5f}" if isinstance(v, float) else f"{'—':>12s}"
        p(row)

    p("\n" + "=" * 78)
    p("E. HOLD-OUT AGREEMENT (control=on)")
    p("=" * 78)
    for tag, blk in sorted(report["holdouts"].items()):
        p(f"  {tag}")
        for ho, vals in blk.items():
            row = f"    {ho:18s}"
            for c in (*COMPONENTS, "pp_dr"):
                v = vals.get(c)
                row += f"{v:12.5f}" if isinstance(v, float) else f"{'—':>12s}"
            p(row)

    p("\n" + "=" * 78)
    p("F. ABLATION — one step off per arm")
    p("=" * 78)
    p(
        f"{'group|arm':30s}{'global %':>26s}{'d_rms':>8s}{'ord':>5s}{'gapmin':>8s}"
        f"{'bband':>9s}{'phase':>9s}{'rough':>9s}{'pp_dr':>9s}"
    )
    for tag, b in sorted(report["ablation"].items()):
        row = f"{tag:30s}{fmt_ci(b['global_pct']):>26s}"
        for k, w, nd in (
            ("d_rms", 8, 3),
            ("order_kept", 5, 0),
            ("gap_ratio_min", 8, 3),
            ("broadband", 9, 4),
            ("phase_noise", 9, 4),
            ("roughness", 9, 4),
            ("pp_dr", 9, 4),
        ):
            v = b.get(k)
            row += f"{v:{w}.{nd}f}" if isinstance(v, (int, float)) else f"{'—':>{w}s}"
        p(row)

    p("\n" + "=" * 78)
    p("G. RESIDUAL — systematic and tachometer parts, separately")
    p("=" * 78)
    p(
        f"{'group|arm':30s}{'d_mean':>9s}{'d_rms':>9s}{'resid':>9s}{'lag_s':>9s}"
        f"{'cond':>11s}{'tachfrac':>10s}{'flat':>8s}"
    )
    for tag, b in sorted(report["residual"].items()):
        row = f"{tag:30s}"
        for k, w in (
            ("d_mean", 9),
            ("d_rms", 9),
            ("resid_rms", 9),
            ("lag_s", 9),
            ("design_cond", 11),
            ("tach_bound_frac", 10),
            ("tach_flatness", 8),
        ):
            v = b.get(k)
            row += f"{v:{w}.4f}" if isinstance(v, (int, float)) else f"{'—':>{w}s}"
        p(row)

    p("\n" + "=" * 78)
    p("H. IDENTITY — gaps per window (arm=main) and the permutation null")
    p("=" * 78)
    for key, v in report["identity"]["per_window"].items():
        if not key.endswith("|main"):
            continue
        raw = ", ".join(f"{g:.3f}" for g in v["gaps_raw"])
        got = ", ".join(f"{g:.3f}" for g in v["gaps_fit"])
        p(
            f"  {key[:-5]:40s} order_kept={v['order_kept']!s:5s} "
            f"raw [{raw}] -> fit [{got}]  x{v['gap_ratio']}"
        )
    p("\n  permutation null (residual pairing d_rms, rev/s):")
    for tag, blk in sorted(report["identity"]["permutation"].items()):
        p(f"    {tag:56s} " + "  ".join(f"{k}={v}" for k, v in sorted(blk.items())))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refit", default="results/telemetry_refit/campaign")
    ap.add_argument("--fit", default="results/telemetry_fitness/campaign")
    ap.add_argument("--profile", default="results/telemetry_fitness/scale_profile")
    ap.add_argument("--profile-component", default="phase_noise")
    ap.add_argument("--out", default="results/telemetry_report.json")
    args = ap.parse_args()

    refit = load_units(Path(args.refit))
    fit = load_units(Path(args.fit))
    prof = load_units(Path(args.profile)) if Path(args.profile).exists() else []

    arms = sorted({str(r.get("arm")) for r in refit})
    cands = sorted({short_candidate(str(r.get("candidate"))) for r in fit})
    report = {
        "provenance": {
            "n_windows": len({r["key"] for r in refit}),
            "arms": arms,
            "candidates": cands,
            "n_refit": len(refit),
            "n_fit": len(fit),
            "n_profile": len(prof),
            "prep_sha1": {
                r["key"]: r.get("prep_sha1") for r in sorted(refit, key=lambda r: r["key"])
            },
        },
        "scale": scale_section(refit),
        "pp": pp_section(fit),
        "profile": profile_section(prof, args.profile_component) if prof else {},
        # The three components disagree about WHERE the minimum is only if the
        # criterion matters; reporting all three is how that gets checked.
        "profile_all": {
            c: profile_section(prof, c) for c in COMPONENTS if prof and c != args.profile_component
        },
        "controls": controls_section(fit),
        "holdouts": holdout_section(fit, cands),
        "ablation": ablation_section(refit, fit),
        "residual": residual_section(refit),
        "identity": identity_section(refit, fit),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    show(report)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
