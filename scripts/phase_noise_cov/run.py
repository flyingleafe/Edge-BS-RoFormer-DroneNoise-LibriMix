#!/usr/bin/env python3
"""WP18 — measure the across-harmonic covariance of the rate opinions.

Runs :mod:`estimate` over every window of :mod:`windows` and writes

    results/phase_noise_covariance/manifest.json     windows + provenance
    results/phase_noise_covariance/raw/<window>.json one file per window
    results/phase_noise_covariance/summary.json      the aggregated verdicts
    results/phase_noise_covariance/v_k.csv           the empirical weight curves

**Restartable** (a window whose JSON exists is skipped) and **parallel** (one
process per window, BLAS pinned to one thread each).  The 44.1 kHz load +
resample of Michael's recordings and the ``beatvk-valid-raw`` stream happen
ONCE, in the parent, into ``.cache/phase_noise_cov`` — workers only ever read
small NPZs.

Cluster invocation (CPU-only)::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 4h -- \\
        python scripts/phase_noise_cov/run.py --jobs 16
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

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "scripts"), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)

import estimate as E  # noqa: E402
import windows as W  # noqa: E402

RESULTS = Path("results/phase_noise_covariance")
CACHE = Path(".cache/phase_noise_cov")
WIN_CACHE = CACHE / "win"


# ---------------------------------------------------------------------------
# window cache


def cache_windows(which: str, force: bool = False) -> dict[str, Any]:
    """Materialize every window as an NPZ; return the manifest."""
    WIN_CACHE.mkdir(parents=True, exist_ok=True)
    man: dict[str, Any] = {"windows": []}
    for win in W.build(which):
        p = WIN_CACHE / f"{win.name}.npz"
        entry = {
            "name": win.name,
            "dataset": win.dataset,
            "environment": win.environment,
            "rid": win.rid,
            "widx": win.widx,
            "trajectory": win.trajectory,
            "n_channels": int(win.audio.shape[0]),
            "duration_s": round(win.audio.shape[-1] / W.SR, 3),
            "rotor_means": np.round(win.r_traj.mean(axis=1), 3).tolist(),
            "rotor_min_split": round(
                float(
                    np.min(
                        np.abs(win.r_traj.mean(1)[:, None] - win.r_traj.mean(1)[None, :])[
                            np.triu_indices(W.N_ROTORS, 1)
                        ]
                    )
                ),
                3,
            ),
            **win.meta,
        }
        man["windows"].append(entry)
        if p.exists() and not force:
            continue
        hi_t, hi_v = win.r_hires if win.r_hires is not None else (np.zeros(0), np.zeros(0))
        np.savez(
            p,
            audio=win.audio.astype(np.float32),
            ft=win.ft,
            r_traj=win.r_traj,
            jitter=(win.jitter if win.jitter is not None else np.zeros(0)),
            hires_t=hi_t,
            hires_v=hi_v,
            meta=json.dumps(entry),
        )
    return man


def load_window(name: str) -> dict[str, Any]:
    with np.load(WIN_CACHE / f"{name}.npz", allow_pickle=False) as z:
        jit = np.asarray(z["jitter"], dtype=np.float64)
        hi_t = np.asarray(z["hires_t"], dtype=np.float64)
        hi_v = np.asarray(z["hires_v"], dtype=np.float64)
        return {
            "audio": np.asarray(z["audio"], dtype=np.float64),
            "ft": np.asarray(z["ft"], dtype=np.float64),
            "r_traj": np.asarray(z["r_traj"], dtype=np.float64),
            "jitter": jit if jit.size else None,
            "hires": (hi_t, hi_v) if hi_t.size else None,
            "meta": json.loads(str(z["meta"].item())),
        }


# ---------------------------------------------------------------------------
# per-window work


def predicted_common(
    r_aud: np.ndarray, n_aud: np.ndarray, bands: list[float], stride: int, n_env: int, fs_env: float
) -> dict[str, Any]:
    """The common-term variance the INJECTED jitter must produce, per band.

    For a small arrival-time error ``n(t)`` the demodulated envelope is
    ``A exp(i 2 pi k r n)``, so after the brickwall the harmonic's rate opinion
    carries ``J = d/dt (r n)`` band-limited the same way — independent of ``k``.
    This pushes the *known* ``n(t)`` through exactly that chain, giving the
    number the estimator has to reproduce on the synthetic control.
    """
    out: dict[str, Any] = {}
    u = (r_aud * n_aud).astype(np.complex64)[None, None, :]
    for b in bands:
        ub = np.real(E.pit._zoom_lp_decimate(u, stride, n_env, b / E.SR))[0, 0]
        j = np.diff(ub) * fs_env
        n_trim = max(1, int(round(E.EDGE_TRIM_S * fs_env)))
        seg = j[n_trim : len(j) - n_trim]
        per_cut = {}
        for fc in E.CUTOFFS:
            if fc > 0 and fc > E.CUTOFF_BAND_FRAC * b:
                continue
            s = E._brickwall(j, fc, fs_env, high=True) if fc > 0 else j
            per_cut[f"{fc:g}"] = float(np.var(s[n_trim : len(j) - n_trim]))
        out[f"{b:g}"] = {"var_full": float(np.var(seg)), "per_cutoff": per_cut}
    return out


def run_window(
    name: str, channels: int | None, arms: tuple[str, ...], traj: str = "auto"
) -> dict[str, Any]:
    """All rotors x all arms for one window.

    ``traj="auto"`` demodulates along the highest-rate trajectory the window
    carries (DREGON's native ~929 Hz telemetry, a synthetic's exact one);
    ``traj="framegrid"`` forces the 0.032 s grid, which is how much of the
    measured common term is an artefact of representing the trajectory.
    """
    tic = time.perf_counter()
    w = load_window(name)
    use = [a for a in E.ARMS if a.name in arms]
    res: dict[str, Any] = {
        "window": name,
        "meta": w["meta"],
        "traj_mode": traj,
        "traj_hires": bool(w["hires"] is not None and traj == "auto"),
        "rotors": [],
    }
    stride = max(1, int(round(E.SR / E.FS_ENV)))
    t_aud = np.arange(w["audio"].shape[-1]) / E.SR
    for i in range(w["r_traj"].shape[0]):
        carrier = None
        if traj == "auto" and w["hires"] is not None:
            hi_t, hi_v = w["hires"]
            carrier = np.interp(t_aud, hi_t, hi_v[i])
        dm = E.demod_rotor(w["audio"], w["r_traj"], w["ft"], i, fs_env=E.FS_ENV, r_carrier=carrier)
        if dm is None:
            res["rotors"].append({"rotor": i, "skipped": "silent or no harmonic"})
            continue
        row: dict[str, Any] = {
            "rotor": i,
            "mean_rate": round(dm.mean_rate, 3),
            "demod": dm.diag,
            "arms": {},
        }
        for arm in use:
            row["arms"][arm.name] = E.arm_covariance(dm, arm, channels=channels)
        row["chan_coherence"] = E.channel_coherence(dm, E.ARMS[0])
        if w["jitter"] is not None:
            r_aud = carrier if carrier is not None else np.interp(t_aud, w["ft"], w["r_traj"][i])
            bands = sorted({float(a.band(k)) for a in use for k in dm.ks})
            row["predicted_common"] = predicted_common(
                r_aud, w["jitter"][i], bands, stride, dm.z.shape[-1], dm.fs_env
            )
        res["rotors"].append(row)
    res["wall_s"] = round(time.perf_counter() - tic, 1)
    return res


def _worker(args: tuple[str, int | None, tuple[str, ...], str, str]) -> tuple[str, bool, str]:
    name, channels, arms, out_dir, traj = args
    p = Path(out_dir) / f"{name}.json"
    try:
        res = run_window(name, channels, arms, traj)
        p.write_text(json.dumps(res))
        return name, True, f"{res['wall_s']}s"
    except Exception:  # noqa: BLE001 — one window must not kill the sweep
        return name, False, traceback.format_exc(limit=6)


# ---------------------------------------------------------------------------
# aggregation


def _med(xs: list[float]) -> float | None:
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    return float(np.median(a)) if a.size else None


def _iqr(xs: list[float]) -> float | None:
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    return float(np.percentile(a, 75) - np.percentile(a, 25)) if a.size >= 4 else None


def aggregate(raw_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Pool per-(window, rotor) fits into per-group verdicts."""
    by_name = {w["name"]: w for w in manifest["windows"]}
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for p in sorted(raw_dir.glob("*.json")):
        res = json.loads(p.read_text())
        meta = by_name.get(res["window"], res.get("meta", {}))
        grp = meta.get("dataset", "?")
        if grp == "synth":
            grp = f"synth_{res['window'].split('_')[1]}"
        for row in res["rotors"]:
            if "skipped" in row:
                continue
            groups.setdefault(grp, []).append((meta, row))

    summary: dict[str, Any] = {"groups": {}}
    for grp, rows in sorted(groups.items()):
        g: dict[str, Any] = {
            "n_units": len(rows),
            "n_windows": len({m.get("name") for m, _ in rows}),
            "environment": rows[0][0].get("environment", "?"),
            "chan_coherence": _med(
                [r.get("chan_coherence", {}).get("chan_coherence_mean") for _, r in rows]
            ),
            "arms": {},
        }
        for arm in E.ARMS:
            aro = [r["arms"][arm.name] for _, r in rows if arm.name in r.get("arms", {})]
            aro = [a for a in aro if "cov" in a]
            if not aro:
                continue
            allo = [r["arms"][arm.name] for _, r in rows if arm.name in r.get("arms", {})]
            ent: dict[str, Any] = {
                "n": len(aro),
                "n_units_attempted": len(allo),
                "n_units_failed": len(allo) - len(aro),
                "n_keep": _med([a["n_keep"] for a in aro]),
                # coverage is a result in its own right: at a fixed band the
                # twin-collision radius in rev/s is B/k, so a wide band on a
                # tight pair empties the harmonic set entirely
                "n_keep_all": _med([a["n_keep"] for a in allo]),
                "lost_twin_gate": _med([a.get("lost_twin_gate") for a in allo]),
                "lost_snr": _med([a.get("lost_snr") for a in allo]),
            }
            # v_k profile on the common k grid
            prof: dict[int, list[float]] = {}
            snr_prof: dict[int, list[float]] = {}
            for a in aro:
                for k, v in zip(a.get("k_used", []), a.get("v_k_used", []), strict=False):
                    if np.isfinite(v) and v > 0:
                        prof.setdefault(int(k), []).append(float(v))
                for k, s in zip(a["ks"], a["snr"], strict=False):
                    snr_prof.setdefault(int(k), []).append(float(s))
            ent["v_k"] = {str(k): _med(v) for k, v in sorted(prof.items())}
            ent["v_k_n"] = {str(k): len(v) for k, v in sorted(prof.items())}
            ent["snr_k"] = {str(k): _med(v) for k, v in sorted(snr_prof.items())}
            for key in ("alpha_raw", "alpha_signal", "alpha_snr"):
                sl = [a[key]["slope"] for a in aro if key in a]
                r2 = [a[key]["r2"] for a in aro if key in a]
                if sl:
                    ent[key] = {"median": _med(sl), "iqr": _iqr(sl), "r2_median": _med(r2)}
            # cutoff sweep of the common term + fit quality
            cuts: dict[str, Any] = {}
            for fc in E.CUTOFFS:
                tag = f"{fc:g}"
                fits = [a["cov"][tag] for a in aro if tag in a.get("cov", {})]
                if not fits:
                    continue
                # A loading exponent read off an unresolved common term is a
                # noise eigenvector, so beta is pooled over the units where the
                # term is significant (|sigma_c2| > 3 SE) and the count is
                # reported next to it.
                sig = [f for f in fits if abs(f.get("sigma_c2_signif") or 0.0) >= 3.0]
                cuts[tag] = {
                    "n": len(fits),
                    "n_significant": len(sig),
                    "sigma_c2_mean": _med([f.get("sigma_c2_mean") for f in fits]),
                    "sigma_c2_median": _med([f.get("sigma_c2_median") for f in fits]),
                    "sigma_c2_iqr": _iqr([f.get("sigma_c2_median") for f in fits]),
                    "sigma_c2_se": _med([f.get("sigma_c2_se") for f in fits]),
                    "sigma_c2_signif": _med([f.get("sigma_c2_signif") for f in fits]),
                    "offdiag_resid_rel": _med([f.get("offdiag_resid_rel") for f in fits]),
                    "offdiag_chi2": _med([f.get("offdiag_chi2") for f in fits]),
                    "offdiag_excess_rel": _med([f.get("offdiag_excess_rel") for f in fits]),
                    "rank1_energy_frac": _med([f.get("rank1_energy_frac") for f in fits]),
                    "loading_beta": _med([f.get("loading_beta") for f in sig]),
                    "loading_snr_corr": _med([f.get("loading_snr_corr") for f in sig]),
                    "offdiag_corr_min": _med([f.get("offdiag_corr_min") for f in fits]),
                    "offdiag_corr_absdiff": _med([f.get("offdiag_corr_absdiff") for f in fits]),
                    "offdiag_neg_frac": _med([f.get("offdiag_neg_frac") for f in fits]),
                    "k_star": _med([a.get(f"k_star_fc{tag}") for a in aro]),
                    "k_star_none_frac": float(
                        np.mean([a.get(f"k_star_fc{tag}") is None for a in aro])
                    ),
                    "floor_share": _med([a.get(f"floor_share_fc{tag}") for a in aro]),
                    "inv_W_stage": _med([a.get(f"inv_W_stage_fc{tag}") for a in aro]),
                }
            ent["cutoffs"] = cuts
            # synthetic control: measured vs injected
            preds = []
            for _, r in rows:
                pc = r.get("predicted_common")
                if not pc or arm.name not in r.get("arms", {}):
                    continue
                a = r["arms"][arm.name]
                if "cov" not in a:
                    continue
                b = f"{arm.band(6):g}"
                if b not in pc:
                    continue
                for tag, fit in a["cov"].items():
                    if tag in pc[b]["per_cutoff"]:
                        preds.append(
                            {
                                "fc": tag,
                                "measured": fit.get("sigma_c2_median"),
                                "predicted": pc[b]["per_cutoff"][tag],
                            }
                        )
            if preds:
                ent["control"] = {
                    tag: {
                        "measured": _med([p["measured"] for p in preds if p["fc"] == tag]),
                        "predicted": _med([p["predicted"] for p in preds if p["fc"] == tag]),
                    }
                    for tag in sorted({p["fc"] for p in preds}, key=float)
                }
            g["arms"][arm.name] = ent
        summary["groups"][grp] = g

    # the headline contrast
    ind = summary["groups"].get("dregon")
    out = summary["groups"].get("michaels")
    if ind and out:
        contrast: dict[str, Any] = {}
        for arm in ("fixB1.5", "fixB3", "fixB6", "kscale0.25", "kscale0.5"):
            a_i, a_o = ind["arms"].get(arm), out["arms"].get(arm)
            if not a_i or not a_o:
                continue
            row: dict[str, Any] = {}
            for tag in a_i.get("cutoffs", {}):
                if tag not in a_o.get("cutoffs", {}):
                    continue
                si = a_i["cutoffs"][tag]["sigma_c2_median"]
                so = a_o["cutoffs"][tag]["sigma_c2_median"]
                row[f"sigma_c2_fc{tag}"] = {"indoor": si, "outdoor": so}
            vi = [v for v in a_i["v_k"].values() if v]
            vo = [v for v in a_o["v_k"].values() if v]
            row["v_k_median"] = {"indoor": _med(vi), "outdoor": _med(vo)}
            contrast[arm] = row
        summary["indoor_outdoor"] = contrast
    return summary


def write_vk_csv(summary: dict[str, Any], path: Path) -> None:
    lines = ["group,arm,k,v_k,n,snr_k"]
    for grp, g in summary["groups"].items():
        for arm, a in g.get("arms", {}).items():
            for k, v in a.get("v_k", {}).items():
                lines.append(f"{grp},{arm},{k},{v!r},{a['v_k_n'].get(k, 0)},{a['snr_k'].get(k)!r}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="WP18 harmonic phase-noise covariance")
    ap.add_argument("--windows", default="all", help="synth,dregon,michaels | all")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--channels", type=int, default=None, help="cap analysed mic channels")
    ap.add_argument("--arms", default=",".join(a.name for a in E.ARMS))
    ap.add_argument(
        "--traj",
        choices=("auto", "framegrid"),
        default="auto",
        help="auto = the highest-rate trajectory a window carries; framegrid = the 0.032 s grid",
    )
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--force", action="store_true", help="recompute existing windows")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.results)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    man_p = out / "manifest.json"

    if args.aggregate_only and man_p.exists():
        man = json.loads(man_p.read_text())
    else:
        tic = time.perf_counter()
        man = cache_windows(args.windows, force=args.rebuild_cache)
        man["built_s"] = round(time.perf_counter() - tic, 1)
        man["fs_env"] = E.FS_ENV
        man["b_wide"] = E.B_WIDE
        man["k_max"] = E.K_MAX
        man["cutoffs"] = list(E.CUTOFFS)
        man["arms"] = {a.name: {"kind": a.kind, "b": a.b, "cap": a.cap} for a in E.ARMS}
        man_p.write_text(json.dumps(man, indent=2))
        print(f"[cache] {len(man['windows'])} windows in {man['built_s']}s", flush=True)

    if not args.aggregate_only:
        arms = tuple(a.strip() for a in args.arms.split(","))
        todo = [
            (str(w["name"]), args.channels, arms, str(raw), args.traj)
            for w in man["windows"]
            if args.force or not (raw / f"{w['name']}.json").exists()
        ]
        print(f"[run] {len(todo)} windows, {args.jobs} jobs", flush=True)
        if todo:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(_worker, t): t[0] for t in todo}
                for fut in as_completed(futs):
                    name, ok, msg = fut.result()
                    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}", flush=True)

    summary = aggregate(raw, man)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_vk_csv(summary, out / "v_k.csv")
    print(json.dumps(summary.get("indoor_outdoor", {}), indent=2), flush=True)
    for grp, g in summary["groups"].items():
        arm = g["arms"].get("fixB1.5", {})
        cuts = arm.get("cutoffs", {})
        c4 = cuts.get("0", {})
        print(
            f"{grp:22s} n={g['n_units']:3d} chan_coh={g['chan_coherence']} "
            f"sigma_c2(fc4)={c4.get('sigma_c2_median')} k*={c4.get('k_star')} "
            f"floor_share={c4.get('floor_share')} alpha_sig={arm.get('alpha_signal', {}).get('median')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
