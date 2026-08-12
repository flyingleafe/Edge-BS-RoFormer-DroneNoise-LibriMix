#!/usr/bin/env python
"""Attribute the VK broadband residual to individual rotors using the mic array.

Three stages, each writing JSON + figures under ``--out``:

``identifiability``
    Geometry only, no data: is the ``sum_r P_r g_r g_r^H + diag(D)`` design
    separable on this array, and where does it break with frequency?

``synthetic``
    Render known per-rotor noises through the repo's own ``propagate``, add a
    known per-mic term, recover ``P_r``. Calibrates both the estimator and the
    null controls.

``real``
    Fit the published VK residual (``artifacts/vk-decompose/<rid>/`` on R2),
    with null controls, a segment bootstrap, and coherence diagnostics.

Example::

    set -a; source .env; set +a
    PYTHONPATH=src python scripts/residual_attribution.py all \\
        --recording free-flight_nosource_room1

Memory: the whole campaign fits in well under 6 GB; the CSD of a 64 s, 8-mic
recording at ``nperseg=4096`` is ~65 MB of segment spectra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.residual_attribution import controls, csd, data, design, fit, plots, steering
from experiments.residual_attribution.synth import make_sources, render, run_case

BANDS = [(20, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
CONTROL_BANDS = [(100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
DEFAULT_OUT = Path("results/residual_attribution")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _write(out: Path, name: str, payload) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(_jsonable(payload), indent=1))
    print(f"[write] {p}")
    return p


# ---------------------------------------------------------------------------
# stage 1 — identifiability
# ---------------------------------------------------------------------------


def stage_identifiability(args) -> dict:
    out = Path(args.out)
    mic, rot = data.geometry_for(args.drone)
    freqs = np.unique(
        np.concatenate([np.arange(20.0, 200.0, 10.0), np.geomspace(200.0, 8000.0, 120)])
    )
    g = steering.steering(mic, rot, freqs)
    plan = design.index_plan(len(mic))
    diag = design.identifiability(g, plan)
    alias = steering.aliasing_frequency(mic)

    # where does each rotor become individually identifiable?
    thr = 10.0
    ok = (diag["vif"] < thr).all(axis=1)
    f_ok = float(freqs[ok][0]) if ok.any() else float("nan")

    payload = {
        "drone": args.drone,
        "geometry_note": (
            "DREGON geometry is TDOA-validated; Michael's is a synthetic ring model (unvalidated)"
        ),
        "n_mic": int(len(mic)),
        "n_rotor": int(len(rot)),
        "max_mic_spacing_m": steering.max_mic_spacing(mic),
        "aliasing_frequency_hz": alias,
        "rotor_mic_distance_m": {
            "min": float(steering.distances(mic, rot).min()),
            "max": float(steering.distances(mic, rot).max()),
        },
        "dof": {
            "real_equations": int(len(mic) ** 2),
            "diagonal_equations": int(len(mic)),
            "offdiagonal_equations": int(plan.n_off_rows),
            "unknowns_P": int(len(rot)),
            "unknowns_D": int(len(mic)),
            "note": (
                "the D unknowns span the whole diagonal subspace, so the diagonal "
                "equations carry no information about P: attribution rests on the "
                f"{plan.n_off_rows} cross-microphone equations alone"
            ),
        },
        "first_frequency_all_vif_below_10_hz": f_ok,
        "by_band": {},
    }
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        payload["by_band"][f"{lo}-{hi}"] = {
            "cond_median": float(np.median(diag["cond"][m])),
            "max_cos_median": float(np.median(diag["max_cos"][m])),
            "vif_median": np.median(diag["vif"][m], axis=0).round(2).tolist(),
            "noise_gain_median": np.median(diag["noise_gain"][m], axis=0).round(2).tolist(),
        }
    _write(out, f"identifiability_{args.drone}.json", payload)
    plots.plot_identifiability(freqs, diag, alias, out / f"fig_identifiability_{args.drone}.png")
    return payload


# ---------------------------------------------------------------------------
# stage 2 — synthetic validation
# ---------------------------------------------------------------------------


def stage_synthetic(args) -> dict:
    out = Path(args.out)
    mic, rot = data.geometry_for(args.drone)
    plan = design.index_plan(len(mic))
    levels = np.array([0.0, -3.0, 3.0, -6.0])[: len(rot)]
    slopes = np.array([1.0, 1.5, 0.5, 2.0])[: len(rot)]

    recovery = {}
    cases = []
    for dc in args.diag_to_coh_db:
        r = run_case(
            mic,
            rot,
            duration_s=args.duration_s,
            diag_to_coh_db=dc,
            levels_db=levels,
            slopes=slopes,
            seed=args.seed,
        )
        row = {
            "share_true": r.share_true.round(4).tolist(),
            "share_hat": r.share_hat.round(4).tolist(),
            "share_abs_err": float(np.abs(r.share_true - r.share_hat).max()),
            "by_band": {},
        }
        for lo, hi in BANDS:
            m = (r.freqs >= lo) & (r.freqs < hi)
            # per-rotor median log-error, restricted to rotors that carry
            # >1% of the true power in the band (a silent rotor's dB error is
            # meaningless and would dominate any pooled statistic)
            tot = r.p_true[m].sum(axis=0)
            live = tot > 0.01 * tot.sum()
            db = 10 * np.log10(np.maximum(r.p_hat[m], 1e-300) / np.maximum(r.p_true[m], 1e-300))
            row["by_band"][f"{lo}-{hi}"] = {
                "off_explained": float(np.nanmean(r.off_explained[m])),
                "live_rotors": live.astype(int).tolist(),
                "median_err_db": np.median(db, axis=0).round(2).tolist(),
                "median_abs_err_db_live": float(np.median(np.abs(np.median(db, axis=0)[live]))),
            }
        recovery[f"{dc:+g}dB"] = row
        cases.append((f"diag/coh {dc:+g} dB", r.freqs, r.p_true, r.p_hat))
        print(f"[synthetic] diag/coh {dc:+g} dB  share err max {row['share_abs_err']:.4f}")

    # control calibration: does the null control discriminate on ideal data?
    rng = np.random.default_rng(args.seed + 11)
    src = make_sources(
        len(rot),
        int(args.duration_s * 16000),
        16000.0,
        levels_db=np.array([0.0, -1.0, 1.0, -2.0])[: len(rot)],
        slopes=np.zeros(len(rot)) + 1.0,
        rng=rng,
    )
    coh = render(src, mic, rot, 16000.0)
    del src
    p_coh = float((coh**2).mean())
    calib = {}
    for dc in args.control_diag_db:
        x = coh + rng.standard_normal(coh.shape) * np.sqrt(p_coh * 10 ** (dc / 10))
        c = csd.welch_csd(x, 16000.0, nperseg=args.nperseg)
        del x
        R = c.matrix()
        calib[f"{dc:+g}dB"] = controls.run_controls(
            R, c.freqs, mic, rot, plan, CONTROL_BANDS, n_draw=args.n_draw, seed=args.seed
        )
        offs, curve = controls.displacement_curve(
            R, c.freqs, mic, rot, plan, (500.0, 2000.0), n_draw=6, seed=args.seed
        )
        calib[f"{dc:+g}dB"]["displacement_500_2000"] = {
            "offsets_m": offs.tolist(),
            "explained": curve.round(4).tolist(),
        }
        del R, c
        print(f"[synthetic] controls at diag/coh {dc:+g} dB done")

    # dynamic range: flat spectra, only the level differs, so a per-rotor error
    # is attributable to level alone. How far below the loudest rotor can a
    # rotor sit and still be recovered?
    dyn_levels = np.array([0.0, -10.0, -20.0, -30.0])[: len(rot)]
    dyn = run_case(
        mic,
        rot,
        duration_s=args.duration_s,
        diag_to_coh_db=0.0,
        levels_db=dyn_levels,
        slopes=np.zeros(len(rot)),
        seed=args.seed + 3,
    )
    dyn_rows = {}
    for lo, hi in BANDS:
        m = (dyn.freqs >= lo) & (dyn.freqs < hi)
        db = np.median(
            10 * np.log10(np.maximum(dyn.p_hat[m], 1e-300) / np.maximum(dyn.p_true[m], 1e-300)),
            axis=0,
        )
        dyn_rows[f"{lo}-{hi}"] = db.round(2).tolist()
    print(f"[synthetic] dynamic range, levels {dyn_levels.tolist()} dB -> err_dB {dyn_rows}")

    payload = {
        "levels_db": levels.tolist(),
        "psd_slopes": slopes.tolist(),
        "duration_s": args.duration_s,
        "recovery": recovery,
        "dynamic_range": {
            "note": (
                "flat per-rotor spectra at these relative levels, diag/coh 0 dB; "
                "value is the median dB error of the fitted PSD per rotor"
            ),
            "levels_db": dyn_levels.tolist(),
            "err_db_by_band": dyn_rows,
        },
        "control_calibration": calib,
    }
    _write(out, f"synthetic_{args.drone}.json", payload)
    plots.plot_synthetic_recovery(cases, out / f"fig_synthetic_recovery_{args.drone}.png")
    return payload


# ---------------------------------------------------------------------------
# stage 3 — real data
# ---------------------------------------------------------------------------


def stage_real(args) -> dict:
    out = Path(args.out)
    rid = args.recording
    drone = args.drone or data.drone_of(rid)
    res = data.fetch_residual(rid)
    mic, rot = data.geometry_for(drone)
    x = res.audio.astype(np.float64)
    if x.shape[0] != len(mic):
        raise ValueError(f"residual has {x.shape[0]} mics, geometry has {len(mic)}")

    c = csd.welch_csd(x, float(res.sample_rate), nperseg=args.nperseg)
    R = c.matrix()
    g = steering.steering(mic, rot, c.freqs)
    plan = design.index_plan(len(mic))
    att = fit.fit_offdiag(R, g, plan)
    attj = fit.fit_joint(R, g, plan)

    coh_meas = csd.coherence(R)
    Rm = fit.model_matrix(att.p_rotor, att.d_mic, g)
    coh_mod = csd.coherence(Rm)
    iu, ju = np.triu_indices(len(mic), 1)
    msc_meas = coh_meas[:, iu, ju].mean(axis=1)
    msc_mod = coh_mod[:, iu, ju].mean(axis=1)

    ctrl = controls.run_controls(
        R, c.freqs, mic, rot, plan, CONTROL_BANDS, n_draw=args.n_draw, seed=args.seed
    )
    offs, curve = controls.displacement_curve(
        R, c.freqs, mic, rot, plan, (500.0, 2000.0), n_draw=8, seed=args.seed
    )

    all_bands = [*BANDS, (BANDS[0][0], BANDS[-1][1])]
    masks = [(c.freqs >= lo) & (c.freqs < hi) for lo, hi in all_bands]
    boots = fit.bootstrap_shares(c, g, plan, masks, n_boot=args.n_boot, seed=args.seed)

    by_band = {}
    for (lo, hi), m, boot in zip(all_bands, masks, boots):
        sh = np.concatenate([att.recv_rotor, att.recv_diag[:, None]], axis=1)[m].sum(axis=0)
        sh = sh / max(sh.sum(), 1e-300)
        shj = np.concatenate([attj.recv_rotor, attj.recv_diag[:, None]], axis=1)[m].sum(axis=0)
        shj = shj / max(shj.sum(), 1e-300)
        by_band[f"{lo}-{hi}"] = {
            "mean_msc_measured": float(msc_meas[m].mean()),
            "mean_msc_model": float(msc_mod[m].mean()),
            "off_explained": float(np.nanmean(att.off_explained[m])),
            "shares_offdiag_fit": sh.round(4).tolist(),
            "shares_joint_fit": shj.round(4).tolist(),
            "shares_boot_p05": np.percentile(boot, 5, axis=0).round(4).tolist(),
            "shares_boot_p95": np.percentile(boot, 95, axis=0).round(4).tolist(),
            "d_clipped_fraction": float(att.d_clipped[m].mean()),
            "rotor_share_of_coherent_part": (
                (sh[:-1] / max(sh[:-1].sum(), 1e-300)).round(4).tolist()
            ),
        }
        print(
            f"[real] {lo}-{hi} Hz  shares {np.round(sh, 3)}  expl {by_band[f'{lo}-{hi}']['off_explained']:.3f}"
        )

    payload = {
        "recording_id": rid,
        "drone": drone,
        "geometry_note": (
            "DREGON geometry TDOA-validated; Michael's is a synthetic ring model "
            "(unvalidated) — attribution on it inherits that uncertainty"
        ),
        "span_s": res.report.get("span_s"),
        "residual_fraction_of_recording": res.report.get("energy", {}).get("residual_fraction"),
        "n_welch_segments": c.n_seg,
        "nperseg": args.nperseg,
        "share_order": [f"rotor{r}" for r in range(len(rot))] + ["diagonal"],
        "by_band": by_band,
        "controls": ctrl,
        "displacement_500_2000": {
            "offsets_m": offs.tolist(),
            "explained": curve.round(4).tolist(),
        },
    }
    _write(out, f"real_{rid}.json", payload)

    edges = csd.band_edges(50.0, 8000.0, 24)
    plots.plot_real_fit(
        c.freqs, att, edges, out / f"fig_real_fit_{rid}.png", title=f"{rid} — VK broadband residual"
    )
    plots.plot_coherence(c.freqs[1:], msc_meas[1:], msc_mod[1:], out / f"fig_coherence_{rid}.png")
    return payload


def stage_controls_figure(args) -> None:
    out = Path(args.out)
    real_p = out / f"real_{args.recording}.json"
    syn_p = out / f"synthetic_{args.drone or 'dregon'}.json"
    if not (real_p.exists() and syn_p.exists()):
        print("[controls-fig] need both real_*.json and synthetic_*.json — skipping")
        return
    real = json.loads(real_p.read_text())["controls"]
    syn_all = json.loads(syn_p.read_text())["control_calibration"]
    key = sorted(syn_all, key=lambda k: abs(float(k.replace("dB", ""))))[0]
    syn = {k: v for k, v in syn_all[key].items() if k != "displacement_500_2000"}
    plots.plot_controls(real, syn, out / f"fig_controls_{args.recording}.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["identifiability", "synthetic", "real", "all", "list"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--drone", default=None, help="dregon | michaels (inferred from --recording)")
    ap.add_argument("--recording", default="free-flight_nosource_room1")
    ap.add_argument("--nperseg", type=int, default=4096)
    ap.add_argument("--n-draw", type=int, default=16, help="random-geometry control draws")
    ap.add_argument("--n-boot", type=int, default=32)
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--diag-to-coh-db", type=float, nargs="*", default=[-100.0, -10.0, 0.0, 10.0])
    ap.add_argument("--control-diag-db", type=float, nargs="*", default=[-100.0, 0.0, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stage == "list":
        for r in data.list_decompositions():
            print(r)
        return
    if args.drone is None and args.stage != "real":
        args.drone = data.drone_of(args.recording)

    if args.stage in ("identifiability", "all"):
        stage_identifiability(args)
    if args.stage in ("synthetic", "all"):
        stage_synthetic(args)
    if args.stage in ("real", "all"):
        stage_real(args)
    if args.stage == "all":
        stage_controls_figure(args)


if __name__ == "__main__":
    main()
