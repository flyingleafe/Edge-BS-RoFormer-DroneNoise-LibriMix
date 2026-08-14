#!/usr/bin/env python
"""Attribute the VK broadband residual to rotors by band POWER (the second pass).

The first pass (``scripts/residual_attribution.py``) fitted the array
cross-spectrum with four coherent point sources and was refuted by its own
geometry null controls. This one drops coherence and uses the two levers that
survive it — a MEASURED per-rotor per-microphone transfer from DREGON's
single-motor bench recordings, and the time modulation of the four rotor speeds
— then checks the two against each other under a rotor-permutation null.

Stages, each writing JSON under ``--out``:

``coherence``
    Why the first pass failed: magnitude-squared coherence of the residual per
    band and rig. No model, no fit.

``bench``
    The DREGON single-motor recordings as ground truth: the per-rotor pattern,
    its throttle stability, its conditioning against the free-field ``1/d^2``
    basis, and the additivity test against ``allMotors_70``.

``real``
    Per recording: band power of the residual, the geometry-free modulation
    regression (per-mic per-rotor shares + block bootstrap), the
    basis-constrained fit with its permutation null, and — on DREGON — the
    agreement between the in-flight pattern and the bench pattern.

Example::

    PYTHONPATH=src python scripts/residual_attribution_power.py all \\
        --decomp results/vk_decompose_v2

Cost: minutes on one core, under 2 GB. The band powers of a 178 s 8-mic
residual are 40 kB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from experiments.residual_attribution import data, power

DEFAULT_OUT = Path("results/residual_attribution_power")
RECORDINGS = ["free-flight_nosource_room1", "FLY124", "FLY125"]
COH_BANDS = [(50, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _write(out: Path, name: str, payload) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(_jsonable(payload), indent=1))
    print(f"wrote {p}")
    return p


# ─── Stages ──────────────────────────────────────────────────────────────────


def stage_coherence(args) -> dict:
    """Magnitude-squared coherence of the residual, per band and mic pair."""
    out = {}
    for rid in args.recordings:
        res = data.local_residual(rid, args.decomp)
        sr = res.sample_rate
        a = int(args.coh_start * sr)
        x = np.asarray(res.audio[:, a : a + int(args.coh_seconds * sr)], dtype=np.float64)
        f, cxy = signal.coherence(x[:, None, :], x[None, :, :], fs=sr, nperseg=2048, axis=-1)
        mic, _ = data.geometry_for(data.drone_of(rid), args.cache)
        dist = np.linalg.norm(mic[:, None, :] - mic[None, :, :], axis=-1)
        iu = np.triu_indices(x.shape[0], 1)
        rows = []
        for lo, hi in COH_BANDS:
            m = (f >= lo) & (f < hi)
            msc = cxy[:, :, m].mean(-1)[iu]
            # A diffuse (isotropic) field is the most incoherent PROPAGATING
            # field there is: anything below it is not a sound field at the
            # array at all.
            diffuse = np.sinc(2 * np.sqrt(lo * hi) * dist[iu] / 343.0) ** 2
            rows.append(
                {
                    "band": [lo, hi],
                    "msc_mean": float(msc.mean()),
                    "msc_max": float(msc.max()),
                    "diffuse_mean": float(diffuse.mean()),
                }
            )
        out[rid] = {"aperture_m": float(dist.max()), "bands": rows}
        print(
            f"{rid}: "
            + " ".join(f"{r['band'][0]}-{r['band'][1]}:{r['msc_mean']:.3f}" for r in rows)
        )
    return out


def stage_bench(args) -> dict:
    """The DREGON single-motor recordings as the attribution ground truth."""
    clips, sr = data.bench_clips(args.data_root)
    basis, rep = power.bench_basis(clips, sr, power.BANDS)
    combined, sr_c = data.bench_combined(args.data_root, throttle=args.throttle)
    single = {r: clips[(r, args.throttle)] for r in range(4) if (r, args.throttle) in clips}
    add = power.additivity(single, combined, sr_c, power.BANDS)

    mic, rot = data.geometry_for("dregon", args.cache)
    geom = power.geom_basis(mic, rot)

    ident_meas, ident_geom, match = [], [], []
    for b in range(len(power.BANDS)):
        ident_meas.append(power.basis_identifiability(basis[:, :, b]))
        ident_geom.append(power.basis_identifiability(geom))
        # which free-field column each measured column is closest to
        mu = basis[:, :, b] / np.linalg.norm(basis[:, :, b], axis=0, keepdims=True)
        gu = geom / np.linalg.norm(geom, axis=0, keepdims=True)
        match.append(mu.T @ gu)

    payload = {
        "sample_rate": sr,
        "bands": power.BANDS,
        "basis": basis,
        "level_db": rep["level_db"],
        "speed_cos": rep["speed_cos"],
        "spread_db": rep["spread_db"],
        "identifiability_measured": {
            "cond": [float(d["cond"]) for d in ident_meas],
            "max_cos": [float(d["max_cos"]) for d in ident_meas],
            "vif": [d["vif"] for d in ident_meas],
        },
        "identifiability_geom": {
            "cond": float(ident_geom[0]["cond"]),
            "max_cos": float(ident_geom[0]["max_cos"]),
            "vif": ident_geom[0]["vif"],
        },
        "measured_vs_geom_cos": match,
        "additivity": add,
    }
    for b, (lo, hi) in enumerate(power.BANDS):
        print(
            f"{lo:6.0f}-{hi:6.0f} Hz  cond(meas)={float(ident_meas[b]['cond']):6.1f}"
            f"  max_cos={float(ident_meas[b]['max_cos']):.3f}"
            f"  spread={rep['spread_db'][:, b].mean():5.1f} dB"
            f"  excess(add)={add['excess_db'][:, b].mean():+5.2f} dB"
        )
    print(
        f"free-field 1/d^2 basis: cond={float(ident_geom[0]['cond']):.1f}"
        f" max_cos={float(ident_geom[0]['max_cos']):.3f}"
    )
    return payload


def _speeds_on_frames(rid: str, times: np.ndarray, repo_root: str) -> np.ndarray:
    rps, t = data.rotor_speeds(rid, repo_root=repo_root)
    return np.stack([np.interp(times, t, rps[i]) for i in range(rps.shape[0])])


def stage_real(args, bench: dict | None) -> dict:
    out = {}
    basis_all = np.asarray(bench["basis"]) if bench else None
    for rid in args.recordings:
        res = data.local_residual(rid, args.decomp)
        bp = power.band_power(res.audio, res.sample_rate, power.BANDS, frame_s=args.frame_s)
        rps = _speeds_on_frames(rid, bp.times, args.repo_root)
        keep = np.isfinite(rps).all(0) & (rps.max(0) > args.min_rps)
        p, s_rps = bp.power[:, :, keep], rps[:, keep]
        s = power.modulation_regressors(s_rps, exponent=args.exponent)
        s = s / np.maximum(s.mean(-1, keepdims=True), 1e-30)

        # Design collinearity of the four modulations (plus the constant).
        design = np.concatenate([s.T, np.ones((s.shape[1], 1))], axis=1)
        dn = design / np.linalg.norm(design, axis=0, keepdims=True)
        vif = []
        for r in range(s.shape[0]):
            others = np.delete(dn, r, axis=1)
            resid = dn[:, r] - others @ np.linalg.lstsq(others, dn[:, r], rcond=None)[0]
            vif.append(1.0 / max(float(resid @ resid), 1e-12))

        free = power.fit_free_modulation(p, s)
        boot = power.block_bootstrap(
            p, s, n_boot=args.boot, block_frames=args.block, basis=basis_all
        )

        mic, rot = data.geometry_for(data.drone_of(rid), args.cache)
        geom = power.geom_basis(mic, rot)
        arms = {"geom": power.fit_basis_modulation(p, s, geom)}
        for k in (1, 2, 3):
            arms[f"geom_roll{k}"] = power.fit_basis_modulation(p, s, np.roll(geom, k, axis=1))
        agree = None
        if basis_all is not None and data.drone_of(rid) == "dregon":
            arms["bench"] = power.fit_basis_modulation(p, s, basis_all)
            for k in (1, 2, 3):
                arms[f"bench_roll{k}"] = power.fit_basis_modulation(
                    p, s, np.roll(basis_all, k, axis=1)
                )
            agree = power.pattern_agreement(free["gain"], basis_all)

        # THE identifiability test: four rotors that always move together carry
        # one degree of freedom. Only the part of the band power that follows
        # roll / pitch / yaw can ever tell one rotor from another.
        from tracking.rotors import MIXER, NUM_ROTORS

        design, mode_names = power.mode_design(s_rps, MIXER.T / NUM_ROTORS)
        modes = power.mode_information(p, design, n_boot=args.boot, block_frames=args.block)

        # Energy accounting: how much of each mic's band power moves at all.
        modulated = free["contrib"].sum(0) / np.maximum(
            free["contrib"].sum(0) + free["floor"], 1e-30
        )
        rec = {
            "n_frames": int(keep.sum()),
            "bands": power.BANDS,
            "rps_mean": s_rps.mean(-1),
            "modulation_vif": vif,
            "r2_free": free["r2"],
            "share": free["share"],
            "gain": free["gain"],
            "floor": free["floor"],
            "modulated_fraction": modulated,
            "share_q05": boot["share_q05"],
            "share_q95": boot["share_q95"],
            "arms_r2": {k: v["r2"] for k, v in arms.items()},
            "arms_alpha": {k: v["alpha"] for k, v in arms.items()},
            "mode_names": mode_names,
            "mode_r2_common": modes["r2_common"],
            "mode_r2_full": modes["r2_full"],
            "mode_delta_r2": modes["delta_r2"],
            "mode_delta_r2_null_q95": modes["delta_r2_null_q95"],
            "mode_coef": modes["coef"],
        }
        if agree is not None:
            rec |= {
                "pattern_cos": agree["cos"],
                "pattern_cos_mean": agree["cos_mean"],
                "pattern_cos_perm": agree["cos_perm"],
                "pattern_cos_q05": boot["cos_q05"],
                "pattern_cos_q95": boot["cos_q95"],
            }
        out[rid] = rec

        print(f"\n=== {rid}: {int(keep.sum())} frames, modulation VIF {np.round(vif, 1)}")
        for b, (lo, hi) in enumerate(power.BANDS):
            sh = free["share"][:, :, b].mean(1)
            print(
                f" {lo:6.0f}-{hi:6.0f} Hz r2={free['r2'][:, b].mean():.3f}"
                f" mod.frac={modulated[:, b].mean():.2f}"
                f" share={np.round(sh, 3)}"
                + (f" cos={agree['cos'][:, b].mean():+.3f}" if agree is not None else "")
                + (f" perm={agree['cos_perm'][1:, b].max():+.3f}" if agree is not None else "")
            )
        print(" control modes (mean over mics): r2 common-only / +differential / null q95")
        for b, (lo, hi) in enumerate(power.BANDS):
            print(
                f" {lo:6.0f}-{hi:6.0f} Hz  r2c={modes['r2_common'][:, b].mean():.3f}"
                f"  d_r2={modes['delta_r2'][:, b].mean():.4f}"
                f"  null={modes['delta_r2_null_q95'][:, b].mean():.4f}"
            )
        print(" arm r2 per band:")
        for k, v in arms.items():
            print(f"   {k:14s} {np.round(v['r2'], 4)}")
    return out


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["coherence", "bench", "real", "all"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--decomp", default="results/vk_decompose_v2")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--cache", default=".cache/vk_decompose")
    ap.add_argument("--recordings", nargs="*", default=RECORDINGS)
    ap.add_argument("--frame-s", type=float, default=0.25)
    ap.add_argument("--exponent", type=float, default=5.0)
    ap.add_argument("--min-rps", type=float, default=20.0)
    ap.add_argument("--throttle", type=int, default=70)
    ap.add_argument("--boot", type=int, default=48)
    ap.add_argument("--block", type=int, default=40)
    ap.add_argument("--coh-start", type=float, default=20.0)
    ap.add_argument("--coh-seconds", type=float, default=20.0)
    args = ap.parse_args()

    bench = None
    if args.stage in ("coherence", "all"):
        _write(args.out, "coherence.json", stage_coherence(args))
    if args.stage in ("bench", "all"):
        bench = stage_bench(args)
        _write(args.out, "bench.json", bench)
    if args.stage in ("real", "all"):
        if bench is None and (args.out / "bench.json").exists():
            bench = json.loads((args.out / "bench.json").read_text())
        _write(args.out, "real.json", stage_real(args, bench))


if __name__ == "__main__":
    main()
