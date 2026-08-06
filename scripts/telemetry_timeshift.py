#!/usr/bin/env python3
"""Is DREGON's ``motors_measured`` late, and is it late by a DIFFERENT amount at
each microphone? (GitHub issue 17, phase 6e.)

Two claims live inside the one theory, they are of very different sizes, and
they need two different instruments. This driver builds both and reports them
side by side, because the honest answer to the second one is a resolution
statement rather than a number.

**The common part.** A period counter reports the revolution that has just
finished, so its reading at log time ``t`` describes the shaft some ``tau``
earlier; the logging hold adds more. The correction is therefore to read the
telemetry at ``t + tau`` — the trace moves EARLIER — and ``shift:`` in the
phase-6a candidate language is exactly that (``scripts/telemetry_fitness.py``).
Sign convention, once, everywhere below: **positive tau means the telemetry
LAGS the shaft by tau**, and the candidate ``r(t + tau)`` removes that lag.

The instrument is phase 6d's ridge, unchanged and at the frozen settings. It can
see the shift at all only because a shift acts through the trajectory's own
slope: the carrier moves by ``tau dr/dt``, and on these windows the 5 Hz
low-passed ``|dr/dt|`` is 8.5-34 rev/s^2, so 20 ms of lag displaces the carrier
by 0.17-0.67 rev/s. That is one to seven times the ridge's own 0.10 rev/s
window, so the common part is inside the instrument's reach. It is also
orthogonal in SHAPE to the scale the campaign already fitted — a scale error is
proportional to ``r``, a lag is proportional to ``dr/dt``, and on a cruise window
those two are uncorrelated — which is why ``--scales`` sweeps the second axis
instead of assuming it away.

**The per-microphone part.** Sound from rotor ``j`` reaches microphone ``c``
after ``d_cj / 343``, so the carrier that fits microphone ``c`` is the shaft
history delayed by that much: the best ``tau`` should fall as distance rises,
with slope ``-1/343`` s/m. On DREGON's rig the whole spread of ``d_cj`` is
0.22-0.40 m, i.e. 0.48 ms between the nearest and the farthest cell of one
rotor and **0.156 ms** between microphones once the four rotors are averaged.
Through the same ``tau dr/dt`` channel that is 0.002-0.005 rev/s of carrier
displacement — a twentieth of the ridge window, on an eighth of the cells. The
ridge cannot resolve it and this script does not pretend otherwise: it measures
the per-microphone best ``tau`` anyway, and reports its scatter beside that
prediction so the comparison is a number rather than an excuse.

**The instrument that CAN resolve it** is ``--mode tdoa``. A propagation delay
is a phase ramp across the comb, not a rate error: harmonic ``k`` of rotor ``j``
arrives at microphone ``c`` with phase ``-2 pi k rate_j d_cj / 343`` relative to
the reference microphone, so the mean harmonic-to-harmonic phase INCREMENT of
the cross-spectrum gives the inter-mic delay directly,

    delay_cj = -mean_k wrap(psi_{k+1} - psi_k) / (2 pi rate_j)

with no unwrapping ambiguity while the delay stays under ``1 / (2 rate)`` =
6.25 ms, against delays of at most 0.5 ms. The phases come from the same
demodulated envelopes the ridge reads, at the envelope-spectrum bin the comb's
own line sits in (found once per (rotor, block) by a joint scan over a rate
offset, never per harmonic — a per-harmonic peak search is the bias that has
already cost this campaign two claims). ``--self-test`` injects a known delay
into a synthetic comb and requires it back, which is also what pins the sign.

Modes:
  ridge      the tau x scale profile, gridrun, pooled + per-mic + per-rotor
  tdoa       the cross-channel comb phase, per (rotor, mic), against geometry
  report     read both unit trees, produce the tables and the verdict inputs

Examples:
  python scripts/telemetry_timeshift.py --mode ridge --pilot --jobs 6
  python scripts/telemetry_timeshift.py --mode tdoa --jobs 4
  python scripts/telemetry_timeshift.py --mode report
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
sys.path.insert(0, str(ROOT / "scripts"))

from telemetry_fitness import (  # noqa: E402
    ALL_WINDOWS,
    DREGON_WINDOWS,
    FLY124_WINDOWS,
    PARTNER,
    _load,
    _uid,
    build_candidate,
    build_preps,
    prep_sha1,
    resolve_prep_dir,
)
from telemetry_report import _argmin_parabola, boot_ci  # noqa: E402

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

#: Speed of sound, m/s. One number, stated once, used by the prediction and by
#: the tdoa regression alike.
C_SOUND = 343.0
#: The scale phase 6d's ridge profile settled on, as a multiplier. The second
#: axis of the grid exists so that neither correction can hide inside the other.
SCALE_6D = 0.99317  # -0.683 %
OUT_DEFAULT = "results/telemetry_timeshift"
CONTROLS = ("on", "offcomb")
LIB_CONTROL = {"on": "none", "offcomb": "offcomb"}
#: Frozen at phase 6d's reading settings, so a ridge here is comparable to a
#: ridge there without a caveat.
PROTO = {"k_min": 2, "k_max": 40, "b0": 1.0, "fs_env": 250.0, "n_blocks": 8, "gate_band_frac": 0.25}

PROTOCOL = {
    "dataset": "beatvk-valid-raw@54849c13ed3a",
    "statistic": "phase-6d ridge concentration (dB over a local floor, more is "
    "better), at the 6d reading settings (b0 = 1 rev/s, gate_band_frac = 0.25, "
    "k = 2..40, fs_env = 250 Hz, 8 blocks)",
    "tau_convention": "positive tau = the telemetry LAGS the shaft by tau; the "
    "candidate is r(t + tau), i.e. the trace read tau later / moved earlier",
    "sensitivity": "a shift acts only through the slope: the carrier moves by "
    "tau * dr/dt. Every unit carries drdt_rms so the reach of the instrument "
    "travels with the number",
    "controls": {
        "on": "the measurement",
        "offcomb": "half-integer comb (k + 0.5) g(t): no rotor line can exist",
    },
    "tdoa": "inter-mic delay from the mean harmonic-to-harmonic phase increment "
    "of the comb cross-spectrum, read at the line bin of a per-(rotor, block) "
    "joint rate-offset scan; null = the same estimator on the half-integer comb",
}


# ---------------------------------------------------------------------------
# geometry: the prediction the whole theory rests on


def dregon_delays() -> tuple[np.ndarray, np.ndarray]:
    """``(d (8, 4) metres, tau_pred (8, 4) seconds)`` rotor-to-mic propagation.

    Imported lazily and from ``scripts``, not from ``src/tracking`` — the
    tracking package may not import ``data_processing`` (the purity rule), and a
    driver may.
    """
    from data_processing.sources import geometry

    mic, rot = geometry("DREGON")
    d = np.linalg.norm(np.asarray(mic)[:, None, :] - np.asarray(rot)[None, :, :], axis=-1)
    return d, d / C_SOUND


def geometry_summary() -> dict[str, Any]:
    """The predicted delays and the two spreads that decide what is measurable."""
    try:
        d, tau = dregon_delays()
    except Exception as exc:  # a cluster worktree need not have the raw tree
        return {"available": False, "error": str(exc)}
    per_mic = tau.mean(axis=1)
    return {
        "available": True,
        "c_sound": C_SOUND,
        "d_m": np.round(d, 4).tolist(),
        "tau_ms": np.round(tau * 1e3, 4).tolist(),
        "tau_mean_ms": round(float(tau.mean() * 1e3), 4),
        # What a per-(rotor, mic) best-tau would have to resolve...
        "spread_rotor_mic_ms": round(float((tau.max(axis=0) - tau.min(axis=0)).mean() * 1e3), 4),
        # ...and what a per-MIC best-tau would have to resolve, which is smaller
        # still because the array is symmetric about the rotor plane.
        "spread_mic_ms": round(float((per_mic.max() - per_mic.min()) * 1e3), 4),
        "per_mic_ms": np.round(per_mic * 1e3, 4).tolist(),
    }


# ---------------------------------------------------------------------------
# mode: ridge — the tau x scale profile


def _spec(tau_s: float, scale: float) -> str:
    """The candidate spec for one grid point, in the phase-6a language."""
    parts = ["lp:5"]
    if scale != 1.0:
        parts.append(f"scale:{scale:.6g}")
    parts.append(f"shift:{tau_s:.6g}")
    return "+".join(parts)


def drdt_rms(r: np.ndarray, ft: np.ndarray) -> float:
    """RMS slope of the 5 Hz pre-smoothed telemetry, rev/s^2.

    This is the gain of the whole experiment: a lag ``tau`` displaces the
    carrier by ``tau`` times this. It must be read on the SMOOTHED trajectory —
    a raw gradient of the 0.269 rev/s staircase on a 32 ms grid reads 8 rev/s^2
    of pure quantisation noise.
    """
    from tracking.telemetry_refit import presmooth

    return float(np.sqrt((np.gradient(presmooth(r, ft, 5.0), ft, axis=1) ** 2).mean()))


def ridge_worker(unit: Unit) -> dict[str, Any]:
    """One (window, tau, scale, control): pooled, per-mic and per-rotor ridge."""
    from tracking.fitness import FitnessConfig, Holdout, score_cells, window_cells

    p = unit.params
    key, control = str(p["key"]), str(p["control"])
    tau, scale = float(p["tau_s"]), float(p["scale"])
    win = _load(key)
    cfg = FitnessConfig(
        k_min=int(p["k_min"]),
        k_max=int(p["k_max"]),
        b0_revs=float(p["b0"]),
        fs_env=float(p["fs_env"]),
        n_blocks=int(p["n_blocks"]),
        gate_band_frac=float(p["gate_band_frac"]),
    )
    spec = _spec(tau, scale)
    cand = build_candidate(spec, win["r"], win["ft"], key)
    # The confound, measured rather than argued: a lag on a trajectory with a
    # sustained trend LOOKS like a scale (it adds tau dr/dt, and a trend makes
    # that a constant). ``d_mean_pct`` is what the shift did to the mean rate,
    # in the same units as the campaign's scale, so the two axes can be read
    # against each other instead of assumed independent.
    base = build_candidate("lp:5", win["r"], win["ft"], key)
    d = np.asarray(cand) - np.asarray(base)
    mean_rate = float(np.mean(win["r"]))
    cells = window_cells(
        win["audio"], win["ft"], cand, win["r"], cfg=cfg, control=LIB_CONTROL[control]
    )
    n_ch = int(cells[0].shape[0])

    def one(ho: Holdout) -> dict[str, Any]:
        s = score_cells(cells, ho, cfg=cfg)
        return {
            "ridge": s.as_dict()["ridge"],
            "broadband": s.as_dict()["broadband"],
            "n_cells_ridge": s.n_cells_ridge,
            "per_rotor": {i: v["ridge"] for i, v in s.per_rotor.items()},
        }

    pooled = one(Holdout.none())
    # "Only microphone c" is the complement of "fit every other microphone" —
    # the hold-out object already expresses it, so the per-mic reading is the
    # same aggregation over the same cells, not a second statistic.
    per_mic = {
        c: one(Holdout(kind="channels", fit=tuple(i for i in range(n_ch) if i != c)))
        for c in range(n_ch)
    }
    return {
        "key": key,
        "recording": key.split("__")[0],
        "regime": win["regime"],
        "dataset": "fly124" if key.startswith("FLY124") else "dregon",
        "tau_s": tau,
        "tau_ms": round(tau * 1e3, 4),
        "scale": scale,
        "scale_pct": round((scale - 1.0) * 100.0, 4),
        "control": control,
        "candidate": spec,
        "pooled": pooled,
        "per_mic": per_mic,
        "n_ch": n_ch,
        "admit_frac_ridge": cells[0].diag.get("admit_frac_ridge"),
        "line_share_ridge": cells[0].diag.get("line_share_ridge"),
        "rotor_mean_rev_s": [round(float(v), 3) for v in win["r"].mean(axis=1)],
        "drdt_rms": round(drdt_rms(win["r"], win["ft"]), 4),
        "d_mean_pct": round(100.0 * float(np.mean(d)) / mean_rate, 4),
        "d_rms": round(float(np.sqrt(np.mean(d**2))), 4),
        "prep_sha1": prep_sha1(key),
    }


# ---------------------------------------------------------------------------
# mode: tdoa — the instrument that can actually resolve a fraction of a ms


def _line_bins(
    pw: np.ndarray, freqs: np.ndarray, ks: np.ndarray, dr_step: float, n_step: int = 60
) -> tuple[float, np.ndarray]:
    """``(the residual rate offset in rev/s, (K,) bin index of each line)``.

    The comb's residual rate error is ONE number per (rotor, block): harmonic
    ``k`` sits at ``k dr`` Hz. So the offset is found by a joint scan that sums
    the harmonics' power along each candidate comb — never by a peak search per
    harmonic, which returns about half a window width on pure noise and which
    has already cost this campaign two published claims.
    """
    grid = np.arange(-n_step, n_step + 1) * dr_step  # rev/s
    kk = np.arange(ks.size)
    idx = np.stack(
        [np.abs(freqs[None, :] - (ks * dr)[:, None]).argmin(axis=1) for dr in grid]
    )  # (G, K)
    score = np.asarray([float(np.sum(pw[:, kk, idx[g]])) for g in range(grid.size)])
    best = int(np.argmax(score))
    return float(grid[best]), idx[best]


def tdoa_worker(unit: Unit) -> dict[str, Any]:
    """Inter-mic delay per (rotor, mic) from the comb's cross-channel phase."""
    from tracking.comb_displacement import demod_comb_bank
    from tracking.fitness import FitnessConfig, _blocks, admission

    p = unit.params
    key, control = str(p["key"]), str(p["control"])
    win = _load(key)
    cfg = FitnessConfig(
        k_min=int(p["k_min"]),
        k_max=int(p["k_max"]),
        b0_revs=float(p["b0"]),
        fs_env=float(p["fs_env"]),
        n_blocks=int(p["n_blocks"]),
        gate_band_frac=float(p["gate_band_frac"]),
    )
    cand = build_candidate(str(p["candidate"]), win["r"], win["ft"], key)
    out = measure_tdoa(
        win["audio"],
        win["ft"],
        cand,
        win["r"],
        cfg=cfg,
        half=(control == "offcomb"),
        dr_step=float(p["dr_step"]),
        gate=bool(p.get("gate", True)),
        demod=demod_comb_bank,
        blocks_fn=_blocks,
        admission_fn=admission,
    )
    return {
        "key": key,
        "recording": key.split("__")[0],
        "regime": win["regime"],
        "dataset": "fly124" if key.startswith("FLY124") else "dregon",
        "control": control,
        "candidate": str(p["candidate"]),
        "gate": bool(p.get("gate", True)),
        "rotor_mean_rev_s": [round(float(v), 3) for v in win["r"].mean(axis=1)],
        "prep_sha1": prep_sha1(key),
        **out,
    }


def measure_tdoa(
    audio: np.ndarray,
    ft: np.ndarray,
    carrier: np.ndarray,
    reference: np.ndarray,
    *,
    cfg: Any,
    half: bool = False,
    dr_step: float = 0.02,
    ref_ch: int = 0,
    gate: bool = True,
    demod: Any = None,
    blocks_fn: Any = None,
    admission_fn: Any = None,
) -> dict[str, Any]:
    """``delay_ms (R, C)`` relative to ``ref_ch``, plus the weights behind it.

    One rotor at a time. Per block the envelope spectrum is taken with the same
    Hann taper the ridge uses, the comb's own line bin is found jointly over
    harmonics, and the cross-spectrum against ``ref_ch`` is accumulated across
    blocks COHERENTLY — the geometry-induced phase is the one thing that does
    not change from block to block, so summing complex values averages
    everything else down.
    """
    if demod is None:  # pragma: no cover - the workers always pass them
        from tracking.comb_displacement import demod_comb_bank as demod
    if blocks_fn is None:  # pragma: no cover
        from tracking.fitness import _blocks as blocks_fn
    if admission_fn is None:  # pragma: no cover
        from tracking.fitness import admission as admission_fn

    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))[: cfg.max_channels]
    ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    cand = np.atleast_2d(np.asarray(carrier, dtype=np.float64))
    ks = np.asarray(cfg.ks, dtype=np.float64)
    n_env = x.shape[-1] // cfg.stride
    blocks = blocks_fn(n_env, cfg)
    length = blocks[0].stop - blocks[0].start
    freqs = np.fft.fftfreq(length, d=1.0 / cfg.fs_env)
    taper = np.hanning(length)
    taper /= np.sqrt(np.mean(taper**2))

    delay_ms: list[list[float | None]] = []
    weight: list[list[float]] = []
    n_pairs: list[list[int]] = []
    for rot in range(cand.shape[0]):
        rate_ref = float(np.mean(ref[rot]))
        if rate_ref < cfg.min_rate:
            continue
        band = cfg.band_hz(rate_ref)
        z, _ = demod(x, cand[rot], ft, cfg.ks, cfg=cfg.displacement(), half=half, band_hz_k=band)
        res_hz = cfg.fs_env / length
        dc_hz = np.minimum(np.maximum(cfg.dc_revs * ks, cfg.dc_bins * res_hz), 0.9 * band)
        _, admit_ridge, _ = admission_fn(
            ref, ft, rot, blocks, band, cfg=cfg, rate_ref=rate_ref, dc_hz=dc_hz
        )
        if not gate:
            # The ungated arm. DREGON's ridge gate leaves 4-8 harmonic pairs,
            # and the delay's own error falls as 1/sqrt(pairs), so the gate is
            # worth turning off ONCE to see whether coverage or the estimator
            # is the limit. It admits contested harmonics, whose phase is a
            # mixture of two rotors at two directions — a bias, reported as a
            # separate arm rather than folded into the measurement.
            admit_ridge = np.ones_like(admit_ridge)
        n_ch = z.shape[0]
        # (C, K) complex cross-spectrum against the reference microphone,
        # accumulated over blocks at each block's own line bin.
        cross = np.zeros((n_ch, ks.size), dtype=np.complex128)
        power = np.zeros(ks.size)
        for b, sl in enumerate(blocks):
            zb = np.asarray(z[:, :, sl], dtype=np.complex128) * taper
            Z = np.fft.fft(zb, axis=-1)  # (C, K, L)
            pw = np.abs(Z) ** 2
            _, bins = _line_bins(pw, freqs, ks, dr_step)
            val = Z[:, np.arange(ks.size), bins]  # (C, K)
            ok = admit_ridge[:, b]
            cross += np.where(ok[None, :], val * np.conj(val[ref_ch])[None, :], 0.0)
            power += np.where(ok, np.abs(val[ref_ch]) ** 2, 0.0)
        psi = np.angle(cross)  # (C, K)
        w = np.minimum(np.abs(cross)[:, :-1], np.abs(cross)[:, 1:])  # (C, K-1)
        # Harmonic-to-harmonic increment: 2 pi rate * delay per unit k, wrapped
        # into (-pi, pi]. Unambiguous while |delay| < 1 / (2 rate) = 6.25 ms.
        dpsi = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1])))
        good = np.asarray(power > 0)[:-1] & np.asarray(power > 0)[1:]
        w = w * good[None, :]
        tot = w.sum(axis=1)
        mean_dpsi = np.where(tot > 0, (w * dpsi).sum(axis=1) / np.maximum(tot, 1e-300), np.nan)
        delay = -mean_dpsi / (2.0 * np.pi * rate_ref)
        delay_ms.append([None if not np.isfinite(v) else round(float(v) * 1e3, 5) for v in delay])
        weight.append([round(float(v), 6) for v in tot])
        n_pairs.append([int(v) for v in (w > 0).sum(axis=1)])
    return {
        "delay_ms": delay_ms,
        "weight": weight,
        "n_pairs": n_pairs,
        "ref_ch": ref_ch,
        "half": bool(half),
    }


def self_test(seed: int = 0) -> int:
    """Inject a known inter-mic delay into a synthetic comb and require it back.

    This is what pins the SIGN of the estimator: the whole per-mic claim is a
    sign statement (farther microphone, later arrival, smaller best tau), and a
    sign convention taken on faith from a demodulation kernel is a coin flip.
    """
    from tracking.fitness import FitnessConfig

    rng = np.random.default_rng(seed)
    cfg = FitnessConfig(k_min=2, k_max=25, b0_revs=1.0, fs_env=250.0, n_blocks=4)
    sr, dur, rate = cfg.sr, 8.0, 80.0
    t = np.arange(int(sr * dur)) / sr
    ft = np.arange(0, dur, 0.032)
    r = rate + 2.0 * np.sin(2 * np.pi * 0.3 * ft)
    rate_t = rate + 2.0 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(rate_t) / sr
    true_ms = np.array([0.0, 0.35, -0.20, 0.80])
    chans = []
    for d_ms in true_ms:
        # A pure propagation delay: the same waveform, evaluated d earlier.
        ph = phase - 2 * np.pi * rate_t * (d_ms * 1e-3)
        sig = sum(np.cos(k * ph) / k for k in range(2, 26))
        chans.append(sig + 0.30 * rng.standard_normal(t.size))
    audio = np.asarray(chans)
    ref = r[None, :]
    got = measure_tdoa(audio, ft, ref, ref, cfg=cfg, dr_step=0.02)
    meas = np.asarray([v if v is not None else np.nan for v in got["delay_ms"][0]])
    err = np.abs(meas - true_ms)
    ok = bool(np.nanmax(err) < 0.03)
    print("[self-test] injected ms:", np.round(true_ms, 3).tolist())
    print("[self-test] measured ms:", np.round(meas, 4).tolist())
    print(f"[self-test] max error {np.nanmax(err) * 1e3:.1f} us — {'PASS' if ok else 'FAIL'}")
    null = measure_tdoa(audio, ft, ref, ref, cfg=cfg, dr_step=0.02, half=True)
    nm = np.asarray([v if v is not None else np.nan for v in null["delay_ms"][0]])
    print("[self-test] half-integer null ms:", np.round(nm, 4).tolist())
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# report


def _profile(rows: list[dict[str, Any]], pick: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(tau grid, (n_window, n_tau) values)`` for one selection of units."""
    taus = np.array(sorted({r["tau_s"] for r in rows}))
    keys = sorted({r["key"] for r in rows})
    mat = np.full((len(keys), taus.size), np.nan)
    for r in rows:
        v = pick(r)
        if v is None:
            continue
        mat[keys.index(r["key"]), int(np.argmin(np.abs(taus - r["tau_s"])))] = float(v)
    return taus, mat


def _best_tau(taus: np.ndarray, y: np.ndarray) -> tuple[float | None, str]:
    """Sub-grid MAXIMUM of a ridge curve, via the report's own basin parabola."""
    return _argmin_parabola(taus * 1e3, -np.asarray(y, dtype=np.float64))


def ridge_section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"global": {}, "per_mic": {}, "per_rotor": {}}
    groups = sorted({_group(r) for r in rows})
    scales = sorted({r["scale"] for r in rows})
    for grp in groups:
        for sc in scales:
            for ctl in CONTROLS:
                sel = [
                    r for r in rows if _group(r) == grp and r["scale"] == sc and r["control"] == ctl
                ]
                if not sel:
                    continue
                taus, mat = _profile(sel, lambda r: r["pooled"]["ridge"])
                keys = sorted({r["key"] for r in sel})
                mean = np.nanmean(mat, axis=0)
                tau_hat, note = _best_tau(taus, mean)
                per_win = [_best_tau(taus, mat[i])[0] for i in range(mat.shape[0])]
                tag = f"{grp}|scale{(sc - 1) * 100:+.3f}|{ctl}"
                out["global"][tag] = {
                    "n_windows": len(keys),
                    "tau_ms": tau_hat,
                    "note": note,
                    "ci_windows": boot_ci([v for v in per_win if v is not None]),
                    "per_window_tau_ms": dict(zip(keys, per_win, strict=True)),
                    "depth_db": round(float(np.nanmax(mean) - np.nanmin(mean)), 4),
                    "ridge_at_best": round(float(np.nanmax(mean)), 4),
                    "ridge_at_zero": round(float(mean[int(np.argmin(np.abs(taus)))]), 4),
                    "grid_ms": np.round(taus * 1e3, 3).tolist(),
                    "curve": np.round(mean, 4).tolist(),
                }
                if ctl == "on":
                    n_ch = int(sel[0]["n_ch"])
                    mics: dict[str, Any] = {}
                    for c in range(n_ch):
                        _, m = _profile(sel, lambda r, c=c: r["per_mic"][str(c)]["ridge"])
                        t_hat, _n = _best_tau(taus, np.nanmean(m, axis=0))
                        mics[str(c)] = {
                            "tau_ms": t_hat,
                            "ci_windows": boot_ci(
                                [
                                    v
                                    for i in range(m.shape[0])
                                    if (v := _best_tau(taus, m[i])[0]) is not None
                                ]
                            ),
                        }
                    out["per_mic"][tag] = mics
                    rots: dict[str, Any] = {}
                    for j in sorted({k for r in sel for k in r["pooled"]["per_rotor"]}):
                        _, m = _profile(sel, lambda r, j=j: r["pooled"]["per_rotor"].get(j))
                        rots[j] = {"tau_ms": _best_tau(taus, np.nanmean(m, axis=0))[0]}
                    out["per_rotor"][tag] = rots
    return out


def _group(r: dict[str, Any]) -> str:
    return r["dataset"] if r["dataset"] == "dregon" else f"fly124-{r.get('regime', '?')}"


def per_mic_section(ridge: dict[str, Any], geo: dict[str, Any]) -> dict[str, Any]:
    """Do the per-mic best taus track the geometry, and could they possibly?"""
    out: dict[str, Any] = {"geometry": geo}
    if not geo.get("available"):
        return out
    pred = np.asarray(geo["per_mic_ms"], dtype=np.float64)
    for tag, mics in ridge.get("per_mic", {}).items():
        if not tag.startswith("dregon"):
            continue
        got = np.asarray(
            [
                mics[str(c)]["tau_ms"] if mics.get(str(c), {}).get("tau_ms") is not None else np.nan
                for c in range(len(pred))
            ],
            dtype=np.float64,
        )
        ok = np.isfinite(got)
        if ok.sum() < 3:
            continue
        r = float(np.corrcoef(pred[ok], got[ok])[0, 1])
        slope, icept = np.polyfit(pred[ok], got[ok], 1)
        out[tag] = {
            "pred_ms": np.round(pred, 4).tolist(),
            "meas_ms": np.round(got, 3).tolist(),
            "pearson_r": round(r, 4),
            "slope": round(float(slope), 3),
            "slope_predicted": 1.0,
            "intercept_ms": round(float(icept), 3),
            "meas_spread_ms": round(float(np.nanmax(got) - np.nanmin(got)), 3),
            "pred_spread_ms": round(float(pred.max() - pred.min()), 4),
        }
    return out


#: Which rig each dataset's microphones belong to. FLY124 is not a DREGON
#: control here: it is a SECOND rig with its own geometry, and it is the
#: positive control for the estimator itself — its rotors are 1.65 rev/s apart,
#: so its comb is resolvable where DREGON's twin pair is not.
RIG_OF = {"dregon": "DREGON", "fly124": "michaels"}


def _rig_pred(rig: str, ref: int = 0) -> np.ndarray | None:
    """``(R, C)`` predicted inter-mic delay in ms, relative to ``ref``."""
    from data_processing.sources import geometry

    try:
        mic, rot = geometry(rig)
    except Exception:
        return None
    d = np.linalg.norm(np.asarray(mic)[:, None, :] - np.asarray(rot)[None, :, :], axis=-1)
    return ((d - d[ref]) / C_SOUND * 1e3).T


def _fit(pred: np.ndarray, meas: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(pred) & np.isfinite(meas)
    if m.sum() < 4 or np.std(pred[m]) == 0:
        return float("nan"), float("nan")
    return (
        float(np.corrcoef(pred[m], meas[m])[0, 1]),
        float(np.polyfit(pred[m], meas[m], 1)[0]),
    )


def tdoa_section(rows: list[dict[str, Any]], geo: dict[str, Any]) -> dict[str, Any]:
    """Measured inter-mic delays against each rig's own geometry prediction.

    Two things are asked of every block, and the second exists because the
    first has a trap. (1) Does the identity rotor labelling correlate? (2) Does
    ANY of the 24 rotor labellings, and is that best-of-24 beyond what
    relabelling noise produces? Rotor slot ``j`` of the telemetry is not
    guaranteed to be rotor ``j`` of the geometry file, so the search is
    necessary — and a best-of-24 without its own null is exactly the kind of
    selected maximum this campaign has already had to withdraw once. The null
    permutes the MICROPHONE labels, which destroys the spatial pattern and
    keeps every value, then takes the same best-of-24.
    """
    import itertools

    out: dict[str, Any] = {"geometry": geo}
    for ds, rig in RIG_OF.items():
        pred = _rig_pred(rig)
        if pred is None:
            continue
        for ctl in CONTROLS:
            for regime in (None, "cruise"):
                sel = [
                    r
                    for r in rows
                    if r["dataset"] == ds
                    and r["control"] == ctl
                    and (regime is None or r.get("regime") == regime)
                ]
                if len(sel) < 2:
                    continue
                arr = np.asarray(
                    [
                        [[np.nan if v is None else v for v in row] for row in r["delay_ms"]]
                        for r in sel
                    ],
                    dtype=np.float64,
                )  # (W, R, C)
                med = np.nanmedian(arr, axis=0)
                perms = list(itertools.permutations(range(pred.shape[0])))
                stats = [_fit(pred[list(p)], med) for p in perms]
                best = int(np.nanargmax([s[0] for s in stats]))
                rng = np.random.default_rng(0)
                null = []
                for _ in range(400):
                    sh = med[:, rng.permutation(med.shape[1])]
                    null.append(np.nanmax([_fit(pred[list(p)], sh)[0] for p in perms]))
                nullmax = np.asarray(null)
                r_id, s_id = _fit(pred, med)
                r_b, s_b = stats[best]
                out[f"{ds}|{ctl}|{regime or 'all'}"] = {
                    "rig": rig,
                    "n_windows": len(sel),
                    "median_n_pairs": int(np.median([np.median(r["n_pairs"]) for r in sel])),
                    "identity": {"pearson_r": round(r_id, 4), "slope": round(s_id, 4)},
                    "best_perm": list(perms[best]),
                    "best": {"pearson_r": round(r_b, 4), "slope": round(s_b, 4)},
                    "slope_predicted": 1.0,
                    "null_best_of_24": {
                        "mean": round(float(np.nanmean(nullmax)), 4),
                        "p95": round(float(np.nanpercentile(nullmax, 95)), 4),
                        "p_value": round(float(np.mean(nullmax >= r_b)), 4),
                    },
                    "window_sd_ms": round(float(np.nanmean(np.nanstd(arr, axis=0))), 4),
                    "pred_spread_ms": round(float(np.nanmax(pred) - np.nanmin(pred)), 4),
                    "meas_spread_ms": round(float(np.nanmax(med) - np.nanmin(med)), 4),
                }
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kind = "tdoa" if rows and "delay_ms" in rows[0] else "ridge"
    geo = geometry_summary()
    if kind == "tdoa":
        return {"protocol": PROTOCOL, "kind": kind, "tdoa": tdoa_section(rows, geo)}
    ridge = ridge_section(rows)
    return {
        "protocol": PROTOCOL,
        "kind": kind,
        "sensitivity": {
            r["key"]: {"drdt_rms": r["drdt_rms"], "rev_s_per_10ms": round(r["drdt_rms"] * 0.01, 4)}
            for r in rows
        },
        "ridge": ridge,
        "per_mic_vs_geometry": per_mic_section(ridge, geo),
    }


def print_ridge(summary: dict[str, Any]) -> None:
    g = summary.get("ridge", {}).get("global", {})
    print(
        f"\n{'group|scale|control':44s} {'tau* ms':>9s} {'CI (windows)':>22s} "
        f"{'depth dB':>9s} {'ridge@tau*':>11s} {'ridge@0':>9s}"
    )
    for tag in sorted(g):
        m = g[tag]
        ci = m["ci_windows"]
        span = f"[{ci['lo']:+.1f},{ci['hi']:+.1f}]" if ci.get("lo") is not None else "—"
        val = f"{m['tau_ms']:+.2f}" if m["tau_ms"] is not None else "—"
        print(
            f"{tag:44s} {val:>9s} {span:>22s} {m['depth_db']:9.3f} "
            f"{m['ridge_at_best']:11.3f} {m['ridge_at_zero']:9.3f}"
        )
    pm = summary.get("per_mic_vs_geometry", {})
    for tag, m in pm.items():
        if tag == "geometry" or not isinstance(m, dict) or "pearson_r" not in m:
            continue
        print(f"\nper-mic tau vs geometry [{tag}]")
        print(
            f"  predicted spread {m['pred_spread_ms']:.4f} ms | "
            f"measured spread {m['meas_spread_ms']:.3f} ms | "
            f"r = {m['pearson_r']:+.3f} | slope {m['slope']:+.2f} (predicted 1.0)"
        )


def print_tdoa(summary: dict[str, Any]) -> None:
    print(
        f"\n{'block':30s} {'rig':9s} {'W':>2s} {'k-pairs':>7s} {'r(id)':>7s} "
        f"{'r(best)':>8s} {'slope':>7s} {'perm':14s} {'null p95':>8s} {'p':>6s} {'sd ms':>7s}"
    )
    for tag, m in summary.get("tdoa", {}).items():
        if not isinstance(m, dict) or "best" not in m:
            continue
        n = m["null_best_of_24"]
        print(
            f"{tag:30s} {m['rig']:9s} {m['n_windows']:2d} {m['median_n_pairs']:7d} "
            f"{m['identity']['pearson_r']:+7.3f} {m['best']['pearson_r']:+8.3f} "
            f"{m['best']['slope']:+7.3f} {str(m['best_perm']):14s} "
            f"{n['p95']:8.3f} {n['p_value']:6.3f} {m['window_sd_ms']:7.3f}"
        )


# ---------------------------------------------------------------------------
# main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", default="ridge", choices=("ridge", "tdoa", "report"))
    ap.add_argument("--dataset", default="all", choices=("dregon", "fly124", "all"))
    ap.add_argument("--windows", default="")
    ap.add_argument("--tau-ms", default="-40:120:4", help="lo:hi:step in milliseconds")
    ap.add_argument(
        "--scales",
        default=f"1.0,{SCALE_6D}",
        help="rate-scale arms; the second axis, so a lag cannot hide inside a scale",
    )
    ap.add_argument("--controls", default=",".join(CONTROLS))
    ap.add_argument("--candidate", default=f"lp:5+scale:{SCALE_6D}", help="tdoa mode carrier")
    ap.add_argument("--dr-step", type=float, default=0.02, help="tdoa line-scan step, rev/s")
    ap.add_argument(
        "--tdoa-gate",
        default="ridge",
        choices=("ridge", "none"),
        help="tdoa: 'ridge' keeps the phase-6d gate (4-8 harmonic pairs on DREGON); "
        "'none' takes every harmonic, trading twin contamination for pair count",
    )
    ap.add_argument("--pilot", action="store_true", help="3 windows, 1 scale, on-comb only")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--build-preps", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--report-dirs", default="", help="report mode: comma-separated unit trees to read"
    )
    add_gridrun_args(ap, jobs=6)
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    out = Path(args.out or f"{OUT_DEFAULT}/{args.mode}")
    if args.mode == "report":
        # NAMED unit trees only. Globbing the output root would silently mix a
        # pilot's coarse grid into the campaign's, which is the kind of quiet
        # provenance loss this campaign has already had to unpick once.
        for d in [Path(p) for p in (args.report_dirs or f"{OUT_DEFAULT}/ridge").split(",")]:
            raw = d / "raw" if (d / "raw").is_dir() else d
            chunk = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))]
            if not chunk:
                print(f"[report] {raw}: no units")
                continue
            s = summarize(chunk)
            s["provenance"] = {"dir": str(d), "n_units": len(chunk)}
            (print_tdoa if s["kind"] == "tdoa" else print_ridge)(s)
            dst = Path(OUT_DEFAULT, f"report_{s['kind']}.json")
            dst.write_text(json.dumps(s, indent=2))
            print(f"[report] {len(chunk)} units -> {dst}")
        raise SystemExit(0)

    if args.windows:
        keys = [k.strip() for k in args.windows.split(",") if k.strip()]
    elif args.pilot:
        keys = [
            "free-flight_nosource_room1__w00",
            "free-flight_nosource_room1__w01",
            "FLY124__w02",
        ]
    else:
        keys = list(
            {"dregon": DREGON_WINDOWS, "fly124": FLY124_WINDOWS, "all": ALL_WINDOWS}[args.dataset]
        )
    for bad in [k for k in keys if k not in ALL_WINDOWS]:
        ap.error(f"unknown window {bad!r}")
    if args.build_preps:
        build_preps(sorted({k for key in keys for k in (key, PARTNER[key])}), resolve_prep_dir())

    ctls = [c.strip() for c in args.controls.split(",") if c.strip()]
    if args.pilot:
        ctls = ["on"]
    for bad in [c for c in ctls if c not in CONTROLS]:
        ap.error(f"unknown control {bad!r}")

    if args.mode == "tdoa":
        units = [
            Unit(
                f"{k}__{ctl}",
                {
                    "key": k,
                    "control": ctl,
                    "candidate": args.candidate,
                    "dr_step": args.dr_step,
                    "gate": args.tdoa_gate == "ridge",
                    **PROTO,
                },
            )
            for k in sorted(keys)
            for ctl in ctls
        ]
        worker = tdoa_worker
    else:
        lo, hi, step = (float(v) for v in args.tau_ms.split(":"))
        taus = np.round(np.arange(lo, hi + 0.5 * step, step) * 1e-3, 6)
        scales = [float(v) for v in args.scales.split(",")]
        if args.pilot:
            scales = scales[:1]
        units = [
            Unit(
                f"{k}__t{int(round(t * 1e3)):+04d}__s{_uid(f'{s:.5f}')}__{ctl}",
                {"key": k, "tau_s": float(t), "scale": float(s), "control": ctl, **PROTO},
            )
            for k in sorted(keys)
            for t in taus
            for s in scales
            for ctl in ctls
        ]
        worker = ridge_worker

    print(f"[telemetry_timeshift/{args.mode}] {len(units)} units -> {out}", flush=True)
    res = gridrun_from_args(args, units, worker, str(out), summarize=summarize)
    (print_tdoa if res.summary.get("kind") == "tdoa" else print_ridge)(res.summary)
    raise SystemExit(res.exit_code)


if __name__ == "__main__":
    main()
