#!/usr/bin/env python3
"""Coupled VK envelopes vs independent demodulation, as the tracker's front end.

The decisive validation of GitHub issue #15: should ``pi_kalman``'s rate
observations come from the COUPLED envelope solve ``A_k(t)``
(:func:`tracking.vk_tracking.vk_envelopes`) instead of the independently
demodulated ``z_k(t)`` (:func:`tracking.phase_increment_tracker.demod_bank`)?

The comparison is estimator-vs-estimator, never trajectory-vs-trajectory: both
front ends run at the SAME carriers (the window's raw telemetry, exactly as
``scripts/tracking_ref.py`` inits), on the same envelope grid, with matched
capture in rev/s, and both are reduced to the same observation

    dr_k(t) = arg(u_k[n] conj(u_k[n-1])) * fs_env / (2 pi k)    [rev/s]

which is the shaft-rate ERROR of the carrier. Because a rate error displaces
harmonic ``k`` by ``k dr``, that observation is k-INDEPENDENT: every harmonic
of one rotor must report the same ``dr``. That is the accuracy criterion used
here — no ground truth is needed, and the telemetry's known ~0.5 % bias is
what both estimators are measuring.

Band correspondence (documented because it is the whole experiment):

* ``demod_bank`` brickwalls the demodulated harmonic to ``+-B_k`` with
  ``B_k = min(k * b0, 0.45 fs_env)`` Hz — the ``band_mode="k_scaled"``,
  ``band_b0=b0`` setting of ``pi_kalman_refine`` that the displacement
  campaign found best at ``b0 = 1`` (``docs/experiments/dregon-comb-displacement.md``).
  Capture radius: ``b0`` rev/s at every harmonic, clamped above
  ``k = 0.45 fs_env / b0``.
* ``vk_envelopes`` takes a **-3 dB** bandwidth ``bw_k = min(k * bw_rps, 0.9 fs_env)``
  Hz, that is a HALF width of ``bw_k / 2``. So ``bw_rps = 2 * b0`` matches the
  capture radius and the clamp knee exactly. The shapes still differ (soft
  4th-order VK passband vs brickwall); that is a property of the estimator
  under test, not of the setup.
* ``VKConfig.sep_bw_factor`` caps a coupled GROUP's band at a multiple of its
  minimum track separation, with ``cfg.bw_hz`` as the floor. With 4 rotors and
  ``k_max = 80`` the coupling predicate links ~all tracks into ONE group that
  contains a near-coincident pair, so the clamp collapses every band to
  ``cfg.bw_hz``. Therefore ``bw_hz`` is this script's clamp knob: the ``wide``
  arm raises it to ``0.9 fs_env`` (k-scaled bands actually run), the
  ``clamped`` arm leaves it at 1 Hz (what the flagship peel actually solves).

Outputs per window: ``<out>/<window>.json`` (per-track rows, near-coincident
pairs, aggregates) plus a compact printed table. Windows come from the frozen
``beatvk`` protocol prep cache (``scripts/beatvk_vk_arms.py`` materializes it).

Run::

    python scripts/vk_frontend_probe.py                       # 3 default windows
    python scripts/vk_frontend_probe.py --seconds 4           # smoke
    python scripts/vk_frontend_probe.py --windows FLY124:2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from tracking.comb_displacement import (  # noqa: E402
    DisplacementConfig,
    carrier_collision_mask,
    nearest_interloper_hz,
)
from tracking.phase_increment_tracker import demod_bank  # noqa: E402
from tracking.vk_tracking import VKConfig, env_stride, vk_envelopes  # noqa: E402

SR = 16000
N_ROTORS = 4
#: DREGON w00 is the frozen ``tracking_ref`` clip; w01 is the displacement
#: campaign's window (r0 k=7 vs r1 k=8 at 1.7 Hz, r0 vs r2 at 27 Hz at k=70);
#: FLY124 w02 is the first FLY124 cruise window.
DEFAULT_WINDOWS = ("free-flight_nosource_room1:0", "free-flight_nosource_room1:1", "FLY124:2")
#: ``VKConfig`` overrides per arm. ``bw_rps`` (multiplied by ``--b0``) is the
#: k-scaled band scale, ``None`` meaning "the scalar ``bw_hz`` band"; ``bw_hz``
#: is the group-clamp floor for the k-scaled arms (see the module docstring).
#:
#: ``wide`` is issue #15's proposal: coupled envelopes at the tracker's own
#: capture. ``peel`` is what the flagship actually solves today — the 1 Hz
#: scalar band of ``pipelines.PEEL_BW_HZ`` — kept as the smoothness-bias
#: extreme, not as a candidate front end.
ARMS: dict[str, dict[str, float | None]] = {
    "wide": {"bw_rps": 2.0, "bw_hz": 90.0},
    "peel": {"bw_rps": None, "bw_hz": 1.0},
}
EDGE_TRIM_S = 0.5
HP_CUT_HZ = 5.0  # smoothness-bias cutoff on the rate-observation spectrum


# ---------------------------------------------------------------------------
# small statistics


def _mad(x: np.ndarray) -> float:
    """Robust standard deviation (1.4826 * median absolute deviation)."""
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def _hp_frac(series: np.ndarray, fs_env: float, f_cut: float = HP_CUT_HZ) -> float:
    """Share of the (mean-removed) series' power above ``f_cut``.

    The smoothness-bias readout of issue #15 risk 6: the ``rho^2 D2' D2`` prior
    shrinks fast variation of ``A_k``, so its rate observations must carry less
    high-frequency power than the demodulated ones if the bias is real.
    """
    if series.ndim == 1:
        series = series[None, :]
    if series.shape[-1] < 8:
        return float("nan")
    x = series - series.mean(axis=-1, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs_env)
    tot = p[:, f > 0].sum(axis=-1)
    hi = p[:, f >= f_cut].sum(axis=-1)
    ok = tot > 0
    return float(np.mean(hi[ok] / tot[ok])) if ok.any() else float("nan")


def _weighted_median(vals: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(vals) & (w > 0)
    if not ok.any():
        return float("nan")
    v, ww = vals[ok], w[ok]
    order = np.argsort(v)
    v, ww = v[order], ww[order]
    c = np.cumsum(ww) / ww.sum()
    return float(v[int(np.searchsorted(c, 0.5))])


def _rates(bank: np.ndarray, ks: np.ndarray, fs_env: float) -> np.ndarray:
    """``(C, K, N-1)`` rev/s rate error from a ``(C, K, N)`` envelope bank."""
    inc = np.angle(bank[..., 1:] * np.conj(bank[..., :-1]))
    return inc * fs_env / (2.0 * np.pi * ks[None, :, None].astype(np.float64))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8:
        return float("nan")
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


@contextmanager
def _count_splu_fallbacks() -> Iterator[dict[str, int]]:
    """Count the coupled groups whose banded Cholesky reported a non-PD system.

    ``vk_envelopes`` silently falls back to the (much slower) sparse LU when
    ``cholesky_banded`` raises, and that fallback IS the conditioning readout of
    issue #15 risk 5 — a wide band on a group that holds a near-coincident pair
    is exactly when the VK normal equations stop being numerically PD. Nothing
    in the public API reports it, so the probe wraps the private core; it reads
    the count only, and never changes what is computed.
    """
    import tracking.vk_tracking as vk

    counter = {"n": 0}
    original = vk._solve_group_splu

    def counted(*a: Any, **kw: Any) -> Any:
        counter["n"] += 1
        return original(*a, **kw)

    vk._solve_group_splu = counted  # type: ignore[assignment]
    try:
        yield counter
    finally:
        vk._solve_group_splu = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# data


def load_window(prep_dir: Path, rid: str, widx: int, seconds: float | None) -> dict[str, Any]:
    """One protocol window: the ``beatvk_vk_arms`` prep cache, or the dataset.

    The cache is the local path (it is what every beat-VK driver already
    wrote). Off this machine there is none, and rebuilding it through
    ``beatvk_vk_arms.build_preps`` would also resolve the raw DREGON source for
    mic weights this probe never uses — so the fallback is
    ``tracking_ref.load_window``, which streams only the protocol dataset.
    """
    path = prep_dir / f"{rid}__w{widx:02d}.npz"
    if not path.exists():
        return _load_window_streaming(rid, widx, seconds)
    d = np.load(path, allow_pickle=False)
    n = d["audio"].shape[-1] if seconds is None else min(d["audio"].shape[-1], int(seconds * SR))
    return {
        "recording": rid,
        "window": widx,
        "regime": str(d["regime"]),
        "start_s": float(d["start_s"]),
        "audio": np.asarray(d["audio"][:, :n], dtype=np.float64),
        "ft": np.asarray(d["ft"], dtype=np.float64),
        "r_meas": np.asarray(d["r_meas"], dtype=np.float64),
    }


def _load_window_streaming(rid: str, widx: int, seconds: float | None) -> dict[str, Any]:
    """The prep-cache fallback: stream the protocol dataset for one window."""
    import tracking_ref

    from tracking.stages import get_audio, get_rps

    frame, spec, _prov = tracking_ref.load_window(rid, widx, version=None, seconds=seconds)
    audio, _sr = get_audio(frame)
    r_meas, ft = get_rps(frame)
    return {
        "recording": rid,
        "window": widx,
        "regime": str(spec.regime),
        "start_s": float(spec.start_s or 0.0),
        "audio": np.asarray(audio, dtype=np.float64),
        "ft": np.asarray(ft, dtype=np.float64),
        "r_meas": np.asarray(r_meas, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# the probe


def run_window(win: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Both front ends on one window: per-track rows, pairs, aggregates."""
    clip = win["audio"][: args.channels]
    n_t = clip.shape[-1]
    t_aud = np.arange(n_t) / SR
    r_aud = np.vstack([np.interp(t_aud, win["ft"], win["r_meas"][i]) for i in range(N_ROTORS)])
    band_cap = 0.45 * args.fs_env

    # --- the coupled solves (one per arm) -------------------------------
    envs: dict[str, Any] = {}
    walls: dict[str, float] = {}
    splu_fallbacks: dict[str, int] = {}
    for arm in args.arms:
        over = ARMS[arm]
        scale = over["bw_rps"]
        cfg = VKConfig(
            fs=float(SR),
            fs_env=args.fs_env,
            bw_rps=None if scale is None else scale * args.b0,
            bw_hz=float(over["bw_hz"] or 1.0),
            k_max=args.k_max,
            f_min=args.f_min,
            f_max=args.f_max,
            n_outer=1,
        )
        tic = time.perf_counter()
        with _count_splu_fallbacks() as counter:
            envs[arm] = vk_envelopes(clip, r_aud, cfg)
        walls[arm] = round(time.perf_counter() - tic, 1)
        splu_fallbacks[arm] = counter["n"]
        print(
            f"    vk_envelopes[{arm}] {walls[arm]}s  groups={len(envs[arm].groups)} "
            f"max_group={max(len(g) for g in envs[arm].groups)} "
            f"non_PD_groups={counter['n']}",
            flush=True,
        )
    ref_env = envs[args.arms[0]]
    stride, fs_env = env_stride(
        VKConfig(fs=float(SR), fs_env=args.fs_env)
    )  # identical for every arm
    n_env = ref_env.x.shape[-1]
    t_env = ref_env.t_env
    r_env = r_aud[:, ::stride][:, :n_env]
    edge = (t_env > EDGE_TRIM_S) & (t_env < t_env[-1] - EDGE_TRIM_S)
    inc_mask = edge[1:] & edge[:-1]

    # Collision geometry: the tracker's own band, re-derived against the true
    # rotor lines. ``search_revs``/``search_hz_cap``/``collision_guard`` are set
    # so that ``collision_guard * search_hz(k) == guard * B_k``.
    coll_cfg = DisplacementConfig(
        sr=SR,
        fs_env=fs_env,
        search_revs=args.b0,
        search_hz_cap=band_cap,
        collision_guard=args.collision_guard,
        f_max=args.f_max,
        min_rate=args.min_rate,
    )

    y32 = np.asarray(clip, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    banks: dict[tuple[str, int], np.ndarray] = {}
    phase_env: dict[int, np.ndarray] = {}

    for rot in range(N_ROTORS):
        if float(np.mean(r_env[rot])) < args.min_rate:
            continue
        sel = ref_env.rotor == rot
        cov = ref_env.valid[sel][:, edge].mean(axis=1)
        ks_all = ref_env.k[sel]
        ks = [int(k) for k, c in zip(ks_all, cov) if c > 0.99]
        if not ks:
            continue
        ka = np.asarray(ks, dtype=np.int64)
        band_k = np.minimum(ka.astype(np.float64) * args.b0, band_cap)

        phi = 2.0 * np.pi * np.cumsum(r_aud[rot]) / SR
        phase_env[rot] = phi[::stride][:n_env]
        z_on, _ = demod_bank(
            y32,
            phi,
            t_aud,
            ks,
            50.0,  # unused: no probe band is requested
            stride,
            n_env,
            float(band_k[0] / SR),
            band_cyc_k=band_k / SR,
            sr=float(SR),
        )
        banks[("demod", rot)] = z_on

        # Track admission: is there a LINE in this band at all? An off-comb
        # noise probe cannot answer that here — the bands reach +-45 Hz, wider
        # than the gaps of a 4-rotor comb, so no probe of the same width can be
        # placed clear of every line, and a narrow probe scaled by bandwidth
        # rejects every high-k harmonic whose line is displaced out of it by the
        # very rate error under measurement. So the statistic is VKConfig's own
        # ``update_gate`` one, which is resolution- and band-independent: the
        # peak-over-median ratio of the demodulated envelope's periodogram
        # (a line peaks far above the in-band floor; white noise tops out ~15).
        pk = np.abs(np.fft.fft(z_on[:, :, edge], axis=-1)) ** 2
        pk = pk.sum(axis=0)  # (K, F) channel-incoherent
        peak_ratio = pk.max(axis=-1) / np.maximum(np.median(pk, axis=-1), 1e-30)
        snr_db = 10.0 * np.log10(np.maximum(peak_ratio, 1e-12))

        dr_z = _rates(z_on, ka, fs_env)
        nearest = nearest_interloper_hz(
            r_env, r_env[rot], rot, ks, f_max=args.f_max, min_rate=args.min_rate
        )
        contested = carrier_collision_mask(r_env, r_env[rot], rot, ks, cfg=coll_cfg)

        arm_dr = {}
        for arm in args.arms:
            e = envs[arm]
            take = np.array([int(np.flatnonzero((e.rotor == rot) & (e.k == k))[0]) for k in ks])
            xa = e.x[:, take]  # (C, K, n_env)
            banks[(arm, rot)] = xa
            arm_dr[arm] = _rates(xa, ka, fs_env)

        m = np.broadcast_to(inc_mask[None, :], (z_on.shape[0], len(inc_mask)))
        n_ok = int(m.sum())
        for a, k in enumerate(ks):
            frac_cont = float(contested[a][edge].mean())
            near_med = float(np.median(nearest[a][edge]))
            row: dict[str, Any] = {
                "rotor": rot,
                "k": k,
                "f_hz": round(float(np.median(k * r_env[rot][edge])), 1),
                "band_hz": round(float(band_k[a]), 2),
                "snr_db": round(float(snr_db[a]), 1),
                "n_obs": n_ok,
                "contested_frac": round(frac_cont, 3),
                "nearest_hz": round(near_med, 2),
                "sep_over_band": round(near_med / max(float(band_k[a]), 1e-9), 2),
                "clean": bool(near_med >= args.clean_ratio * float(band_k[a])),
                "contested": bool(near_med <= float(band_k[a])),
                "est": {},
            }
            if n_ok >= 32 and peak_ratio[a] >= args.peak_min:
                zz = dr_z[:, a][m]
                row["est"]["demod"] = {
                    "med_dr": round(float(np.median(zz)), 4),
                    "mad_dr": round(_mad(zz), 4),
                    "hp_frac": round(_hp_frac(dr_z[:, a], fs_env), 3),
                    "amp": round(float(np.median(np.abs(z_on[:, a][:, edge]))), 6),
                }
                for arm in args.arms:
                    xx = arm_dr[arm][:, a][m]
                    row["est"][arm] = {
                        "med_dr": round(float(np.median(xx)), 4),
                        "mad_dr": round(_mad(xx), 4),
                        "hp_frac": round(_hp_frac(arm_dr[arm][:, a], fs_env), 3),
                        "amp": round(float(np.median(np.abs(banks[(arm, rot)][:, a][:, edge]))), 6),
                        "amp_ratio": round(
                            float(np.median(np.abs(banks[(arm, rot)][:, a][:, edge])))
                            / max(float(np.median(np.abs(z_on[:, a][:, edge]))), 1e-30),
                            3,
                        ),
                        "med_abs_diff": round(float(np.median(np.abs(xx - zz))), 4),
                        "corr": round(_corr(xx, zz), 3),
                        "bw_track": round(
                            float(
                                envs[arm].bw_track[
                                    np.flatnonzero((envs[arm].rotor == rot) & (envs[arm].k == k))[0]
                                ]
                            ),
                            2,
                        ),
                    }
            rows.append(row)

    # --- consensus reference per rotor (clean harmonics, k^2 weights) ----
    consensus: dict[str, dict[str, float]] = {}
    for est in ("demod", *args.arms):
        per_rotor: dict[str, float] = {}
        for rot in range(N_ROTORS):
            sel = [r for r in rows if r["rotor"] == rot and r["clean"] and est in r["est"]]
            if not sel:
                continue
            v = np.array([r["est"][est]["med_dr"] for r in sel])
            w = np.array([float(r["k"]) ** 2 for r in sel])
            per_rotor[str(rot)] = round(_weighted_median(v, w), 4)
        consensus[est] = per_rotor
    for r in rows:
        for stats in r["est"].values():
            ref = consensus["demod"].get(str(r["rotor"]))
            stats["cons_err"] = None if ref is None else round(abs(float(stats["med_dr"]) - ref), 4)

    # --- near-coincident pairs (risk 5) ---------------------------------
    pairs = _pair_report(rows, r_env, edge, phase_env, banks, args, fs_env, inc_mask)

    return {
        "window": {k: win[k] for k in ("recording", "window", "regime", "start_s")},
        "config": {
            "b0_rev_s": args.b0,
            "bw_rps": 2.0 * args.b0,
            "fs_env": fs_env,
            "k_max": args.k_max,
            "f_min": args.f_min,
            "f_max": args.f_max,
            "channels": int(clip.shape[0]),
            "seconds": round(n_t / SR, 2),
            "band_cap_hz": band_cap,
            "arms": {a: ARMS[a] for a in args.arms},
            "peak_min": args.peak_min,
            "hp_cut_hz": HP_CUT_HZ,
        },
        "wall_s": walls,
        "non_pd_groups": splu_fallbacks,
        "r_mean": [round(float(np.mean(r_env[i])), 2) for i in range(N_ROTORS)],
        "consensus_dr": consensus,
        "tracks": rows,
        "pairs": pairs,
        "summary": summarize(rows, args),
    }


def _pair_report(
    rows: list[dict[str, Any]],
    r_env: np.ndarray,
    edge: np.ndarray,
    phase_env: dict[int, np.ndarray],
    banks: dict[tuple[str, int], np.ndarray],
    args: argparse.Namespace,
    fs_env: float,
    inc_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Near-coincident track pairs: individual vs PAIR-SUM rate stability.

    Issue #15 risk 5: inside the envelope band two near-collinear columns leave
    the SUM well determined and the individuals free to take large cancelling
    values. The pair sum is formed exactly, in the first track's frame:
    ``x_m + x_n exp(i (k_n phi_n - k_m phi_m))`` — the beat is below the
    envelope Nyquist by construction, so the decimated grid represents it.
    """
    live = [r for r in rows if r["est"]]
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            ra, rb = live[a], live[b]
            if ra["rotor"] == rb["rotor"]:
                continue
            sep = abs(ra["f_hz"] - rb["f_hz"])
            if sep > args.pair_sep_hz:
                continue
            key = (ra["rotor"], ra["k"], rb["rotor"], rb["k"])
            if key in seen:
                continue
            seen.add(key)
            ka, kb = ra["k"], rb["k"]
            dphi = kb * phase_env[rb["rotor"]] - ka * phase_env[ra["rotor"]]
            rot_a, rot_b = ra["rotor"], rb["rotor"]
            ia = _row_index(rows, rot_a, ka)
            ib = _row_index(rows, rot_b, kb)
            entry: dict[str, Any] = {
                "a": [rot_a, ka],
                "b": [rot_b, kb],
                "sep_hz": round(sep, 2),
                "est": {},
            }
            for est in ("demod", *args.arms):
                xa = banks[(est, rot_a)][:, ia]
                xb = banks[(est, rot_b)][:, ib]
                s = xa + xb * np.exp(1j * dphi)[None, :]
                d_a = _rates(xa[:, None, :], np.array([ka]), fs_env)[:, 0]
                d_s = _rates(s[:, None, :], np.array([ka]), fs_env)[:, 0]
                entry["est"][est] = {
                    "mad_a": round(_mad(d_a[:, inc_mask]), 4),
                    "mad_b": round(
                        _mad(_rates(xb[:, None, :], np.array([kb]), fs_env)[:, 0][:, inc_mask]), 4
                    ),
                    "mad_sum": round(_mad(d_s[:, inc_mask]), 4),
                    "amp_a": round(float(np.median(np.abs(xa[:, edge]))), 6),
                    "amp_b": round(float(np.median(np.abs(xb[:, edge]))), 6),
                    "amp_sum": round(float(np.median(np.abs(s[:, edge]))), 6),
                }
            out.append(entry)
    out.sort(key=lambda e: e["sep_hz"])
    return out[: args.max_pairs]


def _row_index(rows: list[dict[str, Any]], rot: int, k: int) -> int:
    """Position of harmonic ``k`` inside rotor ``rot``'s bank (rows are in order)."""
    ks = [r["k"] for r in rows if r["rotor"] == rot]
    return ks.index(k)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Clean/contested aggregates of the five comparison metrics."""
    out: dict[str, Any] = {}
    live = [r for r in rows if r["est"]]
    for label, sel in (
        ("clean", [r for r in live if r["clean"]]),
        ("contested", [r for r in live if r["contested"]]),
        ("clean_k_ge_20", [r for r in live if r["clean"] and r["k"] >= 20]),
        ("contested_k_ge_45", [r for r in live if r["contested"] and r["k"] >= 45]),
        ("all", live),
    ):
        block: dict[str, Any] = {"n_tracks": len(sel)}
        for est in ("demod", *args.arms):
            vals = [r["est"][est] for r in sel if est in r["est"]]
            if not vals:
                continue

            def med(key: str, v: list[dict[str, Any]] = vals) -> float | None:
                x = np.array([q[key] for q in v if q.get(key) is not None], dtype=np.float64)
                x = x[np.isfinite(x)]
                return None if x.size == 0 else round(float(np.median(x)), 4)

            block[est] = {
                "cons_err": med("cons_err"),
                "mad_dr": med("mad_dr"),
                "hp_frac": med("hp_frac"),
                "med_abs_diff": med("med_abs_diff"),
                "corr": med("corr"),
                "amp_ratio": med("amp_ratio"),
            }
        out[label] = block
    return out


def print_table(res: dict[str, Any], arms: list[str]) -> None:
    w = res["window"]
    print(f"\n=== {w['recording']} w{w['window']:02d} ({w['regime']}) rates {res['r_mean']} ===")
    print(f"  consensus dr (rev/s): {json.dumps(res['consensus_dr'])}")
    ests = ["demod", *arms]
    print(
        "\n  [cons_err | mad_dr | hp_frac | med|A-Z| | corr | amp/demod]"
        "  (rev/s, medians over tracks)"
    )
    print(f"  {'set':<18}{'n':>4}  " + "  ".join(f"{e:>46}" for e in ests))
    for label, block in res["summary"].items():
        cells = []
        for e in ests:
            b = block.get(e)
            if b is None:
                cells.append(f"{'-':>46}")
                continue
            cells.append(
                f"{_f(b['cons_err']):>8}{_f(b['mad_dr']):>8}{_f(b['hp_frac']):>7}"
                f"{_f(b['med_abs_diff']):>8}{_f(b['corr']):>7}{_f(b.get('amp_ratio')):>8}"
            )
        print(f"  {label:<18}{block['n_tracks']:>4}  " + "  ".join(cells))
    print("\n  contested tracks with the strongest lines:")
    cont = sorted(
        [r for r in res["tracks"] if r["est"] and r["contested"]],
        key=lambda r: -r["snr_db"],
    )[:8]
    for r in cont:
        parts = " ".join(
            f"{e}={_f(r['est'][e]['med_dr'])}/{_f(r['est'][e]['mad_dr'])}" for e in ests
        )
        print(
            f"    r{r['rotor']} k={r['k']:<3} f={r['f_hz']:<7.0f} near={r['nearest_hz']:<6} "
            f"snr={r['snr_db']:<5.1f} {parts}"
        )
    if res["pairs"]:
        print("\n  near-coincident pairs (mad_a / mad_b / mad_sum, rev/s):")
        for p in res["pairs"][:6]:
            parts = " ".join(
                f"{e}={_f(p['est'][e]['mad_a'])}/{_f(p['est'][e]['mad_b'])}"
                f"/{_f(p['est'][e]['mad_sum'])}"
                for e in ests
            )
            print(
                f"    r{p['a'][0]} k={p['a'][1]} vs r{p['b'][0]} k={p['b'][1]} "
                f"sep={p['sep_hz']:<6} {parts}"
            )


def _f(v: Any) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:g}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--windows", default=",".join(DEFAULT_WINDOWS), help="RID:WIDX,...")
    ap.add_argument("--arms", default="wide,peel", help=f"subset of {sorted(ARMS)}")
    ap.add_argument("--out", default="results/vk_frontend_probe")
    ap.add_argument("--prep-dir", default="results/beatvk_vk_arms/prep_cache")
    ap.add_argument("--b0", type=float, default=1.0, help="demod band scale (rev/s per order)")
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--f-min", type=float, default=60.0)
    ap.add_argument("--f-max", type=float, default=7500.0)
    ap.add_argument("--fs-env", type=float, default=100.0)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--min-rate", type=float, default=5.0)
    ap.add_argument(
        "--peak-min", type=float, default=10.0, help="envelope peak/median admission ratio"
    )
    ap.add_argument("--collision-guard", type=float, default=1.2)
    ap.add_argument(
        "--clean-ratio",
        type=float,
        default=2.0,
        help="a track is CLEAN when the nearest foreign line is this many bands away",
    )
    ap.add_argument("--pair-sep-hz", type=float, default=5.0)
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=None, help="truncate (smoke runs)")
    args = ap.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        ap.error(f"unknown arms {sorted(unknown)}; known: {sorted(ARMS)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = Path(args.prep_dir)
    for spec in args.windows.split(","):
        rid, widx = spec.rsplit(":", 1)
        print(f"[{spec}] loading", flush=True)
        win = load_window(prep, rid, int(widx), args.seconds)
        res = run_window(win, args)
        (out_dir / f"{rid}__w{int(widx):02d}.json").write_text(json.dumps(res, indent=1))
        print_table(res, args.arms)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
