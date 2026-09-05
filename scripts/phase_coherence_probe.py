#!/usr/bin/env python3
"""How coherent is one rotor's harmonic comb? — the measurement behind the
stochastic comb noise model.

The model this probe motivates is the one in the module docstring of
``data_processing.stochastic_rotor_noise``: a smooth floor plus one
**Lorentzian** line per harmonic, the half width growing with the harmonic
index as ``gamma_k = gamma_0 + s k``. Three claims sit under it, and none of
them was measured on real audio before this probe:

1. Each line is Lorentzian, not Gaussian, and its half width grows linearly
   in ``k``.
2. The disturbance is a SHAFT disturbance, so the harmonics move together —
   the covariance of the per-harmonic rate opinions is rank one plus a small
   diagonal, and the correlation ``rho_k`` is near one.
3. What is left after the shared part is removed is heavy tailed or it is not.

Three conditions of increasing complexity, so that a claim which survives all
three is a claim about rotors and not about one recording:

``single``
    One DREGON motor at a steady setpoint, the other three off
    (``motor_Motor{1-4}_{50,60,70,80,90}``, 8 microphones). No telemetry: the
    reference starts at ``0.98 x`` the nominal setpoint (the measured true
    speed of these runs) and is refined by
    ``tracking.phase_increment_tracker.pi_kalman_refine``. One 20 s segment
    from the middle of each recording.
``bench``
    All four motors at nominal 70 on the ground (``motor_allMotors_70``).
    Same treatment with four rotor tracks, and the inits are STAGGERED —
    four identical constants are not separable, so refinement cannot engage
    from them.
``flight_dregon`` / ``flight_fly125``
    Free flight, TRAIN split only. The five DREGON room-2 recordings carry a
    commanded rotor speed and no measured one, so their reference is the
    spike-cleaned command, clock-aligned to the audio, then refined per window
    by ``pi_kalman_refine``. Michael's FLY125 carries the committed refined
    sidecar (``src/data_processing/refined_labels/FLY125.npz``), which is used
    as it is. Every non-overlapping 20 s window whose four rotors all stay at
    or above 45 rev/s.

Each unit's JSON records which reference it used (``sidecar`` or
``telemetry+pi_kalman`` or ``nominal+pi_kalman``) and the refined mean rate per
rotor, because the three are not the same measurement of the same thing.

The three readings, per (window, rotor)
---------------------------------------
Everything is computed by ``tracking.phase_noise`` on the harmonics ``k = 1..K``
of one rotor along its reference trajectory. This script is the DATA side only.

**1. Line width and line shape.** From the wide-band complex envelope
``z_k(t)`` (``demod_rotor`` at a band of ``min(40, 0.45 x rate)`` Hz on a
500 Hz envelope grid) the normalized complex autocorrelation is taken out to
``--max-lag-s``. The coherence time ``tau_k`` is the lag where its magnitude
falls to ``exp(-1)`` and the half width is ``gamma_k = 1 / (2 pi tau_k)``,
because a Lorentzian of half width ``gamma`` has envelope autocorrelation
``exp(-2 pi gamma |lag|)`` exactly. ``gamma_k = gamma_0 + s k`` is then fitted
by least squares over the admitted harmonics. Separately the averaged
periodogram of ``z_k`` is fitted, in the log domain, by a Lorentzian and by a
Gaussian of the SAME half width at half maximum, so the verdict is about the
tail and not about the width.

**2. Cross-harmonic correlation.** The WP18 estimator, unchanged: the per-frame
rate opinions ``delta_k = arg(z_k[t+1] conj(z_k[t])) fs_env / (2 pi k)`` on the
62.5 Hz grid, at the fixed 1.5 Hz arm and at a k-scaled arm. Reported as the
correlation of harmonic ``k`` with the other admitted harmonics (``rho_k``),
the rank-one share of the off-diagonal energy, and ``sigma_J^2`` against the
median per-harmonic variance ``v_k``.

**3. Residual shape.** ``e_k(t) = delta_k(t) - c(t)``, where ``c(t)`` is the
inverse-variance weighted mean of the admitted opinions at that frame. Pooled
over frames and over the harmonics of a band, standardized by the median
absolute deviation, and reported as an excess kurtosis plus the per-sample log
likelihood ratio of a Cauchy fit against a Gaussian fit. A wrapped increment is
BOUNDED by ``pi / (2 pi k dt)`` rev/s, so at high ``k`` the tails are clipped by
construction and this ratio understates the Cauchy case.

Outputs
-------
One ``utils.gridrun`` unit per (condition, recording, window, rotor) under
``<out>/raw/``, so the run is restartable; ``<out>/summary.json``;
``<out>/summary.csv`` (one row per condition x harmonic); ``<out>/conditions.csv``
(one row per condition); and both CSVs printed as Markdown. The per-window
reference trajectory is cached under ``<out>/refs/`` so the four rotor units of
one window refine it once.

Examples::

    # the laptop check: one motor recording, 5 s, k <= 10 — under a minute
    PYTHONPATH=src python scripts/phase_coherence_probe.py --smoke --out /tmp/pc-smoke

    # the full probe (a CPU node)
    PYTHONPATH=src python scripts/phase_coherence_probe.py \
        --conditions single,bench,flight --out results/phase_coherence --jobs 8
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (process-level parallelism instead) —
# the shared harness convention (utils.gridrun re-asserts it).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

SR = 16000
#: The frozen evaluation frame grid every campaign of this project reads.
HOP_S = 0.032
OUT_DEFAULT = "results/phase_coherence"

#: Condition (i): one motor at a steady setpoint, the others off.
SINGLE_RIDS = tuple(f"motor_Motor{m}_{s}" for m in (1, 2, 3, 4) for s in (50, 60, 70, 80, 90))
#: Condition (ii): four motors at nominal 70 on the bench.
BENCH_RID = "motor_allMotors_70"
#: Condition (iii): free flight, TRAIN split only. The room-1 free flight and
#: FLY124 are the VALIDATION recordings and are deliberately absent.
FLIGHT_DREGON_RIDS = (
    "free-flight_nosource_room2",
    "hovering_nosource_room2",
    "updown_nosource_room2",
    "rectangle_nosource_room2",
    "spinning_nosource_room2",
)
FLY_RID = "FLY125"
CONDITIONS = ("single", "bench", "flight_dregon", "flight_fly125")

DREGON_SOURCE = "dload:DREGON"
MICHAELS_SPEC = "frames:michaels-frames"
LABEL_DIR = "src/data_processing/refined_labels"

#: The motor runs carry no tachometer. The S3 rung of
#: ``scripts/vk_phase_validation.py`` measured the true speed of these runs at
#: about 0.98 x the nominal setpoint in the recording id, which is the init.
NOMINAL_SCALE = 0.98
#: Four identical initial rates are structurally unrefinable (the twin gate
#: excludes every window), so the bench inits are staggered — the ``|stag``
#: control of ``vk_phase_validation.S3B_STAGGER``, reused.
BENCH_STAGGER = (-0.6, -0.2, 0.2, 0.6)
#: A cruise window: every rotor at or above this, for the whole window.
CRUISE_REV_S = 45.0
#: The in-flight mask on the commanded speed (the ``vk_validation`` convention).
DREGON_MIN_RPS = 30.0
#: Harmonic cap of the reference refinement (NOT of the measurement).
REFINE_K_MAX = 40

#: The line measurement's own envelope grid. It has to resolve a coherence
#: time of a few milliseconds — at ``gamma_k = 0.6 k`` the k=30 line has
#: ``tau = 8.8 ms``, which is half a sample on WP18's 62.5 Hz grid — so it runs
#: an order of magnitude faster than the covariance grid. The band is
#: oversampled, not aliased: the brickwall stays at ``b_wide``.
FS_LINE = 500.0
#: The widest band the line measurement demodulates, and the fraction of the
#: rotor rate it is capped at. The cap is what keeps the NEXT tooth out of the
#: band; the absolute maximum is what keeps the transform honest.
B_LINE_MAX = 40.0
B_LINE_FRAC = 0.45
MAX_LAG_S = 2.0
#: The line measurement's own envelope-SNR gate, and it is STRICTER than
#: WP18's ``MIN_SNR`` = 1 on purpose: the autocorrelation is normalized by the
#: line power AFTER the off-comb floor is subtracted, so at SNR ~ 1 that
#: denominator is a small difference of two similar numbers and the curve reads
#: above 1 at random lags. Measured on one DREGON motor run: at SNR 1.3 the
#: normalized curve reached 1.85.
LINE_MIN_SNR = 3.0
#: The two arms of reading 2: WP18's main fixed band, and the k-scaled arm
#: whose band matches the fixed one at k = 3.
DEFAULT_ARMS = ("fixB1.5", "kscale0.5")


# ---------------------------------------------------------------------------
# data


def frame_grid(n_t: int) -> np.ndarray:
    """The uniform ``HOP_S`` frame grid of ``n_t`` samples, in relative seconds."""
    return np.arange(0.0, n_t / SR - 1e-9, HOP_S)


def _dregon_dir(source: str) -> Path:
    from data_processing.streams import resolve_source

    return Path(resolve_source(source))


def _load_dregon_motor(rid: str, source: str) -> dict[str, Any]:
    """One motor-bench recording: audio + a constant init at ``0.98 x`` nominal."""
    from data_processing.sources.dregon import discover_recordings, get_geometry, load_timeframe

    ddir = _dregon_dir(source)
    by_id = {s["recording_id"]: s for s in discover_recordings(ddir)}
    if rid not in by_id:
        raise KeyError(f"{rid} not found under {ddir}")
    sample = by_id[rid]
    frame = load_timeframe(sample, geometry=get_geometry(ddir), target_sr=SR)
    audio = np.asarray(frame["audio"].data, dtype=np.float64)
    nominal = float(sample["motor_speed"])
    n_rotors = 4 if rid == BENCH_RID else 1
    ft = frame_grid(audio.shape[-1])
    r_init = np.full((n_rotors, len(ft)), NOMINAL_SCALE * nominal)
    if n_rotors > 1:
        r_init = r_init + np.asarray(BENCH_STAGGER, dtype=np.float64)[:, None]
    return {
        "recording_id": rid,
        "audio": audio,
        "ft": ft,
        "r_init": r_init,
        "ref_kind": "nominal+pi_kalman",
        "nominal_rps": nominal,
        "span_s": (0.0, float(audio.shape[-1]) / SR),
    }


def _load_dregon_flight(rid: str, source: str) -> dict[str, Any]:
    """One DREGON room-2 flight: audio + the clock-aligned cleaned command.

    These recordings have no ``motors_measured``, so the reference starts from
    ``motors_command``. Two corrections are the free-flight prep convention of
    ``scripts/vk_validation.prepare_recording`` and are not optional: the
    leading logging artifact plus a median filter (``clean_command_spikes``),
    and the audio-telemetry clock offset ``tau`` from the stage-A comb scan. An
    unaligned init would be refined into the wrong place at every harmonic.
    """
    from data_processing.sources.dregon import (
        clean_command_spikes,
        discover_recordings,
        get_geometry,
        load_timeframe,
    )
    from tracking.rps_refinement import RefineConfig, compute_logmag, estimate_clock_offset

    ddir = _dregon_dir(source)
    by_id = {s["recording_id"]: s for s in discover_recordings(ddir)}
    if rid not in by_id:
        raise KeyError(f"{rid} not found under {ddir}")
    frame = load_timeframe(by_id[rid], geometry=get_geometry(ddir), target_sr=SR)
    audio = np.asarray(frame["audio"].data, dtype=np.float64)
    t0 = float(frame["audio"].tindex.t_start)
    command = np.asarray(frame["motors_command"].data)
    mt = np.asarray(frame["motors_command"].tindex.abs_stamps) - t0
    cleaned = clean_command_spikes(command)

    idx = np.where(np.median(command, axis=0) > DREGON_MIN_RPS)[0]
    if idx.size < 2:
        raise RuntimeError(f"{rid}: no in-flight span above {DREGON_MIN_RPS} rev/s")
    t_lo, t_hi = float(mt[idx[0]]) + 0.2, float(mt[idx[-1]]) - 0.2

    cfg = RefineConfig()
    a0, a1 = int(t_lo * SR), int(min(t_lo + 30.0, t_hi) * SR)
    tau, _, _ = estimate_clock_offset(
        compute_logmag(audio[:1, a0:a1], cfg), mt - t_lo, cleaned, cfg
    )

    ft = frame_grid(audio.shape[-1])
    r_init = np.stack([np.interp(ft + tau, mt, cleaned[i]) for i in range(cleaned.shape[0])])
    return {
        "recording_id": rid,
        "audio": audio,
        "ft": ft,
        "r_init": r_init,
        "ref_kind": "telemetry+pi_kalman",
        "tau_s": float(tau),
        "span_s": (t_lo, t_hi),
    }


def _load_michaels(rid: str, spec: str, label_dir: str | Path) -> dict[str, Any]:
    """FLY125 as the published frame, with the committed refined sidecar applied.

    The sidecar's times are defined against the PUBLISHED recording's audio
    ``t_start``, which is why this reads the published frames and applies the
    labels through ``apply_rps_override`` instead of loading the raw tree: the
    two would need the same tick-exact offset, and only one of them has it.
    """
    import tdseries as td  # noqa: I001 — tdseries is a seam, not a stdlib import

    from data_processing.frames import meta_dict
    from data_processing.noise_rps_dataset import load_published_noise_sources
    from tracking.decompose import interp_rps

    directory = Path(label_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    for src in load_published_noise_sources(
        spec, SR, origin="michaels", rps_key="rps", rps_override_dir=directory
    ):
        frame = src.frame
        if str(meta_dict(frame).get("recording_id") or "") != rid:
            continue
        audio_s = frame["audio"]
        audio = np.atleast_2d(np.asarray(audio_s.data, dtype=np.float64))
        rps_s = frame[src.rps_key]
        # Tick-exact relative seconds: the published frames sit at absolute
        # epoch ticks (~1e18) that float64 subtraction cannot hold.
        t0 = int(audio_s.tindex.t_start_ticks)
        ticks = np.asarray(rps_s.tindex.abs_stamps_ticks, dtype=np.int64)
        stamps = (ticks - t0) / float(td.TICKS_PER_SECOND)
        ft = frame_grid(int(audio.shape[-1]))
        return {
            "recording_id": rid,
            "audio": audio,
            "ft": ft,
            "r_init": interp_rps(np.asarray(rps_s.data), stamps, ft),
            "ref_kind": "sidecar",
            "span_s": (0.0, float(audio.shape[-1]) / SR),
        }
    raise KeyError(f"{rid} not found in {spec}")


#: Per-process recording cache. Under a fork start method a worker inherits the
#: parent's copy and decodes nothing.
_RECORDINGS: dict[tuple[str, str], dict[str, Any]] = {}


def get_recording(condition: str, rid: str, opts: dict[str, Any]) -> dict[str, Any]:
    key = (condition, rid)
    if key not in _RECORDINGS:
        if condition in ("single", "bench"):
            rec = _load_dregon_motor(rid, str(opts.get("dregon_source", DREGON_SOURCE)))
        elif condition == "flight_dregon":
            rec = _load_dregon_flight(rid, str(opts.get("dregon_source", DREGON_SOURCE)))
        elif condition == "flight_fly125":
            rec = _load_michaels(
                rid,
                str(opts.get("michaels_spec", MICHAELS_SPEC)),
                str(opts.get("label_dir", LABEL_DIR)),
            )
        else:
            raise ValueError(f"unknown condition {condition!r}")
        _RECORDINGS[key] = rec
    return _RECORDINGS[key]


def plan_windows(rec: dict[str, Any], condition: str, window_s: float) -> list[tuple[int, int]]:
    """The audio sample ranges this recording contributes, non-overlapping.

    The motor conditions give ONE window from the middle of the recording — a
    steady setpoint has nothing else in it. Flight gives every non-overlapping
    cruise window: all rotors at or above ``CRUISE_REV_S`` for the whole window,
    which is what makes the harmonics of one window comparable at all.
    """
    n_t = int(rec["audio"].shape[-1])
    n_seg = int(round(window_s * SR))
    if n_seg >= n_t:
        return [(0, n_t)]
    if condition in ("single", "bench"):
        lo = max(0, (n_t - n_seg) // 2)
        return [(lo, lo + n_seg)]
    t_lo, t_hi = rec["span_s"]
    ft, r = rec["ft"], rec["r_init"]
    out: list[tuple[int, int]] = []
    start = float(t_lo)
    while start + window_s <= float(t_hi):
        sel = (ft >= start) & (ft < start + window_s)
        if sel.sum() >= 2 and float(r[:, sel].min()) >= CRUISE_REV_S:
            a0 = int(round(start * SR))
            out.append((a0, a0 + n_seg))
            start += window_s
        else:
            start += HOP_S * 8  # slide past the non-cruise stretch
    return out


# ---------------------------------------------------------------------------
# the reference trajectory


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def reference_trajectory(
    rec: dict[str, Any], audio: np.ndarray, ft_w: np.ndarray, r_init: np.ndarray, p: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """The window's reference, refined when the labels are not already refined.

    Cached on disk per WINDOW, because the four rotor units of one window would
    otherwise each pay for the same refinement.
    """
    from tracking.phase_increment_tracker import pi_kalman_refine

    if rec["ref_kind"] == "sidecar":
        return r_init, {"kind": "sidecar"}
    cache = Path(str(p["out"])) / "refs" / f"{p['window_uid']}__k{int(p['refine_k_max'])}.npz"
    if cache.is_file():
        try:
            with np.load(cache) as z:
                return np.asarray(z["r_ref"], dtype=np.float64), {
                    "kind": rec["ref_kind"],
                    "cached": True,
                }
        except Exception:  # noqa: BLE001 — a torn cache file is recomputed, not fatal
            pass
    r_ref, diag = pi_kalman_refine(audio, r_init, ft_w, sr=SR, k_max=int(p["refine_k_max"]))
    r_ref = np.asarray(r_ref, dtype=np.float64)
    _atomic_npz(cache, r_ref=r_ref)
    return r_ref, {
        "kind": rec["ref_kind"],
        "cached": False,
        "fs_env_actual": float(diag.get("fs_env_actual", float("nan"))),
        "move_rev_s": [round(float(np.mean(np.abs(a - b))), 5) for a, b in zip(r_ref, r_init)],
    }


# ---------------------------------------------------------------------------
# the unit


def _line_readings(pn: Any, dm: Any, b_line: float, max_lag_s: float) -> list[dict[str, Any]]:
    """Per-harmonic coherence time, half width and line shape."""
    n_env = int(dm.z.shape[-1])
    n_trim = max(1, int(round(pn.EDGE_TRIM_S * dm.fs_env)))
    interior = slice(n_trim, max(n_trim + 1, n_env - n_trim))
    rows: list[dict[str, Any]] = []
    for a, k in enumerate(dm.ks):
        z = dm.z[:, a, interior]
        npw = 2.0 * b_line * dm.noise_psd[:, a]
        snr = float(np.mean(np.abs(z) ** 2) / max(float(np.mean(npw)), 1e-30))
        valid_frac = float(np.mean(~dm.coll[a]))
        admitted = bool(snr >= LINE_MIN_SNR and valid_frac >= pn.MIN_VALID_FRAC)
        row: dict[str, Any] = {
            "k": int(k),
            "snr": round(snr, 4),
            "valid_frac": round(valid_frac, 4),
            "admitted": admitted,
        }
        if admitted:
            lw = pn.linewidth(
                z, dm.fs_env, max_lag_s=max_lag_s, noise_power=npw, noise_band_hz=b_line
            )
            row.update(
                {
                    "tau_s": lw["tau_s"],
                    "gamma_hz": lw["gamma_hz"],
                    "censored": lw["censored"],
                    "gamma_bound_hz": lw["gamma_bound_hz"],
                    "acf_at_max_lag": lw["acf_at_max_lag"],
                    "gamma_slope_hz": lw["gamma_slope_hz"],
                    "slope_n": lw["slope_n"],
                    "slope_r2": lw["slope_r2"],
                }
            )
            # The shape fit needs a starting width and a resolution finer than
            # it. The crossing is preferred, the log slope is next, and the
            # censoring bound is the last resort.
            hw0 = next(
                (
                    float(v)
                    for v in (lw["gamma_hz"], lw["gamma_slope_hz"], lw["gamma_bound_hz"])
                    if v is not None and np.isfinite(v) and v > 0
                ),
                float(lw["gamma_bound_hz"]),
            )
            f, pw = pn.welch_envelope(z, dm.fs_env, target_df=max(hw0 / 3.0, 1e-3))
            sh = pn.fit_line_shape(f, pw, hwhm0=max(hw0, 1e-3))
            row["shape"] = {
                "verdict": sh.get("verdict", ""),
                "log_resid_ratio": sh.get("log_resid_ratio"),
                "hwhm_lorentz_hz": sh.get("lorentz", {}).get("hwhm_hz"),
                "hwhm_gauss_hz": sh.get("gauss", {}).get("hwhm_hz"),
                "resid_lorentz": sh.get("lorentz", {}).get("resid_rms_log10"),
                "resid_gauss": sh.get("gauss", {}).get("resid_rms_log10"),
                "span_hz": sh.get("span_hz"),
            }
        rows.append(row)
    return rows


def _law_inputs(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """The (k, gamma) pairs of one width estimator, admitted and finite only."""
    pairs = [
        (float(r["k"]), float(r[key]))
        for r in rows
        if r["admitted"] and r.get(key) is not None and np.isfinite(r.get(key, np.nan))
    ]
    if not pairs:
        return np.zeros(0), np.zeros(0)
    a, b = zip(*pairs)
    return np.asarray(a), np.asarray(b)


def _shape_bands(pn: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The Lorentz-against-Gauss verdict pooled into the reporting bands."""
    out: dict[str, Any] = {}
    for name, lo, hi in pn.K_BANDS:
        sel = [
            r
            for r in rows
            if r["admitted"] and lo <= r["k"] <= hi and r.get("shape", {}).get("verdict")
        ]
        if not sel:
            continue
        ratios = [
            float(r["shape"]["log_resid_ratio"])
            for r in sel
            if r["shape"]["log_resid_ratio"] is not None
            and np.isfinite(r["shape"]["log_resid_ratio"])
        ]
        n_lor = sum(1 for r in sel if r["shape"]["verdict"] == "lorentz")
        out[name] = {
            "n": len(sel),
            "n_lorentz": n_lor,
            "lorentz_frac": round(n_lor / len(sel), 4),
            "log_resid_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
            "verdict": "lorentz" if n_lor * 2 >= len(sel) else "gauss",
        }
    return out


def _residual_bands(pn: Any, ser: Any, cov: dict[str, Any]) -> dict[str, Any]:
    """Residual ``e_k = delta_k - c`` pooled per band, with its tail statistics."""
    if "v_k_used" not in cov or "k_used" not in cov:
        return {}
    c, mask = pn.shared_rate_opinion(
        ser, np.asarray(cov["v_k_used"], dtype=float), np.asarray(cov["k_used"], dtype=int)
    )
    ks = np.asarray(ser.ks)
    idx = np.where(ser.keep)[0]
    out: dict[str, Any] = {}
    for name, lo, hi in pn.K_BANDS:
        sel = np.array([a for a in idx if lo <= ks[a] <= hi], dtype=int)
        if sel.size == 0:
            continue
        e = ser.delta[:, sel, :] - c[:, None, :]
        keep = np.broadcast_to(ser.valid[sel][None, :, :] & mask[None, None, :], e.shape)
        stats = pn.residual_tail_stats(e[keep])
        stats["n_harmonics"] = int(sel.size)
        out[name] = stats
    return out


def _jsonable(obj: Any) -> Any:
    """numpy scalars and arrays out of a unit payload, so it serializes."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def probe_worker(unit: Unit) -> dict[str, Any]:
    """One (condition, recording, window, rotor): the three readings."""
    return _jsonable(_probe(unit))


def _probe(unit: Unit) -> dict[str, Any]:
    from tracking import phase_noise as pn

    p = dict(unit.params)
    cond, rid, rotor = str(p["condition"]), str(p["recording"]), int(p["rotor"])
    rec = get_recording(cond, rid, p)
    a0, a1 = int(p["a0"]), int(p["a1"])
    audio = np.ascontiguousarray(rec["audio"][:, a0:a1], dtype=np.float64)
    ft_w = frame_grid(a1 - a0)
    r_init = np.stack([np.interp(ft_w + a0 / SR, rec["ft"], row) for row in rec["r_init"]])

    r_ref, ref_diag = reference_trajectory(rec, audio, ft_w, r_init, p)
    mean_rate = float(np.mean(r_ref[rotor]))
    out: dict[str, Any] = {
        "condition": cond,
        "recording": rid,
        "window": int(p["window"]),
        "rotor": rotor,
        "start_s": round(a0 / SR, 4),
        "end_s": round(a1 / SR, 4),
        "n_channels": int(audio.shape[0]),
        "reference": {
            **ref_diag,
            "mean_rps": round(mean_rate, 5),
            "mean_rps_per_rotor": [round(float(np.mean(row)), 5) for row in r_ref],
            "init_mean_rps": round(float(np.mean(r_init[rotor])), 5),
        },
    }
    if mean_rate < pn.MIN_RATE:
        return {**out, "failed": f"rotor mean rate {mean_rate:.2f} below {pn.MIN_RATE}"}

    # --- 1. line width and line shape -------------------------------------
    k_max = int(p["k_max"])
    b_line = float(min(B_LINE_MAX, B_LINE_FRAC * mean_rate))
    dm = pn.demod_rotor(
        audio, r_ref, ft_w, rotor, sr=SR, fs_env=FS_LINE, k_max=k_max, b_wide=b_line
    )
    if dm is None:
        return {**out, "failed": "demod_rotor rejected the rotor"}
    rows = _line_readings(pn, dm, b_line, float(p["max_lag_s"]))
    out["line"] = {
        "b_wide_hz": round(b_line, 4),
        "fs_env": float(dm.fs_env),
        "max_lag_s": float(p["max_lag_s"]),
        "k_top": int(dm.diag["k_top"]),
        "n_admitted": int(sum(1 for r in rows if r["admitted"])),
        "n_censored": int(sum(1 for r in rows if r.get("censored"))),
        "min_snr": LINE_MIN_SNR,
        "harmonics": rows,
    }
    # The law is fitted on the crossing estimator, and AGAIN on the log-slope
    # one: a condition whose lines are too narrow to cross has no first fit at
    # all, and reporting only that would say nothing rather than "very narrow".
    out["law"] = pn.fit_linewidth_law(*_law_inputs(rows, "gamma_hz"))
    out["law_slope"] = pn.fit_linewidth_law(*_law_inputs(rows, "gamma_slope_hz"))
    out["shape_bands"] = _shape_bands(pn, rows)

    # --- 2 and 3. the covariance arms, on WP18's own grid ------------------
    dm_wp = pn.demod_rotor(
        audio, r_ref, ft_w, rotor, sr=SR, fs_env=pn.FS_ENV, k_max=k_max, b_wide=pn.B_WIDE
    )
    arms_by_name = {a.name: a for a in pn.ARMS}
    arms_out: dict[str, Any] = {}
    residual: dict[str, Any] = {}
    if dm_wp is not None:
        for i, name in enumerate(list(p["arms"])):
            arm = arms_by_name[name]
            ser = pn.arm_increments(dm_wp, arm)
            cov = pn.arm_covariance(dm_wp, arm, series=ser)
            corr = pn.cross_harmonic_correlation(ser)
            fit0 = cov.get("cov", {}).get("0", {})
            v_used = np.asarray(cov.get("v_k_used", []), dtype=float)
            v_med = float(np.nanmedian(v_used)) if v_used.size else float("nan")
            sigma = fit0.get("sigma_c2_mean")
            arms_out[name] = {
                "kind": arm.kind,
                "b": arm.b,
                "n_keep": int(cov.get("n_keep", 0)),
                "k": corr.get("k"),
                "rho_k": corr.get("rho_k"),
                "rho_mean": corr.get("rho_mean"),
                "rank1_energy_frac": fit0.get("rank1_energy_frac"),
                "offdiag_resid_rel": fit0.get("offdiag_resid_rel"),
                "loading_beta": fit0.get("loading_beta"),
                "sigma_j2": sigma,
                "sigma_j2_signif": fit0.get("sigma_c2_signif"),
                "v_k_median": v_med,
                "sigma_j2_over_v_median": (
                    float(sigma) / v_med
                    if sigma is not None and np.isfinite(v_med) and v_med > 0
                    else float("nan")
                ),
                "failed": cov.get("failed"),
            }
            if i == 0:  # the residual shape is read off the WP18 main arm
                residual = _residual_bands(pn, ser, cov)
    out["arms"] = arms_out
    out["residual_bands"] = residual
    return out


# ---------------------------------------------------------------------------
# units


def build_units(conditions: list[str], args: argparse.Namespace) -> list[Unit]:
    common = {
        "k_max": int(args.k_max),
        "max_lag_s": float(args.max_lag_s),
        "refine_k_max": int(args.refine_k_max),
        "arms": list(args.arms),
        "out": str(args.out),
        "dregon_source": str(args.dregon_source),
        "michaels_spec": str(args.michaels_spec),
        "label_dir": str(args.label_dir),
    }
    plan: list[tuple[str, str]] = []
    for cond in conditions:
        if cond == "single":
            rids = list(SINGLE_RIDS)
            if args.recording:
                rids = [r for r in rids if r in args.recording]
            plan += [(cond, r) for r in rids]
        elif cond == "bench":
            plan.append((cond, BENCH_RID))
        elif cond == "flight_dregon":
            plan += [(cond, r) for r in FLIGHT_DREGON_RIDS]
        elif cond == "flight_fly125":
            plan.append((cond, FLY_RID))
    if args.smoke:
        plan = plan[:1]

    units: list[Unit] = []
    for cond, rid in plan:
        rec = get_recording(cond, rid, common)
        for i, (a0, a1) in enumerate(plan_windows(rec, cond, float(args.window_s))):
            wuid = f"{cond}__{rid}__w{i}"
            for rotor in range(int(rec["r_init"].shape[0])):
                units.append(
                    Unit(
                        uid=f"{wuid}__r{rotor}",
                        params={
                            **common,
                            "condition": cond,
                            "recording": rid,
                            "window": i,
                            "window_uid": wuid,
                            "a0": int(a0),
                            "a1": int(a1),
                            "rotor": rotor,
                        },
                    )
                )
    return units


# ---------------------------------------------------------------------------
# aggregation


def _med(values: list[float]) -> float:
    v = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.median(v)) if v else float("nan")


def _rho_map(row: dict[str, Any], arm: str) -> dict[int, float]:
    a = (row.get("arms") or {}).get(arm) or {}
    ks, rho = a.get("k"), a.get("rho_k")
    if not ks or not rho:
        return {}
    return {int(k): float(r) for k, r in zip(ks, rho) if r is not None and np.isfinite(float(r))}


def _acc() -> dict[str, list[float]]:
    """The per-(condition, harmonic) accumulator of :func:`per_harmonic_rows`."""
    return {"gamma": [], "gamma_slope": [], "tau": [], "cens": [], "fix": [], "ks": [], "lor": []}


def per_harmonic_rows(rows: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    """One row per (condition, harmonic): the curves the paper plots."""
    arm_fix = arms[0]
    arm_ks = arms[1] if len(arms) > 1 else arms[0]
    by_cond: dict[str, dict[int, dict[str, list[float]]]] = {}
    for r in rows:
        if r.get("failed"):
            continue
        cond = str(r["condition"])
        cell = by_cond.setdefault(cond, {})
        for h in (r.get("line") or {}).get("harmonics", []):
            if not h["admitted"]:
                continue
            acc = cell.setdefault(int(h["k"]), _acc())
            if h.get("gamma_hz") is not None and np.isfinite(h["gamma_hz"]):
                acc["gamma"].append(float(h["gamma_hz"]))
                acc["tau"].append(float(h["tau_s"]))
            if h.get("gamma_slope_hz") is not None and np.isfinite(h["gamma_slope_hz"]):
                acc["gamma_slope"].append(float(h["gamma_slope_hz"]))
            acc["cens"].append(1.0 if h.get("censored") else 0.0)
            v = (h.get("shape") or {}).get("verdict")
            if v:
                acc["lor"].append(1.0 if v == "lorentz" else 0.0)
        for arm, key in ((arm_fix, "fix"), (arm_ks, "ks")):
            for k, rho in _rho_map(r, arm).items():
                cell.setdefault(int(k), _acc())[key].append(rho)
    out: list[dict[str, Any]] = []
    for cond in sorted(by_cond):
        for k in sorted(by_cond[cond]):
            acc = by_cond[cond][k]
            out.append(
                {
                    "condition": cond,
                    "k": k,
                    "n_gamma": len(acc["gamma"]),
                    "gamma_hz_median": _med(acc["gamma"]),
                    "tau_s_median": _med(acc["tau"]),
                    "censored_frac": _med(acc["cens"]) if acc["cens"] else float("nan"),
                    "n_gamma_slope": len(acc["gamma_slope"]),
                    "gamma_slope_hz_median": _med(acc["gamma_slope"]),
                    "lorentz_frac": _med(acc["lor"]) if acc["lor"] else float("nan"),
                    "n_rho_fixed": len(acc["fix"]),
                    f"rho_k_{arm_fix}": _med(acc["fix"]),
                    "n_rho_kscaled": len(acc["ks"]),
                    f"rho_k_{arm_ks}": _med(acc["ks"]),
                }
            )
    return out


def per_condition_rows(rows: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    """One row per condition: the numbers a subsection quotes."""
    from tracking import phase_noise as pn

    arm_fix = arms[0]
    arm_ks = arms[1] if len(arms) > 1 else arms[0]
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if not r.get("failed"):
            by_cond.setdefault(str(r["condition"]), []).append(r)
    out: list[dict[str, Any]] = []
    for cond in sorted(by_cond):
        got = by_cond[cond]
        harm = [h for r in got for h in (r.get("line") or {}).get("harmonics", [])]
        law = pn.fit_linewidth_law(*_law_inputs(harm, "gamma_hz"))
        law_slope = pn.fit_linewidth_law(*_law_inputs(harm, "gamma_slope_hz"))
        adm = [h for h in harm if h["admitted"]]
        row: dict[str, Any] = {
            "condition": cond,
            "n_windows": len({(r["recording"], r["window"]) for r in got}),
            "n_rotors": len({(r["recording"], r["window"], r["rotor"]) for r in got}),
            "n_recordings": len({r["recording"] for r in got}),
            "reference": sorted({str((r.get("reference") or {}).get("kind", "")) for r in got})[0],
            "mean_rps": _med([float((r.get("reference") or {})["mean_rps"]) for r in got]),
            "n_admitted": len(adm),
            "censored_frac": (
                round(sum(1 for h in adm if h.get("censored")) / len(adm), 4)
                if adm
                else float("nan")
            ),
            "gamma0_hz": law.get("gamma0_hz", float("nan")),
            "slope_hz_per_k": law.get("slope_hz_per_k", float("nan")),
            "fit_resid_rms_hz": law.get("resid_rms_hz", float("nan")),
            "fit_r2": law.get("r2", float("nan")),
            "n_points": law.get("n", 0),
            "gamma0_slope_hz": law_slope.get("gamma0_hz", float("nan")),
            "slope_slope_hz_per_k": law_slope.get("slope_hz_per_k", float("nan")),
            "fit_slope_resid_rms_hz": law_slope.get("resid_rms_hz", float("nan")),
            "n_points_slope": law_slope.get("n", 0),
        }
        for name, _lo, _hi in pn.K_BANDS:
            fr = [
                float((r.get("shape_bands") or {}).get(name, {}).get("lorentz_frac", np.nan))
                for r in got
            ]
            row[f"lorentz_frac_{name}"] = _med(fr)
            row[f"shape_{name}"] = (
                "lorentz" if np.isfinite(_med(fr)) and _med(fr) >= 0.5 else "gauss"
            )
        for arm, tag in ((arm_fix, "fixed"), (arm_ks, "kscaled")):
            row[f"rho_mean_{tag}"] = _med(
                [(r.get("arms") or {}).get(arm, {}).get("rho_mean") for r in got]
            )
            row[f"rank1_share_{tag}"] = _med(
                [(r.get("arms") or {}).get(arm, {}).get("rank1_energy_frac") for r in got]
            )
            row[f"sigma_j2_over_v_{tag}"] = _med(
                [(r.get("arms") or {}).get(arm, {}).get("sigma_j2_over_v_median") for r in got]
            )
        for name, _lo, _hi in pn.K_BANDS:
            row[f"kurtosis_{name}"] = _med(
                [(r.get("residual_bands") or {}).get(name, {}).get("excess_kurtosis") for r in got]
            )
            row[f"llr_{name}"] = _med(
                [(r.get("residual_bands") or {}).get(name, {}).get("llr_per_sample") for r in got]
            )
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return "" if not np.isfinite(v) else round(v, 6)
    return v


def markdown(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"### {title}\n\n(no rows)\n"
    fields = list(rows[0].keys())
    lines = [f"### {title}", "", "| " + " | ".join(fields) + " |"]
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(_fmt(r.get(k))) for k in fields) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--conditions",
        default="single,bench,flight",
        help=(
            "comma-separated: single, bench, flight_dregon, flight_fly125. "
            "'flight' expands to both flight sub-conditions"
        ),
    )
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--k-max", type=int, default=30, help="highest harmonic measured")
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--max-lag-s", type=float, default=MAX_LAG_S)
    ap.add_argument("--refine-k-max", type=int, default=REFINE_K_MAX)
    ap.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="comma-separated tracking.phase_noise.ARMS names; the FIRST carries the residual shape",
    )
    ap.add_argument("--recording", default="", help="comma-separated ids, for the single condition")
    ap.add_argument("--dregon-source", default=DREGON_SOURCE)
    ap.add_argument("--michaels-spec", default=MICHAELS_SPEC)
    ap.add_argument("--label-dir", default=LABEL_DIR)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one motor recording, 5 s, k <= 10 — the laptop check",
    )
    add_gridrun_args(ap, jobs=4)
    args = ap.parse_args()

    if args.smoke:
        args.conditions = "single"
        args.window_s = 5.0
        args.k_max = 10
        args.refine_k_max = 10
        args.max_lag_s = min(float(args.max_lag_s), 1.0)
        args.recording = args.recording or SINGLE_RIDS[0]

    conditions: list[str] = []
    for name in (v.strip() for v in str(args.conditions).split(",")):
        if not name:
            continue
        if name == "flight":
            conditions += ["flight_dregon", "flight_fly125"]
        elif name in CONDITIONS:
            conditions.append(name)
        else:
            raise SystemExit(f"unknown condition {name!r}; pick from {CONDITIONS} or 'flight'")
    args.recording = {v.strip() for v in str(args.recording).split(",") if v.strip()}
    arms = [v.strip() for v in str(args.arms).split(",") if v.strip()]
    from tracking.phase_noise import ARMS as KNOWN_ARMS

    known = {a.name for a in KNOWN_ARMS}
    bad = [a for a in arms if a not in known]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; pick from {sorted(known)}")
    args.arms = arms

    out = Path(args.out)
    units = build_units(conditions, args)
    if not units:
        raise SystemExit("no units — check --conditions / --recording")
    print(f"[phase_coherence_probe] {len(units)} units -> {out}", flush=True)

    def summarize(raw: list[dict[str, Any]]) -> dict[str, Any]:
        harm = per_harmonic_rows(raw, arms)
        cond = per_condition_rows(raw, arms)
        _write_csv(out / "summary.csv", harm)
        _write_csv(out / "conditions.csv", cond)
        text = markdown(cond, "Conditions") + "\n" + markdown(harm, "Per harmonic")
        (out / "report.md").write_text(text)
        print("\n" + text, flush=True)
        return _jsonable(
            {
                "n_units": len(raw),
                "n_failed_units": sum(1 for r in raw if r.get("failed")),
                "conditions": cond,
                "arms": arms,
                "params": {
                    "k_max": int(args.k_max),
                    "window_s": float(args.window_s),
                    "max_lag_s": float(args.max_lag_s),
                    "refine_k_max": int(args.refine_k_max),
                    "fs_line": FS_LINE,
                    "b_line_max_hz": B_LINE_MAX,
                    "b_line_frac": B_LINE_FRAC,
                    "cruise_rev_s": CRUISE_REV_S,
                },
            }
        )

    res = gridrun_from_args(args, units, probe_worker, out, summarize=summarize)
    raise SystemExit(res.exit_code)


if __name__ == "__main__":
    main()
