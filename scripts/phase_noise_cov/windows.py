#!/usr/bin/env python3
"""Window providers for the phase-noise covariance measurement (WP18).

Three families, chosen so the indoor/outdoor contrast the model predicts is
actually testable and so the estimator itself is validated against a case whose
answer is known:

``dregon``   — INDOOR (DREGON room 1, a reverberant flight arena).  The frozen
               beat-VK protocol windows of the three ``*_room1`` recordings,
               read from (or built into) ``results/beatvk_vk_arms/prep_cache``
               via :func:`beatvk_vk_arms.build_preps`.  Trajectory = DREGON
               ``motors_measured``, the ~1 kHz *measured* rotor telemetry — by
               far the best label in the project.
``michaels`` — OUTDOOR (FLY124 + FLY125, DJI Matrice 100 flights).  Rebuilt
               from the raw WAV+CSV through ``scripts/michaels_calib/windows.py``
               so both recordings are available and both carry the 2026-07-31
               calibration (offset + dilation + rev/s scale).  Trajectory =
               that calibrated ~29 Hz telemetry, linearly interpolated.
``synth``    — the CONTROL.  A locked-phase harmonic comb over OU trajectories,
               demodulated along the EXACT generating trajectory.  With no
               injected timing jitter the common term must measure ~0; with an
               injected arrival-time jitter of known spectrum the measured
               ``sigma_J^2`` must match the value obtained by pushing that same
               jitter through the analysis chain.  Without this arm a nonzero
               ``sigma_J^2`` on real data proves nothing.

Only *cruise* windows are used: at idle/warm-up the comb is weak and the
telemetry lag calibration was fitted on cruise.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "scripts"), str(REPO / "scripts" / "michaels_calib"), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SR = 16000
FRAME_S = 0.032
N_ROTORS = 4
DREGON_RECS = (
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
)
BEATVK_OUT = REPO / "results/beatvk_vk_arms"


@dataclass
class AnalysisWindow:
    """One 16 s window: audio + the trajectory to demodulate along."""

    name: str
    dataset: str  # dregon | michaels | synth
    environment: str  # indoor | outdoor | synthetic
    rid: str
    widx: int
    audio: np.ndarray  # (C, T) @ SR
    ft: np.ndarray  # (N,) window-relative seconds
    r_traj: np.ndarray  # (4, N) rev/s — what we demodulate along
    trajectory: str  # provenance of r_traj
    meta: dict[str, Any] = field(default_factory=dict)
    #: synth only: per-rotor injected arrival-time jitter at audio rate (s).
    jitter: np.ndarray | None = None
    #: Optional higher-rate trajectory for the demodulation carrier:
    #: ``(times, (4, M) rev/s)``, window-relative seconds.  ``r_traj`` on the
    #: 0.032 s frame grid cannot represent shaft motion above ~15 Hz, and the
    #: residual FM that leaves is COMMON to every harmonic — it would be
    #: measured as ``sigma_J^2``.  DREGON's ``motors_measured`` is ~929 Hz and
    #: the synthetic's trajectory is exact, so both supply one.
    r_hires: tuple[np.ndarray, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# DREGON — indoor


def dregon_windows(cache: Path = BEATVK_OUT, build: bool = True) -> list[AnalysisWindow]:
    """The frozen beat-VK cruise windows of the three DREGON room-1 recordings."""
    man_p = cache / "manifest.json"
    if not man_p.exists() and build:
        _build_dregon(cache)
    man = json.loads(man_p.read_text())["recordings"]
    wanted = {
        rid: [int(w["index"]) for w in man[rid]["windows"]] for rid in DREGON_RECS if rid in man
    }
    missing = {
        rid: [w for w in ws if not (cache / "prep_cache" / f"{rid}__w{w:02d}.npz").exists()]
        for rid, ws in wanted.items()
    }
    if any(missing.values()) and build:
        _build_dregon(cache)
    hires = _dregon_native_telemetry(set(wanted))
    out: list[AnalysisWindow] = []
    for rid, ws in wanted.items():
        for widx in ws:
            p = cache / "prep_cache" / f"{rid}__w{widx:02d}.npz"
            if not p.exists():
                continue
            with np.load(p) as z:
                regime = str(z["regime"])
                if regime != "cruise":
                    continue
                start, end = float(z["start_s"]), float(z["end_s"])
                hi = None
                if rid in hires:
                    ts, vals = hires[rid]
                    sel = (ts >= start - 0.05) & (ts <= end + 0.05)
                    if sel.sum() > 100:
                        hi = (ts[sel] - start, vals[:, sel])
                out.append(
                    AnalysisWindow(
                        name=f"dregon_{rid.split('_')[1]}_w{widx:02d}",
                        dataset="dregon",
                        environment="indoor",
                        rid=rid,
                        widx=widx,
                        audio=np.asarray(z["audio"], dtype=np.float64),
                        ft=np.asarray(z["ft"], dtype=np.float64),
                        r_traj=np.asarray(z["r_meas"], dtype=np.float64),
                        trajectory=(
                            "DREGON motors_measured, native ~929 Hz"
                            if hi is not None
                            else "DREGON motors_measured resampled to the 0.032 s frame grid"
                        ),
                        r_hires=hi,
                        meta={
                            "regime": regime,
                            "start_s": start,
                            "end_s": end,
                            "co_recorded_source": rid != "free-flight_nosource_room1",
                            "room": "room1",
                            "hires_traj": hi is not None,
                        },
                    )
                )
    return out


def _dregon_native_telemetry(rids: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{rid: (times, (4, M) rev/s)}`` at the NATIVE ~929 Hz sample rate.

    Cached, because the only route to it is a (cheap, audio-free) pass over the
    published ``beatvk-valid-raw`` frames.  Returns ``{}`` if the stream is not
    reachable — the frame-grid trajectory then stands, flagged in the manifest.
    """
    p = REPO / ".cache/phase_noise_cov/dregon_native_rps.npz"
    if p.exists():
        with np.load(p) as z:
            return {r: (z[f"ts__{r}"], z[f"vals__{r}"]) for r in rids if f"ts__{r}" in z}
    try:
        import beatvk_eval as bve

        recs = bve.load_recordings(None, rids, keep_audio=False)
    except Exception as exc:  # noqa: BLE001 — optional refinement, never fatal
        print(f"[warn] native DREGON telemetry unavailable ({exc}); using the frame grid")
        return {}
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for r in recs:
        payload[f"ts__{r['recording_id']}"] = r["ts"]
        payload[f"vals__{r['recording_id']}"] = r["vals"]
    np.savez(p, **payload)
    return {r["recording_id"]: (r["ts"], r["vals"]) for r in recs}


def _build_dregon(cache: Path) -> None:
    """Materialize the prep NPZs (streams ``beatvk-valid-raw`` if needed)."""
    import beatvk_vk_arms as bva

    man = bva.load_manifest(cache, set(DREGON_RECS), None)
    jobs = {
        rid: [int(w["index"]) for w in man["recordings"][rid]["windows"]]
        for rid in DREGON_RECS
        if rid in man["recordings"]
    }
    # build_preps also writes per-recording mic weights, and computing those
    # for DREGON materializes the whole ~GB dataset just to read its geometry.
    # Nothing here uses them (the covariance is per channel, unweighted), so a
    # placeholder is written first and the download is skipped entirely.
    (cache / "prep_cache").mkdir(parents=True, exist_ok=True)
    for rid in jobs:
        wp = bva.weights_path(cache, rid)
        if not wp.exists():
            np.savez(wp, weights=np.full((8, N_ROTORS), 1.0 / 8.0))
    bva.build_preps(cache, jobs, None, "dload:DREGON")


# ---------------------------------------------------------------------------
# Michael's — outdoor


def _load_michaels_calib_windows() -> Any:
    """``scripts/michaels_calib/windows.py``, loaded BY PATH.

    A plain ``import windows`` would return *this* module: both files are named
    ``windows`` and both directories are on ``sys.path``, so whichever imports
    first wins ``sys.modules`` and the Michael's arm would silently rebuild
    itself out of the DREGON loader.
    """
    import importlib.util

    path = REPO / "scripts/michaels_calib/windows.py"
    spec = importlib.util.spec_from_file_location("michaels_calib_windows", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["michaels_calib_windows"] = mod
    spec.loader.exec_module(mod)
    return mod


def michaels_windows(
    cache_dir: Path = REPO / ".cache/phase_noise_cov",
    rids: tuple[str, ...] = ("FLY124", "FLY125"),
) -> list[AnalysisWindow]:
    """Every CRUISE window of both Michael's recordings, calibrated telemetry."""
    MW = _load_michaels_calib_windows()
    cache_dir.mkdir(parents=True, exist_ok=True)
    man = MW.build_cache(cache_dir, rids=rids)
    out: list[AnalysisWindow] = []
    for rid in rids:
        rec = man["recordings"][rid]
        for w in rec["windows"]:
            if w["regime"] != "cruise":
                continue
            win = MW.load_cached(cache_dir, str(w["name"]))
            out.append(
                AnalysisWindow(
                    name=f"{rid.lower()}_w{win.widx:02d}",
                    dataset="michaels",
                    environment="outdoor",
                    rid=rid,
                    widx=win.widx,
                    audio=win.audio,
                    ft=win.ft,
                    r_traj=win.r_meas,
                    trajectory=(
                        "Michael's calibrated telemetry (~29 Hz, WP13/WP14 offset "
                        f"{rec['time_offset']:.6f} s, dilation {rec['time_dilation']:.9f}, "
                        f"rev/s scale {MW.shipped_rps_scale(rid):.5f}), linearly interpolated"
                    ),
                    meta={
                        "regime": win.regime,
                        "start_s": win.start_s,
                        "end_s": win.end_s,
                        "co_recorded_source": False,
                    },
                )
            )
    return out


# ---------------------------------------------------------------------------
# synthetic control


def synth_window(
    seed: int,
    jitter_s: float,
    *,
    dur: float = 16.0,
    snr_db: float = 0.0,
    n_channels: int = 4,
    jitter_fc: float = 25.0,
    shaft_fc: float | None = 8.0,
) -> AnalysisWindow:
    """Locked-phase 4-rotor comb + a KNOWN arrival-time jitter.

    Mirrors ``scripts/rps_refine_lab.py:synth_window``'s generation path (OU
    free-flight modes, 2-blade emphasis, 30 harmonics, noise set relative to the
    comb RMS) with one addition: each rotor's shaft phase carries an
    arrival-time error ``n(t)`` — white, lowpassed to ``jitter_fc`` Hz, std
    ``jitter_s`` seconds — so harmonic ``k``'s phase error is exactly
    ``2 pi k r n(t)``, the model under test.  ``jitter_s = 0`` is the NULL
    control: the only common term left is the (zero) trajectory error.

    ``shaft_fc`` band-limits the OU shaft speed BEFORE synthesis (WP4 item 5's
    physical convention: a real rotor's inertia cannot follow a drive that is
    white to 250 Hz).  It is load-bearing HERE, not cosmetic: without it the
    null control measures a common term as large as a 100 us jitter, because
    the shaft motion above the frame-grid Nyquist is un-representable and its
    residual FM is common to every harmonic.  That is the same artefact real
    data can suffer, which is why the demodulation carrier is the exact
    audio-rate trajectory (``r_hires``) rather than the frame-grid one.

    Channels are independent noise realizations of the same comb: the shaft
    jitter is common to all of them, so the channel-coherence diagnostic has a
    known answer here too (1.0).
    """
    from scipy.signal import filtfilt, firwin

    from data_processing.rps_synthesis import MIXER, OUModeParams, RPSSynthConfig
    from data_processing.rps_synthesis import generate as rps_generate

    rng = np.random.default_rng(seed)
    for _ in range(200):
        modes = np.array(
            [rng.uniform(76.0, 94.0), rng.uniform(-3, 3), rng.uniform(-6, 6), rng.uniform(-4, 4)]
        )
        rotor_means = MIXER @ modes
        seps = np.abs(rotor_means[:, None] - rotor_means[None, :])[np.triu_indices(4, 1)]
        if rotor_means.min() >= 70.0 and rotor_means.max() <= 100.0 and seps.min() >= 2.0:
            break
    else:
        raise RuntimeError(f"synth seed {seed}: no valid rotor-mean draw")
    cfg = RPSSynthConfig(
        common=OUModeParams(mean=float(modes[0]), std=1.5, tau=0.70),
        roll=OUModeParams(mean=float(modes[1]), std=0.70, tau=0.60),
        pitch=OUModeParams(mean=float(modes[2]), std=0.85, tau=0.75),
        yaw=OUModeParams(mean=float(modes[3]), std=1.40, tau=1.00),
    )
    n_t = int(dur * SR)
    t = np.arange(n_t) / SR
    fs_traj = 250.0
    r_lo = rps_generate(dur, fs_traj, config=cfg, aggressiveness=1.0, rng=rng)
    if shaft_fc is not None:
        taps_s = firwin(255, shaft_fc / (fs_traj / 2), window="hamming")
        r_lo = filtfilt(taps_s, [1.0], r_lo, axis=1)
    t_lo = np.arange(r_lo.shape[1]) / fs_traj
    r_true = np.stack([np.interp(t, t_lo, r_lo[i]) for i in range(N_ROTORS)])

    jit = np.zeros((N_ROTORS, n_t))
    if jitter_s > 0:
        taps = firwin(511, jitter_fc / (SR / 2), window="hamming")
        for i in range(N_ROTORS):
            w = rng.normal(0.0, 1.0, n_t)
            f = filtfilt(taps, [1.0], w)
            jit[i] = jitter_s * f / max(float(np.std(f)), 1e-30)

    k_max = 30
    psi = rng.uniform(0, 2 * np.pi, (N_ROTORS, k_max))
    comb = np.zeros(n_t)
    for i in range(N_ROTORS):
        phi = 2 * np.pi * np.cumsum(r_true[i]) / SR + 2 * np.pi * r_true[i] * jit[i]
        for k in range(1, k_max + 1):
            amp = (1.6 if k % 2 == 0 else 0.5) / k
            comb += amp * np.cos(k * phi + psi[i, k - 1])
    comb_rms = float(np.sqrt(np.mean(comb**2)))
    sigma_n = comb_rms * 10 ** (-snr_db / 20.0)
    audio = np.stack([comb + rng.normal(0.0, sigma_n, n_t) for _ in range(n_channels)])

    ft = np.arange(0.0, dur - FRAME_S / 2, FRAME_S)
    r_ft = np.stack([np.interp(ft, t, r_true[i]) for i in range(N_ROTORS)])
    tag = ("null" if jitter_s == 0 else f"jit{jitter_s * 1e6:g}us") + f"@{snr_db:g}dB"
    return AnalysisWindow(
        name=f"synth_{tag}_s{seed}",
        dataset="synth",
        environment="synthetic",
        rid=f"synth_{tag}",
        widx=seed,
        audio=audio,
        ft=ft,
        r_traj=r_ft,
        r_hires=(t, r_true),
        trajectory="EXACT generating trajectory (zero trajectory error by construction)",
        meta={
            "regime": "cruise",
            "snr_db": snr_db,
            "jitter_s": jitter_s,
            "jitter_fc_hz": jitter_fc,
            "shaft_fc_hz": shaft_fc,
            "rotor_means": np.round(np.sort(rotor_means), 3).tolist(),
            "co_recorded_source": False,
        },
        jitter=jit if jitter_s > 0 else None,
    )


#: (jitter seconds, comb-vs-noise SNR dB).  The 20 dB pair is where the
#: estimator's recovery of a KNOWN common term is actually testable — at 0 dB
#: the additive term ``v_k`` is ~100x larger and buries it, which is itself a
#: result worth having on the record.
SYNTH_CASES: tuple[tuple[float, float], ...] = (
    (0.0, 20.0),
    (1e-4, 20.0),
    (0.0, 0.0),
    (1e-4, 0.0),
)


def synth_windows(
    seeds: tuple[int, ...] = (11, 12), cases: tuple[tuple[float, float], ...] = SYNTH_CASES
) -> list[AnalysisWindow]:
    return [synth_window(s, j, snr_db=snr) for j, snr in cases for s in seeds]


# ---------------------------------------------------------------------------


def build(which: str) -> list[AnalysisWindow]:
    """``which`` = comma list of ``dregon`` | ``michaels`` | ``synth`` | ``all``."""
    names = [w.strip() for w in which.split(",") if w.strip()]
    if "all" in names:
        names = ["synth", "dregon", "michaels"]
    out: list[AnalysisWindow] = []
    for n in names:
        if n == "dregon":
            out += dregon_windows()
        elif n == "michaels":
            out += michaels_windows()
        elif n == "synth":
            out += synth_windows()
        else:
            raise SystemExit(f"unknown window family {n!r}")
    return out
