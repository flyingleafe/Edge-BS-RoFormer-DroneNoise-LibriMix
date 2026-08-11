#!/usr/bin/env python3
"""F_VK arms campaign — paper plan steps 4 and 5, one restartable grid.

Two questions, one driver, one unit format (:mod:`utils.gridrun`).

**Step 4 — telemetry IMPROVEMENT.** Given a close but wrong init (real
tachometer telemetry, or a synthetic truth corrupted the same way), how much
does each refiner move it, and at what cost? Three arms, every one a
``Frame -> Frame`` composition of shipped stages:

``pikalman``
    :func:`tracking.refit_stage` — the phase 6c procedure exactly: presmooth
    5 Hz, then coarse-to-fine (peel -> pi_kalman) to convergence.
``iavkf``
    The closest honest IAVKF analogue this stack can assemble from shipped
    stages: :func:`tracking.vk_stage` under ``CAPTURE_CFG`` (annealed grow
    schedule, k 6-30, bw 1.5 Hz, ``n_outer`` 12) with ``bw_adapt=True`` — the
    IEEE TII 2024 per-track bandwidth adaptation — followed by
    :func:`tracking.refine_coherent_stage`, the envelope-phase-slope IF
    estimator. Adaptive-bandwidth VK order tracking plus a phase-based IF
    update is what the paper is; nothing else in the stack is closer.
``lbfgs``
    :func:`tracking.fvk_refine_stage` — L-BFGS on F_VK under the default
    ``k_max`` annealing schedule.

**Step 5 — BLIND annotation Pareto.** No close init at all. Four contestants:

``ours_full``
    :func:`tracking.blind_fullrange` — blind seed (arms K, R) -> coarse
    full-range Viterbi -> the calibrated vit2dsp ladder.
``seed_only``
    Its first two stages alone (blind seed -> coarse Viterbi c(t)) — the
    DP/ridge-tracking baseline.
``seed_vk``
    ``seed_only`` plus ONE :func:`tracking.vk_stage` refine pass
    (``REFINE_CFG``) — the cheap middle rung.
``multistart``
    :func:`tracking.optimize_trajectory` from ``N`` random constant-rate
    inits, keeping the best final objective — the expensive precision anchor.
    Cost control: the first rungs of the schedule screen every start, and only
    the best-objective start pays the remaining rungs (``--ms-screen`` /
    ``--ms-finish``); with ``--ms-finish`` empty every start runs the full
    schedule.

Windows
-------
``synth``
    In-script comb windows (``--synth-seconds``, 16 kHz, 4 rotors including one
    DREGON-like twin pair 0.3-1.0 rev/s apart). The trajectory comes from
    :func:`data_processing.rps_synthesis.generate`; the comb is ``1/k``
    amplitudes with WP18-shaped phase noise (harmonic ``k``'s phase carries an
    OU jitter of std ``k * phase_sigma`` rad, i.e. variance ~ ``k^2``, so the
    high harmonics decohere first — the measured behaviour), plus white noise
    at ``--snr-db``. Truth is exact.
``dregon`` / ``fly124``
    The frozen ``beatvk`` prep windows (:func:`tracking.load_prep_window`).
    DREGON carries no truth, so it is scored by F_VK alone; FLY124 is scored
    against its stored labels as well (those labels are the FROZEN
    pre-2026-07-31 alignment — read as "does the arm degrade them", not as an
    absolute accuracy).

Run::

    python scripts/fvk_arms.py --step 4 --classes synth --jobs 8 --out results/fvk_arms
    python scripts/fvk_arms.py --step 5 --classes synth,dregon,fly124 --jobs 8
    python scripts/fvk_arms.py --figs            # figures from summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# The harness convention: process-level parallelism, one BLAS thread per
# worker (utils.gridrun re-asserts it, but the tracking stack reads its own
# thread knob at import time).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TRACKING_FFT_WORKERS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

# ---------------------------------------------------------------------------
# window definitions

SR = 16000
#: The three DREGON prep recordings x 3 windows each (all of them).
DREGON_WINDOWS = tuple(
    f"{rid}__w{i:02d}"
    for rid in (
        "free-flight_nosource_room1",
        "free-flight_speech-low_room1",
        "free-flight_whitenoise-low_room1",
    )
    for i in range(3)
)
#: Three FLY124 cruise windows (sanity — the stored labels must not degrade).
FLY124_WINDOWS = ("FLY124__w03", "FLY124__w04", "FLY124__w05")
#: The five real windows of the step-5 blind Pareto: one cruise window from each
#: DREGON recording (not w00, which is the takeoff ramp) plus two FLY124 cruise
#: windows — a mix of both rigs, and every recording represented once.
BLIND_REAL = (
    "free-flight_nosource_room1__w01",
    "free-flight_speech-low_room1__w01",
    "free-flight_whitenoise-low_room1__w01",
    "FLY124__w04",
    "FLY124__w05",
)

#: DREGON's measured telemetry scale bias (the step-4 synthetic corruption).
DREGON_SCALE = 0.99317
#: The tachometer lattice and refresh of DREGON's flight controller.
TACH_STEP = 0.269
TACH_REFRESH_HZ = 49.7

#: Plausible blind search range for the multi-start inits (rev/s) — the blind
#: seeder's own scan range, narrowed to the cruise band the windows live in.
BLIND_LO, BLIND_HI = 60.0, 105.0

#: Synthetic-window constants, the same values as ``scripts/fvk_bench.py`` so
#: the two campaigns' synthetic material is one signal model.
FS_TRAJ = 250.0  #: OU drive rate before the shaft-inertia low pass
TRAJ_FC_HZ = 8.0  #: shaft inertia — also what makes the truth frame-representable
JITTER_K1 = 0.01  #: rad/sqrt(s) of Wiener phase noise at k = 1; variance prop k^2

STEP4_ARMS = ("pikalman", "iavkf", "lbfgs")
STEP5_ARMS = ("ours_full", "seed_only", "seed_vk", "multistart")


# ---------------------------------------------------------------------------
# synthetic windows


def _draw_rotor_means(rng: Any) -> tuple[tuple[float, ...], Any]:
    """Four control-mode means whose rotors hold ONE DREGON-like twin pair.

    Rejection sampling on the mixer: the smallest pairwise rotor separation
    must sit in [0.3, 1.0] rev/s (the twins DREGON's audio cannot resolve) and
    every other separation must exceed 2 rev/s, so the window has exactly one
    hard pair and three resolvable rotors.
    """
    import numpy as np

    from tracking.rotors import MIXER

    for _ in range(500):
        means = (
            float(rng.uniform(76.0, 94.0)),
            float(rng.uniform(-2.0, 2.0)),
            float(rng.uniform(-3.0, 3.0)),
            float(rng.uniform(-2.0, 2.0)),
        )
        rotor_means = MIXER @ np.array(means)
        seps = np.sort(np.abs(rotor_means[:, None] - rotor_means[None, :])[np.triu_indices(4, 1)])
        if (
            0.3 <= seps[0] <= 1.0
            and seps[1] >= 2.0
            and rotor_means.min() >= 70.0
            and rotor_means.max() <= 100.0
        ):
            return means, rotor_means
    raise RuntimeError("no valid rotor-mean draw in 500 tries")


def synth_window(
    seed: int,
    *,
    dur: float = 2.0,
    sr: int = SR,
    k_max: int = 40,
    snr_db: float = 0.0,
    jitter_k1: float = JITTER_K1,
    n_mic: int = 2,
    blade_pass: bool = True,
) -> dict[str, Any]:
    """One synthetic comb window with an exactly known trajectory.

    Draw order is fixed (means, OU trajectory, harmonic phase + walk, white
    noise), so a seed reproduces the window bit-exactly.

    Three things make the TRUTH exact rather than nearly exact:

    - the OU drive is low-passed at :data:`TRAJ_FC_HZ` (shaft inertia), so
      point-sampling it onto the frame grid does not alias;
    - the audio is synthesized from the FRAME-grid truth interpolated up, so
      the array a candidate is compared against is the array the comb was made
      from;
    - a fixed 8 s OU draw is filtered and then cut, so ``filtfilt`` always has
      the samples it needs and a short window is a prefix of a long one.

    The comb: rotor ``i``, harmonic ``k`` contributes
    ``a_k cos(k phi_i(t) + psi_ik + w_ik(t))`` with ``w_ik`` a Wiener process of
    rate ``jitter_k1 * k`` rad/sqrt(s) — variance ``prop k^2``, the WP18 law,
    drawn INDEPENDENTLY per harmonic because WP18 refuted the rank-one
    (pure shaft-jitter) form. Same law and same constant as
    ``scripts/fvk_bench.render_comb``, so the two campaigns' synthetic material
    differs only in what each needs (this one is 4 rotors, because the blind
    ladder is a 4-track algorithm). White noise is added at ``snr_db`` relative
    to the comb RMS.

    ``a_k`` is ``1/k`` with a 2-blade blade-pass emphasis (even ``1.6/k``, odd
    ``0.5/k`` — the convention of
    :func:`data_processing.rps_synthesis.synth_comb_window`), because that is
    the regime the stack's blind octave rule is calibrated for and the regime a
    2-blade quadrotor is in. ``blade_pass=False`` is the flat ``1/k`` comb; it
    is measurably outside that calibration (measured: every blind ladder
    octave-fails on it), which is why it is not the default.
    """
    import math

    import numpy as np
    from scipy.signal import filtfilt, firwin

    from data_processing.rps_synthesis import generate

    rng = np.random.default_rng(seed)
    means, rotor_means = _draw_rotor_means(rng)
    hop_s = 0.032
    ft = np.arange(0.0, dur - hop_s / 2, hop_s)
    r_lo = generate(8.0, FS_TRAJ, aggressiveness=1.0, rng=rng)
    taps = firwin(255, TRAJ_FC_HZ / (FS_TRAJ / 2.0), window="hamming")
    r_lo = np.asarray(filtfilt(taps, [1.0], r_lo, axis=1))
    r_lo = np.stack([row - row.mean() + m for row, m in zip(r_lo, rotor_means, strict=True)])
    t_lo = np.arange(r_lo.shape[1]) / FS_TRAJ
    r_ft = np.stack([np.interp(ft, t_lo, row) for row in r_lo])

    n_t = int(round(dur * sr))
    t = np.arange(n_t) / sr
    r_true = np.stack([np.interp(t, ft, row) for row in r_ft])

    dt = 1.0 / sr
    comb = np.zeros(n_t)
    for i in range(r_true.shape[0]):
        phi = 2 * np.pi * np.cumsum(r_true[i]) / sr
        for k in range(1, k_max + 1):
            psi = float(rng.uniform(0.0, 2 * np.pi))
            walk = np.cumsum(rng.normal(0.0, jitter_k1 * k * math.sqrt(dt), n_t))
            amp = ((1.6 if k % 2 == 0 else 0.5) if blade_pass else 1.0) / k
            comb += amp * np.cos(k * phi + psi + walk)
    comb_rms = float(np.sqrt(np.mean(comb**2)))
    noise = rng.normal(0.0, comb_rms * 10 ** (-snr_db / 20.0), (n_mic, n_t))
    audio = comb[None, :] + noise

    return {
        "audio": audio,
        "ft": ft,
        "r_true": r_ft,
        "r_true_audio": r_true,
        "rotor_means": [float(v) for v in rotor_means],
        "sr": sr,
        "meta": {
            "seed": seed,
            "snr_db": snr_db,
            "k_max": k_max,
            "duration_s": dur,
            "blade_pass": blade_pass,
        },
    }


def corrupt_init(r_true_audio: Any, ft: Any, sr: int, seed: int) -> Any:
    """DREGON-style corruption of a true trajectory -> the step-4 init.

    Constant scale ``DREGON_SCALE``, then the tachometer measurement model
    (:func:`data_processing.tachometer_corrupt` — refresh-interval mean,
    0.269 rev/s lattice, zero-order hold), then a small smooth OU wander
    (0.15 rev/s, 1 s) so the init is not a pure deterministic function of the
    truth. Returned on the frame grid ``ft``.
    """
    import numpy as np
    from scipy.signal import lfilter

    from data_processing.rps_corruption import tachometer_corrupt

    corrupt = tachometer_corrupt(
        np.asarray(r_true_audio, dtype=np.float64),
        float(sr),
        step=TACH_STEP,
        refresh_hz=TACH_REFRESH_HZ,
        scale=DREGON_SCALE,
    )
    t = np.arange(corrupt.shape[-1]) / sr
    rng = np.random.default_rng(10_000 + seed)
    a = float(np.exp(-(1.0 / sr) / 1.0))
    out = []
    for row in corrupt:
        innov = rng.normal(0.0, 0.15 * float(np.sqrt(1.0 - a * a)), size=row.size)
        innov[0] = rng.normal(0.0, 0.15)
        out.append(row + lfilter([1.0], [1.0, -a], innov))
    corrupt = np.stack(out)
    return np.stack([np.interp(ft, t, row) for row in corrupt])


# ---------------------------------------------------------------------------
# real windows


def load_window(params: dict[str, Any]) -> dict[str, Any]:
    """One window's arrays, whatever its class."""
    import numpy as np

    if params["class"] == "synth":
        w = synth_window(
            int(params["seed"]),
            dur=float(params["seconds"]),
            snr_db=float(params["snr_db"]),
            n_mic=int(params["max_channels"]),
            blade_pass=bool(params.get("blade_pass", True)),
        )
        w["r_ref"] = w["r_true"]
        return w

    from tracking.protocols import load_prep_window

    pdir = params.get("prep_dir")
    z = load_prep_window(str(params["window"]), Path(pdir) if pdir else None)
    audio, ft, r = z["audio"], z["ft"], z["r"]
    sec = params.get("seconds")
    if sec:
        n_a = min(audio.shape[-1], int(round(float(sec) * SR)))
        keep = ft < (n_a / SR - 1e-9)
        audio, ft, r = audio[:, :n_a], ft[keep], r[:, keep]
    n_ch = int(params["max_channels"])
    audio = np.ascontiguousarray(audio[:n_ch], dtype=np.float64)
    return {
        "audio": audio,
        "ft": np.asarray(ft, dtype=np.float64),
        "r_true": r if params["class"] == "fly124" else None,
        "r_ref": np.asarray(r, dtype=np.float64),
        "sr": SR,
        "meta": {"window": params["window"], "regime": z["regime"]},
    }


# ---------------------------------------------------------------------------
# metrics


def _interior(ft: Any, trim_s: float = 0.25) -> Any:
    import numpy as np

    ft = np.asarray(ft)
    return (ft > ft[0] + trim_s) & (ft < ft[-1] - trim_s)


def traj_error(pred: Any, truth: Any, ft: Any) -> dict[str, Any]:
    """PIT-aligned rms / mae of ``pred`` against ``truth`` on the interior."""
    import numpy as np

    from tracking.protocols import pit_align

    mask = _interior(ft)
    aligned, perm = pit_align(pred, truth, cost="mse", edge_mask=mask)
    d = (aligned - truth)[:, mask]
    return {
        "rms": float(np.sqrt(np.mean(d**2))),
        "mae": float(np.mean(np.abs(d))),
        "rms_per_rotor": [float(np.sqrt(np.mean(d[i] ** 2))) for i in range(d.shape[0])],
        "perm": perm,
    }


def score_fvk(audio: Any, sr: int, r: Any, ft: Any, ref: Any, k_max: int, max_ch: int) -> dict:
    """The window's F_VK reading of one trajectory (cells pinned by ``ref``)."""
    from tracking.fitness_vk import FVKConfig, fvk_score

    cfg = FVKConfig(sr=sr, k_max=k_max, bw_rps=1.0, max_channels=max_ch)
    s = fvk_score(audio, sr, r, ft, cfg, reference=ref)
    return {
        "objective": float(s["objective"]),
        "r2": float(s["r2"]),
        "residual": float(s["residual"]),
        "k_hi": int(s["k_hi"]),
        "n_cells": int(s["n_cells"]),
    }


# ---------------------------------------------------------------------------
# the arms


def _fvk_cfg(sr: int, k_max: int, max_ch: int) -> Any:
    from tracking.fitness_vk import FVKConfig

    return FVKConfig(sr=sr, k_max=k_max, bw_rps=1.0, max_channels=max_ch)


def step4_stage(arm: str, sr: int, n_rotors: int, params: dict[str, Any]) -> Any:
    """The step-4 arm as one ``Frame -> Frame`` stage."""
    from dataclasses import replace

    import tracking as trk
    from tracking.pipelines import CAPTURE_CFG
    from tracking.telemetry_refit import RefitConfig

    k_max = int(params["k_max"])
    max_ch = int(params["max_channels"])
    if arm == "pikalman":
        return trk.refit_stage(cfg=RefitConfig(max_iters=int(params["refit_max_iters"])))
    if arm == "iavkf":
        vk = replace(CAPTURE_CFG, fs=float(sr), bw_adapt=True)
        return trk.pipeline(
            trk.vk_stage(vk, name="iavkf_vk"),
            trk.refine_coherent_stage(
                k_min=6, k_max=min(30, k_max), bandwidth_hz=3.0, n_iter=4, name="iavkf_phase"
            ),
        )
    if arm == "lbfgs":
        return trk.fvk_refine_stage(_fvk_cfg(sr, k_max, max_ch), knot_s=0.25, smooth_lambda=1.0)
    raise ValueError(f"unknown step-4 arm {arm!r}")


def step5_stage(arm: str, sr: int, params: dict[str, Any]) -> Any:
    """The step-5 contestant as one ``Frame -> Frame`` stage (blind)."""
    from dataclasses import replace

    import tracking as trk
    from tracking.pipelines import REFINE_CFG, SEED_CFG, CoarseConfig

    seed_pipe = trk.pipeline(
        trk.blind_seed_stage(4, SEED_CFG, arms=("K", "R")),
        trk.coarse_init_stage(CoarseConfig()),
    )
    if arm == "ours_full":
        return trk.blind_fullrange()
    if arm == "seed_only":
        return seed_pipe
    if arm == "seed_vk":
        return trk.pipeline(seed_pipe, trk.vk_stage(replace(REFINE_CFG, fs=float(sr))))
    if arm == "multistart":
        return _multistart_stage(sr, params)
    raise ValueError(f"unknown step-5 arm {arm!r}")


def _parse_schedule(spec: str) -> tuple:
    """``"5:15,10:15"`` -> a tuple of :class:`FVKStage` (``k_max:max_iter``)."""
    from tracking.fitness_vk import FVKStage

    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        k, _, it = part.partition(":")
        out.append(FVKStage(int(k), max_iter=int(it) if it else 20))
    return tuple(out)


def _multistart_stage(sr: int, params: dict[str, Any]) -> Any:
    """Multi-start L-BFGS on F_VK from ``n_starts`` random constant inits.

    Start ``s`` is ONE base rate drawn uniform in ``[BLIND_LO, BLIND_HI]``,
    given to all rotors with the seeder's own ``blind_offsets`` split
    (±0.5, ±1.5 rev/s) so the tracks are not degenerate at init.

    The basin is ``bw_rps / 2`` rev/s (``fitness_vk`` § "Which knob is the
    basin"), so a start 10 rev/s from truth cannot be pulled in at the shipped
    ``bw_rps = 1``. The screening rungs therefore run under a WIDE
    ``ms_bw_rps`` (capture radius ``ms_bw_rps / 2``) and the finishing rungs
    under the standard config — a coarse-to-fine continuation in the knob that
    actually moves the basin. Every start pays the screen; the best start, by
    the STANDARD config's objective so the selection is not made under the wide
    band, pays the finish. An empty ``--ms-finish`` means every start runs the
    whole schedule and the best final objective wins.
    """
    from dataclasses import replace

    import numpy as np

    import tracking as trk
    from tracking.fitness_vk import fvk_score, optimize_trajectory
    from tracking.pipelines import SEED_CFG

    n_starts = int(params["ms_starts"])
    screen = _parse_schedule(str(params["ms_screen"]))
    finish = _parse_schedule(str(params["ms_finish"]))
    k_max, max_ch = int(params["k_max"]), int(params["max_channels"])
    seed = int(params["ms_seed"])
    bw_wide = float(params["ms_bw_rps"])

    def run(frame):
        audio, sr_f = trk.get_audio(frame)
        ref = trk.get_rps(frame, "rps_meas")[0] if "rps_meas" in frame else None
        _, ft = trk.get_rps(frame) if "rps" in frame else (None, None)
        if ft is None:
            raise ValueError("multistart needs a frame grid — give the frame an 'rps' entry")
        t0 = float(frame["audio"].t_start)
        cfg = _fvk_cfg(int(round(sr_f)), k_max, max_ch)
        cfg_wide = replace(cfg, bw_rps=bw_wide)
        rng = np.random.default_rng(seed)
        n_rot = ref.shape[0] if ref is not None else 4
        offsets = np.asarray(SEED_CFG.blind_offsets, dtype=np.float64)[:n_rot]
        bases = rng.uniform(BLIND_LO, BLIND_HI, size=n_starts)
        rel = ft - t0
        cands = []
        for s in range(n_starts):
            start = bases[s] + offsets
            r0 = np.tile(start[:, None], (1, ft.size))
            r_s, diag_s = optimize_trajectory(
                audio, sr_f, r0, rel, cfg_wide, schedule=screen or None, reference=ref
            )
            obj = fvk_score(audio, sr_f, r_s, rel, cfg, reference=ref)["objective"]
            cands.append({"start": [float(v) for v in start], "obj": float(obj)})
            cands[-1]["_r"] = r_s
            cands[-1]["_diag"] = diag_s
        best = min(range(len(cands)), key=lambda i: cands[i]["obj"])
        r_best = cands[best]["_r"]
        diag_finish = None
        if finish:
            r_best, diag_finish = optimize_trajectory(
                audio, sr_f, r_best, rel, cfg, schedule=finish, reference=ref
            )
        info = {
            "n_starts": n_starts,
            "best_start": best,
            "screen_objs": [round(c["obj"], 6) for c in cands],
            "starts": [c["start"] for c in cands],
            "screen_diag": cands[best]["_diag"],
            "finish_diag": diag_finish,
        }
        return trk.with_rps(frame, r_best, ft, stage="multistart", info=info)

    return run


# ---------------------------------------------------------------------------
# the worker


def worker(unit: Unit) -> dict[str, Any]:
    import numpy as np

    import tracking as trk

    p = dict(unit.params)
    win = load_window(p)
    audio, ft, sr = win["audio"], win["ft"], int(win["sr"])
    truth = win.get("r_true")
    ref = win["r_ref"]
    k_max, max_ch = int(p["k_max"]), int(p["max_channels"])

    if p["step"] == 4:
        if p["class"] == "synth":
            r_init = corrupt_init(win["r_true_audio"], ft, sr, int(p["seed"]))
        else:
            r_init = np.asarray(ref, dtype=np.float64)
        frame = trk.tracking_frame(
            audio, sr, rps=r_init, frame_times=ft, rps_meas=r_init, dtype=np.float64
        )
        stage = step4_stage(str(p["arm"]), sr, r_init.shape[0], p)
    else:
        r_init = np.asarray(ref, dtype=np.float64)
        frame = trk.tracking_frame(
            audio, sr, rps=r_init, frame_times=ft, rps_meas=r_init, dtype=np.float64
        )
        stage = step5_stage(str(p["arm"]), sr, p)

    row: dict[str, Any] = {
        "uid": unit.uid,
        "step": int(p["step"]),
        "arm": str(p["arm"]),
        "class": str(p["class"]),
        # the reporting group: a synthetic SNR is its own window class
        "group": (
            f"synth{float(p['snr_db']):+.0f}dB" if p["class"] == "synth" else str(p["class"])
        ),
        "window": str(p.get("tag") or p.get("window")),
        "n_frames": int(ft.size),
        "n_channels": int(audio.shape[0]),
        "duration_s": round(float(audio.shape[-1]) / sr, 3),
        "meta": win["meta"],
    }

    # the init reading (step 5's init is a placeholder, so only its score of
    # the reference trajectory is meaningful — reported as "before")
    row["fvk_before"] = score_fvk(audio, sr, r_init, ft, ref, k_max, max_ch)
    if truth is not None and p["step"] == 4:
        row["err_before"] = traj_error(r_init, truth, ft)

    tic = time.perf_counter()
    out = stage(frame)
    row["wall_s"] = round(time.perf_counter() - tic, 2)

    r_out, ft_out = trk.get_rps(out)
    if r_out.shape[-1] != ft.size or not np.allclose(ft_out, ft):
        r_out = np.stack([np.interp(ft, ft_out, rowv) for rowv in r_out])
    row["fvk_after"] = score_fvk(audio, sr, r_out, ft, ref, k_max, max_ch)
    if truth is not None:
        row["err_after"] = traj_error(r_out, truth, ft)
    if p["step"] == 5 and p["class"] != "synth":
        # no truth on DREGON; on FLY124 the stored labels ARE the reference
        row["err_vs_ref"] = traj_error(r_out, ref, ft)
    row["stages"] = [
        {k: v for k, v in e.items() if k in ("stage", "wall_s", "move_total_max", "n_starts")}
        for e in out["meta"]["tracking"]
    ]
    row["r_out"] = [[round(float(v), 4) for v in rowv] for rowv in r_out]
    return row


# ---------------------------------------------------------------------------
# units + summary


def build_units(args: argparse.Namespace) -> list[Unit]:
    classes = [c for c in args.classes.split(",") if c]
    arms = [a for a in args.arms.split(",") if a] if args.arms else None
    common = {
        "k_max": args.k_max,
        "max_channels": args.max_channels,
        "snr_db": 0.0,
        "blade_pass": not args.flat_amps,
        "refit_max_iters": args.refit_max_iters,
        "ms_starts": args.ms_starts,
        "ms_screen": args.ms_screen,
        "ms_finish": args.ms_finish,
        "ms_seed": args.ms_seed,
        "ms_bw_rps": args.ms_bw_rps,
        "prep_dir": args.prep_dir or None,
    }
    snrs = [float(s) for s in str(args.snr_db).split(",") if s.strip()]
    units: list[Unit] = []
    for step in [int(s) for s in args.step.split(",")]:
        names = arms or list(STEP4_ARMS if step == 4 else STEP5_ARMS)
        for cls in classes:
            if cls == "synth":
                windows = [
                    {
                        "class": "synth",
                        "seed": s,
                        "seconds": args.synth_seconds,
                        "snr_db": snr,
                        "tag": f"synth{s:02d}_snr{snr:+.0f}",
                    }
                    for snr in snrs
                    for s in range(args.synth_seed0, args.synth_seed0 + args.n_synth)
                ]
            elif cls == "dregon":
                windows = [
                    {"class": "dregon", "window": w, "seconds": args.real_seconds}
                    for w in (DREGON_WINDOWS[: args.n_real] if args.n_real else DREGON_WINDOWS)
                ]
            elif cls == "fly124":
                windows = [
                    {"class": "fly124", "window": w, "seconds": args.real_seconds}
                    for w in FLY124_WINDOWS
                ]
            else:
                raise ValueError(f"unknown window class {cls!r}")
            if step == 5 and cls in ("dregon", "fly124"):
                windows = [w for w in windows if w["window"] in BLIND_REAL]
                if args.n_blind_real:
                    windows = windows[: args.n_blind_real]
            for w in windows:
                tag = w.get("tag") or w.get("window")
                for arm in names:
                    units.append(
                        Unit(
                            uid=f"s{step}__{tag}__{arm}",
                            params={**common, **w, "step": step, "arm": arm},
                        )
                    )
    return units


def build_preps(args: argparse.Namespace) -> Path:
    """Materialize the needed ``beatvk`` prep windows from dload — the cluster path.

    A fresh cluster worktree has no prep cache (it is a gitignored artifact), so
    the windows are rebuilt from the pinned ``beatvk-valid-raw`` dataset with the
    campaign's own builder (``scripts/beatvk_vk_arms.build_preps``), into that
    driver's own cache directory. Returns the directory to read from.
    """
    import beatvk_vk_arms as bva

    root = Path(bva.DEFAULT_OUT)
    jobs: dict[str, list[int]] = {}
    for key in (*DREGON_WINDOWS, *FLY124_WINDOWS):
        rid, _, wtag = key.rpartition("__w")
        jobs.setdefault(rid, []).append(int(wtag))
    bva.build_preps(root, jobs, args.dataset_version, "dload:DREGON")
    return bva.prep_dir(root)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {"n_units": len(rows), "table": {}}
    keyed: dict[tuple, list[dict]] = {}
    for r in rows:
        keyed.setdefault((r["step"], r["group"], r["arm"]), []).append(r)
    for (step, cls, arm), rs in sorted(keyed.items()):
        entry: dict[str, Any] = {
            "n": len(rs),
            "wall_s": round(float(np.mean([r["wall_s"] for r in rs])), 1),
        }
        entry["obj_before"] = round(float(np.mean([r["fvk_before"]["objective"] for r in rs])), 5)
        entry["obj_after"] = round(float(np.mean([r["fvk_after"]["objective"] for r in rs])), 5)
        entry["r2_after"] = round(float(np.mean([r["fvk_after"]["r2"] for r in rs])), 4)
        have_before = [r for r in rs if "err_before" in r]
        have_after = [r for r in rs if "err_after" in r]
        if have_before:
            entry["rms_before"] = round(
                float(np.mean([r["err_before"]["rms"] for r in have_before])), 4
            )
        if have_after:
            entry["rms_after"] = round(
                float(np.mean([r["err_after"]["rms"] for r in have_after])), 4
            )
        vs_ref = [r for r in rs if "err_vs_ref" in r]
        if vs_ref:
            entry["rms_vs_ref"] = round(float(np.mean([r["err_vs_ref"]["rms"] for r in vs_ref])), 4)
        out["table"][f"s{step}/{cls}/{arm}"] = entry
    return out


# ---------------------------------------------------------------------------
# figures


#: Panel titles for the reporting groups.
PANEL_TITLE = {
    "dregon": "DREGON (no truth)",
    "fly124": "FLY124 (stored labels)",
    "synth+0dB": "Synthetic, 0 dB (truth)",
    "synth-10dB": "Synthetic, −10 dB (truth)",
}


def _mean(values: Any) -> float:
    """Mean of a possibly empty array, as a plain float (empty -> NaN)."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def make_figures(out_dir: Path) -> None:
    """The two deck figures, styled like the 2026-08-04 deck's prepare_figs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    INK = "#222222"
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.edgecolor": "#888888",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    rows = [json.loads(p.read_text()) for p in sorted((out_dir / "raw").glob("*.json"))]
    if not rows:
        raise SystemExit(f"no unit JSONs under {out_dir}/raw")

    arm_color = {
        "pikalman": "#1f77b4",
        "iavkf": "#ff7f0e",
        "lbfgs": "#d62728",
        "ours_full": "#1f77b4",
        "seed_only": "#7f7f7f",
        "seed_vk": "#2ca02c",
        "multistart": "#d62728",
    }
    label = {
        "pikalman": "peel↔pi-Kalman\n(ours, 6c)",
        "iavkf": "IAVKF-style\nVK + phase",
        "lbfgs": "L-BFGS on $F_{VK}$",
        "ours_full": "ours: blind full-range",
        "seed_only": "seed only (DP ridge)",
        "seed_vk": "seed + 1 VK refine",
        "multistart": "multi-start L-BFGS",
    }

    def groups(step: int) -> list[str]:
        """Reporting groups present at ``step``, synthetic SNRs first."""
        gs = {r["group"] for r in rows if r["step"] == step}
        return sorted(gs, key=lambda g: (not g.startswith("synth"), g))

    def pick(step, grp, arm, key, sub=None):
        vals = []
        for r in rows:
            if r["step"] == step and r["group"] == grp and r["arm"] == arm and key in r:
                vals.append(r[key][sub] if sub else r[key])
        return np.array(vals, dtype=float)

    def has_truth(grp: str) -> bool:
        return grp.startswith("synth")

    # ---- figure (a): step-4 arm comparison -------------------------------
    s4 = [a for a in STEP4_ARMS if any(r["step"] == 4 and r["arm"] == a for r in rows)]
    g4 = groups(4)
    if s4 and g4:
        fig, axes = plt.subplots(1, len(g4), figsize=(4.6 * len(g4), 4.4), squeeze=False)
        x = np.arange(len(s4))
        w = 0.36
        for ax, grp in zip(axes[0], g4, strict=True):
            truth = has_truth(grp) or grp == "fly124"
            key_b, key_a, sub = (
                ("err_before", "err_after", "rms")
                if truth
                else ("fvk_before", "fvk_after", "objective")
            )
            if grp == "fly124":  # no err_before on real telemetry init: it IS the init
                key_b, key_a, sub = "fvk_before", "fvk_after", "objective"
            b = np.array([_mean(pick(4, grp, a, key_b, sub)) for a in s4])
            aa = np.array([_mean(pick(4, grp, a, key_a, sub)) for a in s4])
            ax.bar(x - w / 2, b, w, color="#cccccc", label="init (telemetry)")
            ax.bar(x + w / 2, aa, w, color=[arm_color[a] for a in s4], label="after arm")
            for i, v in enumerate(aa):
                if np.isfinite(v):
                    ax.text(x[i] + w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel(
                "rms rate error vs truth (rev/s)"
                if truth and sub == "rms"
                else "$F_{VK}$ objective (lower = better)"
            )
            ax.set_title(PANEL_TITLE.get(grp, grp))
            ax.set_xticks(x)
            ax.set_xticklabels([label[a] for a in s4], fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            wall = [_mean(pick(4, grp, a, "wall_s")) for a in s4]
            ax.set_xlabel(
                "wall/window: " + ", ".join(f"{v:.0f}s" for v in wall), fontsize=9, color="#666666"
            )
        axes[0][0].legend(fontsize=9, frameon=False)
        fig.suptitle("Step 4 — improving a telemetry init: three refiners", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_step4_arms.png", dpi=150)
        plt.close(fig)
        print("fig_step4_arms.png: written")

    # ---- figure (b): step-5 Pareto ---------------------------------------
    s5 = [a for a in STEP5_ARMS if any(r["step"] == 5 and r["arm"] == a for r in rows)]
    g5 = groups(5)
    if s5 and g5:
        fig, axes = plt.subplots(1, len(g5), figsize=(4.9 * len(g5), 4.6), squeeze=False)
        for ax, grp in zip(axes[0], g5, strict=True):
            if has_truth(grp):
                ykey, ysub, ylab = "err_after", "rms", "rms rate error vs truth (rev/s)"
            elif grp == "fly124":
                ykey, ysub, ylab = "err_vs_ref", "rms", "rms vs stored labels (rev/s)"
            else:
                ykey, ysub, ylab = "fvk_after", "objective", "$F_{VK}$ objective"
            for a in s5:
                t = pick(5, grp, a, "wall_s")
                y = pick(5, grp, a, ykey, ysub)
                n = min(len(t), len(y))
                if not n:
                    continue
                ax.scatter(t[:n], y[:n], s=24, alpha=0.28, color=arm_color[a], marker="o")
                ax.scatter(
                    [t[:n].mean()],
                    [y[:n].mean()],
                    s=200,
                    color=arm_color[a],
                    marker="o",
                    edgecolor="white",
                    linewidth=1.4,
                    zorder=5,
                    label=label[a],
                )
            ref = pick(5, grp, s5[0], "fvk_before", "objective")
            if not has_truth(grp) and grp != "fly124" and ref.size:
                ax.axhline(
                    float(ref.mean()), color="#444444", linestyle="--", linewidth=1.0, zorder=1
                )
                ax.text(
                    0.02,
                    float(ref.mean()),
                    " telemetry",
                    transform=ax.get_yaxis_transform(),
                    va="bottom",
                    fontsize=9,
                    color="#444444",
                )
            ax.set_xscale("log")
            ax.set_xlabel("wall time per window (s, log)")
            ax.set_ylabel(ylab)
            ax.set_title(PANEL_TITLE.get(grp, grp))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, linestyle=":")
        axes[0][0].legend(fontsize=9, frameon=False, loc="best")
        fig.suptitle("Step 5 — blind annotation: precision against compute", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_step5_pareto.png", dpi=150)
        plt.close(fig)
        print("fig_step5_pareto.png: written")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--step", default="4,5", help="comma list of steps to run (4, 5)")
    ap.add_argument("--classes", default="synth,dregon,fly124")
    ap.add_argument("--arms", default="", help="comma list; default = every arm of the step")
    ap.add_argument("--out", default="results/fvk_arms")
    ap.add_argument("--n-synth", type=int, default=10)
    ap.add_argument("--synth-seed0", type=int, default=0)
    ap.add_argument("--synth-seconds", type=float, default=2.0)
    ap.add_argument("--real-seconds", type=float, default=0.0, help="0 = the whole 16 s window")
    ap.add_argument("--n-real", type=int, default=0, help="cap on DREGON windows (0 = all 9)")
    ap.add_argument(
        "--n-blind-real", type=int, default=0, help="extra cap on the step-5 BLIND_REAL windows"
    )
    ap.add_argument(
        "--snr-db", default="0,-10", help="comma list of synthetic comb-to-noise ratios (dB)"
    )
    ap.add_argument(
        "--flat-amps", action="store_true", help="pure 1/k comb (no 2-blade blade-pass emphasis)"
    )
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--max-channels", type=int, default=2)
    ap.add_argument("--refit-max-iters", type=int, default=6)
    ap.add_argument("--ms-starts", type=int, default=8)
    ap.add_argument("--ms-screen", default="5:15,10:15")
    ap.add_argument("--ms-finish", default="20:20,40:20")
    ap.add_argument("--ms-seed", type=int, default=7)
    ap.add_argument(
        "--ms-bw-rps", type=float, default=6.0, help="wide bw_rps of the multi-start SCREEN rungs"
    )
    ap.add_argument(
        "--prep-dir", default="", help="frozen beatvk prep cache (default: resolve_prep_dir())"
    )
    ap.add_argument(
        "--build-preps",
        action="store_true",
        help="materialize the prep windows from dload first (the cluster path)",
    )
    ap.add_argument("--dataset-version", default=None, help="beatvk-valid-raw version override")
    ap.add_argument("--figs", action="store_true", help="only rebuild the figures")
    add_gridrun_args(ap)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.figs:
        make_figures(out_dir)
        return 0
    if args.build_preps:
        args.prep_dir = str(build_preps(args))
    units = build_units(args)
    print(f"[fvk_arms] {len(units)} units -> {out_dir}", flush=True)
    result = gridrun_from_args(args, units, worker, out_dir, summarize=summarize)
    try:
        make_figures(out_dir)
    except Exception as exc:  # noqa: BLE001 — figures must not fail the grid
        print(f"[fvk_arms] figures skipped: {exc}", flush=True)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
