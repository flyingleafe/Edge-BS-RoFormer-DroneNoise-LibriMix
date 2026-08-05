#!/usr/bin/env python3
"""FLAGSHIP beat-VK method: blind Viterbi init + peeled pi_kalman alternation.

The declared flagship pipeline, end-to-end on the frozen ``beatvk-valid-raw``
protocol:

1. **Blind init** — the ``blind_fullrange`` v2 chain exactly as
   ``scripts/beatvk_vk_arms.py`` composes it (blind_KR seeds + BPF octave
   check + coarse full-range frame-Viterbi + energy-timed takeoff bridge +
   DP trust gates + the vit2dsp ladder, stage_guard on). No neural model, no
   telemetry.
2. **Peeled alternation** — per application: solve the coherent VK envelopes
   at the current track (``vk_tracking.vk_envelopes``, bw 1 Hz, k <= 40),
   give each rotor the audio minus the OTHER rotors' comb reconstructions
   (twin pairs get audio minus the non-pair rotors), then one full
   ``pi_kalman_refine`` pass (protocol settings: ``pair_mode=joint``,
   ``n_iter=3`` internal demod iterations, band 6 Hz, k caps 8/20/40) on the
   peeled residuals. Iterate to plateau (~4 applications).

A ``naive`` arm (plain re-application, no peel) runs for comparison. The
peel is sanity-gated per application: the subtraction must REMOVE energy
(``e_resid_all_ratio < 1``); violations are flagged in the report, never
averaged over silently.

Scoring reuses ``beatvk_eval.score_recording`` (the frozen scorer) on the
assembled per-iteration trajectories, so the leaderboard rows are exactly
``beatvk_eval.py`` numbers: ``init`` (blind_fullrange alone), ``naive`` x
1..A, ``peeled`` x 1..A, pooled per class (dregon_cruise / fly124_cruise /
fly124_warmup / dregon_ramp / dregon_steady / all).

Also dumps explainer-artifact slider traces (``meta/tgrid/gt/snapshots`` +
``arms.{naive,peeled}``, ``meta.init = "blind_fullrange"``) for selected
windows plus the artifact's synthetic case (seed-99 OU-driven 4-rotor comb,
0 dB SNR — the ``trace_pipeline.synth_prep`` configuration).

Run (full protocol, an omnirun uni-cpu job)::

    python scripts/beatvk_flagship.py --jobs 8
    python scripts/beatvk_flagship.py --synthetic-only     # local smoke
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (process-level parallelism instead) —
# the beatvk_vk_arms / vk_blind_sweep convention.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import beatvk_eval  # noqa: E402
import beatvk_vk_arms as vka  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from vk_validation import Prepared, smooth_frames  # noqa: E402

SR: int = beatvk_eval.SR
FRAME_S: float = beatvk_eval.FRAME_S
N_ROTORS: int = beatvk_eval.N_ROTORS

# Protocol pi_kalman settings (the 0.641 row; findings.md "Iterated
# pi_kalman: mechanism findings").
PI_N_ITER = 3
PI_BAND_HZ = 6.0
PI_PAIR_MODE = "joint"
# Bandwidth-and-admission revision rows (docs/experiments/beat-vk.md):
# extra pi_kalman_refine kwargs per --pi-variant. "k_anneal"/"full" thread
# the annealed per-rotor band_b0 across applications via band_b0_final.
PI_VARIANTS: dict[str, dict[str, Any]] = {
    "protocol": {},
    "k_scaled": {"band_mode": "k_scaled"},
    "k_anneal": {"band_mode": "k_scaled", "band_anneal": "posterior"},
    "full": {
        "band_mode": "k_scaled",
        "band_anneal": "posterior",
        "lowk_gate": "consistency",
        "probe_mode": "clean",
    },
}
# Peel settings: 1 Hz envelope bandwidth ~ 1 s coherence, inside the
# measured 0.5-1.5 s tau_k window at k = 8-40.
PEEL_BW_HZ = 1.0
PEEL_K_MAX = 40

DEFAULT_OUT = Path("results/beatvk_flagship")
DEFAULT_APPS = 5
ARMS = ("naive", "peeled")
DEFAULT_TRACES = "FLY124:3,free-flight_nosource_room1:0"
TRACE_GRID_N = 400
SYNTH_SEED = 99


# ---------------------------------------------------------------------------
# peel: VK envelope solve at the current track + per-rotor comb subtraction


def make_peels(
    clip: np.ndarray, r_ft: np.ndarray, ft: np.ndarray, sr: int
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Return ``(peel_audio, pair_audio, diag)``.

    ``peel_audio[i]`` = the audio minus the OTHER rotors' coherent comb
    reconstructions; ``pair_audio[(lo, hi)]`` = the audio minus the NON-pair
    rotors' reconstructions (for the joint twin observations). ``diag``
    carries the energy bookkeeping for the peel sanity gate.
    """
    from tracking.vk_tracking import VKConfig, vk_envelopes, vk_reconstruct

    cfg = VKConfig(fs=float(sr), bw_hz=PEEL_BW_HZ, k_max=PEEL_K_MAX, f_max=6000.0, n_outer=1)
    t_aud = np.arange(clip.shape[-1]) / sr
    r_aud = np.vstack([np.interp(t_aud, ft, r_ft[r]) for r in range(N_ROTORS)])
    env = vk_envelopes(clip, r_aud, cfg)
    n_t = clip.shape[-1]
    recon: dict[int, np.ndarray] = {}
    for rot in range(N_ROTORS):
        x_mask = env.x.copy()
        x_mask[:, env.rotor != rot, :] = 0.0
        recon[rot] = vk_reconstruct(dataclasses.replace(env, x=x_mask), n_samples=n_t)
    e_audio = float(np.mean(clip**2))
    peel_audio: dict[int, np.ndarray] = {}
    diag: dict[str, Any] = {"bw_hz": PEEL_BW_HZ, "e_audio": e_audio, "per_rotor": []}
    for rot in range(N_ROTORS):
        others = sum(recon[j] for j in range(N_ROTORS) if j != rot)
        peeled = clip - others
        peel_audio[rot] = peeled.astype(np.float32)
        diag["per_rotor"].append(
            {
                "rotor": rot,
                "e_removed_frac": round(float(np.mean(others**2)) / e_audio, 5),
                "e_resid_ratio": round(float(np.mean(peeled**2)) / e_audio, 5),
            }
        )
    resid_all = clip - sum(recon[j] for j in range(N_ROTORS))
    diag["e_resid_all_ratio"] = round(float(np.mean(resid_all**2)) / e_audio, 5)
    diag["recon_energy_frac"] = [
        round(float(np.mean(recon[j] ** 2)) / e_audio, 5) for j in range(N_ROTORS)
    ]
    # The gate: a correctly-phased peel removes energy. A residual above the
    # window energy (or a per-rotor residual above it) means the peel is
    # mis-phased and would INJECT interference — flag, never average over.
    diag["energy_ok"] = bool(
        diag["e_resid_all_ratio"] < 1.0 and all(d["e_resid_ratio"] < 1.0 for d in diag["per_rotor"])
    )
    pair_audio: dict[tuple[int, int], np.ndarray] = {}
    for lo in range(N_ROTORS):
        for hi in range(N_ROTORS):
            if lo == hi:
                continue
            nonpair = sum(
                (recon[j] for j in range(N_ROTORS) if j not in (lo, hi)),
                np.zeros_like(clip),
            )
            pair_audio[(lo, hi)] = (clip - nonpair).astype(np.float32)
    return peel_audio, pair_audio, diag


# ---------------------------------------------------------------------------
# peel injection: swap the per-rotor / per-pair audio into the tracker's
# private passes (the tracker has no per-rotor-audio interface; the swap is
# the validated instrumentation of the mechanism study).

_CTX: dict[str, Any] = {"peel_audio": None, "pair_audio": None}
_PATCHED = False


def _install_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import tracking.phase_increment_tracker as pit

    orig_rotor_pass = pit._rotor_pass
    orig_pair_joint_obs = pit._pair_joint_obs

    def rotor_pass_wrap(*args: Any, **kw: Any) -> Any:
        peel = _CTX.get("peel_audio")
        i = args[3]  # rotor index
        if peel is not None and i in peel:
            args = (peel[i],) + tuple(args[1:])
        return orig_rotor_pass(*args, **kw)

    def pair_joint_obs_wrap(*args: Any, **kw: Any) -> Any:
        pair = _CTX.get("pair_audio")
        lo, hi = args[3], args[4]
        if pair is not None and (lo, hi) in pair:
            args = (pair[(lo, hi)],) + tuple(args[1:])
        return orig_pair_joint_obs(*args, **kw)

    pit._rotor_pass = rotor_pass_wrap
    pit._pair_joint_obs = pair_joint_obs_wrap
    _PATCHED = True


def run_arm(
    clip: np.ndarray,
    r0: np.ndarray,
    ft: np.ndarray,
    arm: str,
    n_apps: int,
    tag: str,
    pi_variant: str = "protocol",
    band_b0: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Iterate the pi_kalman stage ``n_apps`` times from ``r0``.

    Returns ``(iters (n_apps+1, 4, N), per-application diagnostics)``.
    ``pi_variant`` selects a :data:`PI_VARIANTS` row; annealed variants
    carry the per-rotor ``band_b0`` across applications. ``band_b0``
    overrides the initial k-scaled band scale (rev/s) of that row.
    """
    from tracking.phase_increment_tracker import pi_kalman_refine

    _install_patches()
    pi_kwargs = dict(PI_VARIANTS[pi_variant])
    if band_b0 is not None:
        pi_kwargs["band_b0"] = float(band_b0)
    iters = [r0.copy()]
    app_diag: list[dict[str, Any]] = []
    r_cur = r0.copy()
    for app in range(1, n_apps + 1):
        peel_diag: dict[str, Any] | None = None
        _CTX["peel_audio"] = _CTX["pair_audio"] = None
        tic = time.perf_counter()
        if arm == "peeled":
            peel_audio, pair_audio, peel_diag = make_peels(clip, r_cur, ft, SR)
            _CTX["peel_audio"] = peel_audio
            _CTX["pair_audio"] = pair_audio
        wall_peel = time.perf_counter() - tic
        tic = time.perf_counter()
        r_next, pi_diag = pi_kalman_refine(
            clip,
            r_cur,
            ft,
            sr=SR,
            n_iter=PI_N_ITER,
            pair_mode=PI_PAIR_MODE,
            band_hz=PI_BAND_HZ,
            **pi_kwargs,
        )
        wall_pi = time.perf_counter() - tic
        _CTX["peel_audio"] = _CTX["pair_audio"] = None
        b0_final = pi_diag.get("band_b0_final")
        if pi_kwargs.get("band_anneal") == "posterior" and b0_final is not None:
            pi_kwargs["band_b0"] = tuple(b0_final)  # trust region carries over
        step = r_next - r_cur
        app_diag.append(
            {
                "app": app,
                "wall_peel_s": round(wall_peel, 1),
                "wall_pi_s": round(wall_pi, 1),
                "step_rms": [
                    round(float(np.sqrt(np.mean(step[r] ** 2))), 4) for r in range(N_ROTORS)
                ],
                "step_mean": [round(float(np.mean(step[r])), 4) for r in range(N_ROTORS)],
                **({"band_b0_final": b0_final} if b0_final is not None else {}),
                **({"peel": peel_diag} if peel_diag is not None else {}),
            }
        )
        iters.append(r_next.copy())
        r_cur = r_next
        print(
            f"  [{tag}/{arm}] app {app}: peel {wall_peel:.0f}s pi {wall_pi:.0f}s"
            + (
                f" resid_all {peel_diag['e_resid_all_ratio']:.3f} ok={peel_diag['energy_ok']}"
                if peel_diag
                else ""
            ),
            flush=True,
        )
    return np.stack(iters), app_diag


# ---------------------------------------------------------------------------
# per-window job (worker process)


def variant_tag(cfg: dict[str, Any]) -> str:
    """Cache-key suffix for the pi_kalman variant + mic subset ('' = protocol
    row on the full array), so rows of the ladder never share a cache entry."""
    pi = str(cfg.get("pi_variant", "protocol"))
    tag = "" if pi == "protocol" else f"__{pi}"
    b0 = cfg.get("band_b0")
    if b0 is not None:
        tag += f"__b0{float(b0):g}"
    return tag + vka.chan_tag(int(cfg.get("channels", 8)), cfg.get("channel_seed"))


def flag_path(out: Path, rid: str, widx: int, arm: str, suffix: str = "") -> Path:
    return out / "runs" / f"{rid}__w{widx:02d}__{arm}{suffix}.npz"


def run_flagship_window(rid: str, widx: int, cfg: dict[str, Any]) -> str:
    out, vk_out = Path(cfg["out"]), Path(cfg["vk_out"])
    n_apps = int(cfg["apps"])
    suffix = variant_tag(cfg)
    arms = [a for a in cfg["arms"] if not flag_path(out, rid, widx, a, suffix).exists()]
    if not arms:
        return "cached"
    channels, channel_seed = int(cfg.get("channels", 8)), cfg.get("channel_seed")
    prep, regime = vka.load_prep(vk_out, rid, widx, channels=channels, channel_seed=channel_seed)
    clip = np.asarray(prep.audio, dtype=np.float64)
    init_path = vka.run_path(
        vk_out,
        rid,
        widx,
        cfg["init_arm"],
        cfg["neural_model"],
        vka.chan_tag(channels, channel_seed),
    )
    with np.load(init_path) as z:
        ft = np.asarray(z["ft"], np.float64)
        r0 = np.asarray(z["traj"], np.float64)
        start_s = float(z["start_s"])
    tag = f"{rid} w{widx:02d}"
    for arm in arms:
        iters, app_diag = run_arm(
            clip,
            r0,
            ft,
            arm,
            n_apps,
            tag,
            pi_variant=cfg.get("pi_variant", "protocol"),
            band_b0=cfg.get("band_b0"),
        )
        path = flag_path(out, rid, widx, arm, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            allow_pickle=False,
            start_s=np.float64(start_s),
            end_s=np.float64(prep.seg_hi),
            regime=np.str_(regime),
            ft=ft,
            iters=iters,
            app_diag=np.str_(json.dumps(app_diag)),
        )
    return "ran"


# ---------------------------------------------------------------------------
# synthetic case — the explainer artifact's seed-99 configuration
# (trace_pipeline.synth_prep, reproduced verbatim)


def synth_prep(seed: int = SYNTH_SEED) -> tuple[Prepared, np.ndarray, dict[str, Any]]:
    from data_processing.rps_synthesis import OUModeParams, RPSSynthConfig
    from data_processing.rps_synthesis import generate as rps_generate

    rng = np.random.default_rng(seed)
    dur = 16.0
    n_t = int(dur * SR)
    t = np.arange(n_t) / SR
    cfg = RPSSynthConfig(
        common=OUModeParams(mean=86.0, std=1.5, tau=0.70),
        roll=OUModeParams(mean=0.0, std=0.70, tau=0.60),
        pitch=OUModeParams(mean=-5.5, std=0.85, tau=0.75),
        yaw=OUModeParams(mean=-2.5, std=1.40, tau=1.00),
    )
    aggressiveness = 1.0
    fs_traj = 250.0
    r_lo = rps_generate(dur, fs_traj, config=cfg, aggressiveness=aggressiveness, rng=rng)
    t_lo = np.arange(r_lo.shape[1]) / fs_traj
    r_true = np.stack([np.interp(t, t_lo, r_lo[i]) for i in range(4)])
    k_max = 30
    psi = rng.uniform(0, 2 * np.pi, (4, k_max))  # locked initial phases
    # 1/k envelope with the 2-blade blade-pass structure (even harmonics
    # 1.6/k, odd 0.5/k) so the BPF octave check sees the calibrated regime.
    comb = np.zeros(n_t)
    for i in range(4):
        phi = 2 * np.pi * np.cumsum(r_true[i]) / SR
        for k in range(1, k_max + 1):
            amp = (1.6 if k % 2 == 0 else 0.5) / k
            comb += amp * np.cos(k * phi + psi[i, k - 1])
    comb_rms = float(np.sqrt(np.mean(comb**2)))
    noise = rng.normal(0.0, comb_rms, n_t)  # 0 dB SNR vs the comb
    x = (comb + noise).astype(np.float64)
    audio = np.stack([x, x])

    ft = np.arange(0.0, n_t / SR - FRAME_S / 2, FRAME_S)
    r_meas = np.stack([np.interp(ft, t, r_true[i]) for i in range(4)])
    edge = (ft > 0.5) & (ft < ft[-1] - 0.5)
    prep = Prepared(
        rid="synthetic",
        tau=0.0,
        seg_lo=0.0,
        seg_hi=dur,
        audio=audio,
        ft=ft,
        r_init=r_meas.copy(),
        r_meas=r_meas,
        r_meas_sm=smooth_frames(r_meas),
        edge=edge,
    )
    weights = np.full((2, N_ROTORS), 0.5)
    meta = {
        "source": "synthetic 4-rotor comb (k=1..30, amp 1/k with 1.6x even-harmonic "
        "blade-pass emphasis, locked phases, white noise at 0 dB SNR vs comb) driven by "
        "the DREGON-calibrated free-flight OU control-mode RPS generator "
        f"(rps_synthesis.generate, aggressiveness {aggressiveness}), seed {seed}",
        "rotor_mean_rev_s": [78.0, 83.0, 89.0, 94.0],
        "ou_modes": {
            "common": {"mean": 86.0, "std": 1.5, "tau": 0.70},
            "roll": {"mean": 0.0, "std": 0.70, "tau": 0.60},
            "pitch": {"mean": -5.5, "std": 0.85, "tau": 0.75},
            "yaw": {"mean": -2.5, "std": 1.40, "tau": 1.00},
        },
        "aggressiveness": aggressiveness,
    }
    return prep, weights, meta


def run_blind_chain(
    prep: Prepared, weights: np.ndarray, init_arm: str = vka.FULLRANGE_ARM
) -> tuple[np.ndarray, dict[str, Any]]:
    """The blind_fullrange chain on an in-memory window (beatvk_vk_arms.run_job
    logic, without the file plumbing): blind_KR seed -> fullrange init ->
    vit2dsp ladder with stage guard."""
    from dataclasses import replace

    from vk_blind_annotation import MIDBAND_CFGS, REFINE_CFG, pit_perm, vit2dsp_pipeline

    from tracking.vk_blind_seeding import blind_seed

    tic = time.perf_counter()
    seed = blind_seed(prep.audio, float(SR), N_ROTORS, vka.SEED_CFG, arms=frozenset({"K", "R"}))
    if init_arm == vka.FULLRANGE_2X_ARM:
        r0, seed, coarse_diag = vka.fullrange_init(
            prep, seed, nfft=2 * vka.COARSE_NFFT, hop=1024, gamma=vka.COARSE_GAMMA / 2.0
        )
    else:
        r0, seed, coarse_diag = vka.fullrange_init(prep, seed)
    wall_seed = time.perf_counter() - tic
    gate = seed.update_gate
    p = pit_perm(r0, prep.r_meas, prep.edge)
    phys_map = np.empty(N_ROTORS, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        phys_map[track_row] = truth_row
    mid_cfg = MIDBAND_CFGS[0] if gate is None else replace(MIDBAND_CFGS[0], update_gate=gate)
    ref_cfg = REFINE_CFG if gate is None else replace(REFINE_CFG, update_gate=gate)
    tic = time.perf_counter()
    stages, _, _, wall_scan, wall_vk = vit2dsp_pipeline(
        prep, r0, weights, phys_map, midband_cfg=mid_cfg, refine_cfg=ref_cfg, stage_guard=True
    )
    info = {
        "seed_bases": [round(float(b), 2) for b in seed.bases],
        "coarse_mode": coarse_diag.get("coarse_mode"),
        "coarse_bridge": coarse_diag.get("coarse_bridge"),
        "wall_seed_s": round(wall_seed, 1),
        "wall_ladder_s": round(wall_scan + wall_vk, 1),
    }
    return stages[-1][1], info


# ---------------------------------------------------------------------------
# scoring + assembly


def pit_align_full(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, list[int]]:
    cost = np.array([[float(np.mean((p - g) ** 2)) for g in gt] for p in pred])
    r_ind, c_ind = linear_sum_assignment(cost)
    perm = np.empty(len(gt), dtype=int)
    for r, c in zip(r_ind, c_ind):
        perm[c] = r
    return pred[perm], [int(x) for x in perm]


def window_score(
    pred_ft: np.ndarray,
    ft_abs: np.ndarray,
    gt_ts: np.ndarray,
    gt_vals: np.ndarray,
    t0: float,
    t1: float,
) -> dict[str, Any]:
    """Protocol-style window PIT-MAE on the 0.032 s grid, with per-rotor split."""
    tg = np.arange(int(np.ceil(t1 / FRAME_S)) + 1, dtype=np.float64) * FRAME_S
    mask = (tg >= t0 - 1e-6) & (tg < t1 - 1e-6)
    tgm = tg[mask]
    pred = np.vstack([np.interp(tgm, ft_abs, pred_ft[r]) for r in range(N_ROTORS)])
    gt = np.vstack([np.interp(tgm, gt_ts, gt_vals[r]) for r in range(N_ROTORS)])
    aligned, perm = pit_align_full(pred, gt)
    err = aligned - gt
    return {
        "mean": round(float(np.mean(np.abs(err))), 4),
        "per_rotor": [round(float(np.mean(np.abs(err[r]))), 4) for r in range(N_ROTORS)],
        "per_rotor_bias": [round(float(np.mean(err[r])), 4) for r in range(N_ROTORS)],
        "perm": perm,
    }


POOLS: dict[str, Any] = {
    "dregon_cruise": lambda r: r["recording"] in beatvk_eval.DREGON_RECS
    and r["regime"] == "cruise",
    "fly124_cruise": lambda r: r["recording"] == beatvk_eval.FLY124_REC and r["regime"] == "cruise",
    "fly124_warmup": lambda r: r["recording"] == beatvk_eval.FLY124_REC and r["regime"] == "warmup",
    "dregon_ramp": lambda r: r["recording"] in beatvk_eval.DREGON_RECS and r["window"] == 0,
    "dregon_steady": lambda r: r["recording"] in beatvk_eval.DREGON_RECS and r["window"] in (1, 2),
    "all": lambda r: True,
}


def pool_means(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, pred in POOLS.items():
        sel = [r["mae"] for r in rows if pred(r)]
        out[name] = round(float(np.mean(sel)), 4) if sel else None
    return out


def score_row(
    recs: list[dict[str, Any]],
    trajs: dict[str, tuple[np.ndarray, np.ndarray]],
    jobs_windows: dict[str, list[int]],
) -> list[dict[str, Any]]:
    """Frozen-scorer rows for one leaderboard row (dict rid -> (ft, rps)).

    Rows are restricted to the windows actually run — windows outside the
    run set would only see edge-clamped trajectory values.
    """
    rows: list[dict[str, Any]] = []
    for rec in recs:
        rid = rec["recording_id"]
        if rid not in trajs:
            continue
        ft, rps = trajs[rid]
        keep = set(jobs_windows.get(rid, []))
        rows.extend(
            r for r in beatvk_eval.score_recording(rec, ft, rps, ["none"]) if r["window"] in keep
        )
    return rows


# ---------------------------------------------------------------------------
# trace JSON output (explainer artifact slider schema)


def to_grid(r_ft: np.ndarray, tgrid: np.ndarray, ft: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 3) for v in np.interp(tgrid, ft, r_ft[r])] for r in range(N_ROTORS)]


def build_trace(
    name: str,
    meta: dict[str, Any],
    ft_abs: np.ndarray,
    t0: float,
    t1: float,
    gt_ts: np.ndarray,
    gt_vals: np.ndarray,
    arm_iters: dict[str, np.ndarray],
    arm_diags: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    tgrid = np.linspace(0.0, t1 - t0, TRACE_GRID_N)
    result: dict[str, Any] = {
        "meta": {
            **meta,
            "name": name,
            "t0_s": t0,
            "t1_s": t1,
            "duration_s": t1 - t0,
            "init": meta.pop("init_arm", "blind_fullrange"),
            "pipeline": (
                "blind_fullrange (beatvk_vk_arms: blind_KR seed + BPF octave check + "
                "coarse full-range Viterbi + energy bridge + DP trust gates + vit2dsp "
                f"ladder) + iterated pi_kalman(pair_mode={PI_PAIR_MODE}, "
                f"n_iter={PI_N_ITER}, band {PI_BAND_HZ:g} Hz); peeled arm subtracts the "
                f"other rotors' VK comb reconstructions (bw {PEEL_BW_HZ:g} Hz) before "
                "each application"
            ),
        },
        "tgrid": [round(float(t), 4) for t in tgrid],
        "gt": [
            [round(float(v), 3) for v in np.interp(tgrid + t0, gt_ts, gt_vals[r])]
            for r in range(N_ROTORS)
        ],
        "arms": {},
    }
    for arm, iters in arm_iters.items():
        snaps: list[dict[str, Any]] = []
        for it in range(iters.shape[0]):
            s = window_score(iters[it], ft_abs, gt_ts, gt_vals, t0, t1)
            if it == 0:
                label = "Iteration 0 — blind Viterbi init"
                desc = (
                    "blind_fullrange chain output (coarse full-range frame-Viterbi + "
                    "BPF octave check + energy bridge + DP trust gates + vit2dsp "
                    "ladder). No neural model, no telemetry."
                )
            else:
                label = f"Iteration {it} — {arm} pi_kalman application {it}"
                desc = (
                    f"Full pi_kalman stage ({PI_N_ITER} internal demod iters, k caps "
                    f"8/20/40) re-applied at iteration {it - 1}'s track"
                    + (
                        ", on the per-rotor peeled audio (other rotors' VK comb "
                        "reconstructions subtracted)."
                        if arm == "peeled"
                        else " (naive re-application)."
                    )
                )
            extras: dict[str, Any] = {
                "pit_mae": {
                    "mean": s["mean"],
                    "per_rotor": s["per_rotor"],
                    "per_rotor_bias": s["per_rotor_bias"],
                    "perm": s["perm"],
                }
            }
            if it > 0:
                d = arm_diags[arm][it - 1]
                extras["step_rms"] = d["step_rms"]
                extras["step_mean"] = d["step_mean"]
                if "peel" in d:
                    extras["peel"] = d["peel"]
            snaps.append(
                {
                    "stage": f"iter_{it}",
                    "label": label,
                    "desc": desc,
                    "tracks": to_grid(iters[it], tgrid + t0, ft_abs),
                    "extras": extras,
                }
            )
        result["arms"][arm] = {"snapshots": snaps}
    # top-level snapshots = the flagship (peeled) arm when present
    lead = "peeled" if "peeled" in result["arms"] else next(iter(result["arms"]))
    result["snapshots"] = result["arms"][lead]["snapshots"]
    return result


# ---------------------------------------------------------------------------


def parse_traces(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        rid, _, widx = item.rpartition(":")
        out.append((rid, int(widx)))
    return out


def short_name(rid: str, widx: int) -> str:
    base = "fly124" if rid == vka.FLY124_REC else "dregon_" + rid.split("_")[1].split("-")[0]
    if rid.startswith("free-flight_"):
        base = "dregon_" + rid.split("_")[1]
    return f"{base}_w{widx:02d}"


def run_synthetic(
    out: Path,
    arms: list[str],
    n_apps: int,
    init_arm: str = vka.FULLRANGE_ARM,
    pi_variant: str = "protocol",
    band_b0: float | None = None,
) -> dict[str, Any]:
    """The synthetic case end-to-end: blind chain + both arms + trace JSON."""
    cache = out / "runs" / "synthetic_chain.npz"
    prep, weights, meta = synth_prep()
    if cache.exists():
        with np.load(cache) as z:
            r0 = np.asarray(z["traj"], np.float64)
            info = json.loads(str(z["info"]))
    else:
        r0, info = run_blind_chain(prep, weights, init_arm)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, allow_pickle=False, traj=r0, info=np.str_(json.dumps(info)))
    print(f"[synthetic] blind chain: {info}", flush=True)
    clip = np.asarray(prep.audio, dtype=np.float64)
    arm_iters: dict[str, np.ndarray] = {}
    arm_diags: dict[str, list[dict[str, Any]]] = {}
    suffix = variant_tag({"pi_variant": pi_variant, "band_b0": band_b0})
    for arm in arms:
        apath = flag_path(out, "synthetic", 0, arm, suffix)
        if apath.exists():
            with np.load(apath) as z:
                arm_iters[arm] = np.asarray(z["iters"], np.float64)
                arm_diags[arm] = json.loads(str(z["app_diag"]))
            continue
        iters, app_diag = run_arm(clip, r0, prep.ft, arm, n_apps, "synthetic", pi_variant, band_b0)
        arm_iters[arm], arm_diags[arm] = iters, app_diag
        apath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            apath,
            allow_pickle=False,
            start_s=np.float64(0.0),
            end_s=np.float64(prep.seg_hi),
            regime=np.str_("synthetic"),
            ft=prep.ft,
            iters=iters,
            app_diag=np.str_(json.dumps(app_diag)),
        )
    meta = {
        **meta,
        "n_channels": int(clip.shape[0]),
        "blind_chain": info,
        "init_arm": init_arm,
    }
    trace = build_trace(
        "synthetic",
        meta,
        prep.ft,
        0.0,
        float(prep.seg_hi),
        prep.ft,
        prep.r_meas,
        arm_iters,
        arm_diags,
    )
    tdir = out / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / f"blind_synthetic{suffix}.json", "w") as f:
        json.dump(trace, f)
    curves = {
        arm: [s["extras"]["pit_mae"]["mean"] for s in trace["arms"][arm]["snapshots"]]
        for arm in arm_iters
    }
    print(f"[synthetic] PIT-MAE curves: {curves}", flush=True)
    return {"blind_chain": info, "curves": curves}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--vk-out", default=None, help="blind_fullrange run dir (default <out>/vk_arms)"
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--apps", type=int, default=DEFAULT_APPS, help="pi_kalman applications")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--recordings", default="", help="comma subset (default: all 4)")
    ap.add_argument("--windows", default="", help="comma window indices (default: all)")
    ap.add_argument("--dataset-version", default=None)
    ap.add_argument("--dregon-dir", default="data/DREGON")
    ap.add_argument("--traces", default=DEFAULT_TRACES, help="'<rid>:<widx>,...' trace windows")
    ap.add_argument(
        "--init-arm",
        default=vka.FULLRANGE_ARM,
        choices=list(vka.FULLRANGE_ARMS),
        help="blind init variant (2xwin = 4096/1024 coarse STFT, gamma halved)",
    )
    ap.add_argument(
        "--pi-variant",
        default="protocol",
        choices=sorted(PI_VARIANTS),
        help="pi_kalman option set (bandwidth-and-admission revision rows)",
    )
    ap.add_argument(
        "--band-b0",
        type=float,
        default=None,
        help="override the k-scaled band scale (rev/s) of --pi-variant (default: its own)",
    )
    ap.add_argument(
        "--channels", type=int, default=8, help="mic channels for init + refinement (<=8)"
    )
    ap.add_argument(
        "--channel-seed",
        type=int,
        default=None,
        help="random per-window mic subset seed (default: first --channels mics)",
    )
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument(
        "--synthetic-only", action="store_true", help="local smoke: synthetic case only"
    )
    opts = ap.parse_args()

    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [a for a in opts.arms.split(",") if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; valid: {list(ARMS)}")

    if opts.synthetic_only:
        run_synthetic(out, arms, opts.apps, opts.init_arm, opts.pi_variant, opts.band_b0)
        return

    vk_out = Path(opts.vk_out) if opts.vk_out else out / "vk_arms"
    wanted = {r for r in opts.recordings.split(",") if r} or None
    manifest = vka.load_manifest(vk_out, wanted, opts.dataset_version)
    version = manifest["dataset_version"]
    print(f"[flagship] {beatvk_eval.DATASET}@{version[:12]}", flush=True)
    widx_filter = {int(v) for v in opts.windows.split(",") if v} or None
    jobs_windows: dict[str, list[int]] = {}
    for rid, rec in manifest["recordings"].items():
        if wanted is not None and rid not in wanted:
            continue
        idxs = [int(w["index"]) for w in rec["windows"]]
        if widx_filter is not None:
            idxs = [i for i in idxs if i in widx_filter]
        if idxs:
            jobs_windows[rid] = idxs

    # ── stage 1: blind_fullrange init on every window (beatvk_vk_arms) ──
    vka.build_preps(vk_out, jobs_windows, opts.dataset_version, opts.dregon_dir)
    cfg1 = {
        "out": str(vk_out),
        "channels": opts.channels,
        "channel_seed": opts.channel_seed,
        "neural_model": vka.DEFAULT_NEURAL_MODEL,
    }
    ctag = vka.chan_tag(opts.channels, opts.channel_seed)
    init_jobs = [
        (rid, widx)
        for rid, ws in jobs_windows.items()
        for widx in ws
        if not vka.run_path(vk_out, rid, widx, opts.init_arm, cfg1["neural_model"], ctag).exists()
    ]
    ctx = multiprocessing.get_context("spawn")
    if init_jobs:
        print(f"[stage 1] {len(init_jobs)} blind_fullrange jobs on {opts.jobs} workers", flush=True)
        with ProcessPoolExecutor(max_workers=opts.jobs, mp_context=ctx) as pool:
            futs = [
                pool.submit(vka.run_job, rid, widx, opts.init_arm, cfg1) for rid, widx in init_jobs
            ]
            for f in futs:
                f.result()
    vka.assemble(vk_out, [opts.init_arm], jobs_windows, cfg1["neural_model"], version, ctag)

    # ── stage 2: iterated arms per window ──
    cfg2 = {
        "out": str(out),
        "vk_out": str(vk_out),
        "apps": opts.apps,
        "arms": arms,
        "neural_model": vka.DEFAULT_NEURAL_MODEL,
        "init_arm": opts.init_arm,
        "pi_variant": opts.pi_variant,
        "channels": opts.channels,
        "channel_seed": opts.channel_seed,
        "band_b0": opts.band_b0,
    }
    vtag = variant_tag(cfg2)
    iter_jobs = [
        (rid, widx)
        for rid, ws in jobs_windows.items()
        for widx in ws
        if any(not flag_path(out, rid, widx, a, vtag).exists() for a in arms)
    ]
    if iter_jobs:
        print(f"[stage 2] {len(iter_jobs)} window jobs on {opts.jobs} workers", flush=True)
        with ProcessPoolExecutor(max_workers=opts.jobs, mp_context=ctx) as pool:
            futs = [pool.submit(run_flagship_window, rid, widx, cfg2) for rid, widx in iter_jobs]
            for f in futs:
                f.result()

    # ── stage 3: assemble + frozen-scorer leaderboard ──
    recs = beatvk_eval.load_recordings(opts.dataset_version, set(jobs_windows), keep_audio=False)
    init_trajs = beatvk_eval.preds_from_npz(vk_out / (opts.init_arm + ctag), list(jobs_windows))

    def assembled(arm: str, app: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        trajs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for rid, ws in jobs_windows.items():
            fts, rps = [], []
            for widx in sorted(ws):
                with np.load(flag_path(out, rid, widx, arm, vtag)) as z:
                    fts.append(float(z["start_s"]) + np.asarray(z["ft"], np.float64))
                    rps.append(np.asarray(z["iters"], np.float64)[app])
            trajs[rid] = (np.concatenate(fts), np.concatenate(rps, axis=1))
        return trajs

    leaderboard: dict[str, Any] = {}
    per_window: dict[str, list[dict[str, Any]]] = {}
    rows0 = score_row(recs, init_trajs, jobs_windows)
    leaderboard["init"] = pool_means(rows0)
    per_window["init"] = rows0
    for arm in arms:
        for app in range(1, opts.apps + 1):
            rows = score_row(recs, assembled(arm, app), jobs_windows)
            leaderboard[f"{arm}_x{app}"] = pool_means(rows)
            per_window[f"{arm}_x{app}"] = rows

    # peel energy flags
    peel_flags: list[dict[str, Any]] = []
    peel_diags: dict[str, Any] = {}
    if "peeled" in arms:
        for rid, ws in jobs_windows.items():
            for widx in sorted(ws):
                with np.load(flag_path(out, rid, widx, "peeled", vtag)) as z:
                    diags = json.loads(str(z["app_diag"]))
                key = f"{rid}__w{widx:02d}"
                peel_diags[key] = [d.get("peel") for d in diags]
                for d in diags:
                    p = d.get("peel")
                    if p is not None and not p["energy_ok"]:
                        peel_flags.append(
                            {
                                "window": key,
                                "app": d["app"],
                                "e_resid_all_ratio": p["e_resid_all_ratio"],
                                "per_rotor": p["per_rotor"],
                            }
                        )

    # plateau: first peeled application within 0.005 rev/s of the pooled-all
    # minimum (the curve flattens; later applications buy nothing).
    plateau = None
    if "peeled" in arms:
        curve = [leaderboard[f"peeled_x{a}"]["all"] for a in range(1, opts.apps + 1)]
        best = min(v for v in curve if v is not None)
        plateau = next(a for a, v in enumerate(curve, 1) if v is not None and v <= best + 0.005)

    hdr = f"{'row':<14}" + "".join(f"{p:>15}" for p in POOLS)
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for row, pools in leaderboard.items():
        cells = "".join(
            f"{pools[p]:>15.3f}" if pools[p] is not None else f"{'—':>15}" for p in POOLS
        )
        print(f"{row:<14}{cells}")
    if plateau is not None:
        print(f"\nplateau (peeled): application {plateau}")
    if peel_flags:
        print(f"PEEL ENERGY FLAGS: {json.dumps(peel_flags, indent=1)}")
    else:
        print("peel energy gate: all windows/applications OK")

    # ── stage 4: traces ──
    trace_files: dict[str, str] = {}
    tdir = out / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    for rid, widx in parse_traces(opts.traces):
        if rid not in jobs_windows or widx not in jobs_windows[rid]:
            print(f"[trace] {rid} w{widx} not in the run set — skipped", flush=True)
            continue
        rec = next(r for r in recs if r["recording_id"] == rid)
        w = next(x for x in rec["windows"] if int(x["index"]) == widx)
        arm_iters: dict[str, np.ndarray] = {}
        arm_diags: dict[str, list[dict[str, Any]]] = {}
        ft_abs = np.array([])
        for arm in arms:
            with np.load(flag_path(out, rid, widx, arm, vtag)) as z:
                ft_abs = float(z["start_s"]) + np.asarray(z["ft"], np.float64)
                arm_iters[arm] = np.asarray(z["iters"], np.float64)
                arm_diags[arm] = json.loads(str(z["app_diag"]))
        ipath = vka.run_path(vk_out, rid, widx, opts.init_arm, cfg1["neural_model"], ctag)
        with np.load(ipath) as z:
            blind_info = {
                "seed_bases": [round(float(v), 2) for v in z["seed_bases"]],
                "coarse_mode": str(z["coarse_mode"]) if "coarse_mode" in z else None,
                "wall_seed_s": round(float(z["wall_seed_s"]), 1),
                "wall_ladder_s": round(float(z["wall_scan_s"] + z["wall_vk_s"]), 1),
            }
        name = short_name(rid, widx)
        meta = {
            "source": f"{beatvk_eval.DATASET}@{version[:12]}",
            "recording_id": rid,
            "window_index": widx,
            "regime": str(w["regime"]),
            "n_channels": opts.channels,
            "blind_chain": blind_info,
            "init_arm": opts.init_arm,
        }
        trace = build_trace(
            name,
            meta,
            ft_abs,
            float(w["start_s"]),
            float(w["end_s"]),
            rec["ts"],
            rec["vals"],
            arm_iters,
            arm_diags,
        )
        fpath = tdir / f"blind_{name}{vtag}.json"
        with open(fpath, "w") as f:
            json.dump(trace, f)
        trace_files[name] = str(fpath)
        print(f"[trace] wrote {fpath}", flush=True)

    synth_summary = None
    if not opts.no_synthetic:
        synth_summary = run_synthetic(
            out, arms, opts.apps, opts.init_arm, opts.pi_variant, opts.band_b0
        )
        stag = variant_tag({"pi_variant": opts.pi_variant, "band_b0": opts.band_b0})
        trace_files["synthetic"] = str(tdir / f"blind_synthetic{stag}.json")

    report = {
        "dataset": {"name": beatvk_eval.DATASET, "version": version},
        "pipeline": {
            "init": opts.init_arm + " (beatvk_vk_arms vit2dsp chain)",
            "pi_kalman": {"n_iter": PI_N_ITER, "band_hz": PI_BAND_HZ, "pair_mode": PI_PAIR_MODE},
            "pi_variant": {
                "name": opts.pi_variant,
                **PI_VARIANTS[opts.pi_variant],
                **({} if opts.band_b0 is None else {"band_b0": opts.band_b0}),
            },
            "peel": {"bw_hz": PEEL_BW_HZ, "k_max": PEEL_K_MAX},
            "channels": opts.channels,
            "channel_seed": opts.channel_seed,
            "apps": opts.apps,
        },
        "leaderboard": leaderboard,
        "plateau_peeled": plateau,
        "peel_flags": peel_flags,
        "peel_diags": peel_diags,
        "per_window": per_window,
        "traces": trace_files,
        "synthetic": synth_summary,
    }
    rpath = out / f"report{vtag}.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[flagship] wrote {rpath}", flush=True)


if __name__ == "__main__":
    main()
