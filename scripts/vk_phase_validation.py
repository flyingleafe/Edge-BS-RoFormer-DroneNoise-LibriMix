#!/usr/bin/env python3
"""Progressive-difficulty ladder: does the VK machinery recover instantaneous PHASE?

Campaign context: on real DREGON/FLY124 audio, ``vk_track`` capture+refine and
stage D (``rps_refinement.refine_coherent``) are inert — they cannot pull
tracks with > 0.3 rev/s error and add no precision on the Viterbi-DP ridge —
yet the synthetic selftests pass. Suspects: (a) per-harmonic phase diffusion
(DREGON telemetry shows +-0.6 rev/s zero-mean fast jitter, smearing harmonic
k's phase by k * jitter), (b) per-harmonic SNR ~0.03 triggering the
Wiener/Fisher shrink in ``vk_track``, (c) room reverberation corrupting phase
slopes. This ladder isolates which rung breaks the method by testing *phase*
recovery (not just frequency) from fully-synthetic to real data:

  S0  pure additive comb (full control): 1/k harmonic amplitudes on a
      DREGON-aggressiveness synthetic RPS at mean 80 rev/s, grid over
      broadband SNR {clean, 10, 0, -10 dB} x harmonic jitter {none,
      coherent (shared shaft OU, the extra bridge arm), perharm (independent
      per-harmonic OU calibrated to +-0.6 rev/s IF spread at the shaft)}.
  S1  the trained noise generator, 1 rotor (dregon-conditioned, learned
      per-drone OU linewidth ON as in ``GeneratedNoisePool``): truth shaft
      phase = the input RPS integral; the generator's own linewidth is
      nuisance and lowers C_k — that is part of the measurement. The
      per-harmonic initial phases psi_k are chosen by this script (passed via
      ``initial_phases``), so they are known exactly; C_k is |.|-invariant to
      them anyway.
  S2  the generator with 4 rotors at the DREGON twin structure
      {74.0, 74.7, 85.3, 86.3} rev/s, 8 mics: PIT-aligned metrics per rotor
      + a twin-capture indicator.
  S3  real single-motor DREGON (``motor_Motor{1-4}_{50..90}``, 8ch @ 16 kHz):
      no GT phase — recovered-IF stability, per-harmonic lock quality of the
      audio against the RECOVERED phase, residual harmonic-energy drop under
      phase-locked comb subtraction, and VK confidence. Init = the nominal
      setpoint from the recording id, and nominal +-1.
  S3b real 4-motor static DREGON (``motor_allMotors_70``): all 4 motors at
      nominal 70 on the ground, same treatment as S3 but with 4 rotor tracks
      (init = the constant nominal per rotor; the S3 finding is true speed
      ~ 0.98x nominal, so inits are nominal and nominal-1). Isolates
      multi-rotor masking from flight. No GT -> no PIT (rotors start
      identical and are reported by index). Identical inits are structurally
      unrefinable by iter_warp (twin rejection excludes every window), so a
      second ``|stag`` cell staggers the init per rotor (S3B_STAGGER) as the
      degeneracy-breaking control.
  S3c real quasi-stationary flight (``hovering_nosource_room2``, command
      telemetry only): init = tau-aligned ``clean_command_spikes``-cleaned
      command on a 16 s mid-recording window (the free-flight prep
      conventions, but the in-flight mask uses command only — this recording
      has no ``motors_measured``). Isolates aero loading + minor motion from
      translation. The S3b->S3c->S4 deltas decompose the S3->S4 collapse
      into masking vs aero-load phase noise vs translation/Doppler.
  S4  one 16 s cruise window of ``free-flight_nosource_room1`` (the
      ``vk_blind_annotation`` prep segment), init = tau-aligned command
      telemetry: same S3-style no-GT metrics — the S3->S4 delta isolates
      reverb + multi-rotor masking. (IF MAE vs ``motors_measured`` is also
      reported since it exists here.)

Methods per run: ``init`` (metrics of the init trajectory as-is — the metric
floor/ceiling), ``stage_d`` (``refine_coherent`` with its defaults),
``iter_warp`` (``tracking.warp_refinement.iter_warp_refine`` —
iterated angular-resampling / generalized-demodulation refinement,
coarse-to-fine in harmonic order; per-round per-order lock diagnostics are
stored in the run NPZ as ``diag``), ``pi_kalman``
(``tracking.phase_increment_tracker.pi_kalman_refine`` — ML
instantaneous frequency from per-harmonic envelope phase increments fused
by an RTS-smoothed random-walk Kalman posterior; per-harmonic diffusion
rates ``q_k`` estimated from the data and stored in the run NPZ ``diag``),
``pi_kalman_joint`` (the same with ``pair_mode="joint"`` — tight twin
pairs' self-collided harmonics contribute two-tone rate observations
instead of being gated away),
``vk_refine`` (``vk_track`` with the
campaign REFINE config), and ``vk_capture_refine`` (annealed CAPTURE then
REFINE — offsets where the refine basin ~bw/(2k) cannot reach). Inits for
S0-S2: truth + {0, 0.3, 1.0, 2.0} rev/s constant offsets (capture from +2.0
only, as specified; the smoke mode also runs it from +1.0). ``--methods``
restricts the per-run method set (e.g. ``--methods init,stage_d,iter_warp``).

Metrics with GT (S0-S2): edge-trimmed IF MAE (rev/s); shaft-phase circular
RMSE after fitting the initial phase phi0 that maximises coherence, plus the
accumulated phase drift in revolutions; per-harmonic coherence
C_k = |mean_t exp(i k (phi_hat - phi))| for k in {1, 2, 5, 10, 20, 40}.
No-GT lock metrics (all stages, so S2->S3 bridges sim->real on the SAME
metric): lock_k = |mean_t z_k| / mean_t |z_k| of the demodulated envelope
z_k = LP[x e^{-i k phi_hat}]; S3/S4 additionally report the harmonic-energy
drop (dB) after subtracting the coupled-VK comb reconstruction at the
recovered trajectory, and the VKResult confidence. Two decoherence
diagnostics ride on the SAME envelopes: (a) an off-comb noise floor —
lock{k}_off re-demodulates at k phi_hat + 2 pi Delta t (Delta = 3 Hz, off
every comb line) and lock{k}_corr = sqrt(max(lock^2 - lock_off^2, 0)) is the
bias-corrected lock; (b) a coherence-time curve — L(T) = mean over disjoint
spans of length T of |mean_span z| / mean|z|, T in {0.125..8} s, stored as
the compact-JSON column ``coh_time`` ({k: {T: L}} per rotor row). L(T) decays
with T at the rate the comb decoheres, so the rung-to-rung curves separate
fast phase noise (short plateaus) from slow drift (long plateaus).

Everything is deterministic (fixed per-cell seeds; the generator renders
under ``torch.manual_seed``). Per-run NPZs with the refined trajectories go
to ``<out>/npz/``; one CSV (``rows.csv``) + ``summary.json`` at ``<out>``.

Run examples::

    python scripts/vk_phase_validation.py --smoke            # S0 sanity cell
    python scripts/vk_phase_validation.py --stages S0 --quick --workers 4
    python scripts/vk_phase_validation.py --stages S1,S2 --gen-ckpt \
        r2://ml-data/artifacts/gen_v1_corrected/checkpoints/best.ckpt
    python scripts/vk_phase_validation.py            # full ladder (hours)
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy import (parallelism is process-level).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Pin this repo's src/ ahead of the .venv's absolute-path editable install
# (same rationale as vk_blind_annotation.py / the vk_blind_sweep post-mortem).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vk_blind_annotation import (  # noqa: E402
    CAPTURE_CFG,
    REFINE_CFG,
)
from vk_blind_annotation import (
    prepare_recording as prepare_flight_recording,
)

from data_processing import rps_synthesis  # noqa: E402
from tracking.phase_increment_tracker import (  # noqa: E402
    DEFAULTS as PI_KALMAN_DEFAULTS,
)
from tracking.phase_increment_tracker import pi_kalman_refine  # noqa: E402
from tracking.rps_refinement import (  # noqa: E402
    RefineConfig,
    compute_logmag,
    estimate_clock_offset,
    refine_coherent,
)
from tracking.vk_tracking import (  # noqa: E402
    VKConfig,
    _demod_tracks_fft,
    demodulate,
    vk_envelopes,
    vk_reconstruct,
    vk_track,
)
from tracking.warp_refinement import DEFAULT_RUNGS as WARP_RUNGS  # noqa: E402
from tracking.warp_refinement import iter_warp_refine  # noqa: E402

SR = 16000
FRAME_HOP_S = 0.032  # evaluation grid (campaign convention)
EDGE_TRIM_S = 0.5  # metric exclusion at segment edges
ENV_TRIM_S = 1.0  # envelope-grid trim for the lock / energy metrics
KS = (1, 2, 5, 10, 20, 40)  # coherence / lock harmonics
F_MAX = 6000.0  # harmonic cap (matches the tracker's f_max)
PHASE_STRIDE = 8  # subsample stride for the audio-rate phase statistics

S0_DUR_S = 20.0
S0_MEAN_RPS = 80.0
S0_K_MAX = 40
S0_SNRS_FULL = ("clean", 10.0, 0.0, -10.0)
S0_SNRS_QUICK = ("clean", 0.0)
S0_JITTERS_FULL = ("none", "coherent", "perharm")
S0_JITTERS_QUICK = ("none", "perharm")
JITTER_SIGMA = 0.6  # rev/s IF spread at the shaft (DREGON telemetry)
JITTER_TAU = 0.016  # OU time constant (scripts/calibrate_rps_jitter.py fit)

GEN_DUR_S = 16.0  # S1/S2 clip length (one 16 s render keeps phase continuous)
S2_MEANS = (74.0, 74.7, 85.3, 86.3)  # DREGON twin structure (rev/s)
DEFAULT_GEN_CKPT = "r2://ml-data/artifacts/gen_v1_corrected/checkpoints/best.ckpt"
GEN_N_HARMONICS = 100  # must match the checkpoint (positional_harmonic_gen_cond_*)

S3_DUR_S = 20.0  # mid-recording segment of each motor run
S3_MOTORS_FULL = tuple(f"motor_Motor{m}_{s}" for m in (1, 2, 3, 4) for s in (50, 60, 70, 80, 90))
S3_MOTORS_QUICK = ("motor_Motor1_50", "motor_Motor1_70", "motor_Motor1_90")
S3_OFFSETS = (0.0, -1.0, 1.0)

S3B_RID = "motor_allMotors_70"  # all 4 motors static at nominal 70
S3B_OFFSETS = (0.0, -1.0)  # true speed ~ 0.98x nominal (S3 finding)
# Degeneracy-breaking control: 4 identical inits are structurally unrefinable
# by iter_warp (every order of every rotor collides with its 3 twins ->
# _order_collides excludes ALL windows; measured no-op). The staggered arm
# spreads the init per rotor so the refiner can engage; at offset -1.0 it
# brackets the observed comb fundamentals (~67.6-69.9 Hz on this recording).
S3B_STAGGER = (-0.6, -0.2, 0.2, 0.6)

S3C_RID = "hovering_nosource_room2"  # flying but quasi-stationary, command-only
S3C_WIN_S = 16.0
DREGON_MIN_RPS = 30.0  # in-flight mask threshold (vk_validation convention)

S4_RID = "free-flight_nosource_room1"
S4_WIN_S = 16.0

# Decoherence diagnostics (computed from the SAME demodulated envelopes as
# lock_k): disjoint-span coherence-time curve L(T), and the off-comb noise
# floor at k phi + 2 pi OFF_COMB_HZ t (3 Hz sits off every comb line for all
# k at the ladder's rotor speeds and off the twin-pair beat spacings).
COH_SPANS_S = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
OFF_COMB_HZ = 3.0

INIT_OFFSETS = (0.0, 0.3, 1.0, 2.0)
CAPTURE_OFFSETS = (2.0,)  # vk_capture_refine runs from these inits only
STAGE_D_CFG = RefineConfig(sample_rate=SR, device="cpu")
# One fixed envelope config for the no-GT metrics (comb fit + demod): full
# k range, the campaign's refine bandwidth / coupling.
EVAL_ENV_CFG = VKConfig(
    fs=float(SR),
    k_min=1,
    k_max=40,
    bw_hz=1.5,
    couple_hz=20.0,
    k_schedule="fixed",
    n_outer=1,
)

ALL_STAGES = ("S0", "S1", "S2", "S3", "S3b", "S3c", "S4")
CSV_FIELDS = (
    "stage",
    "cell",
    "method",
    "init_offset",
    "rotor",
    "gt_kind",
    "wall_s",
    "if_mae",
    "if_bias",
    "phase_rmse_rad",
    "drift_revs",
    "c1",
    "c2",
    "c5",
    "c10",
    "c20",
    "c40",
    *(f"lock{k}" for k in KS),
    *(f"lock{k}_off" for k in KS),
    *(f"lock{k}_corr" for k in KS),
    "harm_drop_db",
    "confidence",
    "twin_capture",
    "coh_time",  # compact JSON {k: {T_s: L}} — keep last (long column)
)


# ---------------------------------------------------------------------------
# Cells and run specs


@dataclass
class Cell:
    """One audio clip + its truth/init trajectories (a grid point of a stage)."""

    stage: str
    cell_id: str
    audio: np.ndarray  # (C, T) float64 at SR
    ft: np.ndarray  # (N,) seconds, clip-relative frame grid
    r_init_base: np.ndarray  # (R, N) offset-0 init (truth / nominal / telemetry)
    r_true_aud: np.ndarray | None  # (R, T) audio-rate GT trajectory (None: S3/S4)
    r_meas_ft: np.ndarray | None = None  # (R, N) measured telemetry (S4 IF MAE only)
    with_energy_drop: bool = False  # compute the comb-subtraction metric (S3/S4)
    pit: bool = False  # PIT-align predictions to truth before scoring (S2)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSpec:
    cell: Cell
    method: str  # init | stage_d | vk_refine | vk_capture_refine
    offset: float
    out_dir: str


# ---------------------------------------------------------------------------
# Synthesis helpers


def _ou_paths(n_paths: int, n: int, dt: float, sigma: float, tau: float, rng) -> np.ndarray:
    """``(n_paths, n)`` stationary OU paths (std ``sigma``, time constant ``tau``)."""
    alpha = 1.0 - dt / tau
    eps = rng.standard_normal((n_paths, n))
    x0 = rng.normal(0.0, sigma, size=(n_paths, 1))
    scale = sigma * np.sqrt(2.0 * dt / tau)
    # x[t] = alpha x[t-1] + scale eps[t], plus the decaying initial condition.
    driven = lfilter([scale], [1.0, -alpha], eps, axis=-1)
    decay = x0 * alpha ** np.arange(n)[None, :]
    return np.asarray(driven) + decay


def _synth_rps(dur_s: float, seed: int, means: tuple[float, ...]) -> np.ndarray:
    """``(R, T)`` audio-rate synthetic RPS: DREGON-calibrated OU modes, re-centred
    per rotor to ``means`` (rps_synthesis fixes its own hover level).

    The OU modes are generated ON the 0.032 s frame grid and linearly
    interpolated to audio rate, so the truth is exactly representable by a
    frame-grid trajectory: a truth init reconstructs the shaft phase perfectly
    (C_k ceiling = 1 for all k). Sub-frame phase diffusion — which no frame-
    grid trajectory can encode — is added *explicitly* by the S0 jitter arms
    (and by the generator's own linewidth in S1/S2), keeping the two effects
    separable. (A 100 Hz control grid instead leaves ~0.01-rev/s interp
    wiggle whose integral destroys C_40 even at truth init — measured.)
    """
    rng = np.random.default_rng(seed)
    fs_ctrl = 1.0 / FRAME_HOP_S
    r_ctrl = rps_synthesis.generate(dur_s, fs_ctrl, aggressiveness=1.0, rng=rng)  # (4, M)
    t_ctrl = np.arange(r_ctrl.shape[-1]) * FRAME_HOP_S
    t_aud = np.arange(int(round(dur_s * SR))) / SR
    rows = []
    for i, mean in enumerate(means):
        row = r_ctrl[i % 4]
        rows.append(np.interp(t_aud, t_ctrl, row - row.mean() + mean))
    return np.stack(rows)


def _frame_grid(n_samples: int) -> np.ndarray:
    return np.arange(0.0, n_samples / SR - FRAME_HOP_S / 2, FRAME_HOP_S)


def _interp_rows(t_dst: np.ndarray, t_src: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_dst, t_src, rows[i]) for i in range(rows.shape[0])])


def build_s0_cell(snr: str | float, jitter: str, seed: int) -> Cell:
    """Pure additive comb: x = sum_k a_k cos(k phi + psi_k [+ d_k]) + broadband."""
    rng = np.random.default_rng(seed)
    r_aud = _synth_rps(S0_DUR_S, seed, (S0_MEAN_RPS,))  # (1, T)
    n_t = r_aud.shape[-1]
    phi = 2.0 * np.pi * np.cumsum(r_aud[0]) / SR
    psi = rng.uniform(0.0, 2.0 * np.pi, S0_K_MAX)
    ks = np.arange(1, S0_K_MAX + 1)
    ks = ks[ks * float(r_aud.max()) <= F_MAX]  # cap at 6 kHz (all pass at ~80 rev/s)

    # Harmonic jitter: d_k adds 2 pi k * integral(delta r_k) to harmonic k.
    delta_r = np.zeros((len(ks), n_t))
    if jitter == "coherent":
        delta_r[:] = _ou_paths(1, n_t, 1.0 / SR, JITTER_SIGMA, JITTER_TAU, rng)
    elif jitter == "perharm":
        delta_r = _ou_paths(len(ks), n_t, 1.0 / SR, JITTER_SIGMA, JITTER_TAU, rng)
    elif jitter != "none":
        raise ValueError(f"unknown jitter mode {jitter!r}")

    sig = np.zeros(n_t)
    for i, k in enumerate(ks):
        d_k = 2.0 * np.pi * float(k) * np.cumsum(delta_r[i]) / SR
        sig += (1.0 / k) * np.cos(k * phi + psi[k - 1] + d_k)
    if snr != "clean":
        p_sig = float(np.mean(sig**2))
        noise = rng.standard_normal(n_t)
        noise *= np.sqrt(p_sig / 10.0 ** (float(snr) / 10.0) / float(np.mean(noise**2)))
        sig = sig + noise

    ft = _frame_grid(n_t)
    t_aud = np.arange(n_t) / SR
    return Cell(
        stage="S0",
        cell_id=f"snr={snr}|jit={jitter}",
        audio=sig[None, :],
        ft=ft,
        r_init_base=_interp_rows(ft, t_aud, r_aud),
        r_true_aud=r_aud,
        meta={"snr": snr, "jitter": jitter, "seed": seed, "n_harmonics": int(len(ks))},
    )


# ---------------------------------------------------------------------------
# Generator rendering (S1 / S2)


def _load_gen_bundle(ckpt: str, device: str):
    from data_processing.generated_noise import _load_generator

    params = {
        "checkpoint": ckpt,
        "sample_rate": SR,
        "n_harmonics": GEN_N_HARMONICS,
        "no_diff_noise": False,
        "model_name": "positional_harmonic_gen",
        "rps_jitter_sigma_init": 0.6,
        "rps_jitter_tau": 0.016,
    }
    return _load_generator(params, device)


def build_gen_cell(
    stage: str,
    means: tuple[float, ...],
    seed: int,
    bundle,
    *,
    dregon_dir: str,
    device: str,
    n_channels: int,
) -> Cell:
    """Render a generator clip and return it with the exact input-RPS truth.

    Truth shaft phase = 2 pi * cumsum(input rps) / SR. The generator's learned
    per-drone OU linewidth is forced ON (as ``GeneratedNoisePool`` renders),
    so its own phase diffusion is part of the measurement; the per-harmonic
    initial phases are drawn here and passed in, hence known.
    """
    import torch

    from data_processing.generated_noise import load_geometry
    from models.generative.codebook import geometry_to_rel_pos

    rng = np.random.default_rng(seed)
    n_rotors = len(means)
    r_aud = _synth_rps(GEN_DUR_S, seed, means)  # (R, T)
    mic_pos, rotor_pos = load_geometry("dregon", dregon_dir)
    rel = geometry_to_rel_pos(
        np.asarray(mic_pos)[:n_channels], np.asarray(rotor_pos)[:n_rotors]
    )  # (M, R, 3)

    torch.manual_seed(seed)  # OU jitter + diff-noise draws inside the model
    ip = rng.uniform(0.0, 2.0 * np.pi, size=(1, n_rotors, GEN_N_HARMONICS))
    sigma = bundle.sigma_map.get("dregon")
    kwargs: dict[str, Any] = {
        "initial_phases": torch.from_numpy(ip).float().to(device),
    }
    if sigma is not None:
        # Force the learned linewidth ON at eval (GeneratedNoisePool convention).
        kwargs["rps_jitter"] = True
        kwargs["rps_jitter_sigma"] = torch.full((1,), float(sigma), device=device)
    with torch.no_grad():
        audio_t = bundle.model(
            torch.from_numpy(r_aud).float().to(device)[None],  # (1, R, T)
            torch.from_numpy(np.asarray(rel, dtype=np.float32)).to(device)[None],
            bundle.z_map["dregon"][None].to(device),
            **kwargs,
        )
    audio = np.asarray(audio_t[0].cpu().numpy(), dtype=np.float64)  # (M, T)

    ft = _frame_grid(r_aud.shape[-1])
    t_aud = np.arange(r_aud.shape[-1]) / SR
    return Cell(
        stage=stage,
        cell_id=f"{stage.lower()}_seed{seed}",
        audio=audio,
        ft=ft,
        r_init_base=_interp_rows(ft, t_aud, r_aud),
        r_true_aud=r_aud,
        pit=(n_rotors > 1),
        meta={
            "seed": seed,
            "means": list(means),
            "gen_sigma": sigma,
            "n_channels": n_channels,
            "initial_phases": "drawn here (known); C_k is |.|-invariant to them",
        },
    )


# ---------------------------------------------------------------------------
# Real-data cells (S3 / S4)


def build_s3_cells(dregon_dir: str, quick: bool) -> list[Cell]:
    from data_processing.sources.dregon import discover_recordings, get_geometry, load_timeframe
    from data_processing.streams import resolve_source

    ddir = Path(resolve_source(dregon_dir))
    by_id = {s["recording_id"]: s for s in discover_recordings(ddir)}
    geometry = get_geometry(ddir)
    rids = S3_MOTORS_QUICK if quick else S3_MOTORS_FULL
    cells = []
    for rid in rids:
        if rid not in by_id:
            print(f"[S3] {rid} not found under {ddir} — skipped", flush=True)
            continue
        sample = by_id[rid]
        frame = load_timeframe(sample, geometry=geometry, target_sr=SR)
        audio = np.asarray(frame["audio"].data, dtype=np.float64)
        n_seg = int(S3_DUR_S * SR)
        lo = max(0, (audio.shape[-1] - n_seg) // 2)  # mid-recording segment
        audio = audio[:, lo : lo + n_seg]
        nominal = float(sample["motor_speed"])
        ft = _frame_grid(audio.shape[-1])
        cells.append(
            Cell(
                stage="S3",
                cell_id=rid,
                audio=audio,
                ft=ft,
                r_init_base=np.full((1, len(ft)), nominal),
                r_true_aud=None,
                with_energy_drop=True,
                meta={"nominal": nominal, "motor": sample["motor_id"], "seg_lo_s": lo / SR},
            )
        )
    return cells


def build_s3b_cells(dregon_dir: str) -> list[Cell]:
    """All 4 motors static at nominal 70 (``motor_allMotors_70``) — S3 treatment
    with 4 rotor tracks. No GT -> no PIT; rotors by index. Two cells: the
    spec'd degenerate init (all rotors at the same constant — twin rejection
    makes iter_warp a measured no-op there) and the ``|stag`` control with
    per-rotor S3B_STAGGER added so refinement can engage."""
    from data_processing.sources.dregon import discover_recordings, get_geometry, load_timeframe
    from data_processing.streams import resolve_source

    ddir = Path(resolve_source(dregon_dir))
    by_id = {s["recording_id"]: s for s in discover_recordings(ddir)}
    sample = by_id[S3B_RID]
    frame = load_timeframe(sample, geometry=get_geometry(ddir), target_sr=SR)
    audio = np.asarray(frame["audio"].data, dtype=np.float64)
    n_seg = int(S3_DUR_S * SR)
    lo = max(0, (audio.shape[-1] - n_seg) // 2)  # mid-recording segment
    audio = audio[:, lo : lo + n_seg]
    nominal = float(sample["motor_speed"])
    ft = _frame_grid(audio.shape[-1])
    flat = np.full((4, len(ft)), nominal)
    stag = flat + np.asarray(S3B_STAGGER)[:, None]
    cells = []
    for suffix, base in (("", flat), ("|stag", stag)):
        cells.append(
            Cell(
                stage="S3b",
                cell_id=S3B_RID + suffix,
                audio=audio,
                ft=ft,
                r_init_base=base,
                r_true_aud=None,
                with_energy_drop=True,
                meta={"nominal": nominal, "seg_lo_s": lo / SR, "stagger": bool(suffix)},
            )
        )
    return cells


def build_s3c_cell(dregon_dir: str) -> Cell:
    """Quasi-stationary flight (``hovering_nosource_room2``), command-only.

    Free-flight prep conventions (tau via stage-A ``estimate_clock_offset``,
    ``clean_command_spikes``-cleaned command interpolated to the frame grid)
    but the in-flight mask uses the command alone — this recording carries no
    ``motors_measured``. One 16 s mid-recording window of the in-flight span.
    """
    from data_processing.sources.dregon import (
        clean_command_spikes,
        discover_recordings,
        get_geometry,
        load_timeframe,
    )
    from data_processing.streams import resolve_source

    ddir = Path(resolve_source(dregon_dir))
    by_id = {s["recording_id"]: s for s in discover_recordings(ddir)}
    frame = load_timeframe(by_id[S3C_RID], geometry=get_geometry(ddir), target_sr=SR)
    audio = np.asarray(frame["audio"].data, dtype=np.float64)
    t0 = float(frame["audio"].tindex.t_start)
    command = np.asarray(frame["motors_command"].data)
    mt = np.asarray(frame["motors_command"].tindex.abs_stamps) - t0
    command_clean = clean_command_spikes(command)

    idx = np.where(np.median(command, axis=0) > DREGON_MIN_RPS)[0]
    t_lo = float(mt[idx[0]]) + 0.2
    t_hi = float(mt[idx[-1]]) - 0.2

    # Stage-A clock offset from the cleaned command, channel 0, first 30 s of
    # the in-flight window — identical to vk_validation.prepare_recording.
    cfg_r = RefineConfig()
    spec0 = compute_logmag(audio[:1, int(t_lo * SR) : int(min(t_lo + 30.0, t_hi) * SR)], cfg_r)
    tau, _, _ = estimate_clock_offset(spec0, mt - t_lo, command_clean, cfg_r)

    mid = 0.5 * (t_lo + t_hi)
    seg_lo = max(t_lo, mid - S3C_WIN_S / 2.0)
    seg_hi = min(t_hi, seg_lo + S3C_WIN_S)
    a0, a1 = int(round(seg_lo * SR)), int(round(seg_hi * SR))
    seg = audio[:, a0:a1]
    ft = _frame_grid(seg.shape[-1])
    r_init = np.stack([np.interp(ft + seg_lo + tau, mt, command_clean[i]) for i in range(4)])
    return Cell(
        stage="S3c",
        cell_id=f"{S3C_RID}@{seg_lo:.1f}s",
        audio=seg,
        ft=ft,
        r_init_base=r_init,
        r_true_aud=None,
        with_energy_drop=True,
        meta={
            "window_start_s": float(seg_lo),
            "tau": float(tau),
            "mean_command_rps": float(r_init.mean()),
        },
    )


def build_s4_cell(dregon_dir: str = "data/DREGON") -> Cell:
    """One 16 s cruise window of the vk_blind_annotation prep segment,
    init = tau-aligned cleaned command telemetry (the campaign's init)."""
    try:
        prep = prepare_flight_recording(S4_RID)
    except KeyError:
        # No local data/DREGON checkout (cluster): bypass the annotation
        # prep cache and load via the dload-capable validation loader.
        from vk_validation import prepare_recording as _prep_dir

        prep = _prep_dir(S4_RID, dregon_dir=dregon_dir)
    start = max(0.0, (float(prep.ft[-1]) - S4_WIN_S) / 2.0)
    a0 = int(round(start * SR))
    audio = prep.audio[:, a0 : a0 + int(S4_WIN_S * SR)].astype(np.float64)
    ft = _frame_grid(audio.shape[-1])
    r_init = _interp_rows(ft + start, prep.ft, prep.r_init)
    r_meas = _interp_rows(ft + start, prep.ft, prep.r_meas)
    mean_rps = float(r_meas.mean())
    if mean_rps < 45.0:
        raise RuntimeError(f"S4 window mean measured RPS {mean_rps:.1f} < 45 (not cruise)")
    return Cell(
        stage="S4",
        cell_id=f"{S4_RID}@{start:.1f}s",
        audio=audio,
        ft=ft,
        r_init_base=r_init,
        r_true_aud=None,
        r_meas_ft=r_meas,
        with_energy_drop=True,
        meta={"window_start_s": start, "mean_measured_rps": mean_rps, "tau": prep.tau},
    )


# ---------------------------------------------------------------------------
# Metrics


def _phase_stats(r_hat_ft: np.ndarray, ft: np.ndarray, r_true_aud: np.ndarray) -> dict[str, float]:
    """Shaft-phase metrics of ONE rotor: circular RMSE (phi0 fitted), drift, C_k."""
    n_t = len(r_true_aud)
    t_aud = np.arange(n_t) / SR
    r_hat_aud = np.interp(t_aud, ft, r_hat_ft)
    # phi_hat - phi_true, integrated as one cumsum of the IF error (float64).
    dphi = 2.0 * np.pi * np.cumsum(r_hat_aud - r_true_aud) / SR
    lo, hi = int(EDGE_TRIM_S * SR), n_t - int(EDGE_TRIM_S * SR)
    d = dphi[lo:hi:PHASE_STRIDE]
    out: dict[str, float] = {}
    for k in KS:
        out[f"c{k}"] = float(np.abs(np.mean(np.exp(1j * k * d))))
    z1 = np.mean(np.exp(1j * d))
    resid = np.angle(np.exp(1j * (d - np.angle(z1))))
    out["phase_rmse_rad"] = float(np.sqrt(np.mean(resid**2)))
    out["drift_revs"] = float((d[-1] - d[0]) / (2.0 * np.pi))
    return out


def _lock_stats(cell: Cell, r_hat_ft: np.ndarray) -> list[dict[str, Any]]:
    """Per-rotor lock quality of the audio against the recovered phase.

    lock_k = |mean_t z| / mean_t |z| of the demodulated, decimated envelope
    z = LP[x e^{-i k phi_hat}] (channel-averaged, envelope-grid edges trimmed).
    Rides two decoherence diagnostics on the same envelopes:

    * lock{k}_off — the same statistic demodulated at k phi_hat +
      2 pi OFF_COMB_HZ t (off every comb line): the finite-length noise
      floor of lock. lock{k}_corr = sqrt(max(lock^2 - lock_off^2, 0)).
    * ``coh_time`` (compact JSON {k: {T_s: L}}) — L(T) = mean over disjoint
      spans of length T of |mean_span z| / mean|z|, T in COH_SPANS_S; the
      span-length lock curve whose decay rate is the comb's coherence time.
    """
    n_t = cell.audio.shape[-1]
    t_aud = np.arange(n_t) / SR
    phases: list[np.ndarray] = []
    index: list[tuple[int, int, bool]] = []  # (rotor, k, off_comb) per track row
    for i in range(r_hat_ft.shape[0]):
        r_aud = np.interp(t_aud, cell.ft, r_hat_ft[i])
        phi = 2.0 * np.pi * np.cumsum(r_aud) / SR
        mean_r = float(r_aud.mean())
        for k in KS:
            if k * mean_r <= F_MAX:
                phases.append(k * phi)
                index.append((i, k, False))
                phases.append(k * phi + 2.0 * np.pi * OFF_COMB_HZ * t_aud)
                index.append((i, k, True))
    z = demodulate(cell.audio, np.stack(phases), EVAL_ENV_CFG)  # (C, M, T_env)
    stride = max(1, int(round(SR / EVAL_ENV_CFG.fs_env)))
    fs_env = SR / stride
    n_trim = int(round(ENV_TRIM_S * SR / stride))
    zt = z[..., n_trim : z.shape[-1] - n_trim]
    denom = np.maximum(np.abs(zt).mean(axis=-1), 1e-30)  # (C, M)
    lock_c = (np.abs(zt.mean(axis=-1)) / denom).mean(axis=0)  # (M,) channel average
    n_rotors = r_hat_ft.shape[0]
    rows: list[dict[str, Any]] = [{} for _ in range(n_rotors)]
    coh: list[dict[str, dict[str, float]]] = [{} for _ in range(n_rotors)]
    n_env = zt.shape[-1]
    for m, (i, k, off_comb) in enumerate(index):
        if off_comb:
            rows[i][f"lock{k}_off"] = float(lock_c[m])
            continue
        rows[i][f"lock{k}"] = float(lock_c[m])
        curve: dict[str, float] = {}
        for span_s in COH_SPANS_S:
            n_span = int(round(span_s * fs_env))
            n_spans = n_env // n_span
            if n_span < 1 or n_spans < 1:
                continue  # span longer than the trimmed envelope
            zs = zt[:, m, : n_spans * n_span].reshape(zt.shape[0], n_spans, n_span)
            span_lock = np.abs(zs.mean(axis=-1)) / denom[:, m][:, None]  # (C, n_spans)
            curve[f"{span_s:g}"] = round(float(span_lock.mean()), 6)
        coh[i][str(k)] = curve
    for i in range(n_rotors):
        for k in KS:
            l_on, l_off = rows[i].get(f"lock{k}"), rows[i].get(f"lock{k}_off")
            if l_on is not None and l_off is not None:
                rows[i][f"lock{k}_corr"] = float(np.sqrt(max(l_on**2 - l_off**2, 0.0)))
        rows[i]["coh_time"] = json.dumps(coh[i], separators=(",", ":"))
    return rows


def _energy_drop(cell: Cell, r_hat_ft: np.ndarray) -> float:
    """Harmonic-energy drop (dB) under phase-locked comb subtraction.

    Fits the coupled-VK envelopes at the recovered trajectory (k 1..40),
    reconstructs and subtracts the comb, then compares in-band demodulated
    energy of the original vs the residual over the same tracks.
    """
    n_t = cell.audio.shape[-1]
    t_aud = np.arange(n_t) / SR
    r_aud = _interp_rows(t_aud, cell.ft, r_hat_ft)
    env = vk_envelopes(cell.audio, r_aud, EVAL_ENV_CFG)
    recon = vk_reconstruct(env, n_samples=n_t)
    z_res = _demod_tracks_fft(cell.audio - recon, env.phase, env.rotor, env.k, EVAL_ENV_CFG)
    stride = max(1, int(round(SR / env.fs_env)))
    n_trim = int(round(ENV_TRIM_S * SR / stride))
    sl = slice(n_trim, env.z.shape[-1] - n_trim)
    v = env.valid[None, :, sl].astype(np.float64)
    e_orig = float(np.sum(np.abs(env.z[..., sl]) ** 2 * v))
    e_res = float(np.sum(np.abs(z_res[..., sl]) ** 2 * v))
    return 10.0 * np.log10(max(e_orig, 1e-30) / max(e_res, 1e-30))


def _pit_align(r_hat: np.ndarray, r_true_ft: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Permute prediction rows to minimise pooled MAE vs truth (S2 has no identity)."""
    n = r_hat.shape[0]
    best = min(
        itertools.permutations(range(n)),
        key=lambda p: float(np.mean(np.abs(r_hat[list(p)][:, edge] - r_true_ft[:, edge]))),
    )
    return r_hat[list(best)]


# ---------------------------------------------------------------------------
# One run


def _run_method(cell: Cell, method: str, r0: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Execute one method from init ``r0``; returns (trajectory, confidence, extras)."""
    if method == "init":
        return r0.copy(), float("nan"), {}
    if method == "stage_d":
        return refine_coherent(cell.audio, r0, cell.ft, STAGE_D_CFG), float("nan"), {}
    if method == "iter_warp":
        r_hat, diag = iter_warp_refine(cell.audio, r0, cell.ft, sr=SR)
        return r_hat, float("nan"), diag
    if method == "pi_kalman":
        r_hat, diag = pi_kalman_refine(cell.audio, r0, cell.ft, sr=SR)
        return r_hat, float("nan"), diag
    if method == "pi_kalman_joint":
        r_hat, diag = pi_kalman_refine(cell.audio, r0, cell.ft, sr=SR, pair_mode="joint")
        return r_hat, float("nan"), diag
    if method == "vk_refine":
        res = vk_track(cell.audio, r0, cell.ft, REFINE_CFG)
        return res.r_refined, float(np.mean(res.confidence)), {}
    if method == "vk_capture_refine":
        cap = vk_track(cell.audio, r0, cell.ft, CAPTURE_CFG)
        res = vk_track(cell.audio, cap.r_refined, cell.ft, REFINE_CFG)
        return res.r_refined, float(np.mean(res.confidence)), {}
    raise ValueError(f"unknown method {method!r}")


def run_one(spec: RunSpec) -> list[dict[str, Any]]:
    """Worker: one (cell, method, init offset) -> per-rotor metric rows + NPZ."""
    cell = spec.cell
    r0 = cell.r_init_base + spec.offset
    tic = time.perf_counter()
    r_hat, conf, extras = _run_method(cell, spec.method, r0)
    wall = time.perf_counter() - tic

    ft = cell.ft
    edge = (ft > EDGE_TRIM_S) & (ft < ft[-1] - EDGE_TRIM_S)
    n_rotors = r_hat.shape[0]
    gt_kind = (
        "synthetic"
        if cell.r_true_aud is not None
        else ("measured" if cell.r_meas_ft is not None else "none")
    )

    r_ref_ft: np.ndarray | None = None  # frame-grid reference for IF MAE
    if cell.r_true_aud is not None:
        t_aud = np.arange(cell.r_true_aud.shape[-1]) / SR
        r_ref_ft = _interp_rows(ft, t_aud, cell.r_true_aud)
        if cell.pit:
            r_hat = _pit_align(r_hat, r_ref_ft, edge)
    elif cell.r_meas_ft is not None:
        r_ref_ft = cell.r_meas_ft

    lock_rows = _lock_stats(cell, r_hat)
    drop_db = _energy_drop(cell, r_hat) if cell.with_energy_drop else float("nan")

    rows: list[dict[str, Any]] = []
    for i in range(n_rotors):
        row: dict[str, Any] = dict.fromkeys(CSV_FIELDS, float("nan"))
        row.update(
            stage=cell.stage,
            cell=cell.cell_id,
            method=spec.method,
            init_offset=spec.offset,
            rotor=i,
            gt_kind=gt_kind,
            wall_s=round(wall, 1),
            confidence=conf,
            harm_drop_db=drop_db,
        )
        if r_ref_ft is not None:
            d = (r_hat[i] - r_ref_ft[i])[edge]
            row["if_mae"] = float(np.mean(np.abs(d)))
            row["if_bias"] = float(np.mean(d))
        if cell.r_true_aud is not None:
            row.update(_phase_stats(r_hat[i], ft, cell.r_true_aud[i]))
            if cell.pit:
                assert r_ref_ft is not None
                maes = [
                    float(np.mean(np.abs(r_hat[i] - r_ref_ft[j])[edge])) for j in range(n_rotors)
                ]
                row["twin_capture"] = int(int(np.argmin(maes)) != i)
        row.update(lock_rows[i])
        rows.append(row)

    tag = f"{cell.stage}_{cell.cell_id}__{spec.method}@{spec.offset:+.1f}".replace("|", "_")
    npz_dir = Path(spec.out_dir) / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_dir / f"{tag}.npz",
        ft=ft,
        r_init=r0,
        r_hat=r_hat,
        r_ref_ft=r_ref_ft if r_ref_ft is not None else np.zeros(0),
        diag=json.dumps(extras),
    )

    lead = rows[0]
    bits = [f"wall {wall:.0f}s"]
    if np.isfinite(lead["if_mae"]):
        pooled = float(np.mean([r["if_mae"] for r in rows]))
        bits.append(f"if_mae {pooled:.3f}")
    if np.isfinite(lead.get("c40", float("nan"))):
        bits.append(f"C1 {lead['c1']:.3f} C40 {lead['c40']:.3f}")
    lk = [float(r["lock10"]) for r in rows if np.isfinite(r.get("lock10", float("nan")))]
    if lk:
        bits.append(f"lock10 {float(np.mean(lk)):.3f}")
    if np.isfinite(drop_db):
        bits.append(f"drop {drop_db:.1f}dB")
    print(
        f"[{cell.stage} {cell.cell_id}] {spec.method}@{spec.offset:+.1f}: " + "  ".join(bits),
        flush=True,
    )
    return rows


# ---------------------------------------------------------------------------
# Grid assembly


def methods_for(offset: float, smoke: bool) -> list[str]:
    base = ["init", "stage_d", "iter_warp", "pi_kalman", "pi_kalman_joint", "vk_refine"]
    if smoke:  # smoke also probes the annealed capture at +1.0 (basin check)
        return base + (["vk_capture_refine"] if offset > 0 else [])
    return base + (["vk_capture_refine"] if offset in CAPTURE_OFFSETS else [])


def build_specs(opts: argparse.Namespace, out_dir: Path) -> list[RunSpec]:
    stages = [s.strip() for s in opts.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; expected subset of {ALL_STAGES}")
    specs: list[RunSpec] = []
    allowed = {m.strip() for m in opts.methods.split(",") if m.strip()} if opts.methods else None

    def add(cell: Cell, offsets: tuple[float, ...], smoke: bool = False) -> None:
        for off in offsets:
            for method in methods_for(off, smoke):
                if allowed is not None and method not in allowed:
                    continue
                specs.append(RunSpec(cell=cell, method=method, offset=off, out_dir=str(out_dir)))

    if opts.smoke:
        cell = build_s0_cell("clean", "none", seed=1000)
        add(cell, (0.0, 1.0), smoke=True)
        return specs

    if "S0" in stages:
        snrs = S0_SNRS_QUICK if opts.quick else S0_SNRS_FULL
        jits = S0_JITTERS_QUICK if opts.quick else S0_JITTERS_FULL
        offs = (0.0, 1.0, 2.0) if opts.quick else INIT_OFFSETS
        for i, (snr, jit) in enumerate(itertools.product(snrs, jits)):
            add(build_s0_cell(snr, jit, seed=1000 + i), offs)

    if "S1" in stages or "S2" in stages:
        bundle = _load_gen_bundle(opts.gen_ckpt, opts.device)
        seeds = (0,) if opts.quick else (0, 1)
        if "S1" in stages:
            for s in seeds:
                cell = build_gen_cell(
                    "S1",
                    (S0_MEAN_RPS,),
                    2000 + s,
                    bundle,
                    dregon_dir=opts.dregon_dir,
                    device=opts.device,
                    n_channels=1,
                )
                add(cell, (0.0, 1.0, 2.0) if opts.quick else INIT_OFFSETS)
        if "S2" in stages:
            for s in seeds:
                cell = build_gen_cell(
                    "S2",
                    S2_MEANS,
                    3000 + s,
                    bundle,
                    dregon_dir=opts.dregon_dir,
                    device=opts.device,
                    n_channels=8,
                )
                add(cell, (0.0, 1.0) if opts.quick else INIT_OFFSETS)

    if "S3" in stages:
        keep = {r.strip() for r in opts.s3_recordings.split(",") if r.strip()}
        for cell in build_s3_cells(opts.dregon_dir, opts.quick):
            if keep and cell.cell_id not in keep:
                continue
            add(cell, S3_OFFSETS)

    if "S3b" in stages:
        for cell in build_s3b_cells(opts.dregon_dir):
            add(cell, S3B_OFFSETS)

    if "S3c" in stages:
        add(build_s3c_cell(opts.dregon_dir), (0.0,))

    if "S4" in stages:
        add(build_s4_cell(opts.dregon_dir), (0.0,))

    return specs


# ---------------------------------------------------------------------------
# Reporting


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6) if np.isfinite(v) else ""
    return v


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(
        rows, key=lambda r: (r["stage"], r["cell"], r["method"], r["init_offset"], r["rotor"])
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r[k]) for k in CSV_FIELDS})


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled-over-rotors means per (stage, cell, method, init offset)."""
    numeric = [
        f
        for f in CSV_FIELDS
        if f not in ("stage", "cell", "method", "init_offset", "rotor", "gt_kind", "coh_time")
    ]
    groups: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["stage"], r["cell"], r["method"], r["init_offset"]), []).append(r)
    summary: dict[str, Any] = {}
    for (stage, cell, method, off), grp in sorted(groups.items()):
        entry: dict[str, Any] = {"n_rotors": len(grp), "gt_kind": grp[0]["gt_kind"]}
        for fld in numeric:
            vals = [float(g[fld]) for g in grp if np.isfinite(g.get(fld, float("nan")))]
            if vals:
                entry[fld] = round(float(np.mean(vals)), 6)
        if any(np.isfinite(g.get("twin_capture", float("nan"))) for g in grp):
            entry["n_twin_captured"] = int(
                sum(int(g["twin_capture"]) for g in grp if np.isfinite(g["twin_capture"]))
            )
        summary.setdefault(stage, {}).setdefault(cell, {})[f"{method}@{off:+.1f}"] = entry
    return summary


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--stages", default=",".join(ALL_STAGES), help="comma-separated stage subset")
    ap.add_argument("--out", default="results/vk_phase_validation", help="output directory")
    ap.add_argument("--quick", action="store_true", help="reduced grid per stage")
    ap.add_argument(
        "--methods",
        default="",
        help="comma-separated method filter (default: all methods per rung)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="S0 clean/no-jitter cell only, inits {truth, truth+1.0}, all methods",
    )
    ap.add_argument(
        "--gen-ckpt",
        default=DEFAULT_GEN_CKPT,
        help="S1/S2 generator checkpoint (local path or r2:// URI); the corrected-"
        "geometry dregon-conditioned gen_v1_corrected by default",
    )
    ap.add_argument("--device", default="cpu", help="torch device for the generator render")
    ap.add_argument("--dregon-dir", default="data/DREGON", help="path or dload:DREGON")
    ap.add_argument(
        "--s3-recordings",
        default="",
        help="comma-separated S3 recording-id filter (default: the full/quick S3 set)",
    )
    ap.add_argument("--workers", type=int, default=4, help="parallel worker processes")
    opts = ap.parse_args()

    out_dir = Path(opts.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    specs = build_specs(opts, out_dir)
    print(
        f"[vk_phase_validation] {len(specs)} runs "
        f"(stages={opts.stages}{' smoke' if opts.smoke else ''}"
        f"{' quick' if opts.quick else ''}), workers={opts.workers}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    if opts.workers <= 1:
        for spec in specs:
            rows.extend(run_one(spec))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=opts.workers, mp_context=ctx) as pool:
            for res in pool.map(run_one, specs):
                rows.extend(res)

    write_rows_csv(rows, out_dir / "rows.csv")
    summary = {
        "args": {k: v for k, v in vars(opts).items()},
        "configs": {
            "refine": asdict(REFINE_CFG),
            "capture": asdict(CAPTURE_CFG),
            "eval_env": asdict(EVAL_ENV_CFG),
            "stage_d": asdict(STAGE_D_CFG),
            "iter_warp": {"rungs": [asdict(g) for g in WARP_RUNGS]},
            "pi_kalman": dict(PI_KALMAN_DEFAULTS),
        },
        "grid": {
            "ks": list(KS),
            "init_offsets": list(INIT_OFFSETS),
            "jitter": {"sigma_revs": JITTER_SIGMA, "tau_s": JITTER_TAU},
            "s2_means": list(S2_MEANS),
            "coh_spans_s": list(COH_SPANS_S),
            "off_comb_hz": OFF_COMB_HZ,
            "s3b_offsets": list(S3B_OFFSETS),
        },
        "wall_s": round(time.perf_counter() - tic, 1),
        "results": summarize(rows),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"done: {len(rows)} rows -> {out_dir}/rows.csv + summary.json "
        f"({summary['wall_s']:.0f}s total)",
        flush=True,
    )


if __name__ == "__main__":
    main()
