#!/usr/bin/env python3
"""Fast CPU lab for iterating on the blind-VK RPS refinement chain (WP0 of
docs/experiments/rps-refine-precision.md).

Repo-ified from the scratchpad trace_pipeline.py: runs the fullrange-v2 blind
chain (blind_seed -> BPF octave / coarse full-range Viterbi / trust gates +
energy bridge [= beatvk_vk_arms.fullrange_init] -> vit2dsp ladder [viterbi_c
-> spatial joint DP -> optional midband VK capture -> optional refine VK] ->
pi_kalman) on named 16 s windows, with configurable stage chains and a
synthetic free-flight battery, reporting per-stage PIT-MAE plus per-rotor
bias/shape decomposition.

Windows:
  dregon_ramp    free-flight_nosource_room1 w00 (beatvk prep cache)
  fly124_cruise  FLY124 w03 (beatvk prep cache)
  synth          battery of --synth-battery N windows (OU free-flight RPS via
                 rps_synthesis.generate + locked-phase harmonic comb at 0 dB
                 SNR, exact GT), varying seed and aggressiveness {0.7,1.0,1.4},
                 rotor means kept in ~[70, 100] rev/s
  synthNN        a single battery window (e.g. synth03) for chunked runs
  synth_trace    the WP1-WP3 trace/probe "synthetic" window (seed 99, fixed
                 mode means -> rotor means [78, 83, 89, 94])
  synthbl        the PHYSICAL battery: identical draws to `synth` but the OU
                 shaft trajectory is zero-phase lowpassed at --synth-fc (8 Hz)
                 BEFORE audio synthesis and GT definition, at --synth-snr
                 (0 dB).  Removes the unphysical white-to-250-Hz OU drive and
                 the frame-grid aliasing of point-sampled GT (WP4 item 4/5).
  synthbl_hi     synthbl at 20 dB SNR (evidence-rich reference)
  synthblNN      a single synthbl window for chunked runs

Chains:
  baseline   the exact fullrange-v2 chain (reproduces the trace numbers:
             final PIT-MAE 3.262 dregon_ramp / 1.148 fly124_cruise on the GT
             frame grid = 3.214 / 1.140 on the trace's 400-pt tgrid)
  no_refine  baseline minus the 5 VK-refine rounds
  pk_only    ladder (viterbi_c + vit2dsp) -> pi_kalman (no VK capture/refine)
  pk_custom  baseline (or with --skip-capture = drop VK capture+refine) with
             pi_kalman kwargs overridable via --pk-kwargs '<json>' and
             --pk-repeat R sequential pi_kalman calls
  alt_loop   Round-2 chain: ladder (capture, no refine) -> alternating
             M1 (residual corridor Viterbi) / M2 (residual-audio pi_kalman)
             rounds with per-round corridor shrink (+-8@0.25 -> +-4@0.1 ->
             +-2@0.1), convergence-stopped, -> final narrow gate-mode
             pi_kalman polish. M1/M2 ported from the WP3 decouple probe.
  reseed_alt alt_loop preceded by the comb-invisible-rotor residual re-seed:
             subtract the well-tracked siblings' VK reconstruction, comb-scan
             the 8192-FFT whitened residual for the weak rotor's true base,
             corridor-track it on the residual spec, then run the alt loop
             with that rotor scored on per-round residual specs
  refine_v2  the WP4 chain (supersedes alt_loop): ladder (capture, no VK
             refine) -> M1 corridor round 1 (+-8 @ 0.25, 3 sweeps) -> M1 round
             2 (+-4 @ 0.1, 2 sweeps) -> ONE M2-solo pass -> STOP.  No
             convergence loop and no polish pass: WP4 measured that every
             extra application injects ~0.05 rev/s of fresh estimation noise
             that accumulates additively (iteration is HARMFUL).
  refine_v3  the WP7 chain: refine_v2 with the GENERALIZED residual re-seed
             (M3) inserted between the ladder and the M1 rounds.  M3 subtracts
             ALL FOUR reconstructed combs, comb-scans the residual for an
             unexplained comb (robust-z gate, >=1.5 rev/s from every track,
             small-integer-ratio guard), and hands it to the most REDUNDANT
             rotor — the one whose leave-one-out removal least increases the
             residual, i.e. a duplicate seed whose comb its twin also owns —
             then corridor-tracks it; up to 3 iterations.
  joint_beam WP17: replace `fullrange_init` + the viterbi_c / vit2dsp ladder
             stages with a JOINT 4-rotor beam search over the full speed vector
             (tracking.joint_beam_tracker), then feed the existing
             capture -> M1 -> M2 stages unchanged.  The coarse DP's state is one
             scalar c(t), so all four tracks share one shape by construction
             (WP3); this searches four independent trajectories under an OU
             control-mode prior (cheap along the common mode, mean-reverting on
             the differential modes) with a soft shared-evidence correction.
             Needs NO seed for its own stage — the candidates come from the
             score surface — which also removes the seeding lottery WP15 found
             on FLY124 w03.
  cd_iter    the baseline ladder up to the vit2dsp output (no VK capture, no
             VK refine), then EXACTLY ONE raw vk_track call (no stage guard,
             no pi_kalman, no M1/M2).  The call's config is the refine-stage
             config overridden by --cd-kwargs '<json>' (VKConfig fields);
             --entry-offset adds a constant rev/s to all four tracks first.
             The result JSON records the call's max_deltas, residual_ratios
             and extras (schedules / bw_adapt state) for capture diagnostics.

For synthetic windows an ORACLE floor is also reported: M2 run from the
chain's own M1 output (or, for chains without M1, from the track entering the
final stage) with the SIBLINGS reconstructed from ground truth — the best any
sibling-aware method could do on that window from that entry point.

Run from the repo root:
    PYTHONPATH=src .venv/bin/python scripts/rps_refine_lab.py \
        --chain baseline --windows dregon_ramp,fly124_cruise
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import itertools
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import detrend

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)  # results/* caches are repo-relative

from beatvk_vk_arms import (  # noqa: E402
    COARSE_F_MIN,
    COARSE_GAMMA,
    COARSE_NORM_SOFT,
    COARSE_SMOOTH_FRAMES,
    FRAME_S,
    SR,
    _coarse_spec,
    fullrange_init,
    load_prep,
    weights_path,
)
from vk_blind_annotation import pit_perm  # noqa: E402
from vk_validation import Prepared, smooth_frames  # noqa: E402

import tracking.phase_increment_tracker as pit  # noqa: E402
from data_processing.rps_synthesis import synth_comb_window  # noqa: E402
from tracking.joint_beam_tracker import (  # noqa: E402
    BeamCfg,
    EmissionCfg,
    OUPrior,
    joint_beam_track,
)
from tracking.pipelines import (  # noqa: E402
    MIDBAND_CFGS,
    REFINE_CFG,
    SEED_CFG,
    VIT2D_DELTA,
    VIT2D_STEP,
    VIT_GAMMA_MULT,
    joint_viterbi,
    pair_score_2d_spatial,
    tooth_cube,
    vit_stage1,
    viterbi_lattice,
    whitened_logmag_multi,
)
from tracking.vk_blind_seeding import (  # noqa: E402
    SeedConfig,
    SeedResult,
    blind_seed,
    stage_guard,
    track_comb_confidence,
    whitened_logmag,
)
from tracking.vk_tracking import (  # noqa: E402
    VKConfig,
    vk_envelopes,
    vk_reconstruct,
    vk_track,
)

#: Directory holding the beat-VK per-window prep cache (`prep_cache/*.npz` +
#: `*__weights.npz`) that `real:`/`dregon_ramp`/`fly124_cruise` windows read.
#: Overridable with ``--beatvk-out`` so the same chain can be scored against a
#: DIFFERENT protocol build (e.g. pre- vs post-recalibration labels) without
#: disturbing the production cache — see `scripts/beatvk_rescore.py`.
BEATVK_OUT = Path("results/beatvk_vk_arms")
LAB_OUT = Path("results/rps_refine_lab")
#: Where `get_seed` caches `blind_seed` results.  A seed is a pure function of
#: the window AUDIO, so it must be namespaced by the prep build: two builds of
#: the same protocol can hold DIFFERENT audio for the same (recording, window)
#: — the FLY124 recalibration moved every window's audio 86 ms — and a shared
#: cache silently scores one build with the other's seeds.  Measured cost of
#: getting this wrong: FLY124 w03 seeds 82.7 vs 54.45 rev/s on its 4th rotor.
SEED_CACHE_DIR = LAB_OUT / "seed_cache"
N_ROTORS = 4
CHAINS = (
    "baseline",
    "no_refine",
    "pk_only",
    "pk_custom",
    "alt_loop",
    "reseed_alt",
    "refine_v2",
    "refine_v3",
    "joint_beam",
    "cd_iter",
)
DEFAULT_PK: dict[str, Any] = {"n_iter": 3, "band_hz": 6.0}  # trace/baseline call
# cd_iter: VKConfig overrides for the single vk_track call (--cd-kwargs) and
# the constant rev/s added to every track before it (--entry-offset).
CD_KWARGS: dict[str, Any] = {}
ENTRY_OFFSET = 0.0
AGGR_CYCLE = (0.7, 1.0, 1.4)
# OU mode means (common, roll, pitch, yaw) of the WP1-WP3 trace synthetic
# window -> per-rotor means [78, 83, 89, 94] rev/s.
TRACE_MODES = (86.0, 0.0, -5.5, -2.5)
# Trace-verified final pooled PIT-MAE of the baseline chain on the real
# windows, in BOTH conventions: "ft" = the native GT frame grid (the trace
# JSONs' meta.final_pit_mae.pi_smoothed), "tgrid" = the trace's 400-point
# uniform visualization grid (the numbers quoted in
# docs/experiments/rps-refine-precision.md: 3.214 / 1.140).
#
# `fly124_cruise` MOVED with the protocol republish (`beatvk-valid-raw`
# 268c7660 -> 54849c13, docs/experiments/beat-vk.md § "Protocol recalibrated
# and re-scored"): the recalibrated `time_offset` cuts every FLY124 window
# 86.188 ms earlier in the WAV.  That briefly flipped the window's blind seed
# (the comb-invisible 4th rotor 82.7 -> a spurious 54.45, 1.148 -> 7.273);
# WP15's two arm-R guards (`r_span_max` + mutual dedup of accepted residual
# candidates) fixed it, and the window now scores 0.978 on the corrected
# protocol — better than the 1.148 it scored pre-recalibration, which is the
# label correction showing through.  The pre-recalibration pair is kept below
# because it is still what the OLD build produces, and reproducing it is how
# the re-score harness was validated.
# `dregon_ramp` is unaffected — DREGON telemetry was never recalibrated.
BASELINE_REF = {
    "dregon_ramp": {"ft": 3.262, "tgrid": 3.214},
    "fly124_cruise": {"ft": 0.978, "tgrid": 0.966},
}
#: The same two windows on `beatvk-valid-raw@268c766052cb` (pass
#: `--beatvk-out` at a prep cache built from that pin to reproduce).
BASELINE_REF_PRE_RECALIB = {
    "dregon_ramp": {"ft": 3.262, "tgrid": 3.214},
    "fly124_cruise": {"ft": 1.148, "tgrid": 1.140},
}
REF_TOL = 0.01
N_TGRID = 400


def r3(x: Any) -> Any:
    """Recursively round floats to 3 decimals for compact JSON."""
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return round(v, 3) if np.isfinite(v) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return r3(x.tolist())
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    if isinstance(x, dict):
        return {str(k): r3(v) for k, v in x.items()}
    return x


# ---------------------------------------------------------------------------
# metrics


def stage_metrics(traj: np.ndarray, prep: Prepared) -> dict[str, Any]:
    """Pooled + per-rotor decomposition vs raw telemetry on the GT frame grid.

    Permutation is chosen edge-masked (pit_perm), the metrics are computed on
    the FULL grid — exactly the trace_pipeline final-MAE convention, so the
    baseline chain reproduces its numbers.
    """
    p = pit_perm(traj, prep.r_meas, prep.edge)
    a = traj[list(p)]
    per_mae = np.mean(np.abs(a - prep.r_meas), axis=1)
    rows = []
    for i in range(N_ROTORS):
        pred, gt = a[i], prep.r_meas[i]
        pd = detrend(pred)
        gd = detrend(gt)
        denom = float(np.sqrt(np.sum(pd**2) * np.sum(gd**2)))
        rows.append(
            {
                "mae": float(per_mae[i]),
                "bias": float(np.mean(pred - gt)),
                "shape_corr": float(np.sum(pd * gd) / denom) if denom > 0 else None,
                "std_ratio": float(np.std(pred) / max(float(np.std(gt)), 1e-12)),
            }
        )
    return {"pooled_mae": float(per_mae.mean()), "perm": list(p), "per_rotor": rows}


class Recorder:
    """Per-stage trajectory metrics + wall time."""

    def __init__(self, prep: Prepared):
        self.prep = prep
        self.stages: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def add(self, stage: str, traj: np.ndarray) -> None:
        m = stage_metrics(traj, self.prep)
        m["stage"] = stage
        m["wall_s"] = round(time.perf_counter() - self._t0, 1)
        self._t0 = time.perf_counter()
        self.stages.append(m)
        maes = [r["mae"] for r in m["per_rotor"]]
        print(
            f"  [{stage:>14s}] pooled {m['pooled_mae']:7.3f}  "
            f"per-rotor {np.round(maes, 2).tolist()}  ({m['wall_s']:.1f}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# seeding (cached: blind_seed is the slowest fixed stage and is deterministic)


def seed_cfg_tag(cfg: SeedConfig) -> str:
    """Cache-key suffix naming every knob that differs from ``SEED_CFG``.

    Empty string at the shipped defaults, so the existing per-window seed
    caches keep their names; ANY differing field changes the key (the WP8
    sweep varies `dedup_rps` / `prefer_distinct_candidate` / `min_sep_rps`,
    and a stale cache would silently measure the old seeds).
    """
    base = asdict(SEED_CFG)
    diff = {k: v for k, v in asdict(cfg).items() if base.get(k) != v}
    if not diff:
        return ""
    return "__" + "_".join(f"{k}{v}" for k, v in sorted(diff.items())).replace(" ", "")


def get_seed(name: str, prep: Prepared, use_cache: bool, cfg: SeedConfig = SEED_CFG) -> SeedResult:
    path = SEED_CACHE_DIR / f"{name}{seed_cfg_tag(cfg)}.npz"
    if use_cache and path.exists():
        with np.load(path) as z:
            gate, bw = float(z["update_gate"]), float(z["bw_hz"])
            return SeedResult(
                bases=z["bases"].copy(),
                candidates=[],
                template=None,
                update_gate=None if np.isnan(gate) else gate,
                bw_hz=None if np.isnan(bw) else bw,
            )
    tic = time.perf_counter()
    seed = blind_seed(prep.audio, float(SR), N_ROTORS, cfg, arms=frozenset({"K", "R"}))
    print(f"  blind_seed {time.perf_counter() - tic:.0f}s", flush=True)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: several chains on the SAME window run concurrently under
        # scripts/beatvk_rescore.py and would otherwise race on this file, so a
        # reader could pick up a half-written NPZ.
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(
            tmp,
            bases=np.asarray(seed.bases, dtype=np.float64),
            update_gate=np.float64(np.nan if seed.update_gate is None else seed.update_gate),
            bw_hz=np.float64(np.nan if seed.bw_hz is None else seed.bw_hz),
        )
        os.replace(tmp, path)
    return seed


# ---------------------------------------------------------------------------
# the vit2dsp ladder (vk_blind_annotation.vit2dsp_pipeline with stage_guard,
# de-inlined so the VK capture/refine stages are individually skippable)


def run_ladder(
    prep: Prepared,
    r0: np.ndarray,
    weights: np.ndarray,
    phys_map: np.ndarray,
    mid_cfg,
    ref_cfg,
    rec: Recorder,
    do_capture: bool,
    do_refine: bool,
    skip_dp: bool = False,
) -> np.ndarray:
    """The vit2dsp ladder.  ``skip_dp`` drops stages 1-2 (the shared-c Viterbi
    and the spatial joint 2-rotor DP) so a different init stage can supply the
    entry tracks; the VK capture/refine stages and their `stage_guard` are
    untouched, which is what keeps `joint_beam` comparable to `refine_v2`."""
    lm_avg, bin_hz, st = whitened_logmag(prep.audio, float(SR), SEED_CFG)
    lm_multi, _, _ = whitened_logmag_multi(prep.audio, float(SR), SEED_CFG)
    ks = np.arange(1, 31)
    deltas = np.arange(-VIT2D_DELTA, VIT2D_DELTA + VIT2D_STEP / 2, VIT2D_STEP)
    r_cur = r0.copy()
    order = np.argsort(r_cur.mean(axis=1))
    pairs = [(int(order[0]), int(order[1])), (int(order[2]), int(order[3]))]

    def guard(label: str, r_prev: np.ndarray, r_new: np.ndarray) -> np.ndarray:
        guarded, reverted, gdiag = stage_guard(r_prev, r_new, lm_avg, bin_hz, st, prep.ft, SEED_CFG)
        if reverted:
            print(f"    stage_guard[{label}] reverted {reverted}: {gdiag['reasons']}", flush=True)
        return guarded

    # -- stage 1: Viterbi pair-mean c(t)
    if not skip_dp:
        r_prev = r_cur.copy()
        r_cur, _ = vit_stage1(prep.ft, r_cur, pairs, lm_avg, bin_hz, st, VIT_GAMMA_MULT)
        r_cur = guard("viterbi_c", r_prev, r_cur)
        rec.add("viterbi_c", r_cur)

    # -- stage 2: spatial joint 2-rotor Viterbi
    c_trajs = [r_cur[list(pair)].mean(axis=0) for pair in pairs]
    r_prev = r_cur.copy()
    for pi, pair in [] if skip_dp else list(enumerate(pairs)):
        rot_a, rot_b = int(phys_map[pair[0]]), int(phys_map[pair[1]])
        lm_a = np.tensordot(weights[:, rot_a], lm_multi, axes=(0, 0))
        lm_b = np.tensordot(weights[:, rot_b], lm_multi, axes=(0, 0))
        centers, cube_a = tooth_cube(lm_a, bin_hz, st, prep.ft, c_trajs[pi], deltas, ks)
        _, cube_b = tooth_cube(lm_b, bin_hz, st, prep.ft, c_trajs[pi], deltas, ks)
        s2 = np.stack(
            [
                pair_score_2d_spatial(cube_a[w], cube_b[w], ks, bin_hz)
                for w in range(cube_a.shape[0])
            ]
        )
        flat = s2.reshape(s2.shape[0], -1)
        contrast = float(np.median(np.max(flat, axis=1)) - np.median(np.median(flat, axis=1)))
        d1_idx, d2_idx = joint_viterbi(s2, VIT_GAMMA_MULT * contrast)
        d1 = np.interp(prep.ft, centers, deltas[d1_idx.astype(int)])
        d2 = np.interp(prep.ft, centers, deltas[d2_idx.astype(int)])
        r_cur[pair[0]] = np.maximum(c_trajs[pi] + d1, 0.0)
        r_cur[pair[1]] = np.maximum(c_trajs[pi] + d2, 0.0)
    if not skip_dp:
        r_cur = guard("vit2dsp", r_prev, r_cur)
        rec.add("vit2dsp", r_cur)

    # -- stage 3/4: midband VK capture + refine VK (each skippable)
    if do_capture:
        mid = vk_track(prep.audio, r_cur, prep.ft, mid_cfg)
        r_cur = guard("midband_bw6", r_cur, mid.r_refined)
        rec.add("capture", r_cur)
    if do_refine:
        ref = vk_track(prep.audio, r_cur, prep.ft, ref_cfg)
        r_cur = guard("refine", r_cur, ref.r_refined)
        rec.add("refine", r_cur)
    return r_cur


# ---------------------------------------------------------------------------
# M1 / M2 decoupling stages + the alternating loop (Round 2; ported from the
# WP3 scratchpad decouple_probe.py — see docs/experiments/rps-refine-precision.md)

# M1 — residual corridor Viterbi knobs (probe-frozen)
M1_K_SCORE = 8  # teeth used in the single-rotor contrast (coarse convention)
M1_K_MASK = 12  # sibling teeth masked (covers the k<=8.5 scoring range)
M1_MASK_HALF_BINS = 1.5
M1_PROTECT_HALF_BINS = 1.0
# Sub-bin corridor scoring: sample comb teeth at FRACTIONAL bin positions by
# linear interpolation along frequency.  The coarse spectrogram bin is 7.8 Hz,
# so at k=8 a nearest-bin tooth quantizes the corridor to ~1 bin / 8 = 1 Hz
# ~= 0.5 rev/s of offset resolution — coarser than the 0.1 rev/s corridor step
# the second M1 round uses.  Flip to False to score at integer bins.
M1_SUBBIN = True
# Surface-quality gate (frozen at the first M1 evaluation of a loop): a rotor
# whose own masked contrast surface is junk gets its junk inflated to ~1 by
# the per-frame normalization and the DP wanders onto residual sibling
# structure; a runaway rotor's quality then RISES because it steals a sibling
# comb, so the gate must not be re-decided mid-loop.
M1_GATE_FRAC = 0.4  # relative: fraction of the best rotor's quality
M1_GATE_ABS = 0.15  # absolute floor (dregon_ramp: all four surfaces ~0.10-0.12)

# M2 — residual-audio pi_kalman: sibling reconstruction = MIDBAND_CFGS[0]
# (bw 6 Hz) widened to k 1..30 — the subtraction MUST include the low-k lines
# that twin-collide; that is the whole point of the mechanism.
RECON_CFG = dc_replace(MIDBAND_CFGS[0], k_min=1, k_max=30)
M2_RESID_GUARD = 1.0  # skip subtraction when resid/orig RMS exceeds this
# WP2-winner pi_kalman kwargs (wide_joint_n6), used inside M2
PK_WIDE: dict[str, Any] = {
    "n_iter": 6,
    "band_hz": (12.0, 9.0, 6.0, 4.0, 2.5, 2.5),
    "k_caps": (4, 6, 8, 12, 20, 40),
    "off_comb_hz": 16.0,
    "pair_mode": "joint",
}

# Alternating loop: per-round (corridor half-width rev/s, step rev/s, M1 sweeps)
ALT_ROUNDS = ((8.0, 0.25, 3), (4.0, 0.1, 2), (2.0, 0.1, 2), (2.0, 0.1, 2))
ALT_TIGHT_SPLIT = 1.5  # any pair mean split below this -> M2 full4 (twin-gated)
ALT_CONV_TOL = 0.02  # max per-rotor mean |delta| per round below this -> stop
PK_POLISH: dict[str, Any] = {
    "n_iter": 3,
    "band_hz": (2.5, 2.0, 1.5),
    "k_caps": (20, 30, 40),
    "off_comb_hz": 11.0,
    "pair_mode": "gate",
}

# joint_beam (WP17) knobs — see tracking.joint_beam_tracker for what each
# one means and the measurement behind its default.  Overridable from the CLI so
# the emission/prior balance can be swept on the cluster without code edits.
JB_OU: dict[str, Any] = {}
JB_EMIS: dict[str, Any] = {}
JB_BEAM: dict[str, Any] = {}
JB_DEVICE = "cpu"


# refine_v2 (WP4): M1 corridor rounds only — coarse capture then a fine
# corridor — followed by exactly ONE M2-solo pass.  ALT_ROUNDS' rounds 3/4 and
# the PK_POLISH pass are deliberately absent: measured, each further
# application adds ~0.05 rev/s of estimation noise to the track.
V2_ROUNDS = ((8.0, 0.25, 3), (4.0, 0.1, 2))

# Re-seed scan (chain reseed_alt): single-rotor comb scan of the residual
RESEED_LO, RESEED_HI, RESEED_STEP = 60.0, 120.0, 0.05
RESEED_K = 12
RESEED_PEAK_SEP = 1.0  # rev/s minimum separation between reported peaks


def _paint_teeth(mask: np.ndarray, r_row: np.ndarray, bin_hz: float, half: float) -> None:
    """Set mask[j, t] = True for bins within `half` bins of k * r_row(t)."""
    n_f, n = mask.shape
    tidx = np.arange(n)
    for k in range(1, M1_K_MASK + 1):
        c = k * r_row / bin_hz  # float bin center per frame
        base = np.round(c).astype(int)
        off_max = int(np.ceil(half)) + 1
        for off in range(-off_max, off_max + 1):
            j = base + off
            sel = (np.abs(j - c) <= half) & (j >= 0) & (j < n_f)
            mask[j[sel], tidx[sel]] = True


def _mask_siblings(lm: np.ndarray, bin_hz: float, r_st: np.ndarray, rot: int) -> np.ndarray:
    """Zero the siblings' comb teeth, protecting bins near rotor `rot`'s own."""
    mask = np.zeros(lm.shape, dtype=bool)
    prot = np.zeros(lm.shape, dtype=bool)
    for sib in range(N_ROTORS):
        if sib != rot:
            _paint_teeth(mask, r_st[sib], bin_hz, M1_MASK_HALF_BINS)
    _paint_teeth(prot, r_st[rot], bin_hz, M1_PROTECT_HALF_BINS)
    out = lm.copy()
    out[mask & ~prot] = 0.0  # whitened spec: median is ~0 by construction
    return out


def _single_comb_scores(
    lm: np.ndarray, bin_hz: float, r_row: np.ndarray, d_grid: np.ndarray
) -> np.ndarray:
    """(D, N) contrast of the SINGLE-rotor comb r_row + d.

    On-teeth mean (k=1..M1_K_SCORE) minus positive half-teeth mean, same
    F_MIN / f-cap conventions as the coarse pass, with a per-frame varying
    template. With M1_SUBBIN tooth values are sampled at FRACTIONAL bin
    positions via linear interpolation along frequency (sub-bin scoring —
    integer-bin sampling quantizes at the 7.8 Hz coarse bin).
    """
    n_f, n = lm.shape
    fmax = min(6000.0, (n_f - 1) * bin_hz)
    tidx = np.arange(n)
    ks = np.arange(1, M1_K_SCORE + 1, dtype=np.float64)

    def comb_mean(freqs: np.ndarray, pos_only: bool) -> np.ndarray:
        # freqs (K, N) per-frame tooth frequencies
        valid = (freqs >= COARSE_F_MIN) & (freqs <= fmax)
        idx = np.clip(freqs, 0.0, fmax) / bin_hz
        if M1_SUBBIN:
            j = np.clip(np.floor(idx).astype(int), 0, n_f - 2)
            frac = idx - j
            vals = (1.0 - frac) * lm[j, tidx] + frac * lm[j + 1, tidx]
        else:
            vals = lm[np.clip(np.round(idx).astype(int), 0, n_f - 1), tidx]
        if pos_only:
            vals = np.maximum(vals, 0.0)
        vals = np.where(valid, vals, 0.0)
        cnt = np.maximum(valid.sum(axis=0), 1)
        return vals.sum(axis=0) / cnt

    out = np.empty((len(d_grid), n))
    for di, d in enumerate(d_grid):
        r = r_row + d
        on = comb_mean(ks[:, None] * r[None, :], False)
        half = comb_mean((ks - 0.5)[:, None] * r[None, :], True)
        out[di] = on - half
    return out


def _norm_smooth(fsc: np.ndarray) -> np.ndarray:
    """Coarse pass's per-frame normalization (smooth -> (s-med)/(peak-med))."""
    kern = np.ones(COARSE_SMOOTH_FRAMES) / COARSE_SMOOTH_FRAMES
    fsc = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, fsc)
    med = np.median(fsc, axis=0, keepdims=True)
    peak = fsc.max(axis=0, keepdims=True)
    glob = float(np.median(peak - med))
    return (fsc - med) / np.maximum(peak - med, COARSE_NORM_SOFT * glob)


def _surface_quality(
    lm: np.ndarray, bin_hz: float, r_st: np.ndarray, rot: int, d_grid: np.ndarray
) -> float:
    """Median over frames of the per-frame (peak - median) of the masked,
    UNNORMALIZED corridor score surface — the gate's quality measure."""
    lm_m = _mask_siblings(lm, bin_hz, r_st, rot)
    fsc = _single_comb_scores(lm_m, bin_hz, r_st[rot], d_grid)
    return float(np.median(fsc.max(axis=0) - np.median(fsc, axis=0)))


def m1_corridor(
    prep: Prepared,
    spec: tuple[np.ndarray, float, np.ndarray],
    r_ft: np.ndarray,
    *,
    d_half: float,
    d_step: float,
    n_sweeps: int,
    gated: set[int] | None,
    resid_specs: dict[int, np.ndarray] | None = None,
    force_pass: frozenset[int] | set[int] = frozenset(),
) -> tuple[np.ndarray, set[int], dict[str, Any]]:
    """M1 — residual corridor Viterbi (per-rotor coordinate descent).

    For rotor r, mask the siblings' current comb teeth out of the coarse
    whitened spec (protecting r's own), score the single-rotor comb contrast
    for offsets d in [-d_half, +d_half] (step d_step) around the current
    track, and Viterbi over (frame, d) with the coarse L1 transition cost.
    `gated=None` decides the surface-quality gate on the ENTRY tracks (then
    frozen: pass the returned set back in on later rounds); rotors in
    `force_pass` bypass the gate. `resid_specs` maps rotor -> residual-audio
    whitened coarse spec used for that rotor's scoring (the re-seed path).
    """
    lm, bin_hz, st = spec
    d_grid = np.arange(-d_half, d_half + d_step / 2, d_step)
    r_st = np.stack([np.interp(st, prep.ft, row) for row in r_ft])
    diag: dict[str, Any] = {"d_half": d_half, "d_step": d_step}

    def rot_lm(rot: int) -> np.ndarray:
        return resid_specs[rot] if resid_specs and rot in resid_specs else lm

    if gated is None:
        quals = [
            _surface_quality(rot_lm(rot), bin_hz, r_st, rot, d_grid) for rot in range(N_ROTORS)
        ]
        thr = max(M1_GATE_FRAC * max(quals), M1_GATE_ABS)
        gated = {r for r in range(N_ROTORS) if quals[r] < thr and r not in force_pass}
        diag["quals"] = r3(quals)
        diag["gate_thr"] = r3(thr)
        print(
            f"    [M1] surface quality {[round(q, 3) for q in quals]} "
            f"(gate {thr:.3f}) -> gated rotors {sorted(gated)}",
            flush=True,
        )
    diag["gated"] = sorted(gated)
    moves: list[dict[str, Any]] = []
    for sw in range(n_sweeps):
        for rot in range(N_ROTORS):
            if rot in gated:
                continue
            # Re-mask/re-score against the UPDATED siblings (coordinate descent).
            lm_m = _mask_siblings(rot_lm(rot), bin_hz, r_st, rot)
            fsc = _single_comb_scores(lm_m, bin_hz, r_st[rot], d_grid)
            d_path = viterbi_lattice(_norm_smooth(fsc).T, d_grid, COARSE_GAMMA)
            moves.append(
                {
                    "sweep": sw + 1,
                    "rotor": rot,
                    "d_med": r3(float(np.median(np.abs(d_path)))),
                    "d_max": r3(float(np.abs(d_path).max())),
                }
            )
            r_st[rot] = r_st[rot] + d_path
    diag["moves"] = moves
    return np.stack([np.interp(prep.ft, st, row) for row in r_st]), gated, diag


def _recon_residual(
    prep: Prepared, r_ft: np.ndarray, rows: Sequence[int]
) -> tuple[np.ndarray, float]:
    """Audio minus the VK reconstruction of the tracks `rows` (RECON_CFG).

    Returns (residual, resid/orig RMS ratio). The envelope solve is coupled,
    so a subset must be RE-FITTED, not sliced out of a full fit — that is what
    makes the leave-one-out ratio a redundancy measure (M3): dropping a track
    whose comb a sibling also explains barely moves the residual.
    """
    n_t = prep.audio.shape[-1]
    t_aud = np.arange(n_t) / SR
    r_aud = np.stack([np.interp(t_aud, prep.ft, r_ft[i]) for i in rows])
    env = vk_envelopes(prep.audio, r_aud, RECON_CFG)
    recon = vk_reconstruct(env, n_samples=n_t)
    resid = prep.audio - recon
    rms0 = float(np.sqrt(np.mean(prep.audio**2)))
    ratio = float(np.sqrt(np.mean(resid**2))) / max(rms0, 1e-30)
    return resid, ratio


def _sibling_residual(prep: Prepared, r_ft: np.ndarray, rot: int) -> tuple[np.ndarray, float]:
    """Audio minus the VK reconstruction of the SIBLINGS of rotor `rot`."""
    return _recon_residual(prep, r_ft, [i for i in range(N_ROTORS) if i != rot])


def m2_residual(
    prep: Prepared,
    r_ft: np.ndarray,
    *,
    mode: str,
    n_sweeps: int = 1,
    damp: float = 1.0,
    tag: str = "M2",
    spec: tuple[np.ndarray, float, np.ndarray] | None = None,
    proposals: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """M2 — residual-audio pi_kalman (fine decoupling).

    Per rotor: subtract the siblings' VK reconstruction from the audio and run
    pi_kalman (PK_WIDE kwargs) on the residual. mode="full4": full 4-track
    init (geometric twin gating still applies), consume only row `rot`;
    mode="solo": single-track init — no twin gating at all. When the sibling
    reconstruction diverges (resid/orig RMS > M2_RESID_GUARD, the dregon
    cancelling mode) subtraction is skipped and the plain audio is used.
    `damp` scales the applied correction (oscillation damper).

    **Move gate** (WP15): a rotor's proposal is applied only if it passes
    :func:`m2_gate_reject` — a blind, per-rotor veto of re-captures, which is
    what M2 does on well-tracked steady windows (it is a *fine* decoupling
    stage; a 1.5-2.4 rev/s "correction" is the track sliding onto a
    neighbouring line, not a bias removal). ``M2_GATE = "off"`` restores the
    ungated WP6/WP12 behaviour bit-exactly.

    `spec` = the whitened coarse spectrogram `(lm, bin_hz, st)` the gate's
    comb-confidence term needs (computed here when omitted). `proposals`, if
    given, is filled with one dict per (sweep, rotor) recording the raw
    proposal and the gate decision — the offline gate-design dump.
    """
    r = r_ft.copy()
    ratios: list[float] = []
    if M2_GATE != "off" and spec is None:
        lm_g, bin_g, st_g, _ = _coarse_spec(prep.audio)
        spec = (lm_g, bin_g, st_g)
    for sw in range(n_sweeps):
        for rot in range(N_ROTORS):
            resid, ratio = _sibling_residual(prep, r, rot)
            ratios.append(round(ratio, 4))
            failed = ratio > M2_RESID_GUARD
            if failed:
                print(
                    f"    [{tag} sw{sw + 1} rot{rot}] resid/orig RMS = {ratio:.3f}"
                    "  RECON FAILED -> plain audio",
                    flush=True,
                )
                resid = prep.audio
            if mode == "solo":
                r_new = pit.pi_kalman_refine(
                    resid, r[rot : rot + 1].copy(), prep.ft, sr=SR, **PK_WIDE
                )[0][0]
            else:
                r_new = pit.pi_kalman_refine(resid, r.copy(), prep.ft, sr=SR, **PK_WIDE)[0][rot]
            r_prop = r[rot] + damp * (r_new - r[rot])
            reject, gdiag = m2_gate_reject(r, rot, r_prop, spec, prep, recon_failed=failed)
            if proposals is not None:
                proposals.append(
                    {
                        "sweep": sw + 1,
                        "rotor": rot,
                        "ratio": round(ratio, 4),
                        "recon_failed": bool(failed),
                        "r_before": r[rot].astype(np.float32),
                        "r_prop": r_prop.astype(np.float32),
                        "rejected": bool(reject),
                        **gdiag,
                    }
                )
            if reject:
                print(
                    f"    [{tag} sw{sw + 1} rot{rot}] gate REJECT: {gdiag.get('reason', '')}",
                    flush=True,
                )
                continue
            r[rot] = r_prop
    return r, ratios


# --- the M2 move gate ------------------------------------------------------
#
# Measured (WP15, `results/refine_gate_probe`): on the 15-window real protocol
# M2-solo is a net LOSS — it costs +0.29..+0.55 rev/s on all six steady DREGON
# cruise windows and helps only marginally (<= 0.08) on the ramp ones, because
# with two tight twin pairs the sibling reconstruction explains only ~1/3 of
# the RMS (resid/orig 0.64-0.67) and the wide first PK_WIDE band (12 Hz at
# k <= 4 = +-12 rev/s of capture at k=1) lets the single-track solve slide onto
# a neighbouring line.  The damage signature is unambiguous: ONE or TWO rotors
# acquire a 1.2-2.4 rev/s NEGATIVE bias while the others barely move.
#
# The gate is therefore a *scale* veto, not a quality veto: M2 exists to remove
# sibling-interference bias, which WP3 measured at 0.3-0.5 rev/s.  A proposal
# an order of magnitude larger than that is a re-capture.
M2_GATE = "off"  # "off" = ungated (WP6/WP12 behaviour) | "move" = the scale veto
#: When set (``--m2-dump``), every M2 proposal is written here as an NPZ so any
#: accept/reject rule can be scored OFFLINE against the truth without re-running
#: the 115 s chain per variant (the gate-design loop).
M2_DUMP_PATH: Path | None = None
M2_MOVE_MAX = 0.5  # rev/s: max accepted |mean move| of one M2 proposal.  WP3
# measured sibling-interference bias at 0.3-0.5 rev/s, and the offline sweep
# over all 60 real-window proposals (`results/refine_gate_probe/m2`) puts the
# optimum flat between 0.25 and 0.75 — 0.5 is the middle of that plateau and
# the physical scale at the same time
M2_DUP_TOL = 1.5  # rev/s: reject a proposal landing this close to a sibling
# it was not already sharing a comb with (`SeedConfig.guard_dup_tol`'s rule,
# applied at M2 scale — the ladder's own guard uses a 3.0 rev/s move floor and
# so cannot see M2-scale re-captures at all)


def m2_gate_reject(
    r: np.ndarray,
    rot: int,
    r_prop: np.ndarray,
    spec: tuple[np.ndarray, float, np.ndarray] | None,
    prep: Prepared,
    recon_failed: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Blind per-rotor veto of an M2 proposal.  ``(reject, diagnostics)``.

    Three rules, all scale-based and all truth-free:

    0. **no residual, no M2** — the sibling reconstruction diverged
       (``resid/orig > M2_RESID_GUARD``).  The ungated code falls back to the
       PLAIN audio there, which is not a decoupling step at all but an
       unconstrained wide pi_kalman on a window the ladder already struggled
       with; measured cost on FLY124 w01, +2.70 rev/s.  M2's premise is void,
       so M2 declines.
    1. **move** — ``mean |r_prop - r[rot]| > M2_MOVE_MAX``.  A fine-decoupling
       correction is 0.1-0.5 rev/s; more than that is a capture event.
    2. **occupied comb** — the proposal ends within ``M2_DUP_TOL`` of a sibling
       track it was not already that close to (`stage_guard` rule 1, at M2
       scale).  This is the mechanism, not just the symptom: the solve is
       rewarded for putting two tracks on one strong comb.

    Comb confidences before/after are recorded (never used as a veto): on the
    measured failures the destination comb is STRONGER, so confidence rises —
    the WP8/`stage_guard` lesson that comb evidence cannot veto a landing on a
    better comb.
    """
    move = float(np.mean(np.abs(r_prop - r[rot])))
    occupied = False
    for j in range(N_ROTORS):
        if j == rot:
            continue
        d_after = float(np.mean(np.abs(r_prop - r[j])))
        d_before = float(np.mean(np.abs(r[rot] - r[j])))
        if d_after < M2_DUP_TOL <= d_before:
            occupied = True
            break
    diag: dict[str, Any] = {"move": round(move, 4), "occupied": occupied}
    if spec is not None:
        lm, bin_hz, st = spec
        r_after = r.copy()
        r_after[rot] = r_prop
        cb = track_comb_confidence(lm, bin_hz, st, prep.ft, r, SEED_CFG)[rot]
        ca = track_comb_confidence(lm, bin_hz, st, prep.ft, r_after, SEED_CFG)[rot]
        diag["conf_before"] = round(float(cb), 4)
        diag["conf_after"] = round(float(ca), 4)
    if M2_GATE == "off":
        return False, diag
    if recon_failed:
        diag["reason"] = "sibling reconstruction diverged — no residual to refine on"
        return True, diag
    if move > M2_MOVE_MAX:
        diag["reason"] = f"move {move:.2f} > {M2_MOVE_MAX} (re-capture, not decoupling)"
        return True, diag
    if occupied:
        diag["reason"] = f"move {move:.2f} onto an occupied comb"
        return True, diag
    return False, diag


def min_pair_split(r: np.ndarray) -> float:
    """Minimum over rotor pairs of the mean |r_i - r_j| (tight-pair check)."""
    return min(
        float(np.mean(np.abs(r[i] - r[j]))) for i in range(N_ROTORS) for j in range(i + 1, N_ROTORS)
    )


def _reseed_scan_scores(vec: np.ndarray, bin_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-rotor comb-contrast curve over the RESEED_LO..RESEED_HI grid.

    k=1..RESEED_K teeth, on-teeth mean minus positive half-teeth mean, with
    fractional-bin sampling and the coarse pass's F_MIN / f-cap conventions.
    """
    n_f = len(vec)
    fmax = min(6000.0, (n_f - 1) * bin_hz)
    grid = np.arange(RESEED_LO, RESEED_HI + RESEED_STEP / 2, RESEED_STEP)
    ks = np.arange(1, RESEED_K + 1, dtype=np.float64)

    def mean_at(freqs: np.ndarray, pos_only: bool) -> np.ndarray:
        valid = (freqs >= COARSE_F_MIN) & (freqs <= fmax)
        idx = np.clip(freqs, 0.0, fmax) / bin_hz
        j = np.clip(np.floor(idx).astype(int), 0, n_f - 2)
        frac = idx - j
        vals = (1.0 - frac) * vec[j] + frac * vec[j + 1]
        if pos_only:
            vals = np.maximum(vals, 0.0)
        vals = np.where(valid, vals, 0.0)
        return vals.sum(axis=-1) / np.maximum(valid.sum(axis=-1), 1)

    s = mean_at(grid[:, None] * ks[None, :], False) - mean_at(
        grid[:, None] * (ks - 0.5)[None, :], True
    )
    return grid, s


def _reseed_comb_scan(vec: np.ndarray, bin_hz: float) -> list[tuple[float, float]]:
    """Top-3 local peaks of :func:`_reseed_scan_scores`, RESEED_PEAK_SEP apart."""
    grid, s = _reseed_scan_scores(vec, bin_hz)
    peaks: list[tuple[float, float]] = []
    for i in np.argsort(s)[::-1]:
        b = float(grid[i])
        if all(abs(b - pb) >= RESEED_PEAK_SEP for pb, _ in peaks):
            peaks.append((b, float(s[i])))
        if len(peaks) == 3:
            break
    return peaks


def run_reseed(
    prep: Prepared,
    r: np.ndarray,
    spec: tuple[np.ndarray, float, np.ndarray],
    rec: Recorder,
) -> tuple[np.ndarray, int | None, dict[str, Any]]:
    """Comb-invisible-rotor residual re-seed (Round-2 item c).

    Identify the weakest rotor by M1 surface quality; subtract the other
    three rotors' VK reconstruction from the audio; comb-scan the 8192-FFT
    whitened, time-averaged residual spectrum for its true base; replace the
    rotor's track with (top peak + corridor Viterbi on the residual coarse
    spec). Returns (r_new, weak_rotor_or_None, info).
    """
    lm, bin_hz, st = spec
    d_half, d_step, _ = ALT_ROUNDS[0]
    d_grid = np.arange(-d_half, d_half + d_step / 2, d_step)
    r_st = np.stack([np.interp(st, prep.ft, row) for row in r])
    quals = [_surface_quality(lm, bin_hz, r_st, rot, d_grid) for rot in range(N_ROTORS)]
    thr = max(M1_GATE_FRAC * max(quals), M1_GATE_ABS)
    weak = int(np.argmin(quals))
    info: dict[str, Any] = {
        "quals": r3(quals),
        "gate_thr": r3(thr),
        "weak_rotor": weak,
        "weak_gated": bool(quals[weak] < thr),
    }
    if not info["weak_gated"]:
        print("    [reseed] no comb-invisible rotor (all pass the gate) — skipping", flush=True)
        return r, None, info

    resid, ratio = _sibling_residual(prep, r, weak)
    info["resid_ratio"] = r3(ratio)
    lm8, bin8, _ = whitened_logmag(resid, float(SR), SEED_CFG)
    peaks = _reseed_comb_scan(lm8.mean(axis=1), bin8)
    p = pit_perm(r, prep.r_meas, prep.edge)
    gt_row = int(list(p).index(weak))
    gt_mean = float(np.mean(prep.r_meas[gt_row]))
    info.update(
        {
            "peaks": r3(peaks),
            "gt_row": gt_row,
            "gt_mean": r3(gt_mean),
            "track_mean_before": r3(float(np.mean(r[weak]))),
        }
    )
    print(
        f"    [reseed] weak rotor {weak} (gt row {gt_row}, gt mean {gt_mean:.2f}, "
        f"track mean {np.mean(r[weak]):.2f}); resid/orig {ratio:.3f}; "
        f"scan peaks {[(round(b, 2), round(s, 3)) for b, s in peaks]}",
        flush=True,
    )

    # Corridor-track the re-seeded constant base on the residual coarse spec.
    base = peaks[0][0]
    lm_r = _coarse_spec(resid)[0]
    r_st[weak] = np.full(len(st), base)
    lm_m = _mask_siblings(lm_r, bin_hz, r_st, weak)
    fsc = _single_comb_scores(lm_m, bin_hz, r_st[weak], d_grid)
    d_path = viterbi_lattice(_norm_smooth(fsc).T, d_grid, COARSE_GAMMA)
    info["resid_quality"] = r3(float(np.median(fsc.max(axis=0) - np.median(fsc, axis=0))))
    info["corridor_d_med"] = r3(float(np.median(np.abs(d_path))))
    info["corridor_d_max"] = r3(float(np.abs(d_path).max()))
    r = r.copy()
    r[weak] = np.interp(prep.ft, st, r_st[weak] + d_path)
    rec.add("reseed", r)
    return r, weak, info


# ---------------------------------------------------------------------------
# M3 — residual re-seed, GENERALIZED (WP7)
#
# WP7(a) found that the entry tracks of the failure windows contain a wholly
# mis-assigned rotor because arm R of `blind_seed` could not find a 4th
# DISTINCT comb and filled the slot with a `split_nudge = 0.1 rev/s` duplicate
# of an existing base (18 of 21 battery/real windows carry such a pair).  The
# true 4th comb is usually 2-3 rev/s from a used base — inside arm R's
# `dedup_rps = 4.0` exclusion — so it can never be admitted at seed time from
# a static, time-averaged spectrum.
#
# M3 attacks it after the ladder, where all four tracks are time-varying and
# VK can therefore *cancel* the explained combs rather than mask them: subtract
# all four reconstructions, comb-scan the residual, and if an unexplained comb
# survives the guards, hand it to the most REDUNDANT rotor (the one whose
# removal least increases the residual — a duplicate's comb is still explained
# by its twin, so its own removal is nearly free).  Surface quality cannot see
# duplicates: a duplicate sits on a genuine comb and scores well.
#
# The scan score is only a PROPOSER, never the acceptance test.  Measured on
# the ladder exit of synth01/02/synthbl03, the genuinely missing rotor scores
# robust z = 1.8-3.1 in the residual while comb-residue artefacts reach 3.5-4.0
# — a z gate at the seeder's 3.0 both admits junk and rejects the truth.  M3
# therefore VERIFIES: each surviving candidate is corridor-tracked in place of
# the redundant rotor and kept only if the 4-comb reconstruction residual
# actually drops.  That is the same objective the whole VK model optimizes, so
# a spurious comb cannot win it.

RESEED3_MAX_ITERS = 3
RESEED3_Z_MIN = 1.5  # loose junk pre-filter on the residual scan (proposer only)
# rev/s a candidate comb must clear every current track by.  Set to the minimum
# pairwise rotor separation the battery itself assumes distinct rotors have
# (`synth_window`: seps.min() >= 2.0), i.e. anything closer is by construction
# the SAME rotor seen through a mis-placed track, not a new one.  At 1.5 this
# admitted FLY124's 93.1 residual peak against a track the ladder had dragged
# to 91.16 — re-seeding a second track onto one strong comb, which the coupled
# VK solve rewards (near-degenerate pairs cancel) while the tracking collapses.
RESEED3_MIN_SEP = 2.0
# Rotor-band prior: a quadrotor's four base speeds sit within ~1.25x of each
# other (the same premise as the seeder's one-sided `r_span_pad = 1.15`).  A
# residual "comb" outside that span is a wander tail or a noise ridge, and it
# is the failure mode that bit synth00: with the two genuine proposals gone the
# verifier happily took a 66 rev/s ridge against tracks at 87-96, because ANY
# extra comb with freely fitted VK envelopes absorbs some residual energy.
RESEED3_MAX_SPAN = 1.30
RESEED3_HARM_TOL = 1.0  # rev/s tolerance of the small-integer-ratio guard
RESEED3_HARM_RATIOS = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 2), (2, 3), (4, 1), (1, 4))
RESEED3_RESID_GUARD = 1.0  # abort when the 4-comb reconstruction diverges
RESEED3_CORRIDOR = ((8.0, 0.25), (2.0, 0.1))  # (half-width, step) of the re-track
RESEED3_N_CAND = 6  # residual-scan peaks examined
RESEED3_N_TRY = 3  # of those, how many are corridor-tracked and verified
# Acceptance threshold on the drop of the 4-comb resid/orig RMS ratio.  Scale-
# free reference: the leave-one-out gain of a rotor that genuinely owns a comb
# (0.05-0.08 at 0 dB, larger at high SNR).  Measured on synth01/synthbl03, a
# real repair drops the residual by 0.027-0.040 while the spurious combs the
# scan keeps proposing on later iterations drop it by 0.003-0.006 — an order of
# magnitude apart, so a quarter of a typical unique-rotor contribution
# separates them with margin at any SNR.
RESEED3_DROP_FRAC = 0.25
RESEED3_DROP_ABS = 0.008
# Redundancy gate (WP10).  The leave-one-out gain identifies the rotor to
# reassign, but by itself it CANNOT tell a `split_nudge` duplicate from a
# genuine tight twin pair: in both cases each member's own removal is cheap,
# because the sibling comb covers most of the same teeth.  On FLY124 w04 —
# whose telemetry really does hold a 74.1/74.3 twin pair, so the duplicate seed
# is the physically correct one — M3 declared the 74.44 track redundant and
# re-seeded it onto the 90 rev/s comb, collapsing the window 1.088 -> 4.743.
# The discriminator is the SIZE of that gain relative to the other rotors':
#   duplicate seed (synth00/01, synthbl*): weak gain is 5-15x below the median
#     of the other three  (0.007 vs 0.066 on synth00);
#   genuine twin (FLY124 w04):             weak gain is only ~2x below
#     (0.017 vs 0.032) — each twin still owns real, unique tooth energy.
# So require the candidate rotor to be *provably* redundant before any
# reassignment is even considered.  Applied BEFORE the corridor verification,
# which is the expensive part and, as WP7 already noted, is not self-limiting:
# any extra comb with freely fitted VK envelopes absorbs some residual energy,
# and two tracks on one strong comb are actively REWARDED by the coupled solve
# (near-degenerate pairs cancel).
RESEED3_REDUNDANCY_FRAC = 0.50
# Which track M3 may VACATE.  Only a track with another track within
# RESEED3_SPARE_NN rev/s can be spare — if nothing else covers its comb,
# moving it can only lose evidence.  The leave-one-out gain merely ORDERS the
# candidates; it does not choose, because it is non-monotonic in exactly the
# regime that matters (FLY124 w02: the 0.19 rev/s pair scores gain < 0 from the
# degenerate coupled solve although both members are genuine rotors, while the
# actually-spare member of the 1.05 rev/s pair scores the HIGHEST gain).  The
# residual-drop verification below picks the (rotor, base) pair jointly.
RESEED3_SPARE_NN = 2.0
RESEED3_N_MOVERS = 4


def _residual_comb_candidates(
    vec: np.ndarray, bin_hz: float, n_max: int = RESEED3_N_CAND
) -> list[tuple[float, float, float]]:
    """Top local peaks of the residual comb scan as (base, score, robust z)."""
    grid, s = _reseed_scan_scores(vec, bin_hz)
    med = float(np.median(s))
    mad = 1.4826 * float(np.median(np.abs(s - med)))
    peaks: list[tuple[float, float, float]] = []
    for i in np.argsort(s)[::-1]:
        b = float(grid[i])
        if all(abs(b - pb) >= RESEED_PEAK_SEP for pb, _, _ in peaks):
            peaks.append((b, float(s[i]), float((s[i] - med) / max(mad, 1e-12))))
        if len(peaks) == n_max:
            break
    return peaks


def _harmonic_of_track(base: float, means: np.ndarray) -> str | None:
    """Non-None iff `base` is a small-integer multiple/submultiple of a track.

    A comb at k*r or r/k shares every k-th tooth with the track's own comb, so
    an imperfectly cancelled track leaves exactly such a pseudo-comb in the
    residual — accepting it would re-seed a rotor onto a ghost.
    """
    for m in means:
        for p, q in RESEED3_HARM_RATIOS:
            if abs(base - float(m) * p / q) < RESEED3_HARM_TOL:
                return f"{p}/{q} of track {float(m):.2f}"
    return None


def cross_window_pool(
    means_by_window: dict[str, Sequence[float]],
    exclude: str,
    *,
    cluster_rps: float = RESEED_PEAK_SEP,
    min_votes: int = 2,
) -> list[float]:
    """Base speeds corroborated by >= `min_votes` OTHER windows of a recording.

    ASSUMPTION (stated because it is the whole content of this stage): within one
    cruise recording each rotor holds a near-constant mean speed, so a rotor that
    a window's seeder missed is still visible in its siblings' tracks.  It is a
    *proposal* source only — every base still passes M3's guards and, decisively,
    M3's residual-drop verification on THIS window's audio, so a base that the
    other windows agree on but this window's audio does not support is rejected.

    The `min_votes >= 2` corroboration rule matters: a single other window's
    track can itself be a tracking failure (FLY124 w02's tracks sit at 72.1/72.5
    against true rotors at 73.6/75.2), and unanimity across windows is exactly
    what distinguishes a real rotor from one window's artefact.  Single-linkage
    clusters are greedy and SPAN-limited to `cluster_rps` (single linkage chains
    two distinct rotors together whenever a mis-tracked window bridges them);
    each cluster votes once per source window and is
    represented by its median.
    """
    pts = sorted((float(m), w) for w, ms in means_by_window.items() if w != exclude for m in ms)
    out: list[float] = []
    i = 0
    while i < len(pts):
        j = i + 1
        while j < len(pts) and pts[j][0] - pts[i][0] < cluster_rps:
            j += 1
        chunk = pts[i:j]
        if len({w for _, w in chunk}) >= min_votes:
            out.append(round(float(np.median([m for m, _ in chunk])), 3))
        i = j
    return out


def m3_reseed(
    prep: Prepared,
    r_ft: np.ndarray,
    spec: tuple[np.ndarray, float, np.ndarray],
    rec: Recorder,
    *,
    max_iters: int = RESEED3_MAX_ITERS,
    pool: Sequence[float] = (),
    ref_means: Sequence[Sequence[float]] = (),
) -> tuple[np.ndarray, frozenset[int], dict[str, Any]]:
    """Generalized residual re-seed. Returns (tracks, reseeded rotors, diag).

    `pool` = extra candidate bases from OUTSIDE this window (WP10: the other
    cruise windows of the same recording, whose rotor speeds are near-constant).
    They are PROPOSALS only — they pass through exactly the same guards and the
    same residual-drop verification as the scan peaks, because the campaign's
    standing lesson is that contrast/z cannot gate and only tracking-then-
    measuring can.  See `cross_window_pool`.
    """
    lm, bin_hz, st = spec
    r = r_ft.copy()
    reseeded: set[int] = set()
    iters: list[dict[str, Any]] = []
    for it in range(1, max_iters + 1):
        info: dict[str, Any] = {"iter": it}
        resid_all, ratio_all = _recon_residual(prep, r, range(N_ROTORS))
        info["recon_ratio"] = r3(ratio_all)
        if ratio_all > RESEED3_RESID_GUARD:
            info["stop"] = "4-comb reconstruction diverged"
            print(f"    [M3 it{it}] recon ratio {ratio_all:.3f} > guard — stop", flush=True)
            iters.append(info)
            break
        lm8, bin8, _ = whitened_logmag(resid_all, float(SR), SEED_CFG)
        cands = _residual_comb_candidates(lm8.mean(axis=1), bin8)
        means = r.mean(axis=1)
        info["cands"] = r3(cands)
        proposals: list[tuple[float, float, float]] = []
        rejected: list[list[Any]] = []
        for base, score, z in cands:
            if z < RESEED3_Z_MIN:
                rejected.append([r3(base), r3(z), f"z < {RESEED3_Z_MIN}"])
                continue
            dmin = float(np.min(np.abs(means - base)))
            if dmin < RESEED3_MIN_SEP:
                rejected.append([r3(base), r3(z), f"{dmin:.2f} rev/s from a track"])
                continue
            harm = _harmonic_of_track(base, means)
            if harm is not None:
                rejected.append([r3(base), r3(z), harm])
                continue
            span = max(float(means.max()), base) / min(float(means.min()), base)
            if span > RESEED3_MAX_SPAN:
                rejected.append([r3(base), r3(z), f"rotor-band span {span:.2f}"])
                continue
            proposals.append((base, score, z))
            if len(proposals) == RESEED3_N_TRY:
                break
        # Cross-window pool: bases carried in from the other windows of the same
        # recording.  Same guards; a pool base already covered by a scan proposal
        # (or by a current track) adds nothing and is dropped.
        pooled: list[float] = []
        for base in sorted(float(b) for b in pool):
            dmin = float(np.min(np.abs(means - base)))
            if dmin < RESEED3_MIN_SEP:
                rejected.append([r3(base), "pool", f"{dmin:.2f} rev/s from a track"])
                continue
            near = [b for b, _, _ in proposals if abs(base - b) < RESEED_PEAK_SEP]
            if near:
                # The residual scan already found this comb: the cross-window
                # evidence CORROBORATES it (which is what lets M3 act on it at
                # all when nothing else in the window is provably redundant).
                pooled.append(near[0])
                rejected.append([r3(base), "pool", f"corroborates scan peak {near[0]:.2f}"])
                continue
            harm = _harmonic_of_track(base, means)
            if harm is not None:
                rejected.append([r3(base), "pool", harm])
                continue
            span = max(float(means.max()), base) / min(float(means.min()), base)
            if span > RESEED3_MAX_SPAN:
                rejected.append([r3(base), "pool", f"rotor-band span {span:.2f}"])
                continue
            proposals.append((base, float("nan"), float("nan")))
            pooled.append(base)
        info["rejected"] = rejected
        info["proposals"] = r3([b for b, _, _ in proposals])
        info["pooled_proposals"] = r3(pooled)
        if not proposals:
            info["stop"] = "no unexplained comb"
            print(
                f"    [M3 it{it}] recon {ratio_all:.3f}; no unexplained comb "
                f"(top {[(round(b, 2), round(z, 1)) for b, _, z in cands[:3]]}) — stop",
                flush=True,
            )
            iters.append(info)
            break

        # Worst-fitting rotor = least unique energy explained.  The leave-one-out
        # residuals are re-fitted, so a duplicate seed scores ~0 (its twin covers
        # the same comb) while a rotor that uniquely owns a comb scores high.
        loo: list[tuple[float, np.ndarray]] = []
        for rot in range(N_ROTORS):
            resid_wo, ratio_wo = _recon_residual(prep, r, [i for i in range(N_ROTORS) if i != rot])
            loo.append((ratio_wo - ratio_all, resid_wo))
        gain = [g for g, _ in loo]
        info["loo_gain"] = r3(gain)

        def _ref_median(rot: int, gain: list[float] = gain) -> float:
            return float(np.median([g for i, g in enumerate(gain) if i != rot]))

        # --- redundancy gate (WP10): M3 may only fire on a window that actually
        # CONTAINS a redundant track.  A non-positive gain is unambiguous (the
        # 4-comb fit is no better than the 3-comb one); otherwise the rotor must
        # keep less than RESEED3_REDUNDANCY_FRAC of the median unique energy of
        # the other three.  A genuine tight twin (FLY124 w04) keeps ~0.5 of it.
        red: dict[int, float] = {}
        for rot in range(N_ROTORS):
            ref_r = _ref_median(rot)
            if gain[rot] <= 0.0:
                red[rot] = -1.0  # unambiguous: the 4-comb fit is no better without it
            elif ref_r <= 0.0:
                red[rot] = float("inf")  # it owns energy while the others do not
            else:
                red[rot] = float(gain[rot]) / ref_r
        info["redundancy_ratio"] = r3([red[i] for i in range(N_ROTORS)])
        redundant = [
            rot
            for rot in range(N_ROTORS)
            if rot not in reseeded and red[rot] <= RESEED3_REDUNDANCY_FRAC
        ]
        info["redundant"] = redundant
        # --- multiplicity gate (WP10).  M3 can only ever perform ONE verified
        # repair per iteration, and the residual objective cannot rank two
        # repairs against each other: whichever degeneracy is broken, the other
        # still absorbs energy, so the drop test endorses the move that improves
        # the audio fit most, which need not be the one that fixes an unowned
        # comb.  FLY124 w02 (tracks 72.51/72.70 and 90.02/91.07 — TWO collapsed
        # pairs) is the case: vacating a member of the 0.19 rev/s pair drops the
        # residual by 0.055 while vacating a member of the spare 1.05 rev/s pair
        # drops it by -0.003, and the first is the wrong rotor (5.211 -> 5.418).
        # Same mechanism as WP7's "seed 102 needs TWO repairs" residual failure.
        # Counted as CONNECTED COMPONENTS of the near-neighbour graph, not as
        # pairs: three tracks piled on one comb are one ambiguity (any of them
        # may be the spare, and the verifier can rank them), whereas two
        # disjoint collapsed pairs are two independent repairs that a single
        # move cannot address.
        order_m = list(np.argsort(means))
        n_degen, k = 0, 0
        while k < N_ROTORS:
            j = k + 1
            while (
                j < N_ROTORS
                and float(means[order_m[j]]) - float(means[order_m[j - 1]]) <= RESEED3_SPARE_NN
            ):
                j += 1
            n_degen += 1 if j - k > 1 else 0
            k = j
        info["n_degenerate_pairs"] = n_degen
        blind_ok = bool(redundant) and n_degen <= 1
        movers_ref: list[int] = []
        if not blind_ok:
            # Without a provably redundant track (or with more than one
            # degeneracy to choose between) a residual-scan peak is not
            # trustworthy evidence that a rotor is missing: on FLY124 w03/w04
            # (all four combs correctly tracked) the scan still offers z = 2.3-2.5
            # ridges beside the 91 rev/s comb, and the drop test HAPPILY takes
            # them — two tracks on one strong comb make the coupled VK solve
            # cancel, so the residual falls (w04: 1.088 -> 4.743).  M3 may then
            # only act on EXTERNAL evidence: a base corroborated by other windows
            # of the same recording.
            why = (
                f"no LOO-redundant track (redundancy "
                f"{np.round([red[i] for i in range(N_ROTORS)], 2).tolist()})"
                if not redundant
                else f"{n_degen} disjoint degenerate track groups — one repair cannot rank them"
            )
            if not pooled:
                info["stop"] = f"{why}; no external proposal"
                print(
                    f"    [M3 it{it}] recon {ratio_all:.3f}; tracks "
                    f"{np.round(means, 2).tolist()}; leave-one-out gain "
                    f"{np.round(gain, 4).tolist()} -> {why} — stop",
                    flush=True,
                )
                iters.append(info)
                break
            proposals = [p for p in proposals if p[0] in pooled]
            info["proposals"] = r3([b for b, _, _ in proposals])
            info["pool_only"] = True
            print(
                f"    [M3 it{it}] {why} — restricting proposals to the "
                f"cross-window pool {r3(pooled)}",
                flush=True,
            )
            if ref_means:
                # Cross-window RANK evidence picks the track to vacate.  The
                # residual-drop referee cannot: on FLY124 w05 vacating a member
                # of the genuine 74/75 pair drops the residual by 0.017 while
                # vacating the spare member of the 91 pair drops it by 0.003-0.005
                # — the 81 rev/s rotor is ~10x weaker than its siblings (the
                # comb-invisible rotor), so the objective prefers the wrong move
                # by a wide margin.  Sorted rank r of a 4-rotor cruise recording
                # is stable across windows, so the track that deviates most from
                # the other windows' median rank-r speed is the misplaced one.
                ranks = np.median(np.sort(np.asarray(ref_means, dtype=float), axis=1), axis=0)
                order_idx = [int(i) for i in np.argsort(means)]
                dev = [abs(float(means[order_idx[k]]) - float(ranks[k])) for k in range(N_ROTORS)]
                k_out = int(np.argmax(dev))
                movers_ref = [order_idx[k_out]]
                info["rank_ref"] = r3(ranks.tolist())
                info["rank_dev"] = r3(dev)
                info["rank_mover"] = movers_ref
                print(
                    f"    [M3 it{it}] cross-window rank medians {np.round(ranks, 2).tolist()}; "
                    f"deviation {np.round(dev, 2).tolist()} -> vacate rotor "
                    f"{movers_ref[0]} (mean {means[movers_ref[0]]:.2f})",
                    flush=True,
                )
            else:
                movers_ref = []

        # --- which rotor may be vacated.  Only a track that has a NEAR NEIGHBOUR
        # can be spare: vacating a track whose comb nothing else covers can only
        # lose evidence.  The leave-one-out gain orders them but does NOT decide
        # — on FLY124 w02 the two most "redundant" tracks (0.19 rev/s apart, gain
        # < 0 from the degenerate coupled solve) are a genuine rotor pair, while
        # the actually-spare track sits in a 1.05 rev/s pair with the HIGHEST
        # gain.  The residual-drop verification decides both the rotor and the
        # base; the LOO gate above only decides whether M3 fires at all.
        movers = [
            rot
            for rot in range(N_ROTORS)
            if rot not in reseeded
            and min(abs(float(means[rot]) - float(means[i])) for i in range(N_ROTORS) if i != rot)
            <= RESEED3_SPARE_NN
        ]
        movers.sort(key=lambda rot: gain[rot])
        movers = movers[:RESEED3_N_MOVERS]
        if movers_ref:
            movers = [rot for rot in movers_ref if rot not in reseeded]
        info["movers"] = movers
        if not movers:
            info["stop"] = "no track has a near neighbour that could cover its comb"
            print(f"    [M3 it{it}] no vacatable track — stop", flush=True)
            iters.append(info)
            break
        print(
            f"    [M3 it{it}] recon {ratio_all:.3f}; tracks "
            f"{np.round(means, 2).tolist()}; proposals "
            f"{[(round(b, 2), round(z, 1)) for b, _, z in proposals]}; "
            f"leave-one-out gain {np.round(gain, 4).tolist()}; redundant {redundant}; "
            f"movers {movers}",
            flush=True,
        )

        # Corridor-track each (mover, proposal) on the mover's SIBLING residual
        # (the track that still owns the old comb is subtracted, so the DP cannot
        # snap back onto it), then keep the pair that best explains the audio.
        r_st_base = np.stack([np.interp(st, prep.ft, row) for row in r])
        trials: list[dict[str, Any]] = []
        best: tuple[float, np.ndarray, float, int] | None = None
        for weak in movers:
            resid_w = loo[weak][1]
            lm_r = _coarse_spec(resid_w)[0] if gain[weak] + ratio_all <= RESEED3_RESID_GUARD else lm
            for base, _score, z in proposals:
                row = np.full(len(st), base)
                for d_half, d_step in RESEED3_CORRIDOR:
                    d_grid = np.arange(-d_half, d_half + d_step / 2, d_step)
                    r_st = r_st_base.copy()
                    r_st[weak] = row
                    lm_m = _mask_siblings(lm_r, bin_hz, r_st, weak)
                    fsc = _single_comb_scores(lm_m, bin_hz, row, d_grid)
                    row = row + viterbi_lattice(_norm_smooth(fsc).T, d_grid, COARSE_GAMMA)
                r_try = r.copy()
                r_try[weak] = np.interp(prep.ft, st, row)
                # The corridor DP may land the re-seeded rotor close to a sibling.
                # That is NOT rejected: M1's sibling-masked coordinate descent is
                # exactly the mechanism that separates two co-located tracks, and
                # rejecting here cost synth01 its (correct) repair.  Recorded only.
                new_mean = float(np.mean(r_try[weak]))
                sib_gap = min(abs(new_mean - float(means[i])) for i in range(N_ROTORS) if i != weak)
                _, ratio_try = _recon_residual(prep, r_try, range(N_ROTORS))
                trials.append(
                    {
                        "rotor": weak,
                        "base": r3(base),
                        "z": r3(z),
                        "tracked_mean": r3(new_mean),
                        "sib_gap": r3(sib_gap),
                        "ratio": r3(ratio_try),
                        "drop": r3(ratio_all - ratio_try),
                    }
                )
                if best is None or ratio_try < best[0]:
                    best = (ratio_try, r_try, base, weak)
        info["trials"] = trials
        print(
            f"    [M3 it{it}] verify "
            f"{[(t['rotor'], t['base'], t['tracked_mean'], t['drop']) for t in trials]}",
            flush=True,
        )
        if best is None:
            info["stop"] = "every proposal re-tracked onto an existing comb"
            print(f"    [M3 it{it}] all proposals are duplicates — stop", flush=True)
            iters.append(info)
            break
        weak = best[3]
        ref = _ref_median(weak)
        # With an externally corroborated base AND an externally chosen rotor the
        # audio only has to NOT CONTRADICT: the absolute floor exists to stop the
        # residual scan's junk proposals, and neither input came from the scan.
        # It stays a floor of ZERO rather than being removed — on FLY124 w02 the
        # rank-chosen rotor's repair INCREASES the residual (-0.003) and is
        # correctly refused.
        min_drop = 0.0 if movers_ref else max(RESEED3_DROP_ABS, RESEED3_DROP_FRAC * ref)
        info["weak_rotor"] = weak
        info["weak_mean_before"] = r3(float(means[weak]))
        info["min_drop"] = r3(min_drop)
        if ratio_all - best[0] < min_drop:
            info["stop"] = f"best drop {ratio_all - best[0]:.4f} < {min_drop:.4f}"
            print(
                f"    [M3 it{it}] best drop {ratio_all - best[0]:.4f} < {min_drop:.4f} — stop",
                flush=True,
            )
            iters.append(info)
            break
        r = best[1]
        info["accepted_base"] = r3(best[2])
        info["weak_mean_after"] = r3(float(np.mean(r[weak])))
        reseeded.add(weak)
        iters.append(info)
        rec.add(f"m3_r{it}", r)
    return r, frozenset(reseeded), {"iters": iters, "reseeded": sorted(reseeded)}


def run_alt_chain(
    prep: Prepared,
    r: np.ndarray,
    rec: Recorder,
    *,
    reseed: bool,
    damp: float = 1.0,
    max_rounds: int = len(ALT_ROUNDS),
    m2_rounds: int = 99,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Alternating M1/M2 refinement (Round-2 item b), then the narrow polish.

    Rounds of M1 -> M2 with per-round corridor shrink (ALT_ROUNDS); M2 mode
    per round from the current tracks (any pair mean split < ALT_TIGHT_SPLIT
    -> full4 twin-gated, else solo); stop when the max per-rotor mean |delta|
    of a round drops below ALT_CONV_TOL. Final PK_POLISH gate-mode pi_kalman.
    """
    diag: dict[str, Any] = {}
    lm, bin_hz, st, _ = _coarse_spec(prep.audio)
    spec = (lm, bin_hz, st)
    resid_rotors: set[int] = set()
    if reseed:
        r, weak, reseed_info = run_reseed(prep, r, spec, rec)
        diag["reseed"] = reseed_info
        if weak is not None:
            resid_rotors = {weak}
    gated: set[int] | None = None
    rounds: list[dict[str, Any]] = []
    r_m1 = r.copy()
    for rd, (d_half, d_step, m1_sweeps) in enumerate(ALT_ROUNDS[:max_rounds], 1):
        r_prev = r.copy()
        resid_specs = (
            {rot: _coarse_spec(_sibling_residual(prep, r, rot)[0])[0] for rot in resid_rotors}
            if resid_rotors
            else None
        )
        r, gated, m1_diag = m1_corridor(
            prep,
            spec,
            r,
            d_half=d_half,
            d_step=d_step,
            n_sweeps=m1_sweeps,
            gated=gated,
            resid_specs=resid_specs,
            force_pass=resid_rotors,
        )
        rec.add(f"m1_r{rd}", r)
        r_m1 = r.copy()
        rinfo: dict[str, Any] = {"round": rd, "m1": m1_diag}
        split = min_pair_split(r)
        rinfo["min_split"] = r3(split)
        if rd <= m2_rounds:
            mode = "full4" if split < ALT_TIGHT_SPLIT else "solo"
            r, ratios = m2_residual(prep, r, mode=mode, damp=damp, tag=f"M2 r{rd}")
            rec.add(f"m2_r{rd}_{mode}", r)
            rinfo["m2_mode"] = mode
            rinfo["m2_ratios"] = ratios
            rinfo["m2_damp"] = damp
        move = float(np.max(np.mean(np.abs(r - r_prev), axis=1)))
        rinfo["move"] = r3(move)
        rinfo["mae_after"] = r3(rec.stages[-1]["pooled_mae"])
        rounds.append(rinfo)
        print(f"    [alt r{rd}] move {move:.4f} (conv tol {ALT_CONV_TOL})", flush=True)
        if move < ALT_CONV_TOL:
            break
    diag["rounds"] = rounds
    r = pit.pi_kalman_refine(prep.audio, r, prep.ft, sr=SR, **PK_POLISH)[0]
    rec.add("polish", r)
    return r, diag, r_m1


def run_v2_chain(
    prep: Prepared,
    r: np.ndarray,
    rec: Recorder,
    n_rounds: int = len(V2_ROUNDS),
    *,
    spec: tuple[np.ndarray, float, np.ndarray] | None = None,
    force_pass: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """refine_v2 (WP4) — M1 corridor rounds, then ONE M2-solo pass, then stop.

    Entry is the ladder WITH midband VK capture and WITHOUT the VK refine
    rounds (WP1: refine is a measured no-op).  V2_ROUNDS gives a coarse
    corridor (+-8 @ 0.25, 3 sweeps) followed by a fine one (+-4 @ 0.1, 2
    sweeps); the surface-quality gate is decided on the entry tracks and then
    frozen.  There is deliberately no convergence loop and no polish pass:
    WP4 measured that each further application of the estimator adds ~0.05
    rev/s of fresh noise to the track, so the chain peaks after one M2.

    Returns ``(tracks, diag, M1 output)``; the M1 output is the entry point
    the oracle floor is measured from.
    """
    if spec is None:
        lm, bin_hz, st, _ = _coarse_spec(prep.audio)
        spec = (lm, bin_hz, st)
    gated: set[int] | None = None
    rounds: list[dict[str, Any]] = []
    for rd, (d_half, d_step, m1_sweeps) in enumerate(V2_ROUNDS[:n_rounds], 1):
        r, gated, m1_diag = m1_corridor(
            prep,
            spec,
            r,
            d_half=d_half,
            d_step=d_step,
            n_sweeps=m1_sweeps,
            gated=gated,
            force_pass=force_pass,
        )
        rec.add(f"m1_r{rd}", r)
        rounds.append({"round": rd, "m1": m1_diag, "mae_after": r3(rec.stages[-1]["pooled_mae"])})
    r_m1 = r.copy()
    proposals: list[dict[str, Any]] = []
    r, ratios = m2_residual(
        prep, r, mode="solo", n_sweeps=1, tag="M2", spec=spec, proposals=proposals
    )
    rec.add("m2_solo", r)
    if M2_DUMP_PATH is not None:
        _dump_m2_proposals(prep, r_m1, proposals)
    diag: dict[str, Any] = {
        "rounds": rounds,
        "n_m1_rounds": n_rounds,
        "m2_mode": "solo",
        "m2_ratios": ratios,
        "m2_gate": M2_GATE,
        "m2_proposals": [
            {k: v for k, v in p.items() if not isinstance(v, np.ndarray)} for p in proposals
        ],
        "subbin": M1_SUBBIN,
    }
    return r, diag, r_m1


def _dump_m2_proposals(
    prep: Prepared, r_entry: np.ndarray, proposals: list[dict[str, Any]]
) -> None:
    """Write the M2 entry tracks + every per-rotor proposal to ``M2_DUMP_PATH``.

    Everything an offline gate study needs: the M1 output the proposals are
    relative to, each proposal's replacement row, its blind diagnostics, and
    the truth (`r_meas` + `edge`) so a candidate rule's final PIT-MAE is exactly
    computable without re-running the chain.
    """
    assert M2_DUMP_PATH is not None
    M2_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "rid": prep.rid,
        "ft": prep.ft,
        "edge": prep.edge,
        "r_meas": prep.r_meas,
        "r_entry": r_entry,
        "rotor": np.array([p["rotor"] for p in proposals]),
        "sweep": np.array([p["sweep"] for p in proposals]),
        "ratio": np.array([p["ratio"] for p in proposals]),
        "move": np.array([p["move"] for p in proposals]),
        "occupied": np.array([p["occupied"] for p in proposals]),
        "rejected": np.array([p["rejected"] for p in proposals]),
        "conf_before": np.array([p.get("conf_before", np.nan) for p in proposals]),
        "conf_after": np.array([p.get("conf_after", np.nan) for p in proposals]),
        "r_before": np.stack([p["r_before"] for p in proposals]),
        "r_prop": np.stack([p["r_prop"] for p in proposals]),
    }
    np.savez_compressed(M2_DUMP_PATH, **payload)
    print(f"  [M2] proposals dumped -> {M2_DUMP_PATH}", flush=True)


def gt_aligned(prep: Prepared, r: np.ndarray) -> np.ndarray:
    """Ground truth permuted into the TRACK row order of `r` (edge-masked PIT)."""
    p = pit_perm(r, prep.r_meas, prep.edge)
    gt = np.empty_like(prep.r_meas)
    for truth_row, track_row in enumerate(list(p)):
        gt[track_row] = prep.r_meas[truth_row]
    return gt


def oracle_m2_floor(prep: Prepared, r_entry: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """ORACLE M2 — the per-window best any sibling-aware method could do.

    Synthetic windows only (GT must be exact).  For each rotor the SIBLINGS
    are reconstructed from ground-truth tracks, so the residual holds only
    that rotor's own comb plus additive noise; the same estimator (PK_WIDE)
    and the same entry tracks as the chain's own M2 pass are used, which
    makes the gap to the chain exactly the cost of imperfect sibling
    knowledge.  Ported from `refine-diagnosis/sibsub_clean.oracle_m2`.
    """
    gt = gt_aligned(prep, r_entry)
    r = r_entry.copy()
    ratios: list[float] = []
    skipped: list[int] = []
    for rot in range(N_ROTORS):
        r_sib = gt.copy()
        r_sib[rot] = r_entry[rot]  # only this rotor's own track is imperfect
        resid, ratio = _sibling_residual(prep, r_sib, rot)
        ratios.append(round(ratio, 4))
        if ratio > M2_RESID_GUARD:
            skipped.append(rot)
            continue
        r[rot] = pit.pi_kalman_refine(
            resid, r_entry[rot : rot + 1].copy(), prep.ft, sr=SR, **PK_WIDE
        )[0][0]
    return r, {"ratios": ratios, "skipped": skipped}


# ---------------------------------------------------------------------------
# chain driver


def run_chain(
    name: str,
    prep: Prepared,
    weights: np.ndarray,
    meta: dict[str, Any],
    chain: str,
    pk_kwargs: dict[str, Any],
    pk_repeat: int,
    skip_capture: bool,
    seed_cache: bool,
    do_oracle: bool,
    v2_rounds: int,
    seed_cfg: SeedConfig = SEED_CFG,
    m3_pool: Sequence[float] = (),
    m3_ref: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    print(f"\n=== {name} ({prep.rid}) chain={chain} ===", flush=True)
    t_start = time.perf_counter()
    rec = Recorder(prep)

    joint_chain = chain == "joint_beam"
    seed = get_seed(name, prep, seed_cache, seed_cfg)
    bases0 = np.sort(np.asarray(seed.bases, dtype=np.float64))
    rec.add("seed", np.repeat(bases0[:, None], len(prep.ft), axis=1))

    spec_c: tuple[np.ndarray, float, np.ndarray] | None = None
    jb_diag: dict[str, Any] | None = None
    if joint_chain:
        # The joint search needs NO seed bases: its candidates come from the
        # score surface itself.  `seed` is still computed because the midband
        # VK capture stage downstream reads its `update_gate` calibration, and
        # so the seed row stays in the per-stage table for comparability.
        lm_c, bin_c, st_c, _ = _coarse_spec(prep.audio)
        spec_c = (lm_c, bin_c, st_c)
        t_jb = time.perf_counter()
        r0, jb_diag = joint_beam_track(
            lm_c,
            bin_c,
            st_c,
            prep.ft,
            ou=OUPrior(**JB_OU),
            emis=EmissionCfg(**JB_EMIS),
            beam=BeamCfg(**JB_BEAM),
            device=JB_DEVICE,
        )
        assert jb_diag is not None
        jb_diag["wall_s"] = round(time.perf_counter() - t_jb, 1)
        seed_eff, coarse_diag = seed, {"coarse_mode": "joint_beam"}
        rec.add("joint_beam", r0)
    else:
        r0, seed_eff, coarse_diag = fullrange_init(prep, seed)
        rec.add("coarse_init", r0)

    gate = seed_eff.update_gate
    mid_cfg = MIDBAND_CFGS[0] if gate is None else dc_replace(MIDBAND_CFGS[0], update_gate=gate)
    ref_cfg = REFINE_CFG if gate is None else dc_replace(REFINE_CFG, update_gate=gate)

    # track -> physical rotor map (PIT vs measured; run_job convention)
    p = pit_perm(r0, prep.r_meas, prep.edge)
    phys_map = np.empty(N_ROTORS, dtype=int)
    for truth_row, track_row in enumerate(list(p)):
        phys_map[track_row] = truth_row

    alt_chain = chain in ("alt_loop", "reseed_alt")
    v2_chain = chain in ("refine_v2", "refine_v3", "joint_beam")
    vk_stages = chain == "baseline" or (chain == "pk_custom" and not skip_capture)
    # alt / v2 chains enter from the ladder WITH midband capture but WITHOUT
    # the (measured no-op, WP1) VK refine rounds.
    do_capture = vk_stages or chain == "no_refine" or alt_chain or v2_chain
    do_refine = vk_stages
    r = run_ladder(
        prep,
        r0,
        weights,
        phys_map,
        mid_cfg,
        ref_cfg,
        rec,
        do_capture,
        do_refine,
        skip_dp=joint_chain,
    )

    alt_diag: dict[str, Any] | None = None
    if alt_chain:
        r, alt_diag, r_entry = run_alt_chain(prep, r, rec, reseed=chain == "reseed_alt")
    elif v2_chain:
        if spec_c is None:
            lm_c, bin_c, st_c, _ = _coarse_spec(prep.audio)
            spec_c = (lm_c, bin_c, st_c)
        force_pass: frozenset[int] = frozenset()
        m3_diag: dict[str, Any] | None = None
        if chain == "refine_v3":
            r, force_pass, m3_diag = m3_reseed(prep, r, spec_c, rec, pool=m3_pool, ref_means=m3_ref)
        r, alt_diag, r_entry = run_v2_chain(
            prep, r, rec, n_rounds=v2_rounds, spec=spec_c, force_pass=force_pass
        )
        if m3_diag is not None:
            alt_diag["m3"] = m3_diag
        if jb_diag is not None:
            alt_diag["joint_beam"] = jb_diag
    elif chain == "cd_iter":
        # One RAW vk_track from the ladder output: no stage guard, no
        # pi_kalman, no M1/M2 — the call's own diagnostics are the point.
        if ENTRY_OFFSET != 0.0:
            r = r + ENTRY_OFFSET
        r_entry = r.copy()
        cd_cfg = dc_replace(ref_cfg, **CD_KWARGS)
        cd = vk_track(prep.audio, r, prep.ft, cd_cfg)
        r = cd.r_refined
        rec.add("cd_iter", r)
        alt_diag = {
            "cd_iter": {
                "entry_offset": ENTRY_OFFSET,
                "cd_kwargs": CD_KWARGS,
                "max_deltas": cd.max_deltas,
                "residual_ratios": cd.residual_ratios,
                "extras": cd.extras,
            }
        }
    else:
        # Chains without an M1 stage: the oracle entry is the track that goes
        # into the final estimator call.
        r_entry = r.copy()
        for rep in range(pk_repeat):
            r = pit.pi_kalman_refine(prep.audio, r, prep.ft, sr=SR, **pk_kwargs)[0]
            rec.add(f"pi_kalman_{rep + 1}" if pk_repeat > 1 else "pi_kalman", r)

    oracle: dict[str, Any] | None = None
    if do_oracle and meta.get("synthetic"):
        t_or = time.perf_counter()
        r_or, or_diag = oracle_m2_floor(prep, r_entry)
        oracle = {**stage_metrics(r_or, prep), **or_diag}
        oracle["wall_s"] = round(time.perf_counter() - t_or, 1)
        print(
            f"  [  oracle_floor] pooled {oracle['pooled_mae']:7.3f}  "
            f"per-rotor {np.round([q['mae'] for q in oracle['per_rotor']], 2).tolist()}  "
            f"({oracle['wall_s']:.1f}s)",
            flush=True,
        )

    # Final PIT-MAE on the trace's 400-pt tgrid (the design-doc convention).
    tgrid = np.linspace(0.0, float(prep.ft[-1]), N_TGRID)
    tr_t = np.stack([np.interp(tgrid, prep.ft, row) for row in r])
    gt_t = np.stack([np.interp(tgrid, prep.ft, row) for row in prep.r_meas])
    mae_tgrid = min(
        float(np.mean(np.abs(tr_t[list(p)] - gt_t)))
        for p in itertools.permutations(range(N_ROTORS))
    )

    wall = time.perf_counter() - t_start
    print(f"  window wall {wall:.0f}s", flush=True)
    return {
        "final_pooled_mae_tgrid": mae_tgrid,
        "meta": {
            **meta,
            "name": name,
            "chain": chain,
            "pk_kwargs": r3(pk_kwargs),
            "pk_repeat": pk_repeat,
            "seed_bases": r3(bases0),
            "final_means": r3(np.sort(r.mean(axis=1))),
            "m3_pool": r3(list(m3_pool)),
            "seed_cfg": seed_cfg_tag(seed_cfg) or "default",
            "coarse": r3(
                {k: v for k, v in coarse_diag.items() if isinstance(v, (int, float, str, bool))}
            ),
            "wall_s": round(wall, 1),
            **({"alt": alt_diag} if alt_diag is not None else {}),
        },
        "stages": rec.stages,
        "final_pooled_mae": rec.stages[-1]["pooled_mae"],
        "oracle_floor": oracle,
    }


# ---------------------------------------------------------------------------
# windows


def real_window(rid: str, widx: int) -> tuple[Prepared, np.ndarray, dict[str, Any]]:
    prep, regime = load_prep(BEATVK_OUT, rid, widx, channels=8)
    with np.load(weights_path(BEATVK_OUT, rid)) as z:
        weights = z["weights"][:8]
    meta = {
        "source": "beatvk-valid-raw",
        "recording_id": rid,
        "window_index": widx,
        "regime": regime,
    }
    return prep, weights, meta


def synth_window(
    seed: int,
    aggressiveness: float,
    mode_means: tuple[float, float, float, float] | None = None,
    fc_hz: float | None = None,
    snr_db: float = 0.0,
) -> tuple[Prepared, np.ndarray, dict[str, Any]]:
    """One synthetic battery window: a `rps_synthesis.synth_comb_window` draw
    (OU free-flight modes + locked-phase 2-blade harmonic comb) wrapped into
    this lab's `Prepared` container on the 31.25 Hz evaluation frame grid.
    With `mode_means=None` the mode means are drawn per window so rotor means
    land in ~[70, 100] rev/s with >= 2 rev/s pairwise separation.

    `fc_hz` band-limits the commanded shaft speed (rotor inertia) BEFORE audio
    synthesis, and GT is defined from that same band-limited trajectory — the
    physical convention of WP4 item 5.  Without it the OU drive is white to
    the 250 Hz generation rate and point-sampling GT onto the 31.25 Hz frame
    grid folds all of it back into the comparison band (~47% of the residual
    power sits at 8-16 Hz).  The random draws are unchanged by either knob, so
    `fc_hz=None, snr_db=0` reproduces the original battery bit-exactly.
    """
    win = synth_comb_window(
        seed,
        aggressiveness=aggressiveness,
        mode_means=mode_means,
        fc_hz=fc_hz,
        snr_db=snr_db,
        sr=SR,
        n_mic=2,
    )
    dur = win.meta["duration_s"]
    n_t = win.audio.shape[1]

    ft = np.arange(0.0, n_t / SR - FRAME_S / 2, FRAME_S)
    r_meas = np.stack([np.interp(ft, win.t, win.r_true[i]) for i in range(N_ROTORS)])
    edge = (ft > 0.5) & (ft < ft[-1] - 0.5)
    bl_tag = "" if fc_hz is None else f"_bl{fc_hz:g}"
    prep = Prepared(
        rid=f"synthetic{bl_tag}_s{seed}",
        tau=0.0,
        seg_lo=0.0,
        seg_hi=dur,
        audio=win.audio,
        ft=ft,
        r_init=r_meas.copy(),
        r_meas=r_meas,
        r_meas_sm=smooth_frames(r_meas),
        edge=edge,
    )
    weights = np.full((2, N_ROTORS), 0.5)
    meta = {
        "source": "synthetic OU free-flight + locked-phase comb (trace_pipeline path)",
        "synthetic": True,
        "shaft_fc_hz": fc_hz,
        "snr_db": snr_db,
        "seed": seed,
        "aggressiveness": aggressiveness,
        "mode_means": r3(list(win.mode_means)),
        "rotor_means": r3(np.sort(win.rotor_means)),
    }
    return prep, weights, meta


# ---------------------------------------------------------------------------
# main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast CPU lab for the blind-VK RPS refinement chain (WP0)."
    )
    ap.add_argument("--chain", choices=CHAINS, default="baseline")
    ap.add_argument(
        "--windows",
        default="dregon_ramp,fly124_cruise,synth",
        help="comma list of dregon_ramp | fly124_cruise | synth (= the battery)",
    )
    ap.add_argument("--synth-battery", type=int, default=6, help="battery size for 'synth'")
    ap.add_argument("--synth-base-seed", type=int, default=100)
    ap.add_argument(
        "--synth-fc",
        type=float,
        default=8.0,
        help="synthbl: shaft-speed lowpass cutoff in Hz (physical GT convention)",
    )
    ap.add_argument("--synth-snr", type=float, default=0.0, help="synthbl: comb-vs-noise SNR in dB")
    ap.add_argument(
        "--no-subbin",
        action="store_true",
        help="score the M1 corridor at integer spectrogram bins (M1_SUBBIN=False)",
    )
    ap.add_argument(
        "--v2-rounds",
        type=int,
        default=len(V2_ROUNDS),
        help="refine_v2: how many M1 corridor rounds to run (V2_ROUNDS prefix)",
    )
    ap.add_argument(
        "--no-oracle",
        action="store_true",
        help="skip the oracle sibling-removal floor on synthetic windows",
    )
    ap.add_argument(
        "--m3-redundancy-frac",
        type=float,
        default=None,
        help="refine_v3: max leave-one-out gain of the rotor M3 wants to reassign, "
        "as a fraction of the median gain of the other three (default 0.30; a huge "
        "value disables the gate and reproduces the WP7 behaviour)",
    )
    ap.add_argument(
        "--m3-pool",
        default=None,
        help="refine_v3: comma list of extra base rev/s M3 may propose (the "
        "cross-window pool; each still passes every guard AND the residual-drop "
        "verification)",
    )
    ap.add_argument(
        "--m3-ref",
        default=None,
        help="refine_v3: the OTHER windows' sorted final track means, as "
        "'a,b,c,d;a,b,c,d;...'.  Used only in the pool-escape branch, to pick "
        "which track to vacate by sorted-rank deviation.",
    )
    ap.add_argument(
        "--m2-gate",
        choices=("off", "move"),
        default=None,
        help="refine_v2/v3: per-rotor veto of an M2 proposal. 'move' rejects a "
        "proposal whose |mean move| exceeds --m2-move-max or which lands on a "
        "sibling's comb; 'off' = the ungated WP6/WP12 behaviour (current default)",
    )
    ap.add_argument(
        "--m2-move-max",
        type=float,
        default=None,
        help=f"--m2-gate move: max accepted |mean move| in rev/s (default {M2_MOVE_MAX})",
    )
    ap.add_argument(
        "--m2-dump",
        default=None,
        help="write every M2 proposal (+ truth) to this NPZ for offline gate scoring",
    )
    ap.add_argument(
        "--jb-ou",
        default=None,
        help="joint_beam: JSON kwargs for OUPrior (tau_common/tau_diff/"
        "sigma_level_diff/s_random_walk/s_scale/huber_knee)",
    )
    ap.add_argument(
        "--jb-emis",
        default=None,
        help="joint_beam: JSON kwargs for EmissionCfg (lo/hi/step/k_max/b0_rps/"
        "n_band/k_weight) — b0_rps is the k-scaled-bandwidth lever",
    )
    ap.add_argument(
        "--jb-beam",
        default=None,
        help="joint_beam: JSON kwargs for BeamCfg (width/n_global/n_peaks/"
        "n_local/local_half_rps/lambda_e/overlap_sigma_rps/overlap_gain/"
        "dedup_rps/mu_mode)",
    )
    ap.add_argument("--jb-device", default="cpu", help="joint_beam: torch device")
    ap.add_argument("--pk-kwargs", default=None, help="JSON kwargs for pi_kalman_refine")
    ap.add_argument("--pk-repeat", type=int, default=1, help="sequential pi_kalman calls")
    ap.add_argument(
        "--cd-kwargs",
        default=None,
        help="cd_iter: JSON VKConfig field overrides applied to the refine-stage config",
    )
    ap.add_argument(
        "--entry-offset",
        type=float,
        default=0.0,
        help="cd_iter: constant rev/s added to all four tracks before the vk_track call",
    )
    ap.add_argument(
        "--skip-capture",
        action="store_true",
        help="pk_custom: drop the VK capture+refine stages (ladder -> pi_kalman)",
    )
    ap.add_argument("--no-seed-cache", action="store_true")
    ap.add_argument(
        "--dedup-rps",
        type=float,
        default=None,
        help="SeedConfig.dedup_rps override (arm R: rev/s a NEW residual comb "
        "must clear every used base by; default 4.0)",
    )
    ap.add_argument(
        "--prefer-distinct",
        action="store_true",
        help="SeedConfig.prefer_distinct_candidate: promote an accepted distinct "
        "candidate ahead of a split_nudge duplicate seed",
    )
    ap.add_argument(
        "--min-sep-rps",
        type=float,
        default=None,
        help="SeedConfig.min_sep_rps override (promotion separation; default 2.0)",
    )
    ap.add_argument(
        "--promote-z-min",
        type=float,
        default=None,
        help="SeedConfig.promote_z_min override (promotion contrast gate; default 2.0)",
    )
    ap.add_argument(
        "--promote-span",
        type=float,
        default=None,
        help="SeedConfig.promote_span override (rotor-band ratio; default 1.30)",
    )
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument(
        "--beatvk-out",
        default=None,
        help="override BEATVK_OUT: the directory whose prep_cache/ the real: "
        "windows are read from (a different protocol build)",
    )
    args = ap.parse_args()

    if args.beatvk_out:
        globals()["BEATVK_OUT"] = Path(args.beatvk_out)
        # Seeds follow the audio they were computed from, never the default
        # cache — see the SEED_CACHE_DIR comment.
        globals()["SEED_CACHE_DIR"] = Path(args.beatvk_out) / "seed_cache"
        print(f"beatvk prep cache: {args.beatvk_out} (seeds cached beside it)")
    if args.no_subbin:
        globals()["M1_SUBBIN"] = False
    if args.m2_gate is not None:
        globals()["M2_GATE"] = args.m2_gate
        print(f"M2 gate: {args.m2_gate}")
    if args.m2_move_max is not None:
        globals()["M2_MOVE_MAX"] = float(args.m2_move_max)
        print(f"M2 move ceiling: {args.m2_move_max} rev/s")
    if args.m2_dump:
        globals()["M2_DUMP_PATH"] = Path(args.m2_dump)
    m3_pool = tuple(float(s) for s in args.m3_pool.split(",") if s.strip()) if args.m3_pool else ()
    if m3_pool:
        print(f"M3 cross-window proposal pool: {list(m3_pool)}")
    m3_ref = (
        tuple(
            tuple(float(x) for x in grp.split(",")) for grp in args.m3_ref.split(";") if grp.strip()
        )
        if args.m3_ref
        else ()
    )
    if m3_ref:
        print(f"M3 cross-window rank reference: {[list(g) for g in m3_ref]}")
    if args.m3_redundancy_frac is not None:
        globals()["RESEED3_REDUNDANCY_FRAC"] = float(args.m3_redundancy_frac)
        print(f"M3 redundancy gate: {args.m3_redundancy_frac}")

    seed_overrides: dict[str, Any] = {}
    if args.dedup_rps is not None:
        seed_overrides["dedup_rps"] = float(args.dedup_rps)
    if args.prefer_distinct:
        seed_overrides["prefer_distinct_candidate"] = True
    if args.min_sep_rps is not None:
        seed_overrides["min_sep_rps"] = float(args.min_sep_rps)
    if args.promote_z_min is not None:
        seed_overrides["promote_z_min"] = float(args.promote_z_min)
    if args.promote_span is not None:
        seed_overrides["promote_span"] = float(args.promote_span)
    seed_cfg = dc_replace(SEED_CFG, **seed_overrides) if seed_overrides else SEED_CFG
    if seed_overrides:
        print(f"seed config overrides: {seed_overrides} (cache tag {seed_cfg_tag(seed_cfg)})")

    for flag, target in (("jb_ou", "JB_OU"), ("jb_emis", "JB_EMIS"), ("jb_beam", "JB_BEAM")):
        raw = getattr(args, flag)
        if raw:
            globals()[target] = json.loads(raw)
            print(f"joint_beam {target}: {globals()[target]}")
    globals()["JB_DEVICE"] = args.jb_device

    if (args.cd_kwargs or args.entry_offset != 0.0) and args.chain != "cd_iter":
        ap.error("--cd-kwargs / --entry-offset require --chain cd_iter")
    if args.cd_kwargs:
        cd_kwargs = json.loads(args.cd_kwargs)
        unknown = sorted(set(cd_kwargs) - {f.name for f in dc_fields(VKConfig)})
        if unknown:
            ap.error(f"--cd-kwargs unknown VKConfig fields: {unknown}")
        globals()["CD_KWARGS"] = cd_kwargs
        print(f"cd_iter VKConfig overrides: {cd_kwargs}")
    if args.entry_offset != 0.0:
        globals()["ENTRY_OFFSET"] = float(args.entry_offset)
        print(f"cd_iter entry offset: {args.entry_offset} rev/s")

    pk_kwargs = dict(DEFAULT_PK)
    if args.pk_kwargs:
        if args.chain != "pk_custom":
            ap.error("--pk-kwargs requires --chain pk_custom")
        user = json.loads(args.pk_kwargs)
        pk_kwargs.update({k: tuple(v) if isinstance(v, list) else v for k, v in user.items()})

    jobs: list[tuple[str, Prepared, np.ndarray, dict[str, Any]]] = []
    for w in [s.strip() for s in args.windows.split(",") if s.strip()]:
        if w == "dregon_ramp":
            jobs.append(("dregon_ramp", *real_window("free-flight_nosource_room1", 0)))
        elif w == "fly124_cruise":
            jobs.append(("fly124_cruise", *real_window("FLY124", 3)))
        elif w.startswith("real:"):
            # real:<recording_id>:<window_index> — any window in the beatvk
            # prep cache (the production scoreboard set), for regression checks
            # beyond the two named windows.
            _, rid, widx = w.split(":")
            jobs.append((f"{rid}_w{int(widx):02d}", *real_window(rid, int(widx))))
        elif w == "synth":
            for i in range(args.synth_battery):
                seed = args.synth_base_seed + i
                aggr = AGGR_CYCLE[i % len(AGGR_CYCLE)]
                jobs.append((f"synth{i:02d}", *synth_window(seed, aggr)))
        elif w == "synth_trace":
            jobs.append(("synth_trace", *synth_window(99, 1.0, mode_means=TRACE_MODES)))
        elif w in ("synthbl", "synthbl_hi") or (w.startswith("synthbl") and w[7:].isdigit()):
            snr = 20.0 if w == "synthbl_hi" else args.synth_snr
            tag = "synthblhi" if w == "synthbl_hi" else "synthbl"
            idxs = [int(w[7:])] if w[7:].isdigit() else list(range(args.synth_battery))
            for i in idxs:
                jobs.append(
                    (
                        f"{tag}{i:02d}",
                        *synth_window(
                            args.synth_base_seed + i,
                            AGGR_CYCLE[i % len(AGGR_CYCLE)],
                            fc_hz=args.synth_fc,
                            snr_db=snr,
                        ),
                    )
                )
        elif w == "synthbl_trace":
            jobs.append(
                (
                    "synthbl_trace",
                    *synth_window(
                        99, 1.0, mode_means=TRACE_MODES, fc_hz=args.synth_fc, snr_db=args.synth_snr
                    ),
                )
            )
        elif w.startswith("synth") and w[5:].isdigit():
            i = int(w[5:])
            jobs.append(
                (
                    f"synth{i:02d}",
                    *synth_window(args.synth_base_seed + i, AGGR_CYCLE[i % len(AGGR_CYCLE)]),
                )
            )
        else:
            ap.error(f"unknown window {w!r}")

    results: dict[str, Any] = {}
    for name, prep, weights, meta in jobs:
        results[name] = run_chain(
            name,
            prep,
            weights,
            meta,
            args.chain,
            pk_kwargs,
            args.pk_repeat,
            args.skip_capture,
            not args.no_seed_cache,
            not args.no_oracle,
            args.v2_rounds,
            seed_cfg,
            m3_pool,
            m3_ref,
        )

    # -- summary table
    print(f"\n===== summary (chain={args.chain}) =====")
    hdr = f"{'window':<14s} {'stage':<14s} {'pooled':>7s}  per-rotor mae | bias | corr | std-ratio"
    print(hdr)
    for name, res in results.items():
        for m in res["stages"]:
            pr = m["per_rotor"]

            def fmt(key: str, rows: list[dict[str, Any]] = pr) -> str:
                return "/".join("--" if r[key] is None else f"{r[key]:.2f}" for r in rows)

            print(
                f"{name:<14s} {m['stage']:<14s} {m['pooled_mae']:7.3f}  "
                f"{fmt('mae')} | {fmt('bias')} | {fmt('shape_corr')} | {fmt('std_ratio')}"
            )
        orc = res.get("oracle_floor")
        if orc:
            pr = orc["per_rotor"]

            def ofmt(key: str, rows: list[dict[str, Any]] = pr) -> str:
                return "/".join("--" if r[key] is None else f"{r[key]:.2f}" for r in rows)

            print(
                f"{name:<14s} {'ORACLE_FLOOR':<14s} {orc['pooled_mae']:7.3f}  "
                f"{ofmt('mae')} | {ofmt('bias')} | {ofmt('shape_corr')} | {ofmt('std_ratio')}"
            )

    # -- pooled numbers
    print(f"\n{'window':<14s} {'final':>7s} {'oracle':>7s} {'wall_s':>8s}")
    for name, res in results.items():
        orc = res.get("oracle_floor")
        o = f"{orc['pooled_mae']:7.3f}" if orc else "     --"
        print(f"{name:<14s} {res['final_pooled_mae']:7.3f} {o} {res['meta']['wall_s']:8.1f}")
    pooled_final = [res["final_pooled_mae"] for res in results.values()]
    print(
        f"\nfinal pooled PIT-MAE mean over {len(pooled_final)} windows: "
        f"{float(np.mean(pooled_final)):.3f}"
    )
    orcs = [
        res["oracle_floor"]["pooled_mae"] for res in results.values() if res.get("oracle_floor")
    ]
    if orcs:
        print(f"oracle-floor mean over {len(orcs)} synthetic windows: {float(np.mean(orcs)):.3f}")
    # Pooled per-rotor decomposition of the FINAL stage (mean over windows x
    # rotors; rotor identity is not comparable across windows, so |bias| is
    # reported alongside the signed mean).
    rows = [q for res in results.values() for q in res["stages"][-1]["per_rotor"]]
    corrs = [q["shape_corr"] for q in rows if q["shape_corr"] is not None]
    print(
        f"pooled final per-rotor: bias {float(np.mean([q['bias'] for q in rows])):+.3f}  "
        f"|bias| {float(np.mean([abs(q['bias']) for q in rows])):.3f}  "
        f"shape_corr {float(np.mean(corrs)):.3f}  "
        f"std_ratio {float(np.mean([q['std_ratio'] for q in rows])):.3f}  "
        f"(n={len(rows)})"
    )
    print(
        f"wall_s per window: mean {float(np.mean([r['meta']['wall_s'] for r in results.values()])):.1f}"
    )

    # -- baseline verification vs the trace numbers
    if args.chain == "baseline":
        for name, ref in BASELINE_REF.items():
            if name not in results:
                continue
            got_ft = results[name]["final_pooled_mae"]
            got_tg = results[name]["final_pooled_mae_tgrid"]
            ok_ft = abs(got_ft - ref["ft"]) <= REF_TOL
            ok_tg = abs(got_tg - ref["tgrid"]) <= REF_TOL
            print(
                f"[verify] {name}: frame-grid {got_ft:.3f} vs trace {ref['ft']:.3f} "
                f"-> {'PASS' if ok_ft else 'FAIL'}; tgrid {got_tg:.3f} vs doc "
                f"{ref['tgrid']:.3f} -> {'PASS' if ok_tg else 'FAIL'}"
            )

    out = (
        Path(args.out)
        if args.out
        else LAB_OUT / (f"{args.chain}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r3({"chain": args.chain, "windows": results}), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
