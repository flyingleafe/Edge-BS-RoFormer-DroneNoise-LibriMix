#!/usr/bin/env python3
"""joint_beam diagnosis: is the OBJECTIVE wrong, or is the SEARCH losing?

The joint 4-rotor beam search (WP19) individuates the rotors — per-rotor
std_ratio 1.34 vs the coarse stage's 0.33, shape-correlation spread 0.79 vs
0.10 — and is still ~9x worse than the stage it replaces.  That combination has
exactly two explanations and they need opposite fixes, so nothing should be
tuned until they are told apart:

``--mode cost``
    Evaluate the tracker's OWN cost at the ground-truth trajectory and at every
    competing stage's output (:func:`joint_beam_tracker.score_trajectory`).
    A search and its objective fail in opposite directions:

    - ``cost(GT) < cost(beam output)`` -> the objective is right and the beam
      threw the answer away.  Fix the search: proposals, beam width, diversity.
    - ``cost(GT) > cost(beam output)`` -> the tracker found what it was asked
      for and the ask is wrong.  No search budget can help.

    Reported per term (emission / transition / band) and per frame, so a mixed
    verdict localises to the frames and the term that disagree.

``--mode ceiling``
    The upstream question: can a single rotor be LOCALISED at all by this
    emission?  For every (rotor, frame) it asks whether the true speed is a
    local maximum of the single-comb surface, how far the nearest local maximum
    is, and whether the true speed survives into the ``n_peaks`` shortlist the
    beam actually proposes from.  Swept over analysis configurations
    (``n_fft``, ``k_max``, ``b0_rps``), because the shipped emission reads only
    ``k <= 8`` off 7.8 Hz bins: two rotors 0.89 rev/s apart (FLY124's twin
    pair) put their k=8 teeth 7.1 Hz apart, i.e. inside one bin, so the
    surface CANNOT resolve them and no tracker reading it can either.

Both modes are per-window and restartable; one JSON per (mode, window) unit.

Cluster::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 24 --time 2h -- \\
        python scripts/jb_probe.py --mode cost --jobs 16
    omnirun submit --backend uni-gpushort --gpus 1 --time 1h -- \\
        python scripts/jb_probe.py --mode ceiling --device cuda --jobs 1
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402

#: The window set both modes run on: the two lab reference windows (one DREGON
#: ramp, one FLY124 cruise) plus a steady DREGON, a second FLY124 cruise and the
#: synthetic trace, so a verdict cannot come from one regime.
WINDOWS = (
    "real:free-flight_nosource_room1:0",  # dregon ramp
    "real:free-flight_nosource_room1:1",  # dregon steady
    "real:free-flight_speech-low_room1:1",  # dregon steady, speech present
    "real:FLY124:3",  # fly124 cruise (comb-invisible rotor)
    "real:FLY124:4",  # fly124 cruise
    "synth_trace",  # four distinct shapes, known truth
)

#: Emission configurations for `--mode ceiling`.  `n_fft` is NOT an EmissionCfg
#: field — it selects the spectrogram the tables are built on.
CEILING_CFGS: dict[str, dict[str, Any]] = {
    "shipped": {"n_fft": 2048, "k_max": 8, "b0_rps": 0.0},
    "k16": {"n_fft": 2048, "k_max": 16, "b0_rps": 0.0},
    "k30": {"n_fft": 2048, "k_max": 30, "b0_rps": 0.0},
    "nfft4096_k16": {"n_fft": 4096, "k_max": 16, "b0_rps": 0.0},
    "nfft4096_k30": {"n_fft": 4096, "k_max": 30, "b0_rps": 0.0},
    "nfft4096_k50": {"n_fft": 4096, "k_max": 50, "b0_rps": 0.0},
    "nfft8192_k30": {"n_fft": 8192, "k_max": 30, "b0_rps": 0.0},
    "nfft8192_k50": {"n_fft": 8192, "k_max": 50, "b0_rps": 0.0},
    "nfft8192_k50_b025": {"n_fft": 8192, "k_max": 50, "b0_rps": 0.25},
    "nfft4096_k30_step025": {"n_fft": 4096, "k_max": 30, "b0_rps": 0.0, "step": 0.25},
    # --- POOLING.  The first sweep showed resolution buys at most ~1.6x while
    # the objective ranks the ground truth DEARER than a 27 rev/s-error track,
    # so the defect is in the score function, not the analysis.  A weighted MEAN
    # over teeth rewards any loud content at multiples of c; a quantile or a
    # positive-fraction demands that MOST predicted teeth are actually present,
    # which is the property that separates a rotor from a coincidence.
    "pool_frac_k16": {"n_fft": 4096, "k_max": 16, "b0_rps": 0.0, "pool": "frac_pos"},
    "pool_frac_k30": {"n_fft": 4096, "k_max": 30, "b0_rps": 0.0, "pool": "frac_pos"},
    "pool_frac_k50": {"n_fft": 4096, "k_max": 50, "b0_rps": 0.0, "pool": "frac_pos"},
    "pool_q25_k16": {"n_fft": 4096, "k_max": 16, "b0_rps": 0.0, "pool": "quantile"},
    # --- ABSOLUTE normalisation.  The gate run closed 40-70% of the objective
    # gap with quantile pooling and flipped one window, and the residual is
    # entirely emission: the tracker still claims MORE distinct comb mass than
    # the truth (3.80-3.86 vs 2.90-3.77), because "peak" normalisation makes
    # the best comb of every frame score 1.0 whether or not a rotor is there.
    "mad_q25_k16": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "norm": "mad",
    },
    "mad_mean_k16": {"n_fft": 4096, "k_max": 16, "b0_rps": 0.0, "norm": "mad"},
    # --- claim_q: the PRICE of claiming a comb.  The union emission is
    # (u - mass*ref)/denom; `norm` moves denom (a pure scale, measurably unable
    # to reorder anything) while `claim_q` moves ref, which is subtracted once
    # per claimed comb.  This is the only parameter that can stop the tracker
    # buying a fourth comb that is not there.
    "q25_claim75": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "claim_q": 0.75,
    },
    "q25_claim90": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "claim_q": 0.90,
    },
    "q25_claim95": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "claim_q": 0.95,
    },
    "q25_claim98": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "claim_q": 0.98,
    },
    # --- THE DECISIVE PAIR.  claim_q prices a comb correctly but cannot reorder
    # anything while every reachable assignment carries the same comb mass, so
    # the price and the proposal set have to move together.  Both halves are
    # run separately so a win cannot be attributed to the wrong one.
    "q25_shared": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "beam": {"allow_shared_peaks": True},
    },
    "q25_claim90_shared": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "claim_q": 0.90,
        "beam": {"allow_shared_peaks": True},
    },
    "q25_claim95_shared": {
        "n_fft": 4096,
        "k_max": 16,
        "b0_rps": 0.0,
        "pool": "quantile",
        "claim_q": 0.95,
        "beam": {"allow_shared_peaks": True},
    },
    "mad_q25_k30": {
        "n_fft": 4096,
        "k_max": 30,
        "b0_rps": 0.0,
        "pool": "quantile",
        "norm": "mad",
    },
    "pool_q25_k30": {"n_fft": 4096, "k_max": 30, "b0_rps": 0.0, "pool": "quantile"},
    "pool_q50_k30": {
        "n_fft": 4096,
        "k_max": 30,
        "b0_rps": 0.0,
        "pool": "quantile",
        "pool_q": 0.5,
    },
    "pool_q25_k50": {"n_fft": 4096, "k_max": 50, "b0_rps": 0.0, "pool": "quantile"},
    "pool_frac_k30_step025": {
        "n_fft": 4096,
        "k_max": 30,
        "b0_rps": 0.0,
        "pool": "frac_pos",
        "step": 0.25,
    },
    # --- temporal integration: the truth is stable over ~0.1-0.3 s, so a
    # per-frame score throws away evidence the tracker has no other way to get.
    "pool_frac_k30_sm9": {
        "n_fft": 4096,
        "k_max": 30,
        "b0_rps": 0.0,
        "pool": "frac_pos",
        "smooth_frames": 9,
    },
    "mean_k16_sm9": {"n_fft": 4096, "k_max": 16, "b0_rps": 0.0, "smooth_frames": 9},
}

#: Localisation tolerances reported by `--mode ceiling`, rev/s.
TOLS = (0.25, 0.5, 1.0, 2.0)


# --------------------------------------------------------------------------
# shared helpers


def load_window(window: str) -> tuple[Any, dict[str, Any]]:
    """``(prep, meta)`` for a window spec, using the lab's own builders."""
    import rps_refine_lab as lab

    if window.startswith("real:"):
        _, rid, widx = window.split(":")
        prep, _weights, meta = lab.real_window(rid, int(widx))
        del _weights
        return prep, meta
    if window == "synth_trace":
        prep, _weights, meta = lab.synth_window(99, 1.0, mode_means=lab.TRACE_MODES)
        return prep, meta
    if window.startswith("synthbl"):
        i = int(window[7:])
        prep, _weights, meta = lab.synth_window(
            100 + i, lab.AGGR_CYCLE[i % 3], fc_hz=8.0, snr_db=0.0
        )
        return prep, meta
    i = int(window[5:])
    prep, _weights, meta = lab.synth_window(100 + i, lab.AGGR_CYCLE[i % 3])
    return prep, meta


def whitened_spec(audio: np.ndarray, n_fft: int) -> tuple[np.ndarray, float, np.ndarray]:
    """``_coarse_spec`` with the FFT size as a parameter.

    Same whitening (running median over frequency at ``SEED_CFG.whiten_hz``,
    then the channel mean), same 32 ms hop — only ``n_fft`` moves, so a ceiling
    measured here is comparable to the shipped emission at ``n_fft = 2048``.
    """
    from beatvk_vk_arms import SEED_CFG, SR
    from scipy.ndimage import median_filter

    from data_processing.rps_refinement import RefineConfig, compute_logmag

    spec = compute_logmag(audio, RefineConfig(sample_rate=SR, n_fft=n_fft, device="cpu"))
    lm_raw = spec.logmag.cpu().numpy()
    bin_hz = float(spec.bin_hz)
    win = int(round(SEED_CFG.whiten_hz / bin_hz)) | 1
    white = (lm_raw - median_filter(lm_raw, size=(1, win, 1))).mean(axis=0)
    return white, bin_hz, np.asarray(spec.frame_times, dtype=np.float64)


def gt_on(prep: Any, st: np.ndarray) -> np.ndarray:
    """``(4, len(st))`` telemetry resampled onto a spectrogram frame grid."""
    return np.stack([np.interp(st, prep.ft, row) for row in prep.r_meas])


def unit_path(results: Path, mode: str, window: str) -> Path:
    return results / "raw" / f"{mode}__{window.replace(':', '_')}.json"


# --------------------------------------------------------------------------
# mode: cost


def cost_unit(task: tuple[str, Path, str, str]) -> tuple[str, str]:
    """Objective value at GT vs at every stage's output, on one window."""
    window, results, device, cfg_name = task
    out = unit_path(results, "cost", window)
    if out.exists():
        return window, "skip"
    tic = time.perf_counter()
    try:
        import rps_refine_lab as lab
        from beatvk_vk_arms import _coarse_spec, fullrange_init

        from data_processing.joint_beam_tracker import (
            BeamCfg,
            EmissionCfg,
            OUPrior,
            build_objective,
            joint_beam_track,
            score_trajectory,
        )

        prep, meta = load_window(window)
        # `--cfg` selects a CEILING_CFGS entry so the cost gate is evaluated on
        # the SAME emission the ceiling sweep ranked; `n_fft` is not an
        # EmissionCfg field, it picks the spectrogram underneath.
        kw = dict(CEILING_CFGS[cfg_name])
        n_fft = int(kw.pop("n_fft"))
        # A cfg entry may carry a `beam` sub-dict: the objective is not only the
        # emission, and the proposal set is now the thing under test.
        beam = BeamCfg(**kw.pop("beam", {}))
        emis = EmissionCfg(**kw)
        lm, bin_hz, st = (
            _coarse_spec(prep.audio)[:3] if n_fft == 2048 else whitened_spec(prep.audio, n_fft)
        )
        obj = build_objective(lm, bin_hz, ou=OUPrior(), emis=emis, beam=beam, device=device)

        trajs: dict[str, np.ndarray] = {
            "gt": gt_on(prep, st),
            # DREGON's reciprocal-period label noise costs the TRUE trajectory
            # 1100-1400 transition units (WP20), which is roughness the tracker
            # is not meant to reproduce.  Scoring the smoothed telemetry too
            # keeps the comparison from flattering the tracker.
            "gt_smooth": np.stack([np.interp(st, prep.ft, r) for r in prep.r_meas_sm]),
        }
        r_jb, jb_diag = joint_beam_track(
            lm,
            bin_hz,
            st,
            prep.ft,
            ou=OUPrior(),
            emis=EmissionCfg(),
            beam=BeamCfg(),
            device=device,
        )
        trajs["joint_beam"] = np.stack([np.interp(st, prep.ft, row) for row in r_jb])
        try:
            _, rid, widx = window.split(":")
            seed = lab.get_seed(f"{rid}_w{int(widx):02d}", prep, True)
            r_fri = fullrange_init(prep, seed)[0]
            trajs["fullrange_init"] = np.stack([np.interp(st, prep.ft, row) for row in r_fri])
        except ValueError:  # synthetic windows have no `real:` name to cache under
            r_fri = None

        rec: dict[str, Any] = {
            "window": window,
            "emis_cfg": cfg_name,
            "regime": meta.get("regime"),
            "jb_diag": jb_diag,
            "n_spec_frames": int(len(st)),
            "arms": {},
        }
        for key, w in trajs.items():
            sc = score_trajectory(obj, w)
            pf = sc.pop("per_frame")
            # PIT-MAE of the SAME trajectory, on the metric grid, so cost and
            # accuracy are read off one row and cannot be paired up wrongly.
            on_ft = np.stack([np.interp(prep.ft, st, row) for row in w])
            sc.update(lab.stage_metrics(on_ft, prep))
            sc["per_frame_total"] = (pf["emission"] + pf["transition"] + pf["band"]).tolist()
            rec["arms"][key] = sc

        # The decisive comparison, precomputed so a reader cannot mis-derive it.
        c_gt = rec["arms"]["gt"]["total"]
        c_jb = rec["arms"]["joint_beam"]["total"]
        rec["verdict"] = "search" if c_gt < c_jb else "objective"
        rec["cost_gt_minus_jb"] = c_gt - c_jb
        if "fullrange_init" in rec["arms"]:
            rec["cost_fri_minus_jb"] = rec["arms"]["fullrange_init"]["total"] - c_jb
        rec["wall_s"] = round(time.perf_counter() - tic, 1)
        _write(out, lab.r3(rec))
        return window, "ok"
    except Exception:  # noqa: BLE001 - one bad unit must not kill the probe
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(traceback.format_exc())
        return window, "ERROR"


# --------------------------------------------------------------------------
# mode: ceiling


def _peak_stats(
    surf: np.ndarray, grid: np.ndarray, gt: np.ndarray, n_peaks: int, half_local: float
) -> dict[str, Any]:
    """Per-rotor localisability of ``gt`` on one ``(D, T)`` score surface.

    Four numbers, each answering a different failure:

    - ``argmax_err``: distance from the truth to the best grid point in an
      ISOLATED neighbourhood of it — half-width ``min(3, 0.45 * gap)`` where
      ``gap`` is the distance to the nearest sibling's truth at that frame.
      The isolation matters: a fixed +-3 rev/s window on a window whose rotors
      sit 2.0 rev/s apart contains the NEIGHBOUR's peak, so the metric would
      report a rotor as unlocalisable whenever its sibling is simply louder.
    - ``peak_err``: distance to the nearest strict local maximum.  A rotor whose
      truth sits on a FLANK (the twin case) has a small ``argmax_err`` and a
      large ``peak_err`` — and the beam proposes maxima, so the flank is
      invisible to it.
    - ``in_shortlist``: whether a local maximum within 0.5 rev/s of the truth
      survives into the top-``n_peaks`` the beam actually proposes from.  The
      hard upper bound on ACQUISITION for any beam reading this surface.
    - ``step_hit``: the TRACKING ceiling, which acquisition numbers do not
      bound.  Placed at the truth at ``t-1`` with no prior and no history, take
      the best grid point within ``local_half_rps`` — the beam's own local move
      — and ask whether it lands within tolerance of the truth at ``t``.  A
      rotor can be tracked through frames where it is not proposable, so this
      is the number that says whether per-frame evidence supports continuity.
    """
    n_r, n_t = gt.shape
    step = float(grid[1] - grid[0])
    out: dict[str, Any] = {"per_rotor": []}
    for r in range(n_r):
        argmax_err = np.full(n_t, np.nan)
        peak_err = np.full(n_t, np.nan)
        step_err = np.full(n_t, np.nan)
        shortlist = np.zeros(n_t, dtype=bool)
        others = [q for q in range(n_r) if q != r]
        for t in range(n_t):
            g = gt[r, t]
            if not np.isfinite(g) or g < grid[0] or g > grid[-1]:
                continue
            s = surf[:, t]
            gap = float(np.min(np.abs(gt[others, t] - g))) if others else np.inf
            half = min(3.0, max(0.45 * gap, step))
            lo = int(max(0, np.searchsorted(grid, g - half)))
            hi = int(min(len(grid), np.searchsorted(grid, g + half) + 1))
            if hi > lo:
                argmax_err[t] = abs(grid[lo + int(np.argmax(s[lo:hi]))] - g)
            if t > 0 and np.isfinite(gt[r, t - 1]):
                p = gt[r, t - 1]
                plo = int(max(0, np.searchsorted(grid, p - half_local)))
                phi = int(min(len(grid), np.searchsorted(grid, p + half_local) + 1))
                if phi > plo:
                    step_err[t] = abs(grid[plo + int(np.argmax(s[plo:phi]))] - g)
            ismax = np.zeros(len(s), dtype=bool)
            ismax[1:-1] = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:])
            pk = np.flatnonzero(ismax)
            if len(pk) == 0:
                continue
            d = np.abs(grid[pk] - g)
            peak_err[t] = float(d.min())
            # non-maximum suppression at the beam's own `peak_sep_rps`, then the
            # top-n_peaks by score: exactly `_frame_peaks`' first pass
            order = pk[np.argsort(-s[pk])]
            keep: list[int] = []
            for i in order:
                if len(keep) >= n_peaks:
                    break
                if all(abs(grid[i] - grid[j]) >= 0.2 for j in keep):
                    keep.append(int(i))
            shortlist[t] = bool(keep) and float(np.abs(grid[keep] - g).min()) <= 0.5
        ok = np.isfinite(argmax_err)
        ok_s = np.isfinite(step_err)
        row: dict[str, Any] = {
            "rotor": r,
            "n_frames_scored": int(ok.sum()),
            "argmax_err_median": float(np.nanmedian(argmax_err)) if ok.any() else None,
            "peak_err_median": float(np.nanmedian(peak_err)) if ok.any() else None,
            "step_err_median": float(np.nanmedian(step_err)) if ok_s.any() else None,
            "in_shortlist_frac": float(shortlist[ok].mean()) if ok.any() else None,
            "grid_step": step,
        }
        for tol in TOLS:
            row[f"step_hit_{tol}"] = (
                float(np.nanmean(step_err[ok_s] <= tol)) if ok_s.any() else None
            )
        for tol in TOLS:
            row[f"argmax_hit_{tol}"] = (
                float(np.nanmean(argmax_err[ok] <= tol)) if ok.any() else None
            )
            row[f"peak_hit_{tol}"] = float(np.nanmean(peak_err[ok] <= tol)) if ok.any() else None
        out["per_rotor"].append(row)

    def worst(key: str) -> float | None:
        vals = [q[key] for q in out["per_rotor"] if q[key] is not None]
        return float(min(vals)) if vals else None

    out["argmax_hit_0.5_worst"] = worst("argmax_hit_0.5")
    out["in_shortlist_worst"] = worst("in_shortlist_frac")
    out["step_hit_0.5_worst"] = worst("step_hit_0.5")
    out["step_hit_0.25_worst"] = worst("step_hit_0.25")
    return out


def ceiling_unit(task: tuple[str, Path, str, str]) -> tuple[str, str]:
    """Emission ceiling of every :data:`CEILING_CFGS` entry on one window."""
    window, results, device, _cfg = task
    out = unit_path(results, "ceiling", window)
    if out.exists():
        return window, "skip"
    tic = time.perf_counter()
    try:
        import rps_refine_lab as lab
        import torch

        from data_processing.joint_beam_tracker import (
            BeamCfg,
            EmissionCfg,
            _smooth_frames,
            comb_scores_from_tables,
            comb_tables,
        )

        prep, meta = load_window(window)
        rec: dict[str, Any] = {"window": window, "regime": meta.get("regime"), "cfgs": {}}
        n_peaks = BeamCfg().n_peaks
        half_local = BeamCfg().local_half_rps
        specs: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
        for name, kw in CEILING_CFGS.items():
            kw = dict(kw)
            n_fft = int(kw.pop("n_fft"))
            kw.pop("beam", None)  # ceiling mode measures the emission only
            if n_fft not in specs:
                specs[n_fft] = whitened_spec(prep.audio, n_fft)
            lm, bin_hz, st = specs[n_fft]
            emis = EmissionCfg(**kw)
            lm_t = torch.as_tensor(lm, device=device, dtype=torch.float32)
            grid_t = torch.as_tensor(emis.grid(), device=device, dtype=torch.float32)
            surf = _smooth_frames(
                comb_scores_from_tables(comb_tables(lm_t, bin_hz, emis, grid_t), emis), emis
            )
            stats = _peak_stats(
                surf.cpu().numpy(), emis.grid(), gt_on(prep, st), n_peaks, half_local
            )
            stats["n_fft"] = n_fft
            stats["bin_hz"] = round(bin_hz, 3)
            rec["cfgs"][name] = stats
        # The twin separation this window actually asks for, so a ceiling can be
        # read against the physics rather than in the abstract.
        means = np.sort(prep.r_meas.mean(axis=1))
        rec["rotor_means"] = [round(float(v), 3) for v in means]
        rec["min_pair_sep_rps"] = round(float(np.min(np.diff(means))), 3)
        rec["wall_s"] = round(time.perf_counter() - tic, 1)
        _write(out, lab.r3(rec))
        return window, "ok"
    except Exception:  # noqa: BLE001
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(traceback.format_exc())
        return window, "ERROR"


# --------------------------------------------------------------------------


def _write(out: Path, payload: Any) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, out)


def summarise(results: Path, mode: str) -> dict[str, Any]:
    rows = [json.loads(f.read_text()) for f in sorted((results / "raw").glob(f"{mode}__*.json"))]
    if mode == "cost":
        table = []
        for r in rows:
            row = {"window": r["window"], "verdict": r["verdict"]}
            for key, sc in r["arms"].items():
                row[f"{key}_cost"] = round(sc["total"], 1)
                row[f"{key}_mae"] = round(sc["pooled_mae"], 3)
            table.append(row)
        n_search = sum(1 for r in rows if r["verdict"] == "search")
        return {"n_windows": len(rows), "n_verdict_search": n_search, "table": table}
    table = []
    for r in rows:
        for name, st in r["cfgs"].items():
            table.append(
                {
                    "window": r["window"],
                    "cfg": name,
                    "min_pair_sep_rps": r["min_pair_sep_rps"],
                    "argmax_hit_0.5_worst": st["argmax_hit_0.5_worst"],
                    "in_shortlist_worst": st["in_shortlist_worst"],
                    "step_hit_0.5_worst": st["step_hit_0.5_worst"],
                    "peak_err_median": [q["peak_err_median"] for q in st["per_rotor"]],
                }
            )
    return {"n_windows": len(rows), "table": table}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("cost", "ceiling"), required=True)
    ap.add_argument("--windows", nargs="*", default=list(WINDOWS))
    ap.add_argument("--results", default="results/jb_probe")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--cfg",
        default="shipped",
        choices=sorted(CEILING_CFGS),
        help="cost mode: which emission configuration to evaluate the objective on",
    )
    ap.add_argument(
        "--build-preps",
        action="store_true",
        help="materialise the beat-VK prep cache first.  REQUIRED on a cluster: "
        "`results/beatvk_vk_arms/prep_cache` is a gitignored local artefact, so a "
        "fresh worktree has no windows to score and every unit dies on "
        "FileNotFoundError — silently, because per-unit exceptions become .err files.",
    )
    args = ap.parse_args()

    if args.build_preps:
        import beatvk_rescore as brs
        from beatvk_vk_arms import DEFAULT_OUT

        brs.build_prep_cache(Path(DEFAULT_OUT), None, brs.resolve_dregon_dir())

    results = Path(args.results)
    fn = cost_unit if args.mode == "cost" else ceiling_unit
    tasks = [(w, results, args.device, args.cfg) for w in args.windows]
    if args.jobs <= 1:
        for t in tasks:
            print(fn(t), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(fn, t) for t in tasks]
            for f in as_completed(futs):
                print(f.result(), flush=True)

    summary = summarise(results, args.mode)
    (results / f"summary_{args.mode}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    n_err = len(list((results / "raw").glob(f"{args.mode}__*.err")))
    if n_err:
        print(f"!! {n_err} unit(s) failed — see {results}/raw/*.err", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
