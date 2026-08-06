"""Fit the rotors to the harmonics: the telemetry refitter (GitHub issue 17, phase 6b).

Phase 6a built the JUDGE (:mod:`tracking.fitness`). This module is the thing it
judges — the procedure of issue 17 § "Proposed procedure", all six steps, wiring
the pieces that already exist rather than growing a second copy of any of them.

The six steps, and where each one lives here
--------------------------------------------
1. **Pre-smooth the telemetry carrier** (:func:`presmooth`). The tachometer's
   0.269 rev/s / 49.7 Hz staircase is measurement noise; left in the carrier it
   is residual the demodulation band has to accommodate — 19 Hz of it at k = 70.
   The filter is ``tracking.phase_noise.brickwall``, the SAME low pass the 6a
   driver's ``lp:5`` candidate uses, so "the smoothed telemetry" means one thing
   in both halves of the campaign.
2. **Coarse-to-fine in k** (:func:`k_cap_for_error`, :func:`advance_k`). Each
   outer iteration runs at ONE harmonic cap, and the cap is a function of the
   current error estimate, never a constant. The rule is written against the
   quantity that actually goes wrong at high k: at harmonic ``k`` a residual
   rate error ``e`` turns into a phase increment ``2 pi k e / fs_env`` per
   envelope sample, and the tracker discards increments beyond
   ``wrap_guard_rad``. So the next rung is admissible while
   ``k <= wrap_guard_rad fs_env / (2 pi e)``. (The k-scaled band itself is NOT
   the binding constraint — its capture is ``b0`` rev/s at every ``k`` by
   construction, which is the whole point of the k-scaled shape.) The campaign's
   ``k_caps=(80, 80, 80)`` is what this replaces: it put the decision on
   out-of-capture harmonics weighted ``k^2`` and let them outvote the in-capture
   low ones.
3. **Alternate with the envelope solve and the peel.** One outer iteration is
   exactly one :func:`tracking.pipelines.pi_kalman_arm_stage` application — VK
   envelopes at the current track, the peel, then one ``pi_kalman_refine`` pass
   through the tracker's peel seam. The flagship's alternation, at the
   displacement campaign's settings, is precisely what issue 17 says was never
   run.
4. **Least-squares-projected subtraction.** ``peel_mode="ls"`` is
   :data:`tracking.pipelines.DEFAULT_PEEL_MODE`, and this module keeps it:
   fitting a complex gain per harmonic per block makes the residual
   ``<= ||y||^2`` by construction, so a peel cannot inject energy.
5. **Stop on convergence, not on a fixed count** (:attr:`RefitConfig.tol_rev_s`,
   :attr:`RefitConfig.max_iters`). The loop ends when the largest trajectory
   update inside the trimmed window falls below the tolerance, and every
   iteration's delta is recorded, so a run that hit the cap says so
   (:attr:`RefitResult.stop_reason`). A second stop is there because the
   measurement demanded it: on real audio the MAXIMUM update stays near
   1 rev/s — isolated tracker spikes — long after the trajectory has stopped
   moving in bulk, so the alternation also stops when the 95th-percentile
   update stops improving. Only the tolerance stop is reported as convergence.
6. **Twins.** ``pair_mode="joint"`` is kept for the tight pairs, and the peel's
   twin rule is the existing one, unchanged and asserted rather than rebuilt:
   :func:`tracking.pipelines.make_peels` builds ``pair_audio[(lo, hi)]`` by
   subtracting only the NON-pair rotors, so a sibling's reconstruction is never
   subtracted from the two-tone observation that estimates the pair.

What this module deliberately does not do
-----------------------------------------
It does not score itself. A fitted trajectory has more freedom than fixed
telemetry and will fit better whether or not it is more correct — that is the
premise of the whole issue — so the verdict comes from
:mod:`tracking.fitness`, at fixed degrees of freedom, with its four controls.
The only fitness call here is :func:`tracking.fitness.residual_decompose`,
which is not a score: it is the scale/lag/tachometer-signature reading of
``fit - raw telemetry``, reported so a caller never has to collapse the two
distinct corrections (a small systematic shift and a large de-staircasing) into
one number.

Purity: numpy plus tracking siblings only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tracking.fitness import FitnessConfig, residual_decompose
from tracking.phase_noise import brickwall
from tracking.pipelines import DEFAULT_PEEL_MODE, PEEL_BW_HZ, PEEL_K_MAX, pi_kalman_arm_stage
from tracking.stages import get_rps, tracking_frame

__all__ = [
    "RefitConfig",
    "RefitResult",
    "advance_k",
    "k_cap_for_error",
    "order_and_gaps",
    "presmooth",
    "refit_stage",
    "refit_window",
    "scale_summary",
]


@dataclass(frozen=True)
class RefitConfig:
    """Every knob of the procedure, with the reason for each default.

    The defaults are the displacement campaign's best-behaved arm — k-scaled
    band, ``b0 = 1`` rev/s, telemetry init, ``pair_mode`` for the twins — plus
    the three things issue 17 says that arm was missing: the pre-smoothing, the
    coarse-to-fine ladder, and the peel.
    """

    # --- step 1: the carrier -------------------------------------------------
    #: Low-pass cutoff (Hz) applied to telemetry before it becomes the init.
    #: 5 Hz keeps the shaft dynamics (DREGON free-flight modes have tau
    #: 0.6-1.0 s) and removes the 49.7 Hz refresh staircase.
    smooth_cut_hz: float = 5.0

    # --- step 2: the k ladder ------------------------------------------------
    #: Prior error before the first iteration (rev/s). The issue's own range for
    #: the bias is 0.2-0.8 % of ~80 rev/s = 0.2-0.7 rev/s; 1.5 leaves margin for
    #: the staircase and the flight wander on top.
    e0_rev_s: float = 1.5
    #: Hard cap on the first rung ("start k <= 20").
    k_start_max: int = 20
    #: Ceiling of the ladder ("climb to 80-100").
    k_top: int = 96
    #: Floor of the ladder — below this the comb carries too little information.
    k_min_rung: int = 8
    #: Maximum multiplicative growth of the cap per iteration. Without it a
    #: single good iteration would jump from the first rung to the ceiling.
    k_growth: float = 2.0
    #: Floor of the error estimate (rev/s) — the measured shaft-jitter linewidth
    #: (``pi_kalman_refine``'s ``anneal_w_line``). No rung is ever chosen as if
    #: the trajectory were exact.
    e_floor_rev_s: float = 0.08
    #: Quantile of ``|delta r|`` used as the next error estimate. The maximum is
    #: dominated by isolated de-staircasing spikes; the 95th percentile tracks
    #: how far the trajectory is still moving in bulk.
    step_quantile: float = 0.95

    # --- steps 3, 4, 6: the alternation --------------------------------------
    peel: bool = True
    peel_mode: str = DEFAULT_PEEL_MODE  # "ls" — cannot inject energy
    peel_bw_hz: float = PEEL_BW_HZ
    peel_k_max: int = PEEL_K_MAX
    pair_mode: str = "joint"

    # --- the inner tracker ---------------------------------------------------
    fs_env: float = 62.5
    band_mode: str = "k_scaled"
    #: Capture in rev/s at EVERY harmonic (the identity-preserving shape).
    band_b0: float = 1.0
    band_hz: float = 6.0
    off_comb_hz: float = 11.0
    f_max: float = 7500.0
    wrap_guard_rad: float = 2.8
    #: Inner ``pi_kalman_refine`` iterations per outer application. One, because
    #: the outer loop IS the iteration and each outer step re-peels.
    n_iter_inner: int = 1
    max_step: float = 3.0
    edge_trim_s: float = 0.25
    min_rate: float = 5.0

    # --- step 5: convergence -------------------------------------------------
    #: Stop when ``max |delta r|`` over the trimmed window falls below this.
    tol_rev_s: float = 0.02
    max_iters: int = 8
    #: Never stop before this many applications (the first is the capture).
    min_iters: int = 2
    #: Secondary stop. The tolerance above is on the MAXIMUM update, which on
    #: real audio is dominated by isolated tracker spikes and can sit near
    #: 1 rev/s while the trajectory has stopped moving in bulk. When the
    #: ``step_quantile`` update improves by less than this fraction of the
    #: previous iteration's, the alternation has plateaued and further
    #: applications only spend compute. ``0`` disables it.
    plateau_rel: float = 0.05

    def fitness_cfg(self) -> FitnessConfig:
        """The geometry :func:`tracking.fitness.residual_decompose` reads."""
        return FitnessConfig(edge_trim_s=self.edge_trim_s, min_rate=self.min_rate)


# ---------------------------------------------------------------------------
# step 1: the carrier


def presmooth(r: np.ndarray, ft: np.ndarray, cut_hz: float) -> np.ndarray:
    """Low-pass a trajectory at ``cut_hz`` on its own frame grid (issue 17 step 1).

    THE pre-smoothing of the campaign — the 6a driver's ``lp:`` candidate spec
    calls this function, so "the smoothed telemetry" is one array however it is
    reached. ``cut_hz <= 0`` is the identity, so turning the step off needs no
    second code path.

    The trajectory is **detrended first**. ``tracking.phase_noise.brickwall`` is
    a whole-window FFT filter, so it treats the series as periodic; a window
    whose rate drifts by 1 rev/s end to end therefore carries a step
    discontinuity at the wrap, and the filter rings on it across the whole
    window. Measured on the synthetic window of ``tests/tracking/test_telemetry_refit.py``:
    the bare filter turns a 0.087 rev/s staircase error into 0.112 rev/s — it
    makes the carrier WORSE — while removing the least-squares line first turns
    it into 0.075. Only the line is removed, so nothing the filter is supposed
    to reject is subtracted before it runs.
    """
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    ft = np.asarray(ft, dtype=np.float64)
    if cut_hz <= 0 or ft.size < 4:
        return r.copy()
    fs = 1.0 / float(np.median(np.diff(ft)))
    des = np.stack([ft, np.ones_like(ft)], axis=1)
    coef, *_ = np.linalg.lstsq(des, r.T, rcond=None)  # (2, R)
    trend = (des @ coef).T  # (R, N)
    return np.asarray(brickwall(r - trend, float(cut_hz), fs).real + trend, dtype=np.float64)


# ---------------------------------------------------------------------------
# step 2: the ladder


def k_cap_for_error(e_rev_s: float, cfg: RefitConfig) -> int:
    """Highest harmonic whose line the current error estimate keeps in band.

    A residual rate error ``e`` displaces harmonic ``k`` by ``k e`` Hz, which
    the demodulated envelope carries as a phase increment ``2 pi k e / fs_env``
    per sample. Beyond ``wrap_guard_rad`` the increment is ambiguous and
    ``pi_kalman_refine`` discards it, so the rung is admissible while::

        k <= wrap_guard_rad * fs_env / (2 pi e)

    Clamped to ``[k_min_rung, k_top]``. The k-scaled band does not enter: its
    capture is ``band_b0`` rev/s at every ``k`` by construction, so it cannot be
    the constraint that makes one rung safer than the next.
    """
    e = max(float(e_rev_s), cfg.e_floor_rev_s)
    k = int(np.floor(cfg.wrap_guard_rad * cfg.fs_env / (2.0 * np.pi * e)))
    return int(np.clip(k, cfg.k_min_rung, cfg.k_top))


def advance_k(k_cur: int, e_rev_s: float, cfg: RefitConfig) -> int:
    """The next rung: never lower, never more than ``k_growth`` times higher."""
    want = k_cap_for_error(e_rev_s, cfg)
    return int(min(max(k_cur, want), max(k_cur, int(np.ceil(k_cur * cfg.k_growth)))))


# ---------------------------------------------------------------------------
# identity (the twin-collapse failure mode)


def order_and_gaps(r: np.ndarray) -> tuple[list[int], list[float]]:
    """Rotor order by mean rate (fastest first) and the consecutive gaps, rev/s.

    THE identity test of the displacement campaign
    (``docs/experiments/dregon-comb-displacement.md``). The obvious
    nearest-neighbour test is WRONG here: every rotor moves down together under
    a common-mode scale, so a refined rate lands nearest its neighbour's old
    value and the test reports a collapse that did not happen.
    """
    m = np.asarray(r, dtype=np.float64).mean(axis=1)
    order = [int(i) for i in np.argsort(-m)]
    return order, [round(float(x), 4) for x in -np.diff(m[order])]


# ---------------------------------------------------------------------------
# the scale reading


def scale_summary(
    r_fit: np.ndarray,
    r_ref: np.ndarray,
    ft: np.ndarray,
    *,
    cfg: RefitConfig | None = None,
) -> dict[str, Any]:
    """Rate scale of ``r_fit`` against ``r_ref``, in percent, two well-posed ways.

    This exists beside :func:`tracking.fitness.residual_decompose` because that
    function's per-rotor design matrix ``[r, dr/dt, 1]`` is **near-degenerate on
    a cruise window**: a rotor holding ~85 rev/s with 1 % of wander makes its
    rate column and its intercept column collinear, so least squares splits the
    scale and the offset arbitrarily between them. The split is meaningless
    there, and the two readings below are not:

    ``per_rotor_pct``
        ``100 d_mean / mean_rate``, one number per rotor. No regression, so
        nothing to be ill-conditioned; it is exactly the quantity the
        displacement campaign reported as "% of rate".
    ``global_pct``
        ONE shared scale over all rotors at once,
        ``100 sum(d r) / sum(r r)``. Well conditioned because the rotors sit at
        genuinely different rates, and it is the issue's "joint 4-rotor global
        scale" estimator restricted to a fitted trajectory.

    Both are computed on the edge-trimmed interior, since the tracker's own
    edge trim leaves the first and last ``edge_trim_s`` untouched.
    """
    cfg = cfg or RefitConfig()
    fit = np.atleast_2d(np.asarray(r_fit, dtype=np.float64))
    ref = np.atleast_2d(np.asarray(r_ref, dtype=np.float64))
    ft = np.asarray(ft, dtype=np.float64)
    dt = float(np.median(np.diff(ft))) if ft.size > 1 else 1.0
    trim = max(1, int(round(cfg.edge_trim_s / max(dt, 1e-9))))
    sel = slice(trim, max(trim + 1, ft.size - trim))
    d, r = (fit - ref)[:, sel], ref[:, sel]
    live = r.mean(axis=1) >= cfg.min_rate
    per = [
        round(float(100.0 * d[i].mean() / r[i].mean()), 5) if live[i] else None
        for i in range(fit.shape[0])
    ]
    num = float(np.sum(d[live] * r[live]))
    den = float(np.sum(r[live] ** 2))
    return {
        "per_rotor_pct": per,
        "global_pct": round(100.0 * num / den, 5) if den > 0 else None,
        "d_mean": [round(float(v), 5) for v in d.mean(axis=1)],
        "d_rms": [round(float(v), 5) for v in np.sqrt(np.mean(d**2, axis=1))],
        "mean_rate": [round(float(v), 3) for v in r.mean(axis=1)],
    }


# ---------------------------------------------------------------------------
# the result


@dataclass
class RefitResult:
    """One window's refit: the trajectory, the ladder, and the residual reading."""

    r_fit: np.ndarray  # (R, N) the refined trajectory
    r_init: np.ndarray  # (R, N) the pre-smoothed telemetry that seeded it
    r_raw: np.ndarray  # (R, N) the untouched telemetry
    ft: np.ndarray  # (N,) frame times
    iters: list[dict[str, Any]] = field(default_factory=list)
    converged: bool = False
    #: ``"tolerance"`` | ``"plateau"`` | ``"max_iters"`` — only the first is
    #: convergence in the sense of :attr:`RefitConfig.tol_rev_s`.
    stop_reason: str = "max_iters"
    residual: dict[str, Any] = field(default_factory=dict)
    scale: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def k_ladder(self) -> list[int]:
        return [int(d["k_cap"]) for d in self.iters]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable report — everything but the trajectories themselves."""
        return {
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "n_iters": len(self.iters),
            "k_ladder": self.k_ladder,
            "iters": self.iters,
            "residual": self.residual,
            "scale": self.scale,
            "identity": self.identity,
            "params": self.params,
            "rotor_mean_raw": [round(float(v), 4) for v in self.r_raw.mean(axis=1)],
            "rotor_mean_fit": [round(float(v), 4) for v in self.r_fit.mean(axis=1)],
        }


# ---------------------------------------------------------------------------
# the driver


def refit_window(
    audio: np.ndarray,
    r_meas: np.ndarray,
    ft: np.ndarray,
    sr: float | int = 16000,
    *,
    cfg: RefitConfig | None = None,
    verbose: bool = False,
) -> RefitResult:
    """Fit the rotors to the harmonics on one window (issue 17 steps 1-6).

    Args:
        audio: ``(C, T)`` or ``(T,)`` at ``sr``.
        r_meas: ``(R, N)`` telemetry, rev/s on the ``ft`` grid — the carrier the
            procedure refines, and the reference every reading is against.
        ft: ``(N,)`` frame times, audio-relative seconds.
        sr: audio sample rate (must be exact — an int or an integral float).
        cfg: :class:`RefitConfig`; ``None`` takes the defaults.
        verbose: print one line per iteration.

    Returns:
        A :class:`RefitResult`. ``residual`` is
        :func:`tracking.fitness.residual_decompose` of the fit against the RAW
        telemetry (not the pre-smoothed init), because the raw channel is what
        every historical DREGON number was measured against.
    """
    cfg = cfg or RefitConfig()
    r_raw = np.atleast_2d(np.asarray(r_meas, dtype=np.float64))
    ft = np.asarray(ft, dtype=np.float64)
    if r_raw.shape[-1] != ft.size:
        raise ValueError(f"r_meas has {r_raw.shape[-1]} frames, ft has {ft.size}")
    n_rot = r_raw.shape[0]

    r_init = presmooth(r_raw, ft, cfg.smooth_cut_hz)
    frame = tracking_frame(
        audio,
        sr,
        rps=r_init,
        frame_times=ft,
        rps_meas=r_raw,
        dtype=np.float64,
    )

    trim = max(1, int(round(cfg.edge_trim_s / max(float(np.median(np.diff(ft))), 1e-9))))
    interior = slice(trim, max(trim + 1, ft.size - trim))

    e_est = float(cfg.e0_rev_s)
    k_cap = min(k_cap_for_error(e_est, cfg), cfg.k_start_max)
    r_prev = r_init
    iters: list[dict[str, Any]] = []
    converged = False
    stop_reason = "max_iters"
    d_q_prev = float("inf")

    for it in range(1, cfg.max_iters + 1):
        stage = pi_kalman_arm_stage(
            peel=cfg.peel,
            peel_mode=cfg.peel_mode,
            peel_bw_hz=cfg.peel_bw_hz,
            peel_k_max=cfg.peel_k_max,
            n_rotors=n_rot,
            name=f"refit_it{it}",
            n_iter=cfg.n_iter_inner,
            k_caps=(k_cap,),
            k_max=k_cap,
            fs_env=cfg.fs_env,
            band_mode=cfg.band_mode,
            band_b0=cfg.band_b0,
            band_hz=cfg.band_hz,
            off_comb_hz=cfg.off_comb_hz,
            f_max=cfg.f_max,
            wrap_guard_rad=cfg.wrap_guard_rad,
            pair_mode=cfg.pair_mode,
            max_step=cfg.max_step,
            edge_trim_s=cfg.edge_trim_s,
            min_rate=cfg.min_rate,
        )
        tic = time.perf_counter()
        frame = stage(frame)
        wall = time.perf_counter() - tic
        r_new, _ = get_rps(frame)
        delta = np.abs(r_new - r_prev)[:, interior]
        d_max = float(delta.max()) if delta.size else 0.0
        d_q = float(np.quantile(delta, cfg.step_quantile)) if delta.size else 0.0
        info = frame["meta"]["tracking"][-1]
        peel_diag = info.get("peel")
        rec: dict[str, Any] = {
            "iter": it,
            "k_cap": int(k_cap),
            "e_est_in": round(e_est, 4),
            "delta_max": round(d_max, 5),
            "delta_q": round(d_q, 5),
            "delta_rms": [
                round(float(np.sqrt(np.mean((r_new[i] - r_prev[i])[interior] ** 2))), 5)
                for i in range(n_rot)
            ],
            "delta_mean": [
                round(float(np.mean((r_new[i] - r_prev[i])[interior])), 5) for i in range(n_rot)
            ],
            "wall_s": round(wall, 1),
            "wall_peel_s": info.get("wall_peel_s"),
            "wall_pi_s": info.get("wall_pi_s"),
        }
        if peel_diag is not None:
            rec["peel"] = {
                "mode": peel_diag["mode"],
                "energy_ok": peel_diag["energy_ok"],
                "e_resid_all_ratio": peel_diag["e_resid_all_ratio"],
                "e_resid_ratio": [d["e_resid_ratio"] for d in peel_diag["per_rotor"]],
            }
        iters.append(rec)
        if verbose:
            print(
                f"  [refit] it {it}: k<={k_cap} e_in {e_est:.3f} "
                f"|dr|max {d_max:.4f} q{cfg.step_quantile:.2f} {d_q:.4f} ({wall:.0f}s)",
                flush=True,
            )
        r_prev = r_new
        if it >= cfg.min_iters:
            if d_max < cfg.tol_rev_s:
                converged, stop_reason = True, "tolerance"
                rec["stop"] = stop_reason
                break
            if cfg.plateau_rel > 0 and d_q > (1.0 - cfg.plateau_rel) * d_q_prev:
                stop_reason = "plateau"
                rec["stop"] = stop_reason
                break
        d_q_prev = d_q
        e_est = max(d_q, cfg.e_floor_rev_s)
        k_cap = advance_k(k_cap, e_est, cfg)

    r_fit, _ = get_rps(frame)
    order0, gaps0 = order_and_gaps(r_raw)
    order1, gaps1 = order_and_gaps(r_fit)
    return RefitResult(
        r_fit=r_fit,
        r_init=r_init,
        r_raw=r_raw,
        ft=ft,
        iters=iters,
        converged=converged,
        stop_reason=stop_reason,
        residual=residual_decompose(r_fit, r_raw, ft, cfg=cfg.fitness_cfg()),
        scale=scale_summary(r_fit, r_raw, ft, cfg=cfg),
        identity={
            "order_raw": order0,
            "order_fit": order1,
            "order_kept": order1 == order0,
            "gaps_raw": gaps0,
            "gaps_fit": gaps1,
            "gap_ratio": [
                round(float(b / a), 4) if abs(a) > 1e-9 else None
                for a, b in zip(gaps0, gaps1, strict=True)
            ],
        },
        params={
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in cfg.__dict__.items()
            if not k.startswith("_")
        },
    )


# ---------------------------------------------------------------------------
# Stage adapter


def refit_stage(
    *,
    cfg: RefitConfig | None = None,
    reference_entry: str = "rps_meas",
    name: str = "telemetry_refit",
) -> Any:
    """The whole refit as one :data:`tracking.stages.Stage`.

    The carrier is the frame's ``reference_entry`` (``"rps_meas"`` by default,
    the untouched telemetry) — NOT the frame's current ``"rps"``, because the
    procedure is defined as a refinement OF the measurement. The stage replaces
    ``"rps"`` with the fit and appends the report (minus the trajectories) as
    its diagnostics entry.
    """
    from tracking.stages import get_audio, with_rps

    use = cfg or RefitConfig()

    def run(frame: Any) -> Any:
        audio, sr = get_audio(frame)
        ref, ft = get_rps(frame, reference_entry)
        t0 = float(frame["audio"].t_start)
        res = refit_window(audio, ref, ft - t0, sr, cfg=use)
        return with_rps(frame, res.r_fit, ft, stage=name, info=res.as_dict())

    return run
