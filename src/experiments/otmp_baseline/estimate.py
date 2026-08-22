"""Frame-level and clip-level multi-pitch estimation front end.

Wraps :mod:`experiments.otmp_baseline.solver` into the pipeline the paper
describes: analytic signal -> sample autocovariance -> eq (27) -> read the
pitch masses ``M^T 1_F`` -> restrict the pitch grid to the active candidates
and re-solve the debiased program eq (37) -> report the ``K`` strongest
pitches.

Reference: A. Björkman and F. Elvander, "Inverse Harmonic Clustering for
Multi-Pitch Estimation: An Optimal Transport Approach", IEEE TSP 2026
(arXiv:2508.02471).

Choices the paper leaves open are marked ``[choice]`` in the docstrings and
collected in the module-level list below.

[choice] 1. The paper never states how the real measurement becomes the
   complex covariance ``r_hat``. Its reference [32] is Marple's discrete-time
   analytic signal, so the signal is converted to its analytic form, restricted
   to the analysis band, and scaled to ``r_hat(0) = 1`` (the paper normalizes
   to unit variance; fixing ``r_hat(0)`` instead makes ``beta`` a fixed
   fraction of total power regardless of how much of the band is kept). The
   target is ``r0_target``, and it is not cosmetic: ``nu`` scales with it while
   the data-fit term scales with its square, so it sets the *strength* of the
   ``beta`` and ``zeta`` regularization relative to the data. A real signal of
   unit variance has an analytic covariance with ``r_hat(0) = 2``, so 1 and 2
   are both defensible readings of the paper.
[choice] 2. The autocovariance uses the unbiased normalization ``1/(N - tau)``.
   The biased one tapers by ``(N - tau)/N``, which at ``tau = 2N/3`` attenuates
   by 3x and would fight ``beta``. Switch with ``unbiased_acf=False``.
[choice] 3. Outer / inner iteration counts and stopping tolerance are not in
   the paper. The outer loop is *not* run to its fixed point: the sparsity
   terms keep eroding weak components long after the pitch ranking has settled,
   so ``max_iter`` is a real parameter, not just a budget.
[choice] 4. Identifying the pitches from the mass vector ``M^T 1_F`` (Sec.
   VII-C says only "inspecting the support"): greedy descending peak-picking
   with a minimum *relative* separation (``min_sep_rel``), which merges the
   several adjacent grid candidates that one true fundamental always lights up,
   and a relative floor (``active_rel_thresh``) below which nothing is
   selected. If fewer than ``K`` peaks survive, the strongest is repeated.
[choice] 5. The debiased solve of eq (37) is restricted to exactly those ``K``
   identified candidates. A loose absolute threshold instead admits most of the
   grid, which makes ``c_min`` vanish everywhere and destroys the debiased
   solution entirely.
[choice] 6. Table II (real data) gives no debiased ``zeta`` / ``beta`` row, so
   ``debias_zeta`` / ``debias_beta`` default to the main values.
[choice] 7. The size ``G`` of the pitch-candidate grid is never stated. It
   matters a lot: a coarse grid has no good representative for an off-grid
   fundamental, and the octave-up candidate — which fits the *even* partials
   better in relative terms — then steals them. See :func:`simulated_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from experiments.otmp_baseline.cost import ground_cost, linear_grid
from experiments.otmp_baseline.solver import (
    Quadratic,
    build_quadratic,
    solve_debiased,
    solve_stochastic,
)

__all__ = [
    "FrameEstimate",
    "OTMPConfig",
    "analytic_band_signal",
    "autocovariance",
    "drone_config",
    "estimate_clip",
    "estimate_frame",
    "real_config",
    "simulated_config",
]


@dataclass(frozen=True)
class OTMPConfig:
    """Everything the estimator needs. Defaults = Table II, ``Proposed_c``.

    Table II is the paper's *real data* column (Bach10, 44.1 kHz music); the
    grid fields default to that experiment's 30 ms frame. Use
    :func:`simulated_config` for Table I and :func:`drone_config` for the
    rotor-speed setting.
    """

    # ---- analysis ----
    sample_rate: int = 44100
    frame_len: int = 1323  # 30 ms, the paper's Bach10 frame
    hop: int | None = None  # None -> non-overlapping
    n_lags: int | None = None  # None -> round(2/3 * frame_len), the paper's T
    unbiased_acf: bool = True
    r0_target: float = 1.0  # r_hat(0) after normalization; see [choice] 1

    # ---- grids ----
    freq_lo_hz: float = 0.0
    freq_hi_hz: float | None = None  # None -> Nyquist
    n_freq: int | None = 2260
    freq_step_hz: float | None = None
    pitch_lo_hz: float = 50.0
    pitch_hi_hz: float = 500.0
    n_pitch: int | None = 226
    pitch_step_hz: float | None = None

    # ---- regularization (Table II, Proposed_c) ----
    eta: float = 1e-1
    zeta: float = 1e-1
    eps: float = 1e-5
    beta: float = 3.4e-2
    debias: bool = True
    debias_zeta: float | None = None  # None -> zeta
    debias_beta: float | None = None  # None -> beta

    # ---- solver budget ----
    max_iter: int = 800
    tol: float = 1e-6
    inner_iters: int = 1
    max_active: int = 64
    debias_max_iter: int = 800

    # ---- read-out ----
    n_pitches: int = 4
    active_rel_thresh: float = 1e-3
    min_sep_rel: float = 0.03

    @property
    def lags(self) -> int:
        return int(self.n_lags) if self.n_lags else int(round(2.0 * self.frame_len / 3.0))

    @property
    def band(self) -> tuple[float, float]:
        hi = self.freq_hi_hz if self.freq_hi_hz is not None else self.sample_rate / 2.0
        return float(self.freq_lo_hz), float(hi)

    def freq_grid_hz(self) -> NDArray:
        lo, hi = self.band
        # A frequency grid that starts at exactly 0 Hz gives a degenerate
        # ratio in the ground cost, so the open end of the band is used.
        lo = max(lo, (hi - lo) / max(self.n_freq or 1, 1) * 1e-6)
        return linear_grid(lo, hi, step=self.freq_step_hz, n=self.n_freq)

    def pitch_grid_hz(self) -> NDArray:
        return linear_grid(
            self.pitch_lo_hz, self.pitch_hi_hz, step=self.pitch_step_hz, n=self.n_pitch
        )


def simulated_config(**overrides) -> OTMPConfig:
    """Table I (``Proposed_c``) plus the Sec. VIII-A simulation grids.

    4 pitches, 8 kHz, ``N = 250``. The frequency grid has the 2260 uniformly
    spaced points the paper matches to the reference methods, over the full
    band; the pitch grid spans 50-500 Hz.

    [choice] The paper does not state ``G``. 226 points (2 Hz) is the tempting
    reading — the reference methods' 2260-point frequency grid is exactly the
    integer multiples up to order 10 of such a pitch grid — but it estimates
    badly: with the true fundamental 1 Hz off the nearest candidate, the
    *octave-up* candidate matches the even partials better in relative terms
    and takes them, and the displaced pitch is lost. 451 points (1 Hz) is used
    instead, which measurably lowers the gross error rate.
    """
    return replace(
        OTMPConfig(
            sample_rate=8000,
            frame_len=250,
            eta=1e-1,
            zeta=1e1,
            eps=1e-6,
            beta=1.5e-2,
            debias_zeta=1e0,
            debias_beta=4.5e-2,
            n_freq=2260,
            pitch_lo_hz=50.0,
            pitch_hi_hz=500.0,
            n_pitch=451,
            min_sep_rel=0.03,
        ),
        **overrides,
    )


def real_config(**overrides) -> OTMPConfig:
    """Table II (``Proposed_c``) — the paper's real-data parameters."""
    return replace(OTMPConfig(), **overrides)


def drone_config(**overrides) -> OTMPConfig:
    """Table II parameters on a 4-rotor drone grid (0.5 s at 16 kHz).

    Only the grids change: 40-1200 Hz at 0.5 Hz for the spectrum (the band that
    holds the rotor comb), 30-120 Hz at 0.25 Hz for the pitch candidates (i.e.
    30-120 rev/s), ``K = 4``. Rotor speeds sit within a few percent of each
    other, so ``min_sep_rel`` drops to 0.008 (about two pitch-grid steps at
    cruise).

    On the frozen DREGON-LM validation clips this does *not* work: per-frame
    PIT-MAE is 38-45 rev/s against targets of 34-62 rev/s, i.e. no better than
    guessing. See :mod:`experiments.otmp_baseline.drone_smoke`. The spectral
    estimate puts its mass on the 50-60 Hz broadband floor rather than on the
    rotor comb, and the reported fundamentals come out near half the true rotor
    rate. Three things stand against the method here and none of them are bugs:
    the comb decoheres inside a 0.5 s window because the rotor speed drifts;
    the four rotors run at only two distinct speeds, so ``K = 4`` forces the
    estimator to invent two more; and the mixture also contains speech, whose
    partials are a competing harmonic source inside the same band.
    """
    return replace(
        OTMPConfig(
            sample_rate=16000,
            frame_len=8000,
            freq_lo_hz=40.0,
            freq_hi_hz=1200.0,
            n_freq=None,
            freq_step_hz=0.5,
            pitch_lo_hz=30.0,
            pitch_hi_hz=120.0,
            n_pitch=None,
            pitch_step_hz=0.25,
            n_pitches=4,
            min_sep_rel=0.008,
        ),
        **overrides,
    )


# --------------------------------------------------------------------------
# signal -> covariance
# --------------------------------------------------------------------------


def analytic_band_signal(x: NDArray, sample_rate: float, lo_hz: float, hi_hz: float) -> NDArray:
    """Discrete-time analytic signal restricted to ``[lo_hz, hi_hz]``.

    The dictionary only spans the analysis band, so out-of-band power would
    otherwise appear in ``r_hat`` as unmodelable residual and be paid for by
    ``beta``. Zeroing the non-analysis bins of the analytic spectrum removes it
    (Marple's FFT construction, the paper's [32], with an extra band mask).

    A complex ``x`` is taken to be analytic already (as the paper's model (2)
    is) and is only band-masked, never doubled.
    """
    already_analytic = np.iscomplexobj(x)
    x = np.asarray(x).ravel()
    n = x.size
    spec = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    keep = (freqs >= lo_hz) & (freqs <= hi_hz)
    out = np.zeros(n, dtype=np.complex128)
    if already_analytic:
        out[keep] = spec[keep]
        return np.fft.ifft(out)
    out[keep] = 2.0 * spec[keep]
    if lo_hz <= 0.0:  # DC and (for even n) Nyquist are not doubled
        out[0] = spec[0]
    if n % 2 == 0 and keep[n // 2]:
        out[n // 2] = spec[n // 2]
    return np.fft.ifft(out)


def autocovariance(z: NDArray, n_lags: int, unbiased: bool = True) -> NDArray:
    """``r_hat(tau) = E[z(t + tau) conj(z(t))]`` for ``tau = 0 .. n_lags-1``."""
    z = np.asarray(z, dtype=np.complex128).ravel()
    n = z.size
    if n_lags > n:
        raise ValueError(f"n_lags={n_lags} exceeds signal length {n}")
    size = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.fft(z, size)
    acf = np.fft.ifft(spec * np.conj(spec))[:n_lags]
    counts = (n - np.arange(n_lags)) if unbiased else np.full(n_lags, n)
    return acf / counts


# --------------------------------------------------------------------------
# the per-frame estimator
# --------------------------------------------------------------------------


@dataclass
class FrameEstimate:
    """One frame's result."""

    pitches_hz: NDArray  # (K,)
    masses: NDArray  # (K,) power assigned to each reported pitch
    pitch_grid_hz: NDArray  # (G,)
    pitch_mass: NDArray  # (G,) M^T 1_F from the biased solve, whole grid
    nu: NDArray  # (F,) spectral estimate (debiased when cfg.debias)
    active: NDArray  # (n_retained,) grid indices kept for the debiased solve
    n_iter: int
    converged: bool


@lru_cache(maxsize=8)
def _grids(cfg: OTMPConfig) -> tuple[NDArray, NDArray, NDArray]:
    """Cached ``(freqs_hz, pitches_hz, cost)`` for a config."""
    freqs = cfg.freq_grid_hz()
    pitches = cfg.pitch_grid_hz()
    return freqs, pitches, ground_cost(freqs, pitches)


def _freqs_rad(cfg: OTMPConfig) -> NDArray:
    return 2.0 * np.pi * _grids(cfg)[0] / cfg.sample_rate


@lru_cache(maxsize=4)
def _gram_cache(cfg: OTMPConfig) -> Quadratic:
    """Cached Gram / step size — they depend only on the grid and ``T``.

    A zero correlation vector is stored; :func:`estimate_frame` fills in the
    per-frame one through :meth:`Quadratic.with_corr`. Building this is the
    expensive part, and it is shared by every frame of every clip analyzed with
    the same config.
    """
    return build_quadratic(np.zeros(cfg.lags, dtype=np.complex128), _freqs_rad(cfg))


def _select_pitches(
    pitches_hz: NDArray, mass: NDArray, k: int, min_sep_rel: float, floor: float
) -> NDArray:
    """Identify the ``k`` present pitches from the transport plan's mass.

    Greedy descending peak-pick with a minimum relative separation: adjacent
    grid points around one true fundamental all carry mass, so the runner-up
    must be far enough away in *relative* frequency to count as a second pitch.
    Candidates below ``floor`` times the strongest mass are never selected.

    Returns indices into the pitch grid, strongest first.
    """
    order = np.argsort(-mass)
    threshold = floor * float(mass[order[0]])
    chosen: list[int] = []
    for idx in order:
        if mass[idx] <= threshold:
            break
        if all(abs(np.log(pitches_hz[idx] / pitches_hz[j])) > min_sep_rel for j in chosen):
            chosen.append(int(idx))
        if len(chosen) == k:
            break
    if not chosen:
        chosen = [int(order[0])]
    while len(chosen) < k:  # [choice] pad by repeating the strongest
        chosen.append(chosen[0])
    return np.asarray(chosen[:k], dtype=int)


def estimate_frame(x: NDArray, sample_rate: float, cfg: OTMPConfig | None = None) -> FrameEstimate:
    """Estimate the ``cfg.n_pitches`` strongest fundamentals of one frame.

    ``x`` is a real, single-channel frame of ``cfg.frame_len`` samples (longer
    input is truncated). Returns a :class:`FrameEstimate`; the two headline
    fields are ``pitches_hz`` and ``masses``, both length ``cfg.n_pitches`` and
    sorted by descending mass.
    """
    cfg = cfg or OTMPConfig()
    if int(sample_rate) != int(cfg.sample_rate):
        raise ValueError(f"sample_rate={sample_rate} != cfg.sample_rate={cfg.sample_rate}")
    x = np.asarray(x).ravel()[: cfg.frame_len]
    if x.size < cfg.lags:
        raise ValueError(f"frame of {x.size} samples is shorter than T={cfg.lags}")

    lo, hi = cfg.band
    z = analytic_band_signal(x, cfg.sample_rate, lo, hi)
    r_hat = autocovariance(z, cfg.lags, unbiased=cfg.unbiased_acf)
    scale = float(np.real(r_hat[0]))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("frame has no in-band energy")
    r_hat = r_hat * (cfg.r0_target / scale)  # [choice] fix r_hat(0)

    freqs_hz, pitches_hz, cost = _grids(cfg)
    quad = _gram_cache(cfg).with_corr(_corr_chunked(r_hat, _freqs_rad(cfg)), cfg.r0_target)

    res = solve_stochastic(
        quad,
        cost,
        eta=cfg.eta,
        zeta=cfg.zeta,
        eps=cfg.eps,
        beta=cfg.beta,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        inner_iters=cfg.inner_iters,
        max_active=cfg.max_active,
    )

    # Sec. VII-C: the pitches present are read off the biased transport plan's
    # mass over the candidate grid. Debiasing then re-solves on *only* those
    # candidates; it corrects the amplitudes, it does not re-choose the pitches.
    active = _select_pitches(
        pitches_hz, res.pitch_mass, cfg.n_pitches, cfg.min_sep_rel, cfg.active_rel_thresh
    )
    retained = np.unique(active)

    nu = res.nu
    masses = res.pitch_mass[active]
    n_iter, converged = res.n_iter, res.converged
    if cfg.debias:
        deb = solve_debiased(
            quad,
            cost[:, retained].min(axis=1),
            zeta=cfg.debias_zeta if cfg.debias_zeta is not None else cfg.zeta,
            beta=cfg.debias_beta if cfg.debias_beta is not None else cfg.beta,
            max_iter=cfg.debias_max_iter,
            tol=cfg.tol,
        )
        nu = deb.nu
        n_iter += deb.n_iter
        converged = converged and deb.converged
        # [choice] re-read the masses: in the eps -> 0, eta = 0 limit each
        # frequency's power goes wholly to the retained pitch(es) of minimum
        # cost (Prop. 3's remark), ties split equally.
        sub = cost[:, retained]
        is_min = sub <= sub.min(axis=1, keepdims=True) + 1e-15
        share = is_min / is_min.sum(axis=1, keepdims=True)
        per_retained = nu @ share
        masses = per_retained[np.searchsorted(retained, active)]

    order = np.argsort(-masses)
    return FrameEstimate(
        pitches_hz=pitches_hz[active[order]],
        masses=masses[order],
        pitch_grid_hz=pitches_hz,
        pitch_mass=res.pitch_mass,
        nu=nu,
        active=retained,
        n_iter=n_iter,
        converged=converged,
    )


def _corr_chunked(r_hat: NDArray, freqs_rad: NDArray, chunk: int = 512) -> NDArray:
    tau = np.arange(r_hat.size, dtype=np.float64)
    out = np.empty(freqs_rad.size, dtype=np.float64)
    for start in range(0, freqs_rad.size, chunk):
        stop = min(start + chunk, freqs_rad.size)
        out[start:stop] = np.real(np.exp(-1j * np.outer(freqs_rad[start:stop], tau)) @ r_hat)
    return out


# --------------------------------------------------------------------------
# clip level
# --------------------------------------------------------------------------


def link_frames(pitches: NDArray) -> NDArray:
    """Reorder per-frame pitch sets for continuity (Hungarian, frame to frame).

    ``pitches`` is ``(K, T')`` with each column independently sorted by mass.
    Each column after the first is permuted to minimize the total absolute jump
    from the previous (already linked) column — the convention the salience-RPS
    tracking evaluation uses.
    """
    from scipy.optimize import linear_sum_assignment

    out = np.array(pitches, dtype=np.float64, copy=True)
    out[:, 0] = np.sort(out[:, 0])
    for t in range(1, out.shape[1]):
        cost = np.abs(out[:, t - 1][:, None] - out[:, t][None, :])
        rows, cols = linear_sum_assignment(cost)
        col = out[:, t].copy()
        out[rows, t] = col[cols]
    return out


@overload
def estimate_clip(
    x: NDArray,
    sample_rate: float,
    cfg: OTMPConfig | None = ...,
    *,
    link: bool = ...,
    return_masses: Literal[False] = ...,
) -> tuple[NDArray, NDArray]: ...


@overload
def estimate_clip(
    x: NDArray,
    sample_rate: float,
    cfg: OTMPConfig | None = ...,
    *,
    link: bool = ...,
    return_masses: Literal[True],
) -> tuple[NDArray, NDArray, NDArray]: ...


def estimate_clip(
    x: NDArray,
    sample_rate: float,
    cfg: OTMPConfig | None = None,
    *,
    link: bool = True,
    return_masses: bool = False,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray]:
    """Run :func:`estimate_frame` over a clip and link the tracks.

    Returns ``(times_s, pitches_hz)`` with ``pitches_hz`` of shape ``(K, T')``
    and ``times_s`` the frame *centers*. With ``return_masses=True`` a third
    ``(K, T')`` array of the reported masses (in pre-link order per frame) is
    appended.
    """
    cfg = cfg or OTMPConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    hop = int(cfg.hop or cfg.frame_len)
    starts = list(range(0, max(x.size - cfg.frame_len + 1, 0), hop))
    if not starts:
        raise ValueError(f"clip of {x.size} samples is shorter than one frame")

    pitches = np.empty((cfg.n_pitches, len(starts)), dtype=np.float64)
    masses = np.empty_like(pitches)
    for i, start in enumerate(starts):
        est = estimate_frame(x[start : start + cfg.frame_len], sample_rate, cfg)
        pitches[:, i] = est.pitches_hz
        masses[:, i] = est.masses

    times = (np.asarray(starts, dtype=np.float64) + cfg.frame_len / 2.0) / cfg.sample_rate
    if link:
        pitches = link_frames(pitches)
    if return_masses:
        return times, pitches, masses
    return times, pitches
