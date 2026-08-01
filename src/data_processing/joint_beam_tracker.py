"""Joint 4-rotor beam-search RPS tracker with an OU control-mode prior.

Replaces the coarse stage's single shared trajectory ``c(t)`` (all four rotors
riding at fixed offsets from it) with a genuinely joint search over the full
speed vector ``w_t = (w1, w2, w3, w4)``.  The shared-shape constraint is the
defect WP3 measured: on the DREGON ramp the prediction std is ~26 for ALL FOUR
rotors while ground truth is 27 / 23.5 / 27 / 23, shape correlation 0.975 — the
common shape is right and the per-rotor deviation is unrepresented, so nothing
downstream can invent it.

Three ideas make a joint search tractable and useful:

1. **An OU prior in control-allocation mode space** (§ :class:`OUPrior`).
   Rotor-speed increments are measured to be nearly ISOTROPIC in mode space
   (sigma_common / sigma_diff = 0.97-1.07 on DREGON, 1.64-1.93 on Michael's;
   WP16), so a first-order prior on increments has no power to prefer
   correlated motion — below sqrt(5) it actively prefers the uncorrelated move.
   The anisotropy is in the mode LEVEL: a quadrotor's differential trim is
   restored by the flight controller while its throttle is free.  That is a
   MEAN-REVERSION term, not a diffusion term, and it is what this prior adds.

2. **A harmonic-scaled analysis bandwidth** (§ :class:`EmissionCfg`).  Reading
   a comb tooth at a fixed bandwidth ``B`` gives a capture radius ``B/k`` in
   rev/s, which shrinks exactly on the harmonics carrying the most information
   about rotor speed.  Pooling over ``B_k = k*B0`` Hz makes the capture radius
   ``B0`` rev/s at every harmonic; two rotors' k-th teeth are ``k|r_i - r_j|``
   apart, so the separation argument is unaffected and every threshold becomes
   a single k-invariant rev/s number.

3. **Beam search over two candidate families** (§ :func:`joint_beam_track`) —
   global peak assignments for acquisition and recovery, local moves for
   continuity, the latter being what lets a rotor COAST through a stretch where
   its own comb is invisible (FLY124's 81 rev/s rotor).

Everything is torch and device-agnostic: pass ``device="cuda"`` if one is free,
otherwise it runs on CPU in seconds per 16 s window.

See ``docs/experiments/rps-refine-precision.md`` §§ WP16/WP17.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

from data_processing.rps_synthesis import MIXER

NUM_ROTORS = 4
MODE_NAMES = ("common", "roll", "pitch", "yaw")

# --------------------------------------------------------------------------
# The OU transition prior


@dataclass(frozen=True)
class OUPrior:
    """Discrete-time Ornstein-Uhlenbeck prior on the control modes.

    Per mode ``i``, with ``a_i = exp(-dt/tau_i)`` and stationary std
    ``sigma_level_i``::

        residual_i = m_i(t) - a_i*m_i(t-1) - (1 - a_i)*mu_i
        s_i        = sigma_level_i * sqrt(1 - a_i^2)          (innovation scale)
        cost       = sum_i psi_i(residual_i / s_i)

    ``tau_i = inf`` degenerates to a pure random walk (``a_i = 1``), which drops
    the ``mu_i`` term entirely.

    **The common mode is a random walk by default, and that is deliberate.**
    Within a cruise segment the common mode does fit an OU with tau ~0.6-1.5 s,
    but across regimes it is not stationary at all: a takeoff moves it 42-57
    rev/s, i.e. 26-36 stationary sigmas, and |d_common| per 32 ms frame reaches
    15 sigma on DREGON and 219 sigma on FLY124 (WP16).  Mean-reverting the
    common mode to a within-window constant charges a takeoff hundreds of cost
    units and flattens it — precisely the ramp failure the design set out to
    avoid.  The throttle setpoint is free; the differential trim is not.  Set
    ``tau_common`` to a finite value to measure that claim rather than assume it.

    ``psi_common`` is Huber (quadratic within ``huber_knee`` innovation scales,
    linear beyond) for the same reason; the differential modes are quadratic.

    ``mu`` for the differential modes is not a universal constant — it is the
    airframe's trim, which differs per recording and per payload — so it is
    taken from each beam state's OWN frame-0 assignment (see ``mu_mode``).

    **One tau and one sigma are SHARED by roll/pitch/yaw, and that is forced,
    not a simplification.**  Rotor identity is arbitrary under PIT: the tracker
    cannot know which row is RFront.  A rotor relabelling ``P`` preserves the
    all-ones vector, hence maps the differential subspace to itself by an
    orthogonal 3x3 matrix ``M = Bd^T P Bd / 4`` — so the differential cost is
    invariant under relabelling **iff** the three modes share one scale.  With
    per-mode scales the tracker's answer would depend on an arbitrary labelling.
    The measurement says nothing is lost: the fitted per-mode taus (1.95/1.81/
    2.80, 1.55/2.40/1.32, 1.25/1.11/0.91, 2.25/3.71/0.79, 1.11/2.09/2.75 over
    the five DREGON recordings) have no consistent ordering across recordings,
    i.e. their spread is estimator noise.

    Defaults are the WP16 fit (``scripts/mode_covariance_calib.py``), median
    over the five DREGON free-flight room1 recordings' ``motors_measured``
    cruise segments, fitted from the autocovariance decay over lags 2-64 so
    that the reciprocal-period label noise (which contributes only at lag 0)
    does not bias tau.  Sanity check on those values: the implied innovation
    scale ``s`` for the differential modes, 0.14 rev/s per 32 ms frame, agrees
    with the independently measured band-limited (fc = 5 Hz) increment scale of
    0.09-0.11 rev/s.
    """

    #: Mean-reversion time of the COMMON mode, seconds.  ``inf`` = random walk
    #: (the default; see the class docstring).
    tau_common: float = math.inf
    #: Mean-reversion time shared by roll/pitch/yaw, seconds.
    tau_diff: float = 1.81
    #: Stationary std shared by roll/pitch/yaw, rev/s (WP16 median of the
    #: per-recording rms over the three differential modes).
    sigma_level_diff: float = 0.77
    #: Stationary std of the common mode, rev/s.  Used only when
    #: ``tau_common`` is finite.
    sigma_level_common: float = 2.53
    #: Innovation scale of the common mode when it is a random walk, rev/s per
    #: frame.  WP16's band-limited common-mode increment scale (fc 5-8 Hz).
    s_random_walk: float = 0.39
    #: Global multiplier on every innovation scale.  The fitted values describe
    #: the TRUTH; a tracker reading noisy audio needs slack, and this is the
    #: one knob for it, and it is load-bearing: at the fitted (truth) scales the
    #: prior is stiff enough that a rotor cannot follow its own comb across the
    #: 0.5 rev/s grid.  Measured on the synthetic wiggle battery, worst per-rotor
    #: shape correlation by s_scale at lambda_e = 3: 1.0 -> 0.30, 2.0 -> 0.94,
    #: 3.0 -> 0.94, 6.0 -> 0.94; at s_scale = 2, lambda_e = 1 it is 0.40.
    s_scale: float = 3.0
    #: Huber knee for the common mode, in innovation scales.
    huber_knee: float = 1.0
    #: Frame step, seconds.
    dt: float = 0.032

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """``(a, s)`` per mode: reversion factor and innovation scale."""
        taus = (self.tau_common, self.tau_diff, self.tau_diff, self.tau_diff)
        sig = (
            self.sigma_level_common,
            self.sigma_level_diff,
            self.sigma_level_diff,
            self.sigma_level_diff,
        )
        a = np.empty(NUM_ROTORS)
        s = np.empty(NUM_ROTORS)
        for i, (t, sl) in enumerate(zip(taus, sig, strict=True)):
            if math.isinf(t):
                a[i] = 1.0
                s[i] = self.s_random_walk if i == 0 else sl
            else:
                a[i] = math.exp(-self.dt / t)
                s[i] = sl * math.sqrt(max(1.0 - a[i] ** 2, 1e-12))
        return a, s * self.s_scale

    def sustained_cost_rate(self, delta: float) -> dict[str, float]:
        """Cost per frame of HOLDING a sustained offset, per hypothesis.

        The discrimination the whole prior exists for, stated as a number.
        ``delta`` rev/s applied to all four rotors is a pure common offset;
        ``delta`` on ONE rotor decomposes into ``delta/4`` in every mode.
        Returns the two per-frame costs and their ratio (``inf`` when the
        common mode is a random walk, which holds any level for free).
        """
        a, s = self.coefficients()

        def hold(d: np.ndarray) -> float:
            r = (1.0 - a) * d
            z = r / s
            c = 0.5 * z[1:] ** 2
            zc = abs(z[0])
            c0 = (
                0.5 * zc**2
                if zc <= self.huber_knee
                else self.huber_knee * (zc - 0.5 * self.huber_knee)
            )
            return float(c0 + c.sum())

        four = hold(np.array([delta, 0.0, 0.0, 0.0]))
        one = hold(np.full(NUM_ROTORS, delta / 4.0))
        return {
            "common": four,
            "single_rotor": one,
            "ratio": float("inf") if four <= 0 else one / four,
        }


# --------------------------------------------------------------------------
# Emission: single-rotor comb contrast on the whitened spectrogram


@dataclass(frozen=True)
class EmissionCfg:
    """Full-range single-rotor comb-contrast scoring with a k-scaled bandwidth.

    The score of a candidate speed ``c`` at frame ``t`` is the mean whitened
    value on its teeth ``k*c`` minus the mean POSITIVE value on the half-teeth
    ``(k-0.5)*c`` — the same contrast the coarse pass and ``m1_corridor`` use,
    so the two are directly comparable — with one change:

    **each tooth is pooled over ``+- k*b0_rps`` Hz instead of point-sampled.**

    Motivation: with a fixed analysis bandwidth ``B`` Hz the capture radius in
    rev/s is ``B/k``, so it is TIGHTEST on the high harmonics, which are
    precisely the ones that localise rotor speed best.  On a spectrogram the
    fixed bandwidth is the FFT bin (7.8 Hz at ``COARSE_NFFT``), giving a
    capture radius of 7.8 rev/s at k=1 but only 0.98 rev/s at k=8.  Pooling
    ``+- k*b0`` Hz makes the capture radius ``b0`` rev/s at every harmonic.

    The half-tooth reference is pooled over the SAME width, so the upward bias
    of a max over a k-growing band cancels between the two terms and no
    per-harmonic noise re-normalisation is needed.

    ``k_weight="k"`` weights harmonic ``k`` by ``k``: with a k-scaled band the
    admitted noise grows as ``k``, so the variance of the speed estimate that
    harmonic supports goes as ``1/k`` rather than ``1/k^2``, and the
    inverse-variance weight is ``k`` rather than ``k^2``.  ``"uniform"``
    reproduces the existing unweighted mean.

    ``b0_rps = 0`` collapses the pooling to a single interpolated sample and
    reproduces ``rps_refine_lab._single_comb_scores`` exactly (verified in
    ``tests/test_joint_beam_tracker.py``), which is what makes it an A/B knob.
    """

    lo: float = 12.0
    hi: float = 120.0
    step: float = 0.5
    k_max: int = 8
    #: Half-bandwidth of the k-th tooth, in rev/s (the band is +- k*b0 Hz).
    #:
    #: **Measured default 0.0 — the k-scaling costs more than it buys HERE, and
    #: the reason is structural.**  A k-scaled band is a CAPTURE device: it
    #: helps an estimator that must find a tooth near where it currently thinks
    #: the rotor is.  This stage is not that — it is a dense search over an
    #: explicit 12-120 rev/s grid, so it never has to "capture" anything, and
    #: widening the band only blurs the surface it is searching.  Measured on
    #: the synthetic wiggle battery, the ceiling of the emission itself
    #: (per-frame argmax within +-3 rev/s of truth, correlated against the true
    #: per-rotor shape) falls monotonically: b0 = 0 -> 0.92/0.83/0.97/0.98,
    #: 0.25 -> 0.83/0.74/0.97/0.98, 0.5 -> 0.77/0.71/0.95/0.95, 1.0 ->
    #: 0.65/0.83/0.93/0.90.  Kept as the headline A/B knob because the argument
    #: for it is about REAL windows, where the truth drifts between grid points
    #: and the high harmonics are the ones that go first.
    b0_rps: float = 0.0
    #: Sample points across each pooled band (odd, so the centre is included).
    #: ``0`` = auto: enough that the spacing never exceeds half an FFT bin at
    #: ``k_max``, with a floor of 5.  This matters — a fixed 5 points across the
    #: widest band under-samples it (at ``k=16, b0=1.5`` the spacing is 1.5 bins,
    #: wider than the spectral line, and the pooled max MISSES the tooth it was
    #: widened to catch: measured contribution 0.93 -> 0.30).
    n_band: int = 0
    k_weight: str = "k"  # "k" | "uniform"
    #: How the per-tooth contrasts are pooled into one score.
    #:
    #: ``"mean"`` (the shipped form) is a weighted mean, and its defect is that
    #: it rewards ANY loud content at multiples of ``c``: a comb with four loud
    #: teeth and twelve absent ones scores like one with sixteen medium teeth.
    #: Nothing in it says *most of the predicted teeth are actually there*,
    #: which is the property that distinguishes a rotor from a coincidence.
    #: Measured consequence (WP20): on six windows the emission scores the TRUE
    #: speeds barely above the per-frame median, so the objective ranks the
    #: ground truth DEARER than the tracker's own 27 rev/s-error output.
    #:
    #: - ``"quantile"``: the ``pool_q`` quantile of the per-tooth contrast.  A
    #:   real comb has most teeth present, so a low quantile stays high; a
    #:   coincidence riding a few loud lines collapses.
    #: - ``"frac_pos"``: the weighted fraction of teeth with positive contrast —
    #:   scale-free, and the most direct statement of "the teeth are there".
    pool: str = "mean"  # "mean" | "quantile" | "frac_pos"
    #: Quantile for ``pool="quantile"`` (0.25 = the lower quartile of teeth).
    pool_q: float = 0.25
    f_min: float = 20.0  # COARSE_F_MIN — keeps the k1/k2 teeth
    f_max: float = 6000.0
    #: Frames of boxcar smoothing before the per-frame normalisation.
    smooth_frames: int = 3
    #: Soft floor on the per-frame normalisation denominator, x the global
    #: median contrast (COARSE_NORM_SOFT).
    norm_soft: float = 0.3
    #: What the per-frame score is measured AGAINST.
    #:
    #: ``"peak"`` (the shipped form) divides by ``peak - median``, so the best
    #: comb in every frame scores exactly 1.0 **whether or not a rotor is
    #: there**, and the surface has no way to say "this frame contains only
    #: three combs".  A rotor parked on any above-median structure — a
    #: sideband, an alias, a neighbour's flank — collects real score, which is
    #: why the tracker reliably claims more distinct comb mass than the truth
    #: does (measured 3.80-3.86 against the truth's 2.90-3.77, WP20).
    #:
    #: ``"mad"`` divides by a robust scale of the whole surface
    #: (``1.4826 * MAD``), so a score is in units of "how many noise sigmas
    #: above the floor", a real comb reads 10+ and a spurious placement reads
    #: 1-2.  Claiming a fourth comb that is not there then buys almost nothing,
    #: which is the property the peak form cannot express at any ``lambda_e``.
    norm: str = "peak"  # "peak" | "mad"

    def grid(self) -> np.ndarray:
        return np.arange(self.lo, self.hi + self.step / 2, self.step)


class CombTables(NamedTuple):
    """Per-(grid speed, harmonic) tooth values, bin ids and weights.

    Everything the union-comb emission needs, precomputed once per window:

    - ``v_on`` / ``v_half``: ``(D, K, T)`` the (band-pooled, interpolated)
      whitened value at the on-tooth ``k*c`` and at the half-tooth
      ``(k-0.5)*c`` (the latter clamped at 0, as the contrast has always done);
    - ``bid_on`` / ``bid_half``: ``(D, K)`` the ROUNDED bin index of each tooth,
      which is the identity a union has to deduplicate on;
    - ``w``: ``(D, K)`` per-tooth weight, normalised so that
      ``sum_k w[d, k] == 1`` over that speed's VALID teeth.  That normalisation
      is what makes the union score reduce exactly to ``sum_r S(c_r)`` when the
      four combs are disjoint, so the two are directly comparable and
      ``lambda_e`` keeps its meaning.
    """

    v_on: torch.Tensor
    v_half: torch.Tensor
    bid_on: torch.Tensor
    bid_half: torch.Tensor
    w: torch.Tensor


#: Bin id given to a tooth outside ``[f_min, f_max]``.  Sorts last and carries
#: zero weight, so it can never join a union.
_INVALID_BIN = 1 << 20


def comb_tables(
    lm: torch.Tensor,
    bin_hz: float,
    cfg: EmissionCfg,
    grid: torch.Tensor | None = None,
) -> CombTables:
    """Precompute :class:`CombTables` for every grid speed and harmonic."""
    device, dtype = lm.device, lm.dtype
    g = (
        torch.as_tensor(cfg.grid(), device=device, dtype=dtype)
        if grid is None
        else grid.to(device=device, dtype=dtype)
    )
    n_f = lm.shape[0]
    fmax = min(cfg.f_max, (n_f - 1) * bin_hz)
    ks = torch.arange(1, cfg.k_max + 1, device=device, dtype=dtype)
    if cfg.b0_rps <= 0:
        offs = torch.zeros(1, device=device, dtype=dtype)
    else:
        n_band = cfg.n_band or max(5, int(math.ceil(4.0 * cfg.k_max * cfg.b0_rps / bin_hz)) + 1)
        offs = torch.linspace(-1.0, 1.0, n_band, device=device, dtype=dtype)
    if cfg.k_weight == "k":
        w_k = ks.clone()
    elif cfg.k_weight == "uniform":
        w_k = torch.ones_like(ks)
    else:  # pragma: no cover - guarded by the config surface
        raise ValueError(f"unknown k_weight {cfg.k_weight!r}")

    def pooled(
        mult: torch.Tensor, pos_only: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centre = mult[None, :] * g[:, None]  # (D, K)
        band = offs[None, None, :] * (cfg.b0_rps * ks[None, :, None])  # (1, K, P)
        freqs = centre[:, :, None] + band  # (D, K, P)
        valid = (centre >= cfg.f_min) & (centre <= fmax)  # (D, K)
        idx = freqs.clamp(0.0, fmax) / bin_hz
        j = idx.floor().clamp(0, n_f - 2).long()
        frac = (idx - j).unsqueeze(-1)
        vals = ((1.0 - frac) * lm[j] + frac * lm[j + 1]).amax(dim=2)  # (D, K, T)
        if pos_only:
            vals = vals.clamp_min(0.0)
        vals = torch.where(valid[:, :, None], vals, torch.zeros_like(vals))
        bid = torch.where(
            valid,
            (centre / bin_hz).round().long(),
            torch.full_like(valid, _INVALID_BIN, dtype=torch.long),
        )
        return vals, bid, valid

    v_on, bid_on, valid = pooled(ks, pos_only=False)
    v_half, bid_half, _ = pooled(ks - 0.5, pos_only=True)
    wv = w_k[None, :] * valid.to(dtype)
    w = wv / wv.sum(dim=1, keepdim=True).clamp_min(1e-12)  # (D, K)
    return CombTables(v_on, v_half, bid_on, bid_half, w)


def comb_scores_from_tables(tab: CombTables, cfg: EmissionCfg | None = None) -> torch.Tensor:
    """``(D, T)`` single-rotor comb score, pooled per :attr:`EmissionCfg.pool`.

    The per-tooth contrast ``v_on[k] - v_half[k]`` is the same in every mode;
    only the pooling over ``k`` changes.  ``"mean"`` reproduces the shipped
    weighted mean exactly, so it stays an A/B knob.
    """
    cfg = cfg or EmissionCfg()
    w = tab.w[:, :, None]
    if cfg.pool == "mean":
        return (w * tab.v_on).sum(dim=1) - (w * tab.v_half).sum(dim=1)
    d = tab.v_on - tab.v_half  # (D, K, T) per-tooth contrast
    valid = tab.w > 0  # (D, K) — teeth outside [f_min, f_max] carry no weight
    if cfg.pool == "frac_pos":
        hit = (d > 0).to(d.dtype) * tab.w[:, :, None]
        return hit.sum(dim=1)  # w sums to 1 over valid teeth -> a fraction
    if cfg.pool == "quantile":
        # An invalid tooth must not drag the quantile down, so push it to +inf
        # and take the quantile over the valid count only.
        big = torch.finfo(d.dtype).max
        dv = torch.where(valid[:, :, None], d, torch.full_like(d, big))
        srt, _ = dv.sort(dim=1)
        n_valid = valid.sum(dim=1).clamp_min(1)  # (D,)
        pos = ((n_valid - 1).to(d.dtype) * cfg.pool_q).round().long()  # (D,)
        return srt.gather(1, pos[:, None, None].expand(-1, 1, d.shape[2]))[:, 0]
    raise ValueError(f"unknown pool {cfg.pool!r}")


def _union_keep(tab: CombTables, w_idx: torch.Tensor) -> torch.Tensor:
    """``(N, 4, K)`` mask: which teeth of each assignment survive the union.

    A bin claimed by more than one rotor is awarded to the HIGHEST harmonic
    (the highest-weight claim under ``k_weight="k"``), so every spectrogram bin
    is counted once no matter how many rotors want it.  Factored out of
    :func:`union_emission` because the non-mean pooling modes need the mask
    itself rather than the weighted sum it feeds.
    """
    n_k = tab.v_on.shape[1]
    n = w_idx.shape[0]
    b = tab.bid_on[w_idx].reshape(n, -1)  # (N, 4K)
    key = b * 64 + (n_k - torch.arange(n_k, device=b.device)).repeat(NUM_ROTORS)
    order = key.argsort(dim=1)
    b_s = b.gather(1, order)
    keep_s = torch.ones_like(b_s, dtype=torch.bool)
    keep_s[:, 1:] = b_s[:, 1:] != b_s[:, :-1]
    keep_s &= b_s != _INVALID_BIN
    keep = torch.empty_like(keep_s)
    keep.scatter_(1, order, keep_s)  # back to (rotor, k) order
    return keep.reshape(n, NUM_ROTORS, n_k)


def _pool_kept(d: torch.Tensor, keep: torch.Tensor, cfg: EmissionCfg) -> torch.Tensor:
    """``(N, 4)`` pooled per-rotor score over each rotor's SURVIVING teeth."""
    n_keep = keep.sum(dim=2)  # (N, 4)
    if cfg.pool == "frac_pos":
        return ((d > 0) & keep).sum(dim=2).to(d.dtype) / n_keep.clamp_min(1)
    big = torch.finfo(d.dtype).max
    srt, _ = torch.where(keep, d, torch.full_like(d, big)).sort(dim=2)
    pos = ((n_keep - 1).clamp_min(0).to(d.dtype) * cfg.pool_q).round().long()
    return srt.gather(2, pos[:, :, None])[:, :, 0]


def union_emission(
    tab: CombTables, w_idx: torch.Tensor, t: int, cfg: EmissionCfg | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N,)`` UNION-comb emission of ``N`` four-rotor assignments at frame ``t``.

    ``w_idx`` is ``(N, 4)`` of grid indices.  Every spectrogram bin is counted
    **once**, no matter how many of the four rotors claim it.

    This replaces ``sum_r S(c_r)``, and it is not an optimisation — it is the
    correctness fix.  A sum of single-rotor scores double-counts shared teeth,
    so a comb at ``c/2`` (which shares every SECOND tooth with a comb at ``c``)
    is nearly free to add while sitting 40 rev/s away, where no distance-based
    repulsion can see it.  Measured consequence of the sum form: 52 % of tracks
    landed at a small-integer ratio of a sibling and 45 % below 55 rev/s, i.e.
    the four tracks collapsed onto one comb's subharmonic family (21/42/85).

    Implementation: gather the ``4K`` teeth of each assignment, sort by
    ``(bin id, -k)``, and keep the first of each bin group.  Sorting on
    ``bid * 64 + (k_max - k)`` makes the highest harmonic win a collision, which
    is the highest-weight claim under ``k_weight="k"`` up to the per-speed
    normalisation.  Exact, and — because it runs only on the shortlisted
    proposals — cheap: a ``(4096, 32)`` sort per frame.

    Normalisation is inherited from :class:`CombTables`: disjoint combs give
    exactly ``sum_r S(c_r)``, four identical combs give exactly ``S(c)``.

    Returns ``(score, comb_mass)``.  ``comb_mass`` is the surviving weight sum —
    4.0 when the four combs are disjoint, 1.0 when they coincide — i.e. the
    EFFECTIVE number of distinct combs the assignment explains.  The caller
    needs it because :func:`normalise_scores` is affine, ``(x - med)/denom``,
    and the ``med`` shift has to be applied once per distinct comb: four for a
    disjoint assignment, one for a degenerate one.  Getting that wrong silently
    re-introduces a bias for or against degeneracy.

    Under a non-``"mean"`` :attr:`EmissionCfg.pool` the union cannot be a
    weighted sum, so it is applied per rotor over that rotor's SURVIVING teeth
    and the results are added.  ``comb_mass`` then counts the rotors that still
    hold at least one tooth — the same quantity the affine normalisation needs
    (one median shift per distinct comb), and still 4 for a disjoint assignment
    and 1 for a fully degenerate one.
    """
    cfg = cfg or EmissionCfg()
    n_k = tab.v_on.shape[1]

    if cfg.pool != "mean":
        keep = _union_keep(tab, w_idx)  # (N, 4, K)
        d = tab.v_on[w_idx, :, t] - tab.v_half[w_idx, :, t]  # (N, 4, K)
        # QUALITY x SHARE, and the split is what makes the two ideas compose.
        #
        # Quality is pooled over the rotor's OWN teeth, not over the survivors.
        # Pooling over survivors was the obvious thing and the unit test refuted
        # it: the union deletes a different subset of teeth from every rotor, so
        # a quantile over what is left is not comparable between rotors, and on
        # the four-comb synthetic the spread assignment scored BELOW the
        # degenerate one (-0.022 vs +0.108) — reintroducing the very collapse
        # the union exists to prevent.  Quality is an intrinsic property of the
        # comb hypothesis ("are most of my predicted teeth actually there?"),
        # which is exactly what kills a subharmonic whose odd teeth are absent.
        #
        # Share is the fraction of its own comb mass that no other rotor already
        # claimed, and it alone carries the anti-double-counting.  Four
        # coincident combs then have shares summing to 1.0 and one shared
        # quality, so the assignment scores as ONE comb, as it must.
        own = tab.w[w_idx] > 0  # (N, 4, K) valid teeth of each rotor
        share = (tab.w[w_idx] * keep).sum(dim=2)  # (N, 4), 1.0 if untouched
        return (_pool_kept(d, own, cfg) * share).sum(dim=1), share.sum(dim=1)

    def side(vals: torch.Tensor, bid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = (tab.w[w_idx] * vals[w_idx, :, t]).reshape(w_idx.shape[0], -1)  # (N, 4K)
        b = bid[w_idx].reshape(w_idx.shape[0], -1)
        key = b * 64 + (n_k - torch.arange(n_k, device=b.device)).repeat(NUM_ROTORS)
        order = key.argsort(dim=1)
        b_s, v_s = b.gather(1, order), v.gather(1, order)
        keep = torch.ones_like(b_s, dtype=torch.bool)
        keep[:, 1:] = b_s[:, 1:] != b_s[:, :-1]
        keep &= b_s != _INVALID_BIN
        w_s = tab.w[w_idx].reshape(w_idx.shape[0], -1).gather(1, order)
        return (v_s * keep).sum(dim=1), (w_s * keep).sum(dim=1)

    on, mass = side(tab.v_on, tab.bid_on)
    half, _ = side(tab.v_half, tab.bid_half)
    return on - half, mass


def comb_scores(
    lm: torch.Tensor,
    bin_hz: float,
    cfg: EmissionCfg,
    grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(D, N)`` comb contrast of every grid speed at every frame.

    Args:
        lm: ``(F, N)`` whitened log-magnitude spectrogram.
        bin_hz: frequency resolution of ``lm``.
        cfg: scoring configuration.
        grid: optional explicit ``(D,)`` speed grid (defaults to ``cfg.grid()``).
    """
    device, dtype = lm.device, lm.dtype
    g = (
        torch.as_tensor(cfg.grid(), device=device, dtype=dtype)
        if grid is None
        else grid.to(device=device, dtype=dtype)
    )
    n_f = lm.shape[0]
    fmax = min(cfg.f_max, (n_f - 1) * bin_hz)
    ks = torch.arange(1, cfg.k_max + 1, device=device, dtype=dtype)
    if cfg.b0_rps <= 0:
        offs = torch.zeros(1, device=device, dtype=dtype)
    else:
        n_band = cfg.n_band or max(5, int(math.ceil(4.0 * cfg.k_max * cfg.b0_rps / bin_hz)) + 1)
        offs = torch.linspace(-1.0, 1.0, n_band, device=device, dtype=dtype)
    if cfg.k_weight == "k":
        w = ks / ks.sum()
    elif cfg.k_weight == "uniform":
        w = torch.full_like(ks, 1.0 / len(ks))
    else:  # pragma: no cover - guarded by the config surface
        raise ValueError(f"unknown k_weight {cfg.k_weight!r}")

    def pooled(mult: torch.Tensor, pos_only: bool) -> torch.Tensor:
        """``(D, N)`` weighted mean over k of the band-max at ``mult[k] * c``."""
        centre = mult[None, :] * g[:, None]  # (D, K)
        band = offs[None, None, :] * (cfg.b0_rps * ks[None, :, None])  # (1, K, P)
        freqs = centre[:, :, None] + band  # (D, K, P)
        valid = (centre >= cfg.f_min) & (centre <= fmax)  # (D, K)
        idx = freqs.clamp(0.0, fmax) / bin_hz
        j = idx.floor().clamp(0, n_f - 2).long()
        frac = (idx - j).unsqueeze(-1)
        vals = (1.0 - frac) * lm[j] + frac * lm[j + 1]  # (D, K, P, N)
        vals = vals.amax(dim=2)  # (D, K, N) — band pooling
        if pos_only:
            vals = vals.clamp_min(0.0)
        vals = torch.where(valid[:, :, None], vals, torch.zeros_like(vals))
        wv = w[None, :, None] * valid.to(dtype)[:, :, None]
        return (vals * w[None, :, None]).sum(dim=1) / wv.sum(dim=1).clamp_min(1e-12)

    on = pooled(ks, pos_only=False)
    half = pooled(ks - 0.5, pos_only=True)
    return on - half


def _norm_terms(s: torch.Tensor, cfg: EmissionCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame ``(median, denominator)`` of the score normalisation.

    Shared by :func:`normalise_scores` and :func:`_norm_affine` so the surface
    the beam searches and the union score it accumulates can never drift onto
    two different scales — which would silently change what ``lambda_e`` means.
    """
    med = s.median(dim=0, keepdim=True).values
    if cfg.norm == "peak":
        peak = s.max(dim=0, keepdim=True).values
        glob = (peak - med).median()
        return med, (peak - med).clamp_min(cfg.norm_soft * glob)
    if cfg.norm == "mad":
        mad = 1.4826 * (s - med).abs().median(dim=0, keepdim=True).values
        return med, mad.clamp_min(cfg.norm_soft * mad.median())
    raise ValueError(f"unknown norm {cfg.norm!r}")


def normalise_scores(s: torch.Tensor, cfg: EmissionCfg) -> torch.Tensor:
    """Per-frame normalisation of a ``(D, N)`` score surface.

    Boxcar-smooth over frames, then ``(s - median) / (peak - median)`` with a
    soft floor — the coarse pass's convention (``_norm_smooth``), so the
    emission is scale-free and ``lambda_e`` means the same thing on every
    window.
    """
    s = _smooth_frames(s, cfg)
    med, denom = _norm_terms(s, cfg)
    return (s - med) / denom


def _smooth_frames(s: torch.Tensor, cfg: EmissionCfg) -> torch.Tensor:
    """Boxcar-smooth a ``(D, T)`` surface over frames (the coarse convention)."""
    if cfg.smooth_frames <= 1:
        return s
    k = torch.ones(1, 1, cfg.smooth_frames, device=s.device, dtype=s.dtype) / cfg.smooth_frames
    return torch.nn.functional.conv1d(s[:, None, :], k, padding=cfg.smooth_frames // 2)[
        :, 0, : s.shape[1]
    ]


# --------------------------------------------------------------------------
# Beam search


@dataclass(frozen=True)
class BeamCfg:
    """Beam-search configuration (design §§3-4)."""

    width: int = 256  #: beam width B
    n_global: int = 32  #: states that also get the expensive peak family
    n_peaks: int = 15  #: top-k local maxima per frame
    peak_sep_rps: float = 0.2  #: minimum separation between reported peaks
    #: Grid points per rotor in the local family (the family size is n_local^4).
    #: Measured at 3, not the design's 5: on a real 16 s window the local family
    #: is the dominant cost (5 -> 3 is 123 -> 60 s/window) and 3 is no worse on
    #: the synthetic battery — identical shape correlations and mean error, and
    #: it actually RESOLVES the twin case that 5 and 7 lose, because a wider
    #: per-frame reach lets a track wander onto a neighbour's hump.
    n_local: int = 3
    local_half_rps: float = 1.5  #: local-move window half-width
    #: Emission weight against the transition cost.  Only the ratio to
    #: ``OUPrior.s_scale`` matters; see that field for the measurement.
    lambda_e: float = 3.0
    #: Rotor-band prior ON THE ASSIGNMENT (not on seed bases): four rotors of
    #: ONE drone lie within a narrow speed ratio.  ``span_soft`` is the shipped
    #: seeder value (`SeedConfig.r_span_max`); beyond it the assignment pays
    #: ``span_gain * (span - span_soft)^2`` per frame, and beyond ``span_hard``
    #: it is rejected outright.  Soft rather than hard at 1.45 because a takeoff
    #: legitimately widens the spread while the rotors spin up at different
    #: rates; the widest set any real protocol window seeds is 1.31.
    #:
    #: This is the cheap half of the subharmonic fix — the `21/42/85` family
    #: spans 4.0 and dies here before the union score is even evaluated.
    span_soft: float = 1.45
    span_gain: float = 4.0
    span_hard: float = 2.5
    #: Beam de-duplication resolution, rev/s.  Without it the beam fills with
    #: near-identical copies of one hypothesis and silently loses diversity.
    dedup_rps: float = 0.25
    #: De-duplicate on the SORTED speed vector, so two states that differ only
    #: by which row holds which rotor occupy one beam slot instead of two.
    #:
    #: **Not exact, and the reason is worth stating** — it looked exact and the
    #: unit test refuted it.  The emission and the transition cost ARE
    #: permutation-invariant (a relabelling permutes ``m_from`` and ``m_to``
    #: together and the differential modes share one scale, see
    #: :class:`OUPrior`), but ``mu`` is inherited from each state's OWN
    #: ancestry, so two states that are permutations of each other NOW may carry
    #: trims that are not permutations of each other, and their futures differ.
    #: Merging keeps the cheaper one and discards a live hypothesis: measured
    #: ``final_cost_best`` -226.08 with the merge against -227.04 without.
    #:
    #: It is offered because the trade may still be worth taking — the beam has
    #: only ``width`` slots for a ``len(grid)^4`` space and the global family
    #: manufactures permuted duplicates by construction (best-of-24 per PARENT,
    #: and different parents order the same physical set differently) — but it
    #: costs objective value, so it is OFF until a sweep shows it buys accuracy.
    dedup_sorted: bool = False
    #: Beam slots reserved for MARGINAL coverage rather than joint cost.
    #:
    #: The failure this addresses is representational, and it is the reason a
    #: joint beam can lose to an exactly-solved 1-D DP.  ``width`` states drawn
    #: by joint cost from a ``len(grid)^4`` space give each rotor only
    #: ``width^(1/4)`` distinct values — 4 values per rotor at ``width = 256``.
    #: A rotor whose evidence is momentarily weak has its alternatives crowded
    #: out by states that differ only in a rotor that is doing well.  With a
    #: reserve, the cheapest state holding each distinct value of each sorted
    #: slot is admitted regardless of its joint rank, so every rotor keeps a
    #: live set of alternatives to recover onto.  ``0`` = the plain top-B beam.
    diversity_reserve: int = 0
    #: ``"frame0"`` freezes each state's differential trim at its own frame-0
    #: assignment; ``"running"`` tracks it with an exponential mean of length
    #: ``mu_tau_s`` (more robust to a bad frame 0, weaker against slow drift).
    mu_mode: str = "frame0"
    mu_tau_s: float = 8.0


#: Radix for hashing a rounded 4-rotor state into one int64.  The grid tops out
#: at 120 rev/s and `dedup_rps` is >= 0.05, so an index never reaches 4096.
_HASH_MULT: torch.Tensor = torch.tensor([1, 4096, 4096**2, 4096**3], dtype=torch.int64)

_PERMS: torch.Tensor = torch.tensor(
    list(itertools.permutations(range(NUM_ROTORS))), dtype=torch.long
)


def _subsets(n: int, k: int = NUM_ROTORS) -> torch.Tensor:
    return torch.tensor(list(itertools.combinations(range(n), k)), dtype=torch.long)


def _frame_peaks(s_t: torch.Tensor, grid: torch.Tensor, cfg: BeamCfg) -> torch.Tensor:
    """``(n_peaks,)`` candidate speeds from one frame's score column.

    Two passes, and the second one is load-bearing:

    1. **strict local maxima**, best first — one candidate per hump, so every
       distinct comb in the frame is proposable;
    2. **the best remaining grid points**, to fill the quota.

    Pass 2 exists because a genuine twin pair does NOT produce two maxima: on
    a 7.8 Hz-bin spectrogram scored to k <= 8 the resolution is about 1 rev/s,
    so FLY124's 73.96/74.85 pair is a single broad hump and its second rotor
    lives on the FLANK.  With maxima only, the peak family cannot propose the
    pair at all; measured on the twin unit test, the surface had just five
    strict maxima (two of them grid-edge junk), so every 4-subset was forced to
    contain junk and the tracker parked a track at 101 rev/s.

    Both passes are non-maximum-suppressed at ``peak_sep_rps``.
    """
    s = s_t.detach().cpu().numpy()
    g = grid.detach().cpu().numpy()
    is_max = np.zeros(len(s), dtype=bool)
    is_max[1:-1] = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:])
    if len(s) > 1:
        is_max[0], is_max[-1] = s[0] >= s[1], s[-1] >= s[-2]
    keep: list[int] = []

    def add(pool: np.ndarray) -> None:
        for i in pool[np.argsort(-s[pool])]:
            if len(keep) >= cfg.n_peaks:
                return
            if all(abs(g[i] - g[j]) >= cfg.peak_sep_rps for j in keep):
                keep.append(int(i))

    add(np.flatnonzero(is_max))
    add(np.flatnonzero(~is_max))
    return grid[torch.as_tensor(keep, device=grid.device, dtype=torch.long)]


def _mode_cost(
    dw_from: torch.Tensor,
    w_to: torch.Tensor,
    mu: torch.Tensor,
    a: torch.Tensor,
    s: torch.Tensor,
    huber_knee: float,
) -> torch.Tensor:
    """OU transition cost of ``w_to`` given the previous speeds ``dw_from``.

    All tensors broadcast on a leading set of batch dimensions; the last
    dimension is the rotor axis (4).  ``mu`` is in MODE space (4,) and
    broadcasts the same way.
    """
    b = torch.as_tensor(MIXER, device=w_to.device, dtype=w_to.dtype) / NUM_ROTORS
    m_to = w_to @ b
    m_from = dw_from @ b
    resid = m_to - a * m_from - (1.0 - a) * mu
    z = resid / s
    cost = 0.5 * z[..., 1:] ** 2
    zc = z[..., 0].abs()
    c0 = torch.where(
        zc <= huber_knee,
        0.5 * zc**2,
        huber_knee * (zc - 0.5 * huber_knee),
    )
    return c0 + cost.sum(dim=-1)


def _to_idx(w: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Grid indices of a speed assignment.  ``w`` is ``(..., 4)``."""
    return ((w - grid[0]) / (grid[1] - grid[0])).round().long().clamp(0, len(grid) - 1)


def _sum_scores(s_t: torch.Tensor, grid: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``sum_r S(c_r)`` — the (double-counting) upper bound on the union score."""
    return s_t[_to_idx(w, grid)].sum(dim=-1)


def _norm_affine(s: torch.Tensor, cfg: EmissionCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame ``(scale, shift)`` of :func:`normalise_scores`.

    The union score is computed from the RAW tables, so it has to be put on the
    normalised scale by hand.  ``normalise_scores`` is affine per frame,
    ``(x - med) / denom``; a union of four combs carries four copies of the
    shift, hence the ``NUM_ROTORS * shift`` at the call sites.  Frame smoothing
    is skipped here — it mixes frames and has no per-frame affine form — so with
    ``smooth_frames > 1`` the union is normalised by the UNSMOOTHED statistics.
    Both terms then use one consistent scale, which is what ``lambda_e`` needs.
    """
    med, denom = _norm_terms(s, cfg)
    return (1.0 / denom)[0], (-med / denom)[0]


def _band_penalty(w: torch.Tensor, cfg: BeamCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """``(penalty, reject)`` of the rotor-band prior.  ``w`` is ``(..., 4)``."""
    span = w.amax(dim=-1) / w.amin(dim=-1).clamp_min(1e-6)
    pen = cfg.span_gain * (span - cfg.span_soft).clamp_min(0.0) ** 2
    return pen, span > cfg.span_hard


class Objective(NamedTuple):
    """Everything needed to evaluate the tracker's cost at ANY trajectory.

    Built once per window by :func:`build_objective`; consumed both by
    :func:`joint_beam_track` (which searches it) and by
    :func:`score_trajectory` (which evaluates it at a trajectory somebody else
    produced — the ground truth, or a competing stage's output).

    That second use is the whole point.  A search and its objective fail in
    opposite directions and the fix is different for each, so the two must be
    told apart before anything is tuned: if the TRUE trajectory scores CHEAPER
    than the tracker's own output, the objective is right and the search lost
    it; if it scores DEARER, no amount of search will help.
    """

    tab: CombTables
    scores: torch.Tensor  # (D, T) normalised single-comb surface
    nrm_a: torch.Tensor  # (T,) per-frame affine scale of `normalise_scores`
    nrm_b: torch.Tensor  # (T,) per-frame affine shift, PER DISTINCT COMB
    grid: torch.Tensor
    a: torch.Tensor
    s: torch.Tensor
    b_mix: torch.Tensor
    ou: OUPrior
    emis: EmissionCfg
    beam: BeamCfg


def build_objective(
    lm: np.ndarray | torch.Tensor,
    bin_hz: float,
    *,
    ou: OUPrior | None = None,
    emis: EmissionCfg | None = None,
    beam: BeamCfg | None = None,
    device: str = "cpu",
) -> Objective:
    """Assemble the emission tables, the normalisation and the OU coefficients."""
    ou = ou or OUPrior()
    emis = emis or EmissionCfg()
    beam = beam or BeamCfg()
    dev = torch.device(device)
    lm_t = (
        lm.to(device=dev, dtype=torch.float32)
        if isinstance(lm, torch.Tensor)
        else torch.as_tensor(np.ascontiguousarray(lm), device=dev, dtype=torch.float32)
    )
    grid = torch.as_tensor(emis.grid(), device=dev, dtype=torch.float32)
    tab = comb_tables(lm_t, bin_hz, emis, grid)
    raw = comb_scores_from_tables(tab, emis)
    scores = normalise_scores(raw, emis)
    nrm_a, nrm_b = _norm_affine(raw, emis)
    a_np, s_np = ou.coefficients()
    return Objective(
        tab=tab,
        scores=scores,
        nrm_a=nrm_a,
        nrm_b=nrm_b,
        grid=grid,
        a=torch.as_tensor(a_np, device=dev, dtype=torch.float32),
        s=torch.as_tensor(s_np, device=dev, dtype=torch.float32),
        b_mix=torch.as_tensor(MIXER, device=dev, dtype=torch.float32) / NUM_ROTORS,
        ou=ou,
        emis=emis,
        beam=beam,
    )


def score_trajectory(obj: Objective, w: np.ndarray) -> dict[str, Any]:
    """Total and per-term cost of a ``(4, T)`` trajectory under ``obj``.

    ``w`` is in rev/s on the spectrogram's OWN frame grid and is snapped to the
    emission grid, exactly as every beam state is.  The accumulation is the
    tracker's, term for term: ``-lambda_e * e_union`` per frame, the rotor-band
    penalty per frame, and the OU transition cost between consecutive frames
    with ``mu`` taken from frame 0 (``mu_mode="frame0"``) or tracked
    (``"running"``), so ``total`` is directly comparable to the tracker's
    ``final_cost_best``.
    """
    dev = obj.grid.device
    w_t = torch.as_tensor(np.ascontiguousarray(w.T), device=dev, dtype=torch.float32)  # (T, 4)
    n_t = min(w_t.shape[0], obj.scores.shape[1])
    w_t = w_t[:n_t]
    idx = _to_idx(w_t, obj.grid)
    w_snap = obj.grid[idx]  # on-grid, which is what the beam can actually hold
    emis_t = torch.zeros(n_t, device=dev)
    mass_t = torch.zeros(n_t, device=dev)
    for t in range(n_t):
        u, m = union_emission(obj.tab, idx[t : t + 1], t, obj.emis)
        emis_t[t] = obj.nrm_a[t] * u[0] + m[0] * obj.nrm_b[t]
        mass_t[t] = m[0]
    pen, rej = _band_penalty(w_snap, obj.beam)
    a_mu = math.exp(-obj.ou.dt / obj.beam.mu_tau_s) if obj.beam.mu_mode == "running" else 1.0
    mu = w_snap[0] @ obj.b_mix
    trans = torch.zeros(n_t, device=dev)
    for t in range(1, n_t):
        trans[t] = _mode_cost(w_snap[t - 1], w_snap[t], mu, obj.a, obj.s, obj.ou.huber_knee)
        mu = a_mu * mu + (1.0 - a_mu) * (w_snap[t] @ obj.b_mix)
    e_term = -obj.beam.lambda_e * emis_t
    total = float((e_term + pen + trans).sum())
    return {
        "total": total,
        "emission": float(e_term.sum()),
        "transition": float(trans.sum()),
        "band": float(pen.sum()),
        "n_rejected_frames": int(rej.sum()),
        "comb_mass_mean": float(mass_t.mean()),
        "per_frame": {
            "emission": e_term.detach().cpu().numpy(),
            "transition": trans.detach().cpu().numpy(),
            "band": pen.detach().cpu().numpy(),
        },
    }


def joint_beam_track(
    lm: np.ndarray,
    bin_hz: float,
    st: np.ndarray,
    ft: np.ndarray,
    *,
    ou: OUPrior | None = None,
    emis: EmissionCfg | None = None,
    beam: BeamCfg | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Track four rotor speeds jointly by beam search.  ``(4, len(ft))``, diag.

    Args:
        lm: ``(F, N)`` whitened spectrogram (``beatvk_vk_arms._coarse_spec``).
        bin_hz: its frequency resolution.
        st: ``(N,)`` frame times of ``lm``.
        ft: output frame grid (the scorer's 32 ms grid).
        ou / emis / beam: configuration; defaults are the WP16/WP17 values.
        device: torch device.
    """
    ou = ou or OUPrior()
    emis = emis or EmissionCfg()
    beam = beam or BeamCfg()
    dev = torch.device(device)
    lm_t = torch.as_tensor(np.ascontiguousarray(lm), device=dev, dtype=torch.float32)
    grid = torch.as_tensor(emis.grid(), device=dev, dtype=torch.float32)
    tab = comb_tables(lm_t, bin_hz, emis, grid)
    raw = comb_scores_from_tables(tab, emis)
    scores = normalise_scores(raw, emis)  # (D, T) — peaks + the shortlist bound
    # The union score is built from the SAME tables, so it must carry the same
    # per-frame normalisation or `lambda_e` would mean two different things.
    nrm_a, nrm_b = _norm_affine(raw, emis)  # per-frame affine of normalise_scores
    n_t = scores.shape[1]

    a_np, s_np = ou.coefficients()
    a = torch.as_tensor(a_np, device=dev, dtype=torch.float32)
    s = torch.as_tensor(s_np, device=dev, dtype=torch.float32)
    b_mix = torch.as_tensor(MIXER, device=dev, dtype=torch.float32) / NUM_ROTORS
    perms = _PERMS.to(dev)
    step = float(grid[1] - grid[0])
    n_win = int(round(beam.local_half_rps / step))
    win = torch.arange(-n_win, n_win + 1, device=dev)
    a_mu = math.exp(-ou.dt / beam.mu_tau_s) if beam.mu_mode == "running" else 1.0

    # -- frame 0: all 4-subsets of the peak set, sorted (no previous state)
    pk0 = _frame_peaks(scores[:, 0], grid, beam)
    sets0 = pk0[_subsets(len(pk0)).to(dev)]  # (Q, 4)
    sets0, _ = torch.sort(sets0, dim=-1)
    idx0 = _to_idx(sets0, grid)
    pen0, rej0 = _band_penalty(sets0, beam)
    u0, m0 = union_emission(tab, idx0, 0, emis)
    e0 = nrm_a[0] * u0 + m0 * nrm_b[0]
    cost0 = -beam.lambda_e * e0 + pen0
    cost0 = torch.where(rej0, torch.full_like(cost0, float("inf")), cost0)
    keep = torch.topk(-cost0, min(beam.width, len(cost0))).indices
    cur_w = sets0[keep].double().float()
    cur_c = cost0[keep].double()
    mu = cur_w @ b_mix  # (B, 4) each state's own trim
    n_b = cur_w.shape[0]

    states = torch.zeros(n_t, beam.width, NUM_ROTORS, device=dev)
    parents = torch.zeros(n_t, beam.width, dtype=torch.long, device=dev)
    states[0, :n_b] = cur_w
    diag: dict[str, Any] = {
        "n_frames": int(n_t),
        "grid": [float(emis.lo), float(emis.hi), float(emis.step)],
        "a": [round(float(v), 6) for v in a_np],
        "s": [round(float(v), 4) for v in s_np],
        "sustained_2rps": {k: round(v, 4) for k, v in ou.sustained_cost_rate(2.0).items()},
        "b0_rps": emis.b0_rps,
        "k_weight": emis.k_weight,
    }
    n_distinct: list[int] = []
    marg: list[int] = []
    n_rejected: list[float] = []
    shared: list[float] = []

    for t in range(1, n_t):
        s_t = scores[:, t]
        # ---- family (b): local moves (rotor identity preserved)
        base_idx = ((cur_w - grid[0]) / step).round().long()  # (B, 4)
        cand_idx = (base_idx[:, :, None] + win[None, None, :]).clamp(0, len(grid) - 1)
        cand_s = s_t[cand_idx]  # (B, 4, W)
        cand_s[:, :, n_win] = torch.inf  # force-include "stay" -> coasting
        top = torch.topk(cand_s, min(beam.n_local, cand_s.shape[2]), dim=2).indices
        loc = grid[torch.gather(cand_idx, 2, top)]  # (B, 4, m)
        m = loc.shape[2]
        mesh = torch.stack(
            torch.meshgrid(*[torch.arange(m, device=dev)] * NUM_ROTORS, indexing="ij"),
            dim=-1,
        ).reshape(-1, NUM_ROTORS)  # (m^4, 4)
        w_loc = torch.gather(loc, 2, mesh.t()[None, :, :].expand(n_b, -1, -1)).permute(
            0, 2, 1
        )  # (B, m^4, 4)
        c_loc = (
            cur_c[:, None]
            + _mode_cost(cur_w[:, None, :], w_loc, mu[:, None, :], a, s, ou.huber_knee).double()
        )
        prop_w = w_loc.reshape(-1, NUM_ROTORS)
        prop_base = c_loc.reshape(-1)
        prop_p = torch.arange(n_b, device=dev).repeat_interleave(w_loc.shape[1])

        # ---- family (a): global peak assignments, exact best-of-24
        n_g = min(beam.n_global, n_b)
        gsel = torch.topk(-cur_c, n_g).indices
        pk = _frame_peaks(s_t, grid, beam)
        sets = pk[_subsets(len(pk)).to(dev)]  # (Q, 4)
        w_perm = sets[:, perms]  # (Q, 24, 4)
        c_tr = _mode_cost(
            cur_w[gsel][:, None, None, :],
            w_perm[None],
            mu[gsel][:, None, None, :],
            a,
            s,
            ou.huber_knee,
        )  # (n_g, Q, 24)
        best_c, best_p = c_tr.min(dim=2)
        w_best = w_perm[torch.arange(len(sets), device=dev)[None, :], best_p]  # (n_g,Q,4)
        c_gl = cur_c[gsel][:, None] + best_c.double()
        prop_w = torch.cat([prop_w, w_best.reshape(-1, NUM_ROTORS)])
        prop_base = torch.cat([prop_base, c_gl.reshape(-1)])
        prop_p = torch.cat([prop_p, gsel.repeat_interleave(w_best.shape[1])])

        # ---- rotor-band prior, then shortlist on the CHEAP bound, then
        #      re-rank the shortlist with the EXACT union-comb emission.
        #
        # `sum_r S(c_r)` upper bounds the union score (it double-counts shared
        # teeth and can never undercount), so ranking by it is an optimistic
        # bound and the shortlist is admissible.  The band prior is applied
        # FIRST and on every proposal because it is pure arithmetic and it is
        # what removes the subharmonic family before it can crowd the
        # shortlist — a `21/42/85` assignment spans 4.0 and is rejected outright.
        pen, rej = _band_penalty(prop_w, beam)
        prop_base = prop_base + pen.double()
        # The union is evaluated on EVERY surviving proposal, not on a
        # shortlist.  A shortlist has to be ranked by something cheap, and the
        # only cheap thing available is `sum_r S(c_r)` — which is precisely the
        # double-counting score the union exists to replace, so duplicates
        # inflate their own bound and crowd the shortlist out.  Measured: with
        # a 16x shortlist the synthetic four-shape battery dropped from 4/4
        # rotors tracked to 1/4, while exact evaluation restores it.  With the
        # band prior filtering first and `n_local = 3` the proposal count is
        # ~64k per frame and the extra cost is one (64k, 32) sort.
        keep_p = ~rej
        prop_w, prop_base, prop_p = prop_w[keep_p], prop_base[keep_p], prop_p[keep_p]
        if len(prop_w) == 0:  # pragma: no cover - the band prior never empties
            prop_w, prop_base, prop_p = cur_w, cur_c, torch.arange(n_b, device=dev)
            prop_c = prop_base
        else:
            u_raw, u_mass = union_emission(tab, _to_idx(prop_w, grid), t, emis)
            e_union = nrm_a[t] * u_raw + u_mass * nrm_b[t]
            prop_c = prop_base - (beam.lambda_e * e_union).double()
            shared.append(float((_sum_scores(s_t, grid, prop_w) - e_union).clamp_min(0).mean()))
        n_rejected.append(float(rej.float().mean()))

        # ---- keep the cheapest, de-duplicate, keep B
        #
        # Order matters for cost, not just correctness: a full argsort +
        # `np.unique(axis=0)` over the ~200k proposals of one frame is a lexsort
        # of a (200k, 4) array and measured at 72 % of the whole tracker's
        # runtime.  Shortlisting with topk first and hashing the rounded state
        # into ONE int64 makes the unique a 1-D operation over `width * 8` rows.
        n_short = len(prop_c)
        short = torch.argsort(prop_c)  # cost-ascending
        key = (prop_w[short] / beam.dedup_rps).round().to(torch.int64)
        if beam.dedup_sorted:
            key, _ = key.sort(dim=-1)
        key1 = (key * _HASH_MULT.to(dev)).sum(dim=-1)
        _, inv = torch.unique(key1, return_inverse=True)
        rank = torch.arange(n_short, device=dev)
        first = torch.full((int(inv.max()) + 1,), n_short, dtype=torch.long, device=dev)
        first.scatter_reduce_(0, inv, rank, reduce="amin")
        first = first.sort().values
        n_distinct.append(int(len(first)))
        n_take = min(beam.width, len(first))
        chosen = first[:n_take]
        if beam.diversity_reserve > 0 and len(first) > n_take:
            # Marginal coverage: the top-B by joint cost can hold four values of
            # one rotor and one of another, so a rotor with momentarily weak
            # evidence loses every alternative it would need to recover onto.
            # Reserve slots for the cheapest state carrying each value of each
            # SORTED slot that the plain top-B does not already carry, so the
            # beam's per-rotor marginal stays wide even when its joint cost does
            # not justify it.  Sorted slots (not rows) because rotor identity is
            # arbitrary — see `dedup_sorted`.
            n_keep = max(n_take - beam.diversity_reserve, 1)
            head, tail = chosen[:n_keep], first[n_keep:]
            have = key[head].reshape(-1) * NUM_ROTORS + torch.arange(NUM_ROTORS, device=dev).repeat(
                len(head)
            )
            have_set = torch.unique(have)
            cand = key[tail] * NUM_ROTORS + torch.arange(NUM_ROTORS, device=dev)[None, :]
            novel = ~torch.isin(cand, have_set)  # (n_tail, 4)
            # cost-ascending already, so the first hit for a value is its cheapest
            extra = tail[novel.any(dim=1)][: n_take - n_keep]
            chosen = torch.cat([head, extra])
        sel = short[chosen]

        cur_w = prop_w[sel]
        cur_c = prop_c[sel]
        par = prop_p[sel]
        # How wide is the beam's PER-ROTOR marginal?  `width` states drawn by
        # joint cost from a `len(grid)^4` space give each rotor about
        # `width^(1/4)` values, so this is the number that says whether the beam
        # is representing a distribution or a point.  Reported as the MINIMUM
        # over the four sorted slots, because a beam is only as good as the
        # rotor it covers worst.
        slot_k = (cur_w.sort(dim=-1).values / beam.dedup_rps).round().to(torch.int64)
        marg.append(min(int(slot_k[:, i].unique().numel()) for i in range(NUM_ROTORS)))
        mu = a_mu * mu[par] + (1.0 - a_mu) * (cur_w @ b_mix)
        n_b = cur_w.shape[0]
        states[t, :n_b] = cur_w
        parents[t, :n_b] = par

    # -- backtrack
    best = int(torch.argmin(cur_c).item())
    path = torch.zeros(n_t, NUM_ROTORS, device=dev)
    node = best
    for t in range(n_t - 1, -1, -1):
        path[t] = states[t, node]
        node = int(parents[t, node].item())
    traj_st = path.t().cpu().numpy()  # (4, N)
    traj = np.stack([np.interp(ft, st, row) for row in traj_st])

    fc = cur_c.cpu().numpy()
    diag.update(
        {
            "final_cost_best": float(fc.min()),
            "final_cost_spread": float(fc.max() - fc.min()),
            "final_cost_p90_minus_best": float(np.percentile(fc, 90) - fc.min()),
            "beam_distinct_mean": float(np.mean(n_distinct)) if n_distinct else 0.0,
            "beam_marginal_min_mean": float(np.mean(marg)) if marg else 0.0,
            "beam_marginal_min_worst": int(np.min(marg)) if marg else 0,
            "beam_distinct_min": int(np.min(n_distinct)) if n_distinct else 0,
            "band_rejected_frac": float(np.mean(n_rejected)) if n_rejected else 0.0,
            "shared_evidence_mean": float(np.mean(shared)) if shared else 0.0,
            "means": [round(float(v), 3) for v in np.sort(traj.mean(axis=1))],
            "stds": [round(float(v), 3) for v in traj.std(axis=1)],
        }
    )
    return traj, diag
