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
from typing import Any

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
    f_min: float = 20.0  # COARSE_F_MIN — keeps the k1/k2 teeth
    f_max: float = 6000.0
    #: Frames of boxcar smoothing before the per-frame normalisation.
    smooth_frames: int = 3
    #: Soft floor on the per-frame normalisation denominator, x the global
    #: median contrast (COARSE_NORM_SOFT).
    norm_soft: float = 0.3

    def grid(self) -> np.ndarray:
        return np.arange(self.lo, self.hi + self.step / 2, self.step)


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


def normalise_scores(s: torch.Tensor, cfg: EmissionCfg) -> torch.Tensor:
    """Per-frame normalisation of a ``(D, N)`` score surface.

    Boxcar-smooth over frames, then ``(s - median) / (peak - median)`` with a
    soft floor — the coarse pass's convention (``_norm_smooth``), so the
    emission is scale-free and ``lambda_e`` means the same thing on every
    window.
    """
    if cfg.smooth_frames > 1:
        k = torch.ones(1, 1, cfg.smooth_frames, device=s.device, dtype=s.dtype)
        k = k / cfg.smooth_frames
        s = torch.nn.functional.conv1d(s[:, None, :], k, padding=cfg.smooth_frames // 2)[
            :, 0, : s.shape[1]
        ]
    med = s.median(dim=0, keepdim=True).values
    peak = s.max(dim=0, keepdim=True).values
    glob = (peak - med).median()
    return (s - med) / (peak - med).clamp_min(cfg.norm_soft * glob)


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
    #: Width of the shared-evidence (overlap) kernel, rev/s.  See
    #: :func:`_emission_net` — this is the ONLY overlap knob, and it is a
    #: property of the score surface (how wide a comb peak is), not a taste
    #: parameter.  A k <= 8 comb on 7.8 Hz bins resolves to about ``bin/k_max``
    #: ~ 1 rev/s, so 0.8 is a half-width of that scale.
    overlap_sigma_rps: float = 0.8
    #: Multiplier on the overlap correction.  1.0 removes exactly the
    #: double-counted evidence; > 1 additionally repels.
    overlap_gain: float = 1.0
    #: Beam de-duplication resolution, rev/s.  Without it the beam fills with
    #: near-identical copies of one hypothesis and silently loses diversity.
    dedup_rps: float = 0.25
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


def _overlap(s_r: torch.Tensor, w: torch.Tensor, cfg: BeamCfg) -> torch.Tensor:
    """Evidence that ``sum_r S(c_r)`` counts twice.  ``w``, ``s_r`` are ``(..., 4)``.

    ``P_overlap = gain * sum_{r<s} max(min(S_r, S_s), 0) * exp(-d^2 / 2 sigma^2)``

    Design §2 offered a distance-only soft repulsion (v1) and left the exact
    union-comb score as a follow-up (v2).  v1 measurably does not work here,
    and the reason is a genuine conflict rather than bad tuning: the emission
    ``sum_r S(c_r, t)`` double-counts one comb's evidence out to the width of
    the score peak (~1-2 rev/s on a k <= 8 comb over 7.8 Hz bins), but a
    distance-only penalty wide enough to cover that also charges the GENUINE
    twins that real quadrotors run 0.5-2 rev/s apart.  Measured on the unit
    tests: at ``r1 = 1.0`` the tracker put three tracks across one hump and
    dropped a real 96 rev/s rotor; at ``r1 = 2.0`` a real twin pair pays a
    constant ~0.46/frame just for existing.

    This form resolves the conflict because it is not a repulsion at all — it
    is a correction for the specific quantity that is wrong.  At zero
    separation it removes exactly the duplicated score (``gain = 1`` cancels the
    double count identically), and where the scores are low it is small, so a
    track is never pushed onto empty grid to escape it.  It is the cheap
    diagonal of the union-comb score: no extra spectrogram reads, just the
    per-rotor values already gathered.
    """
    iu = torch.triu_indices(NUM_ROTORS, NUM_ROTORS, offset=1, device=w.device)
    # Index the 6 pairs BEFORE the exp: on the local family this is a
    # (B, m^4, 4, 4) tensor, and doing it the other way spends 2.7x the work.
    lo = torch.minimum(s_r[..., iu[0]], s_r[..., iu[1]]).clamp_min(0.0)
    d = w[..., iu[0]] - w[..., iu[1]]
    k = torch.exp(-(d**2) / (2.0 * cfg.overlap_sigma_rps**2))
    return cfg.overlap_gain * (lo * k).sum(dim=-1)


def _emission_net(
    s_t: torch.Tensor, grid: torch.Tensor, w: torch.Tensor, cfg: BeamCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(net emission, overlap penalty)`` of an assignment.  ``w`` is ``(..., 4)``."""
    idx = ((w - grid[0]) / (grid[1] - grid[0])).round().long().clamp(0, len(grid) - 1)
    s_r = s_t[idx]
    ov = _overlap(s_r, w, cfg)
    return s_r.sum(dim=-1) - ov, ov


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
    scores = normalise_scores(comb_scores(lm_t, bin_hz, emis, grid), emis)  # (D, T)
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
    e0, _ = _emission_net(scores[:, 0], grid, sets0, beam)
    cost0 = -beam.lambda_e * e0
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
    rep_active: list[float] = []

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
            + (-beam.lambda_e * _emission_net(s_t, grid, w_loc, beam)[0]).double()
        )
        prop_w = w_loc.reshape(-1, NUM_ROTORS)
        prop_c = c_loc.reshape(-1)
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
        c_gl = (
            cur_c[gsel][:, None]
            + best_c.double()
            + (-beam.lambda_e * _emission_net(s_t, grid, w_best, beam)[0]).double()
        )
        prop_w = torch.cat([prop_w, w_best.reshape(-1, NUM_ROTORS)])
        prop_c = torch.cat([prop_c, c_gl.reshape(-1)])
        prop_p = torch.cat([prop_p, gsel.repeat_interleave(w_best.shape[1])])

        # ---- keep the cheapest, de-duplicate, keep B
        #
        # Order matters for cost, not just correctness: a full argsort +
        # `np.unique(axis=0)` over the ~200k proposals of one frame is a lexsort
        # of a (200k, 4) array and measured at 72 % of the whole tracker's
        # runtime.  Shortlisting with topk first and hashing the rounded state
        # into ONE int64 makes the unique a 1-D operation over `width * 8` rows.
        n_short = min(len(prop_c), beam.width * 8)
        short = torch.topk(-prop_c, n_short, sorted=True).indices  # cost-ascending
        key = (prop_w[short] / beam.dedup_rps).round().to(torch.int64)
        key1 = (key * _HASH_MULT.to(dev)).sum(dim=-1)
        _, inv = torch.unique(key1, return_inverse=True)
        rank = torch.arange(n_short, device=dev)
        first = torch.full((int(inv.max()) + 1,), n_short, dtype=torch.long, device=dev)
        first.scatter_reduce_(0, inv, rank, reduce="amin")
        first = first.sort().values
        n_distinct.append(int(len(first)))
        sel = short[first[: min(beam.width, len(first))]]

        cur_w = prop_w[sel]
        cur_c = prop_c[sel]
        par = prop_p[sel]
        mu = a_mu * mu[par] + (1.0 - a_mu) * (cur_w @ b_mix)
        n_b = cur_w.shape[0]
        states[t, :n_b] = cur_w
        parents[t, :n_b] = par
        ov = _emission_net(s_t, grid, cur_w, beam)[1]
        rep_active.append(float((ov > 0.05).float().mean()))

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
            "beam_distinct_min": int(np.min(n_distinct)) if n_distinct else 0,
            "repulsion_active_frac": float(np.mean(rep_active)) if rep_active else 0.0,
            "means": [round(float(v), 3) for v in np.sort(traj.mean(axis=1))],
            "stds": [round(float(v), 3) for v in traj.std(axis=1)],
        }
    )
    return traj, diag
