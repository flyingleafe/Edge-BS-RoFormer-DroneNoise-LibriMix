"""Rank-one-plus-diagonal covariance of the per-harmonic rate opinions.

WP18 of ``docs/experiments/rps-refine-precision.md``.  The Fisher-weighted
phase-slope stage (``vk_tracking._freq_update``) weights harmonic ``k`` by
``k^2 |p_k|``, a weight derived by assuming each harmonic's phase error is
independent additive noise.  Under **arrival-time jitter** harmonic ``k``'s
phase error is ``2 pi k r n_t``; dividing by ``k`` to get a *rate* opinion
cancels the ``k``, so that term is IDENTICAL across harmonics.  Writing each
harmonic's opinion as

    delta_hat_k(t) = delta(t) + J(t) + e_k(t),

the covariance across harmonics is **rank-one plus diagonal**

    Sigma = sigma_J^2 * 1 1^T + diag(v_k),

which leaves the relative weights at ``w_k ∝ 1/v_k`` but replaces the fused
variance ``1/W`` by ``1/W + sigma_J^2`` — an irreducible floor, hence a
saturation harmonic ``k*`` beyond which more harmonics buy nothing.

This module MEASURES that covariance.  Nothing here assumes an exponent.

Estimator, per (window, rotor, channel)
---------------------------------------
1. demodulate the audio along the window's trajectory at harmonics
   ``k = 1..K``, brickwall to ``+-B_wide`` and decimate to ``fs_env``
   (:func:`demod_rotor`);
2. per *arm* (a band schedule ``B_k``) brickwall the complex envelope down to
   ``B_k`` — exactly equivalent to demodulating in that band, since brickwall
   lowpasses commute — form
   ``delta_hat_k[t] = arg(z_k[t+1] conj(z_k[t])) fs_env / (2 pi k)``, optionally
   time-high-pass it above ``f_c``, and take the K x K empirical covariance
   across harmonics with pairwise-complete observations
   (:func:`arm_covariance`);
3. fit rank-one-plus-diagonal by moments: every off-diagonal entry equals the
   common variance in expectation, so ``sigma_J^2 = mean(off-diagonals)`` and
   ``v_k = C_kk - sigma_J^2`` (:func:`fit_rank_one`).

Two details that the naive recipe gets wrong and this one does not:

* **``v_k`` needs no high-pass.**  The true trajectory error ``delta`` is
  common across harmonics, so it lands in the rank-one term and cancels out of
  ``v_k = C_kk - common_k`` regardless of cutoff.  ``v_k`` is therefore always
  read at ``f_c = 0`` (the FULL in-band variance, which is what a weight
  actually needs).  Only ``sigma_J^2`` — where ``delta`` and ``J`` are genuinely
  confounded — needs the cutoff, and it is reported as a *curve* in ``f_c``
  rather than one number: the cutoff IS the operational boundary between "slow
  enough for the stage to track" and "irreducible".
* **k-scaled arms have a per-harmonic band**, so the common term is filtered
  differently per harmonic and the off-diagonals are NOT constant: with a
  monotone schedule ``C_ij`` measures the common power up to ``min(B_i, B_j)``.
  The common estimate for harmonic ``k`` is therefore taken over the partners
  whose band is at least as wide (``median_{j: B_j >= B_k} C_kj``), which
  reduces to the plain off-diagonal mean for a fixed band and additionally
  hands back the *cumulative spectrum* of the common term for free.

Fit-quality diagnostics (is the single-common-term model even right?):
relative Frobenius residual of the off-diagonals after removing the rank-one
term; the fraction of off-diagonal energy the best rank-one explains; and the
fitted **loading shape** ``a_k ∝ k^beta`` from ``log C_ij = 2 log sigma +
log a_i + log a_j`` — ``beta = 0`` is the delay-like model this module tests,
``beta = -1`` would instead mean a common *phase* (not delay) disturbance.

Pure numpy/scipy, CPU.  The measurement's data side — which recordings and
which time windows to run it over — is injected by the caller.  The WP18
campaign's own window builder was retired with the campaign (its drivers went
first, which left it with no caller), so a new caller must supply its own
windows, for example against ``tracking.protocols``.  The measured numbers are
recorded in ``docs/experiments/rps-refine-precision.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tracking import phase_increment_tracker as pit
from tracking.dsp import demod

SR = 16000
#: Envelope/measurement grid.  Deliberately the stage's own ``fs_env`` so the
#: measured variances are directly the ones ``pi_kalman`` would see (both
#: ``v_k`` and ``sigma_J^2`` scale with ``dt``, so a different grid would move
#: the absolute numbers even though ``k*`` is nearly invariant).
FS_ENV = 62.5
K_MAX = 30
F_MAX_HZ = 7000.0
#: Widest band demodulated once per rotor; every arm is a brickwall of this.
B_WIDE = 25.0
#: Off-comb noise-floor probe: half-band, and offset as a fraction of the
#: rotor rate (0.5 = exactly between two teeth, the widest clean gap).
B_PROBE = 8.0
OFF_FRAC = 0.5
GUARD_HZ = 1.0
#: Increment wrap protection and the per-harmonic quality gates.
WRAP_RAD = 2.5
MAX_WRAP_FRAC = 0.15
MIN_VALID_FRAC = 0.40
MIN_SNR = 1.0
EDGE_TRIM_S = 0.5
MIN_RATE = 5.0
#: High-pass cutoffs (Hz) swept for the common term.  0 = no high-pass.
CUTOFFS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
#: An arm may use a cutoff only if its NARROWEST band leaves usable width.
CUTOFF_BAND_FRAC = 0.6
#: The refine stage's k range (``k_min = 6``), for the "floor share" report.
STAGE_K_MIN = 6


@dataclass(frozen=True)
class Arm:
    """A demodulation band schedule.

    ``kind="fixed"``: ``B_k = b`` for every harmonic — what the stage does
    today.  ``kind="kscaled"``: ``B_k = min(k * b, cap)`` — the INSERT-D
    proposal, whose capture range and inter-rotor discrimination are both
    ``k``-independent.
    """

    name: str
    kind: str
    b: float
    cap: float = B_WIDE

    def band(self, k: int) -> float:
        if self.kind == "fixed":
            return min(self.b, self.cap)
        return float(min(k * self.b, self.cap))


#: Bands span the stage's own range and beyond.  The narrow end is not
#: optional: a real quadrotor's tight pair splits by 0.4-0.8 rev/s, so at a
#: 6 Hz band the twin's k-th tooth is inside the band for every k below ~16 and
#: the collision gate empties the harmonic set.  That starvation is the same
#: mechanism WP1-B measured on ``pi_kalman`` (twin-gated 80-84 %), and it is
#: itself an argument for k-scaling: at fixed ``B`` the collision radius in
#: rev/s is ``B/k`` — it SHRINKS on exactly the harmonics the weighting
#: favours — while ``B_k = k B_0`` holds it at ``B_0`` for every harmonic.
ARMS: tuple[Arm, ...] = (
    Arm("fixB1.5", "fixed", 1.5),  # the VK refine stage's final band
    Arm("fixB3", "fixed", 3.0),
    Arm("fixB6", "fixed", 6.0),  # the pi_kalman default band
    Arm("fixB12", "fixed", 12.0),
    Arm("fixB20", "fixed", 20.0),
    Arm("kscale0.1", "kscaled", 0.1),
    Arm("kscale0.25", "kscaled", 0.25),
    Arm("kscale0.5", "kscaled", 0.5),
    Arm("kscale1.0", "kscaled", 1.0),
)


# ---------------------------------------------------------------------------
# demodulation


def brickwall(z: np.ndarray, band_hz: float, fs_env: float, high: bool = False) -> np.ndarray:
    """FFT brickwall of a time series along the last axis.

    ``high=False``: keep ``|f| <= band_hz`` (lowpass, complex input allowed).
    ``high=True``: keep ``|f| > band_hz`` (highpass; DC always removed).

    Public because it is the one brickwall of the tracking package:
    ``tracking.fitness`` splits a trajectory residual into its smooth and its
    fast part with the same filter this module bands the envelopes with.
    """
    n = z.shape[-1]
    b = int(np.floor(band_hz * n / fs_env))
    if not high and b >= (n - 1) // 2:
        return z
    spec = np.fft.fft(z, axis=-1)
    if high:
        spec[..., : b + 1] = 0.0
        if b > 0:
            spec[..., -b:] = 0.0
    else:
        keep = np.zeros_like(spec)
        keep[..., : b + 1] = spec[..., : b + 1]
        if b > 0:
            keep[..., -b:] = spec[..., -b:]
        spec = keep
    out = np.fft.ifft(spec, axis=-1)
    return out if np.iscomplexobj(z) else np.real(out)


#: Back-compat alias (``tests/tracking/test_phase_noise.py`` and older callers).
_brickwall = brickwall


@dataclass
class RotorDemod:
    """One rotor's wide-band envelope bank plus everything the arms need."""

    rotor: int
    ks: list[int]
    z: np.ndarray  # (C, K, n_env) complex, band +-B_WIDE
    noise_psd: np.ndarray  # (C, K) rev-free power per Hz from the off-comb probe
    coll: np.ndarray  # (K, N_ft) bool twin-collision mask at sep = B_WIDE + guard
    r_ft: np.ndarray  # (R, N_ft) the trajectory used
    ft: np.ndarray
    t_env: np.ndarray
    t_mid: np.ndarray
    fs_env: float
    mean_rate: float
    diag: dict[str, Any] = field(default_factory=dict)


def demod_rotor(
    audio: np.ndarray,
    r_ft: np.ndarray,
    ft: np.ndarray,
    rotor: int,
    *,
    sr: int = SR,
    fs_env: float = FS_ENV,
    k_max: int = K_MAX,
    b_wide: float = B_WIDE,
    b_probe: float = B_PROBE,
    off_frac: float = OFF_FRAC,
    r_carrier: np.ndarray | None = None,
) -> RotorDemod | None:
    """Demodulate one rotor's harmonic bank + its off-comb noise floor.

    The off-comb probe sits at ``off_frac * mean_rate`` Hz from every tooth
    (half a tooth spacing = the widest clean gap) with its own, narrower band
    ``b_probe``; the floor is reported as a **PSD** so any arm's in-band noise
    power is ``2 * B_k * psd``.

    ``r_carrier`` (audio-rate rev/s) overrides the demodulation trajectory.
    This matters more than it looks: linearly interpolating a trajectory off
    the 0.032 s frame grid throws away every shaft fluctuation above ~15 Hz,
    and the residual FM that leaves is COMMON to all harmonics — i.e. it lands
    in exactly the rank-one term being measured.  Where a higher-rate label
    exists (DREGON's ~929 Hz ``motors_measured``, or a synthetic's exact
    trajectory) it must be used, and the difference against the frame-grid
    version is itself worth reporting.  ``r_ft`` is still used for the
    twin-collision geometry.
    """
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    n_t = x.shape[-1]
    t_aud = np.arange(n_t) / sr
    r_aud = (
        np.interp(t_aud, ft, r_ft[rotor])
        if r_carrier is None
        else np.asarray(r_carrier, dtype=np.float64)
    )
    mean_rate = float(np.mean(r_aud))
    if mean_rate < MIN_RATE:
        return None
    k_top = min(k_max, int(np.floor(F_MAX_HZ / max(float(r_aud.max()), 1e-3))))
    if k_top < 2:
        return None
    off_hz = off_frac * mean_rate
    if off_hz <= b_probe + 1.0 or off_hz + b_probe >= mean_rate - 1.0:
        b_probe = max(2.0, 0.35 * mean_rate * off_frac)

    stride = max(1, int(round(sr / fs_env)))
    fs_e = sr / stride
    n_env = len(range(0, n_t, stride))
    ks = list(range(1, k_top + 1))
    y32 = x.astype(np.float32)
    phi = 2.0 * np.pi * np.cumsum(r_aud) / sr
    c1 = np.exp(-1j * phi).astype(np.complex64)[None, :]

    # THE demodulation (``tracking.dsp.demod``), twice: the on-comb bank at
    # the wide band, and the off-comb probe at its own narrower band. The
    # probe is a CONSTANT frequency offset, so it is a pure bin shift of the
    # same forward transform — but it needs a different band, and a band is
    # what a transform is sliced by, so it costs its own call.
    rot0 = np.zeros(len(ks), dtype=np.int64)
    ka = np.asarray(ks, dtype=np.int64)
    z = demod(y32, c1=c1, rotor=rot0, k=ka, stride=stride, n_env=n_env, band_cyc=b_wide / sr)[
        0
    ].astype(np.complex128)
    probe = demod(
        y32,
        c1=c1,
        rotor=rot0,
        k=ka,
        stride=stride,
        n_env=n_env,
        band_cyc=b_probe / sr,
        shift_cyc=off_hz / sr,
    )[1]
    assert probe is not None
    z_off = probe.astype(np.complex128)
    n_trim = max(1, int(round(EDGE_TRIM_S * fs_e)))
    interior = slice(n_trim, max(n_trim + 1, n_env - n_trim))
    psd = np.maximum(np.mean(np.abs(z_off[..., interior]) ** 2, axis=-1), 1e-30) / (2.0 * b_probe)

    coll = pit._twin_collision_mask(r_ft, rotor, k_top, b_wide + GUARD_HZ, F_MAX_HZ, MIN_RATE)
    t_env = np.arange(n_env) / fs_e
    return RotorDemod(
        rotor=rotor,
        ks=ks,
        z=z,
        noise_psd=psd,
        coll=coll,
        r_ft=r_ft,
        ft=ft,
        t_env=t_env,
        t_mid=t_env[:-1] + 0.5 / fs_e,
        fs_env=fs_e,
        mean_rate=mean_rate,
        diag={"off_comb_hz": round(off_hz, 2), "b_probe": round(b_probe, 2), "k_top": k_top},
    )


# ---------------------------------------------------------------------------
# covariance


def _pairwise_cov(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete covariance of ``x`` ``(K, N)`` under mask ``valid``.

    Entry ``(i, j)`` uses only the frames where BOTH series are valid, so the
    per-frame twin gate (which admits different harmonics at different frames)
    does not bias the estimate.  Returns ``(C, n_pairs)``.
    """
    xz = np.where(valid, x, 0.0)
    v = valid.astype(np.float64)
    n = v @ v.T
    s = xz @ xz.T
    m_i = xz @ v.T  # sum_i over frames where both valid
    m_j = v @ xz.T
    denom = np.maximum(n, 1.0)
    c = (s - m_i * m_j / denom) / np.maximum(n - 1.0, 1.0)
    c[n < 20] = np.nan
    return c, n


def fit_rank_one(c: np.ndarray, bands: np.ndarray, se: np.ndarray | None = None) -> dict[str, Any]:
    """Moment fit of ``Sigma = sigma_c^2 1 1^T + diag(v)`` to ``c``.

    ``bands[k]`` is harmonic ``k``'s half-band.  The common estimate for
    harmonic ``k`` is taken over partners whose band is at least as wide, so a
    monotone k-scaled schedule (where ``C_ij`` measures the common power up to
    ``min(B_i, B_j)``) is handled correctly; for a fixed band every partner
    qualifies and this is the plain off-diagonal mean.

    ``se`` is the per-entry standard error of ``c`` (block bootstrap over
    time).  Without it "is the off-diagonal constant?" cannot be answered: a
    common term one tenth the size of the estimator's own scatter produces a
    raw residual of ~1 no matter how right the model is.  With it the report
    carries ``offdiag_chi2`` (scatter measured in standard errors, 1 = the
    off-diagonals are constant to within noise) and ``sigma_c2_snr``.

    Returns the mean and robust (median) common variance, the per-harmonic
    ``common_k`` curve, ``v_k``, and the fit-quality diagnostics.
    """
    k_n = c.shape[0]
    off = ~np.eye(k_n, dtype=bool)
    ok = np.isfinite(c) & off
    vals = c[ok]
    out: dict[str, Any] = {
        "n_harm": int(k_n),
        "n_offdiag": int(ok.sum()),
    }
    if vals.size < 3:
        return {**out, "failed": "too few off-diagonals"}
    out["sigma_c2_mean"] = float(np.mean(vals))
    out["sigma_c2_median"] = float(np.median(vals))
    out["sigma_c2_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    out["offdiag_neg_frac"] = float(np.mean(vals < 0))

    common = np.full(k_n, np.nan)
    for a in range(k_n):
        wide = (bands >= bands[a] - 1e-9) & off[a] & np.isfinite(c[a])
        if wide.sum() >= 1:
            common[a] = float(np.median(c[a][wide]))
    # The widest harmonic of a k-scaled arm has no wider partner; carry the
    # nearest estimate over (the common term is monotone in the band, so this
    # UNDER-states it there — flagged, not hidden).
    order = np.argsort(bands)
    filled = common.copy()
    last = out["sigma_c2_median"]
    for a in order:
        if np.isfinite(filled[a]):
            last = float(filled[a])
        else:
            filled[a] = last
    v = np.diag(c) - filled
    out["common_k"] = common.tolist()
    out["common_extrapolated"] = (~np.isfinite(common)).tolist()
    out["v_k"] = v.tolist()
    out["diag_k"] = np.diag(c).tolist()

    # --- fit quality -------------------------------------------------------
    resid = c - out["sigma_c2_mean"]
    num = float(np.sqrt(np.nansum(np.where(ok, resid, 0.0) ** 2)))
    den = float(np.sqrt(np.nansum(np.where(ok, c, 0.0) ** 2)))
    out["offdiag_resid_rel"] = num / den if den > 0 else float("nan")
    if se is not None:
        m = ok & np.isfinite(se) & (se > 0)
        if m.sum() >= 8:
            chi = (resid[m] / se[m]) ** 2
            out["offdiag_chi2"] = float(np.mean(chi))
            out["se_offdiag_median"] = float(np.median(se[m]))
            # standard error of the MEAN off-diagonal, allowing for the fact
            # that the entries share harmonics (hence are correlated): the
            # block scatter of the off-diagonal mean is what run.py feeds in
            # as ``se_mean``.
            out["sigma_c2_snr"] = float(
                out["sigma_c2_mean"] / (np.median(se[m]) / np.sqrt(m.sum()))
            )
            excess = float(np.mean(resid[m] ** 2) - np.mean(se[m] ** 2))
            out["offdiag_excess_rel"] = (
                float(np.sqrt(max(excess, 0.0)) / abs(out["sigma_c2_mean"]))
                if out["sigma_c2_mean"] != 0
                else float("nan")
            )

    # best rank-one over the off-diagonals (diagonal imputed by the rank-one
    # model itself, so the eigen-problem is not driven by v_k)
    cc = np.where(ok, c, 0.0)
    np.fill_diagonal(cc, out["sigma_c2_mean"])
    good = np.isfinite(cc).all(axis=0)
    if good.sum() >= 3:
        w, u = np.linalg.eigh(cc[np.ix_(good, good)])
        lead = float(w[-1])
        tot = float(np.sum(np.abs(w)))
        out["rank1_energy_frac"] = lead / tot if tot > 0 else float("nan")
        a_vec = np.abs(u[:, -1]) * np.sqrt(max(lead, 0.0))
        kk = np.arange(1, k_n + 1)[good]
        pos = a_vec > 0
        if pos.sum() >= 3:
            sl = np.polyfit(np.log(kk[pos]), np.log(a_vec[pos]), 1)
            out["loading_beta"] = float(sl[0])
            out["loading_a"] = (a_vec / np.median(a_vec)).tolist()
            out["loading_k"] = kk.tolist()
    # structure of the off-diagonals: do they depend on |i-j| or on min(i, j)?
    ii, jj = np.meshgrid(np.arange(1, k_n + 1), np.arange(1, k_n + 1), indexing="ij")
    for tag, pred in (("absdiff", np.abs(ii - jj)), ("min", np.minimum(ii, jj))):
        m = ok & np.isfinite(c)
        if m.sum() >= 8:
            a_, b_ = c[m], pred[m].astype(float)
            if np.std(a_) > 0 and np.std(b_) > 0:
                out[f"offdiag_corr_{tag}"] = float(np.corrcoef(a_, b_)[0, 1])
    return out


@dataclass
class ArmSeries:
    """One arm's per-harmonic rate opinions, with the gates already applied.

    The first half of :func:`arm_covariance`, extracted so a second reading of
    the SAME opinions (the cross-harmonic correlation, the residual shape) does
    not rebuild them and cannot drift from them.
    """

    arm: str
    ks: list[int]
    bands: np.ndarray  # (K,) the arm's half band per harmonic
    delta: np.ndarray  # (C, K, n_m) rate opinions, rev/s
    valid: np.ndarray  # (K, n_m) bool — twin gate and edge trim
    keep: np.ndarray  # (K,) bool — the admitted harmonics
    snr: np.ndarray  # (K,) in-band power over the off-comb floor
    p_sig: np.ndarray
    n_pow: np.ndarray
    wrap_frac: np.ndarray
    valid_frac: np.ndarray
    fs_env: float
    n_trim: int
    n_channels: int
    gates: dict[str, int]

    @property
    def report(self) -> dict[str, Any]:
        """The per-harmonic block every reading of this arm carries."""
        return {
            "arm": self.arm,
            "bands": self.bands.round(3).tolist(),
            "ks": list(self.ks),
            "keep": self.keep.tolist(),
            "n_keep": int(self.keep.sum()),
            "snr": np.round(self.snr, 4).tolist(),
            "p_sig": self.p_sig.tolist(),
            "n_pow": self.n_pow.tolist(),
            "wrap_frac": np.round(self.wrap_frac, 4).tolist(),
            "valid_frac": np.round(self.valid_frac, 4).tolist(),
            "n_channels": int(self.n_channels),
            **self.gates,
        }


def arm_increments(dm: RotorDemod, arm: Arm, *, channels: int | None = None) -> ArmSeries:
    """One arm's per-harmonic rate opinions ``delta_k(t)``, gated.

    Brickwall the wide-band bank down to this arm's band, form
    ``delta_k[t] = arg(z_k[t+1] conj(z_k[t])) fs_env / (2 pi k)``, then apply
    the four deterministic gates (twin collisions, phase-wrap fraction,
    envelope SNR, and "the band resolves the window").
    """
    n_ch = dm.z.shape[0] if channels is None else min(channels, dm.z.shape[0])
    ks = dm.ks
    k_n = len(ks)
    bands = np.array([arm.band(k) for k in ks])
    fs_e = dm.fs_env
    n_env = dm.z.shape[-1]
    n_m = n_env - 1
    n_trim = max(1, int(round(EDGE_TRIM_S * fs_e)))

    # --- per-harmonic envelopes at this arm's band -------------------------
    delta = np.empty((n_ch, k_n, n_m))
    wrapfrac = np.zeros(k_n)
    snr = np.zeros(k_n)
    p_sig = np.zeros(k_n)
    n_pow = np.zeros(k_n)
    keep = np.ones(k_n, dtype=bool)
    for a, k in enumerate(ks):
        zb = brickwall(dm.z[:n_ch, a], bands[a], fs_e)
        pw = np.abs(zb) ** 2
        npw = 2.0 * bands[a] * dm.noise_psd[:n_ch, a]
        prod = zb[..., 1:] * np.conj(zb[..., :-1])
        dpsi = np.angle(prod)
        delta[:, a] = dpsi * fs_e / (2.0 * np.pi * k)
        interior = slice(n_trim, max(n_trim + 1, n_env - n_trim))
        mp = float(np.mean(pw[..., interior]))
        mn = float(np.mean(npw))
        snr[a] = mp / max(mn, 1e-30)
        p_sig[a] = max(mp - mn, 1e-30)
        n_pow[a] = mn
        wrapfrac[a] = float(np.mean(np.abs(dpsi[..., n_trim : n_m - n_trim]) > WRAP_RAD))

    # --- deterministic frame mask: twin collisions at THIS arm's band ------
    # The separation rule is per harmonic (its own band), so one mask per
    # DISTINCT band value; a fixed arm therefore costs a single call.
    coll = np.zeros((k_n, len(dm.ft)), dtype=bool)
    for b_val in np.unique(bands):
        sel = np.where(bands == b_val)[0]
        m_all = pit._twin_collision_mask(
            dm.r_ft, dm.rotor, k_n, float(b_val) + GUARD_HZ, F_MAX_HZ, MIN_RATE
        )
        coll[sel] = m_all[sel]
    coll_mid = np.stack(
        [np.interp(dm.t_mid, dm.ft, coll[a].astype(np.float64)) for a in range(k_n)]
    )
    valid = coll_mid < 0.5
    tmask = np.zeros(n_m, dtype=bool)
    tmask[n_trim : max(n_trim, n_m - n_trim)] = True
    valid &= tmask[None, :]

    validfrac = valid.mean(axis=1)
    g_twin = validfrac >= MIN_VALID_FRAC
    g_wrap = wrapfrac <= MAX_WRAP_FRAC
    g_snr = snr >= MIN_SNR
    g_res = bands > 0.5 / (n_env / fs_e)  # band must resolve the window length
    keep &= g_twin & g_wrap & g_snr & g_res
    gates = {
        "lost_twin_gate": int(np.sum(~g_twin)),
        "lost_wrap": int(np.sum(g_twin & ~g_wrap)),
        "lost_snr": int(np.sum(g_twin & g_wrap & ~g_snr)),
        "lost_resolution": int(np.sum(~g_res)),
    }
    return ArmSeries(
        arm=arm.name,
        ks=list(ks),
        bands=bands,
        delta=delta,
        valid=valid,
        keep=keep,
        snr=snr,
        p_sig=p_sig,
        n_pow=n_pow,
        wrap_frac=wrapfrac,
        valid_frac=validfrac,
        fs_env=fs_e,
        n_trim=n_trim,
        n_channels=n_ch,
        gates=gates,
    )


def arm_covariance(
    dm: RotorDemod,
    arm: Arm,
    *,
    cutoffs: tuple[float, ...] = CUTOFFS,
    channels: int | None = None,
    n_blocks: int = 6,
    series: ArmSeries | None = None,
) -> dict[str, Any]:
    """One arm on one rotor: per-channel covariances at every usable cutoff.

    ``v_k`` is always read at ``f_c = 0`` (see the module docstring); the
    ``sigma_c^2`` curve is read at every cutoff whose band leaves usable width.
    ``series`` lets a caller that already built the opinions
    (:func:`arm_increments`) hand them over instead of rebuilding them.
    """
    ser = arm_increments(dm, arm, channels=channels) if series is None else series
    n_ch = ser.n_channels
    ks = ser.ks
    bands = ser.bands
    fs_e = ser.fs_env
    n_env = dm.z.shape[-1]
    n_m = n_env - 1
    delta, valid, keep = ser.delta, ser.valid, ser.keep
    snr, p_sig, n_pow = ser.snr, ser.p_sig, ser.n_pow

    out: dict[str, Any] = ser.report
    if keep.sum() < 4:
        out["failed"] = f"only {int(keep.sum())} harmonics survive the gates"
        return out

    idx = np.where(keep)[0]
    kk = np.array(ks)[idx]
    bands_k = bands[idx]

    # --- covariance per cutoff --------------------------------------------
    b_min = float(bands_k.min())
    per_cut: dict[str, Any] = {}
    common_chan: dict[str, list[float]] = {}
    v_at_zero: np.ndarray | None = None
    for fc in cutoffs:
        if fc > 0 and fc > CUTOFF_BAND_FRAC * b_min:
            continue
        xs: list[np.ndarray] = []
        for c in range(n_ch):
            x = delta[c, idx]
            if fc > 0:
                # gaps filled by interpolation so the FFT filter sees a
                # continuous series; the gap frames stay excluded below.
                xf = np.empty_like(x)
                tt = np.arange(n_m, dtype=float)
                for a in range(len(idx)):
                    v = valid[idx[a]]
                    xf[a] = np.interp(tt, tt[v], x[a][v]) if v.any() else 0.0
                x = brickwall(xf, fc, fs_e, high=True)
            xs.append(x)
        # Block bootstrap in TIME: the per-entry standard error is what makes
        # "are the off-diagonals constant?" answerable.  Channels are averaged
        # inside each block because the common term is shared across mics, so
        # channel scatter would understate the error on exactly that term.
        blocks = np.array_split(np.arange(n_m), n_blocks)
        cbs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
            for bl in blocks:
                if len(bl) < 64:
                    continue
                cb = np.nanmean(
                    np.stack([_pairwise_cov(x[:, bl], valid[idx][:, bl])[0] for x in xs]), axis=0
                )
                cbs.append(cb)
            cs_full = [_pairwise_cov(x, valid[idx])[0] for x in xs]
            cmat = np.nanmean(np.stack(cs_full), axis=0)
        se = None
        if len(cbs) >= 4:
            stack = np.stack(cbs)
            se = np.nanstd(stack, axis=0) / np.sqrt(np.sum(np.isfinite(stack), axis=0))
        fit = fit_rank_one(cmat, bands_k, se)
        # A common term is only recovered UNATTENUATED where the harmonic's own
        # envelope SNR is good: at SNR ~ 1 the wrapped-phase increment
        # decorrelates from the truth, so the fitted loading falls with SNR
        # rather than being uniform.  Reporting the correlation keeps that
        # confound visible instead of letting it masquerade as physics.
        a_load = fit.get("loading_a")
        if a_load is not None and "loading_k" in fit:
            ai = np.asarray(a_load, dtype=float)
            pos = np.asarray(fit["loading_k"], dtype=int) - 1  # positions, not k
            si = snr[idx][pos]
            if len(si) == len(ai) and np.all(ai > 0) and np.all(si > 0):
                fit["loading_snr_corr"] = float(np.corrcoef(np.log(ai), np.log(si))[0, 1])
            fit["loading_k"] = kk[pos].tolist()  # report TRUE harmonic indices
            if np.all(ai > 0):  # re-fit beta against the TRUE k, not positions
                fit["loading_beta"] = _slope(np.log(kk[pos].astype(float)), np.log(ai))["slope"]
        if se is not None and len(cbs) >= 4:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                offm = [float(np.nanmean(cb[~np.eye(len(idx), dtype=bool)])) for cb in cbs]
            offm = [o for o in offm if np.isfinite(o)]
            if len(offm) >= 4:
                s_e = float(np.std(offm, ddof=1) / np.sqrt(len(offm)))
                fit["sigma_c2_se"] = s_e
                fit["n_blocks"] = len(offm)
                # Everything downstream that reads a SHAPE off the common term
                # (the loading exponent, the channel coherence) is meaningless
                # when the term itself is not resolved, so carry the
                # significance rather than letting a noise eigenvector pose as
                # physics.
                fit["sigma_c2_signif"] = (
                    float(fit["sigma_c2_mean"] / s_e) if s_e > 0 else float("nan")
                )
        per_cut[f"{fc:g}"] = fit
        common_chan[f"{fc:g}"] = [
            float(np.nanmedian(cc[~np.eye(len(idx), dtype=bool)])) for cc in cs_full
        ]
        if fc == 0.0:
            v_at_zero = np.asarray(fit.get("v_k", []), dtype=float)
    out["cov"] = per_cut
    out["common_per_channel"] = common_chan

    # --- the weight curve, its exponent, and the saturation harmonic -------
    if v_at_zero is not None and v_at_zero.size == len(idx):
        pos = v_at_zero > 0
        out["v_k_used"] = v_at_zero.tolist()
        out["k_used"] = kk.tolist()
        out["v_k_pos_frac"] = float(np.mean(pos))
        if pos.sum() >= 4:
            lk = np.log(kk[pos].astype(float))
            w = np.log(1.0 / v_at_zero[pos])
            out["alpha_raw"] = _slope(lk, w)
            out["alpha_signal"] = _slope(lk, w - np.log(p_sig[idx][pos]))
            out["alpha_snr"] = _slope(lk, w - np.log(p_sig[idx][pos] / n_pow[idx][pos]))
        out.update(_saturation(kk, v_at_zero, per_cut))
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS slope + R^2 of ``y`` on ``x``."""
    if len(x) < 3 or np.std(x) == 0:
        return {"slope": float("nan"), "r2": float("nan"), "n": len(x)}
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else float("nan")
    return {"slope": float(b), "r2": float(r2), "n": int(len(x))}


def _saturation(kk: np.ndarray, v: np.ndarray, per_cut: dict[str, Any]) -> dict[str, Any]:
    """``k*`` (where ``1/W`` drops below the floor) + the floor's share."""
    out: dict[str, Any] = {}
    order = np.argsort(kk)
    kk_s, v_s = kk[order], v[order]
    ok = v_s > 0
    if ok.sum() < 3:
        return out
    inv = np.cumsum(1.0 / v_s[ok])
    k_ok = kk_s[ok]
    for tag, fit in per_cut.items():
        s2 = fit.get("sigma_c2_mean")
        # An UNRESOLVED common term has no saturation harmonic: reporting one
        # off a value consistent with zero would invent a floor.
        if (fit.get("sigma_c2_signif") or 0.0) < 3.0:
            s2 = None
        if s2 is None or not np.isfinite(s2) or s2 <= 0:
            out[f"k_star_fc{tag}"] = None
            continue
        below = np.where(1.0 / inv < s2)[0]
        out[f"k_star_fc{tag}"] = int(k_ok[below[0]]) if below.size else None
        stage = k_ok >= STAGE_K_MIN
        if stage.any():
            w_stage = float(np.sum(1.0 / v_s[ok][stage]))
            out[f"floor_share_fc{tag}"] = float(s2 / (1.0 / w_stage + s2))
            out[f"inv_W_stage_fc{tag}"] = float(1.0 / w_stage)
    return out


# ---------------------------------------------------------------------------
# channel coherence of the common term


def channel_coherence(dm: RotorDemod, arm: Arm, fc: float | None = None) -> dict[str, Any]:
    """Is the common term SOURCE-side (shaft) or CHANNEL-side (propagation)?

    Shaft timing jitter is one number for the whole aircraft, so its estimate
    must agree across microphones; a per-microphone propagation fluctuation
    must not.  ``J_c(t)`` is estimated per channel as the across-harmonic
    median of the high-passed opinions; the report is the mean pairwise
    correlation.
    """
    n_ch = dm.z.shape[0]
    if n_ch < 2:
        return {}
    ks = dm.ks
    fs_e = dm.fs_env
    n_env = dm.z.shape[-1]
    n_m = n_env - 1
    n_trim = max(1, int(round(EDGE_TRIM_S * fs_e)))
    bands = np.array([arm.band(k) for k in ks])
    # Default to the widest cutoff this arm's narrowest band can carry: a
    # slower cutoff would leave the (common) trajectory error in, which would
    # read as perfect channel coherence for the wrong reason.
    fc_max = CUTOFF_BAND_FRAC * float(bands.min())
    fc_use = min(fc, fc_max) if fc is not None else fc_max
    if fc_use <= 0.2:
        return {}
    series = []
    for c in range(n_ch):
        rows = []
        for a, k in enumerate(ks):
            zb = brickwall(dm.z[c, a], bands[a], fs_e)
            prod = zb[1:] * np.conj(zb[:-1])
            rows.append(np.angle(prod) * fs_e / (2.0 * np.pi * k))
        x = brickwall(np.stack(rows), fc_use, fs_e, high=True)
        series.append(np.median(x, axis=0))
    s = np.stack(series)[:, n_trim : n_m - n_trim]
    cm = np.corrcoef(s)
    off = cm[~np.eye(n_ch, dtype=bool)]
    return {
        "fc": round(fc_use, 3),
        "chan_coherence_mean": float(np.mean(off)),
        "chan_coherence_min": float(np.min(off)),
        "chan_coherence_max": float(np.max(off)),
        "n_channels": int(n_ch),
    }


# ---------------------------------------------------------------------------
# line width, line shape, cross-harmonic correlation, residual shape
#
# The four readings the stochastic comb model needs (see the module docstring
# of ``data_processing.stochastic_rotor_noise``): a Lorentzian line per
# harmonic whose half width grows as ``gamma_k = gamma_0 + s k``.  Everything
# below is pure array code on the SAME envelopes WP18 measures its covariance
# on, so the two readings cannot disagree about what a harmonic is.

#: The autocorrelation level whose lag IS the coherence time.  A Lorentzian of
#: half width ``gamma`` has envelope autocorrelation ``exp(-2 pi gamma |lag|)``,
#: so the lag at ``exp(-1)`` is ``1 / (2 pi gamma)`` exactly.
ACF_LEVEL = float(np.exp(-1.0))
#: Bands the per-harmonic readings are pooled into for the report.
K_BANDS: tuple[tuple[str, int, int], ...] = (("k1_5", 1, 5), ("k6_15", 6, 15), ("k16_30", 16, 30))
#: Cap on the samples a Cauchy/Gaussian maximum-likelihood fit sees.  The two
#: fits are optimizers, and a million samples buys no precision the verdict can
#: use.
TAIL_FIT_MAX_N = 50000


def envelope_acf(
    z: np.ndarray,
    fs_env: float,
    *,
    max_lag_s: float = 2.0,
    noise_power: float | np.ndarray = 0.0,
    noise_band_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lags (s) and the normalized magnitude autocorrelation of an envelope.

    ``z`` is ``(..., n)`` complex — the leading axes are microphones. The
    unbiased estimator ``R(l) = mean_t z[t+l] conj(z[t])`` is taken along the
    last axis and averaged over the leading ones. A per-channel constant
    propagation phase cancels in ``z(t+l) conj(z(t))``, so the average is
    taken COMPLEX (which averages the estimator noise down) and the magnitude
    is read at the end.

    ``z`` is NOT mean removed. A tone with no phase noise has a flat ``|R|``
    and an infinite coherence time, which is the answer this measurement must
    give; removing the mean would make every line read as white.

    The demodulation band is a brickwall, so a flat floor of power
    ``noise_power`` inside a half band ``noise_band_hz`` contributes
    ``noise_power * sinc(2 B l)`` — subtracted at EVERY lag rather than only at
    lag 0, which is what an oversampled envelope grid needs (on a critically
    sampled one the sinc vanishes at every non-zero lag by itself).
    """
    x = np.atleast_2d(np.asarray(z))
    n = int(x.shape[-1])
    n_lag = int(min(int(round(max_lag_s * fs_env)), n - 2))
    if n_lag < 2:
        raise ValueError(f"window of {n} samples at {fs_env} Hz is too short for an ACF")
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.fft(x, n=n_fft, axis=-1)
    acf = np.fft.ifft(np.abs(spec) ** 2, axis=-1)[..., : n_lag + 1]
    lags = np.arange(n_lag + 1) / float(fs_env)
    acf = acf / (n - np.arange(n_lag + 1))
    npw = np.asarray(noise_power, dtype=np.float64)
    if noise_band_hz is not None and np.any(npw > 0):
        acf = acf - npw.reshape(npw.shape + (1,) * (acf.ndim - npw.ndim)) * np.sinc(
            2.0 * float(noise_band_hz) * lags
        )
    r = acf.reshape(-1, n_lag + 1).mean(axis=0)
    p0 = float(np.real(r[0]))
    if not np.isfinite(p0) or p0 <= 0:
        return lags, np.full(n_lag + 1, np.nan)
    return lags, np.abs(r) / p0


def coherence_time(lags: np.ndarray, rho: np.ndarray, *, level: float = ACF_LEVEL) -> float:
    """First lag where ``rho`` falls below ``level``, linearly interpolated.

    ``nan`` when the curve never falls that far inside the lags given — the
    coherence time is then CENSORED at ``lags[-1]``, not measured, and the
    caller must say so rather than report the last lag as an answer.
    """
    r = np.asarray(rho, dtype=np.float64)
    below = np.where(np.isfinite(r[1:]) & (r[1:] < level))[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0]) + 1
    r0, r1 = float(r[i - 1]), float(r[i])
    if not np.isfinite(r0) or r0 <= r1:
        return float(lags[i])
    frac = (r0 - level) / (r0 - r1)
    return float(lags[i - 1] + frac * (lags[i] - lags[i - 1]))


def linewidth(
    z: np.ndarray,
    fs_env: float,
    *,
    max_lag_s: float = 2.0,
    noise_power: float | np.ndarray = 0.0,
    noise_band_hz: float | None = None,
) -> dict[str, Any]:
    """Coherence time and Lorentzian half width of one harmonic's envelope."""
    lags, rho = envelope_acf(
        z,
        fs_env,
        max_lag_s=max_lag_s,
        noise_power=noise_power,
        noise_band_hz=noise_band_hz,
    )
    tau = coherence_time(lags, rho)
    censored = not np.isfinite(tau)
    gamma = float("nan") if censored else 1.0 / (2.0 * np.pi * tau)
    return {
        "tau_s": float(tau),
        "gamma_hz": gamma,
        "censored": bool(censored),
        "max_lag_s": float(lags[-1]),
        # The floor the width would take if the curve had crossed at the very
        # last lag — the honest bound a censored harmonic reports.
        "gamma_bound_hz": float(1.0 / (2.0 * np.pi * lags[-1])),
        "acf_at_max_lag": float(rho[-1]),
    }


def fit_linewidth_law(
    k: np.ndarray, gamma: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, Any]:
    """Least squares ``gamma_k = gamma_0 + s k`` over the admitted harmonics."""
    kk = np.asarray(k, dtype=np.float64).reshape(-1)
    gg = np.asarray(gamma, dtype=np.float64).reshape(-1)
    ok = np.isfinite(kk) & np.isfinite(gg)
    kk, gg = kk[ok], gg[ok]
    w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)[ok]
    out: dict[str, Any] = {"n": int(kk.size)}
    if kk.size < 3 or np.std(kk) == 0:
        return {**out, "gamma0_hz": float("nan"), "slope_hz_per_k": float("nan")}
    design = np.stack([np.ones_like(kk), kk], axis=1)
    if w is None:
        coef, *_ = np.linalg.lstsq(design, gg, rcond=None)
    else:
        sw = np.sqrt(np.maximum(w, 0.0))
        coef, *_ = np.linalg.lstsq(design * sw[:, None], gg * sw, rcond=None)
    pred = design @ coef
    resid = gg - pred
    ss = float(np.sum((gg - np.mean(gg)) ** 2))
    out.update(
        {
            "gamma0_hz": float(coef[0]),
            "slope_hz_per_k": float(coef[1]),
            "resid_rms_hz": float(np.sqrt(np.mean(resid**2))),
            "resid_rel": float(np.sqrt(np.mean(resid**2)) / max(float(np.mean(gg)), 1e-12)),
            "r2": float(1.0 - np.sum(resid**2) / ss) if ss > 0 else float("nan"),
        }
    )
    return out


def welch_envelope(
    z: np.ndarray, fs_env: float, *, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided averaged periodogram of a complex envelope, DC centered.

    A complex envelope is NOT conjugate symmetric — the line's two flanks carry
    different information — so the spectrum is two sided and stays that way.
    Microphones are averaged in POWER, which is right here and wrong for the
    autocorrelation: a spectrum has no phase to preserve.
    """
    from scipy.signal import welch

    x = np.atleast_2d(np.asarray(z))
    n = int(x.shape[-1])
    nps = int(nperseg) if nperseg else max(64, n // 4)
    nps = min(nps, n)
    f, p = welch(
        x, fs=fs_env, nperseg=nps, return_onesided=False, detrend=False, scaling="density", axis=-1
    )
    order = np.argsort(f)
    f = np.asarray(f)[order]
    p = np.asarray(p)[..., order]
    return f, p.reshape(-1, f.size).mean(axis=0)


def _shape_model(kind: str, u: np.ndarray) -> np.ndarray:
    """Unit-peak line profile at ``u = (f - f0) / hwhm``. Both have HWHM 1."""
    if kind == "lorentz":
        return 1.0 / (1.0 + u**2)
    return np.exp(-np.log(2.0) * u**2)


def _fit_one_shape(
    kind: str, f: np.ndarray, p: np.ndarray, hwhm0: float, f0: float
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    logp = np.log10(np.maximum(p, 1e-300))
    floor0 = max(float(np.median(p[np.abs(f - f0) > 0.6 * float(np.abs(f).max())])), 1e-300)
    amp0 = max(float(np.max(p)) - floor0, floor0)

    def resid(theta: np.ndarray) -> np.ndarray:
        amp, cen, hw, flr = 10.0 ** theta[0], theta[1], 10.0 ** theta[2], 10.0 ** theta[3]
        model = amp * _shape_model(kind, (f - cen) / max(hw, 1e-9)) + flr
        return logp - np.log10(np.maximum(model, 1e-300))

    x0 = np.array([np.log10(amp0), f0, np.log10(max(hwhm0, 1e-6)), np.log10(floor0)])
    span = float(np.abs(f).max())
    bounds = (
        np.array([x0[0] - 6.0, -span, np.log10(max(hwhm0, 1e-6)) - 2.0, x0[3] - 8.0]),
        np.array([x0[0] + 6.0, span, np.log10(max(hwhm0, 1e-6)) + 2.0, x0[3] + 8.0]),
    )
    try:
        sol = least_squares(resid, x0, bounds=bounds, max_nfev=400)
    except Exception:  # noqa: BLE001 — a failed fit is a result, not a crash
        return {"resid_rms_log10": float("nan"), "hwhm_hz": float("nan"), "ok": False}
    r = resid(sol.x)
    return {
        "resid_rms_log10": float(np.sqrt(np.mean(r**2))),
        "hwhm_hz": float(10.0 ** sol.x[2]),
        "center_hz": float(sol.x[1]),
        "floor_frac": float(10.0 ** (sol.x[3] - sol.x[0])),
        "ok": bool(sol.success),
        "n_points": int(f.size),
    }


def fit_line_shape(
    f: np.ndarray, p: np.ndarray, *, hwhm0: float, span_factor: float = 10.0
) -> dict[str, Any]:
    """Lorentzian against Gaussian on one line, both fitted in the log domain.

    Both profiles are written with the SAME parameter — a half width at half
    maximum — so the discriminator is the TAIL and not the width: a Lorentzian
    falls as ``1 / u^2`` where a Gaussian falls as ``exp(-u^2)``. Each model
    fits its own amplitude, center, half width and additive floor by least
    squares on ``log10`` power, and the verdict is the residual ratio.

    The fit band is ``span_factor`` half widths around DC, clipped to the
    spectrum. Too narrow and the two shapes are the same parabola; too wide and
    the fit is scoring the noise floor rather than the line.
    """
    fa = np.asarray(f, dtype=np.float64)
    pa = np.asarray(p, dtype=np.float64)
    span = min(float(np.abs(fa).max()), max(span_factor * float(hwhm0), 3.0 * float(hwhm0)))
    sel = np.abs(fa) <= span
    if int(sel.sum()) < 12:
        return {"verdict": "", "n_points": int(sel.sum())}
    fs_, ps_ = fa[sel], pa[sel]
    f0 = float(fs_[int(np.argmax(ps_))])
    lor = _fit_one_shape("lorentz", fs_, ps_, hwhm0, f0)
    gau = _fit_one_shape("gauss", fs_, ps_, hwhm0, f0)
    rl, rg = lor["resid_rms_log10"], gau["resid_rms_log10"]
    verdict = ""
    if np.isfinite(rl) and np.isfinite(rg):
        verdict = "lorentz" if rl < rg else "gauss"
    return {
        "lorentz": lor,
        "gauss": gau,
        "verdict": verdict,
        # Positive => the Lorentzian fits better. A ratio of residual rms in
        # the log domain, so it is dimensionless and poolable across harmonics.
        "log_resid_ratio": float(np.log10(rg / rl)) if np.isfinite(rl) and rl > 0 else float("nan"),
        "span_hz": float(span),
        "n_points": int(sel.sum()),
    }


def cross_harmonic_correlation(series: ArmSeries) -> dict[str, Any]:
    """Correlation across the admitted harmonics' rate opinions.

    The covariance is the pairwise-complete one :func:`arm_covariance` fits its
    rank-one model to, converted to a CORRELATION so the per-harmonic numbers
    are comparable across harmonics whose own variances differ by orders of
    magnitude. ``rho_k`` is harmonic ``k``'s mean correlation with the other
    admitted harmonics: 1 means one shared disturbance, 0 means independent
    per-harmonic phase noise.
    """
    idx = np.where(series.keep)[0]
    out: dict[str, Any] = {"arm": series.arm, "n_keep": int(idx.size)}
    if idx.size < 3:
        return {**out, "failed": "fewer than 3 admitted harmonics"}
    kk = np.asarray(series.ks)[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cs = [
            _pairwise_cov(series.delta[c, idx], series.valid[idx])[0]
            for c in range(series.n_channels)
        ]
        cmat = np.nanmean(np.stack(cs), axis=0)
        d = np.sqrt(np.clip(np.diag(cmat), 1e-30, None))
        corr = cmat / np.outer(d, d)
        off = ~np.eye(idx.size, dtype=bool)
        rho_k = np.array([np.nanmean(corr[a][off[a]]) for a in range(idx.size)])
        rho_all = float(np.nanmean(corr[off]))
    return {
        **out,
        "k": kk.tolist(),
        "rho_k": np.round(rho_k, 5).tolist(),
        "rho_mean": rho_all,
        "var_k": np.round(np.diag(cmat), 10).tolist(),
    }


def shared_rate_opinion(
    series: ArmSeries, v_k: np.ndarray, k_used: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The inverse-variance weighted mean opinion ``c(t)``, per channel.

    ``v_k`` / ``k_used`` are :func:`arm_covariance`'s own per-harmonic variances
    at ``f_c = 0`` and the harmonics they belong to, so the weights are the ones
    the fusion stage would really use. Returns ``(c, mask)`` — ``c`` is
    ``(C, n_m)`` and ``mask`` marks the frames where any harmonic was admitted.
    """
    ks = np.asarray(series.ks)
    pos = np.array([int(np.where(ks == k)[0][0]) for k in np.asarray(k_used).reshape(-1)])
    v = np.asarray(v_k, dtype=np.float64).reshape(-1)
    good = np.isfinite(v) & (v > 0)
    pos, v = pos[good], v[good]
    n_ch, _, n_m = series.delta.shape
    if pos.size < 2:
        return np.full((n_ch, n_m), np.nan), np.zeros(n_m, dtype=bool)
    w = (1.0 / v)[:, None] * series.valid[pos].astype(np.float64)  # (K', n_m)
    wsum = w.sum(axis=0)
    mask = wsum > 0
    num = np.einsum("kt,ckt->ct", w, series.delta[:, pos, :])
    c = np.where(mask[None, :], num / np.where(mask, wsum, 1.0)[None, :], np.nan)
    return c, mask


def residual_tail_stats(e: np.ndarray, *, max_n: int = TAIL_FIT_MAX_N) -> dict[str, Any]:
    """Excess kurtosis and the Cauchy-against-Gaussian per-sample LLR.

    ``e`` is pooled over frames and over the harmonics of one band. It is
    standardized by the median absolute deviation (not the standard deviation,
    which a heavy tail owns), then the two maximum-likelihood fits are compared.
    A positive LLR favours the Cauchy — that is, a jitter whose per-harmonic
    part is heavy tailed rather than Gaussian.

    Note that a wrapped phase increment is BOUNDED by ``pi / (2 pi k dt)``
    rev/s, so the tails of a high-``k`` residual are clipped by construction and
    this LLR UNDERSTATES the Cauchy case there.
    """
    from scipy.stats import cauchy, norm

    x = np.asarray(e, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    out: dict[str, Any] = {"n": int(x.size)}
    if x.size < 200:
        return {**out, "failed": "fewer than 200 pooled samples"}
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = mad / 0.6744897501960817 if mad > 0 else float(np.std(x))
    if not np.isfinite(scale) or scale <= 0:
        return {**out, "failed": "degenerate scale"}
    z = (x - med) / scale
    m2 = float(np.mean(z**2))
    out["mad_scale"] = scale
    out["excess_kurtosis"] = float(np.mean(z**4) / m2**2 - 3.0) if m2 > 0 else float("nan")
    # A deterministic stride, so a re-run of the same unit gives the same fit.
    step = max(1, int(np.ceil(x.size / max_n)))
    s = z[::step]
    try:
        c_loc, c_scale = cauchy.fit(s)
        n_loc, n_scale = norm.fit(s)
        llr = float(
            np.mean(cauchy.logpdf(s, c_loc, c_scale)) - np.mean(norm.logpdf(s, n_loc, n_scale))
        )
    except Exception:  # noqa: BLE001 — a failed fit is a result, not a crash
        return {**out, "failed": "maximum-likelihood fit did not converge"}
    out.update(
        {
            "llr_per_sample": llr,
            "verdict": "cauchy" if llr > 0 else "gauss",
            "cauchy_scale": float(c_scale),
            "gauss_scale": float(n_scale),
            "n_fit": int(s.size),
        }
    )
    return out


def band_of(k: int) -> str:
    """The reporting band of harmonic ``k`` (``""`` when it is outside them)."""
    for name, lo, hi in K_BANDS:
        if lo <= k <= hi:
            return name
    return ""
