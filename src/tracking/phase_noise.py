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
which time windows to run it over — is injected by
``scripts/phase_noise_cov/windows.py`` (the tracking-purity split of
``tracking.protocols``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tracking import phase_increment_tracker as pit

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


def _demod(
    y32: np.ndarray,
    c1: np.ndarray,
    ks: list[int],
    stride: int,
    n_env: int,
    band_cyc: float,
    ramp: np.ndarray | None = None,
) -> np.ndarray:
    """``(C, K, n_env)`` complex envelopes of ``y32`` demodulated by ``k*phi``.

    ``c1 = exp(-1j phi)`` (complex64); ``ramp`` an optional extra phasor applied
    to every harmonic (the off-comb noise probe).  Harmonic carriers come from
    the power recursion of ``pit._demod_bank``.
    """
    n_ch, n_t = y32.shape
    z = np.empty((n_ch, len(ks), n_env), dtype=np.complex128)
    chunk = max(1, int(24e6 / (max(1, n_ch) * max(1, n_t) * 8)))
    buf = np.empty((n_ch, min(chunk, len(ks)), n_t), dtype=np.complex64)
    idxs: list[int] = []

    def flush() -> None:
        m = len(idxs)
        b = buf[:, :m]
        if ramp is not None:
            b = b * ramp
        z[:, idxs] = pit.zoom_lp_decimate(b, stride, n_env, band_cyc)
        idxs.clear()

    cur = np.ones_like(c1)
    cur_k = 0
    for a, k in enumerate(ks):
        step = k - cur_k
        if step > 2:
            cur = cur * c1**step
        else:
            for _ in range(step):
                cur = cur * c1
        cur_k = k
        np.multiply(y32, cur, out=buf[:, len(idxs)])
        idxs.append(a)
        if len(idxs) == buf.shape[1]:
            flush()
    if idxs:
        flush()
    return z


def _brickwall(z: np.ndarray, band_hz: float, fs_env: float, high: bool = False) -> np.ndarray:
    """FFT brickwall of a time series along the last axis.

    ``high=False``: keep ``|f| <= band_hz`` (lowpass, complex input allowed).
    ``high=True``: keep ``|f| > band_hz`` (highpass; DC always removed).
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
    c1 = np.exp(-1j * phi).astype(np.complex64)
    ramp = np.exp(-2j * np.pi * off_hz * t_aud).astype(np.complex64)

    z = _demod(y32, c1, ks, stride, n_env, b_wide / sr)
    z_off = _demod(y32, c1, ks, stride, n_env, b_probe / sr, ramp=ramp)
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


def arm_covariance(
    dm: RotorDemod,
    arm: Arm,
    *,
    cutoffs: tuple[float, ...] = CUTOFFS,
    channels: int | None = None,
    n_blocks: int = 6,
) -> dict[str, Any]:
    """One arm on one rotor: per-channel covariances at every usable cutoff.

    ``v_k`` is always read at ``f_c = 0`` (see the module docstring); the
    ``sigma_c^2`` curve is read at every cutoff whose band leaves usable width.
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
        zb = _brickwall(dm.z[:n_ch, a], bands[a], fs_e)
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

    out: dict[str, Any] = {
        "arm": arm.name,
        "bands": bands.round(3).tolist(),
        "ks": ks,
        "keep": keep.tolist(),
        "n_keep": int(keep.sum()),
        "snr": np.round(snr, 4).tolist(),
        "p_sig": p_sig.tolist(),
        "n_pow": n_pow.tolist(),
        "wrap_frac": np.round(wrapfrac, 4).tolist(),
        "valid_frac": np.round(validfrac, 4).tolist(),
        "n_channels": int(n_ch),
        **gates,
    }
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
                x = _brickwall(xf, fc, fs_e, high=True)
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
            zb = _brickwall(dm.z[c, a], bands[a], fs_e)
            prod = zb[1:] * np.conj(zb[:-1])
            rows.append(np.angle(prod) * fs_e / (2.0 * np.pi * k))
        x = _brickwall(np.stack(rows), fc_use, fs_e, high=True)
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
