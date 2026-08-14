"""Attribution of the broadband residual by INCOHERENT band POWER, not by coherence.

Why a second instrument
-----------------------
The first pass (``csd`` / ``design`` / ``fit``) models the array covariance as
``sum_r P_r g_r g_r^H + diag(D)`` — four COHERENT point sources at the rotor
hubs. It was refuted by its own geometry null controls
(``docs/experiments/residual-attribution.md``). The measurement below says why:
the residual carries almost no coherent field where that model needs one. On
DREGON the mean magnitude-squared coherence between mic pairs is **0.014** at
50-200 Hz — the band where a propagating field over a 0.17 m aperture must be
near unity — and peaks at only 0.37 around 500-1000 Hz. The DREGON mics sit
inside the rotor wash, so the low band is per-diaphragm flow noise, which is
uncorrelated between mics BY CONSTRUCTION and cannot be steered.

So this module drops the phase and keeps the two levers that survive:

1. **A MEASURED per-rotor transfer**, not a free-field model. DREGON ships
   single-motor recordings (``Motor{1-4}_{50..90}.wav``, 8 channels, one rotor
   running). Each gives the per-microphone broadband pattern of ONE rotor
   directly, airframe shadowing and wake included. :func:`bench_basis`.
2. **Time modulation.** Rotor speeds move quasi-independently, and broadband
   rotor noise follows its own rotor's speed. Regressing per-mic band power on
   the four rotor-speed tracks identifies per-rotor per-mic gains with NO
   geometry and NO coherence at all. :func:`fit_free_modulation`.

Lever 2 is the one that can CONTRADICT lever 1, which is what makes the pair
an experiment rather than an assumption: the in-flight regression pattern is
compared against the bench pattern under a rotor permutation null.

The model
---------
Per band ``b``, microphone ``c``, time frame ``t``::

    p_{c,b}(t) = sum_r B_{r,c,b} s_r(t) + phi_{c,b} + noise

``p`` is a robust band power (median over bins, so leftover comb teeth do not
set it), ``s_r(t)`` is the modulation regressor of rotor ``r`` (its own speed
raised to an aeroacoustic exponent), ``B >= 0`` and ``phi >= 0``. Everything a
consumer wants is a ratio inside one microphone, so the unknown per-microphone
sensitivity divides out and never has to be estimated.

Nothing here imports the rest of the repo: arrays in, arrays out. Geometry,
audio and rotor-speed tracks are the caller's to supply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.optimize import nnls

__all__ = [
    "BANDS",
    "BandPower",
    "band_power",
    "basis_identifiability",
    "bench_basis",
    "geom_basis",
    "additivity",
    "fit_free_modulation",
    "fit_basis_modulation",
    "pattern_agreement",
    "fit_linear",
    "mode_design",
    "mode_information",
    "block_bootstrap",
]

#: Default analysis bands (Hz). The first stops at 100 Hz because below it the
#: residual is dominated by the k1-2 comb leakage of the VK solve.
BANDS: list[tuple[float, float]] = [
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, 14000),
]


# ─── Band power ──────────────────────────────────────────────────────────────


@dataclass
class BandPower:
    """Robust per-(mic, band, frame) power of a multichannel signal."""

    power: np.ndarray  # (C, B, T)
    times: np.ndarray  # (T,) seconds from the signal start
    bands: list[tuple[float, float]]
    n_bins: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # (B,)


def band_power(
    x: np.ndarray,
    sr: float,
    bands: list[tuple[float, float]] | None = None,
    *,
    frame_s: float = 0.25,
    n_fft: int = 1024,
) -> BandPower:
    """``(C, T)`` audio -> band power per microphone, band and frame.

    The reduction over the bins of a band is the MEDIAN, not the mean: the VK
    residual keeps narrow leftovers (foreign tones, high-``k`` comb teeth that
    the 1 Hz prior could not follow), and a mean would let one tooth set the
    level of a whole band. The median is an unbiased scale estimator for an
    exponential periodogram up to the fixed factor ``ln 2``, which cancels in
    every ratio taken here.
    """
    bands = list(BANDS if bands is None else bands)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    hop = max(1, int(round(frame_s * sr)))
    win = np.hanning(n_fft)
    n_frames = 1 + max(0, (x.shape[-1] - n_fft) // hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands]
    n_bins = np.array([int(m.sum()) for m in masks], dtype=int)
    if np.any(n_bins == 0):
        raise ValueError(f"empty band at n_fft={n_fft}, sr={sr}: {bands}")

    out = np.empty((x.shape[0], len(bands), n_frames))
    # Chunk the frames so a long recording never holds a full spectrogram.
    step = max(1, int(4e6 // (n_fft * x.shape[0])))
    for a in range(0, n_frames, step):
        b = min(n_frames, a + step)
        idx = np.arange(a, b) * hop
        seg = x[:, idx[:, None] + np.arange(n_fft)[None, :]]  # (C, n, n_fft)
        spec = np.abs(np.fft.rfft(seg * win, axis=-1)) ** 2
        for j, m in enumerate(masks):
            out[:, j, a:b] = np.median(spec[:, :, m], axis=-1)
    times = (np.arange(n_frames) * hop + n_fft / 2) / sr
    return BandPower(power=out, times=times, bands=bands, n_bins=n_bins)


# ─── Bases: what one rotor does to the array ─────────────────────────────────


def geom_basis(mic_pos: np.ndarray, rotor_pos: np.ndarray, *, exponent: float = 2.0) -> np.ndarray:
    """Free-field power basis ``(C, R)``: ``1 / d^exponent``.

    The model the first pass used, reduced to its magnitude. Kept as the
    control the measured basis is scored against.
    """
    d = np.linalg.norm(np.asarray(mic_pos)[:, None, :] - np.asarray(rotor_pos)[None, :, :], axis=-1)
    return 1.0 / d**exponent


def bench_basis(
    clips: dict[tuple[int, int], np.ndarray],
    sr: float,
    bands: list[tuple[float, float]] | None = None,
    *,
    n_fft: int = 4096,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """MEASURED per-rotor per-mic transfer from single-motor bench recordings.

    Args:
        clips: ``{(rotor index, throttle percent): (C, N) audio}`` — one rotor
            running, every other rotor stopped.
        sr: sample rate of the clips.
        bands: analysis bands.
        n_fft: Welch segment.

    Returns:
        ``(basis (C, R, B), report)``. Each column is normalized to sum to one
        over microphones, so it is a PATTERN and carries no absolute level; the
        report keeps the level (``level_db``), the throttle stability of the
        pattern (``speed_cos``) and the per-band mic spread (``spread_db``).

    The pattern is what attribution needs and the only part that is portable:
    absolute level depends on the throttle, the pattern does not (measured
    cosine 0.95-1.00 across 50-90 % throttle on DREGON).
    """
    bands = list(BANDS if bands is None else bands)
    rotors = sorted({r for r, _ in clips})
    speeds = sorted({s for _, s in clips})
    n_ch = next(iter(clips.values())).shape[0]
    per_speed = np.full((n_ch, len(rotors), len(bands), len(speeds)), np.nan)

    for (r, s), x in clips.items():
        f, psd = signal.welch(np.asarray(x, dtype=np.float64), fs=sr, nperseg=n_fft, axis=-1)
        for j, (lo, hi) in enumerate(bands):
            m = (f >= lo) & (f < hi)
            per_speed[:, rotors.index(r), j, speeds.index(s)] = np.median(psd[:, m], axis=-1)

    level = np.nanmean(per_speed, axis=0)  # (R, B, S) mean over mics
    pat = per_speed / np.nansum(per_speed, axis=0, keepdims=True)
    basis = np.nanmean(pat, axis=-1)  # (C, R, B) throttle-averaged pattern

    # Throttle stability of the pattern, per rotor and band.
    unit = pat / np.linalg.norm(np.nan_to_num(pat), axis=0, keepdims=True)
    speed_cos = np.nansum(unit[..., :-1] * unit[..., 1:], axis=0)  # (R, B, S-1)

    report = {
        "rotors": np.array(rotors),
        "speeds": np.array(speeds),
        "level_db": 10 * np.log10(np.maximum(level, 1e-30)),
        "speed_cos": speed_cos,
        "spread_db": 10
        * np.log10(np.nanmax(basis, axis=0) / np.maximum(np.nanmin(basis, axis=0), 1e-30)),
    }
    return basis, report


def additivity(
    single: dict[int, np.ndarray], combined: np.ndarray, sr: float, bands=None, *, n_fft=4096
) -> dict[str, np.ndarray]:
    """Test the one assumption the whole method rests on: powers ADD.

    Compares the summed band power of the four single-motor clips against the
    all-motors clip at the same throttle, per microphone and band. A method
    that apportions power between rotors is meaningless if the four rotors
    together do not make the sum of their parts.
    """
    bands = list(BANDS if bands is None else bands)

    def bp(x):
        f, psd = signal.welch(np.asarray(x, dtype=np.float64), fs=sr, nperseg=n_fft, axis=-1)
        return np.stack(
            [np.median(psd[:, (f >= lo) & (f < hi)], axis=-1) for lo, hi in bands], axis=-1
        )

    parts = np.stack([bp(single[r]) for r in sorted(single)], axis=0)  # (R, C, B)
    total = bp(combined)  # (C, B)
    summed = parts.sum(0)
    return {
        "sum_db": 10 * np.log10(np.maximum(summed, 1e-30)),
        "combined_db": 10 * np.log10(np.maximum(total, 1e-30)),
        "excess_db": 10 * np.log10(np.maximum(total, 1e-30) / np.maximum(summed, 1e-30)),
        "share_of_sum": parts / np.maximum(summed[None], 1e-30),
    }


def basis_identifiability(basis: np.ndarray, *, with_floor: bool = True) -> dict[str, np.ndarray]:
    """Conditioning of one band's ``(C, R)`` basis, with an optional flat floor column.

    ``cond`` over the column-normalized design, ``max_cos`` the worst pair of
    rotor columns, ``vif`` the variance-inflation factor per rotor. A rotor
    whose column is a near-copy of another's is not attributable however good
    the data is, and this is where that shows.
    """
    a = np.asarray(basis, dtype=np.float64)
    if with_floor:
        a = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    a = a / np.linalg.norm(a, axis=0, keepdims=True)
    n_rot = basis.shape[1]
    gram = a.T @ a
    iu, ju = np.triu_indices(n_rot, k=1)
    cos = np.abs(gram[iu, ju])
    vif = np.empty(n_rot)
    for r in range(n_rot):
        others = np.delete(a, r, axis=1)
        resid = a[:, r] - others @ np.linalg.lstsq(others, a[:, r], rcond=None)[0]
        vif[r] = 1.0 / max(float(resid @ resid), 1e-12)
    return {
        "cond": np.array(np.linalg.cond(a)),
        "max_cos": np.array(cos.max()),
        "cos_pairs": np.stack([iu, ju, cos], axis=-1),
        "vif": vif,
    }


# ─── Fits ────────────────────────────────────────────────────────────────────


def _nnls_cols(a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NNLS of every column of ``y (N, K)`` against ``a (N, P)``."""
    x = np.empty((a.shape[1], y.shape[1]))
    res = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        x[:, j], res[j] = nnls(a, y[:, j])
    return x, res


def _r2(y: np.ndarray, resid_norm: np.ndarray) -> np.ndarray:
    var = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    return 1.0 - resid_norm**2 / np.maximum(var, 1e-30)


def modulation_regressors(rps: np.ndarray, *, exponent: float = 5.0, ref: float = 80.0):
    """``(R, T)`` rotor speeds -> the per-rotor modulation regressor.

    Broadband rotor self-noise scales with a high power of tip speed (the
    trailing-edge / turbulence-ingestion laws put it near ``V^5``), so the
    regressor is ``(rps / ref) ** exponent``. The exponent only reshapes one
    common curve; what identifies a rotor is the DIFFERENCE between the four
    tracks, which no exponent can create or destroy.
    """
    return (np.asarray(rps, dtype=np.float64) / float(ref)) ** float(exponent)


def fit_free_modulation(power: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    """Per-mic, per-band regression on the four rotor modulations. NO geometry.

    ``p_{c,b}(t) = sum_r B_{r,c,b} s_r(t) + phi_{c,b}``, non-negative, solved
    independently for each ``(c, b)``.

    Returns ``gain`` ``(R, C, B)``, ``floor`` ``(C, B)``, ``r2`` ``(C, B)`` and
    ``share`` ``(R, C, B)`` — the fraction of the MODULATED part of that
    microphone's band power that rotor ``r`` explains. The share is the
    quantity the generator can consume, and it is free of the microphone's own
    unknown sensitivity.
    """
    p = np.asarray(power, dtype=np.float64)  # (C, B, T)
    s = np.asarray(s, dtype=np.float64)  # (R, T)
    n_ch, n_band, n_t = p.shape
    a = np.concatenate([s.T, np.ones((n_t, 1))], axis=1)  # (T, R+1)
    y = p.reshape(n_ch * n_band, n_t).T  # (T, C*B)
    x, res = _nnls_cols(a, y)
    r2 = _r2(y, res).reshape(n_ch, n_band)
    gain = x[:-1].reshape(-1, n_ch, n_band)
    floor = x[-1].reshape(n_ch, n_band)
    contrib = gain * s.mean(-1)[:, None, None]
    share = contrib / np.maximum(contrib.sum(0, keepdims=True), 1e-30)
    return {"gain": gain, "floor": floor, "r2": r2, "share": share, "contrib": contrib}


def fit_basis_modulation(
    power: np.ndarray, s: np.ndarray, basis: np.ndarray, *, per_mic_floor: bool = True
) -> dict[str, np.ndarray]:
    """The constrained fit: one per-rotor level per band, spread by a KNOWN basis.

    ``p_{c,b}(t) = alpha_{r,b} B_{c,r,b} s_r(t) + phi_{c,b}``. Four unknown
    levels (plus ``C`` floors) per band against ``C * T`` observations, so the
    basis is doing the attribution and the fit only scales it. Its ``r2``
    against :func:`fit_free_modulation`'s is the price of believing the basis;
    its ``r2`` under a rotor PERMUTATION of the basis is the null control.
    """
    p = np.asarray(power, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    n_ch, n_band, n_t = p.shape
    n_rot = s.shape[0]
    alpha = np.empty((n_rot, n_band))
    floor = np.empty((n_ch, n_band))
    r2 = np.empty(n_band)
    for b in range(n_band):
        bb = basis[:, :, b] if basis.ndim == 3 else basis  # (C, R)
        # rows = (c, t); rotor column r is B_{c,r} s_r(t)
        rot_cols = (bb[:, None, :] * s.T[None, :, :]).reshape(n_ch * n_t, n_rot)
        fl = np.repeat(np.eye(n_ch), n_t, axis=0) if per_mic_floor else np.ones((n_ch * n_t, 1))
        a = np.concatenate([rot_cols, fl], axis=1)
        y = p[:, b, :].reshape(-1, 1)
        x, res = _nnls_cols(a, y)
        alpha[:, b] = x[:n_rot, 0]
        floor[:, b] = x[n_rot:, 0] if per_mic_floor else x[n_rot, 0]
        r2[b] = _r2(y, res)[0]
    return {"alpha": alpha, "floor": floor, "r2": r2}


def pattern_agreement(gain: np.ndarray, basis: np.ndarray) -> dict[str, np.ndarray]:
    """Does the in-flight, geometry-free pattern match the bench basis?

    ``gain (R, C, B)`` from :func:`fit_free_modulation` against ``basis
    (C, R, B)`` from :func:`bench_basis`. Returns the per-rotor cosine
    ``cos (R, B)`` and the same under every cyclic rotor permutation
    (``cos_perm (P, B)``, averaged over rotors) — the identity check. If the
    true assignment does not beat the permuted ones, the two instruments do not
    agree and no attribution has been earned.
    """
    g = np.asarray(gain, dtype=np.float64)
    m = np.asarray(basis, dtype=np.float64)
    n_rot, _, n_band = g.shape
    gu = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
    mu = m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-30)
    mu = np.transpose(mu, (1, 0, 2))  # (R, C, B)
    cos = np.einsum("rcb,rcb->rb", gu, mu)
    perms = np.stack(
        [np.einsum("rcb,rcb->rb", gu, np.roll(mu, k, axis=0)).mean(0) for k in range(n_rot)]
    )
    return {"cos": cos, "cos_mean": cos.mean(0), "cos_perm": perms}


def fit_linear(power: np.ndarray, design: np.ndarray) -> dict[str, np.ndarray]:
    """Ordinary (SIGNED) least squares of band power on an arbitrary design.

    ``power (C, B, T)``, ``design (T, K)``. Signed, because the useful
    parameterization of four near-collinear rotor speeds is not the rotor basis
    at all but the CONTROL MODES (common, roll, pitch, yaw), and three of those
    four coefficients must be free to go negative.

    Returns ``coef (K, C, B)``, ``r2 (C, B)`` and ``resid_var (C, B)``.
    """
    p = np.asarray(power, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    n_ch, n_band, _ = p.shape
    y = p.reshape(n_ch * n_band, -1).T  # (T, C*B)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    var = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    r2 = 1.0 - (resid**2).sum(0) / np.maximum(var, 1e-30)
    return {
        "coef": coef.reshape(-1, n_ch, n_band),
        "r2": r2.reshape(n_ch, n_band),
        "resid_var": (resid**2).mean(0).reshape(n_ch, n_band),
    }


def mode_design(rps: np.ndarray, mixer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(R, T)`` rotor speeds -> the control-mode design and its column names.

    ``mixer (4, R)`` maps rotor speeds to (common, roll, pitch, yaw) — the
    quadrotor's own allocation. Columns are standardized, and a constant is
    prepended, so the coefficients are comparable and the ``delta r^2`` of the
    three DIFFERENTIAL modes over the common one is exactly the per-rotor
    information the recording contains.
    """
    modes = np.asarray(mixer, dtype=np.float64) @ np.asarray(rps, dtype=np.float64)  # (4, T)
    z = (modes - modes.mean(-1, keepdims=True)) / np.maximum(modes.std(-1, keepdims=True), 1e-12)
    design = np.concatenate([np.ones((z.shape[-1], 1)), z.T], axis=1)
    return design, np.array(["const", "common", "roll", "pitch", "yaw"])


def mode_information(
    power: np.ndarray,
    design: np.ndarray,
    *,
    n_boot: int = 64,
    block_frames: int = 40,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """How much of the band power the DIFFERENTIAL modes explain over the common one.

    The whole per-rotor question reduces to this number. Four rotors that
    always move together carry one degree of freedom, not four: only the part
    of the band power that follows roll / pitch / yaw can ever tell rotors
    apart. ``delta_r2`` is that part; the block bootstrap gives it a null —
    ``delta_r2_q95_shuffled`` is what the same three columns buy after the
    blocks of the DESIGN are shuffled against the data.
    """
    full = fit_linear(power, design)
    base = fit_linear(power, design[:, :2])
    delta = full["r2"] - base["r2"]

    rng = np.random.default_rng(seed)
    n_t = design.shape[0]
    null = []
    for _ in range(n_boot):
        n_block = max(1, n_t // block_frames)
        starts = rng.integers(0, max(1, n_t - block_frames), size=n_block)
        idx = np.concatenate([np.arange(a, a + block_frames) for a in starts])[:n_t]
        d = design.copy()
        d[: len(idx), 2:] = design[idx, 2:]
        f = fit_linear(power, d)
        b = fit_linear(power, d[:, :2])
        null.append(f["r2"] - b["r2"])
    null = np.stack(null)
    return {
        "r2_full": full["r2"],
        "r2_common": base["r2"],
        "delta_r2": delta,
        "delta_r2_null_q95": np.percentile(null, 95, axis=0),
        "coef": full["coef"],
    }


def block_bootstrap(
    power: np.ndarray,
    s: np.ndarray,
    *,
    n_boot: int = 64,
    block_frames: int = 40,
    seed: int = 0,
    basis: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Moving-block bootstrap over TIME of the free-modulation shares.

    Band power is strongly autocorrelated (the rotor speeds are), so a
    frame-wise resample would understate the interval by the block length.
    Returns the 5th / 50th / 95th percentile of ``share`` and, when ``basis``
    is given, of the true-assignment pattern cosine.
    """
    p = np.asarray(power, dtype=np.float64)
    n_t = p.shape[-1]
    rng = np.random.default_rng(seed)
    n_block = max(1, n_t // block_frames)
    shares, cosines = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n_t - block_frames), size=n_block)
        idx = np.concatenate([np.arange(a, a + block_frames) for a in starts])
        idx = idx[idx < n_t]
        f = fit_free_modulation(p[:, :, idx], s[:, idx])
        shares.append(f["share"])
        if basis is not None:
            cosines.append(pattern_agreement(f["gain"], basis)["cos"])
    q = np.percentile(np.stack(shares), [5, 50, 95], axis=0)
    out = {"share_q05": q[0], "share_q50": q[1], "share_q95": q[2]}
    if basis is not None:
        qc = np.percentile(np.stack(cosines), [5, 50, 95], axis=0)
        out |= {"cos_q05": qc[0], "cos_q50": qc[1], "cos_q95": qc[2]}
    return out
