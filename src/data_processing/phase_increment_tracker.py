"""Phase-increment ML instantaneous-frequency tracker (``pi_kalman``).

The phase-validation program's principled refiner. The generative model per
rotor is

    phase of harmonic k:  theta_k(t) = k * phi(t) + b_k(t) + psi_k,

where ``phi`` is the shaft phase (``dphi/dt = 2 pi r(t)``) and ``b_k`` is a
per-harmonic phase random walk (diffusion rate ``q_k`` rad^2/s, coherence
time ``tau_k ~ 2 / q_k``) — the decoherence channel the ladder measured on
real rotor audio, which kills long-window coherent methods. The target is
the maximum-likelihood *instantaneous frequency* ``r(t)``, NOT phase
reconstruction: phase increments of the demodulated harmonic envelopes,

    dpsi_k(t) = arg(z_k(t) conj(z_k(t-1))) ~ 2 pi dt k dr(t) + db_k + v,

stay informative about the shaft-rate error ``dr`` long after absolute
phase has decohered, because the diffusion enters the likelihood as
*measurement noise* (variance ``q_k dt`` per frame) instead of breaking the
signal model. Per outer iteration and rotor:

1. demodulate the audio by ``k phi_hat`` for every usable harmonic (twin
   guard + coarse-to-fine harmonic cap), brickwall-lowpass to ``+-band_hz``
   and decimate to the ``~fs_env`` frame grid -> envelopes ``z_k``, all
   channels;
2. measure ``dpsi_k`` per (channel, harmonic, frame) with per-measurement
   variance = envelope-SNR term (high-SNR von Mises: ``noise^2 / |z|^2``
   per envelope sample; the in-band noise floor is estimated from an
   off-comb demodulation at ``k phi_hat + 2 pi off_comb_hz t``) + the
   diffusion term ``q_k dt``; both are deflated by the analytic
   band-limiting factors of :func:`_band_corrections` — the envelope grid
   oversamples the ``+-band_hz`` band, so an *increment* sees only a small
   fraction of either per-sample variance. Frames whose envelope sits
   at/below the noise floor are gated out (their variance -> infinity
   anyway);
3. estimate ``q_k`` per harmonic FROM THE DATA: an initial smoothing pass
   with ``q_k = 0``, then the robust (MAD) excess variance of the
   ``dpsi_k`` residuals over the SNR-predicted variance:
   ``q_k = max(0, excess) / dt``;
4. fuse all surviving increments in a scalar random-walk Kalman smoother
   (RTS) over frames — state ``dr(t)``, process variance
   ``sigma_process^2 dt`` per step — then ``r_hat += dr`` (clipped) and
   re-demodulate on the next outer iteration.

Coarse-to-fine: iteration ``j`` only admits harmonics up to ``k_caps[j]``
(``warp_refinement``'s ambiguity-ladder idea — while the residual is large,
high orders both alias across frames, ``k |dr| dt`` approaching half a
cycle, and fall out of the demod band); a wrap guard additionally drops
near-``pi`` increments. Multi-rotor: rotors are refined sequentially inside
each iteration under the ``warp_refinement`` twin rule, applied *per frame*
— measurements of harmonic ``k`` of rotor ``i`` are gated out at the frames
where any harmonic of any other rotor comes within ``band_hz + guard_hz``
of it (harmonics collided at every frame are dropped outright).

Pair-coupled twin estimation (``pair_mode="joint"``): on tight twin pairs
(rate split below ``pair_max_split``) the gate discards most measurements —
the collided band contains a *two-phasor mixture*, not garbage. In joint
mode each pair's self-collided harmonics are demodulated ONCE by the
pair-mean phase ``k phibar`` in a band wide enough for ``+-k split/2``; per
short window the two-tone spectrum is fitted (two-peak pick + parabolic
interpolation, channel-incoherent power) and the two line frequencies are
assigned to the rotors *by order* (higher line -> faster rotor — immune to
search-band ambiguity), giving direct per-rotor rate observations
(equivalently the pair mean and split) that feed the same per-rotor
smoothers as ``H = 1`` measurements. The mixture's *phase increments* are
deliberately NOT used as a pair-mean measurement: the argument of a
two-phasor sum advances at the STRONGER component's frequency on average
(winding number), which biases increment-based means whenever the twin
amplitudes differ. Harmonics contaminated by non-pair rotors inside the
widened band keep the plain gate.

Stated approximations: noise of adjacent increments is treated as
independent (it is MA(1) — neighbours share one envelope sample); the
wrapped-phase likelihood is Gaussian (fine above the SNR gate); ``q_k`` is
time-constant within an iteration. All are absorbed in practice by the
process/measurement variance split.

Pure numpy/scipy, CPU; cost comparable to ``iter_warp_refine`` (seconds per
20 s mono cell). Diagnostics carry the per-(rotor, iteration) ``q_k`` /
implied ``tau_k`` budgets — the empirical decoherence budgets that the
ladder's coherence-time curves cross-check.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np

from data_processing.vk_tracking import _fft_workers

__all__ = ["DEFAULTS", "pi_kalman_refine"]

_TINY = 1e-30
_MAX_CHANNELS = 8  # multichannel fusion cap (vk_tracking convention)


# ---------------------------------------------------------------------------
# demodulation


def _zoom_lp_decimate(x: np.ndarray, stride: int, n_env: int, band_cyc: float) -> np.ndarray:
    """FFT brickwall lowpass (``|f| <= band_cyc`` cycles/sample) + decimate.

    The zoom-IFFT of :func:`data_processing.vk_tracking._fft_lp_decimate`
    with a *parametric* cutoff below the decimated Nyquist: zero-pad the
    complex input to ``stride * n_env``, keep the ``+-band_cyc`` band
    (positive and negative bins — the input is complex), inverse-FFT at
    length ``n_env`` directly. Circular edge handling; callers trim edges.
    """
    from scipy import fft as sfft

    w = _fft_workers()
    n_pad = stride * n_env
    xc = np.asarray(x, dtype=np.complex64)
    spec = cast(np.ndarray, sfft.fft(xc, n=n_pad, axis=-1, workers=w))
    b = min(int(np.floor(band_cyc * n_pad)), (n_env - 1) // 2)
    low = np.zeros(x.shape[:-1] + (n_env,), dtype=np.complex64)
    low[..., : b + 1] = spec[..., : b + 1]
    if b > 0:
        low[..., -b:] = spec[..., -b:]
    dec = cast(np.ndarray, sfft.ifft(low, axis=-1, workers=w))
    return (dec / np.complex64(stride)).astype(np.complex128)


def _demod_bank(
    y32: np.ndarray,
    phi: np.ndarray,
    t_aud: np.ndarray,
    ks: list[int],
    off_hz: float,
    stride: int,
    n_env: int,
    band_cyc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """On-comb and off-comb envelope banks for one rotor.

    Returns ``(z_on, z_off)``, each ``(C, K, n_env)`` complex128: the audio
    demodulated by ``k * phi`` (resp. ``k * phi + 2 pi off_hz t``),
    brickwall-lowpassed to ``+-band_cyc`` and decimated. Carriers come from
    the harmonic power recursion (one exp for the fundamental, complex64
    multiplies per harmonic step — ``vk_tracking._track_carriers``' trick);
    the off-comb carrier is the on-comb one times one shared ramp phasor,
    so the noise-floor probe costs no extra exp.
    """
    n_ch, n_t = y32.shape
    n_k = len(ks)
    z_on = np.empty((n_ch, n_k, n_env), dtype=np.complex128)
    z_off = np.empty_like(z_on)
    c1 = np.exp(-1j * phi).astype(np.complex64)
    ramp = np.exp(-2j * np.pi * off_hz * t_aud).astype(np.complex64)
    chunk = max(1, int(96e6 / (max(1, n_ch) * max(1, n_t) * 8)))
    buf = np.empty((n_ch, min(chunk, n_k), n_t), dtype=np.complex64)
    idxs: list[int] = []

    def flush() -> None:
        m = len(idxs)
        z_on[:, idxs] = _zoom_lp_decimate(buf[:, :m], stride, n_env, band_cyc)
        buf[:, :m] *= ramp
        z_off[:, idxs] = _zoom_lp_decimate(buf[:, :m], stride, n_env, band_cyc)
        idxs.clear()

    cur = np.ones_like(c1)
    cur_k = 0
    for a, k in enumerate(ks):
        step = k - cur_k
        if step > 2:  # rare gaps (twin-excluded runs): one pow, not many muls
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
    return z_on, z_off


# ---------------------------------------------------------------------------
# gating


def _twin_collision_mask(
    r_ft: np.ndarray, i: int, k_top: int, sep_hz: float, f_max: float, min_rate: float
) -> np.ndarray:
    """Per-frame twin-collision mask for rotor ``i``: ``(k_top, N)`` bool.

    The ``warp_refinement`` twin rule in Hz, evaluated *per frame*: entry
    ``(k-1, t)`` is True when ``|k r_i(t) - k' r_j(t)| < sep_hz`` for any
    other rotor ``j`` and any harmonic ``k'`` (only the two ``k'``
    bracketing ``k r_i / r_j`` can enter the band). Collided (harmonic,
    frame) *measurements* are gated out rather than dropping the whole
    harmonic — on real multi-rotor flight the trajectories cross and part,
    and an any-frame exclusion rule empties the harmonic set entirely.
    Near-silent rotors (mean rate below ``min_rate``) carry no comb and are
    skipped.
    """
    ks = np.arange(1, k_top + 1, dtype=np.float64)
    fi = ks[:, None] * r_ft[i][None, :]  # (K, N)
    coll = np.zeros(fi.shape, dtype=bool)
    for j in range(r_ft.shape[0]):
        if j == i or float(np.mean(r_ft[j])) < min_rate:
            continue
        rj = np.maximum(r_ft[j], 1e-3)[None, :]
        base = fi / rj
        for kp in (np.floor(base), np.ceil(base)):
            fj = np.maximum(kp, 1.0) * rj
            coll |= (np.abs(fj - fi) < sep_hz) & (fj <= f_max + sep_hz)
    return coll


def _band_corrections(band_hz: float, dt: float) -> tuple[float, float]:
    """Adjacent-frame correction factors for the ``+-band_hz`` brickwall.

    The envelope grid oversamples the demod band (``2 band_hz < fs_env``),
    so both noise and diffusion contribute to a *phase increment* far less
    than their per-sample variances suggest:

    * ``c_noise``: in-band white noise has adjacent-sample correlation
      ``rho = sinc(2 B dt)``; the increment noise variance is
      ``(1 - rho)`` times the naive two-sample sum.
    * ``c_diff``: a Brownian phase (rate ``q``) seen through the brickwall
      has increment variance ``c_diff * q * dt`` with
      ``c_diff = (2 / pi) * int_0^{2 pi B dt} (1 - cos u) / u^2 du``
      (the ``|1 - e^{i w dt}|^2 / w^2`` spectrum integrated over the band).

    Both matter for the data-driven ``q_k``: the SNR-predicted variance must
    be deflated by ``c_noise`` before the residual excess is read, and the
    excess maps back to a diffusion rate through ``c_diff``.
    """
    rho = float(np.sinc(2.0 * band_hz * dt))  # np.sinc(x) = sin(pi x)/(pi x)
    c_noise = max(1.0 - rho, 1e-3)
    u = np.linspace(0.0, 2.0 * np.pi * band_hz * dt, 512)[1:]
    integrand = np.concatenate(([0.5], (1.0 - np.cos(u)) / u**2))
    c_diff = max(2.0 / np.pi * float(np.trapezoid(integrand, np.concatenate(([0.0], u)))), 1e-3)
    return c_noise, c_diff


# ---------------------------------------------------------------------------
# pair-coupled twin estimation (pair_mode="joint")


def _assign_pairs(r_ft: np.ndarray, max_split: float, min_rate: float) -> list[tuple[int, int]]:
    """Greedy proximity pairing: ``(lo, hi)`` rotor pairs with mean-rate split
    below ``max_split`` (each rotor in at most one pair; near-silent rotors
    are never paired). Adjacent-in-rate rotors pair first."""
    means = r_ft.mean(axis=-1)
    order = [int(a) for a in np.argsort(means) if means[a] >= min_rate]
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for a in range(len(order) - 1):
        i, j = order[a], order[a + 1]
        if i in used or j in used:
            continue
        if means[j] - means[i] < max_split:
            pairs.append((i, j))
            used.update((i, j))
    return pairs


def _parab_peak(logp: np.ndarray, p: int, freqs: np.ndarray) -> float:
    """Sub-bin peak frequency by parabolic interpolation on log power
    (``freqs`` uniformly spaced, ``0 < p < len - 1``)."""
    y0, y1, y2 = float(logp[p - 1]), float(logp[p]), float(logp[p + 1])
    denom = y0 - 2.0 * y1 + y2
    off = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    off = float(np.clip(off, -0.5, 0.5))
    return float(freqs[p] + off * (freqs[1] - freqs[0]))


def _pair_joint_obs(
    y32: np.ndarray,
    t_aud: np.ndarray,
    r: np.ndarray,
    lo: int,
    hi: int,
    ft: np.ndarray,
    sr: int,
    stride: int,
    n_env: int,
    dt: float,
    k_cap: int,
    *,
    band_hz: float,
    f_max: float,
    guard_hz: float,
    min_rate: float,
    joint_win_s: float,
    joint_snr_min: float,
    n_trim: int,
    pair_max_split: float,
) -> tuple[dict[int, list[tuple[int, float, float]]], dict[str, Any]]:
    """Two-tone rate observations for one twin pair ``(lo, hi)``.

    For each self-collided harmonic ``k`` of the pair (and only those free
    of non-pair contamination inside the widened band), the audio is
    demodulated by the pair-mean phase ``k phibar`` in a band covering both
    lines, and per sliding window the two strongest spectral lines are
    picked (channel-incoherent power, parabolic sub-bin interpolation) and
    assigned to the rotors by order. Returns ``{rotor: [(frame_idx,
    delta_obs_rev_s, var), ...]}`` on the increment grid plus diagnostics.

    The *track* split is treated as unreliable — it is part of the error
    being estimated (telemetry inits routinely carry half the true split).
    Search bands and the split-plausibility window are therefore bounded by
    ``pair_max_split`` (the pairing radius), not by the track split; peak
    exclusion is resolution-based (``2 / window``), and a Hann-sidelobe
    amplitude guard rejects second peaks more than ~27 dB below the first.
    """
    fs_e = 1.0 / dt
    rbar_ft = 0.5 * (r[lo] + r[hi])
    d_ft = r[hi] - r[lo]  # signed track split (rev/s), hi - lo
    d_abs = np.abs(d_ft)
    d_med = float(np.median(d_abs))
    sep = band_hz + guard_hz
    rbar_aud = np.interp(t_aud, ft, rbar_ft)
    k_top = min(k_cap, int(np.floor(f_max / max(float(rbar_aud.max()), 1e-3))))
    others = [
        j for j in range(r.shape[0]) if j not in (lo, hi) and float(np.mean(r[j])) >= min_rate
    ]
    diag: dict[str, Any] = {"pair": [lo, hi], "split_track_med": round(d_med, 3)}
    obs: dict[int, list[tuple[int, float, float]]] = {lo: [], hi: []}

    # Effective split scale for the search geometry: robust (p75, not max —
    # transient excursions must not blow the band past the envelope Nyquist),
    # at least half and at most the full pairing radius — the track split
    # itself is unreliable (it is part of the init error being estimated).
    d_eff = min(max(float(np.percentile(d_abs, 75)), 0.5 * pair_max_split), pair_max_split)
    ks_joint: list[int] = []
    for k in range(1, k_top + 1):
        if not bool((k * d_abs < sep).any()):
            continue  # pair never self-collides at this harmonic
        bw_k = 0.5 * k * d_eff + band_hz
        if bw_k > 0.45 * fs_e:
            continue  # two-tone band does not fit the envelope Nyquist
        contaminated = False
        f_pair = k * rbar_ft
        for j in others:
            rj = np.maximum(r[j], 1e-3)
            base = f_pair / rj
            for kp in (np.floor(base), np.ceil(base)):
                fj = np.maximum(kp, 1.0) * rj
                if bool((np.abs(fj - f_pair) < bw_k + guard_hz).any()):
                    contaminated = True
                    break
            if contaminated:
                break
        if not contaminated:
            ks_joint.append(k)
    diag["ks"] = ks_joint
    if not ks_joint or d_med < 1e-3:
        diag["skipped"] = "no joint harmonics" if ks_joint == [] else "zero split"
        return obs, diag

    phibar = 2.0 * np.pi * np.cumsum(rbar_aud) / sr
    t_env = np.arange(n_env) * dt
    n_m = n_env - 1
    n_locked = 0
    n_windows = 0
    splits: list[float] = []
    for k in ks_joint:
        band_k_hz = 0.5 * k * d_eff + band_hz
        phasor = np.exp(-1j * (k * phibar)).astype(np.complex64)
        z = _zoom_lp_decimate(y32 * phasor[None, :], stride, n_env, band_k_hz / sr)  # (C, n_env)
        w_k_s = max(joint_win_s, 1.5 / (k * max(d_med, 1e-3)))
        n_w = int(round(w_k_s * fs_e))
        n_w = min(n_w, max(8, (n_env - 2 * n_trim) // 2))
        if n_w < 8:
            continue
        w_k_s = n_w * dt
        window = np.hanning(n_w)
        nfft = 8 * (1 << int(np.ceil(np.log2(n_w))))
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))
        df = float(freqs[1] - freqs[0])
        hop = max(n_w // 2, 1)
        if k * d_eff * w_k_s < 1.2:
            continue  # even the effective split is unresolvable in this window
        for a in range(n_trim, n_env - n_trim - n_w + 1, hop):
            n_windows += 1
            t_c = t_env[a] + 0.5 * (n_w - 1) * dt
            d_w = float(np.interp(t_c, ft, d_ft))
            rbar_w = float(np.interp(t_c, ft, rbar_ft))
            seg = z[:, a : a + n_w] * window
            power = np.fft.fftshift((np.abs(np.fft.fft(seg, n=nfft, axis=-1)) ** 2).sum(axis=0))
            in_band = np.abs(freqs) <= band_k_hz
            if in_band.sum() < 8:
                continue
            p_band = np.where(in_band, power, 0.0)
            p1 = int(np.argmax(p_band))
            excl_r = 2.0 / w_k_s  # resolution-based (covers the first Hann sidelobe)
            mask2 = in_band & (np.abs(freqs - freqs[p1]) >= excl_r)
            if not mask2.any():
                continue
            p2 = int(np.argmax(np.where(mask2, power, 0.0)))
            near = (np.abs(freqs - freqs[p1]) < excl_r) | (np.abs(freqs - freqs[p2]) < excl_r)
            floor_bins = in_band & ~near
            if floor_bins.sum() < 4:
                continue
            floor = float(np.median(power[floor_bins]))
            snr2 = float(power[p2] / max(floor, _TINY))  # the weaker of the two lines
            if snr2 < joint_snr_min:
                continue
            if power[p2] < 2e-3 * power[p1]:
                continue  # > ~27 dB below the main line: sidelobe, not a twin
            if not (0 < p1 < nfft - 1 and 0 < p2 < nfft - 1):
                continue
            logp = np.log(power + _TINY)
            f1 = _parab_peak(logp, p1, freqs)
            f2 = _parab_peak(logp, p2, freqs)
            s_meas = abs(f1 - f2)
            if not (2.0 / w_k_s <= s_meas <= k * pair_max_split + 2.0 / w_k_s):
                continue  # unresolved or outside the pairing radius: mis-pick
            rho_lo, rho_hi = sorted((rbar_w + f1 / k, rbar_w + f2 / k))
            if d_w < 0:  # track order inverted: hi-track rotor is the slower line
                rho_lo, rho_hi = rho_hi, rho_lo
            y_lo = rho_lo - float(np.interp(t_c, ft, r[lo]))
            y_hi = rho_hi - float(np.interp(t_c, ft, r[hi]))
            if max(abs(y_lo), abs(y_hi)) > 1.5:
                continue
            var = max(df, 1.0 / w_k_s) ** 2 / (8.0 * max(snr2 - 1.0, 0.5)) / float(k * k)
            var = max(var, 1e-6)
            idx = int(np.clip(round(t_c / dt - 0.5), 0, n_m - 1))
            obs[lo].append((idx, y_lo, var))
            obs[hi].append((idx, y_hi, var))
            splits.append(s_meas / k)
            n_locked += 1
    diag["n_windows"] = n_windows
    diag["n_windows_locked"] = n_locked
    if splits:
        diag["split_meas_med"] = round(float(np.median(splits)), 4)
    return obs, diag


# ---------------------------------------------------------------------------
# scalar random-walk Kalman smoother


def _rw_kalman_rts(
    info: np.ndarray, mean_info: np.ndarray, q_step: float, p0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar random-walk Kalman filter + RTS smoother, information-fed.

    ``info[j] = sum_i H_i^2 / R_i`` and ``mean_info[j] = sum_i H_i y_i / R_i``
    over the measurements landing on frame ``j`` (0 where none — the step
    becomes a pure prediction). Returns the smoothed state mean and
    variance, each ``(n,)``.
    """
    n = len(info)
    m_f = np.empty(n)
    p_f = np.empty(n)
    m_p = np.empty(n)
    p_p = np.empty(n)
    m_prev, p_prev = 0.0, float(p0)
    for j in range(n):
        pp = p_prev + (q_step if j > 0 else 0.0)
        mp = m_prev
        m_p[j] = mp
        p_p[j] = pp
        pf = 1.0 / (1.0 / pp + info[j])
        mf = pf * (mp / pp + mean_info[j])
        m_f[j] = mf
        p_f[j] = pf
        m_prev, p_prev = mf, pf
    m_s = m_f.copy()
    p_s = p_f.copy()
    for j in range(n - 2, -1, -1):
        a = p_f[j] / p_p[j + 1]
        m_s[j] = m_f[j] + a * (m_s[j + 1] - m_p[j + 1])
        p_s[j] = p_f[j] + a * a * (p_s[j + 1] - p_p[j + 1])
    return m_s, p_s


def _smooth_delta(
    dpsi: np.ndarray,
    r_var: np.ndarray,
    valid: np.ndarray,
    h: np.ndarray,
    q_step: float,
    p0: float,
    ess: float = 1.0,
    extra: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RTS-smoothed ``dr`` from gated increments (state scalar per frame).

    ``dpsi`` / ``r_var`` / ``valid``: ``(C, K, n-1)``; ``h``: ``(K,)`` the
    per-harmonic observation gain ``2 pi k dt``. All valid measurements of
    one frame are folded into the information pair before the scan. ``ess``
    scales the information: successive increments are correlated by the
    demod lowpass (only ``~2 band_hz dt`` independent increments per frame
    when the band is oversampled), and without the effective-sample-size
    deflation the posterior overcounts the band and chases in-band noise.
    ``extra``: optional ``(idx, y, var)`` direct-rate observations
    (``H = 1``, rev/s — the joint pair-mode two-tone measurements), folded
    into the same information pair at their frame indices (no ess: the
    windows hop by half their length and are near-independent).
    """
    w = np.where(valid, ess / np.maximum(r_var, _TINY), 0.0)
    hw = h[None, :, None]
    info = np.sum(hw**2 * w, axis=(0, 1))
    mean_info = np.sum(hw * dpsi * w, axis=(0, 1))
    if extra is not None:
        idx, y, var = extra
        np.add.at(info, idx, 1.0 / var)
        np.add.at(mean_info, idx, y / var)
    return _rw_kalman_rts(info, mean_info, q_step, p0)


# ---------------------------------------------------------------------------
# one (rotor, outer iteration) pass


def _rotor_pass(
    y32: np.ndarray,
    t_aud: np.ndarray,
    r: np.ndarray,
    i: int,
    ft: np.ndarray,
    sr: int,
    stride: int,
    n_env: int,
    dt: float,
    t_mid: np.ndarray,
    band_cyc: float,
    k_cap: int,
    *,
    band_hz: float,
    off_comb_hz: float,
    f_max: float,
    guard_hz: float,
    snr_gate: float,
    wrap_guard_rad: float,
    n_trim: int,
    q_step: float,
    p0: float,
    min_rate: float,
    extra_obs: list[tuple[int, float, float]] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """One outer iteration for rotor ``i``: ``(delta on ft grid | None, diag)``."""
    r_aud = np.interp(t_aud, ft, r[i])
    mean_rate = float(np.mean(r_aud))
    d: dict[str, Any] = {"k_cap": int(k_cap), "mean_rate": round(mean_rate, 3)}
    extra = None
    if extra_obs:
        arr = np.asarray(extra_obs, dtype=np.float64)
        extra = (arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2])
        d["n_joint_obs"] = len(extra_obs)
    if mean_rate < min_rate:
        d["skipped"] = f"mean rate {mean_rate:.1f} < min_rate {min_rate}"
        return None, d
    k_top = min(k_cap, int(np.floor(f_max / max(float(r_aud.max()), 1e-3))))
    if k_top < 1:
        d["skipped"] = "no harmonic below f_max"
        return None, d
    coll = _twin_collision_mask(r, i, k_top, band_hz + guard_hz, f_max, min_rate)
    fully = np.asarray(coll.all(axis=1), dtype=bool)
    ks = [k for k in range(1, k_top + 1) if not fully[k - 1]]
    d["n_twin_excluded"] = int(fully.sum())
    d["ks"] = ks
    if not ks:
        if extra is None:
            d["skipped"] = "all harmonics twin-excluded"
            return None, d
        # Joint-pair observations alone: smooth them without any increments.
        n_m = n_env - 1
        info = np.zeros(n_m)
        mean_info = np.zeros(n_m)
        np.add.at(info, extra[0], 1.0 / extra[2])
        np.add.at(mean_info, extra[0], extra[1] / extra[2])
        m1, p1 = _rw_kalman_rts(info, mean_info, q_step, p0)
        d["delta_rms"] = round(float(np.sqrt(np.mean(m1**2))), 4)
        d["post_std_med"] = round(float(np.median(np.sqrt(p1))), 4)
        return np.interp(ft, t_mid, m1), d

    phi = 2.0 * np.pi * np.cumsum(r_aud) / sr
    z, z_off = _demod_bank(y32, phi, t_aud, ks, off_comb_hz, stride, n_env, band_cyc)
    interior = slice(n_trim, max(n_trim + 1, n_env - n_trim))
    noise_pow = np.maximum(np.mean(np.abs(z_off[..., interior]) ** 2, axis=-1), _TINY)  # (C, K)

    a2 = np.abs(z) ** 2  # (C, K, n_env)
    dpsi = np.angle(z[..., 1:] * np.conj(z[..., :-1]))  # (C, K, n_env - 1)
    inv0 = 1.0 / np.maximum(a2[..., :-1], _TINY)
    inv1 = 1.0 / np.maximum(a2[..., 1:], _TINY)
    c_noise, c_diff = _band_corrections(band_hz, dt)
    var_meas = c_noise * 0.5 * noise_pow[..., None] * (inv0 + inv1)
    valid = (a2[..., 1:] > snr_gate * noise_pow[..., None]) & (
        a2[..., :-1] > snr_gate * noise_pow[..., None]
    )
    valid &= np.abs(dpsi) < wrap_guard_rad
    n_m = n_env - 1
    tmask = np.zeros(n_m, dtype=bool)
    tmask[n_trim : max(n_trim, n_m - n_trim)] = True
    valid &= tmask[None, None, :]
    # Per-frame twin gate on the increment grid: a measurement is dropped
    # when its span sits (mostly) inside a collision interval of its harmonic.
    coll_mid = np.stack(
        [np.interp(t_mid, ft, coll[k - 1].astype(np.float64)) for k in ks]
    )  # (K, n_m)
    valid &= coll_mid[None, :, :] < 0.5
    d["twin_gated_frac"] = round(float(np.mean(coll_mid >= 0.5)), 4)
    d["n_meas"] = int(valid.sum())
    d["n_meas_total"] = int(valid.size)
    if not valid.any() and extra is None:
        d["skipped"] = "no measurement passed the SNR/wrap gates"
        return None, d

    kf = np.asarray(ks, dtype=np.float64)
    h = 2.0 * np.pi * kf * dt
    ess = min(1.0, 2.0 * band_hz * dt)  # independent increments per frame
    # Pass A (q_k = 0) -> data-driven q_k from the robust residual excess.
    m0, _ = _smooth_delta(dpsi, var_meas, valid, h, q_step, p0, ess=ess, extra=extra)
    resid = dpsi - h[None, :, None] * m0[None, None, :]
    q_k = np.zeros(len(ks))
    q_ok = np.zeros(len(ks), dtype=bool)
    for a in range(len(ks)):
        v = valid[:, a, :]
        rr = resid[:, a, :][v]
        if rr.size < 16:
            continue  # too few increments to calibrate — leave q at 0
        var_rob = (1.4826 * float(np.median(np.abs(rr - np.median(rr))))) ** 2
        excess = var_rob - float(np.mean(var_meas[:, a, :][v]))
        q_k[a] = max(0.0, excess) / (c_diff * dt)
        q_ok[a] = True
    # Pass B: diffusion-aware measurement variances (the in-band effective
    # diffusion contribution per increment is c_diff * q_k * dt = the excess).
    m1, p1 = _smooth_delta(
        dpsi,
        var_meas + (q_k * c_diff * dt)[None, :, None],
        valid,
        h,
        q_step,
        p0,
        ess=ess,
        extra=extra,
    )

    d["q_k"] = {str(k): round(float(q), 5) for k, q, ok in zip(ks, q_k, q_ok) if ok}
    d["tau_k"] = {
        str(k): (round(2.0 / float(q), 4) if q > 0 else None)
        for k, q, ok in zip(ks, q_k, q_ok)
        if ok
    }
    d["noise_floor_db"] = {
        str(k): round(10.0 * float(np.log10(np.median(noise_pow[:, a]) + _TINY)), 1)
        for a, k in enumerate(ks)
    }
    d["delta_rms"] = round(float(np.sqrt(np.mean(m1**2))), 4)
    d["post_std_med"] = round(float(np.median(np.sqrt(p1))), 4)
    return np.interp(ft, t_mid, m1), d


# ---------------------------------------------------------------------------
# public API


def pi_kalman_refine(
    audio: np.ndarray,
    r_init: np.ndarray,
    ft: np.ndarray,
    sr: int = 16000,
    *,
    n_iter: int = 3,
    fs_env: float = 62.5,
    band_hz: float = 6.0,
    off_comb_hz: float = 11.0,
    k_max: int = 40,
    f_max: float = 6000.0,
    k_caps: tuple[int, ...] = (8, 20, 40),
    sigma_process: float = 2.0,
    sigma_prior: float = 2.0,
    guard_hz: float = 1.0,
    snr_gate: float = 2.0,
    wrap_guard_rad: float = 2.8,
    max_step: float = 3.0,
    edge_trim_s: float = 0.25,
    min_rate: float = 5.0,
    pair_mode: str = "gate",
    pair_max_split: float = 1.5,
    joint_win_s: float = 0.5,
    joint_snr_min: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """ML instantaneous-frequency refinement by phase-increment Kalman smoothing.

    Args:
        audio: ``(C, T)`` or ``(T,)`` audio at ``sr`` (channels capped at 8).
        r_init: ``(R, N)`` initial IF tracks, rev/s on the frame grid ``ft``.
        ft: ``(N,)`` uniform frame times (seconds, audio-relative).
        sr: audio sample rate.
        n_iter: outer demodulate→smooth→correct iterations.
        fs_env: target envelope/measurement frame rate (Hz); the actual rate
            is ``sr / round(sr / fs_env)`` (62.5 Hz = 16 ms at 16 kHz).
        band_hz: demodulation half-band (Hz) — envelopes keep ``+-band_hz``
            around each harmonic.
        off_comb_hz: offset of the noise-floor probe demodulation (Hz);
            must exceed ``band_hz`` (else the probe band contains the comb
            line itself) and stay well below the fundamental.
        k_max: highest harmonic index considered.
        f_max: highest harmonic frequency used (Hz).
        k_caps: per-iteration harmonic caps (coarse-to-fine ambiguity
            ladder); the last entry repeats for extra iterations.
        sigma_process: random-walk process scale of the IF error state,
            rev/s per sqrt(s) — deliberately aggressive by default so real
            shaft fluctuation is trackable.
        sigma_prior: prior std of the initial IF error (rev/s).
        guard_hz: twin-rejection margin on top of ``band_hz``.
        snr_gate: envelope power gate in units of the off-comb noise floor.
        wrap_guard_rad: drop increments with ``|dpsi|`` above this (wrap
            ambiguity protection near ``pi``).
        max_step: per-iteration correction clip (rev/s).
        edge_trim_s: envelope-grid edge exclusion (filter/wrap transients).
        min_rate: rotors whose mean rate is below this (rev/s) are skipped.
        pair_mode: ``"gate"`` discards twin-collided measurements (default);
            ``"joint"`` additionally extracts two-tone rate observations
            from each tight pair's self-collided harmonics (module
            docstring, "Pair-coupled twin estimation").
        pair_max_split: mean-rate split (rev/s) below which two rotors form
            a joint pair.
        joint_win_s: minimum two-tone analysis window (s); low harmonics
            stretch it to keep ``k * split * window >= 1.5`` cycles.
        joint_snr_min: power-SNR gate on the *weaker* of the two lines.

    Returns:
        ``(r_refined, diagnostics)`` — refined ``(R, N)`` tracks and a
        JSON-serializable dict with, per rotor and outer iteration, the
        admitted harmonic set, gate counts, the data-estimated ``q_k`` /
        implied ``tau_k`` per harmonic, off-comb noise floors, and the
        posterior summary (``delta_rms``, ``post_std_med``, step sizes).
    """
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))[:_MAX_CHANNELS]
    r = np.array(r_init, dtype=np.float64, copy=True)
    if r.ndim != 2:
        raise ValueError(f"r_init must be (R, N), got shape {r.shape}")
    ft = np.asarray(ft, dtype=np.float64)
    if r.shape[-1] != len(ft):
        raise ValueError(f"r_init has {r.shape[-1]} frames, ft has {len(ft)}")
    if off_comb_hz <= band_hz:
        raise ValueError(f"off_comb_hz={off_comb_hz} must exceed band_hz={band_hz}")
    if pair_mode not in ("gate", "joint"):
        raise ValueError(f"unknown pair_mode {pair_mode!r} (expected 'gate' or 'joint')")
    n_t = x.shape[-1]
    t_aud = np.arange(n_t) / sr
    stride = max(1, int(round(sr / fs_env)))
    fs_e = sr / stride
    dt = 1.0 / fs_e
    n_env = len(range(0, n_t, stride))
    if n_env < 8:
        raise ValueError(f"clip too short: {n_env} envelope frames at fs_env={fs_e:.1f}")
    t_env = np.arange(n_env) * dt
    t_mid = t_env[:-1] + 0.5 * dt  # increments live between envelope samples
    band_cyc = band_hz / sr
    y32 = x.astype(np.float32)
    n_trim = max(1, int(round(edge_trim_s * fs_e)))
    q_step = sigma_process**2 * dt
    p0 = sigma_prior**2
    schedule = [int(min(k_caps[min(j, len(k_caps) - 1)], k_max)) for j in range(n_iter)]

    rotor_diags: list[dict[str, Any]] = [{"rotor": i, "iters": []} for i in range(r.shape[0])]
    pair_diags: list[list[dict[str, Any]]] = []
    for it, k_cap in enumerate(schedule):
        joint_obs: dict[int, list[tuple[int, float, float]]] = {}
        if pair_mode == "joint":
            it_pair_diags: list[dict[str, Any]] = []
            for lo, hi in _assign_pairs(r, pair_max_split, min_rate):
                obs, pd = _pair_joint_obs(
                    y32,
                    t_aud,
                    r,
                    lo,
                    hi,
                    ft,
                    sr,
                    stride,
                    n_env,
                    dt,
                    k_cap,
                    band_hz=band_hz,
                    f_max=f_max,
                    guard_hz=guard_hz,
                    min_rate=min_rate,
                    joint_win_s=joint_win_s,
                    joint_snr_min=joint_snr_min,
                    n_trim=n_trim,
                    pair_max_split=pair_max_split,
                )
                pd["iter"] = it + 1
                it_pair_diags.append(pd)
                for rot, lst in obs.items():
                    joint_obs.setdefault(rot, []).extend(lst)
            pair_diags.append(it_pair_diags)
        for i in range(r.shape[0]):
            delta_ft, d = _rotor_pass(
                y32,
                t_aud,
                r,
                i,
                ft,
                sr,
                stride,
                n_env,
                dt,
                t_mid,
                band_cyc,
                k_cap,
                band_hz=band_hz,
                off_comb_hz=off_comb_hz,
                f_max=f_max,
                guard_hz=guard_hz,
                snr_gate=snr_gate,
                wrap_guard_rad=wrap_guard_rad,
                n_trim=n_trim,
                q_step=q_step,
                p0=p0,
                min_rate=min_rate,
                extra_obs=joint_obs.get(i),
            )
            d["iter"] = it + 1
            if delta_ft is not None:
                step = np.clip(delta_ft, -max_step, max_step)
                r[i] += step
                d["step_rms"] = round(float(np.sqrt(np.mean(step**2))), 4)
                d["step_max"] = round(float(np.max(np.abs(step))), 4)
            rotor_diags[i]["iters"].append(d)

    diagnostics: dict[str, Any] = {
        "params": {
            "n_iter": n_iter,
            "fs_env": fs_env,
            "band_hz": band_hz,
            "off_comb_hz": off_comb_hz,
            "k_max": k_max,
            "f_max": f_max,
            "k_caps": list(k_caps),
            "sigma_process": sigma_process,
            "sigma_prior": sigma_prior,
            "guard_hz": guard_hz,
            "snr_gate": snr_gate,
            "wrap_guard_rad": wrap_guard_rad,
            "max_step": max_step,
            "edge_trim_s": edge_trim_s,
            "min_rate": min_rate,
            "pair_mode": pair_mode,
            "pair_max_split": pair_max_split,
            "joint_win_s": joint_win_s,
            "joint_snr_min": joint_snr_min,
        },
        "fs_env_actual": fs_e,
        "k_schedule": schedule,
        "rotors": rotor_diags,
    }
    if pair_diags:
        diagnostics["pairs"] = pair_diags
    return r, diagnostics


def _defaults() -> dict[str, Any]:
    sig = inspect.signature(pi_kalman_refine)
    return {
        name: (list(p.default) if isinstance(p.default, tuple) else p.default)
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


#: Default parameter values of :func:`pi_kalman_refine` (for run manifests).
DEFAULTS: dict[str, Any] = _defaults()
