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

from tracking.vk_tracking import fft_workers

__all__ = ["DEFAULTS", "pi_kalman_refine"]

_TINY = 1e-30
_MAX_CHANNELS = 8  # multichannel fusion cap (vk_tracking convention)


# ---------------------------------------------------------------------------
# demodulation


def zoom_lp_decimate(
    x: np.ndarray,
    stride: int,
    n_env: int,
    band_cyc: float,
    band_cyc_rows: np.ndarray | None = None,
) -> np.ndarray:
    """FFT brickwall lowpass (``|f| <= band_cyc`` cycles/sample) + decimate.

    The zoom-IFFT of :func:`tracking.vk_tracking._fft_lp_decimate`
    with a *parametric* cutoff below the decimated Nyquist: zero-pad the
    complex input to ``stride * n_env``, keep the ``+-band_cyc`` band
    (positive and negative bins — the input is complex), inverse-FFT at
    length ``n_env`` directly. Circular edge handling; callers trim edges.

    ``band_cyc_rows`` (optional, ``(x.shape[-2],)``): a per-row cutoff for
    ``(..., rows, T)`` input — the ``band_mode="k_scaled"`` path, where each
    harmonic keeps its own band. ``None`` keeps the shared-cutoff behavior
    bit-identical.
    """
    from scipy import fft as sfft

    w = fft_workers()
    n_pad = stride * n_env
    xc = np.asarray(x, dtype=np.complex64)
    spec = cast(np.ndarray, sfft.fft(xc, n=n_pad, axis=-1, workers=w))
    low = np.zeros(x.shape[:-1] + (n_env,), dtype=np.complex64)
    if band_cyc_rows is None:
        b = min(int(np.floor(band_cyc * n_pad)), (n_env - 1) // 2)
        low[..., : b + 1] = spec[..., : b + 1]
        if b > 0:
            low[..., -b:] = spec[..., -b:]
    else:
        for a, bc in enumerate(band_cyc_rows):
            b = min(int(np.floor(float(bc) * n_pad)), (n_env - 1) // 2)
            low[..., a, : b + 1] = spec[..., a, : b + 1]
            if b > 0:
                low[..., a, -b:] = spec[..., a, -b:]
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
    band_cyc_k: np.ndarray | None = None,
    off_hz_k: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """On-comb and off-comb envelope banks for one rotor.

    Returns ``(z_on, z_off)``, each ``(C, K, n_env)`` complex128: the audio
    demodulated by ``k * phi`` (resp. ``k * phi + 2 pi off_hz t``),
    brickwall-lowpassed to ``+-band_cyc`` and decimated. Carriers come from
    the harmonic power recursion (one exp for the fundamental, complex64
    multiplies per harmonic step — ``vk_tracking._track_carriers``' trick);
    the off-comb carrier is the on-comb one times one shared ramp phasor,
    so the noise-floor probe costs no extra exp.

    ``band_cyc_k`` / ``off_hz_k`` (optional, ``(len(ks),)``): per-harmonic
    demod band (cycles/sample) and per-harmonic *signed* probe offset (Hz)
    — the ``band_mode="k_scaled"`` / ``probe_mode="clean"`` paths. The
    off-comb probe then demodulates in the SAME per-k band as the on-comb
    envelope (one ramp phasor per unique offset). ``None`` keeps the shared
    band / shared ramp behavior bit-identical.
    """
    n_ch, n_t = y32.shape
    n_k = len(ks)
    z_on = np.empty((n_ch, n_k, n_env), dtype=np.complex128)
    z_off = np.empty_like(z_on)
    c1 = np.exp(-1j * phi).astype(np.complex64)
    ramp = (
        None if off_hz_k is not None else np.exp(-2j * np.pi * off_hz * t_aud).astype(np.complex64)
    )
    ramps: dict[float, np.ndarray] = {}

    def get_ramp(off: float) -> np.ndarray:
        if off not in ramps:
            ramps[off] = np.exp(-2j * np.pi * off * t_aud).astype(np.complex64)
        return ramps[off]

    chunk = max(1, int(96e6 / (max(1, n_ch) * max(1, n_t) * 8)))
    buf = np.empty((n_ch, min(chunk, n_k), n_t), dtype=np.complex64)
    idxs: list[int] = []

    def flush() -> None:
        m = len(idxs)
        rows = None if band_cyc_k is None else band_cyc_k[idxs]
        z_on[:, idxs] = zoom_lp_decimate(buf[:, :m], stride, n_env, band_cyc, rows)
        if off_hz_k is None:
            assert ramp is not None
            buf[:, :m] *= ramp
        else:
            for a, g in enumerate(idxs):
                buf[:, a] *= get_ramp(float(off_hz_k[g]))
        z_off[:, idxs] = zoom_lp_decimate(buf[:, :m], stride, n_env, band_cyc, rows)
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
    r_ft: np.ndarray,
    i: int,
    k_top: int,
    sep_hz: float | np.ndarray,
    f_max: float,
    min_rate: float,
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

    ``sep_hz`` may be a ``(k_top,)`` array (the ``band_mode="k_scaled"``
    per-harmonic separation ``k B0 + guard``): in rev/s terms the collision
    condition becomes ``|dr| < B0 + guard / k`` — approximately
    k-independent, so the low harmonics of a twin pair with split above
    ``~B0`` un-mask instead of being collided up to ``sep / split``.
    """
    ks = np.arange(1, k_top + 1, dtype=np.float64)
    sep = np.asarray(sep_hz, dtype=np.float64)
    sep_col = sep[:, None] if sep.ndim == 1 else sep  # (K, 1) | scalar
    fi = ks[:, None] * r_ft[i][None, :]  # (K, N)
    coll = np.zeros(fi.shape, dtype=bool)
    for j in range(r_ft.shape[0]):
        if j == i or float(np.mean(r_ft[j])) < min_rate:
            continue
        rj = np.maximum(r_ft[j], 1e-3)[None, :]
        base = fi / rj
        for kp in (np.floor(base), np.ceil(base)):
            fj = np.maximum(kp, 1.0) * rj
            coll |= (np.abs(fj - fi) < sep_col) & (fj <= f_max + sep_col)
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


def _band_corrections_k(band_hz_k: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_band_corrections` per harmonic band: ``(c_noise, c_diff)``,
    each ``(K,)`` — the ``band_mode="k_scaled"`` variance corrections."""
    pairs = [_band_corrections(float(b), dt) for b in band_hz_k]
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def _clean_probe_offsets(
    r_ft: np.ndarray,
    i: int,
    ks: list[int],
    band_hz_k: np.ndarray,
    fallback_hz: np.ndarray,
    guard_hz: float,
    f_max: float,
    min_rate: float,
    search_lo: float = 8.0,
    search_hi: float = 15.0,
) -> tuple[np.ndarray, int]:
    """Collision-aware off-comb probe offsets for rotor ``i``.

    For each harmonic ``k`` the probe center rides at ``k r_i(t) + off``;
    ``off`` is searched over ``+-search_lo..+-search_hi`` Hz (0.5 Hz grid,
    signs interleaved, small magnitudes first, floored at
    ``band_hz_k + guard`` so the probe band always clears its own line) and
    accepted when at EVERY frame it stays ``band_hz_k + guard`` away from
    every harmonic of every tracked rotor (combs followed to
    ``f_max + 50 Hz``). No clean slot -> the fixed ``fallback_hz`` offset;
    returns ``(off_k signed (len(ks),), n_fallback)``.
    """
    rows = [
        np.maximum(r_ft[j], 1e-3)
        for j in range(r_ft.shape[0])
        if float(np.mean(r_ft[j])) >= min_rate
    ]
    off_k = np.empty(len(ks))
    n_fb = 0
    for a, k in enumerate(ks):
        bk = float(band_hz_k[a])
        clear = bk + guard_hz
        lo = max(search_lo, clear)
        hi = max(search_hi, lo + 4.0)
        fi = k * r_ft[i]  # (N,)
        chosen: float | None = None
        for mag in np.arange(lo, hi + 1e-9, 0.5):
            for off in (float(mag), -float(mag)):
                fp = fi + off
                ok = True
                for rj in rows:
                    base = fp / rj
                    for kp in (np.floor(base), np.ceil(base)):
                        fj = np.maximum(kp, 1.0) * rj
                        if bool(((np.abs(fj - fp) < clear) & (fj <= f_max + 50.0)).any()):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    chosen = off
                    break
            if chosen is not None:
                break
        if chosen is None:
            n_fb += 1
            chosen = float(fallback_hz[a])
        off_k[a] = chosen
    return off_k, n_fb


def _lowk_consistency_scale(
    dpsi: np.ndarray,
    r_var: np.ndarray,
    valid: np.ndarray,
    h: np.ndarray,
    kf: np.ndarray,
    ess: float | np.ndarray,
    split_k: int,
    thresh: float,
    weight: float,
    min_meas: int = 16,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Displacement-aware low-k admission (``lowk_gate="consistency"``).

    Per half-window, the information-weighted mean rate increment implied
    by the ``k < split_k`` measurements alone is compared against the
    ``k >= split_k`` one; disagreement beyond ``thresh`` (rev/s) multiplies
    the low group's fusion weights in that half by ``weight`` (a
    down-weight, not a hard drop — the DREGON displaced-comb fix). Returns
    ``(scale (K, n_m) | None, diag)``; ``None`` when the gate never fires.
    """
    n_m = dpsi.shape[-1]
    low = kf < float(split_k)
    diag: dict[str, Any] = {"split_k": split_k, "halves": []}
    if not low.any() or low.all():
        diag["skipped"] = "one group empty"
        return None, diag
    ess_b = ess[None, :, None] if isinstance(ess, np.ndarray) else ess
    w = np.where(valid, ess_b / np.maximum(r_var, _TINY), 0.0)
    hw = h[None, :, None]
    scale = np.ones((len(kf), n_m))
    fired_any = False
    for h0, h1 in ((0, n_m // 2), (n_m // 2, n_m)):
        sl = slice(h0, h1)
        est: dict[str, tuple[float, int] | None] = {}
        for gname, gmask in (("low", low), ("high", ~low)):
            wg = w[:, gmask, sl]
            info = float(np.sum(hw[:, gmask, :] ** 2 * wg))
            n_g = int(np.sum(valid[:, gmask, sl]))
            if info <= 0.0 or n_g < min_meas:
                est[gname] = None
                continue
            est[gname] = (float(np.sum(hw[:, gmask, :] * dpsi[:, gmask, sl] * wg)) / info, n_g)
        entry: dict[str, Any] = {"t_frames": [h0, h1]}
        e_lo, e_hi = est["low"], est["high"]
        if e_lo is None or e_hi is None:
            entry["skipped"] = "insufficient measurements"
        else:
            dis = abs(e_lo[0] - e_hi[0])
            fired = dis > thresh
            entry.update(
                {
                    "rate_low": round(e_lo[0], 4),
                    "rate_high": round(e_hi[0], 4),
                    "disagreement": round(dis, 4),
                    "n_low": e_lo[1],
                    "n_high": e_hi[1],
                    "fired": fired,
                }
            )
            if fired:
                scale[low, sl] = weight
                fired_any = True
        diag["halves"].append(entry)
    diag["fired"] = fired_any
    return (scale if fired_any else None), diag


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
    b0: float | None = None,
) -> tuple[dict[int, list[tuple[int, float, float, int]]], dict[str, Any]]:
    """Two-tone rate observations for one twin pair ``(lo, hi)``.

    For each self-collided harmonic ``k`` of the pair (and only those free
    of non-pair contamination inside the widened band), the audio is
    demodulated by the pair-mean phase ``k phibar`` in a band covering both
    lines, and per sliding window the two strongest spectral lines are
    picked (channel-incoherent power, parabolic sub-bin interpolation) and
    assigned to the rotors by order. Returns ``{rotor: [(frame_idx,
    delta_obs_rev_s, var, k), ...]}`` on the increment grid plus
    diagnostics. ``b0`` (rev/s, ``band_mode="k_scaled"``) replaces the
    fixed ``band_hz`` base band with the per-harmonic ``k b0`` (clamped to
    the envelope Nyquist) in the self-collision test and the two-tone band.

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
    fs_e_cap = 0.45 / dt
    rbar_aud = np.interp(t_aud, ft, rbar_ft)
    k_top = min(k_cap, int(np.floor(f_max / max(float(rbar_aud.max()), 1e-3))))
    others = [
        j for j in range(r.shape[0]) if j not in (lo, hi) and float(np.mean(r[j])) >= min_rate
    ]
    diag: dict[str, Any] = {"pair": [lo, hi], "split_track_med": round(d_med, 3)}
    obs: dict[int, list[tuple[int, float, float, int]]] = {lo: [], hi: []}

    # Effective split scale for the search geometry: robust (p75, not max —
    # transient excursions must not blow the band past the envelope Nyquist),
    # at least half and at most the full pairing radius — the track split
    # itself is unreliable (it is part of the init error being estimated).
    d_eff = min(max(float(np.percentile(d_abs, 75)), 0.5 * pair_max_split), pair_max_split)
    ks_joint: list[int] = []
    base_band: dict[int, float] = {}
    for k in range(1, k_top + 1):
        bb = band_hz if b0 is None else min(k * b0, fs_e_cap)
        base_band[k] = bb
        if not bool((k * d_abs < bb + guard_hz).any()):
            continue  # pair never self-collides at this harmonic
        bw_k = 0.5 * k * d_eff + bb
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
        band_k_hz = 0.5 * k * d_eff + base_band[k]
        phasor = np.exp(-1j * (k * phibar)).astype(np.complex64)
        z = zoom_lp_decimate(y32 * phasor[None, :], stride, n_env, band_k_hz / sr)  # (C, n_env)
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
            obs[lo].append((idx, y_lo, var, k))
            obs[hi].append((idx, y_hi, var, k))
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
    ess: float | np.ndarray = 1.0,
    extra: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RTS-smoothed ``dr`` from gated increments (state scalar per frame).

    ``dpsi`` / ``r_var`` / ``valid``: ``(C, K, n-1)``; ``h``: ``(K,)`` the
    per-harmonic observation gain ``2 pi k dt``. All valid measurements of
    one frame are folded into the information pair before the scan. ``ess``
    scales the information: successive increments are correlated by the
    demod lowpass (only ``~2 band_hz dt`` independent increments per frame
    when the band is oversampled), and without the effective-sample-size
    deflation the posterior overcounts the band and chases in-band noise;
    a ``(K,)`` array gives the per-harmonic value of the k-scaled bands.
    ``extra``: optional ``(idx, y, var)`` direct-rate observations
    (``H = 1``, rev/s — the joint pair-mode two-tone measurements), folded
    into the same information pair at their frame indices (no ess: the
    windows hop by half their length and are near-independent). ``scale``:
    optional ``(K, n-1)`` extra information multiplier (the low-k
    consistency down-weight).
    """
    ess_b = ess[None, :, None] if isinstance(ess, np.ndarray) else ess
    w = np.where(valid, ess_b / np.maximum(r_var, _TINY), 0.0)
    if scale is not None:
        w = w * scale[None, :, :]
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
    extra_obs: list[tuple[int, float, float, int]] | None = None,
    b0: float | None = None,
    probe_mode: str = "fixed",
    lowk_gate: str = "none",
    lowk_split_k: int = 16,
    lowk_thresh: float = 0.15,
    lowk_weight: float = 0.1,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """One outer iteration for rotor ``i``: ``(delta on ft grid | None, diag)``.

    ``b0`` (rev/s per harmonic order) switches the pass to per-harmonic
    ``k b0`` demod bands (``band_mode="k_scaled"``, clamped to the envelope
    Nyquist) with per-k variance corrections / ess / twin separation; the
    remaining keywords wire ``probe_mode`` and ``lowk_gate``.
    """
    r_aud = np.interp(t_aud, ft, r[i])
    mean_rate = float(np.mean(r_aud))
    d: dict[str, Any] = {"k_cap": int(k_cap), "mean_rate": round(mean_rate, 3)}
    extra = None
    extra_k = None
    if extra_obs:
        arr = np.asarray(extra_obs, dtype=np.float64)
        extra = (arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2].copy())
        extra_k = arr[:, 3].astype(np.int64)
        d["n_joint_obs"] = len(extra_obs)
    if mean_rate < min_rate:
        d["skipped"] = f"mean rate {mean_rate:.1f} < min_rate {min_rate}"
        return None, d
    k_top = min(k_cap, int(np.floor(f_max / max(float(r_aud.max()), 1e-3))))
    if k_top < 1:
        d["skipped"] = "no harmonic below f_max"
        return None, d
    fs_e = 1.0 / dt
    if b0 is None:
        sep_full: float | np.ndarray = band_hz + guard_hz
    else:
        kf_full = np.arange(1, k_top + 1, dtype=np.float64)
        band_full = np.minimum(kf_full * b0, 0.45 * fs_e)
        n_clamped = int((kf_full * b0 > 0.45 * fs_e).sum())
        if n_clamped:
            d["n_band_clamped"] = n_clamped
        sep_full = band_full + guard_hz
    coll = _twin_collision_mask(r, i, k_top, sep_full, f_max, min_rate)
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
        jmask = np.zeros(n_m, dtype=bool)
        jmask[n_trim : max(n_trim, n_m - n_trim)] = True
        p_int = p1[jmask] if jmask.any() else p1
        d["post_std_max"] = round(float(np.max(np.sqrt(p_int))), 4)
        return np.interp(ft, t_mid, m1), d

    phi = 2.0 * np.pi * np.cumsum(r_aud) / sr
    ka = np.asarray(ks)
    band_cyc_k: np.ndarray | None = None
    band_k_hz: np.ndarray | None = None
    if b0 is not None:
        band_k_hz = np.minimum(ka * b0, 0.45 * fs_e)
        band_cyc_k = band_k_hz / sr
        d["band_hz_k"] = {str(k): round(float(bh), 2) for k, bh in zip(ks, band_k_hz)}
    off_hz_k: np.ndarray | None = None
    if b0 is not None or probe_mode == "clean":
        base_bands = band_k_hz if band_k_hz is not None else np.full(len(ks), band_hz)
        fallback = np.maximum(off_comb_hz, base_bands + guard_hz)
        if probe_mode == "clean":
            off_hz_k, n_fb = _clean_probe_offsets(
                r, i, ks, base_bands, fallback, guard_hz, f_max, min_rate
            )
            d["probe_fallbacks"] = n_fb
            d["probe_off_k"] = {str(k): round(float(o), 1) for k, o in zip(ks, off_hz_k)}
        elif bool((fallback > off_comb_hz).any()):
            off_hz_k = fallback  # k-scaled fixed probe: forced band clearance
    z, z_off = _demod_bank(
        y32, phi, t_aud, ks, off_comb_hz, stride, n_env, band_cyc, band_cyc_k, off_hz_k
    )
    interior = slice(n_trim, max(n_trim + 1, n_env - n_trim))
    noise_pow = np.maximum(np.mean(np.abs(z_off[..., interior]) ** 2, axis=-1), _TINY)  # (C, K)

    a2 = np.abs(z) ** 2  # (C, K, n_env)
    dpsi = np.angle(z[..., 1:] * np.conj(z[..., :-1]))  # (C, K, n_env - 1)
    inv0 = 1.0 / np.maximum(a2[..., :-1], _TINY)
    inv1 = 1.0 / np.maximum(a2[..., 1:], _TINY)
    if b0 is None:
        cn: float | np.ndarray
        cd: float | np.ndarray
        cn, cd = _band_corrections(band_hz, dt)
    else:
        assert band_k_hz is not None
        cn_k, cd = _band_corrections_k(band_k_hz, dt)
        cn = cn_k[None, :, None]
    var_meas = cn * 0.5 * noise_pow[..., None] * (inv0 + inv1)
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
    # Independent increments per frame (per-harmonic for the k-scaled bands).
    ess: float | np.ndarray
    if b0 is None:
        ess = min(1.0, 2.0 * band_hz * dt)
    else:
        assert band_k_hz is not None
        ess = np.minimum(1.0, 2.0 * band_k_hz * dt)
    scale = None
    if lowk_gate == "consistency":
        # The comparison needs high-k evidence; on the coarse iterations
        # (k_cap below the split) demodulate a small high-k PROBE group so
        # the displaced low set is checked BEFORE it can move the track
        # (the un-probed gate only sees the displacement after capture,
        # when it is invisible). Probe rows are fused only when the gate
        # fires (below); otherwise they are comparison-only.
        g_dpsi, g_var, g_valid, g_kf, g_ess = dpsi, var_meas, valid, kf, ess
        probe_ks: list[int] = []
        if int((kf >= lowk_split_k).sum()) < 4:
            hi_top = min(int(np.floor(f_max / max(float(r_aud.max()), 1e-3))), lowk_split_k + 7)
            if hi_top >= lowk_split_k:
                if b0 is None:
                    sep_hi: float | np.ndarray = band_hz + guard_hz
                    band_hi_of = None
                else:
                    khi_full = np.arange(1, hi_top + 1, dtype=np.float64)
                    band_hi_of = np.minimum(khi_full * b0, 0.45 * fs_e)
                    sep_hi = band_hi_of + guard_hz
                coll_hi = _twin_collision_mask(r, i, hi_top, sep_hi, f_max, min_rate)
                probe_ks = [
                    k
                    for k in range(lowk_split_k, hi_top + 1)
                    if k not in ks and not coll_hi[k - 1].all()
                ]
                if probe_ks:
                    pk = np.asarray(probe_ks)
                    if b0 is None:
                        p_cyc_k = None
                        p_band = None
                        p_cn: float | np.ndarray = cn if not isinstance(cn, np.ndarray) else 1.0
                        p_cd: float | np.ndarray = cd if not isinstance(cd, np.ndarray) else 1.0
                        p_ess: float | np.ndarray = ess if not isinstance(ess, np.ndarray) else 1.0
                    else:
                        assert band_hi_of is not None
                        p_band = band_hi_of[pk - 1]
                        p_cyc_k = p_band / sr
                        p_cn_k, p_cd = _band_corrections_k(p_band, dt)
                        p_cn = p_cn_k[None, :, None]
                        p_ess = np.minimum(1.0, 2.0 * p_band * dt)
                    zp, zp_off = _demod_bank(
                        y32,
                        phi,
                        t_aud,
                        probe_ks,
                        off_comb_hz,
                        stride,
                        n_env,
                        band_cyc,
                        p_cyc_k,
                        None,
                    )
                    np_pow = np.maximum(np.mean(np.abs(zp_off[..., interior]) ** 2, axis=-1), _TINY)
                    p_a2 = np.abs(zp) ** 2
                    p_dpsi = np.angle(zp[..., 1:] * np.conj(zp[..., :-1]))
                    p_var = (
                        p_cn
                        * 0.5
                        * np_pow[..., None]
                        * (
                            1.0 / np.maximum(p_a2[..., :-1], _TINY)
                            + 1.0 / np.maximum(p_a2[..., 1:], _TINY)
                        )
                    )
                    p_valid = (p_a2[..., 1:] > snr_gate * np_pow[..., None]) & (
                        p_a2[..., :-1] > snr_gate * np_pow[..., None]
                    )
                    p_valid &= np.abs(p_dpsi) < wrap_guard_rad
                    p_valid &= tmask[None, None, :]
                    p_coll_mid = np.stack(
                        [np.interp(t_mid, ft, coll_hi[k - 1].astype(np.float64)) for k in probe_ks]
                    )
                    p_valid &= p_coll_mid[None, :, :] < 0.5
                    g_dpsi = np.concatenate([dpsi, p_dpsi], axis=1)
                    g_var = np.concatenate([var_meas, p_var], axis=1)
                    g_valid = np.concatenate([valid, p_valid], axis=1)
                    g_kf = np.concatenate([kf, pk.astype(np.float64)])
                    if isinstance(ess, np.ndarray) or isinstance(p_ess, np.ndarray):
                        e_main = ess if isinstance(ess, np.ndarray) else np.full(len(kf), ess)
                        e_probe = (
                            p_ess if isinstance(p_ess, np.ndarray) else np.full(len(pk), p_ess)
                        )
                        g_ess = np.concatenate([e_main, e_probe])
                    else:
                        g_ess = ess
        g_h = 2.0 * np.pi * g_kf * dt
        scale_full, lowk_diag = _lowk_consistency_scale(
            g_dpsi, g_var, g_valid, g_h, g_kf, g_ess, lowk_split_k, lowk_thresh, lowk_weight
        )
        if scale_full is not None and probe_ks:
            # Gate fired on a low-k-only iteration: the down-weight alone
            # cannot stop capture when the low group is the only fused
            # evidence, so FUSE the (coherent — the gate measured them)
            # high-k probe rows too: the arm-A high-k anchor, engaged only
            # when the displacement discriminator trips (no-op elsewhere).
            dpsi, var_meas, valid = g_dpsi, g_var, g_valid
            kf, h, ess = g_kf, g_h, g_ess
            if isinstance(cd, np.ndarray) or isinstance(p_cd, np.ndarray):
                cd_main = cd if isinstance(cd, np.ndarray) else np.full(len(ks), cd)
                cd_probe = p_cd if isinstance(p_cd, np.ndarray) else np.full(len(pk), p_cd)
                cd = np.concatenate([cd_main, cd_probe])
            noise_pow = np.concatenate([noise_pow, np_pow], axis=1)
            coll_mid = np.concatenate([coll_mid, p_coll_mid], axis=0)
            ks = ks + probe_ks
            scale = scale_full
            lowk_diag["fused_probe"] = True
        else:
            scale = None if scale_full is None else scale_full[: len(kf)]
        if probe_ks:
            lowk_diag["probe_ks"] = probe_ks
        d["lowk"] = lowk_diag
        if extra is not None and extra_k is not None:
            for hd in lowk_diag.get("halves", []):
                if hd.get("fired"):
                    lo_f, hi_f = hd["t_frames"]
                    m = (extra_k < lowk_split_k) & (extra[0] >= lo_f) & (extra[0] < hi_f)
                    extra[2][m] /= lowk_weight  # down-weight joint low-k obs too
    # Pass A (q_k = 0) -> data-driven q_k from the robust residual excess.
    m0, _ = _smooth_delta(dpsi, var_meas, valid, h, q_step, p0, ess=ess, extra=extra, scale=scale)
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
        cd_a = float(cd[a]) if isinstance(cd, np.ndarray) else cd
        q_k[a] = max(0.0, excess) / (cd_a * dt)
        q_ok[a] = True
    # Pass B: diffusion-aware measurement variances (the in-band effective
    # diffusion contribution per increment is c_diff * q_k * dt = the excess).
    m1, p1 = _smooth_delta(
        dpsi,
        var_meas + (q_k * cd * dt)[None, :, None],
        valid,
        h,
        q_step,
        p0,
        ess=ess,
        extra=extra,
        scale=scale,
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
    # Interior only: the edge-trimmed frames sit at the prior and would
    # pin the posterior-annealed band to its ceiling.
    p_int = p1[tmask] if tmask.any() else p1
    d["post_std_max"] = round(float(np.max(np.sqrt(p_int))), 4)
    d["n_meas_k"] = {str(k): int(valid[:, a, :].sum()) for a, k in enumerate(ks)}
    tg = {
        str(k): round(float(np.mean(coll_mid[a] >= 0.5)), 3)
        for a, k in enumerate(ks)
        if bool((coll_mid[a] >= 0.5).any())
    }
    if tg:
        d["twin_gated_frac_k"] = tg
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
    band_hz: float | tuple[float, ...] = 6.0,
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
    band_mode: str = "fixed",
    band_b0: float | tuple[float, ...] = 0.35,
    band_anneal: str = "none",
    anneal_c: float = 3.0,
    anneal_w_line: float = 0.08,
    anneal_b0_floor: float = 0.12,
    lowk_gate: str = "none",
    lowk_split_k: int = 16,
    lowk_thresh: float = 0.15,
    lowk_weight: float = 0.1,
    probe_mode: str = "fixed",
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
            around each harmonic. A tuple gives a per-iteration schedule
            (last entry repeats), wide -> narrow: wide bands admit large
            initial detunings (capture), the narrow final band excludes
            near-line contamination — twin lines and the low-side
            modulation sidebands real flight audio carries at a ~constant
            -1..-3 Hz offset from every harmonic, which a band-mean
            estimator would otherwise integrate as a coherent downward
            bias (the S4 free-flight finding).
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
        band_mode: ``"fixed"`` (default) keeps one ``band_hz`` demod band for
            every harmonic; ``"k_scaled"`` gives harmonic ``k`` the band
            ``k * band_b0`` Hz (a shaft-rate trust region: a rate error
            ``dr`` displaces harmonic ``k`` by ``k dr`` Hz, so a fixed band
            over-admits at low k and under-admits at high k). All variance
            corrections (``c_noise``/``c_diff``/``ess``), the off-comb noise
            probe band, and the twin separation become per-k; bands are
            clamped to the envelope Nyquist (``0.45 fs_env`` ~ 28 Hz at the
            62.5 Hz default — no clamp below k = 80 at ``band_b0 = 0.35``).
            ``band_hz`` is then ignored for demodulation.
        band_b0: k-scaled band scale (rev/s per harmonic order); scalar or
            per-rotor tuple. Default 0.35 — just under the minimum twin
            pair split, so twin low harmonics un-mask (collision requires
            ``|dr| < band_b0 + guard_hz / k``).
        band_anneal: ``"posterior"`` (requires ``k_scaled``) shrinks each
            rotor's ``band_b0`` after every outer iteration to
            ``max(anneal_c * sqrt(max_t p_s) + anneal_w_line,
            anneal_b0_floor)`` — the smoother's own posterior as a trust
            region (capped at the initial ``band_b0``). The final per-rotor
            values are returned as ``diagnostics["band_b0_final"]`` so
            callers can thread them across repeated applications.
        anneal_c: posterior-std multiplier of the anneal (3x: the posterior
            understates the true residual under model mismatch).
        anneal_w_line: additive linewidth margin (rev/s) — the measured
            shaft-jitter linewidth the band must keep even at zero residual
            (q_k budgets put it at ~0.05-0.1 rev/s; 0.08 is the midpoint).
        anneal_b0_floor: lower bound of the annealed ``band_b0`` (rev/s).
        lowk_gate: ``"consistency"`` compares, per half-window, the
            information-weighted mean rate increment implied by the
            ``k < lowk_split_k`` measurements against the high-k group
            (when the iteration's cap admits too few high harmonics, a
            small high-k probe group ``[split_k, split_k + 7]`` is
            demodulated for the comparison); on disagreement beyond
            ``lowk_thresh`` the low group's weights are multiplied by
            ``lowk_weight`` AND the probe rows are fused as measurements —
            the arm-A high-k anchor, engaged only when the displacement
            discriminator trips (the DREGON displaced-comb admission fix —
            bit-exact no-op where low and high k agree).
        lowk_split_k: the low/high group boundary (DREGON displacement
            lives at k <= 13; 16 leaves margin).
        lowk_thresh: disagreement threshold (rev/s). Default 0.15 = half
            the smallest measured DREGON displacement (0.3-0.5 rev/s), well
            above the per-group estimator noise on cruise windows.
        lowk_weight: multiplicative down-weight of the low-k group.
        probe_mode: ``"clean"`` searches each harmonic's off-comb noise
            probe offset over ``+-8..15`` Hz for a slot whose probe band
            clears every harmonic of every tracked rotor at every frame
            (fallback: the fixed offset; count in
            ``diagnostics["probe_fallbacks"]``). ``"fixed"`` keeps the
            shared ``off_comb_hz`` ramp.

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
    bands = tuple(float(b) for b in (band_hz if isinstance(band_hz, tuple) else (band_hz,)))
    if band_mode not in ("fixed", "k_scaled"):
        raise ValueError(f"unknown band_mode {band_mode!r} (expected 'fixed' or 'k_scaled')")
    if band_anneal not in ("none", "posterior"):
        raise ValueError(f"unknown band_anneal {band_anneal!r} (expected 'none' or 'posterior')")
    if band_anneal == "posterior" and band_mode != "k_scaled":
        raise ValueError("band_anneal='posterior' requires band_mode='k_scaled'")
    if lowk_gate not in ("none", "consistency"):
        raise ValueError(f"unknown lowk_gate {lowk_gate!r} (expected 'none' or 'consistency')")
    if probe_mode not in ("fixed", "clean"):
        raise ValueError(f"unknown probe_mode {probe_mode!r} (expected 'fixed' or 'clean')")
    if band_mode == "fixed" and off_comb_hz <= max(bands):
        raise ValueError(f"off_comb_hz={off_comb_hz} must exceed band_hz={max(bands)}")
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
    y32 = x.astype(np.float32)
    n_trim = max(1, int(round(edge_trim_s * fs_e)))
    q_step = sigma_process**2 * dt
    p0 = sigma_prior**2
    schedule = [int(min(k_caps[min(j, len(k_caps) - 1)], k_max)) for j in range(n_iter)]
    band_schedule = [bands[min(j, len(bands) - 1)] for j in range(n_iter)]
    n_rot = r.shape[0]
    k_scaled = band_mode == "k_scaled"
    b0_arr = np.asarray(band_b0, dtype=np.float64)
    if b0_arr.ndim == 0:
        b0_arr = np.full(n_rot, float(b0_arr))
    elif b0_arr.shape != (n_rot,):
        raise ValueError(f"band_b0 must be scalar or per-rotor ({n_rot},), got {b0_arr.shape}")
    b0_init = b0_arr.copy()

    rotor_diags: list[dict[str, Any]] = [{"rotor": i, "iters": []} for i in range(n_rot)]
    pair_diags: list[list[dict[str, Any]]] = []
    for it, k_cap in enumerate(schedule):
        band_it = band_schedule[it]
        joint_obs: dict[int, list[tuple[int, float, float, int]]] = {}
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
                    band_hz=band_it,
                    f_max=f_max,
                    guard_hz=guard_hz,
                    min_rate=min_rate,
                    joint_win_s=joint_win_s,
                    joint_snr_min=joint_snr_min,
                    n_trim=n_trim,
                    pair_max_split=pair_max_split,
                    b0=float(0.5 * (b0_arr[lo] + b0_arr[hi])) if k_scaled else None,
                )
                pd["iter"] = it + 1
                it_pair_diags.append(pd)
                for rot, lst in obs.items():
                    joint_obs.setdefault(rot, []).extend(lst)
            pair_diags.append(it_pair_diags)
        for i in range(n_rot):
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
                band_it / sr,
                k_cap,
                band_hz=band_it,
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
                b0=float(b0_arr[i]) if k_scaled else None,
                probe_mode=probe_mode,
                lowk_gate=lowk_gate,
                lowk_split_k=lowk_split_k,
                lowk_thresh=lowk_thresh,
                lowk_weight=lowk_weight,
            )
            d["iter"] = it + 1
            if k_scaled:
                d["band_b0"] = round(float(b0_arr[i]), 4)
            if delta_ft is not None:
                step = np.clip(delta_ft, -max_step, max_step)
                r[i] += step
                d["step_rms"] = round(float(np.sqrt(np.mean(step**2))), 4)
                d["step_max"] = round(float(np.max(np.abs(step))), 4)
                if band_anneal == "posterior" and "post_std_max" in d:
                    # Trust region: next iteration's per-k band scale from the
                    # smoother's own posterior (never wider than the initial
                    # capture band, floored to keep capture sane).
                    b0_new = max(anneal_c * d["post_std_max"] + anneal_w_line, anneal_b0_floor)
                    b0_arr[i] = min(b0_new, float(b0_init[i]))
                    d["band_b0_next"] = round(float(b0_arr[i]), 4)
            rotor_diags[i]["iters"].append(d)

    diagnostics: dict[str, Any] = {
        "params": {
            "n_iter": n_iter,
            "fs_env": fs_env,
            "band_hz": list(bands),
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
            "band_mode": band_mode,
            "band_b0": list(band_b0) if isinstance(band_b0, tuple) else band_b0,
            "band_anneal": band_anneal,
            "anneal_c": anneal_c,
            "anneal_w_line": anneal_w_line,
            "anneal_b0_floor": anneal_b0_floor,
            "lowk_gate": lowk_gate,
            "lowk_split_k": lowk_split_k,
            "lowk_thresh": lowk_thresh,
            "lowk_weight": lowk_weight,
            "probe_mode": probe_mode,
        },
        "fs_env_actual": fs_e,
        "k_schedule": schedule,
        "rotors": rotor_diags,
    }
    if k_scaled:
        diagnostics["band_b0_final"] = [round(float(v), 4) for v in b0_arr]
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
