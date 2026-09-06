"""Iterated time-warp (generalized-demodulation) refinement of rotor IF tracks.

First method of the VK-tracking-improvement program. Given multichannel audio
``(C, T)`` and an approximate per-rotor IF trajectory (rev/s on a uniform
frame grid, error up to ~1-2 rev/s), each rotor is refined independently by
iterating, coarse-to-fine in harmonic order:

1. Shaft-phase estimate ``phi_hat(t) = 2 pi cumsum(r_hat) / sr``.
2. **Angular resampling**: interpolate the audio onto a uniform grid in
   ``phi_hat`` (constant ``S`` samples per revolution). In the warped domain
   this rotor's harmonics become quasi-stationary tones at exact integer
   *orders* (cycles/rev); a residual trajectory error ``delta_r(t)`` displaces
   order ``k`` by ``k * delta_r / r`` cycles/rev.
3. **Residual order-error estimation**: per long window (length given in
   seconds, converted to revolutions at the current mean rate), FFT the warped
   audio and, for each order ``k`` in the current rung's set, read the peak
   offset ``eps_k`` (cycles/rev) inside the search band ``k * delta_max / r``
   (parabolic sub-bin interpolation on log power); the implied shaft error is
   ``delta_k = r * eps_k / k`` rev/s. Channels are combined *incoherently*
   (per-channel power spectra summed — mic phases differ, peak positions
   agree). Orders whose search band collides with another rotor's predicted
   orders are excluded (twin rejection); surviving estimates are fused across
   orders with Fisher-style weights ``k^2 * (SNR - 1)`` (frequency precision
   grows with harmonic index and line SNR), gated at ``snr_min``.
4. **Correction**: the per-window fused ``delta`` is interpolated back to real
   time on the frame grid, lightly smoothed (~0.25 s), clipped to the
   per-round step (``max_step``), and added to the track.

The rung schedule is an ambiguity ladder: each round keeps
``k_hi * delta_max / r < 0.5`` cycles/rev (at the ~75-90 rev/s regime), so a
peak found inside the search band can never be a neighbouring order of the
same comb, and each round's correction shrinks the residual enough that the
next round's higher orders stay unambiguous (``|k * delta_phi| < pi``).

Numerics: audio is polyphase-FIR upsampled (``oversample``) before the linear
angular interpolation so high-order tones survive the resampling; ``S`` is
chosen so the warped Nyquist (``S/2`` orders) sits at or above the real-audio
Nyquist — no real content aliases onto integer orders. Pure numpy/scipy, CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from tracking.dsp import boxcar

__all__ = ["DEFAULT_RUNGS", "WarpRung", "iter_warp_refine"]

_TINY = 1e-30


@dataclass(frozen=True)
class WarpRung:
    """One coarse-to-fine round: order set, window length, search half-range."""

    k_lo: int
    k_hi: int
    window_s: float  # long-window length (seconds, converted to revolutions)
    delta_max: float  # rev/s search half-range around the current track


#: Default ambiguity ladder (see module docstring). Residual error from a
#: +2 rev/s init shrinks by <= ``max_step`` per round: 2.0 -> 1.5 -> 1.0 ->
#: 0.5 -> 0, and every rung's ``delta_max`` covers the worst-case residual it
#: can face while keeping ``k_hi * delta_max / r < 0.5`` cycles/rev.
DEFAULT_RUNGS: tuple[WarpRung, ...] = (
    WarpRung(k_lo=1, k_hi=6, window_s=3.0, delta_max=2.5),
    WarpRung(k_lo=4, k_hi=12, window_s=1.5, delta_max=1.8),
    WarpRung(k_lo=8, k_hi=24, window_s=0.75, delta_max=1.2),
    WarpRung(k_lo=12, k_hi=40, window_s=0.375, delta_max=0.6),
)


def _order_collides(
    k: int, r_i: float, r_others: Sequence[float], band: float, guard: float, f_max: float
) -> bool:
    """True iff another rotor's predicted order falls inside this order's band.

    In rotor ``i``'s warped axis, rotor ``j``'s order ``k'`` sits at
    ``k' * r_j / r_i`` cycles/rev; only the ``k'`` nearest ``k * r_i / r_j``
    can enter the ``band + guard`` neighbourhood of ``k``.
    """
    for ro in r_others:
        if ro <= 1e-3:
            continue
        ratio = ro / r_i
        base = k / ratio
        for kp in {int(np.floor(base)), int(np.ceil(base))}:
            if kp < 1 or kp * ro > f_max:
                continue
            if abs(kp * ratio - k) < band + guard:
                return True
    return False


def _peak_in_band(
    power: np.ndarray, k: int, band: float, bin_per_cycle: float, pad_factor: int
) -> tuple[float, float] | None:
    """Strongest line near order ``k``: ``(f_peak cycles/rev, power SNR)``.

    Parabolic sub-bin interpolation on log power; the noise floor is the
    median over a 3x band with the peak's immediate bins excluded.
    """
    center = k * bin_per_cycle
    hb = max(int(np.ceil(band * bin_per_cycle)), 3)
    lo = max(int(np.floor(center)) - hb, 1)
    hi = min(int(np.ceil(center)) + hb, len(power) - 2)
    if hi <= lo:
        return None
    p = lo + int(np.argmax(power[lo : hi + 1]))
    y0, y1, y2 = (float(np.log(power[q] + _TINY)) for q in (p - 1, p, p + 1))
    denom = y0 - 2.0 * y1 + y2
    off = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    off = float(np.clip(off, -0.5, 0.5))
    f_peak = (p + off) / bin_per_cycle

    flo = max(int(np.floor(center - 3 * hb)), 1)
    fhi = min(int(np.ceil(center + 3 * hb)), len(power) - 2)
    idx = np.arange(flo, fhi + 1)
    mask = np.abs(idx - p) > 2 * pad_factor
    if not np.any(mask):
        return None
    floor = float(np.median(power[idx[mask]]))
    snr = float(power[p] / max(floor, _TINY, 1e-12 * float(power[p])))
    return f_peak, snr


def _refine_round(
    x_hi: np.ndarray,
    t_hi: np.ndarray,
    t_aud: np.ndarray,
    r: np.ndarray,
    i: int,
    ft: np.ndarray,
    rung: WarpRung,
    sr: int,
    *,
    f_max: float,
    snr_min: float,
    snr_cap: float,
    guard_cycles: float,
    pad_factor: int,
    min_rate: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """One warp round for rotor ``i``: ``(delta on ft grid | None, diagnostics)``."""
    r_aud = np.interp(t_aud, ft, r[i])
    mean_rate = float(np.mean(r_aud))
    rd: dict[str, Any] = {
        "k_lo": rung.k_lo,
        "k_hi": rung.k_hi,
        "window_s": rung.window_s,
        "delta_max": rung.delta_max,
        "mean_rate": round(mean_rate, 3),
    }
    if mean_rate < min_rate:
        rd["skipped"] = f"mean rate {mean_rate:.1f} < min_rate {min_rate}"
        return None, rd

    # Angular resampling grid: S samples/rev with the warped Nyquist (S/2
    # orders = S/2 * r Hz) at or above the audio Nyquist -> alias-free orders.
    phi_rev = np.cumsum(np.clip(r_aud, 1e-2, None)) / sr
    phi_rev -= phi_rev[0]
    r_lo = max(float(np.percentile(r_aud, 5)), min_rate)
    s_rev = 1 << int(np.ceil(np.log2(max(sr / r_lo, 2.0 * rung.k_hi + 16.0))))
    m_total = int(phi_rev[-1] * s_rev)
    theta = np.arange(m_total) / s_rev
    t_m = np.interp(theta, phi_rev, t_aud)
    y = np.stack([np.interp(t_m, t_hi, ch) for ch in x_hi])

    win_len = int(round(rung.window_s * mean_rate * s_rev))
    win_len = min(max(win_len, 4 * s_rev), m_total)
    if win_len < 2 * s_rev:
        rd["skipped"] = "clip shorter than two revolutions"
        return None, rd
    hop = max(win_len // 2, 1)
    starts = list(range(0, m_total - win_len + 1, hop))
    if starts and starts[-1] + win_len < m_total:
        starts.append(m_total - win_len)
    window = np.hanning(win_len)
    nfft = pad_factor * (1 << int(np.ceil(np.log2(win_len))))
    bin_per_cycle = nfft / s_rev
    ks = list(range(rung.k_lo, rung.k_hi + 1))
    stats: dict[int, dict[str, Any]] = {
        k: {"snr": [], "delta": [], "n_excluded": 0, "n_low_snr": 0} for k in ks
    }
    others = [j for j in range(r.shape[0]) if j != i]

    t_centers: list[float] = []
    deltas: list[float] = []
    for a in starts:
        seg = y[:, a : a + win_len] * window
        power = (np.abs(np.fft.rfft(seg, n=nfft, axis=-1)) ** 2).sum(axis=0)
        t_a, t_b = float(t_m[a]), float(t_m[a + win_len - 1])
        if t_b - t_a <= 0:
            continue
        r_w = (win_len - 1) / s_rev / (t_b - t_a)  # exact mean rate over window
        t_c = 0.5 * (t_a + t_b)
        r_others = [float(np.interp(t_c, ft, r[j])) for j in others]
        guard = guard_cycles + 2.0 * s_rev / win_len  # + two pre-padding bins
        num = den = 0.0
        for k in ks:
            st = stats[k]
            if k * r_w > f_max:
                continue
            band = min(rung.delta_max * k / r_w, 0.45)
            if _order_collides(k, r_w, r_others, band, guard, f_max):
                st["n_excluded"] += 1
                continue
            est = _peak_in_band(power, k, band, bin_per_cycle, pad_factor)
            if est is None:
                continue
            f_peak, snr = est
            if snr < snr_min:
                st["n_low_snr"] += 1
                continue
            snr = min(snr, snr_cap)
            delta_k = r_w * (f_peak - k) / k
            st["snr"].append(snr)
            st["delta"].append(delta_k)
            w = float(k * k) * (snr - 1.0)
            num += w * delta_k
            den += w
        if den > 0.0:
            t_centers.append(t_c)
            deltas.append(num / den)

    rd["samples_per_rev"] = s_rev
    rd["n_windows"] = len(starts)
    rd["n_windows_locked"] = len(t_centers)
    rd["orders"] = [
        {
            "k": k,
            "n_locked": len(st["snr"]),
            "n_excluded": st["n_excluded"],
            "n_low_snr": st["n_low_snr"],
            "snr_med": round(float(np.median(st["snr"])), 2) if st["snr"] else 0.0,
            "delta_med": round(float(np.median(st["delta"])), 4) if st["delta"] else None,
        }
        for k, st in stats.items()
    ]
    if not t_centers:
        return None, rd
    return np.interp(ft, np.asarray(t_centers), np.asarray(deltas)), rd


def iter_warp_refine(
    audio: np.ndarray,
    r_init: np.ndarray,
    ft: np.ndarray,
    sr: int = 16000,
    *,
    rounds: int = 4,
    rungs: Sequence[WarpRung] = DEFAULT_RUNGS,
    max_step: float = 0.5,
    smooth_s: float = 0.25,
    f_max: float = 6000.0,
    snr_min: float = 3.0,
    snr_cap: float = 1e3,
    guard_cycles: float = 0.03,
    pad_factor: int = 2,
    oversample: int = 2,
    min_rate: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Iterated angular-resampling refinement of per-rotor IF tracks.

    Args:
        audio: ``(C, T)`` or ``(T,)`` audio at ``sr``.
        r_init: ``(R, N)`` initial IF tracks, rev/s on the frame grid ``ft``.
        ft: ``(N,)`` uniform frame times (seconds, audio-relative).
        sr: audio sample rate.
        rounds: number of warp rounds; beyond ``len(rungs)`` the last rung
            repeats (extra fine polishing rounds).
        rungs: coarse-to-fine order/window/search schedule.
        max_step: per-round correction clip (rev/s).
        smooth_s: light smoothing of each round's correction (seconds).
        f_max: highest harmonic frequency used (Hz).
        snr_min: per-order power-SNR gate (below -> order ignored).
        snr_cap: weight cap on the power SNR (clean signals otherwise let one
            order dominate the fusion).
        guard_cycles: extra twin-rejection guard band (cycles/rev) on top of
            two FFT bins.
        pad_factor: FFT zero-padding factor for sub-bin peak interpolation.
        oversample: polyphase upsampling factor applied to the audio before
            the linear angular interpolation.
        min_rate: rotors whose mean rate is below this (rev/s) are skipped.

    Returns:
        ``(r_refined, diagnostics)`` — the refined ``(R, N)`` tracks and a
        JSON-serializable dict with per-rotor per-round per-order lock
        quality (windows locked/excluded/low-SNR, median SNR, median delta).
    """
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    r = np.array(r_init, dtype=np.float64, copy=True)
    if r.ndim != 2:
        raise ValueError(f"r_init must be (R, N), got shape {r.shape}")
    ft = np.asarray(ft, dtype=np.float64)
    t_aud = np.arange(x.shape[-1]) / sr
    x_hi = resample_poly(x, oversample, 1, axis=-1) if oversample > 1 else x
    t_hi = np.arange(x_hi.shape[-1]) / (sr * oversample)
    dt_ft = float(ft[1] - ft[0]) if len(ft) > 1 else 0.032
    smooth_n = max(1, int(round(smooth_s / dt_ft)))
    schedule = [rungs[min(j, len(rungs) - 1)] for j in range(rounds)]

    rotor_diags: list[dict[str, Any]] = []
    for i in range(r.shape[0]):
        round_diags: list[dict[str, Any]] = []
        for j, rung in enumerate(schedule):
            delta_ft, rd = _refine_round(
                x_hi,
                t_hi,
                t_aud,
                r,
                i,
                ft,
                rung,
                sr,
                f_max=f_max,
                snr_min=snr_min,
                snr_cap=snr_cap,
                guard_cycles=guard_cycles,
                pad_factor=pad_factor,
                min_rate=min_rate,
            )
            rd["round"] = j + 1
            if delta_ft is not None:
                step = np.clip(boxcar(delta_ft, smooth_n), -max_step, max_step)
                r[i] += step
                rd["step_rms"] = round(float(np.sqrt(np.mean(step**2))), 4)
                rd["step_max"] = round(float(np.max(np.abs(step))), 4)
            round_diags.append(rd)
        rotor_diags.append({"rotor": i, "rounds": round_diags})

    diagnostics: dict[str, Any] = {
        "rungs": [asdict(g) for g in schedule],
        "params": {
            "rounds": rounds,
            "max_step": max_step,
            "smooth_s": smooth_s,
            "f_max": f_max,
            "snr_min": snr_min,
            "snr_cap": snr_cap,
            "guard_cycles": guard_cycles,
            "pad_factor": pad_factor,
            "oversample": oversample,
            "min_rate": min_rate,
        },
        "rotors": rotor_diags,
    }
    return r, diagnostics
