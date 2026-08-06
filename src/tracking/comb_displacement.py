"""Where the acoustic comb sits relative to a rotor-speed carrier.

The measurement. Heterodyne harmonic ``k`` of rotor ``r`` by
``exp(-i 2 pi k int g_r dt)``, where ``g_r`` is a rotor-speed carrier in rev/s
(telemetry, or any candidate trajectory), and brickwall-lowpass the product to a
per-harmonic band. In the demodulated envelope the carrier rate is exactly DC.
Thus an envelope frequency ``f`` is an acoustic shaft-rate offset
``delta_k = f / k`` rev/s. A short-time spectrum of the envelope gives
``delta_k(t)``, and the power ratio over the in-band median gives a per-frame
weight. This module is the one implementation of that measurement. The
DREGON comb-displacement campaign (``docs/experiments/dregon-comb-displacement.md``,
GitHub issue 17) is its first consumer.

Why a null is mandatory. Section B of issue 17 makes the null controls part of
every number, because the campaign lost two claims to their absence. The traps,
in the order they were found:

* A peak-pick inside a search window of half-width ``W`` returns approximately
  ``W / 2`` on PURE NOISE. The half-width here is ``min(1.5 k, 8) Hz``, that is
  ``<= 8 / k`` rev/s, so it shrinks as ``1 / k``. High-k noise then peak-picks to
  a small and impressive number. The withdrawn claim "the high-k comb tracks
  telemetry to 0.086 rev/s" was matched by a null at 0.0857 (ratio 1.00).
* The pulse-pair estimator is window-free: its unambiguous range is
  ``|delta| < fs_env / (2 k)``, far outside the demod band, so no search window
  can bias it. But it returns approximately 0 on symmetric in-band noise. Thus
  agreement between the peak-pick and the pulse-pair is not evidence.
* The half-integer carrier ``(k + 0.5) g_r(t)`` is the null of the first kind.
  The trajectory, the band, the decimation and the gate stay identical, and only
  the carrier rate changes. No rotor line can exist there. The bank reuses the
  tracker's own integer harmonic recursion: ``exp(-i (2k+1) phi / 2)`` is
  ``exp(-i (k + 1/2) phi)``, so the half-integer carrier costs one halved phase
  and the odd harmonic index ``2k + 1``.

Why the collision rule is re-derived. The tracker's twin rule
(:func:`tracking.phase_increment_tracker._twin_collision_mask`) assumes that the
carrier is ``k r_i`` of a rotor that is itself a row of ``r_ft``. Every null
breaks that assumption: the off-comb carrier is ``(k + 0.5) r_i``, the
mismatched carrier is ``k g_partner`` of a different window, and the permuted
carrier is another rotor's telemetry. The interferers, on the other hand, are
always the AUDIO's real rotor lines. A null gated against fictional lines (or a
null not gated against the real ones) catches interference that the measurement
is protected from, which makes the comparison worthless.
:func:`carrier_collision_mask` therefore re-derives the rule against the true
rotor lines for an arbitrary carrier. The rotor index is kept for one purpose
only: the tracker skips its own rotor, and this function skips it the same way.

Purity. This module imports numpy and one sibling only. It does no file
input/output, it holds no frozen window table and no path. The drivers
(``scripts/displacement/nullcontrol.py``) supply the data and the protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tracking.phase_increment_tracker import demod_bank

__all__ = [
    "DisplacementConfig",
    "carrier_collision_mask",
    "combine_k",
    "demod_comb_bank",
    "measure_variant",
    "nearest_interloper_hz",
    "profile_prominence",
    "pulse_pair",
    "pulse_pair_bank",
    "ridge_from_envelope",
    "weighted_stats",
]

#: The "displaced" harmonic set: low k, where the interloper density is low.
LOW_K: tuple[int, ...] = tuple(range(2, 14))
#: The "on-grid" harmonic set: high k, where the null trap above lives.
HIGH_K: tuple[int, ...] = tuple(range(16, 41))


@dataclass(frozen=True)
class DisplacementConfig:
    """The measurement geometry: band, search window, gate, and envelope rate.

    All values are the frozen campaign defaults. The two shapes matter more
    than the values:

    * The band is ``min(b0_revs * k, band_frac_of_rate * mean rate)`` Hz. The
      first term keeps a CONSTANT capture in rev/s at every harmonic, because a
      rate error ``dr`` displaces harmonic ``k`` by ``k dr`` Hz. The second term
      keeps the band away from the next harmonic.
    * The search half-width is ``min(search_revs * k, search_hz_cap)`` Hz. The
      cap keeps the window inside the line spacing of the interleaved 4-rotor
      comb at high k. It is also the source of the ``W / 2`` noise trap, so
      every number that uses it must carry its null.
    """

    sr: int = 16000
    fs_env: float = 250.0
    b0_revs: float = 3.0
    band_frac_of_rate: float = 0.45
    search_revs: float = 1.5
    search_hz_cap: float = 8.0
    collision_guard: float = 1.6
    f_max: float = 6000.0
    min_rate: float = 5.0

    @property
    def stride(self) -> int:
        """Decimation factor from the audio rate to the envelope rate."""
        return int(round(self.sr / self.fs_env))

    def search_hz(self, k: int) -> float:
        """Peak-search half-width in Hz for harmonic ``k``."""
        return min(self.search_revs * k, self.search_hz_cap)

    def band_hz(self, ks: Sequence[int], rate_mean: float) -> np.ndarray:
        """``(K,)`` demodulation half-band in Hz, one entry for each ``k``."""
        return np.array(
            [min(self.b0_revs * k, self.band_frac_of_rate * rate_mean) for k in ks],
            dtype=np.float64,
        )

    def eff_search_revs(self, k: int, band_hz: float) -> float:
        """Effective peak-search HALF width in rev/s, which is what runs.

        The search window is also capped at 0.9 of the band, because a peak on
        the band edge is a filter artifact. Report this value with every
        offset: the noise expectation of a peak-pick is half of it.
        """
        return min(self.search_hz(k), 0.9 * band_hz) / k

    def seg_len_env(self, k: int) -> int:
        """Envelope-STFT segment length in envelope samples for harmonic ``k``.

        The length keeps the rev/s resolution ``fs_env / (n_seg k)`` almost
        constant (about 0.06 rev/s) across k, and it stays below the measured
        coherence time at high k. The floor is 1 s and the ceiling is 8 s.
        """
        seg_s = float(np.clip(16.0 / max(k, 1), 1.0, 8.0))
        n = int(round(seg_s * self.fs_env))
        return n - (n % 2)


def demod_comb_bank(
    audio: np.ndarray,
    r_row: np.ndarray,
    ft: np.ndarray,
    ks: Sequence[int],
    *,
    cfg: DisplacementConfig = DisplacementConfig(),
    half: bool = False,
    rate_ref: float | None = None,
    band_hz_k: np.ndarray | None = None,
    probe_off_hz: float = 0.0,
    return_probe: bool = False,
) -> tuple[np.ndarray, ...]:
    """Demodulation bank around ``k * g(t)``, or ``(k + 0.5) * g(t)`` for the null.

    Args:
        audio: ``(C, T)`` audio at ``cfg.sr``.
        r_row: ``(N,)`` carrier rate in rev/s on the frame grid ``ft``. This is
            ONE rotor's trajectory, not the full array, so that a permuted or a
            mismatched carrier is expressible.
        ft: ``(N,)`` frame times in seconds, audio-relative.
        ks: harmonic indices.
        cfg: the measurement geometry.
        half: use the half-integer null carrier.
        rate_ref: mean rate the BAND is derived from. ``None`` uses the
            carrier's own mean, which makes the band a function of the
            candidate; a fixed-degrees-of-freedom comparison across candidate
            trajectories (``tracking.fitness``) pins it to the window's
            reference rate instead.
        band_hz_k: ``(K,)`` half-band in Hz, overriding ``cfg.band_hz``
            entirely. The most explicit form of the same pin.
        probe_off_hz: off-comb noise probe offset in Hz. The probe rides at a
            constant offset from every harmonic and comes out of the SAME
            forward transform, so it costs nothing.
        return_probe: also return the probe bank.

    Returns:
        ``(z (C, K, n_env) complex64, band_hz_k (K,))``, or
        ``(z, z_probe, band_hz_k)`` when ``return_probe``.

    The half-integer carrier reuses the tracker's own integer recursion: it
    halves the phase and asks for harmonic ``2k + 1``, because
    ``exp(-i (2k+1) phi / 2) = exp(-i (k + 1/2) phi)``. The band and the
    decimation are identical to the on-comb call, so the two banks differ in
    the carrier rate and in nothing else.
    """
    ks = list(ks)
    n_t = audio.shape[-1]
    t_aud = np.arange(n_t) / cfg.sr
    r_aud = np.interp(t_aud, ft, r_row)
    phi = 2.0 * np.pi * np.cumsum(r_aud) / cfg.sr
    if band_hz_k is None:
        rate = float(np.mean(r_row)) if rate_ref is None else float(rate_ref)
        band_hz_k = cfg.band_hz(ks, rate)
    band_hz_k = np.asarray(band_hz_k, dtype=np.float64)
    n_env = n_t // cfg.stride
    y32 = np.asarray(audio, dtype=np.float32)
    phi_use = phi / 2.0 if half else phi
    ks_use = [2 * k + 1 for k in ks] if half else list(ks)
    z_on, z_off = demod_bank(
        y32,
        phi_use,
        t_aud,
        ks_use,
        float(probe_off_hz),
        cfg.stride,
        n_env,
        float(np.max(band_hz_k)) / cfg.sr,
        band_cyc_k=band_hz_k / cfg.sr,
        sr=float(cfg.sr),
    )
    if return_probe:
        return z_on, z_off, band_hz_k
    return z_on, band_hz_k


def ridge_from_envelope(
    z_k: np.ndarray,
    band_hz: float,
    k: int,
    *,
    cfg: DisplacementConfig = DisplacementConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Short-time ridge of one harmonic's envelope.

    Args:
        z_k: ``(C, n_env)`` envelope of one harmonic.
        band_hz: that harmonic's demodulation half-band in Hz.
        k: the harmonic index.
        cfg: the measurement geometry.

    Returns:
        ``(t_frames_s, delta_revs, snr_lin, spec_db (F, T), rev_axis (F,))``.
        For each STFT frame, the parabolically refined peak offset in rev/s and
        its power ratio over the in-band median, plus the channel-averaged
        spectrogram on a rev/s axis for the figures.

    The channel average is INCOHERENT (power, not amplitude). Each microphone
    carries a different static phase on the same line, so a coherent sum across
    channels attenuates the line it is meant to measure.
    """
    n_seg = min(cfg.seg_len_env(k), z_k.shape[-1])
    n_env = z_k.shape[-1]
    hop = max(n_seg // 2, 1)
    starts = list(range(0, n_env - n_seg + 1, hop))
    win = np.hanning(n_seg)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_seg, d=1.0 / cfg.fs_env))
    keep = np.abs(freqs) <= band_hz
    rev_axis = freqs[keep] / k
    spec = np.empty((len(starts), int(keep.sum())))
    for a, s in enumerate(starts):
        seg = z_k[:, s : s + n_seg] * win
        p = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=-1), axes=-1)) ** 2
        spec[a] = p.mean(axis=0)[keep]  # incoherent channel average
    t_frames = (np.array(starts) + n_seg / 2.0) / cfg.fs_env
    search = np.abs(rev_axis) <= cfg.eff_search_revs(k, band_hz)
    idx_off = int(np.argmax(search))  # first True
    delta = np.full(len(starts), np.nan)
    snr = np.zeros(len(starts))
    for a in range(len(starts)):
        row = spec[a]
        sub = row[search]
        j = int(np.argmax(sub)) + idx_off
        # parabolic refinement on the log-power peak
        if 0 < j < len(row) - 1:
            y0, y1, y2 = np.log(row[j - 1 : j + 2] + 1e-300)
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            frac = float(np.clip(frac, -0.5, 0.5))
        else:
            frac = 0.0
        step = rev_axis[1] - rev_axis[0]
        delta[a] = rev_axis[j] + frac * step
        floor = float(np.median(row))
        snr[a] = float(row[j]) / max(floor, 1e-300)
    spec_db = 10.0 * np.log10(spec.T + 1e-300)
    return t_frames, delta, snr, spec_db, rev_axis


def profile_prominence(
    spec_db: np.ndarray,
    rev_axis: np.ndarray,
    keep_f: np.ndarray,
    k: int,
    band_hz: float,
    *,
    cfg: DisplacementConfig = DisplacementConfig(),
) -> tuple[float, float]:
    """``(prominence_db, peak_offset_rev_s)`` of the time-averaged profile.

    The power spectra of the uncollided frames are averaged, normalized by the
    in-band median, smoothed to about 0.05 rev/s, and peak-picked inside the
    search window. The smoothing prevents one noisy bin from claiming the line.

    The prominence is the bar the campaign reports against: a unit counts only
    if it stands a stated number of dB over the same statistic at the
    half-integer null. The pooled SNR-weighted mean is DILUTED by harmonics
    with no line, which contribute noise centred on zero, so the campaign
    reports the bar-restricted statistic and never the pooled one.
    """
    spec = np.power(10.0, spec_db.T / 10.0)  # (T, F)
    src = spec[keep_f] if keep_f.sum() >= 3 else spec
    prof = src.mean(axis=0)
    prof_db = 10.0 * np.log10(prof / np.median(prof) + 1e-300)
    step = float(rev_axis[1] - rev_axis[0])
    n_sm = max(3, int(round(0.05 / step)) | 1)
    kern = np.hanning(n_sm)
    prof_sm = np.convolve(prof_db, kern / kern.sum(), mode="same")
    sw = np.abs(rev_axis) <= cfg.eff_search_revs(k, band_hz)
    if not sw.any():
        return float("nan"), float("nan")
    j = int(np.argmax(prof_sm[sw]))
    return float(prof_sm[sw][j]), float(rev_axis[sw][j])


def pulse_pair(
    z_k: np.ndarray,
    k: int,
    keep_env: np.ndarray | None = None,
    *,
    cfg: DisplacementConfig = DisplacementConfig(),
    fs_env: float | None = None,
    min_samples: int = 8,
) -> tuple[float, float]:
    """Coherent phase-increment (pulse-pair) offset in rev/s, and its coherence.

    The estimator is ``arg(sum_n sum_c z[c,n] conj(z[c,n-1])) / (2 pi k dt_env)``.
    The lag product's phase is the envelope frequency and carries no static
    per-microphone phase, so here the channel sum IS coherent and legitimate.
    Coherence is ``|sum| / sum|.|``.

    It is unambiguous over ``|delta| < fs_env / (2 k)`` rev/s, which is far
    outside the demod band. Thus it is window-free: the ``W / 2`` trap of the
    peak-pick cannot touch it. But it returns approximately 0 on symmetric
    in-band noise, so it agrees with a peak-pick that found nothing. Read the
    two estimators as independent failures, not as a corroboration.

    This is THE centre estimator of the tracking package. It is the ML centre
    of the increment distribution; the MEDIAN of wrapped increments is biased
    toward zero once the phasor is noise dominated (a noise phasor has winding
    number 0), which is what ``docs/experiments/vk-frontend-probe.md`` §Method
    records. ``keep_env=None`` uses every envelope sample, and ``fs_env``
    overrides ``cfg.fs_env`` for a caller on a different envelope grid.
    """
    z_k = np.atleast_2d(z_k)
    n = z_k.shape[-1]
    m = np.ones(n, dtype=bool) if keep_env is None else np.asarray(keep_env, dtype=bool)
    if int((m[1:] & m[:-1]).sum()) < min_samples:
        return float("nan"), 0.0
    off, coh = pulse_pair_bank(
        z_k[:, None, :],
        [k],
        fs_env=cfg.fs_env if fs_env is None else float(fs_env),
        keep=m,
    )
    return float(off[0]), float(coh[0])


def pulse_pair_bank(
    z: np.ndarray,
    ks: Sequence[int],
    *,
    fs_env: float,
    keep: np.ndarray | None = None,
    sum_channels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized pulse-pair centres: ``(offsets rev/s, coherences)``.

    THE core of :func:`pulse_pair` — the scalar form is a one-harmonic call of
    this one. ``z`` is ``(C, K, N)`` or ``(K, N)``; ``keep`` is an optional
    ``(N,)`` envelope-sample mask (a lag product needs both of its samples).
    ``sum_channels`` performs the coherent channel sum the scalar form does;
    ``False`` keeps the channel axis, which is what a per-(channel, harmonic)
    cell statistic needs.
    """
    z = np.asarray(z)
    lag = z[..., 1:] * np.conj(z[..., :-1])
    if keep is not None:
        m = np.asarray(keep, dtype=bool)
        lag = lag[..., m[1:] & m[:-1]]
    s = lag.sum(axis=-1)
    d = np.abs(lag).sum(axis=-1)
    if sum_channels and lag.ndim == 3:
        s, d = s.sum(axis=0), d.sum(axis=0)
    kf = np.asarray(list(ks), dtype=np.float64)
    off = np.angle(s) * fs_env / (2.0 * np.pi * kf)
    return off, np.where(d > 0, np.abs(s) / np.maximum(d, 1e-300), 0.0)


def carrier_collision_mask(
    r_ft_true: np.ndarray,
    r_row_carrier: np.ndarray,
    rot: int,
    ks: Sequence[int],
    *,
    half: bool = False,
    cfg: DisplacementConfig = DisplacementConfig(),
) -> np.ndarray:
    """``(K, N)`` bool: a real rotor line inside harmonic ``k``'s search window.

    Args:
        r_ft_true: ``(R, N)`` the AUDIO's real rotor trajectories. These are the
            interferers, whatever the carrier is.
        r_row_carrier: ``(N,)`` the carrier trajectory.
        rot: the carrier's rotor index, skipped as an interferer exactly as the
            tracker skips its own rotor.
        ks: harmonic indices.
        half: the carrier is at ``k + 0.5``.
        cfg: the measurement geometry.

    Only the two harmonics that bracket ``k_carrier * r_c / r_j`` can enter the
    window, so the search over ``k'`` is two floor/ceil candidates. The
    separation is ``collision_guard * search_hz(k)``, that is the search
    half-width with a margin.
    """
    kf = np.asarray(list(ks), dtype=np.float64) + (0.5 if half else 0.0)
    sep = np.array([cfg.collision_guard * cfg.search_hz(k) for k in ks], dtype=np.float64)[:, None]
    fi = kf[:, None] * r_row_carrier[None, :]
    coll = np.zeros(fi.shape, dtype=bool)
    for j in range(r_ft_true.shape[0]):
        if j == rot or float(np.mean(r_ft_true[j])) < cfg.min_rate:
            continue
        rj = np.maximum(r_ft_true[j], 1e-3)[None, :]
        base = fi / rj
        for kp in (np.floor(base), np.ceil(base)):
            fj = np.maximum(kp, 1.0) * rj
            coll |= (np.abs(fj - fi) < sep) & (fj <= cfg.f_max + sep)
    return coll


def nearest_interloper_hz(
    r_ft_true: np.ndarray,
    r_row_carrier: np.ndarray,
    rot: int,
    ks: Sequence[int],
    *,
    half: bool = False,
    f_max: float = 6000.0,
    min_rate: float = 5.0,
) -> np.ndarray:
    """``(K, N)`` Hz: distance to the NEAREST foreign rotor line, per frame.

    The quantitative form of :func:`carrier_collision_mask`: instead of the
    boolean "a real line is inside the window", it returns how far the closest
    real line of another rotor is from harmonic ``k`` of the carrier. A caller
    that knows its own band gets its own collision rule from this
    (``nearest < band + guard``), and a caller that wants to grade contested
    harmonics by how close the interferer is gets that too.

    Arguments and the bracketing search are those of
    :func:`carrier_collision_mask`; ``f_max`` bounds the interferer combs
    (lines above it are not in the audio band) and ``min_rate`` skips
    near-silent rotors. No interferer anywhere -> ``inf``.
    """
    kf = np.asarray(list(ks), dtype=np.float64) + (0.5 if half else 0.0)
    fi = kf[:, None] * np.asarray(r_row_carrier, dtype=np.float64)[None, :]
    best = np.full(fi.shape, np.inf)
    for j in range(r_ft_true.shape[0]):
        if j == rot or float(np.mean(r_ft_true[j])) < min_rate:
            continue
        rj = np.maximum(np.asarray(r_ft_true[j], dtype=np.float64), 1e-3)[None, :]
        base = fi / rj
        for kp in (np.floor(base), np.ceil(base)):
            fj = np.maximum(kp, 1.0) * rj
            d = np.where(fj <= f_max, np.abs(fj - fi), np.inf)
            best = np.minimum(best, d)
    return best


def weighted_stats(vals: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """``(weighted mean, weighted std, n_eff)`` over the finite, positive-weight
    entries. ``n_eff`` is Kish's effective sample count ``(sum w)^2 / sum w^2``."""
    ok = np.isfinite(vals) & (w > 0)
    if not ok.any():
        return float("nan"), float("nan"), 0.0
    v, ww = vals[ok], w[ok]
    m = float(np.sum(ww * v) / np.sum(ww))
    var = float(np.sum(ww * (v - m) ** 2) / np.sum(ww))
    n_eff = float(np.sum(ww) ** 2 / np.sum(ww**2))
    return m, float(np.sqrt(var)), n_eff


def combine_k(d_grid: np.ndarray, w_grid: np.ndarray, kset: Sequence[int]) -> np.ndarray:
    """Weight-combined offset series over a harmonic set: ``(N,)`` rev/s.

    Row ``a`` of ``d_grid`` / ``w_grid`` is harmonic ``a + 1`` on the frame
    grid, so ``kset`` selects rows ``k - 1``. Harmonics outside the grid are
    dropped, which lets a short probe run (``k <= 8``) reuse the frozen
    ``LOW_K`` / ``HIGH_K`` sets. Frames with no weight become NaN.
    """
    idx = [k - 1 for k in kset if 1 <= k <= d_grid.shape[0]]
    if not idx:
        return np.full(d_grid.shape[1], np.nan)
    d, w = d_grid[idx], w_grid[idx]
    good = np.isfinite(d)
    w = np.where(good, w, 0.0)
    d = np.where(good, d, 0.0)
    tot = w.sum(axis=0)
    return np.where(tot > 0, (w * d).sum(axis=0) / np.maximum(tot, 1e-30), np.nan)


def measure_variant(
    audio: np.ndarray,
    ft: np.ndarray,
    r_row_carrier: np.ndarray,
    r_ft_true: np.ndarray,
    rot: int,
    ks: Sequence[int],
    *,
    half: bool = False,
    cfg: DisplacementConfig = DisplacementConfig(),
    low_k: Sequence[int] = LOW_K,
    high_k: Sequence[int] = HIGH_K,
) -> dict[str, Any]:
    """One (window, rotor, variant) pass: the campaign's unit of measurement.

    The carrier comes in as a ROW, not as an array plus a rotor index, so every
    null of issue 17 section B is one call of this function: the measurement
    passes the rotor's own telemetry, the off-comb null passes the same row with
    ``half=True``, the correspondence-breaking null passes another window's row,
    and the rotor-permutation null passes another rotor's row. ``rot`` stays
    only for the collision gate's "skip my own rotor" rule, and ``r_ft_true``
    always holds the AUDIO's real trajectories.

    Returns a dict with:

    * ``per_k[str(k)]``: a 10-element list. The campaign's JSON readers index
      it by position, so the order is frozen:

      0. peak-pick offset in rev/s, or None if no frame survived
      1. median per-frame SNR (linear) over the uncollided frames
      2. weighted std of the offset in rev/s, or None
      3. effective frame count of the weighted mean
      4. profile prominence in dB over the in-band floor, or None
      5. profile peak offset in rev/s, or None
      6. pulse-pair offset in rev/s, or None
      7. pulse-pair coherence, 0 to 1
      8. uncollided fraction of the frames
      9. search half-width in rev/s (the noise expectation is half of it)

    * ``low_k_series_mae`` / ``high_k_series_mae``: the mean absolute value of
      the weight-combined offset series over ``low_k`` / ``high_k``.
    * ``low_k_series_mean`` / ``high_k_series_mean``: the signed means. The
      displacement is one-sided, so a mean far below the MAE is a sign that the
      set is dominated by noise.
    """
    ks = list(ks)
    z_on, band_hz_k = demod_comb_bank(audio, r_row_carrier, ft, ks, cfg=cfg, half=half)
    clean = ~carrier_collision_mask(r_ft_true, r_row_carrier, rot, ks, half=half, cfg=cfg)
    n_env = z_on.shape[-1]
    t_env = (np.arange(n_env) + 0.5) * cfg.stride / cfg.sr
    d_grid = np.full((len(ks), ft.size), np.nan)
    w_grid = np.zeros((len(ks), ft.size))
    per_k: dict[str, list[float | None]] = {}
    for a, k in enumerate(ks):
        band = float(band_hz_k[a])
        tf, delta, snr, spec_db, rev_axis = ridge_from_envelope(z_on[:, a], band, k, cfg=cfg)
        keep_f = np.interp(tf, ft, clean[a].astype(float)) > 0.999
        w = np.where(keep_f, np.maximum(snr - 1.0, 0.0), 0.0)
        m, sd, n_eff = weighted_stats(delta, w)
        prom, prom_off = profile_prominence(spec_db, rev_axis, keep_f, k, band, cfg=cfg)
        keep_env = np.interp(t_env, ft, clean[a].astype(float)) > 0.999
        pp, coh = pulse_pair(z_on[:, a], k, keep_env, cfg=cfg)
        snr_c = snr[keep_f] if keep_f.any() else snr
        per_k[str(k)] = [
            None if not np.isfinite(m) else round(m, 4),
            round(float(np.median(snr_c)), 4),
            None if not np.isfinite(sd) else round(sd, 4),
            round(n_eff, 2),
            round(prom, 3) if np.isfinite(prom) else None,
            round(prom_off, 4) if np.isfinite(prom_off) else None,
            None if not np.isfinite(pp) else round(pp, 4),
            round(coh, 4),
            round(float(np.mean(keep_f)), 3),
            round(cfg.eff_search_revs(k, band), 4),
        ]
        d_grid[a] = np.interp(ft, tf, delta)
        w_grid[a] = np.interp(ft, tf, w)
    lowk = combine_k(d_grid, w_grid, low_k)
    highk = combine_k(d_grid, w_grid, high_k)
    return {
        "per_k": per_k,
        "low_k_series_mae": round(float(np.nanmean(np.abs(lowk))), 4),
        "high_k_series_mae": round(float(np.nanmean(np.abs(highk))), 4),
        "low_k_series_mean": round(float(np.nanmean(lowk)), 4),
        "high_k_series_mean": round(float(np.nanmean(highk)), 4),
    }
