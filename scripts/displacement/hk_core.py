#!/usr/bin/env python3
"""Shared core for the short-segment high-k demodulation (F1 redraw) and F0.

Everything here is referenced to a rotor-speed telemetry channel.  A harmonic k
of rotor r is demodulated by exp(-i k phi_r(t)) with phi_r(t) = 2 pi \\int g_r dt,
g_r = the telemetry rate in rev/s.  In the demodulated envelope the TELEMETRY
rate is exactly DC, so an envelope frequency f maps to an acoustic shaft-rate
offset delta = f / k rev/s.

DREGON's ``motors_measured`` is the default channel because it is the real
tachometer.  Only the five ``free-flight_*_room1`` recordings carry it; the rest
have ``motors_command`` only, which is a commanded value, not a measurement.
:func:`available_channels` reports which a recording has, and ``load_raw`` takes
the channel as a keyword so a caller can be explicit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DREGON = ROOT / "data/DREGON"
OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"

CORR = 0.99458  # measured telemetry correction (x this) - see dregon_telemetry.md
DISP_REVS = -0.542e-2  # relative displacement (multiplicative)

MEASURED = "motors_measured"
COMMAND = "motors_command"


def _motors_path(rid: str) -> Path:
    return DREGON / f"DREGON_{rid}" / f"DREGON_{rid}_motors.mat"


def available_channels(rid: str) -> list[str]:
    """Telemetry channels present for a DREGON recording, best first.

    ``motors_measured`` (the tachometer) comes before ``motors_command`` (the
    flight controller's demand); the distinction matters scientifically, so
    every figure built from this must say which it used.
    """
    p = _motors_path(rid)
    if not p.exists():
        return []
    names = scipy.io.loadmat(str(p))["motor"].dtype.names or ()
    out = []
    if "measured" in names:
        out.append(MEASURED)
    if "command" in names:
        out.append(COMMAND)
    return out


def load_raw(rid: str, t0: float, dur: float, channel: str = MEASURED):
    """``(audio (C,N) float64, sr, g (R,N) rev/s on the audio grid, rates_mean)``.

    ``g`` is the requested telemetry channel interpolated onto the audio sample
    grid of the requested slice.  ``channel`` is ``motors_measured`` (default,
    the historical behaviour) or ``motors_command``.
    """
    field = {MEASURED: "measured", COMMAND: "command"}[channel]
    d = DREGON / f"DREGON_{rid}"
    mat = scipy.io.loadmat(str(_motors_path(rid)))["motor"]
    ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
    vals = mat[field][0, 0].astype(np.float64).T  # (R, M)
    if channel == COMMAND:
        vals = _clean_command(vals)
    t_a0 = float(
        scipy.io.loadmat(str(d / f"DREGON_{rid}_audiots.mat"))["audio_timestamps"].flatten()[0]
    )
    t_tel = ts - t_a0
    info = sf.info(str(d / f"DREGON_{rid}.wav"))
    sr = info.samplerate
    a0 = int(round(t0 * sr))
    n = int(round(dur * sr))
    x, _ = sf.read(str(d / f"DREGON_{rid}.wav"), start=a0, frames=n, always_2d=True)
    audio = x.T.astype(np.float64)
    t = (a0 + np.arange(audio.shape[1])) / sr
    g = np.stack([np.interp(t, t_tel, vals[r]) for r in range(vals.shape[0])])
    return audio, sr, g, g.mean(axis=1)


def _clean_command(vals: np.ndarray) -> np.ndarray:
    """``motors_command`` with its leading logging freeze removed.

    Delegates to the project's ``clean_command_spikes`` when ``src`` is
    importable, so the cleaning matches the published DREGON frames; falls back
    to a bare median filter otherwise.
    """
    try:
        import sys

        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from data_processing.sources.dregon import clean_command_spikes

        return np.asarray(clean_command_spikes(vals), dtype=np.float64)
    except Exception:
        from scipy.signal import medfilt

        return np.stack([medfilt(v, 21) for v in vals])


def phase(g_r: np.ndarray, sr: int) -> np.ndarray:
    """Cumulative telemetry phase in radians (2 pi * revolutions)."""
    return 2.0 * np.pi * np.cumsum(g_r) / sr


def demod_spec(
    audio: np.ndarray,
    sr: int,
    phi: np.ndarray,
    k: int,
    seg_s: float,
    band_revs: float,
    overlap: float = 0.75,
    pad: int = 4,
):
    """Short-time spectrum of harmonic ``k``'s demodulated envelope.

    Returns ``(t_frames_s, rev_axis, P (F, T))`` with ``P`` the power averaged
    INCOHERENTLY over microphones, and ``rev_axis`` the offset axis in rev/s
    (envelope Hz divided by k).  ``pad`` only interpolates the display grid; the
    true resolution stays ``1 / seg_s / k`` rev/s.
    """
    n_seg = int(round(seg_s * sr))
    n_seg -= n_seg % 2
    hop = max(int(round(n_seg * (1.0 - overlap))), 1)
    n = audio.shape[1]
    starts = np.arange(0, n - n_seg + 1, hop)
    nfft = n_seg * pad
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sr))
    band_hz = band_revs * k
    keep = np.abs(freqs) <= band_hz
    rev_axis = freqs[keep] / k
    win = np.hanning(n_seg)
    carrier = np.exp(-1j * k * phi)
    acc = np.zeros((len(starts), int(keep.sum())))
    for c in range(audio.shape[0]):
        z = audio[c] * carrier
        fr = np.lib.stride_tricks.sliding_window_view(z, n_seg)[::hop] * win
        Z = np.fft.fftshift(np.fft.fft(fr, n=nfft, axis=-1), axes=-1)[:, keep]
        acc += np.abs(Z) ** 2
    acc /= audio.shape[0]
    t_frames = (starts + n_seg / 2.0) / sr
    return t_frames, rev_axis, acc.T


def prominence(rev_axis, P, search_revs=0.85, smooth_revs=0.0):
    """``(prom_db, peak_revs, prof_db)`` of the time-averaged demod profile.

    ``prom_db`` is the peak inside +-``search_revs`` over the MEDIAN of the whole
    displayed band (the in-band noise floor).
    """
    prof = P.mean(axis=1)
    floor = float(np.median(prof))
    prof_db = 10.0 * np.log10(prof / max(floor, 1e-300) + 1e-300)
    if smooth_revs > 0:
        step = float(rev_axis[1] - rev_axis[0])
        n_sm = max(3, int(round(smooth_revs / step)) | 1)
        kern = np.hanning(n_sm)
        prof_db = np.convolve(prof_db, kern / kern.sum(), mode="same")
    sw = np.abs(rev_axis) <= search_revs
    j = int(np.argmax(prof_db[sw]))
    return float(prof_db[sw][j]), float(rev_axis[sw][j]), prof_db


def neighbour_lines(rates: np.ndarray, rot: int, k: int, ylim: float):
    """Offsets (rev/s, telemetry frame of ``rot``) of every OTHER rotor's nearest
    harmonic to ``k * g_rot``, that lands inside +-``ylim``."""
    f0 = k * rates[rot]
    out = []
    for r2 in range(len(rates)):
        if r2 == rot:
            continue
        kk = int(round(f0 / rates[r2]))
        for cand in (kk - 1, kk, kk + 1):
            if cand < 1:
                continue
            off = (cand * rates[r2] - f0) / k
            if abs(off) <= ylim:
                out.append((r2, cand, off))
    return out
