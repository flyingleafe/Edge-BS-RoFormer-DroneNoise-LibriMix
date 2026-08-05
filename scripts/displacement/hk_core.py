#!/usr/bin/env python3
"""DREGON slice loader for the comb-displacement drivers.

One job: give a driver the audio of one time slice, its sample rate, and a rotor
speed telemetry channel interpolated onto that slice's audio grid.

``motors_measured`` is the default channel, because it is the real tachometer:
its values lie on the reciprocal-integer lattice of a period counter. Only the
five ``free-flight_*_room1`` recordings carry it. The rest have
``motors_command`` only, which is a commanded value and not a measurement.
:func:`available_channels` says which channels a recording has, and
:func:`load_raw` takes the channel as a keyword, so a caller stays explicit.

The measurement code that consumes this is ``tracking.comb_displacement`` and
``tracking.order_domain``. Nothing here computes: this file is data resolution
only, and it is the reason the two roots (code, data) stay separate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]  # this checkout (code)
sys.path.insert(0, str(ROOT / "src"))

from utils.paths import get_data_path  # noqa: E402

DREGON = get_data_path("DREGON")

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
