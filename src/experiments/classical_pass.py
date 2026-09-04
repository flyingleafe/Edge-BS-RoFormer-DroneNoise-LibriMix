"""One classical ``pi_kalman`` pass behind a neural seed, on the frozen real split.

Why this exists. The C2 candidate (the HG-CKLA refiner) is judged against the
classical phase-increment pass it mirrors: the synthesis doc says that if C2
has no fixed point at the truth, the classical pass stays as the precision
stage behind C1. This module runs that pass on the same clips, from the same
initial trajectories, with the flagship's protocol row (``PI_PROTOCOL``:
3 iterations, 6 Hz band, joint pair mode), so the two refiners are read on
one table.

The frozen split is 37 clips of 8 s at 16 kHz with 8 microphones; labels sit
on the STFT grid (hop 512, 251 frames). ``pi_kalman_refine`` accepts ``(C, T)``
audio and refines all rotors at once.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from tracking.top import PI_PROTOCOL, get_rps, pi_kalman_stage, tracking_frame

__all__ = ["pi_kalman_pass", "refine_clips"]

HOP = 512
SR = 16000


def pi_kalman_pass(
    audio: np.ndarray,
    init: np.ndarray,
    *,
    sr: int = SR,
    hop: int = HOP,
    n_iter: int | None = None,
    threads: int = 4,
) -> np.ndarray:
    """``(C, N)`` audio + ``(R, T)`` rev/s on the hop grid -> refined ``(R, T)``.

    ``n_iter`` overrides the protocol's 3 inner iterations (one pass of the
    outer loop either way). Rotors at zero stay at zero: the core treats a
    zero rate as stopped, so the caller need not mask them.
    """
    init = np.asarray(init, dtype=np.float64)
    times = np.arange(init.shape[-1], dtype=np.float64) * hop / sr
    frame = tracking_frame(np.asarray(audio, dtype=np.float32), sr, rps=init, frame_times=times)
    kwargs: dict[str, Any] = {"threads": threads}
    if n_iter is not None:
        kwargs["n_iter"] = n_iter
    stage = pi_kalman_stage(PI_PROTOCOL, diagnostics=False, **kwargs)
    out = stage(frame)
    r, _ = get_rps(out)
    return r


def refine_clips(clips: list[dict], inits: np.ndarray, **kw) -> tuple[np.ndarray, float]:
    """Refine every clip of ``experiments.slot_real.real_clips`` from ``inits``.

    ``inits`` is ``(n_clips, R, T)``. Returns the refined array and the wall
    time per clip in seconds.
    """
    out = np.empty_like(np.asarray(inits, dtype=np.float64))
    t0 = time.time()
    for i, clip in enumerate(clips):
        out[i] = pi_kalman_pass(clip["audio"], inits[i], **kw)
    return out, (time.time() - t0) / max(len(clips), 1)
