"""Michael's drone-noise dataset — 8-channel audio + aligned rotor speeds.

The recordings live in `data/recording_with_motor_speed/`:
  - recording_1/124.wav + FLY124.csv
  - recording_2/125.wav + FLY125.csv

Each WAV is an 8-channel in-flight drone recording; each DJI flight-controller
CSV logs per-motor rotation speed (RPM). The audio and telemetry clocks are
*not* aligned by default — a per-file `time_offset` (seconds) plus a
`time_dilation` clock-rate factor brings them into register, and the logged
speeds themselves are low by a small multiplicative factor (`MICHAELS_RPS_SCALE`).
All three constants are now **measured**, not hand-tuned: see the block comment
above `MICHAELS_FILES` and `docs/experiments/rps-refine-precision.md` § WP13.

`load_michaels_timeframes()` returns a list of two `td.Frame`s, one per
recording, each holding:
  - `audio`: 8-channel Series (dims `("mic", "time")`) `(8, N)` at `sr`
  - `rps`  : Series of motor speeds (dims `("rotor", "time")`) `(4, M)` in
    revolutions/second

Audio and RPS share the frame's time anchor, so they are aligned: slicing the
frame slices both consistently.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa as lr
import numpy as np
import pandas as pd
import tdseries as td

from data_processing.frames import make_recording_frame

# Project data root: `$DATA_ROOT` if set, else `<repo>/data` (this file lives
# in `<repo>/data_processing/`).
_DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))

# ── Measured alignment / label calibration (2026-07-31) ─────────────────────
#
# Referee: the LABEL-FREE Vold-Kalman reconstruction residual
# ||x - x_hat||/||x|| (k = 1..30, `scripts/michaels_calib/calib.py:RECON_CFG`,
# identical to `rps_refine_lab.RECON_CFG`) of the harmonic model built ALONG a
# candidate telemetry trajectory, scored on contiguous 16 s windows of the
# frozen beat-VK window protocol. Only the TELEMETRY is shifted/scaled; our own
# blind RPS estimate is never consulted. Sweep: `scripts/michaels_calib/`,
# raw + fits in `omnirun-outputs/python-519b66/results/michaels_calib/`.
#
# TIMING — both recordings show a clock DILATION error (the audio-optimal lag
# drifts linearly with time), not a constant offset. With the old constants the
# audio prefers the telemetry shifted by lag(t) = a + b·t seconds; folding that
# into the loader's parameterisation gives
#     time_dilation_new = time_dilation_old / (1 - b)
#     time_offset_new   = time_offset_old  - a / (1 - b)
# (`scripts/michaels_calib/fit.py:fit_lag`, used for BOTH recordings):
#
#   FLY124 — 4 cruise windows w02..w05 (centres 40/56/72/88 s), lags
#     -61.83 / -49.00 / -34.61 / -31.77 ms; OLS b = +0.65356 ms/s,
#     a = -86.131 ms, R^2 = 0.942, residual RMS 2.90 ms vs 12.04 ms for a
#     constant-lag model. (-20.84, 1.001) -> (-20.753813, 1.001654644);
#     dilation multiplier x1.000654. The cluster sweep independently re-measured
#     w03/w04 at -48.51 / -34.56 ms, i.e. within 0.5 ms of the above.
#   FLY125 — 9 cruise windows w01..w09 (centres 24..152 s), lags -158.4 ->
#     -105.5 ms; OLS b = +0.37656 ms/s, a = -172.086 ms, R^2 = 0.923, residual
#     RMS 4.49 ms vs 16.19 ms constant-lag. (-26.51, 1.0048) ->
#     (-26.337849, 1.005178509); dilation multiplier x1.000377.
#
# VALUE — the logged speeds are LOW by ~0.55-0.68 rev/s at cruise (DREGON shows
# the opposite sign, so this is a Michael's-rig property, not a referee bias).
# Whether the error is additive (b_r = const) or multiplicative (b_r = g·rps_r)
# is statistically UNRESOLVED: the per-rotor discriminator (`prot`, regressing
# each rotor's optimal offset on its own mean rps over the 74-91 rev/s cruise
# spread) returns "additive" for FLY124 by a 4 % RMS margin (0.2939 vs 0.30557)
# and "multiplicative" for FLY125 by 0.04 % (0.38621 vs 0.38607) — a tie; the
# free-line fits have R^2 0.013 / 0.004 (no significant dependence on rotor mean
# rps) and the observed per-rotor spread (0.94 / 1.96 rev/s) dwarfs what either
# model predicts (0.147 / 0.117), i.e. the discriminator has no power here.
# We ship the MULTIPLICATIVE form: the two are indistinguishable in the cruise
# band where they were measured, but these frames cover the WHOLE recording
# including warm-up and ground idle, where an additive +0.6 rev/s would corrupt
# a near-stationary rotor's label (and manufacture harmonics at a standstill)
# while a scale correctly vanishes as rps -> 0. Fitted g: 0.008387 (FLY124,
# +0.671 rev/s at 80) and 0.006904 (FLY125, +0.552 rev/s at 80).
#
# (wav_path, csv_path, time_offset_sec, time_dilation)
MICHAELS_FILES = [
    (
        "recording_with_motor_speed/recording_1/124.wav",
        "recording_with_motor_speed/recording_1/FLY124.csv",
        -20.753813,
        1.001654644,
    ),
    (
        "recording_with_motor_speed/recording_2/125.wav",
        "recording_with_motor_speed/recording_2/FLY125.csv",
        -26.337849,
        1.005178509,
    ),
]

#: Per-recording MULTIPLICATIVE rev/s correction, keyed by CSV stem (the
#: recording id). Applied to the rotor speeds in `_load_michaels_data_raw`, so
#: every consumer of `rps` gets calibrated labels. Recordings without a measured
#: constant (the 103 unaligned `new-drone-noises` logs) fall back to 1.0.
MICHAELS_RPS_SCALE: dict[str, float] = {
    "FLY124": 1.00839,  # g = 0.008387 -> +0.671 rev/s at 80 rev/s
    "FLY125": 1.00690,  # g = 0.006904 -> +0.552 rev/s at 80 rev/s
}


def rps_scale_for(csv_path: str | Path) -> float:
    """Calibrated rev/s scale for a recording, from its CSV stem (1.0 if none)."""
    return MICHAELS_RPS_SCALE.get(Path(csv_path).stem.upper(), 1.0)


# ── Array / airframe geometry ───────────────────────────────────────────────
# Body frame: X = forward, Y = left, Z = up; origin at the drone body centre
# (rotor plane). The microphone array is a HORIZONTAL ring (plane = X-Y, normal
# +Z) mounted forward of and above the body: the array photo is a top-down shot
# in which the ring appears face-on as a full circle (a vertical ring would be
# edge-on), and its extent is labelled in both X (width) and Y (height). It sits
# 33 cm above and ~30 cm forward of the body, cleanly above the rotor downwash.
# Constants read from
# `data/recording_with_motor_speed/Microphone_Array_Configuration.jpeg`;
# wheelbase from the DJI Matrice 100 (the airframe in the rig photos).

N_MICS = 8
NUM_ROTORS = 4
MIC_ARRAY_RADIUS = 0.0825  # m (Ø165 mm)
# The rig spec's horizontal "20cm" is "drone body to centre of array" — i.e. the
# gap from the body's FRONT EDGE to the array centre, NOT from the body centre
# (the wording contrasts "drone body" with the explicit "centre of array"). To
# express the array in the body-centre frame we add the body's forward
# half-extent. Sanity check: the array ring sits clearly *forward* of the front
# props in the rig photos, so its forward offset must exceed the front-rotor
# reach (a = WHEELBASE/2·cos45° ≈ 230 mm); the old centre-referenced 200 mm put
# it *behind* the front rotors, which is wrong. 200 + ~100 ≈ 300 mm clears them.
ARRAY_GAP_FROM_BODY_EDGE = 0.20  # m: measured front-edge -> array centre (spec)
BODY_HALF_FORWARD = 0.10  # m: DJI Matrice 100 body centre -> front edge (estimate)
ARRAY_OFFSET_FORWARD = ARRAY_GAP_FROM_BODY_EDGE + BODY_HALF_FORWARD  # +X, body-centre frame
ARRAY_OFFSET_UP = 0.33  # m (+Z): above drone body, vertical
WHEELBASE = 0.650  # m: DJI Matrice 100 motor-to-motor diagonal
# rps-row order — michaels CSV "Motor:Speed:*" columns, first 4 (see
# `_load_michaels_data_raw`); rotor_positions below is in the SAME order.
ROTOR_ORDER = ("RFront", "LFront", "LBack", "RBack")


def get_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mic_positions (8, 3), rotor_positions (4, 3))`` in metres.

    Body frame: X = forward, Y = left, Z = up; origin at the body centre.
    Mirrors :func:`data_processing.dregon.get_geometry` so Michael's recordings
    can populate the ``"mic_pos"`` / ``"rotor_pos"`` Frame entries the same way
    DREGON does.

    Microphones: 8 evenly-spaced points on a HORIZONTAL ring (radius 82.5 mm) in
    the X-Y plane (normal +Z), numbered counter-clockwise starting 22.5° left of
    top, with the ring centre ``ARRAY_OFFSET_FORWARD`` forward (the spec's 20 cm
    front-edge gap + the body's forward half-extent) and 330 mm above the body.
    The exact in-plane orientation / handedness of the numbering is refined from
    audio by the geometry self-calibration (notebooks/geom_calibration.py).

    Rotors: X-quad, arms at ±45°, motor radius ``(W/2)·cos45°`` per horizontal
    axis, at body height (z = 0). Ordered to match the ``rps`` rows
    (``ROTOR_ORDER``: RFront, LFront, LBack, RBack).
    """
    # Microphones — HORIZONTAL ring in the X-Y plane (normal +Z); z is constant.
    # Photo axes map as image-up -> +X (forward), image-right -> -Y (lateral).
    theta = np.deg2rad(112.5 + 45.0 * np.arange(N_MICS))  # mic 1 at 112.5°, CCW
    fwd = MIC_ARRAY_RADIUS * np.sin(theta)  # image-up component -> +X (forward)
    lat = MIC_ARRAY_RADIUS * np.cos(theta)  # image-right component -> -Y (lateral)
    mic_positions = np.stack(
        [ARRAY_OFFSET_FORWARD + fwd, -lat, np.full(N_MICS, ARRAY_OFFSET_UP)], axis=-1
    )  # (8, 3)

    # Rotors — X-config; per-axis offset a = (W/2)·cos45°.
    a = (WHEELBASE / 2.0) * np.cos(np.deg2rad(45.0))
    rotor_positions = np.array(
        [
            [+a, -a, 0.0],  # RFront
            [+a, +a, 0.0],  # LFront
            [-a, +a, 0.0],  # LBack
            [-a, -a, 0.0],  # RBack
        ]
    )
    return mic_positions, rotor_positions


def _load_michaels_data_raw(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float = 0.0,
    time_dilation: float = 1.0,
    sr: int | None = None,
    rps_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load + align one recording — port of the notebook function.

    `rps_scale` defaults to the recording's calibrated `MICHAELS_RPS_SCALE`
    entry (`None` -> looked up from the CSV stem); pass an explicit float (e.g.
    ``1.0``) to score a hypothesis against the uncalibrated telemetry.

    Returns
    -------
    wav   : (n_channels, n_samples) audio at `sr`
    ts    : (M,) aligned motor timestamps (seconds, anchored at audio start)
    ms    : (4, M) motor speeds in revolutions/second (RPM / 60, calibrated)
    sr    : sample rate actually used
    """
    scale = rps_scale_for(csv_path) if rps_scale is None else float(rps_scale)
    wav, sample_rate = lr.load(str(wav_path), sr=sr, mono=False)
    if len(wav.shape) == 1:
        wav = wav[None, :]

    csv = pd.read_csv(csv_path)
    ms_cols = [c for c in csv.columns if "Motor" in c][:4]
    t_col = "Clock:offsetTime"
    small_csv = csv[[t_col] + ms_cols]

    wav_duration = wav.shape[-1] / sample_rate
    cut_csv = small_csv[
        (small_csv[t_col] >= time_offset) & (small_csv[t_col] <= wav_duration + time_offset)
    ]
    ts = np.asarray(cut_csv[t_col], dtype=np.float64)

    if ts[0] > time_offset:
        wav = wav[:, int((ts[0] - time_offset) * sample_rate) :]
        time_offset = ts[0]
        wav_duration = wav.shape[-1] / sample_rate

    if ts[-1] < wav_duration + time_offset:
        wav = wav[:, : int((ts[-1] - ts[0]) * sample_rate)]

    jump_idx = np.argmax(np.diff(ts))
    ts[0 : jump_idx + 2] = np.linspace(ts[0], ts[jump_idx + 2], jump_idx + 2)
    ts *= time_dilation

    ms = np.asarray(cut_csv[ms_cols], dtype=np.float64).T / 60 * scale
    return wav, ts, ms, int(sample_rate)


def load_michaels_timeframe(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float = 0.0,
    time_dilation: float = 1.0,
    sr: int | None = 16000,
    recording_id: str | None = None,
    rps_scale: float | None = None,
) -> td.Frame:
    """Load one Michael's recording as an aligned ``td.Frame``.

    The frame holds an 8-channel ``audio`` Series (dims ``("mic", "time")``)
    and an ``rps`` Series (dims ``("rotor", "time")``): the audio is anchored at
    t=0, the RPS timestamps are aligned to it via `time_offset` /
    `time_dilation`, and the speeds carry the calibrated `rps_scale`
    (`None` -> the recording's `MICHAELS_RPS_SCALE` entry).
    """
    wav, ts, ms, sample_rate = _load_michaels_data_raw(
        wav_path,
        csv_path,
        time_offset=time_offset,
        time_dilation=time_dilation,
        sr=sr,
        rps_scale=rps_scale,
    )
    audio = td.uniform(wav, sample_rate, dims=("mic", "time"))
    rps = td.events(ts, ms, dims=("rotor", "time"))
    meta = {"recording_id": recording_id or Path(wav_path).stem}
    mic_positions, rotor_positions = get_geometry()
    return make_recording_frame(
        {"audio": audio, "rps": rps},
        meta=meta,
        mic_pos=mic_positions,
        rotor_pos=rotor_positions,
    )


def load_michaels_timeframes(
    data_root: str | Path | None = None,
    sr: int | None = 16000,
) -> list[td.Frame]:
    """Load FLY124 and FLY125 as a list of two aligned 8-channel ``td.Frame``s.

    Each frame contains ``audio`` (8-channel Series, ``(8, N)``) and ``rps``
    (motor speeds Series, ``(4, M)`` in rev/s), aligned and value-calibrated
    with the measured per-recording constants (`MICHAELS_FILES` +
    `MICHAELS_RPS_SCALE`).
    """
    from data_processing.streams import resolve_source

    root = resolve_source(data_root) if data_root is not None else _DATA_ROOT
    frames: list[td.Frame] = []
    for wav_rel, csv_rel, time_offset, time_dilation in MICHAELS_FILES:
        frames.append(
            load_michaels_timeframe(
                root / wav_rel,
                root / csv_rel,
                time_offset=time_offset,
                time_dilation=time_dilation,
                sr=sr,
            )
        )
    return frames
