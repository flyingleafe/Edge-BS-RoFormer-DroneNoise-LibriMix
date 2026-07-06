"""Michael's drone-noise dataset — 8-channel audio + aligned rotor speeds.

The recordings live in `data/recording_with_motor_speed/`:
  - recording_1/124.wav + FLY124.csv
  - recording_2/125.wav + FLY125.csv

Each WAV is an 8-channel in-flight drone recording; each DJI flight-controller
CSV logs per-motor rotation speed (RPM). The audio and telemetry clocks are
*not* aligned by default — a per-file `time_offset` (seconds) plus a
`time_dilation` clock-rate factor brings them into register. These constants
were tuned **manually** (see `notebooks/michael_data_analysis.ipynb`) and are
the values reproduced here verbatim.

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

# (wav_path, csv_path, time_offset_sec, time_dilation) — manual alignment
# constants, copied verbatim from `michael_data_analysis.ipynb`.
MICHAELS_FILES = [
    (
        "recording_with_motor_speed/recording_1/124.wav",
        "recording_with_motor_speed/recording_1/FLY124.csv",
        -20.84,
        1.001,
    ),
    (
        "recording_with_motor_speed/recording_2/125.wav",
        "recording_with_motor_speed/recording_2/FLY125.csv",
        -26.51,
        1.0048,
    ),
]


# ── Array / airframe geometry ───────────────────────────────────────────────
# Body frame: X = forward, Y = left, Z = up; origin at the drone body centre
# (rotor plane). The microphone array is a vertical ring (plane = Y-Z, normal
# +X) mounted forward of and above the body. Constants read from
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

    Microphones: 8 evenly-spaced points on a vertical ring (radius 82.5 mm),
    numbered counter-clockwise starting 22.5° left of top, with the ring centre
    ``ARRAY_OFFSET_FORWARD`` forward (the spec's 20 cm front-edge gap + the body's
    forward half-extent) and 330 mm above the body.

    Rotors: X-quad, arms at ±45°, motor radius ``(W/2)·cos45°`` per horizontal
    axis, at body height (z = 0). Ordered to match the ``rps`` rows
    (``ROTOR_ORDER``: RFront, LFront, LBack, RBack).
    """
    # Microphones — ring in the Y-Z plane (u = lateral -> +Y, v = vertical -> +Z).
    theta = np.deg2rad(112.5 + 45.0 * np.arange(N_MICS))  # mic 1 at 112.5°, CCW
    u = MIC_ARRAY_RADIUS * np.cos(theta)
    v = MIC_ARRAY_RADIUS * np.sin(theta)
    mic_positions = np.stack(
        [np.full(N_MICS, ARRAY_OFFSET_FORWARD), u, ARRAY_OFFSET_UP + v], axis=-1
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load + align one recording — verbatim port of the notebook function.

    Returns
    -------
    wav   : (n_channels, n_samples) audio at `sr`
    ts    : (M,) aligned motor timestamps (seconds, anchored at audio start)
    ms    : (4, M) motor speeds in revolutions/second (RPM / 60)
    sr    : sample rate actually used
    """
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

    ms = np.asarray(cut_csv[ms_cols], dtype=np.float64).T / 60
    return wav, ts, ms, int(sample_rate)


def load_michaels_timeframe(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float = 0.0,
    time_dilation: float = 1.0,
    sr: int | None = 16000,
    recording_id: str | None = None,
) -> td.Frame:
    """Load one Michael's recording as an aligned ``td.Frame``.

    The frame holds an 8-channel ``audio`` Series (dims ``("mic", "time")``)
    and an ``rps`` Series (dims ``("rotor", "time")``), built exactly as in
    the notebook — the audio is anchored at t=0 and the RPS timestamps are
    aligned to it via the `time_offset` / `time_dilation` constants.
    """
    wav, ts, ms, sample_rate = _load_michaels_data_raw(
        wav_path, csv_path, time_offset=time_offset, time_dilation=time_dilation, sr=sr
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
    (motor speeds Series, ``(4, M)`` in rev/s), aligned exactly as in
    `notebooks/michael_data_analysis.ipynb`.
    """
    root = Path(data_root) if data_root is not None else _DATA_ROOT
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
