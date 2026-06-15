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

`load_michaels_timeframes()` returns a list of two `TimeFrame`s, one per
recording, each holding:
  - `audio`: 8-channel `UniformSeries` `(8, N)` at `sr`
  - `rps`  : `EventSeries` of motor speeds `(4, M)` in revolutions/second

Audio and RPS share the frame's time anchor, so they are aligned: slicing the
frame slices both consistently.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa as lr
import numpy as np
import pandas as pd

from utils.data import EventSeries, TimeFrame, UniformSeries

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
    ts = cut_csv[t_col].values

    if ts[0] > time_offset:
        wav = wav[:, int((ts[0] - time_offset) * sample_rate) :]
        time_offset = ts[0]
        wav_duration = wav.shape[-1] / sample_rate

    if ts[-1] < wav_duration + time_offset:
        wav = wav[:, : int((ts[-1] - ts[0]) * sample_rate)]

    jump_idx = np.argmax(np.diff(ts))
    ts[0 : jump_idx + 2] = np.linspace(ts[0], ts[jump_idx + 2], jump_idx + 2)
    ts *= time_dilation

    ms = cut_csv[ms_cols].values.T / 60
    return wav, ts, ms, sample_rate


def load_michaels_timeframe(
    wav_path: str | Path,
    csv_path: str | Path,
    time_offset: float = 0.0,
    time_dilation: float = 1.0,
    sr: int | None = 16000,
    recording_id: str | None = None,
) -> TimeFrame:
    """Load one Michael's recording as an aligned `TimeFrame`.

    The frame holds an 8-channel `audio` `UniformSeries` and an `rps`
    `EventSeries` (4 rotors), built exactly as in the notebook — the audio is
    anchored at t=0 and the RPS timestamps are aligned to it via the
    `time_offset` / `time_dilation` constants.
    """
    wav, ts, ms, sample_rate = _load_michaels_data_raw(
        wav_path, csv_path, time_offset=time_offset, time_dilation=time_dilation, sr=sr
    )
    audio = UniformSeries.from_samples(samples=wav, sr=sample_rate)
    rps = EventSeries.from_events(timestamps=ts, values=ms)
    tags = {"recording_id": recording_id or Path(wav_path).stem}
    return TimeFrame.from_tracks(dict(audio=audio, rps=rps), tags=tags)


def load_michaels_timeframes(
    data_root: str | Path | None = None,
    sr: int | None = 16000,
) -> list[TimeFrame]:
    """Load FLY124 and FLY125 as a list of two aligned 8-channel `TimeFrame`s.

    Each frame contains `audio` (8-channel `UniformSeries`, `(8, N)`) and `rps`
    (motor speeds `EventSeries`, `(4, M)` in rev/s), aligned exactly as in
    `notebooks/michael_data_analysis.ipynb`.
    """
    root = Path(data_root) if data_root is not None else _DATA_ROOT
    frames: list[TimeFrame] = []
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
