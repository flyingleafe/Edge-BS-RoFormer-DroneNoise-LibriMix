"""Noise-generation task module — the inverse of RPS prediction.

Where RPS prediction maps *audio -> rotor speeds*, noise generation maps
*rotor speeds + array geometry -> the drone noise observed at each microphone*.
The model is a position-aware synthesiser (see
``models.generative.PositionalHarmonicNoiseGen``): per-rotor RPS drives a
harmonic + filtered-noise emitter, and the array geometry places each rotor as
a point source and renders it at every mic (1/r attenuation + propagation
delay), summing over rotors.

Geometry convention (the project's `TimeFrame` carries it natively)
-------------------------------------------------------------------
Microphone and rotor positions are **non-temporal** array metadata, so they live
in ``TimeFrame.global_data`` (``mic_positions`` ``(M, 3)``,
``rotor_positions`` ``(R, 3)``) — exactly as ``data_processing.dregon`` already
populates them. The model consumes the per-(mic, rotor) relative vector
``rel_pos[m, r] = mic_positions[m] - rotor_positions[r]`` built by
:func:`geometry_to_rel_pos`.

The training/eval datasets reuse the **same on-disk format as RPS prediction**
(DREGON-LM ``sample_*`` chunks). The only differences: the target is the clean
``noise.wav`` (no speech mixing) and positions are attached from the recording
geometry.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from utils.data import EventSeries, TimeFrame, UniformSeries

# ── Constants ─────────────────────────────────────────────────────────────

SR_AUDIO: float = 16000.0
N_ROTORS: int = 4
SPEED_OF_SOUND: float = 343.0


# ── Geometry → model input ──────────────────────────────────────────────────


def geometry_to_rel_pos(
    mic_positions: np.ndarray,
    rotor_positions: np.ndarray,
) -> np.ndarray:
    """Build per-(mic, rotor) relative position vectors.

    Args:
        mic_positions: ``(M, 3)`` microphone xyz (metres).
        rotor_positions: ``(R, 3)`` rotor xyz (metres), same frame.

    Returns:
        ``(M, R, 3)`` float32 where ``rel_pos[m, r] = mic[m] - rotor[r]`` — the
        vector from rotor ``r`` to mic ``m`` that the generator propagates along.
    """
    mic = np.asarray(mic_positions, dtype=np.float64)
    rotor = np.asarray(rotor_positions, dtype=np.float64)
    if mic.ndim != 2 or mic.shape[-1] != 3:
        raise ValueError(f"mic_positions must be (M, 3), got {mic.shape}")
    if rotor.ndim != 2 or rotor.shape[-1] != 3:
        raise ValueError(f"rotor_positions must be (R, 3), got {rotor.shape}")
    return (mic[:, None, :] - rotor[None, :, :]).astype(np.float32)  # (M, R, 3)


# ── Generator protocol ────────────────────────────────────────────────────


@runtime_checkable
class NoiseGenerator(Protocol):
    """Structural interface for position-aware noise generation.

    ``forward(rps, rel_pos) -> audio`` with:
    * ``rps``     : ``(B, R, T)`` per-rotor speed at audio rate (Hz)
    * ``rel_pos`` : ``(B, M, R, 3)`` rotor->mic vectors (metres)
    * returns     : ``(B, M, T)`` noise at each microphone
    """

    def forward(self, rps: torch.Tensor, rel_pos: torch.Tensor) -> torch.Tensor: ...


# ── TimeFrame input-set loader ────────────────────────────────────────────


def load_input_set(
    path: str | Path,
    mic_positions: np.ndarray,
    rotor_positions: np.ndarray,
    *,
    target_file: str = "noise.wav",
    sr: float = SR_AUDIO,
) -> Iterator[TimeFrame]:
    """Load DREGON-LM-style chunks as ``TimeFrame``s for noise-generation eval.

    Each yielded frame has tracks ``{"audio": UniformSeries (clean noise),
    "rps": EventSeries}`` and ``global_data = {"mic_positions",
    "rotor_positions"}`` — the geometry the generator needs. Mirrors
    :func:`tasks.rps_prediction.load_input_set` but targets the clean noise and
    carries positions.

    Args:
        path: directory of ``sample_*`` chunks.
        mic_positions / rotor_positions: array geometry (``(M,3)`` / ``(R,3)``).
        target_file: per-chunk clean-noise file (default ``noise.wav``).
        sr: expected audio sample rate.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    import torchaudio

    global_data = {
        "mic_positions": np.asarray(mic_positions, dtype=np.float64),
        "rotor_positions": np.asarray(rotor_positions, dtype=np.float64),
    }

    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
            continue
        wav_path = sample_dir / target_file
        rps_path = sample_dir / "rps.npy"
        if not wav_path.is_file() or not rps_path.is_file():
            continue

        waveform, file_sr = torchaudio.load(str(wav_path))  # (C, T)
        if file_sr != sr:
            raise ValueError(f"Expected {sr} Hz audio, got {file_sr} in {wav_path}")
        audio = waveform.numpy().astype(np.float32)
        if audio.shape[0] == 1:
            audio = audio[0]  # (T,) mono

        rps_raw = np.load(str(rps_path)).astype(np.float64)  # (R, M)
        audio_dur_s = audio.shape[-1] / file_sr
        n_motor = rps_raw.shape[-1]
        motor_times = np.arange(n_motor) / (n_motor / audio_dur_s if audio_dur_s > 0 else 1000.0)
        rps_es = EventSeries.from_events(
            timestamps=motor_times, values=rps_raw, t_start=0.0, t_end=audio_dur_s
        )
        audio_us = UniformSeries.from_samples(audio, sr=file_sr, t_start=0.0)

        yield TimeFrame.from_tracks(
            {"audio": audio_us, "rps": rps_es},
            tags={"id": sample_dir.name},
            global_data=global_data,
        )
