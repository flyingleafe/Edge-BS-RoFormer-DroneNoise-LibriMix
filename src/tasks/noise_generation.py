"""Noise-generation task module — the inverse of RPS prediction.

Where RPS prediction maps *audio -> rotor speeds*, noise generation maps
*rotor speeds + array geometry -> the drone noise observed at each microphone*.
The model is a position-aware synthesiser (see
``models.generative.PositionalHarmonicNoiseGen``): per-rotor RPS drives a
harmonic + filtered-noise emitter, and the array geometry places each rotor as
a point source and renders it at every mic (1/r attenuation + propagation
delay), summing over rotors.

Geometry convention (the project's ``tdseries.Frame`` carries it natively)
-------------------------------------------------------------------------
Microphone and rotor positions are **non-temporal** array metadata, so they
live in dedicated Frame entries ``"mic_pos"`` (``(M, 3)``, sharing the
``"mic"`` dim with the audio track) and ``"rotor_pos"`` (``(R, 3)``, sharing
the ``"rotor"`` dim with the ``"rps"`` track) — exactly as
``data_processing.dregon`` already populates them. The model consumes the
per-(mic, rotor) relative vector ``rel_pos[m, r] = mic_positions[m] -
rotor_positions[r]`` built by :func:`geometry_to_rel_pos`.

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
import tdseries as td
import torch

from data_processing.frames import with_meta

# ── Constants ─────────────────────────────────────────────────────────────

SR_AUDIO: float = 16000.0
N_ROTORS: int = 4
SPEED_OF_SOUND: float = 343.0


# ── Generator protocol ────────────────────────────────────────────────────


@runtime_checkable
class NoiseGenerator(Protocol):
    """Structural interface for position-aware noise generation.

    ``forward(rps, rel_pos, z=None) -> audio`` with:
    * ``rps``     : ``(B, R, T)`` per-rotor speed at audio rate (Hz)
    * ``rel_pos`` : ``(B, M, R, 3)`` rotor->mic vectors (metres)
    * ``z``       : ``(B, d)`` optional external per-drone conditioning code
      (from a :class:`DroneCodebook`); required iff the model was built with
      ``cond_dim > 0``
    * returns     : ``(B, M, T)`` noise at each microphone
    """

    def forward(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


# ── Frame input-set loader ────────────────────────────────────────────────


def load_input_set(
    path: str | Path,
    mic_positions: np.ndarray,
    rotor_positions: np.ndarray,
    *,
    target_file: str = "noise.wav",
    sr: float = SR_AUDIO,
) -> Iterator[td.Frame]:
    """Load DREGON-LM-style chunks as ``Frame``s for noise-generation eval.

    Each yielded frame has entries ``"audio"`` (clean noise), ``"rps"``,
    ``"mic_pos"`` (``(M, 3)``, dim ``"mic"``) and ``"rotor_pos"`` (``(R, 3)``,
    dim ``"rotor"``) — the geometry the generator needs. Mirrors
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

    mic_pos_series = td.wrap(np.asarray(mic_positions, dtype=np.float64), dims=("mic", None))
    rotor_pos_series = td.wrap(np.asarray(rotor_positions, dtype=np.float64), dims=("rotor", None))

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
            audio_dims = ("time",)
        else:
            audio_dims = ("mic", "time")

        rps_raw = np.load(str(rps_path)).astype(np.float64)  # (R, M)
        audio_dur_s = audio.shape[-1] / file_sr
        n_motor = rps_raw.shape[-1]
        motor_times = np.arange(n_motor) / (n_motor / audio_dur_s if audio_dur_s > 0 else 1000.0)
        rps_series = td.events(
            motor_times, rps_raw, dims=("rotor", "time"), t_start=0.0, t_end=audio_dur_s
        )
        audio_series = td.uniform(audio, file_sr, dims=audio_dims, t_start=0.0)

        frame = td.Frame(
            {
                "audio": audio_series,
                "rps": rps_series,
                "mic_pos": mic_pos_series,
                "rotor_pos": rotor_pos_series,
            }
        )
        yield with_meta(frame, id=sample_dir.name)
