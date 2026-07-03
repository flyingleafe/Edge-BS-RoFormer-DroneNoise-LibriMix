"""Tiny synthetic rps_prediction dataset + model for the training-framework
tests (collate/validate/loop). Referenced by dotted path
(``tests.training._fixtures.TinyRPSFrameDataset`` /
``tests.training._fixtures.TinyRPSModel``) from hand-built configs, so
:func:`training.config.build_dataset` / :func:`training.config.instantiate_model`
exercise the exact same ``_target_`` dispatch a real Hydra run would use.
"""

from __future__ import annotations

import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

__all__ = ["TinyRPSFrameDataset", "TinyRPSModel", "make_tiny_frame"]


def make_tiny_frame(
    *,
    recording_id: str,
    input_snr: float,
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    hop_length: int = 512,
    num_rotors: int = 4,
    rng: np.random.Generator | None = None,
) -> td.Frame:
    """One deterministic synthetic ``rps_prediction`` sample Frame."""
    rng = rng or np.random.default_rng(0)
    n_audio = int(round(duration_s * sample_rate))
    audio = rng.standard_normal(n_audio).astype(np.float32) * 0.1
    n_frames = n_audio // hop_length + 1
    rps = rng.uniform(20.0, 100.0, size=(num_rotors, n_frames)).astype(np.float32)

    mixture = td.uniform(audio, sample_rate, dims=("time",), t_start=0.0)
    rps_idx = td.GridIndex.create((sample_rate, hop_length), n_frames, t_start=0)
    rps_series = td.Series(rps, ("rotor", "time"), {"time": rps_idx})
    meta = td.Frame({"recording_id": recording_id, "input_snr": float(input_snr)})
    return td.Frame({"mixture": mixture, "rps": rps_series, "meta": meta})


class TinyRPSFrameDataset(Dataset):
    """Map-style dataset of ``n_samples`` deterministic synthetic Frames."""

    def __init__(
        self,
        *,
        n_samples: int = 8,
        duration_s: float = 0.5,
        sample_rate: int = 16000,
        hop_length: int = 512,
        num_rotors: int = 4,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        rng = np.random.default_rng(seed)
        self._frames = [
            make_tiny_frame(
                recording_id=f"tiny_{i}",
                input_snr=float(-30 + 5 * (i % 7)),
                duration_s=duration_s,
                sample_rate=sample_rate,
                hop_length=hop_length,
                num_rotors=num_rotors,
                rng=rng,
            )
            for i in range(self.n_samples)
        ]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> td.Frame:
        return self._frames[idx]


class TinyRPSModel(nn.Module):
    """2-layer Conv1d RPS predictor: mono ``(B, T)`` audio -> ``(B, num_rotors,
    T // hop_length + 1)`` — small enough for a CPU smoke test, no front-end
    dependency (unlike the real ``SimpleConv*`` family)."""

    def __init__(self, *, hop_length: int = 512, num_rotors: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.conv1 = nn.Conv1d(
            1, hidden, kernel_size=hop_length * 2, stride=hop_length, padding=hop_length
        )
        self.conv2 = nn.Conv1d(hidden, num_rotors, kernel_size=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        n_frames = audio.shape[-1] // self.hop_length + 1
        h = F.relu(self.conv1(audio.unsqueeze(1)))
        out = self.conv2(h)
        if out.shape[-1] != n_frames:
            out = F.interpolate(out, size=n_frames, mode="linear", align_corners=False)
        return out
