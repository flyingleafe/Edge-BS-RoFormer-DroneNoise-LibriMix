"""Torch ``Dataset``/``IterableDataset`` adapters that yield per-sample ``td.Frame``s.

The unified training loop (``src/training/loop.py``) is Frame-in throughout —
every dataset, map-style or iterable, yields one ``td.Frame`` per sample (no
``"batch"`` dim) and gets stacked by ``data_processing.collate.frame_collate``.
This module holds the two concrete adapters wired into ``conf/data/``:

- :class:`DregonLMFrameDataset` — a thin re-implementation of
  ``train_rps_predictor.py``'s ``DREGONRPSDataset`` (folder of
  ``sample_*/{mixture.wav,rps.npy}``, RPS resampled to the STFT frame grid by
  the same shape-stretch ``F.interpolate`` the original used — see that
  class's docstring for the alignment caveat this inherits) that returns a
  ``td.Frame`` instead of a raw ``(audio, rps)`` tensor pair, and also
  attaches per-sample ``metadata.json`` fields (``input_snr`` etc., when
  present) under ``"meta"``.
- :class:`OnlineMixFrameDataset` — wraps
  ``data_processing.online_mixing.OnlineMixIterableDataset`` (not modified;
  its ``(audio, rps)``-tensor-stream interface is a public contract used
  elsewhere) and packs each yielded pair into a ``td.Frame``. The online
  mixer's public interface has no per-sample metadata beyond the tensors
  themselves, so ``"meta"`` is an empty nested Frame here.

Both datasets normalize mono audio (1 channel) to a ``(time,)`` Series and
multichannel audio to ``(mic, time)``, matching ``tasks.task``'s
``n_channels=None`` vs ``n_channels=C`` convention.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import tdseries as td
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from data_processing.online_mixing import OnlineMixIterableDataset

__all__ = ["DregonLMFrameDataset", "OnlineMixFrameDataset"]

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_SAMPLE_RATE = 16000


def _audio_series(audio: np.ndarray, sample_rate: int) -> td.Series:
    """``(C, T)`` -> mono ``(time,)`` Series (``C == 1``) or ``(mic, time)``."""
    if audio.shape[0] == 1:
        return td.uniform(audio[0], sample_rate, dims=("time",), t_start=0.0)
    return td.uniform(audio, sample_rate, dims=("mic", "time"), t_start=0.0)


def _rps_series(rps: np.ndarray, *, sample_rate: int, hop_length: int) -> td.Series:
    """``(rotor, n_frames)`` array -> Series on the exact ``sr/hop`` STFT grid."""
    n_frames = rps.shape[-1]
    idx = td.GridIndex.create((sample_rate, hop_length), n_frames, t_start=0)
    return td.Series(rps, ("rotor", "time"), {"time": idx})


class DregonLMFrameDataset(Dataset):
    """Map-style dataset over a DREGON-LM-style split directory.

    ``data_dir`` is a split folder (e.g. ``datasets/DREGON-LM-V4/train``)
    containing ``sample_*/`` subdirectories, each with ``mixture.wav`` and
    ``rps.npy``. Per-sample metadata is read once from the sibling
    ``metadata.json`` (``{"train": [{"id": ..., "input_snr": ...}, ...],
    "valid": [...]}}`` — see ``create_dregon_librimix.py``) and merged under
    ``"meta"``; absent when no ``metadata.json`` exists.

    ``channel``, when set, selects one mic channel (``audio[channel]``) from
    a multichannel recording, producing a genuinely mono ``(time,)`` Frame —
    for mono-only models (e.g. ``simple_conv_v2``) trained against a
    multichannel dataset like DREGON-LM-V4 without the legacy
    channel-as-extra-batch-item flattening (``train_rps_predictor.py``'s
    ``_flatten_channels``, not reproduced here — see
    ``data_processing/AGENTS.md`` § "Multichannel Training & Evaluation
    Wiring"). ``None`` (default) keeps every channel, dims ``(mic, time)``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channel: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.channel = channel
        self.samples = sorted(
            Path(d)
            for d in glob.glob(os.path.join(str(self.data_dir), "sample_*"))
            if os.path.isfile(os.path.join(d, "mixture.wav"))
            and os.path.isfile(os.path.join(d, "rps.npy"))
        )
        if not self.samples:
            raise ValueError(f"no sample_* directories with mixture.wav+rps.npy under {data_dir}")
        self._meta = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        meta_path = self.data_dir.parent / "metadata.json"
        if not meta_path.is_file():
            return {}
        data = json.loads(meta_path.read_text())
        items = data.get(self.data_dir.name, [])
        return {item["id"]: item for item in items if "id" in item}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> td.Frame:
        d = self.samples[idx]
        raw, sr = sf.read(d / "mixture.wav", dtype="float32", always_2d=True)  # (T, C)
        if int(sr) != self.sample_rate:
            raise ValueError(
                f"{d}: mixture.wav sr={sr} != configured sample_rate={self.sample_rate}"
            )
        audio = np.ascontiguousarray(raw.T)  # (C, T)
        if self.channel is not None:
            audio = audio[self.channel : self.channel + 1]  # (1, T) -> mono via _audio_series

        rps_raw = np.load(d / "rps.npy").astype(np.float32)  # (rotor, rps_T)
        n_frames = audio.shape[-1] // self.hop_length + 1
        # Shape-stretch resample (endpoint-to-endpoint), matching
        # train_rps_predictor.py::DREGONRPSDataset — see class docstring.
        rps = (
            F.interpolate(
                torch.from_numpy(rps_raw).unsqueeze(0),
                size=n_frames,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )

        meta = dict(self._meta.get(d.name, {}))
        meta.setdefault("recording_id", d.name)

        return td.Frame(
            {
                "mixture": _audio_series(audio, self.sample_rate),
                "rps": _rps_series(rps, sample_rate=self.sample_rate, hop_length=self.hop_length),
                "meta": td.Frame(meta),
            }
        )


class OnlineMixFrameDataset(IterableDataset):
    """Wraps :class:`~data_processing.online_mixing.OnlineMixIterableDataset`,
    packing each ``(audio, rps)`` tensor pair into a per-sample ``td.Frame``."""

    def __init__(self, inner: OnlineMixIterableDataset) -> None:
        self.inner = inner

    @classmethod
    def from_config(cls, cfg: Any) -> OnlineMixFrameDataset:
        return cls(OnlineMixIterableDataset.from_config(cfg))

    @classmethod
    def from_yaml(cls, path: str) -> OnlineMixFrameDataset:
        """Load an online-mix policy YAML (e.g. ``configs/online_mix_*.yaml``)
        and build the dataset from it — the ``_target_`` this module's
        ``conf/data/online_mix_*.yaml`` configs actually use, since a Hydra
        component config carries a config *path*, not an inlined policy tree."""
        from omegaconf import OmegaConf

        return cls.from_config(OmegaConf.load(path))

    def __iter__(self):
        for audio, rps in self.inner:
            yield self._pack(audio, rps)

    def _pack(self, audio: torch.Tensor, rps: torch.Tensor) -> td.Frame:
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
        rps_np = rps.numpy() if isinstance(rps, torch.Tensor) else np.asarray(rps)
        return td.Frame(
            {
                "mixture": _audio_series(audio_np, self.inner.sample_rate),
                "rps": _rps_series(
                    rps_np, sample_rate=self.inner.sample_rate, hop_length=self.inner.hop_length
                ),
                "meta": td.Frame({}),
            }
        )
