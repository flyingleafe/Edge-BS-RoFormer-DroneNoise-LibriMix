"""Torch ``Dataset``/``IterableDataset`` adapters that yield per-sample ``td.Frame``s.

The unified training loop (``src/training/loop.py``) is Frame-in throughout —
every dataset, map-style or iterable, yields one ``td.Frame`` per sample (no
``"batch"`` dim) and gets stacked by ``data_processing.collate.frame_collate``.
This module holds the concrete adapters wired into ``conf/data/``:

- :class:`DregonLMFrameDataset` — a thin re-implementation of
  ``train_rps_predictor.py``'s ``DREGONRPSDataset`` (folder of
  ``sample_*/{mixture.wav,rps.npy}``, RPS resampled to the STFT frame grid by
  the same shape-stretch ``F.interpolate`` the original used — see that
  class's docstring for the alignment caveat this inherits) that returns a
  ``td.Frame`` instead of a raw ``(audio, rps)`` tensor pair, and also
  attaches per-sample ``metadata.json`` fields (``input_snr`` etc., when
  present) under ``"meta"``.
- :class:`DNLMFrameDataset` — the DN-LM (DroneNoise-LibriMix, Paper 1)
  analogue for ``speech_enhancement``: folder of
  ``sample_*/{mixture.wav,vocals.wav,noise.wav}`` (no ``rps.npy`` — DN-LM
  predates per-sample RPS labels), emitting ``{"mixture", "target", "meta"}``
  (``"target"`` = clean ``vocals.wav``, matching ``losses.MaskedLoss``'s
  default ``target_key``). ``scripts/create_dataset.py`` writes one
  ``metadata.json`` **inside each split directory** (``{"train": [...]}`` /
  ``{"valid": [...]}``), unlike ``DregonLMFrameDataset``'s sibling-of-both-
  splits layout — see :meth:`DNLMFrameDataset._load_metadata`.
- :class:`OnlineMixFrameDataset` — wraps
  ``data_processing.online_mixing.OnlineMixIterableDataset`` (not modified;
  its ``(audio, rps)``-tensor-stream interface is a public contract used
  elsewhere) and packs each yielded pair into a ``td.Frame``. The online
  mixer's public interface has no per-sample metadata beyond the tensors
  themselves, so ``"meta"`` is an empty nested Frame here — except with
  ``flatten_channels=True``, which expands each multichannel chunk into
  per-mic mono Frames tagged ``meta.channel`` (the legacy training-loop
  flatten semantics, mirroring ``DregonLMFrameDataset``'s flag).
- :class:`NoiseGenFrameDataset` — wraps
  ``data_processing.noise_rps_dataset.NoiseRPSDataset``/
  ``build_noise_rps_datasets`` (DREGON `in_flight_noise` + Michael's chunk
  source, not modified) for the ``noise_generation`` task: attaches the
  recording's array geometry (``mic_pos``/``rotor_pos``) and a
  ``meta.drone`` name (``"dregon"``/``"michaels"``, from the inner
  dataset's per-draw ``origin``) to each ``(rps, audio)`` chunk. See its own
  docstring for the single-microphone limitation this inherits from
  ``NoiseRPSDataset``.

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
from torch.utils.data import Dataset, IterableDataset

from data_processing.frames import audio_series as _audio_series
from data_processing.frames import get_meta
from data_processing.frames import rps_series as _rps_series
from data_processing.online_mixing import OnlineMixIterableDataset
from data_processing.streams import (
    iter_published_frames,
    local_repository,
    resolve_source,
    stretch_rps_to_frames,
)

__all__ = [
    "DregonLMFrameDataset",
    "DNLMFrameDataset",
    "OnlineMixFrameDataset",
    "NoiseGenFrameDataset",
    "SEValidFrameDataset",
]

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_SAMPLE_RATE = 16000


class DregonLMFrameDataset(Dataset):
    """Map-style dataset over a DREGON-LM-style split directory.

    ``data_dir`` is a split folder (e.g. ``datasets/DREGON-LM-V4/train``)
    containing ``sample_*/`` subdirectories, each with ``mixture.wav`` and
    ``rps.npy``. Per-sample metadata is read once from the sibling
    ``metadata.json`` (``{"train": [{"id": ..., "input_snr": ...}, ...],
    "valid": [...]}}`` — see ``scripts/create_dregon_librimix.py``) and merged under
    ``"meta"``; absent when no ``metadata.json`` exists.

    ``channel``, when set, selects one mic channel (``audio[channel]``) from
    a multichannel recording, producing a genuinely mono ``(time,)`` Frame —
    for mono-only models (e.g. ``simple_conv_v2``) trained against a
    multichannel dataset like DREGON-LM-V4 without the legacy
    channel-as-extra-batch-item flattening. ``None`` (default) keeps every
    channel, dims ``(mic, time)``.

    ``flatten_channels=True`` instead *reproduces* that legacy
    ``train_rps_predictor.py::_flatten_channels`` scheme at the data level:
    each multichannel sample expands into ``n_channels`` separate mono-view
    Frames (one per mic, index space becomes ``len(samples) *
    n_channels``), each broadcasting the sample's single ``(rotor, T_stft)``
    RPS target — matching the legacy ``(B, C, T) -> (B*C, T)`` batch-flatten
    trick. ``meta`` gains a ``"channel"`` key recording which mic each
    flattened item came from. Mutually exclusive with ``channel=<int>``
    (that already selects one channel deterministically). See
    ``data_processing/AGENTS.md`` § "Multichannel Training & Evaluation
    Wiring" and REPLICATION.md § C9.

    ``rps_corruption`` (a mapping, default ``None`` = off) arms the
    conditional-refiner data seam (``data_processing.rps_corruption``): each
    Frame additionally carries ``"rps_cond"`` — a corrupted copy of the RPS
    target — and ``"rps"`` becomes the GT *in conditioning order* (row-
    permuted only when the corruption's swap event fired; invisible to PIT
    metrics). The corruption is seeded from the dataset index, so a map-style
    validation set gets a FIXED corruption per sample across epochs.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channel: int | None = None,
        flatten_channels: bool = False,
        rps_corruption: dict[str, Any] | None = None,
    ) -> None:
        # `dload:NAME[/subpath]` URIs materialize to a local tree first
        # (data_processing.streams.resolve_source); plain paths pass through.
        self.data_dir = resolve_source(data_dir)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.channel = channel
        self.flatten_channels = flatten_channels
        from data_processing.rps_corruption import RPSCorruption

        self._corruption = RPSCorruption.from_config(
            rps_corruption, frame_rate_hz=self.sample_rate / self.hop_length
        )
        self.samples = sorted(
            Path(d)
            for d in glob.glob(os.path.join(str(self.data_dir), "sample_*"))
            if os.path.isfile(os.path.join(d, "mixture.wav"))
            and os.path.isfile(os.path.join(d, "rps.npy"))
        )
        if not self.samples:
            raise ValueError(f"no sample_* directories with mixture.wav+rps.npy under {data_dir}")
        self._meta = self._load_metadata()

        self._n_channels_per_sample = 1
        if self.flatten_channels:
            if self.channel is not None:
                raise ValueError(
                    "flatten_channels=True is incompatible with channel=<int> "
                    "(channel already selects one mic deterministically)"
                )
            info = sf.info(str(self.samples[0] / "mixture.wav"))
            self._n_channels_per_sample = info.channels
            if self._n_channels_per_sample <= 1:
                raise ValueError(
                    "flatten_channels=True needs multichannel mixture.wav files "
                    f"(>1 channel); {self.samples[0]} has {self._n_channels_per_sample}"
                )

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        meta_path = self.data_dir.parent / "metadata.json"
        if not meta_path.is_file():
            return {}
        data = json.loads(meta_path.read_text())
        items = data.get(self.data_dir.name, [])
        return {item["id"]: item for item in items if "id" in item}

    def __len__(self) -> int:
        return len(self.samples) * self._n_channels_per_sample

    def __getitem__(self, idx: int) -> td.Frame:
        if self.flatten_channels:
            sample_idx, channel = divmod(idx, self._n_channels_per_sample)
        else:
            sample_idx, channel = idx, self.channel

        d = self.samples[sample_idx]
        raw, sr = sf.read(d / "mixture.wav", dtype="float32", always_2d=True)  # (T, C)
        if int(sr) != self.sample_rate:
            raise ValueError(
                f"{d}: mixture.wav sr={sr} != configured sample_rate={self.sample_rate}"
            )
        audio = np.ascontiguousarray(raw.T)  # (C, T)
        if channel is not None:
            audio = audio[channel : channel + 1]  # (1, T) -> mono via _audio_series

        rps_raw = np.load(d / "rps.npy").astype(np.float32)  # (rotor, rps_T)
        n_frames = audio.shape[-1] // self.hop_length + 1
        # Shape-stretch resample (endpoint-to-endpoint), matching
        # train_rps_predictor.py::DREGONRPSDataset — see class docstring.
        rps = stretch_rps_to_frames(rps_raw, n_frames)

        meta = dict(self._meta.get(d.name, {}))
        meta.setdefault("recording_id", d.name)
        if self.flatten_channels:
            meta["channel"] = channel

        entries: dict[str, Any] = {
            "mixture": _audio_series(audio, self.sample_rate),
            "rps": _rps_series(rps, sample_rate=self.sample_rate, hop_length=self.hop_length),
            "meta": td.Frame(meta),
        }
        if self._corruption is not None:
            # Deterministic per dataset index (a flattened index already
            # encodes the channel), so validation corruption is fixed.
            cond, gt_aligned = self._corruption(rps, idx)
            entries["rps"] = _rps_series(
                gt_aligned, sample_rate=self.sample_rate, hop_length=self.hop_length
            )
            entries["rps_cond"] = _rps_series(
                cond, sample_rate=self.sample_rate, hop_length=self.hop_length
            )
        return td.Frame(entries)


class DNLMFrameDataset(Dataset):
    """Map-style dataset over a DN-LM (DroneNoise-LibriMix, Paper 1) split directory.

    ``data_dir`` is a split folder (e.g. ``datasets/DN-LM/train``) containing
    ``sample_*/`` subdirectories, each with ``mixture.wav`` and ``vocals.wav``
    (the clean speech target; ``noise.wav`` also exists on disk but is not
    needed for ``speech_enhancement`` training). DN-LM predates per-sample RPS
    labels — there is no ``rps.npy`` and no ``"rps"`` Frame entry, matching
    ``tasks.task.speech_enhancement``'s ``use_rps=False`` default. Per-sample
    metadata (``input_snr``, ``speech_source``, ``noise_source``,
    ``speech_distance``) is read from ``scripts/create_dataset.py``'s per-split
    ``metadata.json`` (``{"<split>": [{"id": ..., ...}, ...]}``, written
    *inside* the split directory itself — unlike ``DregonLMFrameDataset``'s
    dataset-root-level file shared by both splits).
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.data_dir = resolve_source(data_dir)
        self.sample_rate = int(sample_rate)
        self.samples = sorted(
            Path(d)
            for d in glob.glob(os.path.join(str(self.data_dir), "sample_*"))
            if os.path.isfile(os.path.join(d, "mixture.wav"))
            and os.path.isfile(os.path.join(d, "vocals.wav"))
        )
        if not self.samples:
            raise ValueError(
                f"no sample_* directories with mixture.wav+vocals.wav under {data_dir}"
            )
        self._meta = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.is_file():
            return {}
        data = json.loads(meta_path.read_text())
        items = data.get(self.data_dir.name, [])
        return {item["id"]: item for item in items if "id" in item}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> td.Frame:
        d = self.samples[idx]
        mixture, sr = sf.read(d / "mixture.wav", dtype="float32", always_2d=True)  # (T, C)
        if int(sr) != self.sample_rate:
            raise ValueError(
                f"{d}: mixture.wav sr={sr} != configured sample_rate={self.sample_rate}"
            )
        vocals, sr_v = sf.read(d / "vocals.wav", dtype="float32", always_2d=True)  # (T, C)
        if int(sr_v) != self.sample_rate:
            raise ValueError(
                f"{d}: vocals.wav sr={sr_v} != configured sample_rate={self.sample_rate}"
            )

        meta = dict(self._meta.get(d.name, {}))
        meta.setdefault("recording_id", d.name)

        return td.Frame(
            {
                "mixture": _audio_series(np.ascontiguousarray(mixture.T), self.sample_rate),
                "target": _audio_series(np.ascontiguousarray(vocals.T), self.sample_rate),
                "meta": td.Frame(meta),
            }
        )


class SEValidFrameDataset(Dataset):
    """Map-style dataset over a published ``tdframe-v1`` **SE valid** set.

    The fixed speech-enhancement validation sets (``SE-valid-drone`` /
    ``SE-valid-harmonic``, built by ``scripts/build_se_valid.py``) publish one
    sample per mixture as a Frame carrying ``mixture`` (noisy) + ``target``
    (clean speech as mixed) audio Series and a ``meta`` sub-Frame with
    ``input_snr`` / ``category`` / ``noise_source`` / ``id``. These are exactly
    the entries the ``speech_enhancement`` codec (``mixture`` in) and
    ``losses.MaskedLoss`` / separation metrics (``target``) consume, and
    ``eval.py`` groups per-SNR on ``meta.input_snr``.

    The sets are small (a few hundred–few thousand short clips), so all frames
    are streamed once via :func:`iter_published_frames` and held in memory for
    O(1), worker-safe random access — portable to any backend (streams from R2
    via dload; no local checkout needed). ``category`` filters to one Pass-B
    category (used to score per-category transfer on ``SE-valid-harmonic``).

    ``local_root`` reads the set from a local (unpublished) dload repository
    built by ``scripts/build_se_valid.py --local-repo`` instead of R2 — the
    replication path for a valid set that is not (yet) published.
    """

    def __init__(
        self,
        dataset: str,
        *,
        version: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        category: str | None = None,
        local_root: str | None = None,
    ) -> None:
        self.dataset = str(dataset)
        self.sample_rate = int(sample_rate)
        self.category = category
        self.local_root = local_root
        repo = local_repository(local_root) if local_root else None
        self._frames: list[td.Frame] = []
        for frame in iter_published_frames(self.dataset, version, repo=repo):
            if "mixture" not in frame or "target" not in frame:
                continue
            if category is not None and str(get_meta(frame, "category", "")) != category:
                continue
            self._frames.append(frame)
        if not self._frames:
            raise ValueError(
                f"SE valid dataset {self.dataset!r} yielded no usable frames"
                + (f" for category={category!r}" if category else "")
            )

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> td.Frame:
        return self._frames[idx]


class OnlineMixFrameDataset(IterableDataset):
    """Wraps :class:`~data_processing.online_mixing.OnlineMixIterableDataset`,
    packing each ``(audio, rps)`` tensor pair into a per-sample ``td.Frame``.

    ``flatten_channels=True`` reproduces the legacy ``train_rps_predictor.py``
    channel-as-extra-batch-item semantics at the *data* level, mirroring
    :class:`DregonLMFrameDataset`'s flag: each multichannel ``(C, T)`` chunk is
    expanded into ``C`` mono-view Frames (one per mic, ``meta.channel`` tagged),
    all broadcasting the chunk's single RPS target. Mono ``(T,)`` chunks pass
    through unchanged. Mono RPS models (``('time',)`` input spec) require this
    over multichannel sources — without it validation fails on the
    ``(mic, time)`` mixture dims. Default ``False`` keeps the raw multichannel
    stream byte-identical to before.

    ``rps_corruption`` (mapping, default ``None`` = off) is the conditional-
    refiner seam (``data_processing.rps_corruption``, mirroring
    :class:`DregonLMFrameDataset`'s flag): every emitted Frame additionally
    carries ``"rps_cond"`` (a corrupted copy of the chunk's RPS target) and
    its ``"rps"`` becomes the GT in conditioning order. The corruption id is
    derived from the inner mixer's global chunk id (recomputed here in
    lockstep with ``OnlineMixIterableDataset.__iter__``'s worker-aware
    enumeration) plus the flattened channel index — deterministic per
    ``(base seed, chunk id, channel)``, independent of worker count for the
    chunk content (the mixer's own guarantee) and of nothing else.
    """

    #: sub-id stride for (chunk id, channel) -> corruption sample id; any
    #: bound comfortably above the physical channel count works.
    _COND_ID_STRIDE = 256

    def __init__(
        self,
        inner: OnlineMixIterableDataset,
        *,
        flatten_channels: bool = False,
        rps_corruption: dict[str, Any] | None = None,
    ) -> None:
        self.inner = inner
        self.flatten_channels = bool(flatten_channels)
        from data_processing.rps_corruption import RPSCorruption

        self._corruption = RPSCorruption.from_config(
            rps_corruption, frame_rate_hz=inner.sample_rate / inner.hop_length
        )
        if self._corruption is not None and self.inner.task != "rps_prediction":
            raise ValueError("rps_corruption requires the rps_prediction online-mix task")

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        flatten_channels: bool = False,
        rps_corruption: dict[str, Any] | None = None,
    ) -> OnlineMixFrameDataset:
        return cls(
            OnlineMixIterableDataset.from_config(cfg),
            flatten_channels=flatten_channels,
            rps_corruption=rps_corruption,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str,
        *,
        flatten_channels: bool = False,
        rps_corruption: dict[str, Any] | None = None,
    ) -> OnlineMixFrameDataset:
        """Load an online-mix policy YAML (e.g. ``conf/online_mix/online_mix_*.yaml``)
        and build the dataset from it — the ``_target_`` this module's
        ``conf/data/online_mix_*.yaml`` configs actually use, since a Hydra
        component config carries a config *path*, not an inlined policy tree.
        ``flatten_channels`` sits next to ``path`` in the Hydra ``params``."""
        from omegaconf import OmegaConf

        return cls.from_config(
            OmegaConf.load(path),
            flatten_channels=flatten_channels,
            rps_corruption=rps_corruption,
        )

    def __iter__(self):
        if getattr(self.inner, "task", "rps_prediction") == "speech_enhancement":
            yield from self._iter_se()
            return
        from torch.utils.data import get_worker_info

        # Recompute the inner mixer's global chunk ids in lockstep with its
        # own worker-aware enumeration (OnlineMixIterableDataset.__iter__:
        # gid = start + worker_id + k * num_workers) — both run in the same
        # worker process, so the k-th chunk this iterator sees IS chunk gid.
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        start = int(getattr(self.inner, "start_sample_id", 0))
        for k, (audio, rps) in enumerate(self.inner):
            chunk_gid = start + worker_id + k * num_workers
            audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
            rps_np = rps.numpy() if isinstance(rps, torch.Tensor) else np.asarray(rps)
            if self.flatten_channels and audio_np.ndim == 2 and audio_np.shape[0] > 1:
                for channel in range(audio_np.shape[0]):
                    yield self._pack(
                        audio_np[channel : channel + 1],
                        rps_np,
                        meta={"channel": channel},
                        cond_id=chunk_gid * self._COND_ID_STRIDE + channel,
                    )
            else:
                yield self._pack(audio_np, rps_np, cond_id=chunk_gid * self._COND_ID_STRIDE)

    def _iter_se(self):
        """Speech-enhancement stream: the inner mixer yields ``(mixture, clean)``
        pairs; pack them into the ``{"mixture", "target"}`` Frame the SE task /
        ``losses.MaskedLoss`` consume (the DN-LM ``DNLMFrameDataset`` layout).
        ``flatten_channels`` is a no-op — the SE stream is single-channel."""
        for mixture, target in self.inner:
            mixture_np = (
                mixture.numpy() if isinstance(mixture, torch.Tensor) else np.asarray(mixture)
            )
            target_np = target.numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
            yield td.Frame(
                {
                    "mixture": _audio_series(mixture_np, self.inner.sample_rate),
                    "target": _audio_series(target_np, self.inner.sample_rate),
                    "meta": td.Frame({}),
                }
            )

    def _pack(
        self,
        audio_np: np.ndarray,
        rps_np: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
        cond_id: int = 0,
    ) -> td.Frame:
        entries: dict[str, Any] = {
            "mixture": _audio_series(audio_np, self.inner.sample_rate),
            "rps": _rps_series(
                rps_np, sample_rate=self.inner.sample_rate, hop_length=self.inner.hop_length
            ),
            "meta": td.Frame(dict(meta or {})),
        }
        if self._corruption is not None:
            cond, gt_aligned = self._corruption(rps_np, cond_id)
            entries["rps"] = _rps_series(
                gt_aligned, sample_rate=self.inner.sample_rate, hop_length=self.inner.hop_length
            )
            entries["rps_cond"] = _rps_series(
                cond, sample_rate=self.inner.sample_rate, hop_length=self.inner.hop_length
            )
        return td.Frame(entries)


def _resolve_noise_dir(d: str | Path | None) -> str | Path | None:
    """resolve_source, except ``frames:NAME[@VER]`` specs pass through verbatim.

    ``resolve_source`` would wrap the spec in a Path, defeating the
    string-prefix dispatch in ``noise_rps_dataset.build_noise_rps_datasets``
    (and the geometry dispatch above).
    """
    if d is None:
        return None
    from data_processing.noise_rps_dataset import FRAMES_SPEC_PREFIX

    if isinstance(d, str) and d.startswith(FRAMES_SPEC_PREFIX):
        return d
    return resolve_source(d)


_FRAMES_GEOMETRY_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _frames_spec_geometry(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """Geometry from a published ``frames:NAME[@VER]`` dataset.

    The tdframe-v1 recording frames carry ``mic_pos``/``rotor_pos`` entries
    baked in at publish time (``frames.make_recording_frame``), so a raw data
    tree (``micPos.txt``) is not needed — required for dload-streamed training
    on backends without a local checkout. Decoding a recording just for its
    geometry is wasteful, hence the module-level cache (train+valid datasets
    share it).
    """
    if spec not in _FRAMES_GEOMETRY_CACHE:
        from data_processing.noise_rps_dataset import _parse_frames_spec
        from data_processing.streams import iter_published_frames

        name, version = _parse_frames_spec(spec)
        for tf in iter_published_frames(name, version):
            if "mic_pos" in tf and "rotor_pos" in tf:
                _FRAMES_GEOMETRY_CACHE[spec] = (
                    np.asarray(tf["mic_pos"].data, dtype=np.float64),
                    np.asarray(tf["rotor_pos"].data, dtype=np.float64),
                )
                break
        else:
            raise ValueError(f"no recording in {spec!r} carries mic_pos/rotor_pos geometry")
    return _FRAMES_GEOMETRY_CACHE[spec]


def _noise_gen_geometry(
    origin: str, dregon_dir: str | Path | None
) -> tuple[np.ndarray, np.ndarray]:
    """``(mic_positions, rotor_positions)`` for a ``NoiseRPSDataset`` chunk origin."""
    if origin == "michaels":
        from data_processing import sources

        return sources.geometry("michaels")
    if origin == "dregon":
        if dregon_dir is None:
            raise ValueError("dregon_dir is required to build geometry for 'dregon'-origin chunks")
        from data_processing.noise_rps_dataset import FRAMES_SPEC_PREFIX

        if isinstance(dregon_dir, str) and dregon_dir.startswith(FRAMES_SPEC_PREFIX):
            return _frames_spec_geometry(dregon_dir)
        from data_processing import sources
        from data_processing.streams import resolve_source

        return sources.dregon.get_geometry(resolve_source(dregon_dir))
    raise ValueError(
        f"unknown NoiseRPSDataset chunk origin {origin!r}; expected 'dregon' or 'michaels'"
    )


class NoiseGenFrameDataset(Dataset):
    """Frame adapter around ``noise_rps_dataset.NoiseRPSDataset`` for the
    ``noise_generation`` task.

    Wraps the existing DREGON+Michael's chunkable noise+RPS source (reused,
    not rewritten — see ``data_processing/noise_rps_dataset.py``) and
    attaches what ``tasks.task.noise_generation``/
    ``tasks.codecs.NoiseGenerationCodec`` need beyond ``(rps, audio)``:
    array geometry (``mic_pos``/``rotor_pos``) and a ``meta.drone`` name
    (``"dregon"``/``"michaels"``, straight from the inner dataset's
    per-draw ``origin`` — already exactly the codebook key convention used
    by ``conf/online_mix/noise_gen_online_dregon_michaels*.yaml`` historically).

    **Single-microphone limitation**: ``NoiseRPSDataset`` already reduces
    each draw to one selected audio channel (``channel_policy``) without
    reporting *which* physical index was picked — so this adapter only
    supports ``channel_policy="first"`` (channel 0, deterministic): both
    the audio target and ``mic_pos`` are row 0 of the recording's geometry.
    This is a real reduction from the historical online-mixing noise-gen
    trainer, which rendered all 8 mics jointly (native multi-observer) —
    see REPLICATION.md § E2/E3 for the caveat; extending
    ``NoiseRPSDataset`` to report the drawn channel index is the natural
    follow-up if full multichannel noise-gen training is needed.
    """

    def __init__(
        self,
        inner: Any,
        *,
        dregon_dir: str | Path | None = None,
    ) -> None:
        from data_processing.noise_rps_dataset import NoiseRPSDataset

        if not isinstance(inner, NoiseRPSDataset):
            raise TypeError(f"NoiseGenFrameDataset wraps a NoiseRPSDataset, got {type(inner)}")
        if inner.channel_policy != "first":
            raise ValueError(
                "NoiseGenFrameDataset requires channel_policy='first' (mic index 0 "
                f"deterministic — see class docstring); got {inner.channel_policy!r}"
            )
        self.inner = inner
        self.dregon_dir = _resolve_noise_dir(dregon_dir)
        origins = {r.origin for r in inner.records}
        self._geometry = {o: _noise_gen_geometry(o, self.dregon_dir) for o in origins}

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> td.Frame:
        item = self.inner[idx]
        rps = np.asarray(item["rps"], dtype=np.float32)  # (rotor, T) Hz at audio rate
        audio = np.asarray(item["audio"], dtype=np.float32)[None, :]  # (1, T) -> single mic
        origin = str(item["origin"])
        mic_pos_full, rotor_pos = self._geometry[origin]
        mic_pos = np.asarray(mic_pos_full[:1], dtype=np.float32)  # (1, 3) — channel 0 only
        rotor_pos = np.asarray(rotor_pos, dtype=np.float32)
        sample_rate = self.inner.sample_rate

        return td.Frame(
            {
                "audio": td.uniform(audio, sample_rate, dims=("mic", "time"), t_start=0.0),
                "rps": td.uniform(rps, sample_rate, dims=("rotor", "time"), t_start=0.0),
                "mic_pos": td.wrap(mic_pos, dims=("mic", None)),
                "rotor_pos": td.wrap(rotor_pos, dims=("rotor", None)),
                "meta": td.Frame({"drone": origin}),
            }
        )

    @classmethod
    def build_train(
        cls,
        *,
        dregon_dir: str | Path | None = None,
        michaels_dir: str | Path | None = None,
        sample_rate: int = 16000,
        chunk_size: int = 16000,
        train_samples: int = 4096,
        val_samples: int = 512,
        val_pct: float = 0.1,
        val_at_start: bool = False,
        seed: int = 42,
        **dataset_kwargs: Any,
    ) -> NoiseGenFrameDataset:
        """Build the *train* split — see :func:`build_valid` for the pair."""
        from data_processing.noise_rps_dataset import build_noise_rps_datasets

        dregon_dir = _resolve_noise_dir(dregon_dir)
        michaels_dir = _resolve_noise_dir(michaels_dir)
        train_ds, _val_ds = build_noise_rps_datasets(
            dregon_dir=dregon_dir,
            michaels_dir=michaels_dir,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            train_samples=train_samples,
            val_samples=val_samples,
            val_pct=val_pct,
            val_at_start=val_at_start,
            seed=seed,
            channel_policy="first",
            **dataset_kwargs,
        )
        return cls(train_ds, dregon_dir=dregon_dir)

    @classmethod
    def build_valid(
        cls,
        *,
        dregon_dir: str | Path | None = None,
        michaels_dir: str | Path | None = None,
        sample_rate: int = 16000,
        chunk_size: int = 16000,
        train_samples: int = 4096,
        val_samples: int = 512,
        val_pct: float = 0.1,
        val_at_start: bool = False,
        seed: int = 42,
        **dataset_kwargs: Any,
    ) -> NoiseGenFrameDataset:
        """Build the *valid* split — same call as :meth:`build_train`, source
        loading happens twice (once per split-classmethod call) rather than
        sharing state, matching how every other ``conf/data/*.yaml`` in this
        module gives ``train:``/``valid:`` independent ``_target_`` specs."""
        from data_processing.noise_rps_dataset import build_noise_rps_datasets

        dregon_dir = _resolve_noise_dir(dregon_dir)
        michaels_dir = _resolve_noise_dir(michaels_dir)
        _train_ds, val_ds = build_noise_rps_datasets(
            dregon_dir=dregon_dir,
            michaels_dir=michaels_dir,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            train_samples=train_samples,
            val_samples=val_samples,
            val_pct=val_pct,
            val_at_start=val_at_start,
            seed=seed,
            channel_policy="first",
            **dataset_kwargs,
        )
        return cls(val_ds, dregon_dir=dregon_dir)
