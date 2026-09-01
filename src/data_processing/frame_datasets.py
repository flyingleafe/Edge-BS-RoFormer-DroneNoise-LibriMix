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
  default ``target_key``). The ``dn_lm`` derivation writes one
  ``metadata.json`` **inside each split directory** (``{"train": [...]}`` /
  ``{"valid": [...]}``), unlike ``DregonLMFrameDataset``'s sibling-of-both-
  splits layout — see :meth:`DNLMFrameDataset._load_metadata`.
- :class:`OnlineMixFrameDataset` — wraps the compiled online-mix pipeline
  (``data_processing.online_mixing.build_online_mix_pipeline``); the stream
  yields per-sample ``td.Frame``s directly, with ``meta.sample_id`` carrying
  the global chunk id. ``flatten_channels=True`` appends a ``flat_map``
  expanding each multichannel chunk into per-mic mono Frames tagged
  ``meta.channel`` (the legacy training-loop flatten semantics, mirroring
  ``DregonLMFrameDataset``'s flag).
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
import zlib
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import tdseries as td
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, IterableDataset

from data_processing.frames import audio_series as _audio_series
from data_processing.frames import get_meta, meta_dict
from data_processing.frames import rps_series as _rps_series
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
    "valid": [...]}}`` — see ``derivations.generate_dregon_lm_split``) and merged under
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
    ``speech_distance``) is read from the ``dn_lm`` derivation's per-split
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
    ``SE-valid-harmonic``, built by ``derivations.generate_se_valid``) publish one
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
    committed to a local repository (``streams.local_repository``) instead of R2 — the
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


def _flatten_frame_channels(frame: td.Frame) -> list[td.Frame]:
    """Expand a multichannel sample Frame into per-mic mono-view Frames.

    Each ``(mic, time)`` mixture becomes ``n_channels`` mono ``(time,)``
    Frames (one per mic, ``meta.channel`` tagged), all broadcasting the
    chunk's single RPS target — the legacy channel-as-extra-batch-item
    semantics at the data level. Mono frames pass through unchanged.
    """
    mix = frame["mixture"]
    if "mic" not in mix.dims or mix.dim_size("mic") <= 1:
        return [frame]
    sr = int(cast(td.GridIndex, mix.tindex).sr)
    gid = get_meta(frame, "sample_id", 0)
    data = np.asarray(mix.data)
    out = []
    for ch in range(mix.dim_size("mic")):
        entries: dict[str, Any] = {"mixture": _audio_series(data[ch : ch + 1], sr)}
        if "rps" in frame:
            entries["rps"] = frame["rps"]
        entries["meta"] = td.Frame({**meta_dict(frame), "sample_id": gid, "channel": ch})
        out.append(td.Frame(entries))
    return out


def _corrupt_frame(
    corruption: Any, sample_rate: int, hop_length: int, stride: int, frame: td.Frame
) -> td.Frame:
    """Apply the conditional-refiner RPS corruption (``rps_corruption.py``).

    The corruption id derives from the sample's own metadata (``sample_id``
    carried by the stream + ``meta.channel`` set by the flatten stage) —
    deterministic per ``(base seed, chunk id, channel)``.
    """
    gid = int(get_meta(frame, "sample_id", 0))
    channel = int(get_meta(frame, "channel", 0) or 0)
    rps_np = np.asarray(frame["rps"].data)
    cond, gt_aligned = corruption(rps_np, gid * stride + channel)
    frame = frame.with_entry(
        "rps", _rps_series(gt_aligned, sample_rate=sample_rate, hop_length=hop_length)
    )
    return frame.with_entry(
        "rps_cond", _rps_series(cond, sample_rate=sample_rate, hop_length=hop_length)
    )


def apply_speech_override(cfg: Any, speech: bool | None) -> Any:
    """Turn the speech pool of an online-mix policy on or off in place.

    The salv2 grid asks one question -- does mixed-in speech help or hurt the
    rotor-speed task -- and the only honest way to ask it is for the two arms to
    share every other byte of their policy. So the with/without switch lives
    here rather than in a second, hand-copied YAML that would drift.

    ``speech=False`` deletes ``sources.speech`` (which makes ``speech_present``
    False in ``online_mixing``, so no utterance is drawn at all) AND sets every
    stage's ``source_prob`` to 0. Either alone suffices; both together mean the
    intent survives a later reader of the merged config.
    """
    if speech is None:
        return cfg
    if speech:
        if "speech" not in cfg.sources:
            raise ValueError("policy has no sources.speech to enable")
        return cfg
    if "speech" in cfg.sources:
        del cfg.sources["speech"]
    for stage in cfg.policy.stages:
        stage.source_prob = 0.0
    return cfg


def apply_flight_reuse(cfg: Any, flight_reuse: int | None) -> Any:
    """Override ``sources.noise[*].rps.flight_reuse`` in place.

    The policy value (32) is a TRAINING economy: one full flight is generated
    and then windowed 32 times. On a fixed validation set that is a defect --
    ``n`` counts frames AFTER the per-mic flat_map, so with 8 mics n=96 is 12
    clips, and 12 < 32 makes all of them windows of ONE trajectory. Validation
    sets pass 1, and then every clip is its own flight.
    """
    if flight_reuse is None:
        return cfg
    for src in cfg.sources.noise:
        if "rps" in src:
            src.rps.flight_reuse = int(flight_reuse)
    return cfg


class OnlineMixFrameDataset(IterableDataset):
    """The online-mix training stream: a compiled policy pipeline as a torch
    IterableDataset.

    Thin wrapper over
    :func:`data_processing.online_mixing.build_online_mix_pipeline` (the
    policy YAML compiles to one infinite ``dload.Pipeline`` of per-sample
    Frames). ``flatten_channels=True`` appends the per-mic expansion as a
    ``flat_map`` stage; ``rps_corruption`` (the conditional-refiner seam,
    ``data_processing.rps_corruption``) appends the corruption as a ``map``
    stage reading ids from each frame's own metadata (``meta.sample_id`` +
    ``meta.channel``).
    """

    #: sub-id stride for (chunk id, channel) -> corruption sample id; any
    #: bound comfortably above the physical channel count works.
    _COND_ID_STRIDE = 256

    def __init__(
        self,
        cfg: Any,
        *,
        flatten_channels: bool = False,
        rps_corruption: dict[str, Any] | None = None,
    ) -> None:
        from dload.torch import as_iterable_dataset

        from data_processing.online_mixing import build_online_mix_pipeline
        from data_processing.rps_corruption import RPSCorruption

        plain = cast(
            dict[str, Any],
            OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg,
        )
        self.sample_rate = int(plain.get("sample_rate", 16000))
        self.hop_length = int(plain.get("hop_length", 512))
        self.task = str(plain.get("task", "rps_prediction"))
        self.start_sample_id = int(plain.get("start_sample_id", 0))
        self.base_seed = int(plain.get("base_seed", 1234))
        self.flatten_channels = bool(flatten_channels)
        self._corruption = RPSCorruption.from_config(
            rps_corruption, frame_rate_hz=self.sample_rate / self.hop_length
        )
        if self._corruption is not None and self.task != "rps_prediction":
            raise ValueError("rps_corruption requires the rps_prediction online-mix task")

        pipe = build_online_mix_pipeline(cfg)
        if self.flatten_channels and self.task != "speech_enhancement":
            pipe = pipe.flat_map(_flatten_frame_channels)
        if self._corruption is not None:
            pipe = pipe.map(
                partial(
                    _corrupt_frame,
                    self._corruption,
                    self.sample_rate,
                    self.hop_length,
                    self._COND_ID_STRIDE,
                )
            )
        self._inner = as_iterable_dataset(pipe)

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        flatten_channels: bool = False,
        rps_corruption: dict[str, Any] | None = None,
    ) -> OnlineMixFrameDataset:
        return cls(cfg, flatten_channels=flatten_channels, rps_corruption=rps_corruption)

    @classmethod
    def from_yaml(
        cls,
        path: str,
        *,
        flatten_channels: bool = False,
        rps_corruption: dict[str, Any] | None = None,
        speech: bool | None = None,
        flight_reuse: int | None = None,
    ) -> OnlineMixFrameDataset:
        """Load an online-mix policy YAML (e.g. ``conf/online_mix/online_mix_*.yaml``)
        and build the dataset from it — the ``_target_`` this module's
        ``conf/data/online_mix_*.yaml`` configs actually use, since a Hydra
        component config carries a config *path*, not an inlined policy tree.
        ``flatten_channels`` sits next to ``path`` in the Hydra ``params``."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        apply_speech_override(cfg, speech)
        apply_flight_reuse(cfg, flight_reuse)
        return cls.from_config(
            cfg,
            flatten_channels=flatten_channels,
            rps_corruption=rps_corruption,
        )

    def set_epoch(self, epoch: int) -> None:
        self._inner.set_epoch(int(epoch))

    def __iter__(self):
        yield from self._inner


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

    **Channel policy.** ``channel_policy="first"`` (the default) keeps the
    historical single-microphone behaviour: channel 0 and row 0 of the
    recording's geometry. ``channel_policy="all"`` renders **every** microphone
    jointly, restoring the native multi-observer training the pre-Hydra
    trainer did.

    The distinction is not cosmetic. Anything whose spatial signature differs
    from the coherent field's ``1/r`` law — the wind-wake channel above all,
    whose whole claim is a wake-gated, per-microphone-incoherent field — is
    **unidentifiable from one microphone**: with ``M = 1`` it degenerates into
    just another broadband shape, competing against the far more flexible
    learned noise filter, and it will lose. Multi-observer training is what
    gives such a component something only it can explain. ``"random"`` is
    rejected because the inner dataset does not report which index it drew, so
    the geometry could not be matched to the audio.
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
        if inner.channel_policy not in ("first", "all"):
            raise ValueError(
                "NoiseGenFrameDataset requires channel_policy='first' (mic index 0) "
                "or 'all' (every mic, multi-observer) — 'random' does not report "
                f"which index it drew, so geometry cannot be matched; got "
                f"{inner.channel_policy!r}"
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
        audio = np.asarray(item["audio"], dtype=np.float32)
        origin = str(item["origin"])
        mic_pos_full, rotor_pos = self._geometry[origin]
        if self.inner.channel_policy == "all":
            audio = np.atleast_2d(audio)  # (M, T) — every mic
            mic_pos = np.asarray(mic_pos_full[: audio.shape[0]], dtype=np.float32)
        else:
            audio = audio[None, :]  # (1, T) — channel 0 only
            mic_pos = np.asarray(mic_pos_full[:1], dtype=np.float32)
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
        channel_policy: str = "first",
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
            channel_policy=channel_policy,
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
        channel_policy: str = "first",
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
            channel_policy=channel_policy,
            **dataset_kwargs,
        )
        return cls(val_ds, dregon_dir=dregon_dir)


class StaticCombGenDataset(Dataset):
    """Synthetic frozen-profile comb for the ``noise_generation`` task, with a
    switchable **label** transform — the Phase-7 generator label-sensitivity probe.

    Every sample is one rotor observed by one microphone. The *target* audio is
    :func:`data_processing.rotor_spectral_model.render_fixed_comb` driven by the
    **true** OU trajectory; the *conditioning* the model receives is that same
    trajectory passed through ``label_mode``. So the arms differ in exactly one
    thing — what the generator is told the rotor speed is — while the signal it
    must reproduce is identical:

    ==================  ==========================================================
    ``label_mode``      conditioning
    ==================  ==========================================================
    ``exact``           the truth (the loss-design / capacity control)
    ``scale``           ``label_scale`` x truth (the benign-bias control)
    ``tach``            ``label_scale`` x truth through the tachometer staircase
                        (:func:`data_processing.rps_corruption.tachometer_corrupt`)
    ``tach_presmooth``  ``tach`` low-passed at ``presmooth_cut_hz``
                        (:func:`data_processing.rps_corruption.presmooth_track`)
    ==================  ==========================================================

    Why one rotor and one microphone. Four overlapping combs make the per-``k``
    line readout ambiguous (whose harmonic is this?), and eight microphones only
    replicate a signal whose spatial structure is not under test. Both would add
    variance without adding a degree of freedom the hypothesis speaks about.

    Determinism: sample ``i`` of a split is a pure function of ``(seed, split,
    i)``, so validation is fixed and arms are comparable frame by frame.
    """

    #: Geometry — a plausible rotor/mic offset. The comb itself is rendered
    #: without propagation; the positions only have to be finite and non-zero
    #: so the generator's ``1/r`` + fractional-delay path is well posed.
    MIC_POS = np.array([[0.0, 0.0, 0.05]], dtype=np.float32)
    ROTOR_POS = np.array([[0.15, 0.15, 0.0]], dtype=np.float32)

    LABEL_MODES = ("exact", "scale", "tach", "tach_presmooth")

    def __init__(
        self,
        *,
        n_samples: int,
        split: str = "train",
        duration_s: float = 1.0,
        sample_rate: int = 16000,
        label_mode: str = "exact",
        label_scale: float = 1.0,
        tach_step: float = 0.269,
        tach_refresh_hz: float = 49.7,
        presmooth_cut_hz: float = 5.0,
        traj_fs: float = 250.0,
        shaft_cut_hz: float = 2.0,
        margin_s: float = 2.0,
        aggressiveness: float = 1.0,
        comb: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        from data_processing.rotor_spectral_model import FixedCombSpec

        if label_mode not in self.LABEL_MODES:
            raise ValueError(f"label_mode must be one of {self.LABEL_MODES}, got {label_mode!r}")
        self.n_samples = int(n_samples)
        self.split = str(split)
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)
        self.label_mode = label_mode
        self.label_scale = float(label_scale)
        self.tach_step = float(tach_step)
        self.tach_refresh_hz = float(tach_refresh_hz)
        self.presmooth_cut_hz = float(presmooth_cut_hz)
        self.traj_fs = float(traj_fs)
        self.shaft_cut_hz = float(shaft_cut_hz)
        self.margin_s = float(margin_s)
        self.aggressiveness = float(aggressiveness)
        self.seed = int(seed)

        if isinstance(comb, DictConfig):
            comb = cast(dict, OmegaConf.to_container(comb, resolve=True))
        self.spec = FixedCombSpec(**dict(comb or {}))
        self.profile = self.spec.profile(ref_rps=80.0, sample_rate=self.sample_rate)
        self.gain = self.spec.gain(self.profile)

    def __len__(self) -> int:
        return self.n_samples

    def _rng(self, idx: int) -> np.random.Generator:
        # crc32, not hash(): str hashing is salted per process (PYTHONHASHSEED),
        # so a builtin hash would make DataLoader workers disagree.
        split_key = zlib.crc32(self.split.encode()) & 0x7FFFFFFF
        return np.random.default_rng([self.seed, split_key, int(idx)])

    def true_rps(self, idx: int) -> np.ndarray:
        """The exact ``(T,)`` rotor speed of sample ``idx`` at audio rate.

        Equivalent to ``self._traj(idx)`` cropped to the clip.

        The OU draw is white to the trajectory rate, so it is band-limited by a
        **shaft-inertia** low pass at ``shaft_cut_hz`` before it becomes the
        truth. Two things depend on this, both measured at the 2 Hz default:

        * Without it the truth moves ~0.4 rev/s inside one 20 ms refresh
          interval, so the tachometer's *hold* error swamps its *quantization*
          error and the arm would measure "telemetry is too slow for the
          shaft" — a different claim, and one E6 already answered with the
          emitter's OU jitter. Band-limited, the staircase's total error is
          0.108 rev/s against 0.078 rev/s of pure quantization.
        * ``tach_presmooth`` only means something if the 5 Hz filter passes the
          truth. It does: it moves the *true* track by 0.017 rev/s, a sixth of
          the staircase it removes (0.108 -> 0.041 rev/s). Arm C therefore
          measures the mitigation's ceiling, not the filter's distortion.

        The truth still moves ~2.0 rev/s rms inside a window, so the model is
        not being handed a constant.
        """
        ext, sl = self._traj(idx)
        return ext[sl]

    def _traj(self, idx: int) -> tuple[np.ndarray, slice]:
        """The trajectory with margins, plus the slice that is the clip.

        The label transforms run on the **margined** track and are cropped
        afterwards, because a real deployment smooths a whole telemetry stream
        and then windows it — not the other way round. It also matters
        numerically: :func:`~data_processing.rps_corruption.presmooth_track`
        is a whole-window FFT brickwall, so on a bare 1 s clip a 5 Hz cutoff
        keeps only five harmonics and the filter rings worse than the
        staircase it removes (measured: 0.155 rev/s residual vs the
        staircase's 0.093). With ``margin_s`` of context it has enough
        harmonics to be the low pass it is meant to be.
        """
        from scipy.signal import butter, sosfiltfilt

        from data_processing import rps_synthesis

        rng = self._rng(idx)
        edge_s = 1.0  # discarded after filtering: filtfilt's own edge transient
        low = rps_synthesis.generate(
            self.duration_s + 2.0 * (self.margin_s + edge_s),
            self.traj_fs,
            aggressiveness=self.aggressiveness,
            rng=rng,
        )[0]
        sos = butter(4, self.shaft_cut_hz / (self.traj_fs / 2.0), output="sos")
        low = np.asarray(sosfiltfilt(sos, low), dtype=np.float64)
        n_edge = int(round(edge_s * self.traj_fs))
        low = low[n_edge : low.shape[-1] - n_edge]
        n_ext = int(round((self.duration_s + 2.0 * self.margin_s) * self.sample_rate))
        t_lo = np.arange(low.shape[-1], dtype=np.float64) / self.traj_fs
        t = np.arange(n_ext, dtype=np.float64) / self.sample_rate
        ext = np.interp(t, t_lo, low)
        off = int(round(self.margin_s * self.sample_rate))
        n_t = int(round(self.duration_s * self.sample_rate))
        return ext, slice(off, off + n_t)

    def label_for(self, rps: np.ndarray) -> np.ndarray:
        """Apply this arm's label transform to a true ``(..., T)`` track."""
        from data_processing.rps_corruption import presmooth_track, tachometer_corrupt

        if self.label_mode == "exact":
            return np.asarray(rps, dtype=np.float64)
        if self.label_mode == "scale":
            return np.asarray(rps, dtype=np.float64) * self.label_scale
        label = tachometer_corrupt(
            rps,
            self.sample_rate,
            step=self.tach_step,
            refresh_hz=self.tach_refresh_hz,
            scale=self.label_scale,
        )
        if self.label_mode == "tach_presmooth":
            label = presmooth_track(label, self.sample_rate, cut_hz=self.presmooth_cut_hz)
        return label

    def render(self, rps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Render the ``(T,)`` target audio for a true track (shared with eval)."""
        from data_processing.rotor_spectral_model import render_fixed_comb

        return render_fixed_comb(
            rps, self.profile, sample_rate=self.sample_rate, gain=self.gain, rng=rng
        )

    def __getitem__(self, idx: int) -> td.Frame:
        if idx < 0:
            idx += self.n_samples
        if not 0 <= idx < self.n_samples:
            raise IndexError(idx)
        ext, sl = self._traj(idx)
        rps_true = ext[sl]
        audio = self.render(rps_true, self._rng(idx + self.n_samples))
        label = self.label_for(ext)[sl]
        return td.Frame(
            {
                "audio": td.uniform(
                    audio[None, :].astype(np.float32),
                    self.sample_rate,
                    dims=("mic", "time"),
                    t_start=0.0,
                ),
                "rps": td.uniform(
                    label[None, :].astype(np.float32),
                    self.sample_rate,
                    dims=("rotor", "time"),
                    t_start=0.0,
                ),
                "mic_pos": td.wrap(self.MIC_POS, dims=("mic", None)),
                "rotor_pos": td.wrap(self.ROTOR_POS, dims=("rotor", None)),
                "meta": td.Frame({"drone": "synth", "sample_id": int(idx), "split": self.split}),
            }
        )


class DecompFrameDataset(Dataset):
    """Random chunks of the published Vold-Kalman decompositions.

    The training source of the amplitude-target objective. One published
    recording of ``decomp-frames-v1`` carries, on one common time origin:

    ``rps``
        ``(rotor, time)`` at the audio rate — the EXACT carrier the solve used,
        so the amplitudes are targets for the trajectory the model is
        conditioned on.
    ``amp`` / ``amp_valid``
        ``(mic, rotor, k, time)`` amplitude envelopes at 100 Hz and their
        validity mask (``False`` above a recording's own ``k_hi``).
    ``residual``
        ``(mic, time)`` the broadband part the tracks do not explain.

    Chunks are cut by INDEX, not by ``frame.time[a:b]``: a time cut of a 16 kHz
    entry and a 100 Hz entry can round to different lengths on different draws,
    and a batch of Frames must stack. Starts are therefore snapped to the
    envelope stride (160 samples), which makes the two grids commensurate and
    every sample the same shape.

    ``min_motor_rps`` rejects chunks that are not flying: the decomposition
    spans include the take-off ramp, where there is no comb to demodulate and
    the "envelopes" are floor noise. The draw is retried (bounded) rather than
    the span pre-trimmed, so a recording with a slow ramp still contributes its
    flight portion.

    ``dataset`` accepts a LIST of published datasets, which are concatenated
    into one record pool (draws stay duration-weighted across the union). The v3
    decompositions are published per rig, so a combined DREGON + Michael's arm
    names both; every record carries its own ``drone`` id, which is the rig id
    the model's propagation head is keyed by.
    """

    #: ``decomp-frames-v1``'s envelope stride in audio samples (meta.env_stride).
    ENV_STRIDE = 160
    MAX_DRAWS = 32

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 16000,
        n_samples: int = 4096,
        seed: int = 42,
        min_motor_rps: float = 30.0,
        split: str = "train",
    ) -> None:
        if not records:
            raise ValueError("DecompFrameDataset needs at least one record")
        self.records = records
        self.chunk_size = int(chunk_size)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.min_motor_rps = float(min_motor_rps)
        self.split = str(split)
        if self.chunk_size % self.ENV_STRIDE:
            raise ValueError(
                f"chunk_size {self.chunk_size} must be a multiple of the envelope stride "
                f"{self.ENV_STRIDE} so the audio-rate and 100 Hz grids stay commensurate"
            )
        mics = {int(r["residual"].shape[0]) for r in records}
        if len(mics) > 1:
            raise ValueError(f"records disagree on microphone count {sorted(mics)}; cannot batch")
        self.weights = np.asarray([float(r["span"][1] - r["span"][0]) for r in records])
        self.weights = self.weights / self.weights.sum()

    def __len__(self) -> int:
        return self.n_samples

    def _draw(self, idx: int) -> tuple[dict[str, Any], int]:
        """Pick a record and a stride-aligned start sample above the idle gate."""
        rng = np.random.default_rng([self.seed, idx])
        for _ in range(self.MAX_DRAWS):
            rec = self.records[int(rng.choice(len(self.records), p=self.weights))]
            lo, hi = rec["span"]
            last = hi - self.chunk_size
            if last < lo:
                continue
            n_starts = (last - lo) // self.ENV_STRIDE + 1
            s0 = lo + int(rng.integers(n_starts)) * self.ENV_STRIDE
            rps = rec["rps"][:, s0 : s0 + self.chunk_size]
            if float(rps.mean()) >= self.min_motor_rps:
                return rec, s0
        return rec, max(lo, min(s0, hi - self.chunk_size))

    def __getitem__(self, idx: int) -> td.Frame:
        if idx < 0:
            idx += self.n_samples
        if not 0 <= idx < self.n_samples:
            raise IndexError(idx)
        rec, s0 = self._draw(idx)
        e0 = s0 // self.ENV_STRIDE
        n_env = self.chunk_size // self.ENV_STRIDE
        sr = int(rec["sample_rate"])
        sl = slice(s0, s0 + self.chunk_size)
        esl = slice(e0, e0 + n_env)
        return td.Frame(
            {
                "rps": td.uniform(
                    np.ascontiguousarray(rec["rps"][:, sl]), sr, dims=("rotor", "time"), t_start=0.0
                ),
                "residual": td.uniform(
                    np.ascontiguousarray(rec["residual"][:, sl]),
                    sr,
                    dims=("mic", "time"),
                    t_start=0.0,
                ),
                "amp": td.Series(
                    np.ascontiguousarray(rec["amp"][:, :, :, esl]),
                    ("mic", "rotor", "k", "time"),
                    {"time": td.GridIndex.create((sr, self.ENV_STRIDE), n_env, t_start=0.0)},
                ),
                "amp_valid": td.Series(
                    np.ascontiguousarray(rec["amp_valid"][:, :, esl]),
                    ("rotor", "k", "time"),
                    {"time": td.GridIndex.create((sr, self.ENV_STRIDE), n_env, t_start=0.0)},
                ),
                "mic_pos": td.wrap(rec["mic_pos"], dims=("mic", None)),
                "rotor_pos": td.wrap(rec["rotor_pos"], dims=("rotor", None)),
                "meta": td.Frame(
                    {
                        "drone": rec["drone"],
                        "recording_id": rec["recording_id"],
                        "start_sample": int(s0),
                        "split": self.split,
                    }
                ),
            }
        )

    # ── construction ────────────────────────────────────────────────────────

    @staticmethod
    def _dataset_versions(
        dataset: str | list[str], version: str | list[str | None] | None
    ) -> list[tuple[str, str | None]]:
        """``(name, version)`` pairs — one dataset, or a concatenation of several.

        The v3 decompositions are published **per rig**
        (``decomp-frames-v3-dregon`` / ``decomp-frames-v3-michaels``), so a
        combined arm names both. Concatenating at this level (rather than
        publishing a joint dataset) keeps each rig's solve independently
        re-derivable, and costs nothing downstream: a record already carries its
        own ``drone`` id, which is the rig id the propagation head is keyed by.
        """
        names = [dataset] if isinstance(dataset, str) else list(dataset)
        if not names:
            raise ValueError("DecompFrameDataset needs at least one dataset name")
        if version is None or isinstance(version, str):
            versions: list[str | None] = [version] * len(names)
        else:
            versions = list(version)
            if len(versions) != len(names):
                raise ValueError(
                    f"{len(versions)} versions for {len(names)} datasets; give one per dataset "
                    "or a single version for all"
                )
        return list(zip(names, versions, strict=True))

    @classmethod
    def _load_records(
        cls,
        dataset: str | list[str],
        version: str | list[str | None] | None,
        *,
        split: str,
        val_pct: float,
        val_position: str,
    ) -> list[dict[str, Any]]:
        """Decode the published recordings once and take each one's split span.

        The split is a TIME split inside every recording (there are only three),
        so a validation chunk is never a training chunk of the same recording.
        ``val_position`` defaults to ``"middle"`` and that default is
        load-bearing: a flight recording BEGINS with the take-off ramp and ENDS
        with the landing one, so a held-out block at either end is a different
        flight regime, not a held-out sample of the same one — measured on
        ``decomp-frames-v1``, a leading 10 % block averages 41 rev/s against 79
        in the remainder. A middle block holds out cruise against cruise. The
        train side is then the two pieces around it.
        """
        from data_processing.streams import iter_published_frames

        if val_position not in ("middle", "start", "end"):
            raise ValueError(f"val_position must be middle/start/end, got {val_position!r}")
        records: list[dict[str, Any]] = []
        pairs = cls._dataset_versions(dataset, version)
        for name, ver in pairs:
            for tf in iter_published_frames(name, ver):
                records += cls._records_of(tf, split=split, val_pct=val_pct, v_pos=val_position)
        if not records:
            raise ValueError(f"{[n for n, _ in pairs]}: no published recording decoded")
        return records

    @classmethod
    def _records_of(
        cls, tf: td.Frame, *, split: str, val_pct: float, v_pos: str
    ) -> list[dict[str, Any]]:
        """One decoded recording -> its per-span records (see :meth:`_load_records`)."""
        from data_processing.frames import meta_dict

        meta = meta_dict(tf)
        rps = np.asarray(tf["rps"].data, dtype=np.float32)
        n_t = int(rps.shape[-1])
        n_val = int(round(val_pct * n_t)) // cls.ENV_STRIDE * cls.ENV_STRIDE
        starts = {"start": 0, "middle": (n_t - n_val) // 2, "end": n_t - n_val}
        v0 = starts[v_pos] // cls.ENV_STRIDE * cls.ENV_STRIDE
        spans = (
            [(v0, v0 + n_val)]
            if split == "valid"
            else [s for s in [(0, v0), (v0 + n_val, n_t)] if s[1] - s[0] > 0]
        )
        base = {
            "recording_id": str(meta.get("recording_id")),
            "drone": str(meta.get("drone")),
            "sample_rate": int(meta.get("sample_rate", 16000)),
            "rps": rps,
            "residual": np.asarray(tf["residual"].data, dtype=np.float32),
            "amp": np.asarray(tf["amp"].data, dtype=np.float32),
            "amp_valid": np.asarray(tf["amp_valid"].data, dtype=bool),
            "mic_pos": np.asarray(tf["mic_pos"].data, dtype=np.float32),
            "rotor_pos": np.asarray(tf["rotor_pos"].data, dtype=np.float32),
        }
        # One record per contiguous span: the arrays are SHARED (a view of the
        # same decoded recording), only the draw range differs.
        return [{**base, "span": span} for span in spans]

    @classmethod
    def build_train(
        cls,
        *,
        dataset: str | list[str] = "decomp-frames-v1",
        version: str | list[str | None] | None = None,
        chunk_size: int = 16000,
        train_samples: int = 4096,
        val_samples: int = 256,
        val_pct: float = 0.1,
        val_position: str = "middle",
        seed: int = 42,
        min_motor_rps: float = 30.0,
    ) -> DecompFrameDataset:
        del val_samples
        return cls(
            cls._load_records(
                dataset, version, split="train", val_pct=val_pct, val_position=val_position
            ),
            chunk_size=chunk_size,
            n_samples=train_samples,
            seed=seed,
            min_motor_rps=min_motor_rps,
            split="train",
        )

    @classmethod
    def build_valid(
        cls,
        *,
        dataset: str | list[str] = "decomp-frames-v1",
        version: str | list[str | None] | None = None,
        chunk_size: int = 16000,
        train_samples: int = 4096,
        val_samples: int = 256,
        val_pct: float = 0.1,
        val_position: str = "middle",
        seed: int = 42,
        min_motor_rps: float = 30.0,
    ) -> DecompFrameDataset:
        del train_samples
        return cls(
            cls._load_records(
                dataset, version, split="valid", val_pct=val_pct, val_position=val_position
            ),
            chunk_size=chunk_size,
            n_samples=val_samples,
            # A different seed stream from train's, so the two never draw the
            # same (record, start) pair by index coincidence.
            seed=seed + 1,
            min_motor_rps=min_motor_rps,
            split="valid",
        )


class FixedSynthFrameDataset(Dataset):
    """A FINITE, deterministic validation set drawn from an online-mix policy.

    Every synthetic-only arm in this project validates on the REAL frozen split,
    because that is the only finite RPS dataset the training loop can iterate
    (``run_training``'s validation pass has no sample cap, so an infinite
    ``OnlineMixFrameDataset`` would never terminate). One consequence went
    unnoticed for the whole stochastic-comb campaign: ``monitor: mse`` then
    selects each checkpoint by REAL performance, so a model that is steadily
    learning the synthetic distribution has its best synthetic weights thrown
    away. Measured on ``stoch_s1id_scv2``, the saved ``best`` checkpoint scores
    8.63 all-MAE on held-out synthetic where ``last`` scores 3.70.

    This class closes that hole. It pulls ``n`` frames from the stream ONCE at
    construction and serves them map-style, so the loop sees an ordinary finite
    dataset. The stream is deterministic given ``base_seed``, so the set is
    reproducible across runs and machines; use a seed the training policy does
    not use, or validation overlaps training.

    ``duration_s`` and ``augment`` default to the REAL split's conditions (8 s
    clips, no augmentation) rather than the policy's training settings, so a
    synthetic validation number is directly comparable with a real one. Set
    ``augment=True`` to select on the augmented task the model actually
    optimizes.
    """

    def __init__(
        self,
        path: str | Any,
        n: int = 64,
        base_seed: int = 990001,
        duration_s: float | None = 8.0,
        augment: bool = False,
        flatten_channels: bool = True,
        flight_reuse: int | None = None,
        speech: bool | None = None,
    ):
        cfg = OmegaConf.load(path) if isinstance(path, str) else path
        cfg.base_seed = int(base_seed)
        if duration_s is not None:
            cfg.duration_s = float(duration_s)
        if not augment:
            for stage in cfg.policy.stages:
                for key in ("augmentations", "noise_augmentations", "noise_time_warp"):
                    if key in stage:
                        del stage[key]
        apply_flight_reuse(cfg, flight_reuse)
        apply_speech_override(cfg, speech)
        stream = OnlineMixFrameDataset.from_config(cfg, flatten_channels=flatten_channels)
        self._frames: list[td.Frame] = []
        for frame in stream:
            self._frames.append(frame)
            if len(self._frames) >= int(n):
                break

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> td.Frame:
        return self._frames[int(idx)]


class ConcatFrameDataset(Dataset):
    """Concatenate several finite Frame datasets into one validation set.

    The with-speech arms of the salv2 grid validate on BOTH conditions at once:
    a model trained on mixtures has to keep working when the speech is absent,
    and one number that averages the two says whether it does. Each part is an
    ordinary ``_target_`` block in the Hydra config, so the parts are declared
    where they are used and nothing here knows what they contain.

    The parts are NOT reweighted -- keep them the same length, or the longer one
    sets the average.
    """

    def __init__(self, parts: Sequence[Any]):
        self.parts: list[Any] = list(parts)
        if not self.parts:
            raise ValueError("ConcatFrameDataset needs at least one part")
        self._lens = [len(p) for p in self.parts]

    def __len__(self) -> int:
        return sum(self._lens)

    def __getitem__(self, idx: int) -> td.Frame:
        i = int(idx)
        for part, n in zip(self.parts, self._lens):
            if i < n:
                return part[i]
            i -= n
        raise IndexError(idx)


class SpeechPairedSynthValidDataset(Dataset):
    """Validation on BOTH conditions: half the clips without speech, half with.

    The salv2 grid asks whether mixed-in speech helps or hurts rotor-speed
    estimation. A model trained WITH speech must be scored on both halves, or
    the number cannot say whether it merely learned to ignore a talker that is
    always there. The two halves are equal in size, so neither sets the average
    on its own.

    Both halves come from ONE policy file at the SAME seed, which is what makes
    this a matched pair rather than two samples: measured, the two halves come
    back with byte-identical RPS labels and byte-identical rotor noise, and the
    only difference in the audio is the added speech (residual RMS 0.0063
    against 0.100 of noise, present in every frame). The speech contrast is
    therefore free of trajectory variance -- the same flight is scored twice,
    once quiet and once with a talker over it.

    ``n`` counts frames AFTER the per-mic flat_map and is split evenly, so with
    8 microphones ``n=512`` is 32 clips without speech and 32 with. Pass
    ``flight_reuse=1`` (the default here, unlike the training policy) or the
    clips collapse onto a handful of trajectories -- see
    :func:`apply_flight_reuse`.
    """

    def __init__(
        self,
        path: str,
        n: int = 512,
        base_seed: int = 880101,
        duration_s: float | None = 8.0,
        augment: bool = False,
        flatten_channels: bool = True,
        flight_reuse: int | None = 1,
    ):
        half = int(n) // 2

        def _half(speech: bool) -> FixedSynthFrameDataset:
            return FixedSynthFrameDataset(
                path=path,
                n=half,
                base_seed=int(base_seed),
                duration_s=duration_s,
                augment=augment,
                flatten_channels=flatten_channels,
                flight_reuse=flight_reuse,
                speech=speech,
            )

        self.no_speech = _half(False)
        self.with_speech = _half(True)
        self._inner = ConcatFrameDataset([self.no_speech, self.with_speech])

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> td.Frame:
        return self._inner[idx]


class MixtureMatchedValidDataset(Dataset):
    """Validation whose SOURCE MIX reproduces the training policy's.

    WHY. The mixed-training arms (`m3abl_mixed_*`, real 50 / generated 25 /
    comb 25) validated on the REAL split alone and stopped at 25 epochs. The
    same family validated on a set that contains its own synthetic sources
    (`stoch_long_scv2`, `MixedRealSynthValidDataset`) reached 228. That is a
    9x difference in training length produced by the validation set, not by the
    model or the data, so the mixed arms' verdict was never a fair one.

    This class removes the remaining mismatch in that fix. `MixedRealSynthValid`
    is half real and half synthetic regardless of what the policy trains on;
    here the proportions are READ FROM THE POLICY, so the monitored number is an
    estimate of the training objective's own risk. With the real frozen split at
    296 frames and a 50/25/25 policy, that is 296 real + 148 generated + 148
    comb.

    THE REAL PART IS THE WHOLE FROZEN SPLIT, never a subsample: it is the number
    every other row in the project is quoted against, and shrinking it would
    make this arm incomparable.

    Each synthetic part is drawn from a copy of the policy holding ONE source,
    so the parts keep the policy's SNR range, speech and excitation. A
    ``generated`` source is forced to ``refresh: false`` (a fixed bank) because
    a live producer is not reproducible, and its buffer is shrunk -- a few dozen
    clips do not need 512 slots beside the training producer's.
    """

    #: sources that are the REAL pool; everything else is synthetic and is
    #: sized relative to them.
    REAL_KINDS = ("frames",)

    def __init__(
        self,
        policy_path: str,
        real_data_dir: str = "dload:DREGON-LM-V4-michaels-valid-full",
        base_seed: int = 970001,
        duration_s: float = 8.0,
        augment: bool = False,
        flatten_channels: bool = True,
        flight_reuse: int | None = 1,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
    ):
        self.real = DregonLMFrameDataset(
            data_dir=real_data_dir,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            flatten_channels=flatten_channels,
        )
        cfg = OmegaConf.load(policy_path)
        sources = list(cfg.sources.noise)
        real_w = sum(
            float(src.get("weight", 1.0)) for src in sources if src.get("kind") in self.REAL_KINDS
        )
        if real_w <= 0:
            raise ValueError(f"{policy_path} has no `kind: frames` source to scale against")

        parts: list[Any] = [self.real]
        self.counts: dict[str, int] = {"real": len(self.real)}
        for i, src in enumerate(sources):
            kind = str(src.get("kind"))
            if kind in self.REAL_KINDS:
                continue
            n_k = int(round(len(self.real) * float(src.get("weight", 1.0)) / real_w))
            if n_k <= 0:
                continue
            sub = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            one = OmegaConf.create(OmegaConf.to_container(src, resolve=False))
            if kind == "generated":
                # A LIVE PRODUCER IS NOT REPRODUCIBLE, so the bank is filled once.
                one.refresh = False
                # AND IT MUST BE SMALL. The producer renders `gen_batch` clips at
                # once on the GPU, and a validation clip is 8 s where a training
                # clip is 1 s -- so the policy's `gen_batch: 32` asks for 8x the
                # per-clip work at the same batch. Measured: it tried to allocate
                # 12.21 GiB BESIDE the training stream's own producer (12.61 GiB)
                # and the job died with CUDA OOM. At `gen_batch: 1` the second
                # producer costs about 3 GiB, and it fills a few dozen clips once.
                one.gen_batch = 1
                one.buffer = {"slots": 32, "warmup": 4}
            sub.sources.noise = [one]
            parts.append(
                FixedSynthFrameDataset(
                    path=sub,
                    n=n_k,
                    base_seed=int(base_seed) + i,
                    duration_s=duration_s,
                    augment=augment,
                    flatten_channels=flatten_channels,
                    flight_reuse=flight_reuse,
                )
            )
            self.counts[f"{kind}[{i}]"] = n_k
        self._inner = ConcatFrameDataset(parts)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> td.Frame:
        return self._inner[idx]


class MixedRealSynthValidDataset(Dataset):
    """Validation on HALF REAL clips and HALF SYNTHETIC clips, concatenated.

    Selecting a synthetic-only model on either half alone fails in a different
    direction, and both failures are measured rather than hypothetical.

    Real-only selection — what every arm in the stochastic-comb campaign did,
    because the real frozen split was the only finite RPS dataset available —
    halts training while the synthetic fit is still improving. No arm ever
    converged: ``stoch_s1id_trbig``'s best checkpoint is epoch 1 of 11,
    ``trxl`` reached 20 epochs, ``trxxl`` 34, and the BiGRU's best was epoch 5.
    Conclusions about capacity drawn from those runs measured the stopping rule.

    Synthetic-only selection has the opposite hole: a model that fits the
    generator ever more precisely keeps scoring better and is kept, even while
    it diverges from the real rigs. ``stoch_s1id_scv2``'s final weights fit
    synthetic 2.3x better than its saved best (8.63 -> 3.70 all-MAE) and score
    1.9x worse on real (7.40 -> 14.19); synthetic-only selection would have
    chosen exactly that checkpoint.

    Half and half gives the monitored metric both jobs: it improves while the
    model learns harmonic structure that generalizes, and turns over as soon as
    the model starts fitting generator artifacts the real recordings lack.

    The two halves are matched in size so neither dominates the average, and
    the synthetic half is drawn at 8 s without augmentation — the real half's
    conditions — so the two are on the same scale. Use a ``base_seed`` the
    training policy does not use, or validation overlaps training.
    """

    def __init__(
        self,
        policy_path: str,
        real_data_dir: str = "dload:DREGON-LM-V4-michaels-valid-full",
        base_seed: int = 990001,
        n_synth: int | None = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
        duration_s: float = 8.0,
        augment: bool = False,
        flatten_channels: bool = True,
    ):
        self.real = DregonLMFrameDataset(
            data_dir=real_data_dir,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            flatten_channels=flatten_channels,
        )
        # Default to exactly as many synthetic clips as the real split has, so
        # the monitored metric weights the two domains equally.
        self.synth = FixedSynthFrameDataset(
            path=policy_path,
            n=len(self.real) if n_synth is None else int(n_synth),
            base_seed=base_seed,
            duration_s=duration_s,
            augment=augment,
            flatten_channels=flatten_channels,
        )

    def __len__(self) -> int:
        return len(self.real) + len(self.synth)

    def __getitem__(self, idx: int) -> td.Frame:
        idx = int(idx)
        if idx < len(self.real):
            return self.real[idx]
        return self.synth[idx - len(self.real)]
