"""Naive online-mixing IterableDataset for RPS prediction.

This is intentionally the simplest implementation of the online-mixing design:

- aligned rotating-noise recordings stay as existing ``TimeFrame`` objects;
- unaligned speech/source audio is loaded from ordinary files on demand;
- the public interface is config-in/stream-out;
- optimization layers such as packed source caches can be added behind the same
  ``SourcePool`` interface later.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import IterableDataset, get_worker_info

from data_processing.dregon import clean_command_spikes, load_dregon_timeframes
from data_processing.michaels import load_michaels_timeframes
from utils.data import EventSeries, TimeFrame, UniformSeries


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DURATION_S = 1.0
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def _to_plain(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def make_rng(base_seed: int, global_sample_id: int) -> np.random.Generator:
    """Deterministic per-sample RNG independent of worker process."""
    payload = f"{int(base_seed)}:{int(global_sample_id)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    return np.random.default_rng(seed)


def _resolve_motor_tracks(tf: TimeFrame) -> tuple[str, str, bool]:
    """Return ``(detect_key, rps_key, needs_cleaning)`` for a noise TimeFrame."""
    if "motors_command" in tf or "motors_measured" in tf:
        detect = "motors_measured" if "motors_measured" in tf else "motors_command"
        rps_key = "motors_command" if "motors_command" in tf else "motors_measured"
        return detect, rps_key, True
    if "rps" in tf:
        return "rps", "rps", False
    raise ValueError(
        f"{tf.tags.get('recording_id', '?')} has no rotor-speed track "
        "(expected 'motors_measured', 'motors_command', or 'rps')"
    )


def _inflight_window(
    tf: TimeFrame,
    motor_key: str,
    *,
    min_motor_rps: float,
    clean: bool,
) -> tuple[float, float]:
    motor = cast(EventSeries, tf[motor_key])
    if motor.values is None or len(motor) == 0:
        return motor.t_start, motor.t_end
    values = np.asarray(motor.values, dtype=np.float32)
    if clean:
        values = clean_command_spikes(values)
    mask = np.all(values > float(min_motor_rps), axis=0)
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        raise ValueError(
            f"No in-flight window (all motors > {min_motor_rps} RPS) in "
            f"{tf.tags.get('recording_id', '?')}"
        )
    times = motor.abs_timestamps
    return float(times[idxs[0]]), float(times[idxs[-1]])


def _as_audio_ct(audio: np.ndarray, *, target_len: int | None = None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 2:
        # TimeFrame audio is already (C, T); soundfile audio is normalized before
        # calling this helper.
        pass
    else:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {audio.shape}")
    if target_len is not None:
        if audio.shape[-1] < target_len:
            audio = np.pad(audio, ((0, 0), (0, target_len - audio.shape[-1])))
        elif audio.shape[-1] > target_len:
            audio = audio[..., :target_len]
    return np.ascontiguousarray(audio, dtype=np.float32)


def _extract_audio_array(tf: TimeFrame, *, target_len: int) -> np.ndarray:
    audio = cast(UniformSeries, tf["audio"])
    return _as_audio_ct(audio.samples, target_len=target_len)


def _stft_frame_times(audio: UniformSeries, n_frames: int, hop_length: int) -> np.ndarray:
    # Match the existing training target convention: frame i is placed at
    # t_start + i * hop / sr.  Online data uses timestamp interpolation rather
    # than shape-stretching raw RPS arrays.
    return audio.t_start + (np.arange(n_frames, dtype=np.float64) * hop_length / audio.sr)


def interpolate_rps_to_stft_grid(
    tf: TimeFrame,
    *,
    n_frames: int,
    hop_length: int,
) -> np.ndarray:
    """Interpolate a sliced TimeFrame's RPS track to the model STFT grid."""
    _, rps_key, needs_clean = _resolve_motor_tracks(tf)
    audio = cast(UniformSeries, tf["audio"])
    rps = cast(EventSeries, tf[rps_key])
    if rps.values is None or len(rps) == 0:
        return np.zeros((4, n_frames), dtype=np.float32)

    frame_times = _stft_frame_times(audio, n_frames, hop_length)
    values = np.asarray(rps.values, dtype=np.float32)
    if needs_clean:
        values = clean_command_spikes(values)
    event_times = rps.abs_timestamps

    out = np.empty((values.shape[0], n_frames), dtype=np.float32)
    for i in range(values.shape[0]):
        out[i] = np.interp(frame_times, event_times, values[i]).astype(np.float32)
    return out


class TimeFrameNoisePool:
    """Randomly slice existing aligned noise ``TimeFrame`` recordings."""

    def __init__(
        self,
        recordings: Iterable[TimeFrame],
        *,
        min_motor_rps: float = 30.0,
        duration_s: float = DEFAULT_DURATION_S,
    ):
        self.records: list[dict[str, Any]] = []
        for tf in recordings:
            if "audio" not in tf:
                continue
            try:
                detect_key, _rps_key, needs_clean = _resolve_motor_tracks(tf)
                audio = cast(UniformSeries, tf["audio"])
                detect = cast(EventSeries, tf[detect_key])
                valid_start = max(audio.t_start, detect.t_start)
                valid_end = min(audio.t_end, detect.t_end)
                if min_motor_rps > 0:
                    flight_start, flight_end = _inflight_window(
                        tf, detect_key, min_motor_rps=min_motor_rps, clean=needs_clean
                    )
                    valid_start = max(valid_start, flight_start)
                    valid_end = min(valid_end, flight_end)
                available = valid_end - valid_start
                if available >= duration_s:
                    self.records.append(
                        {"tf": tf, "valid_start": valid_start, "valid_end": valid_end}
                    )
            except Exception as exc:
                print(f"Warning: skipping noise frame {tf.tags.get('recording_id', '?')}: {exc}")

        if not self.records:
            raise ValueError("no usable noise recordings found")
        weights = np.array(
            [r["valid_end"] - r["valid_start"] for r in self.records], dtype=np.float64
        )
        self.weights = weights / weights.sum()

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> "TimeFrameNoisePool":
        cfg = _to_plain(cfg)
        if isinstance(cfg, list):
            combined = object.__new__(cls)
            combined.records = []
            for one_cfg in cfg:
                pool = cls.from_config(one_cfg, duration_s=duration_s, sample_rate=sample_rate)
                combined.records.extend(pool.records)
            if not combined.records:
                raise ValueError("no usable noise recordings found")
            weights = np.array(
                [r["valid_end"] - r["valid_start"] for r in combined.records], dtype=np.float64
            )
            combined.weights = weights / weights.sum()
            return combined

        kind = _cfg_get(cfg, "kind", "dregon")
        root = Path(_cfg_get(cfg, "root", "data"))
        min_motor_rps = float(_cfg_get(cfg, "min_motor_rps", 30.0))
        if kind == "dregon":
            splits = _cfg_get(cfg, "splits", None)
            split = _cfg_get(cfg, "split", None)
            if splits is None and split is not None:
                splits = [split]
            if splits is None:
                splits = ["in_flight_noise"]
            download = bool(_cfg_get(cfg, "download", False))
            frames = load_dregon_timeframes(
                root, splits=list(splits), target_sr=sample_rate, download=download
            )
            ids = _cfg_get(cfg, "recording_ids", None)
            if ids is not None:
                wanted = {str(x) for x in ids}
                frames = [tf for tf in frames if str(tf.tags.get("recording_id", "")) in wanted]
            exclude_ids = _cfg_get(cfg, "exclude_recording_ids", None)
            if exclude_ids is not None:
                excluded = {str(x) for x in exclude_ids}
                frames = [tf for tf in frames if str(tf.tags.get("recording_id", "")) not in excluded]
        elif kind in {"michaels", "michael"}:
            frames = load_michaels_timeframes(data_root=root, sr=sample_rate)
            ids = _cfg_get(cfg, "ids", _cfg_get(cfg, "id", "all"))
            if isinstance(ids, str):
                ids = [ids]
            want_all = any(str(i).strip().lower() == "all" for i in ids)
            wanted = {str(i).strip().lower().removeprefix("fly") for i in ids}
            selected = []
            for tf in frames:
                rid = str(tf.tags.get("recording_id", ""))
                if want_all or rid.lower().removeprefix("fly") in wanted:
                    tags = dict(tf.tags)
                    tags["recording_id"] = f"michaels_FLY{rid}"
                    selected.append(
                        TimeFrame.from_tracks(
                            {k: tf[k] for k in tf},
                            t_start=tf.t_start,
                            tags=tags,
                            global_data=tf.global_data,
                        )
                    )
            exclude_ids = _cfg_get(cfg, "exclude_recording_ids", None)
            if exclude_ids is not None:
                excluded = {str(x) for x in exclude_ids}
                selected = [tf for tf in selected if str(tf.tags.get("recording_id", "")) not in excluded]
            frames = selected
        else:
            raise ValueError(f"unsupported noise source kind: {kind!r}")
        return cls(frames, min_motor_rps=min_motor_rps, duration_s=duration_s)

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> TimeFrame:
        idx = int(rng.choice(len(self.records), p=self.weights))
        rec = self.records[idx]
        start = float(rng.uniform(rec["valid_start"], rec["valid_end"] - duration_s))
        return cast(TimeFrame, rec["tf"]).slice(start, start + duration_s)


class AudioFileSourcePool:
    """Naive source pool that reads individual audio files on demand."""

    AUDIO_SUFFIXES = {".flac", ".wav", ".ogg", ".mp3"}

    def __init__(
        self,
        files: Iterable[str | Path],
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration_s: float = DEFAULT_DURATION_S,
        cache_mode: str = "none",
        cache_dir: str | Path = ".cache/online_mix_sources",
    ):
        self.files = [Path(p) for p in files]
        self.sample_rate = int(sample_rate)
        self.target_len = int(round(duration_s * sample_rate))
        self.cache_mode = str(cache_mode)
        self.cache_dir = Path(cache_dir)
        if not self.files:
            raise ValueError("source pool has no audio files")
        self._memory_cache: list[np.ndarray] | None = None
        self._packed_data: np.memmap | None = None
        self._packed_index: np.ndarray | None = None
        if self.cache_mode == "memory":
            print(f"Creating in-memory source cache for {len(self.files)} files ...")
            self._memory_cache = [self._load_one(p) for p in self.files]
        elif self.cache_mode in {"packed_int16", "auto"}:
            self._open_or_create_packed_cache()
        elif self.cache_mode in {"none", "file_lru"}:
            # `file_lru` currently falls back to direct file reads. It keeps the
            # config schema stable for a later bounded cache implementation.
            pass
        else:
            raise ValueError(f"unsupported source cache mode: {self.cache_mode!r}")

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> "AudioFileSourcePool":
        cfg = _to_plain(cfg)
        root = Path(_cfg_get(cfg, "root", "."))
        globs = _cfg_get(cfg, "globs", None)
        glob_one = _cfg_get(cfg, "glob", None)
        if globs is None:
            globs = [glob_one] if glob_one is not None else ["**/*.flac", "**/*.wav"]
        files: list[Path] = []
        for pattern in globs:
            files.extend(p for p in root.glob(str(pattern)) if p.suffix.lower() in cls.AUDIO_SUFFIXES)
        files = sorted(set(files))
        cache_cfg = _cfg_get(cfg, "cache", {}) or {}
        cache_mode = str(_cfg_get(cache_cfg, "mode", "none"))
        cache_dir = _cfg_get(cache_cfg, "dir", ".cache/online_mix_sources")
        return cls(
            files,
            sample_rate=sample_rate,
            duration_s=duration_s,
            cache_mode=cache_mode,
            cache_dir=cache_dir,
        )

    def _cache_fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update(b"online-mix-source-cache-v1")
        h.update(str(self.sample_rate).encode())
        for path in self.files:
            st = path.stat()
            h.update(str(path.resolve()).encode())
            h.update(str(st.st_size).encode())
            h.update(str(st.st_mtime_ns).encode())
        return h.hexdigest()

    def _open_or_create_packed_cache(self) -> None:
        fingerprint = self._cache_fingerprint()
        cache_path = self.cache_dir / fingerprint
        data_path = cache_path / "audio.i16"
        index_path = cache_path / "index.npy"
        manifest_path = cache_path / "manifest.json"

        if not (data_path.exists() and index_path.exists() and manifest_path.exists()):
            print(
                f"Creating source cache for {len(self.files)} files at {cache_path} "
                "(PCM16 packed audio) ..."
            )
            cache_path.mkdir(parents=True, exist_ok=True)
            tmp_data = cache_path / "audio.i16.tmp"
            tmp_index = cache_path / "index.npy.tmp"
            tmp_manifest = cache_path / "manifest.json.tmp"
            offsets: list[tuple[int, int]] = []
            offset = 0
            with open(tmp_data, "wb") as f:
                for i, path in enumerate(self.files, start=1):
                    audio = self._load_one(path)
                    audio_i16 = np.clip(audio, -1.0, 1.0)
                    audio_i16 = np.round(audio_i16 * 32767.0).astype(np.int16)
                    offsets.append((offset, int(audio_i16.shape[0])))
                    f.write(audio_i16.tobytes(order="C"))
                    offset += int(audio_i16.shape[0])
                    if i == 1 or i % 1000 == 0 or i == len(self.files):
                        print(f"  cached {i}/{len(self.files)} source files")
            index = np.asarray(offsets, dtype=np.int64)
            with open(tmp_index, "wb") as f:
                np.save(f, index)
            manifest = {
                "version": 1,
                "dtype": "int16",
                "sample_rate": self.sample_rate,
                "n_files": len(self.files),
                "total_samples": int(offset),
                "fingerprint": fingerprint,
            }
            tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            tmp_data.replace(data_path)
            tmp_index.replace(index_path)
            tmp_manifest.replace(manifest_path)
        else:
            print(f"Reusing source cache at {cache_path}")

        self._packed_index = np.load(index_path)
        total_samples = int(self._packed_index[:, 1].sum())
        self._packed_data = np.memmap(data_path, dtype=np.int16, mode="r", shape=(total_samples,))

    def _load_one(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            # soundfile returns (T, C); convert to mono for source material.
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return np.asarray(audio, dtype=np.float32)

    def sample_mono(self, rng: np.random.Generator) -> np.ndarray:
        idx = int(rng.integers(0, len(self.files)))
        if self._packed_data is not None and self._packed_index is not None:
            offset, length = self._packed_index[idx]
            audio = self._packed_data[int(offset) : int(offset + length)].astype(np.float32) / 32767.0
        elif self._memory_cache is None:
            path = self.files[idx]
            audio = self._load_one(path)
        else:
            audio = self._memory_cache[idx]
        if audio.shape[0] >= self.target_len:
            start = int(rng.integers(0, audio.shape[0] - self.target_len + 1))
            audio = audio[start : start + self.target_len]
        else:
            audio = np.pad(audio, (0, self.target_len - audio.shape[0]))
        return np.ascontiguousarray(audio, dtype=np.float32)

    def sample_array(
        self,
        rng: np.random.Generator,
        *,
        channels: int,
        mode: str = "independent",
    ) -> np.ndarray:
        if mode == "shared":
            mono = self.sample_mono(rng)
            return np.tile(mono[None, :], (channels, 1)).astype(np.float32, copy=False)
        if mode != "independent":
            raise ValueError(f"unsupported source channel mode: {mode!r}")
        return np.stack([self.sample_mono(rng) for _ in range(channels)], axis=0).astype(np.float32)


def _sample_snr_db(policy: Mapping[str, Any], rng: np.random.Generator) -> float:
    spec = policy.get("snr_db", {"uniform": {"low": -30.0, "high": 0.0}})
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, Mapping) and "uniform" in spec:
        u = cast(Mapping[str, Any], spec["uniform"])
        return float(rng.uniform(float(u.get("low", -30.0)), float(u.get("high", 0.0))))
    raise ValueError(f"unsupported snr_db spec: {spec!r}")


def _resolve_policy(policy: Mapping[str, Any], global_sample_id: int) -> Mapping[str, Any]:
    """Resolve constant or staged policy for a global sample id.

    Phase-1 schedules are intentionally simple: a list of stages with ``until``
    sample ids.  The first stage whose ``until`` is ``None`` or greater than the
    current id is active.
    """
    stages = policy.get("stages") if isinstance(policy, Mapping) else None
    if not stages:
        return policy
    for stage in stages:
        until = stage.get("until")
        if until is None or int(global_sample_id) < int(until):
            return stage
    return stages[-1]


def _mix_at_source_to_noise_snr(
    source: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    per_channel: bool = False,
) -> np.ndarray:
    eps = 1e-12
    if per_channel:
        noise_power = np.mean(noise.astype(np.float64) ** 2, axis=1, keepdims=True)
        source_power = np.mean(source.astype(np.float64) ** 2, axis=1, keepdims=True)
    else:
        noise_power = np.array([[np.mean(noise.astype(np.float64) ** 2)]])
        source_power = np.array([[np.mean(source.astype(np.float64) ** 2)]])
    scale = np.sqrt((noise_power * (10.0 ** (float(snr_db) / 10.0))) / (source_power + eps))
    return (noise + source * scale.astype(np.float32)).astype(np.float32)


def _apply_one_augmentation(
    audio: np.ndarray,
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if not spec:
        return audio
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return audio
    choices = list(spec.get("choices", []))
    if not choices:
        return audio
    choice = choices[int(rng.integers(0, len(choices)))]
    if isinstance(choice, str):
        name, params = choice, {}
    elif isinstance(choice, Mapping):
        if len(choice) != 1:
            raise ValueError(f"augmentation choice must have one key, got {choice!r}")
        name, params = next(iter(choice.items()))
        params = params or {}
    else:
        raise ValueError(f"unsupported augmentation choice: {choice!r}")

    out = audio.copy()
    if name == "random_gain":
        min_db = float(params.get("min_db", -6.0))
        max_db = float(params.get("max_db", 6.0))
        gain = 10.0 ** (float(rng.uniform(min_db, max_db)) / 20.0)
        out *= np.float32(gain)
    elif name == "random_polarity":
        out *= -1.0
    elif name == "channel_drop":
        if out.shape[0] <= 1:
            return out
        max_channels = int(params.get("max_channels", 1))
        n_drop = int(rng.integers(1, min(max_channels, out.shape[0] - 1) + 1))
        drop = rng.choice(out.shape[0], size=n_drop, replace=False)
        out[drop, :] = 0.0
    else:
        raise ValueError(f"unsupported augmentation: {name!r}")
    return out.astype(np.float32, copy=False)


class OnlineMixIterableDataset(IterableDataset):
    """Infinite online-mixing stream yielding ``(audio, rps_target)`` tensors."""

    def __init__(
        self,
        noise_pool: TimeFrameNoisePool,
        source_pool: AudioFileSourcePool | None,
        *,
        policy: Mapping[str, Any] | None = None,
        base_seed: int = 1234,
        duration_s: float = DEFAULT_DURATION_S,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        start_sample_id: int = 0,
    ):
        self.noise_pool = noise_pool
        self.source_pool = source_pool
        self.policy = dict(policy or {})
        self.base_seed = int(base_seed)
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.start_sample_id = int(start_sample_id)
        self.target_len = int(round(self.duration_s * self.sample_rate))

    @classmethod
    def from_config(cls, cfg: Any) -> "OnlineMixIterableDataset":
        cfg = _to_plain(cfg)
        sample_rate = int(_cfg_get(cfg, "sample_rate", DEFAULT_SAMPLE_RATE))
        duration_s = float(_cfg_get(cfg, "duration_s", DEFAULT_DURATION_S))
        n_fft = int(_cfg_get(cfg, "n_fft", DEFAULT_N_FFT))
        hop_length = int(_cfg_get(cfg, "hop_length", DEFAULT_HOP_LENGTH))
        base_seed = int(_cfg_get(cfg, "base_seed", 1234))
        start_sample_id = int(_cfg_get(cfg, "start_sample_id", 0))
        policy = cast(Mapping[str, Any], _cfg_get(cfg, "policy", {}))

        sources = _cfg_get(cfg, "sources", {})
        noise_cfg = _cfg_get(sources, "noise", None)
        if noise_cfg is None:
            raise ValueError("online mix config requires sources.noise")
        noise_pool = TimeFrameNoisePool.from_config(
            noise_cfg, duration_s=duration_s, sample_rate=sample_rate
        )

        speech_cfgs = _cfg_get(sources, "speech", None)
        source_pool = None
        if speech_cfgs is not None:
            speech_cfg = speech_cfgs[0] if isinstance(speech_cfgs, list) else speech_cfgs
            source_pool = AudioFileSourcePool.from_config(
                speech_cfg, duration_s=duration_s, sample_rate=sample_rate
            )

        return cls(
            noise_pool,
            source_pool,
            policy=policy,
            base_seed=base_seed,
            duration_s=duration_s,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            start_sample_id=start_sample_id,
        )

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        k = 0
        while True:
            global_sample_id = self.start_sample_id + worker_id + k * num_workers
            k += 1
            yield self.generate_sample(global_sample_id)

    def generate_sample(self, global_sample_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = make_rng(self.base_seed, int(global_sample_id))
        policy = _resolve_policy(self.policy, int(global_sample_id))
        noise_tf = self.noise_pool.sample_timeframe(rng, self.duration_s)
        audio_track = cast(UniformSeries, noise_tf["audio"])
        noise_audio = _extract_audio_array(noise_tf, target_len=self.target_len)
        n_frames = noise_audio.shape[-1] // self.hop_length + 1

        source_prob = float(policy.get("source_prob", 1.0 if self.source_pool is not None else 0.0))
        mixture = noise_audio
        if self.source_pool is not None and rng.random() < source_prob:
            mode = str(policy.get("speech_per_channel", "independent"))
            source = self.source_pool.sample_array(rng, channels=noise_audio.shape[0], mode=mode)
            snr_db = _sample_snr_db(policy, rng)
            mixture = _mix_at_source_to_noise_snr(
                source,
                noise_audio,
                snr_db,
                per_channel=bool(policy.get("snr_per_channel", False)),
            )

        mixture = _apply_one_augmentation(
            mixture,
            cast(Mapping[str, Any] | None, policy.get("augmentations")),
            rng,
        )
        # Keep amplitude policy deliberately simple for Phase 1: no peak
        # normalization, because that would alter the configured SNR/gain regime.
        rps = interpolate_rps_to_stft_grid(
            noise_tf.select(["audio", _resolve_motor_tracks(noise_tf)[1]]),
            n_frames=n_frames,
            hop_length=self.hop_length,
        )
        # `audio_track` is read above to document that the STFT frame grid is
        # tied to the sliced audio's actual timeline; keep this sanity check here.
        if int(round(audio_track.sr)) != self.sample_rate:
            raise ValueError(f"noise audio sr {audio_track.sr} != configured {self.sample_rate}")
        return torch.from_numpy(mixture), torch.from_numpy(rps)
