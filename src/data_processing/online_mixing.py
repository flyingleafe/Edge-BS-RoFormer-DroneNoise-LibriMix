"""Naive online-mixing IterableDataset for RPS prediction.

This is intentionally the simplest implementation of the online-mixing design:

- aligned rotating-noise recordings stay as existing ``td.Frame`` objects;
- unaligned speech/source audio is loaded from ordinary files on demand;
- the public interface is config-in/stream-out;
- optimization layers such as packed source caches can be added behind the same
  ``SourcePool`` interface later.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import librosa
import numpy as np
import soundfile as sf
import tdseries as td
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import IterableDataset, get_worker_info

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a project dependency.
    load_dotenv = None

if load_dotenv is not None:
    # Let configs use ${oc.env:...} values from the project .env while still
    # respecting variables already provided by the shell/job launcher.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from data_processing.dregon import clean_command_spikes, load_dregon_timeframes
from data_processing.frames import audio_series, get_meta, resample_audio_series, with_meta
from data_processing.michaels import load_michaels_timeframes
from data_processing.streams import (
    iter_published_frames,
    open_repository,
    resolve_source,
    sample_to_frame,
)
from data_processing.time_warp import (
    WarpParams,
    apply_time_warp,
    sample_warp_params,
    source_duration_s,
)


@runtime_checkable
class NoisePool(Protocol):
    """A source of aligned noise slices: real recordings or a generated stream."""

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame: ...


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
    payload = f"{int(base_seed)}:{int(global_sample_id)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    return np.random.default_rng(seed)


def _resolve_motor_tracks(tf: td.Frame) -> tuple[str, str, bool]:
    """Return ``(detect_key, rps_key, needs_cleaning)`` for a noise Frame."""
    if "motors_command" in tf or "motors_measured" in tf:
        detect = "motors_measured" if "motors_measured" in tf else "motors_command"
        rps_key = "motors_command" if "motors_command" in tf else "motors_measured"
        return detect, rps_key, True
    if "rps" in tf:
        return "rps", "rps", False
    raise ValueError(
        f"{get_meta(tf, 'recording_id', '?')} has no rotor-speed track "
        "(expected 'motors_measured', 'motors_command', or 'rps')"
    )


def _inflight_window(
    tf: td.Frame,
    motor_key: str,
    *,
    min_motor_rps: float,
    clean: bool,
) -> tuple[float, float]:
    motor = tf[motor_key]
    if motor.data is None or motor.dim_size("time") == 0:
        return motor.t_start, motor.t_end
    values = np.asarray(motor.data, dtype=np.float32)
    if clean:
        values = clean_command_spikes(values)
    mask = np.all(values > float(min_motor_rps), axis=0)
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        raise ValueError(
            f"No in-flight window (all motors > {min_motor_rps} RPS) in "
            f"{get_meta(tf, 'recording_id', '?')}"
        )
    times = cast(td.StampIndex, motor.tindex).abs_stamps
    return float(times[idxs[0]]), float(times[idxs[-1]])


def _as_audio_ct(audio: np.ndarray, *, target_len: int | None = None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 2:
        # Frame audio is already (C, T); soundfile audio is normalized before
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


def _extract_audio_array(tf: td.Frame, *, target_len: int) -> np.ndarray:
    audio = tf["audio"]
    return _as_audio_ct(audio.data, target_len=target_len)


def _stft_frame_times(audio: td.Series, n_frames: int, hop_length: int) -> np.ndarray:
    # Match the existing training target convention: frame i is placed at
    # t_start + i * hop / sr.  Built from the exact sr/hop frame-rate fraction
    # (never a float division) via a throwaway GridIndex, then read back as
    # absolute sample-edge times.
    ti = cast(td.GridIndex, audio.tindex)
    frame_rate = Fraction(ti.sr_num, ti.sr_den) / hop_length
    grid = td.GridIndex.create(
        (frame_rate.numerator, frame_rate.denominator), n_frames, t_start=ti.t_start_ticks
    )
    return grid.sample_times()


def interpolate_rps_to_stft_grid(
    tf: td.Frame,
    *,
    n_frames: int,
    hop_length: int,
) -> np.ndarray:
    """Interpolate a sliced Frame's RPS track to the model STFT grid."""
    _, rps_key, needs_clean = _resolve_motor_tracks(tf)
    audio = tf["audio"]
    rps = tf[rps_key]
    if rps.data is None or rps.dim_size("time") == 0:
        return np.zeros((4, n_frames), dtype=np.float32)

    frame_times = _stft_frame_times(audio, n_frames, hop_length)
    if needs_clean:
        rps = rps.map_data(clean_command_spikes)
    return rps.interpolate(frame_times).astype(np.float32)


#: Rotor-track entry names recognised in published rich frames, in preference
#: order: the generic ``rps`` (michaels-frames), then DREGON-frames'
#: ``motors_command`` (the canonical, *already cleaned* track — mirrors the
#: ``kind: dregon`` rps-key choice), then ``motors_measured``.
_PUBLISHED_RPS_KEYS = ("rps", "motors_command", "motors_measured")


def _adapt_published_frame(frame: td.Frame, *, sample_rate: int) -> td.Frame | None:
    """Rich published recording -> the minimal (audio + rps) Frame the pool slices.

    Published rich-frame datasets (``scripts/publish_frame_datasets.py``:
    ``DREGON-frames`` / ``michaels-frames``) carry their fixes baked in —
    DREGON's ``motors_command`` is already ``clean_command_spikes``-cleaned and
    michaels' ``rps`` is already aligned. The rotor track is therefore stored
    under the generic ``rps`` name, which ``_resolve_motor_tracks`` treats as
    needing **no** cleaning, so no fix logic is re-applied at load time.
    Everything else (IMU, raw telemetry, geometry, per-sample clocks) is
    dropped so the pool keeps only what it slices; audio is soxr-resampled to
    the pool ``sample_rate`` (same handling as the folder loaders'
    ``target_sr``). Returns ``None`` for frames without audio or a rotor track
    (e.g. clean-source recordings).
    """
    if "audio" not in frame:
        return None
    rps_key = next((k for k in _PUBLISHED_RPS_KEYS if k in frame), None)
    if rps_key is None:
        return None
    entries: dict[str, Any] = {
        "audio": resample_audio_series(cast(td.Series, frame["audio"]), sample_rate),
        "rps": frame[rps_key],
    }
    if "meta" in frame:
        entries["meta"] = frame["meta"]
    return td.Frame(entries)


def _iter_published_noise_frames(
    dataset: str,
    version: str | None,
    *,
    sample_rate: int,
    splits: list[str] | None,
    recording_ids: set[str] | None,
    exclude_recording_ids: set[str] | None,
    take: int | None,
) -> Iterator[td.Frame]:
    """Lazily adapt a published ``tdframe-v1`` dataset for the noise pool.

    One rich frame is decoded at a time and immediately reduced by
    :func:`_adapt_published_frame`, so the pool never holds more than one
    full recording's extra telemetry in memory.
    """
    kept = 0
    for frame in iter_published_frames(dataset, version, splits=splits):
        rid = str(get_meta(frame, "recording_id", ""))
        if recording_ids is not None and rid not in recording_ids:
            continue
        if exclude_recording_ids is not None and rid in exclude_recording_ids:
            continue
        adapted = _adapt_published_frame(frame, sample_rate=sample_rate)
        if adapted is None:
            continue
        yield adapted
        kept += 1
        if take is not None and kept >= int(take):
            return


class TimeFrameNoisePool:
    """Randomly slice existing aligned noise ``td.Frame`` recordings."""

    def __init__(
        self,
        recordings: Iterable[td.Frame],
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
                audio = tf["audio"]
                detect = tf[detect_key]
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
                print(f"Warning: skipping noise frame {get_meta(tf, 'recording_id', '?')}: {exc}")

        if not self.records:
            raise ValueError("no usable noise recordings found")
        weights = np.array(
            [r["valid_end"] - r["valid_start"] for r in self.records], dtype=np.float64
        )
        self.weights = weights / weights.sum()

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> TimeFrameNoisePool:
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
        # `root` may be a plain path (unchanged behaviour) or a `dload:NAME`
        # URI materialized to a local tree first.
        root = resolve_source(_cfg_get(cfg, "root", "data"))
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
                frames = [tf for tf in frames if str(get_meta(tf, "recording_id", "")) in wanted]
            exclude_ids = _cfg_get(cfg, "exclude_recording_ids", None)
            if exclude_ids is not None:
                excluded = {str(x) for x in exclude_ids}
                frames = [
                    tf for tf in frames if str(get_meta(tf, "recording_id", "")) not in excluded
                ]
        elif kind in {"michaels", "michael"}:
            frames = load_michaels_timeframes(data_root=root, sr=sample_rate)
            ids = _cfg_get(cfg, "ids", _cfg_get(cfg, "id", "all"))
            if isinstance(ids, str):
                ids = [ids]
            want_all = any(str(i).strip().lower() == "all" for i in ids)
            wanted = {str(i).strip().lower().removeprefix("fly") for i in ids}
            selected = []
            for tf in frames:
                rid = str(get_meta(tf, "recording_id", ""))
                if want_all or rid.lower().removeprefix("fly") in wanted:
                    selected.append(with_meta(tf, recording_id=f"michaels_FLY{rid}"))
            exclude_ids = _cfg_get(cfg, "exclude_recording_ids", None)
            if exclude_ids is not None:
                excluded = {str(x) for x in exclude_ids}
                selected = [
                    tf for tf in selected if str(get_meta(tf, "recording_id", "")) not in excluded
                ]
            frames = selected
        elif kind == "frames":
            # Published rich-frame dataset (tdframe-v1; see
            # scripts/publish_frame_datasets.py). Fixes are baked in at publish
            # time — nothing is re-cleaned here (`_adapt_published_frame`).
            dataset = _cfg_get(cfg, "dataset", None)
            if not dataset:
                raise ValueError("noise source kind 'frames' requires a 'dataset' name")
            splits = _cfg_get(cfg, "splits", None)
            split = _cfg_get(cfg, "split", None)
            if splits is None and split is not None:
                splits = [split]
            ids = _cfg_get(cfg, "recording_ids", None)
            exclude_ids = _cfg_get(cfg, "exclude_recording_ids", None)
            take = _cfg_get(cfg, "take", None)
            frames = _iter_published_noise_frames(
                str(dataset),
                _cfg_get(cfg, "version", None),
                sample_rate=sample_rate,
                splits=[str(s) for s in splits] if splits is not None else None,
                recording_ids={str(x) for x in ids} if ids is not None else None,
                exclude_recording_ids=(
                    {str(x) for x in exclude_ids} if exclude_ids is not None else None
                ),
                take=int(take) if take is not None else None,
            )
        else:
            raise ValueError(f"unsupported noise source kind: {kind!r}")
        return cls(frames, min_motor_rps=min_motor_rps, duration_s=duration_s)

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        idx = int(rng.choice(len(self.records), p=self.weights))
        rec = self.records[idx]
        start = float(rng.uniform(rec["valid_start"], rec["valid_end"] - duration_s))
        tf = cast(td.Frame, rec["tf"])
        return tf.time[start : start + duration_s]


class MixedNoisePool:
    """Weight-sample among several noise sub-pools, delegating the slice.

    Lets a heterogeneous ``sources.noise`` list — e.g. real recordings plus a
    :class:`data_processing.generated_noise.GeneratedNoisePool` — behave as one
    pool. Each sub-pool has a relative ``weight`` (a generated, infinite pool has
    no natural duration, so its weight is explicit).
    """

    def __init__(self, pools: list[Any], weights: list[float]):
        if not pools:
            raise ValueError("MixedNoisePool needs at least one sub-pool")
        self.pools = list(pools)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.pools),) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be one non-negative value per pool, summing > 0")
        self.weights = w / w.sum()

    @property
    def records(self) -> list[dict[str, Any]]:
        # Aggregate real sub-pool records (generated pools contribute none), so
        # helpers that introspect `.records` (e.g. drone-name discovery) still work.
        recs: list[dict[str, Any]] = []
        for pool in self.pools:
            recs.extend(getattr(pool, "records", []))
        return recs

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        idx = int(rng.choice(len(self.pools), p=self.weights))
        return self.pools[idx].sample_timeframe(rng, duration_s)


def build_noise_pool(cfg: Any, *, duration_s: float, sample_rate: int):
    """Build a noise pool from a source spec (or list), dispatching on ``kind``.

    Real sources (``dregon``/``michaels``) are merged into one
    :class:`TimeFrameNoisePool` (duration-weighted across recordings, unchanged).
    Any ``kind: generated`` source becomes a
    :class:`data_processing.generated_noise.GeneratedNoisePool`; when generated
    and real sources are mixed, they are combined in a :class:`MixedNoisePool`
    with pool-level ``weight`` (default ``1.0`` per source item — so a bare
    ``[dregon, generated]`` list is a 50/50 mix). The pure-real path is
    byte-for-byte the old behaviour.
    """
    cfg = _to_plain(cfg)
    items = list(cfg) if isinstance(cfg, list) else [cfg]
    # Standalone pools, each with its own pool class (not merged into the shared
    # duration-weighted TimeFrameNoisePool):
    #   kind: generated    -> trained PositionalHarmonicNoiseGen (GeneratedNoisePool)
    #   kind: static_comb  -> analytic static-comb model (StaticCombNoisePool, E8)
    #   kind: gp           -> egonoise-GP coefficient table (GPRotorNoisePool, G3)
    #   kind: audio_pool   -> lazy dload-backed audio dataset (DloadAudioPool, F1 SE)
    standalone_kinds = {"generated", "static_comb", "gp", "audio_pool"}
    standalone_items = [c for c in items if _cfg_get(c, "kind") in standalone_kinds]
    if not standalone_items:
        return TimeFrameNoisePool.from_config(cfg, duration_s=duration_s, sample_rate=sample_rate)

    def _build_standalone(c: Any):
        kind = _cfg_get(c, "kind")
        if kind == "generated":
            from data_processing.generated_noise import GeneratedNoisePool

            return GeneratedNoisePool.from_config(c, duration_s=duration_s, sample_rate=sample_rate)
        if kind == "audio_pool":
            return DloadAudioPool.from_config(c, duration_s=duration_s, sample_rate=sample_rate)
        if kind == "gp":
            from data_processing.gp_noise import GPRotorNoisePool

            return GPRotorNoisePool.from_config(c, duration_s=duration_s, sample_rate=sample_rate)
        from data_processing.rotor_spectral_model import StaticCombNoisePool

        return StaticCombNoisePool.from_config(c, duration_s=duration_s, sample_rate=sample_rate)

    real_items = [c for c in items if _cfg_get(c, "kind") not in standalone_kinds]
    pools: list[Any] = []
    weights: list[float] = []
    if real_items:
        pools.append(
            TimeFrameNoisePool.from_config(
                real_items, duration_s=duration_s, sample_rate=sample_rate
            )
        )
        weights.append(sum(float(_cfg_get(c, "weight", 1.0)) for c in real_items))
    for c in standalone_items:
        pools.append(_build_standalone(c))
        weights.append(float(_cfg_get(c, "weight", 1.0)))
    if len(pools) == 1:
        return pools[0]
    return MixedNoisePool(pools, weights)


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
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> AudioFileSourcePool:
        cfg = _to_plain(cfg)
        root = resolve_source(_cfg_get(cfg, "root", "."))
        globs = _cfg_get(cfg, "globs", None)
        glob_one = _cfg_get(cfg, "glob", None)
        if globs is None:
            globs = [glob_one] if glob_one is not None else ["**/*.flac", "**/*.wav"]
        files: list[Path] = []
        for pattern in globs:
            files.extend(
                p for p in root.glob(str(pattern)) if p.suffix.lower() in cls.AUDIO_SUFFIXES
            )
        files = sorted(set(files))
        # Leak-free speaker split: drop any file whose path contains one of the
        # held-out tokens (e.g. LibriSpeech speaker ids reserved for the SE valid
        # set). Matched as path-string substrings so ``/103/`` etc. work.
        exclude = _cfg_get(cfg, "exclude", None) or _cfg_get(cfg, "exclude_speakers", None)
        if exclude:
            tokens = [str(t) for t in exclude]
            files = [p for p in files if not any(f"/{t}/" in f"/{p.as_posix()}/" for t in tokens)]
            if not files:
                raise ValueError("source pool empty after applying 'exclude' filter")
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
            kept: list[Path] = []
            skipped: list[str] = []
            with open(tmp_data, "wb") as f:
                for i, path in enumerate(self.files, start=1):
                    try:
                        audio = self._load_one(path)
                    except Exception as exc:
                        # Corpora ship with the odd unreadable file (e.g. a
                        # corrupt flac in the librispeech dload copy); one bad
                        # file must not kill a multi-hour cache build.
                        print(f"Warning: skipping unreadable source file {path}: {exc}")
                        skipped.append(str(path))
                        continue
                    audio_i16 = np.clip(audio, -1.0, 1.0)
                    audio_i16 = np.round(audio_i16 * 32767.0).astype(np.int16)
                    offsets.append((offset, int(audio_i16.shape[0])))
                    f.write(audio_i16.tobytes(order="C"))
                    offset += int(audio_i16.shape[0])
                    kept.append(path)
                    if i == 1 or i % 1000 == 0 or i == len(self.files):
                        print(f"  cached {i}/{len(self.files)} source files")
            self.files = kept
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
                # Recorded so the reuse path drops the same files: the packed
                # index rows must stay aligned with self.files.
                "skipped": skipped,
            }
            tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            tmp_data.replace(data_path)
            tmp_index.replace(index_path)
            tmp_manifest.replace(manifest_path)
        else:
            print(f"Reusing source cache at {cache_path}")
            manifest_skipped = set(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("skipped", [])
            )
            if manifest_skipped:
                self.files = [p for p in self.files if str(p) not in manifest_skipped]

        packed_index = np.load(index_path)
        if len(packed_index) != len(self.files):
            raise ValueError(
                f"packed cache at {cache_path} has {len(packed_index)} entries but "
                f"{len(self.files)} source files resolved — stale/foreign cache?"
            )
        self._packed_index = packed_index
        total_samples = int(packed_index[:, 1].sum())
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
            audio = (
                self._packed_data[int(offset) : int(offset + length)].astype(np.float32) / 32767.0
            )
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


class DloadAudioPool:
    """Telemetry-free noise pool over an arbitrary dload dataset (``kind: audio_pool``).

    Streams audio lazily at *shard* granularity: one shard (~128 MB) is pinned
    and read through dload's ``PackReader``, giving O(1) random access to every
    sample it holds before the next shard is fetched. The whole dataset is never
    materialized, so this works on the 258 GiB ``MIMII`` / 88 GiB
    ``DroneAudioSet`` pools where :class:`TimeFrameNoisePool` (which holds every
    recording in RAM) would OOM.

    Each draw picks a random *recording* (shard weighted by its sample count,
    then a uniform sample within the shard), a random channel for multichannel
    audio, resamples to the pool ``sample_rate`` (soxr_hq), and loops/pads to the
    requested duration. The returned Frame has only an ``audio`` track (no rotor
    telemetry) — usable for ``speech_enhancement`` mixing, which skips RPS
    interpolation, but not for RPS prediction.

    Supports both published ``tdframe-v1`` datasets (audio under the ``audio``
    entry) and raw one-file-per-sample datasets (audio field named by extension,
    e.g. ``wav``/``flac``). Zip-blob datasets (``zenodo_drone_noises``) are not
    supported.
    """

    AUDIO_EXTS = ("wav", "flac", "ogg", "mp3")

    def __init__(
        self,
        dataset: str,
        *,
        version: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channel: str | int = "random",
        reader_cache: int = 2,
        holdout: Mapping[str, Any] | None = None,
        max_shards: int | None = None,
    ) -> None:
        self.dataset = str(dataset)
        self.version = version
        self.sample_rate = int(sample_rate)
        self.channel = channel
        self.reader_cache = max(1, int(reader_cache))
        # Cap the number of (post-holdout) shards actually streamed. A random
        # shard is drawn per sample, so an uncapped pool over a 2003-shard /
        # 258 GiB dataset (MIMII) would, over a full run, pull essentially the
        # whole dataset from R2 (coupon-collector) — infeasible I/O + cache
        # thrash. For a noise-*augmentation* pool a bounded, diverse shard
        # subset (e.g. 24 shards ≈ hundreds of recordings) is plenty. ``None`` =
        # all shards (fine for small datasets).
        self._max_shards = int(max_shards) if max_shards else None
        # Leak-free train/valid split. Preferred: reserve the last ``valid_shards``
        # *whole shards* (= whole recording groups) as the valid partition and
        # everything before as train — so the fixed SE valid set never shares a
        # recording with the training stream AND the valid build pulls at most
        # ``valid_shards`` shards even on 2003-shard MIMII. Fallback for datasets
        # with too few shards to reserve one (``n_shards <= valid_shards``, e.g.
        # single-shard KAIST/HUSTmotor): a per-shard sample-index split (first
        # ``1-fraction`` train, last ``fraction`` valid). ``None`` = use all.
        frac = 0.1
        side: str | None = None
        valid_shards = 1
        if holdout:
            frac = float(_cfg_get(holdout, "fraction", 0.1))
            if not 0.0 < frac < 1.0:
                raise ValueError(f"audio_pool holdout.fraction must be in (0,1), got {frac}")
            side = str(_cfg_get(holdout, "split", "train"))
            if side not in {"train", "valid"}:
                raise ValueError(f"audio_pool holdout.split must be 'train'/'valid', got {side!r}")
            valid_shards = max(1, int(_cfg_get(holdout, "valid_shards", 1)))

        repo = open_repository()
        manifest = repo.manifest(self.dataset, self.version)
        shards = list(manifest.shards)
        if not shards:
            raise ValueError(f"audio_pool dataset {self.dataset!r} has no shards")
        meta = getattr(manifest, "meta", None) or {}
        self._layout = meta.get("layout")
        self.num_samples = int(getattr(manifest, "num_samples", 0)) or sum(
            int(s.num_samples) for s in shards
        )

        # Apply the shard-level split (or arm the index-split fallback).
        self._holdout_frac: float | None = None
        self._holdout_side: str | None = None
        if side is not None and len(shards) > valid_shards:
            shards = shards[-valid_shards:] if side == "valid" else shards[:-valid_shards]
        elif side is not None:
            self._holdout_frac = frac  # single-shard fallback: split by sample index
            self._holdout_side = side

        # Bound the streamed shard set (train side) to keep the total R2 pull
        # feasible; the valid side already reserves only ``valid_shards``.
        if self._max_shards is not None and side != "valid" and len(shards) > self._max_shards:
            shards = shards[: self._max_shards]

        # Plain, picklable state (crosses the DataLoader fork); the shard structs
        # are dataclasses of str/int.
        self._shards = shards
        nsamp = np.array([max(1, int(s.num_samples)) for s in shards], dtype=np.float64)
        self._shard_weights = nsamp / nsamp.sum()
        # Per-process handles, (re)created lazily so they never cross the fork.
        self._pid: int | None = None
        self._repo: Any = None
        self._lru: OrderedDict[str, tuple[Any, Any]] | None = None

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> DloadAudioPool:
        cfg = _to_plain(cfg)
        dataset = _cfg_get(cfg, "dataset", None)
        if not dataset:
            raise ValueError("noise source kind 'audio_pool' requires a 'dataset' name")
        # `duration_s` is accepted for a uniform pool-builder signature; the slice
        # length is taken from the per-call `sample_timeframe(duration_s)` instead.
        return cls(
            str(dataset),
            version=_cfg_get(cfg, "version", None),
            sample_rate=sample_rate,
            channel=_cfg_get(cfg, "channel", "random"),
            reader_cache=int(_cfg_get(cfg, "reader_cache", 2)),
            holdout=_cfg_get(cfg, "holdout", None),
            max_shards=_cfg_get(cfg, "max_shards", None),
        )

    def _index_range(self, n: int) -> tuple[int, int]:
        """The [lo, hi) sample-index window active for this pool's holdout side."""
        if self._holdout_frac is None or n <= 1:
            return 0, n
        cut = int(np.floor(n * (1.0 - self._holdout_frac)))
        cut = min(max(1, cut), n - 1)  # both sides non-empty when n >= 2
        return (cut, n) if self._holdout_side == "valid" else (0, cut)

    def _ensure_process(self) -> None:
        pid = os.getpid()
        if self._pid != pid:
            self._pid = pid
            self._repo = open_repository()
            self._lru = OrderedDict()

    def _reader(self, shard_idx: int):
        from dload.pack import PackReader

        self._ensure_process()
        assert self._lru is not None
        shard = self._shards[shard_idx]
        cached = self._lru.get(shard.digest)
        if cached is not None:
            self._lru.move_to_end(shard.digest)
            return cached[1]
        pin = self._repo.open_shard(shard)
        reader = PackReader(pin.path)
        self._lru[shard.digest] = (pin, reader)
        while len(self._lru) > self.reader_cache:
            _digest, (old_pin, old_reader) = self._lru.popitem(last=False)
            try:
                old_reader.close()
            finally:
                old_pin.release()
        return reader

    def _decode(self, key: str, fields: Mapping[str, bytes]) -> tuple[np.ndarray, int] | None:
        """Decode one sample to ``((C, T), sr)``, or ``None`` if it holds no audio.

        Some datasets interleave non-audio samples (e.g. ``new-drone-noises``
        ships flight-log ``csv`` samples alongside ``wav``); the caller redraws
        on ``None`` rather than crashing.
        """
        if self._layout == "tdframe-v1":
            frame = sample_to_frame(fields)
            if "audio" not in frame:
                return None
            series = frame["audio"]
            arr = np.asarray(series.data, dtype=np.float32)
            sr = int(cast(td.GridIndex, series.tindex).sr)
            return arr, sr
        ext = next((e for e in self.AUDIO_EXTS if e in fields), None)
        if ext is None:
            return None
        raw, sr = sf.read(io.BytesIO(fields[ext]), dtype="float32", always_2d=True)  # (T, C)
        return np.ascontiguousarray(raw.T), int(sr)  # (C, T)

    def _fit_length(
        self, mono: np.ndarray, target_len: int, rng: np.random.Generator
    ) -> np.ndarray:
        length = int(mono.shape[0])
        if length == 0:
            return np.zeros(target_len, dtype=np.float32)
        if length == target_len:
            return np.ascontiguousarray(mono, dtype=np.float32)
        if length > target_len:
            start = int(rng.integers(0, length - target_len + 1))
            return np.ascontiguousarray(mono[start : start + target_len], dtype=np.float32)
        reps = int(np.ceil(target_len / length))
        tiled = np.tile(mono, reps)
        start = int(rng.integers(0, tiled.shape[0] - target_len + 1))
        return np.ascontiguousarray(tiled[start : start + target_len], dtype=np.float32)

    def _sample_mono(self, rng: np.random.Generator, target_len: int) -> np.ndarray:
        decoded = None
        for _ in range(32):  # redraw past non-audio samples (e.g. csv flight logs)
            shard_idx = int(rng.choice(len(self._shards), p=self._shard_weights))
            reader = self._reader(shard_idx)
            n = len(reader.keys)
            lo, hi = self._index_range(n)
            idx = int(rng.integers(lo, hi))
            key, fields = reader.read(idx)
            decoded = self._decode(key, fields)
            if decoded is not None:
                break
        if decoded is None:
            raise ValueError(f"audio_pool {self.dataset!r}: no audio sample found in 32 draws")
        arr, sr = decoded
        if arr.ndim == 2:
            if isinstance(self.channel, str) and self.channel == "random":
                c = int(rng.integers(0, arr.shape[0]))
            else:
                c = int(self.channel) % arr.shape[0]
            mono = arr[c]
        else:
            mono = arr
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        if sr != self.sample_rate:
            mono = librosa.resample(
                mono, orig_sr=float(sr), target_sr=self.sample_rate, res_type="soxr_hq"
            )
        return self._fit_length(mono, target_len, rng)

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        target_len = int(round(duration_s * self.sample_rate))
        mono = self._sample_mono(rng, target_len)
        return td.Frame({"audio": audio_series(mono[None, :], self.sample_rate)})


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


def _scale_source_to_snr(
    source: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    per_channel: bool = False,
) -> np.ndarray:
    """Return ``source * scale`` — the clean source as it appears in the mixture.

    ``scale`` is chosen so that ``source * scale`` sits at ``snr_db`` relative to
    ``noise`` (globally, or per channel when ``per_channel``). The mixture is
    then ``noise + scaled_source``; the *scaled* source is the correct clean
    reference for speech-enhancement targets (SI-SDR is computed against exactly
    the speech component present in the mixture, up to added augmentation).
    """
    eps = 1e-12
    if per_channel:
        noise_power = np.mean(noise.astype(np.float64) ** 2, axis=1, keepdims=True)
        source_power = np.mean(source.astype(np.float64) ** 2, axis=1, keepdims=True)
    else:
        noise_power = np.array([[np.mean(noise.astype(np.float64) ** 2)]])
        source_power = np.array([[np.mean(source.astype(np.float64) ** 2)]])
    scale = np.sqrt((noise_power * (10.0 ** (float(snr_db) / 10.0))) / (source_power + eps))
    return (source * scale.astype(np.float32)).astype(np.float32)


def _mix_at_source_to_noise_snr(
    source: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    per_channel: bool = False,
) -> np.ndarray:
    return (noise + _scale_source_to_snr(source, noise, snr_db, per_channel=per_channel)).astype(
        np.float32
    )


def _maybe_sample_time_warp(
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> WarpParams | None:
    """Fire-and-sample the noise time-warp, mirroring ``_apply_one_augmentation``.

    Draws the single fire-decision random only when ``spec`` is present with a
    positive probability (Python ``and`` short-circuits, so an absent key or
    ``probability <= 0`` consumes no RNG and keeps the stream byte-identical to
    the un-warped path). On a hit, the warp parameters are then drawn.
    """
    if not spec:
        return None
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return None
    return sample_warp_params(spec, rng)


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


def _apply_one_augmentation_pair(
    mixture: np.ndarray,
    target: np.ndarray,
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one augmentation to the ``(mixture, clean target)`` pair *identically*.

    For speech enhancement the clean target is the speech component present in
    the mixture, so any post-mix transform must hit both signals with the same
    parameters — otherwise the pair stops being consistent (SI-SDR would be
    computed against a differently-scaled reference). ``random_gain`` and
    ``random_polarity`` are scalar multiplications applied to both; ``channel_drop``
    zeros the same channels in both (a no-op for the mono SE stream). Draws the
    same RNG sequence as :func:`_apply_one_augmentation` so behaviour is
    predictable. Unsupported augmentations raise (SE configs list only the
    supported ones).
    """
    if not spec:
        return mixture, target
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return mixture, target
    choices = list(spec.get("choices", []))
    if not choices:
        return mixture, target
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

    mix = mixture.copy()
    tgt = target.copy()
    if name == "random_gain":
        min_db = float(params.get("min_db", -6.0))
        max_db = float(params.get("max_db", 6.0))
        gain = np.float32(10.0 ** (float(rng.uniform(min_db, max_db)) / 20.0))
        mix *= gain
        tgt *= gain
    elif name == "random_polarity":
        mix *= -1.0
        tgt *= -1.0
    elif name == "channel_drop":
        if mix.shape[0] <= 1:
            return mix, tgt
        max_channels = int(params.get("max_channels", 1))
        n_drop = int(rng.integers(1, min(max_channels, mix.shape[0] - 1) + 1))
        drop = rng.choice(mix.shape[0], size=n_drop, replace=False)
        mix[drop, :] = 0.0
        tgt[drop, :] = 0.0
    else:
        raise ValueError(f"unsupported SE augmentation: {name!r}")
    return mix.astype(np.float32, copy=False), tgt.astype(np.float32, copy=False)


class OnlineMixIterableDataset(IterableDataset):
    """Infinite online-mixing stream yielding ``(audio, rps_target)`` tensors.

    With ``task="speech_enhancement"`` the stream instead yields
    ``(mixture, clean_target)`` — the clean target being the gain-scaled speech
    exactly as mixed (plus any post-mix augmentation) — and RPS interpolation is
    skipped entirely (so telemetry-free ``kind: audio_pool`` noise sources work).
    """

    def __init__(
        self,
        noise_pool: NoisePool,
        source_pool: AudioFileSourcePool | None,
        *,
        policy: Mapping[str, Any] | None = None,
        base_seed: int = 1234,
        duration_s: float = DEFAULT_DURATION_S,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        start_sample_id: int = 0,
        task: str = "rps_prediction",
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
        self.task = str(task)
        if self.task not in {"rps_prediction", "speech_enhancement"}:
            raise ValueError(f"unsupported online-mix task: {self.task!r}")
        if self.task == "speech_enhancement" and self.source_pool is None:
            raise ValueError("speech_enhancement online mixing requires a sources.speech pool")
        self.target_len = int(round(self.duration_s * self.sample_rate))

    @classmethod
    def from_config(cls, cfg: Any) -> OnlineMixIterableDataset:
        cfg = _to_plain(cfg)
        sample_rate = int(_cfg_get(cfg, "sample_rate", DEFAULT_SAMPLE_RATE))
        duration_s = float(_cfg_get(cfg, "duration_s", DEFAULT_DURATION_S))
        n_fft = int(_cfg_get(cfg, "n_fft", DEFAULT_N_FFT))
        hop_length = int(_cfg_get(cfg, "hop_length", DEFAULT_HOP_LENGTH))
        base_seed = int(_cfg_get(cfg, "base_seed", 1234))
        start_sample_id = int(_cfg_get(cfg, "start_sample_id", 0))
        task = str(_cfg_get(cfg, "task", "rps_prediction"))
        policy = cast(Mapping[str, Any], _cfg_get(cfg, "policy", {}))

        sources = _cfg_get(cfg, "sources", {})
        noise_cfg = _cfg_get(sources, "noise", None)
        if noise_cfg is None:
            raise ValueError("online mix config requires sources.noise")
        noise_pool = build_noise_pool(noise_cfg, duration_s=duration_s, sample_rate=sample_rate)

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
            task=task,
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
        if self.task == "speech_enhancement":
            return self._generate_se_sample(global_sample_id)
        return self._generate_rps_sample(global_sample_id)

    def _generate_rps_sample(self, global_sample_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = make_rng(self.base_seed, int(global_sample_id))
        policy = _resolve_policy(self.policy, int(global_sample_id))

        # Time-varying time-warp of the noise+RPS pair (before extraction/mixing).
        # The single fire decision is drawn exactly like `_apply_one_augmentation`:
        # when the key is absent or probability 0, no rng is consumed here and the
        # downstream stream is byte-identical to the un-warped path.
        warp = _maybe_sample_time_warp(
            cast("Mapping[str, Any] | None", policy.get("noise_time_warp")), rng
        )
        if warp is not None:
            noise_tf = self.noise_pool.sample_timeframe(
                rng, source_duration_s(self.duration_s, warp)
            )
            noise_tf = apply_time_warp(
                noise_tf,
                warp,
                target_len=self.target_len,
                sample_rate=self.sample_rate,
            )
        else:
            noise_tf = self.noise_pool.sample_timeframe(rng, self.duration_s)
        audio_track = noise_tf["audio"]
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
        audio_sr = cast(td.GridIndex, audio_track.tindex).sr
        if int(round(audio_sr)) != self.sample_rate:
            raise ValueError(f"noise audio sr {audio_sr} != configured {self.sample_rate}")
        return torch.from_numpy(mixture), torch.from_numpy(rps)

    def _generate_se_sample(self, global_sample_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Yield ``(mixture, clean_target)`` for speech-enhancement training.

        The clean target is the speech component exactly as mixed into the
        mixture (gain-scaled to the drawn SNR, plus any post-mix augmentation
        applied identically to both). No RPS is interpolated, so telemetry-free
        ``kind: audio_pool`` noise sources are supported. Speech is always drawn
        (a clean reference is required); ``source_prob`` is ignored here.
        """
        assert self.source_pool is not None  # enforced in __init__
        rng = make_rng(self.base_seed, int(global_sample_id))
        policy = _resolve_policy(self.policy, int(global_sample_id))

        noise_tf = self.noise_pool.sample_timeframe(rng, self.duration_s)
        audio_track = noise_tf["audio"]
        audio_sr = cast(td.GridIndex, audio_track.tindex).sr
        if int(round(audio_sr)) != self.sample_rate:
            raise ValueError(f"noise audio sr {audio_sr} != configured {self.sample_rate}")
        noise_audio = _extract_audio_array(noise_tf, target_len=self.target_len)
        # The SE stream is single-channel: pick a random mic from multichannel
        # noise sources (DREGON/Michael's 8-ch frames) so the mixture/target are
        # mono (1, T) — the codec's mono speech-enhancement contract.
        if noise_audio.shape[0] > 1:
            ch = int(rng.integers(0, noise_audio.shape[0]))
            noise_audio = np.ascontiguousarray(noise_audio[ch : ch + 1])

        source = self.source_pool.sample_array(rng, channels=1, mode="independent")
        snr_db = _sample_snr_db(policy, rng)
        per_channel = bool(policy.get("snr_per_channel", False))
        scaled_source = _scale_source_to_snr(source, noise_audio, snr_db, per_channel=per_channel)
        mixture = (noise_audio + scaled_source).astype(np.float32)

        mixture, target = _apply_one_augmentation_pair(
            mixture,
            scaled_source,
            cast(Mapping[str, Any] | None, policy.get("augmentations")),
            rng,
        )
        return torch.from_numpy(np.ascontiguousarray(mixture)), torch.from_numpy(
            np.ascontiguousarray(target)
        )
