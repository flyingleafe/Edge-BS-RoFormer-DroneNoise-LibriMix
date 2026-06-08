"""
Combined chunkable noise+RPS dataset for training generative models.

Pulls in-flight noise recordings from:
- DREGON (`in_flight_noise` split, has motor telemetry @ ~929 Hz)
- Michael's set (`data/new-drone-noises/`, motor telemetry @ ~30 Hz)

Each `__getitem__` returns:
  - rps_audio_rate: torch.FloatTensor (4, T)  — RPS upsampled to audio rate, in Hz
  - audio_target:   torch.FloatTensor (T,)    — real recorded drone noise

This format is ready for `models.generative.DroneNoisePlusFilterGen`:
  generator(rps_audio_rate.unsqueeze(0)) -> {'audio': pred, ...}
  loss(pred, audio_target.unsqueeze(0))
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from utils.data import TimeFrame

from . import michaels as M
from . import dregon as D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upsample_rps_to_audio_rate(
    rps: np.ndarray,
    motor_ts: np.ndarray,
    audio_ts: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate motor speeds onto the audio sample grid.

    DREGON in particular has a few duplicated motor timestamps; we drop them
    before interpolation to keep `interp1d` from producing NaNs (which would
    poison gradients downstream).

    Args:
        rps:      (4, M) motor speeds in Hz at motor timestamps `motor_ts`.
        motor_ts: (M,) timestamps for the motor samples.
        audio_ts: (N,) timestamps where we want RPS values (one per audio sample).
    Returns:
        (4, N) RPS at the audio sample grid.
    """
    # Deduplicate timestamps (keep first occurrence).
    motor_ts = np.asarray(motor_ts, dtype=np.float64)
    _, uniq_idx = np.unique(motor_ts, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    motor_ts = motor_ts[uniq_idx]
    rps = rps[:, uniq_idx]
    # Clip audio_ts into the motor_ts range to avoid extrapolation NaNs.
    clipped = np.clip(audio_ts, motor_ts[0], motor_ts[-1])
    f = interp1d(motor_ts, rps, kind="linear", axis=-1, assume_sorted=True)
    return f(clipped).astype(np.float32)


# ---------------------------------------------------------------------------
# Unified record wrapper (DREGON → TimeFrame; Michael's → MichaelsRecord)
# ---------------------------------------------------------------------------

@dataclass
class _ChunkSource:
    record: object         # TimeFrame | MichaelsRecord
    origin: str            # "dregon" | "michaels"
    n_channels: int        # number of usable audio channels
    duration: float


def _wrap_dregon(tf: TimeFrame) -> _ChunkSource:
    audio = tf["audio"]
    n_ch = audio.samples.shape[0] if audio.samples.ndim > 1 else 1
    return _ChunkSource(record=tf, origin="dregon", n_channels=n_ch,
                        duration=audio.duration)


def _wrap_michaels(record: "M.MichaelsRecord") -> _ChunkSource:
    n_ch = record.audio.shape[1] if record.audio.ndim > 1 else 1
    return _ChunkSource(record=record, origin="michaels", n_channels=n_ch,
                        duration=record.duration)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoiseRPSDataset(Dataset):
    """In-memory chunkable dataset of drone-noise audio + aligned RPS.

    Args:
        records: list of pre-loaded `_ChunkSource`s (use the loader helpers
            below to build it).
        chunk_size: chunk length in audio samples (e.g. 16000 for 1 s @ 16 kHz).
        sample_rate: audio sample rate (must match the records).
        samples_per_epoch: virtual epoch size — items are drawn randomly each
            time, so __len__ controls dataloader iters per epoch.
        seed: optional seed for reproducible sampling.
        channel_policy: 'first' uses channel 0 always; 'random' picks a random
            channel from each record.
        rps_normalize: divide rps by this value before returning (useful for
            scale-stabilising downstream networks; the harmonic synth needs raw
            Hz, but small auxiliary networks benefit from normalisation).
            If 0/None, no normalisation. Note: the audio-rate RPS returned is
            always RAW (Hz). Use `rps_normalize` to also return a normalised
            copy via the `rps_norm` key.
    """

    def __init__(
        self,
        records: list[_ChunkSource],
        chunk_size: int,
        sample_rate: int = 16000,
        samples_per_epoch: int = 1024,
        seed: int | None = None,
        channel_policy: Literal["first", "random"] = "first",
        rps_normalize: float | None = None,
    ):
        if not records:
            raise ValueError("NoiseRPSDataset got no records")
        self.records = records
        self.chunk_size = int(chunk_size)
        self.sample_rate = int(sample_rate)
        self.samples_per_epoch = int(samples_per_epoch)
        self.channel_policy = channel_policy
        self.rps_normalize = rps_normalize
        self._chunk_duration_sec = self.chunk_size / self.sample_rate

        # Filter records that are too short to chunk.
        self.records = [r for r in self.records if r.duration >= self._chunk_duration_sec]
        if not self.records:
            raise ValueError(
                f"All records shorter than chunk_size={chunk_size} "
                f"({self._chunk_duration_sec:.2f}s @ {sample_rate} Hz)"
            )

        # Length-weighted sampling probabilities (favour longer records).
        weights = np.array([r.duration for r in self.records], dtype=np.float64)
        self.weights = weights / weights.sum()

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.samples_per_epoch

    def _extract_chunk(self, src: _ChunkSource) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (audio [T], rps_motor_rate [4, M], audio_ts [T], motor_ts [M])."""
        rec = src.record
        # Random channel within record
        if self.channel_policy == "random" and src.n_channels > 1:
            ch = int(self.rng.integers(0, src.n_channels))
        else:
            ch = 0

        if src.origin == "dregon":
            # rec is a TimeFrame — use native slice + track access.
            tf: TimeFrame = rec
            # Compute valid time window (intersection of audio and motor domains).
            audio_start = tf["audio"].t_start
            audio_end = tf["audio"].t_end
            motor_track = tf["motors_measured"]
            motor_start = motor_track.t_start
            motor_end = motor_track.t_end
            valid_start = max(audio_start, motor_start)
            valid_end = min(audio_end, motor_end)
            rel_lo = valid_start - audio_start
            rel_hi = valid_end - audio_start - self._chunk_duration_sec
            start = float(self.rng.uniform(rel_lo, rel_hi))

            # Slice both audio and motors simultaneously.
            sliced = tf.slice(audio_start + start,
                              audio_start + start + self._chunk_duration_sec)

            audio_us = sliced["audio"]
            # UniformSeries stores (channels, N) — axis 0 = channels, axis -1 = time
            audio = audio_us.samples[ch, :] if audio_us.samples.ndim > 1 else audio_us.samples
            audio_ts = audio_us.sample_times()

            motor_es = sliced["motors_measured"]
            motor_ts = motor_es.abs_timestamps  # float seconds
            # EventSeries values are already time-last (4, M).
            rps = motor_es.values if motor_es.values is not None else np.zeros((4, 0), dtype=np.float32)
        else:  # michaels
            # Same logic as before, using MichaelsRecord API.
            rec_m: M.MichaelsRecord = rec
            audio_start = rec_m.audio_timestamps[0]
            audio_end = rec_m.audio_timestamps[-1]
            motor_start = rec_m.motors.timestamps[0]
            motor_end = rec_m.motors.timestamps[-1]
            valid_start = max(audio_start, motor_start)
            valid_end = min(audio_end, motor_end)
            rel_lo = valid_start - audio_start
            rel_hi = valid_end - audio_start - self._chunk_duration_sec
            start = float(self.rng.uniform(rel_lo, rel_hi))
            sliced = rec_m.slice_by_time(start, start + self._chunk_duration_sec)
            audio = sliced.audio[:, ch] if sliced.audio.ndim > 1 else sliced.audio
            rps = sliced.motors.measured.T
            motor_ts = sliced.motors.timestamps
            audio_ts = sliced.audio_timestamps

        # Length normalisation — chunks can be off by 1 sample due to int cast
        if len(audio) > self.chunk_size:
            audio = audio[: self.chunk_size]
            audio_ts = audio_ts[: self.chunk_size]
        elif len(audio) < self.chunk_size:
            # Pad audio + audio_ts (extrapolate by sample dt)
            pad = self.chunk_size - len(audio)
            dt = 1.0 / self.sample_rate
            audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
            audio_ts = np.concatenate(
                [audio_ts, audio_ts[-1] + dt * np.arange(1, pad + 1)]
            )
        return audio.astype(np.float32), rps.astype(np.float32), audio_ts.astype(np.float64), motor_ts.astype(np.float64)

    def __getitem__(self, idx: int):
        # idx is ignored — we pick a record randomly.
        src_idx = int(self.rng.choice(len(self.records), p=self.weights))
        src = self.records[src_idx]

        for _ in range(4):  # a few retries if a record fails for some reason
            try:
                audio, rps, audio_ts, motor_ts = self._extract_chunk(src)
                break
            except ValueError:
                src_idx = int(self.rng.choice(len(self.records), p=self.weights))
                src = self.records[src_idx]
        else:
            raise RuntimeError("Failed to extract a noise chunk after retries")

        # Upsample RPS to audio rate
        rps_audio = upsample_rps_to_audio_rate(rps, motor_ts, audio_ts)  # (4, T)
        rps_audio = torch.from_numpy(rps_audio)
        audio_t = torch.from_numpy(audio)

        if self.rps_normalize:
            rps_norm = rps_audio / float(self.rps_normalize)
            return {
                "rps": rps_audio,
                "rps_norm": rps_norm,
                "audio": audio_t,
                "origin": src.origin,
            }
        return {"rps": rps_audio, "audio": audio_t, "origin": src.origin}


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_dregon_noise_sources(
    dregon_dir: str | Path,
    sample_rate: int,
    cache_dir: str | Path | None = None,
) -> list[_ChunkSource]:
    """Load all DREGON `in_flight_noise` recordings with motor data.

    Uses the new TimeFrame-native loader.
    """
    dregon_dir = Path(dregon_dir)
    frames = D.load_dregon_timeframes(
        dregon_dir.parent,
        splits=["in_flight_noise"],
        target_sr=sample_rate,
        download=False,
    )
    sources: list[_ChunkSource] = []
    for tf in frames:
        if "motors_measured" not in tf:
            continue
        sources.append(_wrap_dregon(tf))
    return sources


def load_michaels_noise_sources(
    michaels_dir: str | Path,
    sample_rate: int,
) -> list[_ChunkSource]:
    """Load all Michael's recordings that exist in `michaels_dir`."""
    records = M.load_all_michaels_records(michaels_dir, sample_rate=sample_rate)
    return [_wrap_michaels(r) for r in records]


def build_noise_rps_datasets(
    *,
    dregon_dir: str | Path | None,
    michaels_dir: str | Path | None,
    sample_rate: int = 16000,
    chunk_size: int = 16000,
    train_samples: int = 4096,
    val_samples: int = 512,
    val_pct: float = 0.1,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    **dataset_kwargs,
) -> tuple[NoiseRPSDataset, NoiseRPSDataset]:
    """Build train/val NoiseRPSDataset by holding out a fraction of every record.

    Strategy: each record's *time axis* is split — the first `1 - val_pct`
    fraction is used for training, the last `val_pct` for validation. This
    keeps the same recording diversity in both splits while preventing data
    leakage (random chunks from disjoint time intervals are statistically
    almost-independent for our purposes).

    Args:
        dregon_dir: path to DREGON dataset (or None to skip).
        michaels_dir: path to Michael's recordings (or None to skip).
        sample_rate: target audio sample rate.
        chunk_size: chunk length in samples.
        train_samples / val_samples: virtual epoch sizes.
        val_pct: fraction of each recording held out at the end for validation.
        seed: RNG seed.
        cache_dir: where to cache resampled DREGON audio (kept for API compat).
        **dataset_kwargs: forwarded to `NoiseRPSDataset` (e.g. rps_normalize).
    """
    sources: list[_ChunkSource] = []
    if dregon_dir is not None:
        sources += load_dregon_noise_sources(dregon_dir, sample_rate, cache_dir=cache_dir)
    if michaels_dir is not None:
        sources += load_michaels_noise_sources(michaels_dir, sample_rate)

    if not sources:
        raise ValueError("No noise sources found (DREGON and Michael's both empty/absent)")

    train_sources: list[_ChunkSource] = []
    val_sources: list[_ChunkSource] = []
    for src in sources:
        cut = src.duration * (1.0 - val_pct)

        if src.origin == "dregon":
            tf: TimeFrame = src.record
            # Train: slice(0, cut)  |  Val: slice(cut, end)
            train_tf = tf.slice(tf.t_start, tf.t_start + cut)
            val_tf = tf.slice(tf.t_start + cut, tf.t_end)
            if train_tf["audio"].duration >= chunk_size / sample_rate:
                train_sources.append(_wrap_dregon(train_tf))
            if val_tf["audio"].duration >= chunk_size / sample_rate:
                val_sources.append(_wrap_dregon(val_tf))
        else:
            rec = src.record
            train_rec = rec.slice_by_time(0.0, cut)
            val_rec = rec.slice_by_time(cut, src.duration)
            if train_rec.duration >= chunk_size / sample_rate:
                train_sources.append(_wrap_michaels(train_rec))
            if val_rec.duration >= chunk_size / sample_rate:
                val_sources.append(_wrap_michaels(val_rec))

    train_ds = NoiseRPSDataset(
        train_sources, chunk_size=chunk_size, sample_rate=sample_rate,
        samples_per_epoch=train_samples, seed=seed, **dataset_kwargs,
    )
    val_ds = NoiseRPSDataset(
        val_sources, chunk_size=chunk_size, sample_rate=sample_rate,
        samples_per_epoch=val_samples, seed=seed + 1, **dataset_kwargs,
    )
    return train_ds, val_ds
