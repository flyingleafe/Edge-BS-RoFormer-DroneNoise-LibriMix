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
from typing import Any, Literal, cast

import numpy as np
import tdseries as td
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from . import dregon as D
from . import michaels as M
from .frames import resample_audio_series
from .streams import iter_published_frames

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
# Unified record wrapper — both DREGON and Michael's are now plain
# ``td.Frame``s with an "audio" entry (dims ("mic", "time")) and a
# rotor-speed entry (dims ("rotor", "time")); only the entry name differs
# ("motors_measured" for DREGON, "rps" for Michael's).
# ---------------------------------------------------------------------------


@dataclass
class _ChunkSource:
    frame: td.Frame
    origin: str  # "dregon" | "michaels"
    rps_key: str  # "motors_measured" | "rps"
    n_channels: int  # number of usable audio channels
    duration: float


def _wrap_frame(tf: td.Frame, *, origin: str, rps_key: str) -> _ChunkSource:
    audio = tf["audio"]
    n_ch = audio.dim_size("mic") if "mic" in audio.dims else 1
    return _ChunkSource(
        frame=tf, origin=origin, rps_key=rps_key, n_channels=n_ch, duration=audio.duration
    )


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

    def _extract_chunk(
        self, src: _ChunkSource
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (audio [T], rps_motor_rate [4, M], audio_ts [T], motor_ts [M])."""
        tf = src.frame
        # Random channel within record
        if self.channel_policy == "random" and src.n_channels > 1:
            ch = int(self.rng.integers(0, src.n_channels))
        else:
            ch = 0

        # Compute valid time window (intersection of audio and motor domains).
        audio_start = tf["audio"].t_start
        audio_end = tf["audio"].t_end
        motor_track = tf[src.rps_key]
        motor_start = motor_track.t_start
        motor_end = motor_track.t_end
        valid_start = max(audio_start, motor_start)
        valid_end = min(audio_end, motor_end)
        rel_lo = valid_start - audio_start
        rel_hi = valid_end - audio_start - self._chunk_duration_sec
        start = float(self.rng.uniform(rel_lo, rel_hi))

        # Slice both audio and motors simultaneously.
        sliced = tf.time[audio_start + start : audio_start + start + self._chunk_duration_sec]

        audio_s = sliced["audio"]
        # audio data is (channels, N) — axis 0 = channels, axis -1 = time
        audio = np.asarray(audio_s.data)[ch, :]
        audio_ts = cast(td.GridIndex, audio_s.tindex).sample_times()

        motor_s = sliced[src.rps_key]
        motor_ts = cast(td.StampIndex, motor_s.tindex).abs_stamps
        # values are already time-last (4, M).
        rps = (
            np.asarray(motor_s.data)
            if motor_s.data is not None
            else np.zeros((4, 0), dtype=np.float32)
        )

        # Length normalisation — chunks can be off by 1 sample due to int cast
        if len(audio) > self.chunk_size:
            audio = audio[: self.chunk_size]
            audio_ts = audio_ts[: self.chunk_size]
        elif len(audio) < self.chunk_size:
            # Pad audio + audio_ts (extrapolate by sample dt)
            pad = self.chunk_size - len(audio)
            dt = 1.0 / self.sample_rate
            audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
            audio_ts = np.concatenate([audio_ts, audio_ts[-1] + dt * np.arange(1, pad + 1)])
        return (
            audio.astype(np.float32),
            rps.astype(np.float32),
            audio_ts.astype(np.float64),
            motor_ts.astype(np.float64),
        )

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

    Uses the tdseries-native loader.
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
        sources.append(_wrap_frame(tf, origin="dregon", rps_key="motors_measured"))
    return sources


def load_michaels_noise_sources(
    michaels_dir: str | Path,
    sample_rate: int,
) -> list[_ChunkSource]:
    """Load all Michael's recordings that exist in `michaels_dir`."""
    frames = M.load_michaels_timeframes(data_root=michaels_dir, sr=sample_rate)
    return [_wrap_frame(tf, origin="michaels", rps_key="rps") for tf in frames]


#: ``dregon_dir`` / ``michaels_dir`` values starting with this prefix select a
#: published rich-frame dataset instead of a local folder:
#: ``frames:NAME[@VERSION]`` (e.g. ``frames:DREGON-frames``).
FRAMES_SPEC_PREFIX = "frames:"


def _parse_frames_spec(spec: str) -> tuple[str, str | None]:
    body = spec[len(FRAMES_SPEC_PREFIX) :]
    name, _, version = body.partition("@")
    if not name:
        raise ValueError(
            f"invalid published-frames spec {spec!r}: expected 'frames:NAME[@VERSION]'"
        )
    return name, (version or None)


def load_published_noise_sources(
    spec: str,
    sample_rate: int,
    *,
    origin: str,
    rps_key: str,
    splits: list[str] | None = None,
) -> list[_ChunkSource]:
    """Load a published rich-frame dataset (``frames:NAME[@VERSION]``).

    The dload/tdframe-v1 counterpart of the folder loaders above (see
    ``scripts/publish_frame_datasets.py``): streams the dataset via
    ``streams.iter_published_frames``, keeps only the ``audio`` + ``rps_key``
    tracks (+ ``meta``) of each recording — the published frames carry their
    fixes baked in, so nothing is re-cleaned here — and soxr-resamples audio
    to ``sample_rate``. Recordings missing either track are skipped, matching
    the folder loaders (e.g. DREGON recordings without ``motors_measured``).
    """
    name, version = _parse_frames_spec(spec)
    sources: list[_ChunkSource] = []
    for tf in iter_published_frames(name, version, splits=splits):
        if "audio" not in tf or rps_key not in tf:
            continue
        entries: dict[str, Any] = {
            "audio": resample_audio_series(cast(td.Series, tf["audio"]), sample_rate),
            rps_key: tf[rps_key],
        }
        if "meta" in tf:
            entries["meta"] = tf["meta"]
        sources.append(_wrap_frame(td.Frame(entries), origin=origin, rps_key=rps_key))
    return sources


def build_noise_rps_datasets(
    *,
    dregon_dir: str | Path | None,
    michaels_dir: str | Path | None,
    sample_rate: int = 16000,
    chunk_size: int = 16000,
    train_samples: int = 4096,
    val_samples: int = 512,
    val_pct: float = 0.1,
    val_at_start: bool = False,
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
        dregon_dir: path to DREGON dataset (or None to skip). May also be a
            published-frames spec ``frames:DREGON-frames[@VERSION]`` to stream
            the rich-frame dataset from dload instead of a local folder.
        michaels_dir: path to Michael's recordings (or None to skip). May also
            be ``frames:michaels-frames[@VERSION]``.
        sample_rate: target audio sample rate.
        chunk_size: chunk length in samples.
        train_samples / val_samples: virtual epoch sizes.
        val_pct: fraction of each recording held out for validation.
        val_at_start: hold out the *first* `val_pct` fraction of each
            recording for validation instead of the last (train:
            `[t_start+cut, t_end]`, val: `[t_start, t_start+cut]`). A
            "swapped-split" knob for noise-generation experiments (see
            REPLICATION.md § E2/E3) — not a room-level DREGON split (this
            loader pools all `in_flight_noise` recordings together, it does
            not select by recording id), just which end of each recording's
            time axis is held out.
        seed: RNG seed.
        cache_dir: where to cache resampled DREGON audio (kept for API compat).
        **dataset_kwargs: forwarded to `NoiseRPSDataset` (e.g. rps_normalize).
    """
    sources: list[_ChunkSource] = []
    if dregon_dir is not None:
        if isinstance(dregon_dir, str) and dregon_dir.startswith(FRAMES_SPEC_PREFIX):
            # e.g. "frames:DREGON-frames" — mirror the folder loader: the
            # in_flight_noise split, measured motor speeds only.
            sources += load_published_noise_sources(
                dregon_dir,
                sample_rate,
                origin="dregon",
                rps_key="motors_measured",
                splits=["in_flight_noise"],
            )
        else:
            sources += load_dregon_noise_sources(dregon_dir, sample_rate, cache_dir=cache_dir)
    if michaels_dir is not None:
        if isinstance(michaels_dir, str) and michaels_dir.startswith(FRAMES_SPEC_PREFIX):
            sources += load_published_noise_sources(
                michaels_dir, sample_rate, origin="michaels", rps_key="rps"
            )
        else:
            sources += load_michaels_noise_sources(michaels_dir, sample_rate)

    if not sources:
        raise ValueError("No noise sources found (DREGON and Michael's both empty/absent)")

    train_sources: list[_ChunkSource] = []
    val_sources: list[_ChunkSource] = []
    for src in sources:
        tf = src.frame
        cut = src.duration * (1.0 - val_pct) if not val_at_start else src.duration * val_pct
        # Open-ended slices: recordings with absolute epoch timestamps (the
        # published tdframe-v1 frames) sit at ~1e18 ticks, beyond float64
        # integer precision — round-tripping t_start/t_end through float
        # overshoots the boundary by a few ticks and raises DomainError.
        # Only the interior cut point may round-trip.
        if val_at_start:
            # Val: [t_start, t_start+cut]  |  Train: [t_start+cut, t_end]
            val_tf = tf.time[: tf.t_start + cut]
            train_tf = tf.time[tf.t_start + cut :]
        else:
            # Train: [t_start, t_start+cut]  |  Val: [t_start+cut, t_end]
            train_tf = tf.time[: tf.t_start + cut]
            val_tf = tf.time[tf.t_start + cut :]
        if train_tf["audio"].duration >= chunk_size / sample_rate:
            train_sources.append(_wrap_frame(train_tf, origin=src.origin, rps_key=src.rps_key))
        if val_tf["audio"].duration >= chunk_size / sample_rate:
            val_sources.append(_wrap_frame(val_tf, origin=src.origin, rps_key=src.rps_key))

    train_ds = NoiseRPSDataset(
        train_sources,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        samples_per_epoch=train_samples,
        seed=seed,
        **dataset_kwargs,
    )
    val_ds = NoiseRPSDataset(
        val_sources,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        samples_per_epoch=val_samples,
        seed=seed + 1,
        **dataset_kwargs,
    )
    return train_ds, val_ds
