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

from .frames import get_meta, resample_audio_series
from .streams import iter_published_frames

#: Repository root — relative ``dregon_rps_override_dir`` values resolve here,
#: so a config can name a repo-relative folder and stay machine-independent
#: (same convention as ``zoo.cache.REPO_ROOT``).
REPO_ROOT = Path(__file__).resolve().parents[2]

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
            channel from each record; 'all' keeps EVERY channel, returning
            ``audio`` as ``(C, T)`` instead of ``(T,)``. 'all' is what
            multi-observer noise generation needs: a channel model whose
            spatial law differs from the coherent field's ``1/r`` (the wind-wake
            gate, say) is simply unidentifiable from a single microphone.
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
        channel_policy: Literal["first", "random", "all"] = "first",
        rps_normalize: float | None = None,
        balance_rps: bool = False,
        rps_bins: int = 8,
    ):
        if not records:
            raise ValueError("NoiseRPSDataset got no records")
        self.records = records
        self.chunk_size = int(chunk_size)
        self.sample_rate = int(sample_rate)
        self.samples_per_epoch = int(samples_per_epoch)
        self.channel_policy = channel_policy
        self.rps_normalize = rps_normalize
        # When True, chunk-start positions are drawn to *flatten the RPS
        # histogram* per record, so the brief low-/zero-RPS regions (warm-up,
        # takeoff, landing, rotors-off) are represented far more than their tiny
        # share of wall-clock time — the generator must see them to learn that
        # zero RPS means silence. Off => uniform-in-time (proportional) sampling.
        self.balance_rps = bool(balance_rps)
        self.rps_bins = int(rps_bins)
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
        # Per-record (candidate_start_rel, weight) for RPS-balanced sampling.
        self._start_sampler: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if self.balance_rps:
            for i, src in enumerate(self.records):
                self._start_sampler[i] = self._build_start_sampler(src)

    def __len__(self):
        return self.samples_per_epoch

    def _valid_start_window(self, src: _ChunkSource) -> tuple[float, float]:
        """Relative [lo, hi] chunk-start range (audio-frame seconds) — the
        audio∩motor overlap minus one chunk length."""
        tf = src.frame
        audio_start = tf["audio"].t_start
        motor_track = tf[src.rps_key]
        valid_start = max(audio_start, motor_track.t_start)
        valid_end = min(tf["audio"].t_end, motor_track.t_end)
        return valid_start - audio_start, valid_end - audio_start - self._chunk_duration_sec

    def _build_start_sampler(self, src: _ChunkSource) -> tuple[np.ndarray, np.ndarray]:
        """Candidate chunk-start positions + inverse-RPS-frequency weights.

        Evaluates mean-window RPS on a coarse grid of starts, bins it, and weights
        each candidate by ``1 / (#candidates in its bin)`` so every RPS level —
        including the rare zero/warm-up regions — is drawn about equally often.
        """
        rel_lo, rel_hi = self._valid_start_window(src)
        if rel_hi <= rel_lo:
            return np.array([max(rel_lo, 0.0)]), np.array([1.0])
        tf = src.frame
        motor = tf[src.rps_key]
        m_ts = np.asarray(cast(td.StampIndex, motor.tindex).abs_stamps, dtype=np.float64)
        m_val = np.asarray(motor.data, dtype=np.float64)  # (4, M)
        if m_ts.size == 0:
            return np.array([rel_lo]), np.array([1.0])
        m_mean = m_val.mean(axis=0)  # (M,) mean over rotors
        a0 = tf["audio"].t_start
        # ~4 candidate starts per chunk length, capped for memory.
        n_cand = int(min(4000, max(8, (rel_hi - rel_lo) / self._chunk_duration_sec * 4)))
        starts = np.linspace(rel_lo, rel_hi, n_cand)
        # mean RPS over each candidate window [start, start+chunk].
        win = self._chunk_duration_sec
        abs_starts = a0 + starts
        lo_idx = np.searchsorted(m_ts, abs_starts)
        hi_idx = np.searchsorted(m_ts, abs_starts + win)
        cand_rps = np.array(
            [
                m_mean[lo:hi].mean() if hi > lo else m_mean[min(lo, len(m_mean) - 1)]
                for lo, hi in zip(lo_idx, hi_idx, strict=False)
            ]
        )
        # Inverse-frequency weights over RPS bins (flatten the histogram).
        edges = np.linspace(0.0, max(cand_rps.max(), 1e-6), self.rps_bins + 1)
        bin_idx = np.clip(np.digitize(cand_rps, edges) - 1, 0, self.rps_bins - 1)
        counts = np.bincount(bin_idx, minlength=self.rps_bins).astype(np.float64)
        w = 1.0 / np.maximum(counts[bin_idx], 1.0)
        return starts, w / w.sum()

    def _extract_chunk(
        self, src: _ChunkSource, rec_idx: int = -1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (audio, rps_motor_rate [4, M], audio_ts [T], motor_ts [M]).

        ``audio`` is ``[T]`` for the single-channel policies and ``[C, T]`` for
        ``channel_policy='all'``.
        """
        tf = src.frame
        # Random channel within record
        if self.channel_policy == "all":
            ch = slice(None)
        elif self.channel_policy == "random" and src.n_channels > 1:
            ch = int(self.rng.integers(0, src.n_channels))
        else:
            ch = 0

        # Compute valid time window (intersection of audio and motor domains).
        audio_start = tf["audio"].t_start
        if self.balance_rps and rec_idx in self._start_sampler:
            starts, sw = self._start_sampler[rec_idx]
            start = float(self.rng.choice(starts, p=sw))
        else:
            rel_lo, rel_hi = self._valid_start_window(src)
            start = float(self.rng.uniform(rel_lo, rel_hi))

        # Slice both audio and motors simultaneously.
        sliced = tf.time[audio_start + start : audio_start + start + self._chunk_duration_sec]

        audio_s = sliced["audio"]
        # audio data is (channels, N) — axis 0 = channels, axis -1 = time
        audio = np.asarray(audio_s.data)[ch, :]  # [T], or [C, T] for 'all'
        audio_ts = cast(td.GridIndex, audio_s.tindex).sample_times()

        motor_s = sliced[src.rps_key]
        motor_ts = cast(td.StampIndex, motor_s.tindex).abs_stamps
        # values are already time-last (4, M).
        rps = (
            np.asarray(motor_s.data)
            if motor_s.data is not None
            else np.zeros((4, 0), dtype=np.float32)
        )
        # An empty motor slice (no telemetry sample landed in this window — can
        # happen at a sparse-telemetry edge) is unusable; raise so __getitem__
        # retries cleanly instead of an IndexError downstream in upsampling.
        if motor_ts.size == 0 or rps.shape[-1] == 0:
            raise ValueError("empty motor slice for chunk window")

        # Length normalisation — chunks can be off by 1 sample due to int cast.
        # Time is the LAST axis, which is the only one present for the mono
        # policies and the second of two under `channel_policy='all'`.
        n_samples = audio.shape[-1]
        if n_samples > self.chunk_size:
            audio = audio[..., : self.chunk_size]
            audio_ts = audio_ts[: self.chunk_size]
        elif n_samples < self.chunk_size:
            # Pad audio + audio_ts (extrapolate by sample dt)
            pad = self.chunk_size - n_samples
            dt = 1.0 / self.sample_rate
            pad_shape = (*audio.shape[:-1], pad)
            audio = np.concatenate([audio, np.zeros(pad_shape, dtype=audio.dtype)], axis=-1)
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

        for _ in range(8):  # a few retries if a record/chunk fails for some reason
            try:
                audio, rps, audio_ts, motor_ts = self._extract_chunk(src, src_idx)
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

    ``dregon_dir`` may be a ``frames:DREGON-frames[@VER]`` spec, a plain
    path or a ``dload:...`` URI.
    """
    if isinstance(dregon_dir, str) and dregon_dir.startswith(FRAMES_SPEC_PREFIX):
        return load_published_noise_sources(
            dregon_dir,
            sample_rate,
            origin="dregon",
            rps_key="motors_measured",
            splits=["in_flight_noise"],
        )
    from data_processing.sources.dregon import load_dregon_timeframes
    from data_processing.streams import resolve_source

    dregon_dir = Path(resolve_source(dregon_dir))
    frames = load_dregon_timeframes(
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
    """Load all Michael's recordings.

    ``michaels_dir`` may be a ``frames:michaels-frames[@VER]`` spec, a plain
    path or a ``dload:...`` URI.
    """
    if isinstance(michaels_dir, str) and michaels_dir.startswith(FRAMES_SPEC_PREFIX):
        return load_published_noise_sources(
            michaels_dir, sample_rate, origin="michaels", rps_key="rps"
        )
    from data_processing.sources import michaels as M
    from data_processing.streams import resolve_source

    frames = M.load_michaels_timeframes(data_root=resolve_source(michaels_dir), sr=sample_rate)
    return [_wrap_frame(tf, origin="michaels", rps_key="rps") for tf in frames]


# ---------------------------------------------------------------------------
# Rotor-speed label knobs (DREGON only)
# ---------------------------------------------------------------------------


def resolve_override_dir(override_dir: str | Path) -> Path:
    """Make an override directory absolute against the repository root."""
    path = Path(override_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _available_ids(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.npz"))


def apply_rps_override(frame: td.Frame, rps_key: str, override_dir: str | Path) -> td.Frame:
    """Replace a recording's rotor-speed VALUES with refined labels.

    The refined labels come from a sidecar file ``<override_dir>/<recording
    id>.npz`` with two arrays:

    - ``ft`` (N,): the label times in seconds, RELATIVE to the audio
      ``t_start`` of this (full, untrimmed) frame
    - ``r_refined`` (R, N): the refined rotor speeds at those times

    The frame keeps its timebase, its dims and its dtype. Only the values
    change: each original telemetry stamp gets the linear interpolation of
    ``r_refined`` at its own offset from the audio ``t_start``. The offset is
    computed in exact int64 ticks, because the published frames sit at
    absolute epoch ticks (~1e18) that float64 cannot hold.

    Times outside ``ft`` keep their original telemetry values. The function
    raises if the sidecar is absent — a silent fall-back to the original
    telemetry would make the two arms of the label A/B the same experiment.
    """
    directory = resolve_override_dir(override_dir)
    recording_id = str(get_meta(frame, "recording_id", "") or "")
    sidecar = directory / f"{recording_id}.npz"
    if not recording_id or not sidecar.is_file():
        raise FileNotFoundError(
            f"no refined RPS sidecar for recording {recording_id!r}: expected {sidecar}. "
            f"Available ids in {directory}: {_available_ids(directory)}. "
            "The sidecars come from the trajectory-refinement job; run it first, or "
            "unset dregon_rps_override_dir to train on the original telemetry."
        )
    with np.load(sidecar) as data:
        if "ft" not in data or "r_refined" not in data:
            raise ValueError(f"{sidecar} has no 'ft'/'r_refined' arrays: {sorted(data.files)}")
        ft = np.asarray(data["ft"], dtype=np.float64).reshape(-1)
        refined = np.asarray(data["r_refined"], dtype=np.float64)
    if refined.ndim != 2 or refined.shape[-1] != ft.size or ft.size < 2:
        raise ValueError(
            f"{sidecar}: expected r_refined (R, N) against ft (N,) with N >= 2, "
            f"got {refined.shape} against {ft.shape}"
        )
    if np.any(np.diff(ft) <= 0.0):
        raise ValueError(f"{sidecar}: 'ft' must increase strictly")

    series = cast(td.Series, frame[rps_key])
    values = np.asarray(series.data)
    if values.ndim != 2 or values.shape[0] != refined.shape[0]:
        raise ValueError(
            f"{sidecar}: refined labels have {refined.shape[0]} rotors, but "
            f"{rps_key} has values of shape {values.shape}"
        )
    tindex = series.tindex
    if not isinstance(tindex, td.StampIndex):
        raise TypeError(f"{rps_key} must be an event (StampIndex) track, got {type(tindex)}")

    # Tick-exact offsets: int64 minus int64, THEN one division into seconds.
    audio_start_ticks = cast(td.Series, frame["audio"]).t_start_ticks
    stamp_ticks = np.asarray(tindex.abs_stamps_ticks, dtype=np.int64)
    offsets = (stamp_ticks - int(audio_start_ticks)) / float(td.TICKS_PER_SECOND)

    # Stamps OUTSIDE the sidecar's span keep the original telemetry. Clipping
    # them to the edge assigned cruise-level refined values to the motor
    # shutdown after the audio ends (telemetry 0 vs refined 75 rev/s).
    inside = (offsets >= ft[0]) & (offsets <= ft[-1])
    new_values = np.array(values, copy=True)
    # One sidecar grid step, in stamps — the smear radius of a sub-frame
    # on/off transition when the coarse grid interpolates onto the stamps.
    if offsets.size > 1:
        stamp_dt = float(np.median(np.diff(offsets)))
        grid_dt = float(np.median(np.diff(ft)))
        radius = max(1, int(np.ceil(grid_dt / max(stamp_dt, 1e-9))))
    else:
        radius = 1
    for rotor in range(refined.shape[0]):
        # A stopped motor stays stopped, and stamps within one grid step of a
        # stop keep their telemetry: the 0.032 s sidecar grid smears the
        # sub-frame shutdown step into a ramp (75 rev/s appeared on stamps
        # whose telemetry is 0, and 5 rev/s on running stamps adjacent to the
        # stop). Refined values also clamp to >= 0.
        stopped = values[rotor] <= 0.0
        near_stop = (
            np.convolve(stopped.astype(np.float64), np.ones(2 * radius + 1), mode="same") > 0.0
        )
        running = inside & ~near_stop
        interp = np.interp(offsets[running], ft, refined[rotor])
        new_values[rotor, running] = np.maximum(interp, 0.0).astype(values.dtype)
    return frame.with_entry(rps_key, series.with_data(new_values))


def apply_rps_scale(frame: td.Frame, rps_key: str, scale: float) -> td.Frame:
    """Multiply a recording's rotor-speed values by ``scale``.

    The cheap counterpart of :func:`apply_rps_override`: one constant gain on
    the labels, which is the phase-7 correction of the measured DREGON
    telemetry bias. Timebase, dims and dtype stay the same.
    """
    series = cast(td.Series, frame[rps_key])
    values = np.asarray(series.data)
    return frame.with_entry(rps_key, series.with_data((values * float(scale)).astype(values.dtype)))


def _check_label_knobs(override_dir: str | Path | None, scale: float) -> None:
    """Make sure that only one label knob is active."""
    if override_dir is not None and float(scale) != 1.0:
        raise ValueError(
            "dregon_rps_override_dir and dregon_rps_scale are mutually exclusive: "
            f"got {override_dir!r} and {scale!r}. The refined labels already carry "
            "their own scale."
        )


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
    rps_override_dir: str | Path | None = None,
    rps_scale: float = 1.0,
) -> list[_ChunkSource]:
    """Load a published rich-frame dataset (``frames:NAME[@VERSION]``).

    The dload/tdframe-v1 counterpart of the folder loaders above (see
    the ``source_frames`` derivation): streams the dataset via
    ``streams.iter_published_frames``, keeps only the ``audio`` + ``rps_key``
    tracks (+ ``meta``) of each recording — the published frames carry their
    fixes baked in, so nothing is re-cleaned here — and soxr-resamples audio
    to ``sample_rate``. Recordings missing either track are skipped, matching
    the folder loaders (e.g. DREGON recordings without ``motors_measured``).

    ``rps_override_dir`` / ``rps_scale`` are the two label knobs (see
    :func:`apply_rps_override` / :func:`apply_rps_scale`). They apply to the
    FULL frame, before the overlap trim below, because the sidecar times are
    relative to the full frame's audio ``t_start``. Only the DREGON call of
    :func:`build_noise_rps_datasets` passes them; Michael's labels never
    change.
    """
    _check_label_knobs(rps_override_dir, rps_scale)
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
        frame = td.Frame(entries)
        if rps_override_dir is not None:
            frame = apply_rps_override(frame, rps_key, rps_override_dir)
        elif float(rps_scale) != 1.0:
            frame = apply_rps_scale(frame, rps_key, rps_scale)
        # The published audio can span more than the telemetry (e.g. michaels
        # rps starts after the audio) — chunks sampled outside the overlap
        # would carry an empty motor slice (upsample_rps_to_audio_rate
        # IndexError). Trim to the common window, with a small inward margin
        # so absolute epoch boundaries (~1e18 ticks) never round-trip through
        # float exactly (cf. the open-ended-slice fix in
        # build_noise_rps_datasets).
        # NB: for a StampIndex telemetry series t_start/t_end are the DOMAIN
        # bounds (= the container's), not the first/last stamp — DREGON
        # motors_measured stamps start seconds after the audio. Use the
        # actual stamps.
        stamps = np.asarray(cast(Any, frame[rps_key]).tindex.abs_stamps, dtype=np.float64)
        if stamps.size < 2:
            continue
        margin = 0.01
        lo = max(float(frame["audio"].t_start), float(stamps[0])) + margin
        hi = min(float(frame["audio"].t_end), float(stamps[-1])) - margin
        if hi - lo < 1.0:
            continue
        sources.append(_wrap_frame(frame.time[lo:hi], origin=origin, rps_key=rps_key))
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
    dregon_rps_override_dir: str | Path | None = None,
    dregon_rps_scale: float = 1.0,
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
        dregon_rps_override_dir: folder of refined-label sidecars, one
            ``<recording id>.npz`` per DREGON recording. When set, every
            DREGON recording gets its ``motors_measured`` values replaced by
            the refined trajectory (:func:`apply_rps_override`); a recording
            with no sidecar raises. A relative path resolves against the
            repository root. Michael's labels never change.
        dregon_rps_scale: constant gain on the DREGON labels
            (:func:`apply_rps_scale`), for example ``0.99458`` for the
            measured telemetry bias. Mutually exclusive with
            ``dregon_rps_override_dir``. Michael's labels never change.
        **dataset_kwargs: forwarded to `NoiseRPSDataset` (e.g. rps_normalize).
    """
    _check_label_knobs(dregon_rps_override_dir, dregon_rps_scale)
    label_knobs_set = dregon_rps_override_dir is not None or float(dregon_rps_scale) != 1.0
    if label_knobs_set and dregon_dir is None:
        raise ValueError(
            "dregon_rps_override_dir / dregon_rps_scale apply to DREGON only, "
            "but dregon_dir is None"
        )
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
                rps_override_dir=dregon_rps_override_dir,
                rps_scale=dregon_rps_scale,
            )
        else:
            if label_knobs_set:
                # The sidecar times are relative to the PUBLISHED frame's audio
                # t_start, so they are only correct on the `frames:` path.
                raise ValueError(
                    "dregon_rps_override_dir / dregon_rps_scale need a published-frames "
                    f"spec (dregon_dir='frames:DREGON-frames'), got {dregon_dir!r}"
                )
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
