"""
DREGON-LibriMix Dataset Synthesis Script (v2)

Creates synthetic mixtures from:
- Speech samples from LibriSpeech (train-clean-100)
- Drone noise from DREGON dataset with real command RPS data

Key improvements over v1:
- Uses all 8 microphone channels (each as independent noise source)
- 3-second samples (exactly 48k samples at 16 kHz)
- Command speeds (not measured) cleaned via clean_command_spikes
- Recording-level train/valid split (no overlap)
- Synthetic steady-state motor combinations (~20% of train)
- Optional low-level white noise mixed with speech
"""

import argparse
import json
import random
from glob import glob
from pathlib import Path
from typing import cast

import librosa
import numpy as np
import soundfile as sf
import tdseries as td
from tqdm import tqdm

from data_processing.dregon import (
    clean_command_spikes,
    get_geometry,
    load_dregon_timeframes,
)
from data_processing.frames import get_meta, with_meta
from data_processing.michaels import load_michaels_timeframes
from utils.paths import get_data_path, get_datasets_path

# =============================================================================
# Constants
# =============================================================================

SAMPLE_RATE = 16000
SAMPLE_DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_DURATION * SAMPLE_RATE)  # 48000
MOTOR_SAMPLE_RATE = 929.0  # Hz — default DREGON motor logging rate
NUM_ROTORS = 4

# Train: all in_flight_noise recordings
TRAIN_NOISE_SPLITS = ["in_flight_noise"]
# Valid: specific in_flight_source recordings
VALID_NOISE_RECORDING_IDS = [
    "free-flight_whitenoise-low_room1",
    "free-flight_speech-low_room1",
]


# =============================================================================
# Audio I/O helpers
# =============================================================================


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE, mono: bool = True) -> np.ndarray:
    """Load audio file, resample to target sample rate, convert to mono."""
    audio, sr = sf.read(path)

    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range based on peak."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def adjust_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or randomly crop audio to exact target length."""
    current_length = len(audio)

    if current_length > target_length:
        start = np.random.randint(0, current_length - target_length + 1)
        audio = audio[start : start + target_length]
    elif current_length < target_length:
        pad_length = target_length - current_length
        audio = np.pad(audio, (0, pad_length), mode="constant", constant_values=0)

    return audio


def calculate_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    speech_power = np.sum(speech**2)
    noise_power = np.sum(noise**2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(speech_power / max(noise_power, 1e-10))


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix speech and noise at a target SNR.

    Returns:
        Tuple of (mixture, scaled_speech, scaled_noise)
    """
    speech_power = np.sum(speech**2)
    noise_power = np.sum(noise**2)

    if noise_power > 0 and speech_power > 0:
        target_noise_power = speech_power / (10 ** (target_snr / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        noise = noise * scale

    mixture = speech + noise

    # Anti-clipping: if mixture peaks above 0.95, scale everything down
    max_val = np.abs(mixture).max()
    if max_val > 0.95:
        scale_factor = 0.95 / max_val
        mixture = mixture * scale_factor
        speech = speech * scale_factor
        noise = noise * scale_factor

    return mixture.astype(np.float32), speech.astype(np.float32), noise.astype(np.float32)


def generate_white_noise(length: int, snr_db: float, speech: np.ndarray) -> np.ndarray:
    """
    Generate white noise and mix with speech at specified SNR (noise below speech).

    Args:
        length: Number of samples
        snr_db: SNR of white noise relative to speech (positive = noise quieter)
        speech: Reference speech signal for power calculation

    Returns:
        speech + white_noise mixed at target SNR
    """
    speech_power = np.sum(speech**2)
    noise = np.random.randn(length).astype(np.float32)
    noise_power = np.sum(noise**2)

    if noise_power > 0 and speech_power > 0:
        target_noise_power = speech_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        noise = noise * scale

    return (speech + noise).astype(np.float32)


# =============================================================================
# DREGON record loading
# =============================================================================


def load_dregon_noise_records(
    dregon_dir: Path,
    recording_ids: list[str] | None = None,
    splits: list[str] | None = None,
    cache_dir: Path | None = None,
) -> list[td.Frame]:
    """
    Load DREGON recordings as ``td.Frame`` objects, one per channel per recording.

    Args:
        dregon_dir: Path to DREGON data directory
        recording_ids: Specific recording IDs to load (overrides splits)
        splits: Split names to load (used if recording_ids is None)
        cache_dir: Ignored (the loader handles resampling via target_sr).

    Returns:
        List of ``td.Frame`` objects (one per channel per recording).
        Each frame keeps a single-channel ``"audio"`` entry with dims
        ``("mic", "time")`` (mic size 1 — never squeezed away).
    """
    dregon_dir = Path(dregon_dir)
    get_geometry(dregon_dir)

    # Load all frames resampled to SAMPLE_RATE
    all_frames = load_dregon_timeframes(
        dregon_dir.parent,
        splits=splits or ["in_flight_noise"],
        target_sr=SAMPLE_RATE,
        download=False,
    )

    # Filter by recording_ids if given
    if recording_ids is not None:
        rid_set = set(recording_ids)
        all_frames = [tf for tf in all_frames if get_meta(tf, "recording_id", "") in rid_set]

    # Expand each channel into a separate single-channel td.Frame.
    # ``tf.slice["mic", ch:ch+1]`` (a *slice*, not a bare int) keeps the "mic"
    # dim at size 1 and slices "audio" and "mic_pos" together consistently
    # (they share the "mic" dim) — see docs/refactor-unified-framework.md.
    result: list[td.Frame] = []
    for tf in tqdm(all_frames, desc="Loading DREGON records"):
        rid = get_meta(tf, "recording_id", "?")
        if "motors_measured" not in tf and "motors_command" not in tf:
            print(f"Warning: {rid} has no motor data, skipping")
            continue
        n_channels = tf["audio"].dim_size("mic")
        for ch in range(n_channels):
            ch_tf = tf.slice["mic", ch : ch + 1]
            ch_tf = with_meta(ch_tf, recording_id=f"{rid}_ch{ch}")
            result.append(ch_tf)

    return result


# =============================================================================
# Noise chunk extraction with command RPS
# =============================================================================


def resolve_motor_tracks(tf: td.Frame) -> tuple[str, str, bool]:
    """Resolve the rotor-speed entry names of a noise ``td.Frame``.

    Returns ``(detect_key, rps_key, needs_cleaning)``:

    - ``detect_key``: entry used for in-flight window detection (real measured
      speeds when available — they capture spindown during landing).
    - ``rps_key``: entry saved as ground-truth RPS.
    - ``needs_cleaning``: whether ``clean_command_spikes`` must be applied
      (only DREGON ``motors_command`` carries logging spikes / freezes).

    Two conventions are supported so that any aligned ``td.Frame`` can serve as
    a noise source:

    - **DREGON**: separate ``motors_measured`` (real, preferred for detection)
      and ``motors_command`` (cleaner, preferred as GT). Command values carry
      leading/trailing freezes → ``needs_cleaning=True``.
    - **Generic / Michael's**: a single ``rps`` entry of already-aligned
      measured rotor speeds (rev/s). Detection and GT both use it, and no spike
      cleaning is applied.
    """
    if "motors_command" in tf or "motors_measured" in tf:
        detect = "motors_measured" if "motors_measured" in tf else "motors_command"
        rps_k = "motors_command" if "motors_command" in tf else "motors_measured"
        return detect, rps_k, True
    if "rps" in tf:
        return "rps", "rps", False
    raise ValueError(
        f"{get_meta(tf, 'recording_id', '?')} has no rotor-speed track "
        f"(expected one of 'motors_measured', 'motors_command', 'rps')"
    )


def _find_inflight_window(
    tf: td.Frame,
    motor_key: str,
    min_motor_rps: float,
    clean: bool = True,
) -> tuple[float, float]:
    """Return (t_start, t_end) of the in-flight window (absolute seconds).

    Finds the first and last absolute times where **all 4 rotors** exceed
    ``min_motor_rps``.  For DREGON command telemetry, ``clean_command_spikes``
    is first applied to the motor entry (``clean=True``) to strip the
    pre-takeoff logging artefact; for already-clean measured tracks (e.g.
    Michael's ``rps``) pass ``clean=False``.  This trims the ramp-up from the
    start; landing is usually not captured in DREGON telemetry so the end is
    rarely affected.

    Raises ``ValueError`` if no in-flight window can be found.
    """
    motor = tf[motor_key]
    if motor.data is None or motor.dim_size("time") == 0:
        return motor.t_start, motor.t_end
    vals = np.asarray(motor.data).copy()  # (4, M)
    if clean:
        vals = clean_command_spikes(vals)
    ts = cast(td.StampIndex, motor.tindex).abs_stamps  # (M,)
    in_flight = np.all(vals > min_motor_rps, axis=0)  # (M,) bool
    idxs = np.where(in_flight)[0]
    if len(idxs) == 0:
        raise ValueError(
            f"No in-flight window (all motors > {min_motor_rps} RPS) found "
            f"in {get_meta(tf, 'recording_id', '?')}"
        )
    return float(ts[idxs[0]]), float(ts[idxs[-1]])


def extract_noise_chunk_with_command_rps(
    tf: td.Frame,
    duration_sec: float = SAMPLE_DURATION,
    channel: int = 0,
    min_motor_rps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract a random noise chunk with aligned command RPS (cleaned) from a td.Frame.

    Uses command speeds (not measured), cleaned via clean_command_spikes.

    Args:
        tf: td.Frame with "audio" and "motors_command" entries.
        duration_sec: Duration of chunk in seconds.
        channel: Microphone channel index (0-based) within the audio entry.
        min_motor_rps: When > 0, restrict sampling to the in-flight window
            (all 4 rotors above this threshold after cleaning).  Set to e.g.
            30.0 to exclude takeoff/pre-takeoff; 0.0 disables (default).

    Returns:
        Tuple of (audio, rps, metadata)
        - audio: (n_samples,) mono audio at record's sample rate
        - rps: (4, n_motor_samples) cleaned command RPS at native motor rate
        - metadata: dict with recording_id, start_time, motor_sample_rate
    """
    # Resolve rotor-speed tracks (DREGON command/measured, or generic ``rps``).
    detect_key, rps_key, needs_clean = resolve_motor_tracks(tf)

    audio_s = tf["audio"]
    detect_s = tf[detect_key]

    valid_start = max(audio_s.t_start, detect_s.t_start)
    valid_end = min(audio_s.t_end, detect_s.t_end)

    if min_motor_rps > 0.0:
        t_fl_start, t_fl_end = _find_inflight_window(
            tf, detect_key, min_motor_rps, clean=needs_clean
        )
        valid_start = max(valid_start, t_fl_start)
        valid_end = min(valid_end, t_fl_end)

    valid_duration = valid_end - valid_start

    if valid_duration < duration_sec:
        rec_id = get_meta(tf, "recording_id", "?")
        raise ValueError(
            f"Record {rec_id} has insufficient overlap: {valid_duration:.1f}s < {duration_sec}s"
        )

    # Random start time within valid range (absolute time)
    start_sec = np.random.uniform(valid_start, valid_end - duration_sec)

    # Slice using Frame.time (works on absolute time)
    sliced = tf.time[start_sec : start_sec + duration_sec]

    # Extract mono audio from specified channel
    # audio data is (channels, N) — axis 0 = channels
    audio_samples = np.asarray(sliced["audio"].data)
    audio = audio_samples[channel, :] if audio_samples.ndim > 1 else audio_samples

    motor_sliced = sliced[rps_key]
    if motor_sliced.data is not None:
        vals = np.asarray(motor_sliced.data).copy()  # (4, M) — time-last
        rps = (clean_command_spikes(vals) if needs_clean else vals).astype(np.float32)
    else:
        rps = np.zeros((4, 0), dtype=np.float32)

    # Estimate motor sample rate
    motor_ts = cast(td.StampIndex, motor_sliced.tindex).abs_stamps
    if len(motor_ts) > 1:
        motor_sr = 1.0 / np.median(np.diff(motor_ts.astype(np.float64)))
    else:
        motor_sr = MOTOR_SAMPLE_RATE

    metadata = {
        "recording_id": get_meta(tf, "recording_id", ""),
        "start_time": start_sec - audio_s.t_start,
        "duration": duration_sec,
        "motor_sample_rate": float(motor_sr),
        "channel": channel,
    }

    return audio.astype(np.float32), rps, metadata


# =============================================================================
# Synthetic motor combinations
# =============================================================================


# Cached per-channel SPL targets (computed once, then reused)
_SPL_TARGETS_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _load_all_motor_wavs(motors_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """
    Load all individual motor WAV files (keep all 8 channels).

    Returns:
        Dict mapping (motor_id, speed) -> audio array (n_samples, 8), float32, 44100 Hz
    """
    motor_wavs = {}
    for wav_path in motors_dir.rglob("Motor*_*.wav"):
        stem = wav_path.stem  # e.g., "Motor1_70"
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        motor_id = int(parts[0].replace("Motor", ""))
        speed = int(parts[1])
        audio, _ = sf.read(wav_path)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]  # Ensure 2D
        motor_wavs[(motor_id, speed)] = audio.astype(np.float32)
    return motor_wavs


def _compute_per_channel_spl_targets(
    motor_wavs: dict[tuple[int, int], np.ndarray],
    motors_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel SPL targets for motor combo normalization.

    Returns:
        Tuple of (mean_single_rms_per_channel, allmotors_70_rms_per_channel)
        Each is shape (8,) float64.
    """
    global _SPL_TARGETS_CACHE
    if _SPL_TARGETS_CACHE is not None:
        return _SPL_TARGETS_CACHE

    n_channels = 8

    # --- allMotors_70 RMS per channel ---
    allm_paths = list(motors_dir.rglob("allMotors_70.wav"))
    if allm_paths:
        allm, _ = sf.read(allm_paths[0])
        if allm.ndim == 1:
            allm = allm[:, np.newaxis]
        allm_rms = np.sqrt(np.mean(allm.astype(np.float64) ** 2, axis=0))  # (8,)
    else:
        allm_rms = np.full(n_channels, 0.0348, dtype=np.float64)

    # --- Mean RMS per channel across all single-motor recordings ---
    single_rms_per_ch = np.zeros((len(motor_wavs), n_channels), dtype=np.float64)
    for i, (_key, audio) in enumerate(sorted(motor_wavs.items())):
        single_rms_per_ch[i] = np.sqrt(np.mean(audio.astype(np.float64) ** 2, axis=0))

    mean_single_rms = single_rms_per_ch.mean(axis=0)  # (8,)

    _SPL_TARGETS_CACHE = (mean_single_rms, allm_rms)
    return _SPL_TARGETS_CACHE


def create_motor_combo_sample(
    motor_wavs: dict[tuple[int, int], np.ndarray],
    motors_dir: Path,
    duration_sec: float = SAMPLE_DURATION,
    target_sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a synthetic steady-state rotor combination.

    Picks one microphone channel, then sums the SAME channel across different
    motors (with random speeds). Normalizes SPL to match the expected level
    for that number of rotors.

    Args:
        motor_wavs: Dict of (motor_id, speed) -> audio array (n_samples, 8)
        motors_dir: Path to motor recordings directory (for allMotors_70 reference)
        duration_sec: Target duration
        target_sr: Target sample rate

    Returns:
        Tuple of (audio, rps, metadata)
        - audio: (n_samples,) mono audio at target_sr (resampled if needed)
        - rps: (4, n_motor_samples) constant RPS, zeros for inactive rotors
        - metadata: dict with combo info
    """
    mean_single_rms_per_ch, allm_rms_per_ch = _compute_per_channel_spl_targets(
        motor_wavs, motors_dir
    )

    motor_sr = 44100  # Native sample rate of individual motor WAVs
    motor_target_length = int(duration_sec * motor_sr)  # e.g., 132300 for 3s
    target_length = int(duration_sec * target_sr)  # e.g., 48000 for 3s@16kHz
    n_motor_samples = int(duration_sec * MOTOR_SAMPLE_RATE)
    n_channels = mean_single_rms_per_ch.shape[0]

    # Pick a random microphone channel
    channel = np.random.randint(0, n_channels)

    # Randomly choose number of motors (1–4)
    num_motors = np.random.randint(1, NUM_ROTORS + 1)

    # Randomly choose which motor IDs to use (1-4), and random speeds (50-90)
    available_motors = list(range(1, NUM_ROTORS + 1))
    chosen_motors = random.sample(available_motors, num_motors)
    available_speeds = [50, 60, 70, 80, 90]

    combo_specs = []  # list of (motor_id, speed)
    for motor_id in chosen_motors:
        speed = random.choice(available_speeds)
        combo_specs.append((motor_id, speed))

    # Sum the SAME channel across different motors at native sample rate
    summed_audio = np.zeros(motor_target_length, dtype=np.float64)
    for motor_id, speed in combo_specs:
        key = (motor_id, speed)
        if key not in motor_wavs:
            raise KeyError(f"Motor {motor_id} at {speed} rps not found")
        wav = motor_wavs[key]  # (n_total, 8) at 44100 Hz
        ch_audio = wav[:, channel]  # (n_total,) at 44100 Hz

        # Randomly crop a segment at native rate
        if len(ch_audio) > motor_target_length:
            start = np.random.randint(0, len(ch_audio) - motor_target_length + 1)
            segment = ch_audio[start : start + motor_target_length]
        else:
            segment = np.pad(ch_audio, (0, motor_target_length - len(ch_audio)), mode="constant")
        summed_audio += segment

    # Normalize SPL per channel (at native rate)
    target_rms = (
        mean_single_rms_per_ch[channel]
        + (allm_rms_per_ch[channel] - mean_single_rms_per_ch[channel]) * (num_motors - 1) / 3.0
    )
    actual_rms = float(np.sqrt(np.mean(summed_audio**2)))
    if actual_rms > 1e-10:
        summed_audio = summed_audio * (target_rms / actual_rms)

    audio = summed_audio.astype(np.float32)

    # Resample from motor_sr to target_sr
    if target_sr != motor_sr:
        audio = librosa.resample(audio, orig_sr=motor_sr, target_sr=target_sr)
        # Trim/pad to exact target length after resampling
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")

    # Build RPS array: constant speeds for active motors, 0 for inactive
    rps = np.zeros((NUM_ROTORS, n_motor_samples), dtype=np.float32)
    for motor_id, speed in combo_specs:
        rps[motor_id - 1, :] = float(speed)

    metadata = {
        "num_motors": num_motors,
        "combo": [(int(m), int(s)) for m, s in combo_specs],
        "channel": int(channel),
        "motor_sample_rate": MOTOR_SAMPLE_RATE,
        "target_rms": float(target_rms),
        "actual_rms_before_norm": float(actual_rms),
    }

    return audio, rps, metadata


# =============================================================================
# Multichannel (8-channel) record loading and chunk extraction
# =============================================================================


def load_dregon_multichannel_records(
    dregon_dir: Path,
    recording_ids: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[td.Frame]:
    """Load DREGON recordings as full multi-channel ``td.Frame`` objects.

    Unlike ``load_dregon_noise_records`` (which expands each mic channel into a
    separate single-channel frame), this keeps the full ``(C, N)`` audio entry so
    a single chunk yields all microphone channels at the same time slice.

    Returns:
        List of ``td.Frame`` objects (one per recording), each with a
        multi-channel ``"audio"`` entry and shared ``"motors_command"`` /
        ``"motors_measured"``.
    """
    dregon_dir = Path(dregon_dir)

    # When filtering by explicit recording_ids, search ALL splits (the valid
    # IDs live in ``in_flight_source``, not ``in_flight_noise``).
    load_splits = None if recording_ids is not None else (splits or ["in_flight_noise"])
    all_frames = load_dregon_timeframes(
        dregon_dir.parent,
        splits=load_splits,
        target_sr=SAMPLE_RATE,
        download=False,
    )

    if recording_ids is not None:
        rid_set = set(recording_ids)
        all_frames = [tf for tf in all_frames if get_meta(tf, "recording_id", "") in rid_set]

    result: list[td.Frame] = []
    for tf in all_frames:
        if "motors_measured" not in tf and "motors_command" not in tf:
            print(f"Warning: {get_meta(tf, 'recording_id', '?')} has no motor data, skipping")
            continue
        result.append(tf)
    return result


def load_michaels_sources(
    ids: list[str],
    data_root: Path | None = None,
    sr: int = SAMPLE_RATE,
) -> list[td.Frame]:
    """Load Michael's recordings as noise-source ``td.Frame``s, filtered by id.

    Each id may be given as ``"125"``, ``"FLY125"`` (case-insensitive), or
    ``"all"``.  The returned frames carry an ``rps`` entry (already-aligned
    measured rotor speeds, rev/s) and an 8-channel ``audio`` entry, ready to be
    used as a noise source.  Their ``recording_id`` meta is normalised to
    ``"michaels_FLY<id>"`` so downstream metadata can distinguish them.
    """
    frames = load_michaels_timeframes(data_root=data_root, sr=sr)
    want_all = any(i.strip().lower() == "all" for i in ids)
    wanted = {i.strip().lower().removeprefix("fly") for i in ids}

    out: list[td.Frame] = []
    for tf in frames:
        rid = str(get_meta(tf, "recording_id", ""))  # wav stem, e.g. "124"
        if want_all or rid.lower().removeprefix("fly") in wanted:
            out.append(with_meta(tf, recording_id=f"michaels_FLY{rid}"))
    return out


def load_noise_sources(
    specs: list[str],
    dregon_dir: Path,
    sr: int = SAMPLE_RATE,
    michaels_root: Path | None = None,
) -> list[td.Frame]:
    """Compose a noise-source pool (``list[td.Frame]``) from a list of specs.

    This is the single entry point for mixing and matching any sets of aligned
    source ``td.Frame``s into a train or valid noise pool.  Each spec selects
    one group of recordings:

    - ``"dregon-split:<split>"`` — all DREGON recordings in a split
      (e.g. ``"dregon-split:in_flight_noise"``); ``"dregon:"`` is a short alias.
    - ``"dregon-id:<recording_id>"`` — a specific DREGON recording, searched
      across all splits (e.g. ``"dregon-id:free-flight_speech-low_room1"``);
      ``"dregon-rec:"`` is an alias.
    - ``"michaels:<id>"`` — a Michael's recording by id, ``"125"`` /
      ``"FLY125"`` / ``"all"`` (e.g. ``"michaels:FLY125"``).
    - a bare token with no ``"kind:"`` prefix is treated as a DREGON
      recording id (backward-compatible with ``--valid_recording_ids``).

    The pool is the concatenation of all selected frames, in spec order
    (DREGON splits, then DREGON ids, then Michael's).
    """
    dregon_splits: list[str] = []
    dregon_ids: list[str] = []
    michaels_ids: list[str] = []

    for raw in specs:
        spec = raw.strip()
        if not spec:
            continue
        if ":" in spec:
            kind, val = spec.split(":", 1)
            kind, val = kind.strip().lower(), val.strip()
        else:
            kind, val = "dregon-id", spec  # bare token → DREGON recording id

        if kind in ("dregon", "dregon-split"):
            dregon_splits.append(val)
        elif kind in ("dregon-id", "dregon-rec"):
            dregon_ids.append(val)
        elif kind in ("michaels", "michael", "fly"):
            michaels_ids.append(val)
        else:
            raise ValueError(
                f"Unknown noise source spec '{spec}'. Expected one of: "
                f"'dregon:<split>', 'dregon-id:<recording_id>', 'michaels:<id>'."
            )

    frames: list[td.Frame] = []
    if dregon_splits:
        frames += load_dregon_multichannel_records(dregon_dir, splits=dregon_splits)
    if dregon_ids:
        frames += load_dregon_multichannel_records(dregon_dir, recording_ids=dregon_ids)
    if michaels_ids:
        frames += load_michaels_sources(michaels_ids, data_root=michaels_root, sr=sr)
    return frames


def extract_multichannel_noise_chunk(
    tf: td.Frame,
    duration_sec: float = SAMPLE_DURATION,
    min_motor_rps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract a random ``(C, n_samples)`` noise chunk with aligned command RPS.

    Same logic as ``extract_noise_chunk_with_command_rps`` but keeps ALL mic
    channels instead of selecting one.

    ``min_motor_rps``: when > 0, restricts sampling to the in-flight window
    (all 4 rotors above this threshold after cleaning).  Set to e.g. 30.0 to
    exclude pre-takeoff and ramp-up; 0.0 disables (backward-compat default).

    Returns:
        (audio, rps, metadata)
        - audio: ``(C, n_samples)`` multi-channel audio at record's sample rate
        - rps: ``(4, n_motor_samples)`` cleaned command RPS (shared across channels)
        - metadata: dict with recording_id, start_time, motor_sample_rate, n_channels
    """
    # Resolve rotor-speed tracks (DREGON command/measured, or generic ``rps``).
    detect_key, rps_key, needs_clean = resolve_motor_tracks(tf)

    audio_s = tf["audio"]
    detect_s = tf[detect_key]

    valid_start = max(audio_s.t_start, detect_s.t_start)
    valid_end = min(audio_s.t_end, detect_s.t_end)

    if min_motor_rps > 0.0:
        t_fl_start, t_fl_end = _find_inflight_window(
            tf, detect_key, min_motor_rps, clean=needs_clean
        )
        valid_start = max(valid_start, t_fl_start)
        valid_end = min(valid_end, t_fl_end)

    if valid_end - valid_start < duration_sec:
        rec_id = get_meta(tf, "recording_id", "?")
        raise ValueError(
            f"Record {rec_id} has insufficient overlap: "
            f"{valid_end - valid_start:.1f}s < {duration_sec}s"
        )

    start_sec = np.random.uniform(valid_start, valid_end - duration_sec)
    sliced = tf.time[start_sec : start_sec + duration_sec]

    # Keep all channels: audio data is (channels, N)
    audio_samples = np.asarray(sliced["audio"].data)
    if audio_samples.ndim == 1:
        audio_samples = audio_samples[np.newaxis, :]
    audio = audio_samples.astype(np.float32)  # (C, N)

    motor_sliced = sliced[rps_key]
    if motor_sliced.data is not None:
        # motor data is time-last (4, M); clean_command_spikes keeps shape
        vals = np.asarray(motor_sliced.data).copy()
        rps = (clean_command_spikes(vals) if needs_clean else vals).astype(np.float32)
    else:
        rps = np.zeros((4, 0), dtype=np.float32)

    motor_ts = cast(td.StampIndex, motor_sliced.tindex).abs_stamps
    if len(motor_ts) > 1:
        motor_sr = 1.0 / np.median(np.diff(motor_ts.astype(np.float64)))
    else:
        motor_sr = MOTOR_SAMPLE_RATE

    metadata = {
        "recording_id": get_meta(tf, "recording_id", ""),
        "start_time": start_sec - audio_s.t_start,
        "duration": duration_sec,
        "motor_sample_rate": float(motor_sr),
        "n_channels": int(audio.shape[0]),
    }
    return audio, rps, metadata


def extract_non_overlapping_multichannel_chunks(
    tf: td.Frame,
    duration_sec: float = SAMPLE_DURATION,
    min_motor_rps: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """Extract ALL non-overlapping ``(C, n_samples)`` chunks from a recording.

    Same logic as ``extract_multichannel_noise_chunk`` but instead of picking
    a single random chunk, returns every non-overlapping chunk spanning the
    in-flight window in order.  Any remainder at the end that is shorter than
    ``duration_sec`` is dropped (not zero-padded).

    Returns:
        List of (audio, rps, metadata) tuples, one per non-overlapping chunk.
    """
    detect_key, rps_key, needs_clean = resolve_motor_tracks(tf)

    audio_s = tf["audio"]
    detect_s = tf[detect_key]

    valid_start = max(audio_s.t_start, detect_s.t_start)
    valid_end = min(audio_s.t_end, detect_s.t_end)

    if min_motor_rps > 0.0:
        t_fl_start, t_fl_end = _find_inflight_window(
            tf, detect_key, min_motor_rps, clean=needs_clean
        )
        valid_start = max(valid_start, t_fl_start)
        valid_end = min(valid_end, t_fl_end)

    if valid_end - valid_start < duration_sec:
        rec_id = get_meta(tf, "recording_id", "?")
        raise ValueError(
            f"Record {rec_id} has insufficient overlap: "
            f"{valid_end - valid_start:.1f}s < {duration_sec}s"
        )

    chunks = []
    n_chunks = int((valid_end - valid_start) // duration_sec)
    for i in range(n_chunks):
        start_sec = valid_start + i * duration_sec
        sliced = tf.time[start_sec : start_sec + duration_sec]

        audio_samples = np.asarray(sliced["audio"].data)
        if audio_samples.ndim == 1:
            audio_samples = audio_samples[np.newaxis, :]
        audio = audio_samples.astype(np.float32)  # (C, N)

        motor_sliced = sliced[rps_key]
        if motor_sliced.data is not None:
            vals = np.asarray(motor_sliced.data).copy()
            rps = (clean_command_spikes(vals) if needs_clean else vals).astype(np.float32)
        else:
            rps = np.zeros((4, 0), dtype=np.float32)

        motor_ts = cast(td.StampIndex, motor_sliced.tindex).abs_stamps
        if len(motor_ts) > 1:
            motor_sr = 1.0 / np.median(np.diff(motor_ts.astype(np.float64)))
        else:
            motor_sr = MOTOR_SAMPLE_RATE

        metadata = {
            "recording_id": get_meta(tf, "recording_id", ""),
            "start_time": start_sec - audio_s.t_start,
            "duration": duration_sec,
            "motor_sample_rate": float(motor_sr),
            "n_channels": int(audio.shape[0]),
        }
        chunks.append((audio, rps, metadata))

    return chunks


def adjust_length_mc(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or randomly crop a ``(C, T)`` array along the time axis."""
    current = audio.shape[-1]
    if current > target_length:
        start = np.random.randint(0, current - target_length + 1)
        return audio[:, start : start + target_length]
    if current < target_length:
        return np.pad(audio, ((0, 0), (0, target_length - current)), mode="constant")
    return audio


def render_multichannel_sample(
    noise_records: list[td.Frame],
    speech_files: list[str],
    *,
    target_length: int,
    sample_rate: int,
    sample_duration: float,
    snr_range: tuple[float, float],
    speech_per_channel: str,
    source_white_noise_prob: float,
    white_noise_prob: float,
    white_noise_snr: float,
    min_motor_rps: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Render ONE multichannel DREGON-LibriMix sample — no disk I/O.

    Returns ``(arrays, meta)`` where ``arrays`` holds ``mixture``/``vocals``/
    ``noise`` each ``(T, C)`` interleaved float32 plus ``rps`` ``(4, M)``, and
    ``meta`` is the per-sample metadata dict (without the ``id`` key, which the
    caller adds). Shared by the disk-writing CLI
    (:func:`create_dregon_librimix_multichannel`) and the dload derived-dataset
    generator (``data_processing.derivations.generate_dregon_lm_split``).

    Advances ``random``/``np.random`` in exactly the order the original inline
    loop did (record pick → chunk extraction retries → per-channel source draw
    → per-channel SNR), so routing the CLI through it does not change its
    output for a given seed + ``speech_files`` order. Keep that order stable.
    """
    record = random.choice(noise_records)
    noise = None
    for _ in range(20):
        try:
            noise, rps, noise_meta = extract_multichannel_noise_chunk(
                record,
                duration_sec=sample_duration,
                min_motor_rps=min_motor_rps,
            )
            break
        except ValueError:
            record = random.choice(noise_records)
    else:
        raise ValueError("Could not find a valid noise chunk after 20 attempts")

    noise = adjust_length_mc(noise, target_length)  # (C, T)
    C = noise.shape[0]
    # Per-channel peak normalization
    noise = noise / np.maximum(np.abs(noise).max(axis=1, keepdims=True), 1e-10)

    def _draw_source(force_wn: bool = False) -> np.ndarray:
        """Return a normalised 1-D source signal."""
        if force_wn or (source_white_noise_prob > 0 and random.random() < source_white_noise_prob):
            src = np.random.randn(target_length).astype(np.float32)
        else:
            src = load_audio(random.choice(speech_files), target_sr=sample_rate, mono=True)
            src = adjust_length(src, target_length)
        return normalize_audio(src)

    if speech_per_channel == "shared":
        # One source decision for the whole sample.
        is_wn = source_white_noise_prob > 0 and random.random() < source_white_noise_prob
        shared_src = _draw_source(force_wn=is_wn)
        speech_channels = [shared_src.copy() for _ in range(C)]
    else:  # independent
        speech_channels = [_draw_source() for _ in range(C)]

    mix_ch, voc_ch, noi_ch = [], [], []
    per_channel_snr = []
    for ch in range(C):
        speech = speech_channels[ch]
        if white_noise_prob > 0 and random.random() < white_noise_prob:
            speech = normalize_audio(generate_white_noise(target_length, white_noise_snr, speech))
        target_snr = float(np.random.uniform(snr_range[0], snr_range[1]))
        mixture, speech_scaled, noise_scaled = mix_at_snr(speech, noise[ch], target_snr)
        mix_ch.append(mixture)
        voc_ch.append(speech_scaled)
        noi_ch.append(noise_scaled)
        per_channel_snr.append(calculate_snr(speech_scaled, noise_scaled))

    arrays = {
        "mixture": np.stack(mix_ch, axis=1).astype(np.float32),
        "vocals": np.stack(voc_ch, axis=1).astype(np.float32),
        "noise": np.stack(noi_ch, axis=1).astype(np.float32),
        "rps": rps,
    }
    meta = {
        "n_channels": int(C),
        "input_snr_per_channel": [float(s) for s in per_channel_snr],
        "input_snr": float(np.mean(per_channel_snr)),
        "speech_per_channel": speech_per_channel,
        "source_white_noise_prob": source_white_noise_prob,
        "noise_source": noise_meta["recording_id"],
        "noise_start_time": noise_meta.get("start_time", 0.0),
        "motor_sample_rate": noise_meta.get("motor_sample_rate", MOTOR_SAMPLE_RATE),
        "rps_shape": list(rps.shape),
    }
    return arrays, meta


def create_dregon_librimix_multichannel(
    speech_dir: Path,
    dregon_dir: Path,
    output_dir: Path,
    num_samples: int,
    sample_duration: float = SAMPLE_DURATION,
    sample_rate: int = SAMPLE_RATE,
    snr_range: tuple[float, float] = (-30.0, 0.0),
    split: str = "train",
    seed: int = 42,
    speech_per_channel: str = "independent",
    source_white_noise_prob: float = 0.0,
    white_noise_prob: float = 0.0,
    white_noise_snr: float = 30.0,
    min_motor_rps: float = 0.0,
    noise_records: list[td.Frame] | None = None,
):
    """Create an 8-channel DREGON-LibriMix dataset.

    Each sample is a full multi-channel noise chunk (all mics at the same time
    slice). Speech is mixed into every channel; with ``speech_per_channel=
    "independent"`` each channel gets a different utterance and SNR, so the
    channel axis behaves like a minibatch at train time. ``rps.npy`` is shared
    across channels (``(4, M)``).

    ``source_white_noise_prob``: fraction of samples (or channels, when
    ``speech_per_channel='independent'``) that use white noise as the target
    source *instead of* a LibriSpeech utterance.  The white noise is drawn
    fresh per sample/channel and normalised to the same level as speech would
    be; SNR mixing then proceeds identically.

    ``white_noise_prob``: legacy — adds a small amount of WN *on top of* the
    speech source (not a replacement).  Usually 0 for the real-recording
    validation approach.

    The noise pool is ``noise_records`` when provided (compose it with
    ``load_noise_sources``); otherwise it falls back to the DREGON defaults
    (``TRAIN_NOISE_SPLITS`` for train, ``VALID_NOISE_RECORDING_IDS`` for valid).

    Saved per sample: ``vocals.wav``, ``noise.wav``, ``mixture.wav`` each
    ``(n_samples, C)`` interleaved WAV, plus ``rps.npy``.
    """
    random.seed(seed)
    np.random.seed(seed)

    target_length = int(sample_duration * sample_rate)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    speech_files = []
    for ext in ["*.wav", "*.flac"]:
        speech_files.extend(glob(str(speech_dir / "**" / ext), recursive=True))
    if len(speech_files) == 0:
        raise ValueError(f"No speech files found in {speech_dir}")
    print(f"Found {len(speech_files)} speech files")

    if noise_records is None:
        if split == "train":
            noise_records = load_dregon_multichannel_records(
                dregon_dir,
                splits=TRAIN_NOISE_SPLITS,
            )
        else:
            noise_records = load_dregon_multichannel_records(
                dregon_dir,
                recording_ids=VALID_NOISE_RECORDING_IDS,
            )
    if len(noise_records) == 0:
        raise ValueError("No valid noise records with motor data found")
    print(f"Loaded {len(noise_records)} multichannel noise records")
    print(f"  Sources: {sorted({str(get_meta(tf, 'recording_id', '?')) for tf in noise_records})}")

    metadata_list = []
    for idx in tqdm(range(num_samples), desc=f"Creating {split} samples"):
        sample_id = f"sample_{idx:05d}"
        sample_dir = split_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        arrays, sample_meta = render_multichannel_sample(
            noise_records,
            speech_files,
            target_length=target_length,
            sample_rate=sample_rate,
            sample_duration=sample_duration,
            snr_range=snr_range,
            speech_per_channel=speech_per_channel,
            source_white_noise_prob=source_white_noise_prob,
            white_noise_prob=white_noise_prob,
            white_noise_snr=white_noise_snr,
            min_motor_rps=min_motor_rps,
        )

        sf.write(sample_dir / "vocals.wav", arrays["vocals"], sample_rate)
        sf.write(sample_dir / "noise.wav", arrays["noise"], sample_rate)
        sf.write(sample_dir / "mixture.wav", arrays["mixture"], sample_rate)
        np.save(sample_dir / "rps.npy", arrays["rps"])

        metadata_list.append({"id": sample_id, **sample_meta})

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}
    all_metadata[split] = metadata_list
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Created {num_samples} {split} samples in {split_dir}")
    print(f"Metadata saved to {metadata_path}")


# =============================================================================
# Main dataset creation
# =============================================================================


def create_dregon_librimix(
    speech_dir: Path,
    dregon_dir: Path,
    output_dir: Path,
    num_samples: int,
    sample_duration: float = SAMPLE_DURATION,
    sample_rate: int = SAMPLE_RATE,
    snr_range: tuple[float, float] = (-30.0, 0.0),
    split: str = "train",
    seed: int = 42,
    motor_combo_fraction: float = 0.0,
    white_noise_prob: float = 0.0,
    white_noise_snr: float = 30.0,
    min_motor_rps: float = 0.0,
):
    """
    Create the DREGON-LibriMix dataset with command RPS data.

    Args:
        speech_dir: Directory containing LibriSpeech audio files
        dregon_dir: Path to DREGON dataset directory
        output_dir: Output directory for the dataset
        num_samples: Number of samples to generate
        sample_duration: Duration of each sample in seconds (default 3.0)
        sample_rate: Target sample rate
        snr_range: Range of target SNRs in dB
        split: 'train' or 'valid'
        seed: Random seed
        motor_combo_fraction: Fraction of samples from synthetic motor combos (0.0–1.0)
        white_noise_prob: Probability of adding low-level white noise to speech
        white_noise_snr: SNR of white noise relative to speech (dB, positive = quieter)
    """
    random.seed(seed)
    np.random.seed(seed)

    target_length = int(sample_duration * sample_rate)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # --- Find speech files ---
    speech_files = []
    for ext in ["*.wav", "*.flac"]:
        speech_files.extend(glob(str(speech_dir / "**" / ext), recursive=True))

    if len(speech_files) == 0:
        raise ValueError(f"No speech files found in {speech_dir}")

    print(f"Found {len(speech_files)} speech files")

    # --- Load noise records ---
    cache_dir = dregon_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        noise_records = load_dregon_noise_records(
            dregon_dir,
            splits=TRAIN_NOISE_SPLITS,
            cache_dir=cache_dir,
        )
    else:
        noise_records = load_dregon_noise_records(
            dregon_dir,
            recording_ids=VALID_NOISE_RECORDING_IDS,
            cache_dir=cache_dir,
        )

    if len(noise_records) == 0:
        raise ValueError("No valid DREGON records with motor data found")

    total_noise_duration = sum(tf["audio"].duration for tf in noise_records)
    print(f"Loaded {len(noise_records)} noise records ({total_noise_duration:.1f}s total)")

    # Identify unique base recordings (strip _chN suffix) for reporting
    base_recordings = set(
        str(get_meta(tf, "recording_id", "")).rsplit("_ch", 1)[0] for tf in noise_records
    )
    print(f"  Unique base recordings: {sorted(base_recordings)}")

    # --- Load motor WAVs for synthetic combos (train only) ---
    motor_wavs = None
    motors_dir = None
    num_motor_combo_samples = 0

    if motor_combo_fraction > 0 and split == "train":
        motors_dir = dregon_dir / "DREGON_individual_motors_recordings"
        motor_wavs = _load_all_motor_wavs(motors_dir)

        if len(motor_wavs) == 0:
            print("Warning: No motor WAVs found, disabling motor combos")
            motor_combo_fraction = 0.0
        else:
            num_motor_combo_samples = int(num_samples * motor_combo_fraction)
            # Pre-compute SPL targets
            ms_rms, am_rms = _compute_per_channel_spl_targets(motor_wavs, motors_dir)
            print(
                f"Motor combos: {num_motor_combo_samples} samples ({motor_combo_fraction:.0%}), "
                f"{len(motor_wavs)} motor WAVs loaded"
            )
            # Report SPL targets averaged across channels
            print(
                f"  SPL targets (mean across channels): "
                f"1-motor RMS={ms_rms.mean():.6f}, "
                f"4-motor RMS={am_rms.mean():.6f}"
            )

    # --- Generate samples ---
    metadata_list = []

    for idx in tqdm(range(num_samples), desc=f"Creating {split} samples"):
        sample_id = f"sample_{idx:05d}"
        sample_dir = split_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        # Decide sample type: motor combo or in-flight noise
        is_motor_combo = idx < num_motor_combo_samples

        if is_motor_combo:
            # --- Synthetic motor combo ---
            assert motor_wavs is not None and motors_dir is not None
            noise, rps, noise_meta = create_motor_combo_sample(
                motor_wavs,
                motors_dir=motors_dir,
                duration_sec=sample_duration,
                target_sr=sample_rate,
            )
            noise = adjust_length(noise, target_length)
            noise = normalize_audio(noise)
            noise_source_id = f"motor_combo_{noise_meta['combo']}"
        else:
            # --- In-flight noise ---
            record = random.choice(noise_records)

            # Records are already single-channel; always use channel 0
            ch = 0

            # Try to extract a valid chunk
            noise = None
            for _attempt in range(20):
                try:
                    noise, rps, noise_meta = extract_noise_chunk_with_command_rps(
                        record,
                        duration_sec=sample_duration,
                        channel=ch,
                        min_motor_rps=min_motor_rps,
                    )
                    break
                except ValueError:
                    record = random.choice(noise_records)
            else:
                raise ValueError("Could not find a valid noise chunk after 20 attempts")

            noise = adjust_length(noise, target_length)
            noise = normalize_audio(noise)
            noise_source_id = noise_meta["recording_id"]

        # --- Speech ---
        speech_path = random.choice(speech_files)
        speech = load_audio(speech_path, target_sr=sample_rate, mono=True)
        speech = adjust_length(speech, target_length)
        speech = normalize_audio(speech)

        # --- Optional white noise mixed into speech ---
        if white_noise_prob > 0 and random.random() < white_noise_prob:
            speech = generate_white_noise(target_length, white_noise_snr, speech)
            speech = normalize_audio(speech)

        # --- Random target SNR and mix ---
        target_snr = np.random.uniform(snr_range[0], snr_range[1])
        mixture, speech_scaled, noise_scaled = mix_at_snr(speech, noise, target_snr)
        actual_snr = calculate_snr(speech_scaled, noise_scaled)

        # --- Save files ---
        sf.write(sample_dir / "vocals.wav", speech_scaled, sample_rate)
        sf.write(sample_dir / "noise.wav", noise_scaled, sample_rate)
        sf.write(sample_dir / "mixture.wav", mixture, sample_rate)
        np.save(sample_dir / "rps.npy", rps)

        # --- Record metadata ---
        metadata_list.append(
            {
                "id": sample_id,
                "input_snr": float(actual_snr),
                "target_snr": float(target_snr),
                "speech_source": str(Path(speech_path).name),
                "noise_source": noise_source_id,
                "noise_start_time": noise_meta.get("start_time", 0.0),
                "motor_sample_rate": noise_meta.get("motor_sample_rate", MOTOR_SAMPLE_RATE),
                "rps_shape": list(rps.shape),
                "is_motor_combo": is_motor_combo,
            }
        )

    # --- Save metadata ---
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}

    all_metadata[split] = metadata_list

    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Created {num_samples} {split} samples in {split_dir}")
    print(f"Metadata saved to {metadata_path}")


# =============================================================================
# Real-recording passthrough valid set
# =============================================================================

# Recording IDs for the real valid set (in_flight_source, low-level source).
REAL_VALID_RECORDING_IDS = [
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
]


def _infer_source_type(recording_id: str) -> str:
    """Infer the co-recorded source type from a recording id.

    Michael's recordings are pure drone noise (no co-recorded source), so they
    map to ``"nosource"`` like DREGON ``*_nosource_*`` recordings.
    """
    rid = recording_id.lower()
    if "speech" in rid:
        return "speech"
    if "whitenoise" in rid:
        return "whitenoise"
    if "nosource" in rid or "michaels" in rid:
        return "nosource"
    return "unknown"


def create_dregon_real_valid(
    dregon_dir: Path,
    output_dir: Path,
    num_samples: int,
    sample_duration: float = 8.0,
    sample_rate: int = SAMPLE_RATE,
    recording_ids: list[str] | None = None,
    seed: int = 43,
    min_motor_rps: float = 0.0,
    max_non_overlapping: bool = False,
    records: list[td.Frame] | None = None,
) -> None:
    """Extract real 8-channel clips from DREGON recordings.

    Unlike the synthesised training set, these samples are **not mixed** —
    ``mixture.wav`` IS the raw multichannel recording (drone + co-recorded
    speech or whitenoise source).  No ``vocals.wav`` / ``noise.wav`` is
    written because no clean reference exists.

    Metadata includes ``source_type`` (inferred from recording ID), ``recording_id``,
    and ``start_time`` so downstream code can group samples by condition.

    Two modes:
    - **random** (default): ``num_samples`` chunks drawn at random positions
      from the pool of recordings (may overlap).
    - **max_non_overlapping**: extracts every non-overlapping chunk in every
      recording, covering the full in-flight window.  ``num_samples`` is
      ignored; all available chunks are written.
    """
    random.seed(seed)
    np.random.seed(seed)

    split_dir = output_dir / "valid"
    split_dir.mkdir(parents=True, exist_ok=True)

    # ``records`` (a pre-composed pool) takes precedence; otherwise fall back to
    # loading DREGON recordings by id.
    if records is None:
        if recording_ids is None:
            recording_ids = REAL_VALID_RECORDING_IDS
        records = load_dregon_multichannel_records(dregon_dir, recording_ids=recording_ids)
    if not records:
        raise ValueError(f"No DREGON records found for IDs: {recording_ids}")

    target_length = int(sample_duration * sample_rate)
    metadata_list: list[dict] = []

    if max_non_overlapping:
        # --- Extract every non-overlapping chunk from every recording ---
        all_chunks: list[tuple[np.ndarray, np.ndarray, dict, str]] = []  # (audio, rps, meta, rid)
        for tf in records:
            rid = str(get_meta(tf, "recording_id", "?"))
            try:
                chunks = extract_non_overlapping_multichannel_chunks(
                    tf,
                    duration_sec=sample_duration,
                    min_motor_rps=min_motor_rps,
                )
            except ValueError as e:
                print(f"  Skipping {rid}: {e}")
                continue
            for audio, rps, chunk_meta in chunks:
                all_chunks.append((audio, rps, chunk_meta, rid))

        total_possible = sum(
            int(tf["audio"].duration // sample_duration)
            for tf in records
            if tf["audio"].duration >= sample_duration
        )
        print(
            f"Loaded {len(records)} real recordings "
            f"({sum(tf['audio'].duration for tf in records):.0f}s total, "
            f"~{total_possible} non-overlapping clips available)"
        )
        print(f"Extracted {len(all_chunks)} non-overlapping chunks")

        for idx, (audio, rps, chunk_meta, rid) in enumerate(
            tqdm(all_chunks, desc="Writing valid samples (non-overlapping)")
        ):
            sample_id = f"sample_{idx:05d}"
            sample_dir = split_dir / sample_id
            sample_dir.mkdir(exist_ok=True)

            audio = adjust_length_mc(audio, target_length)  # (C, T)

            sf.write(sample_dir / "mixture.wav", audio.T.astype(np.float32), sample_rate)
            np.save(sample_dir / "rps.npy", rps)

            source_type = _infer_source_type(rid)

            metadata_list.append(
                {
                    "id": sample_id,
                    "recording_id": rid,
                    "source_type": source_type,
                    "start_time": chunk_meta.get("start_time", 0.0),
                    "duration": sample_duration,
                    "n_channels": int(audio.shape[0]),
                    "motor_sample_rate": chunk_meta.get("motor_sample_rate", MOTOR_SAMPLE_RATE),
                    "rps_shape": list(rps.shape),
                    "is_real_recording": True,
                }
            )
    else:
        # --- Original random-sampling behaviour ---
        total_dur = sum(tf["audio"].duration for tf in records)
        print(
            f"Loaded {len(records)} real recordings "
            f"({total_dur:.0f}s total, {total_dur // sample_duration:.0f} non-overlapping clips available)"
        )

        for idx in tqdm(range(num_samples), desc="Creating valid samples (real)"):
            sample_id = f"sample_{idx:05d}"
            sample_dir = split_dir / sample_id
            sample_dir.mkdir(exist_ok=True)

            record = random.choice(records)
            for _ in range(20):
                try:
                    audio, rps, chunk_meta = extract_multichannel_noise_chunk(
                        record,
                        duration_sec=sample_duration,
                        min_motor_rps=min_motor_rps,
                    )
                    break
                except ValueError:
                    record = random.choice(records)
            else:
                raise ValueError("Could not find a valid chunk after 20 attempts")

            audio = adjust_length_mc(audio, target_length)  # (C, T)

            # Save raw recording as-is — no mixing.
            sf.write(sample_dir / "mixture.wav", audio.T.astype(np.float32), sample_rate)
            np.save(sample_dir / "rps.npy", rps)

            rid = chunk_meta["recording_id"]
            source_type = _infer_source_type(rid)

            metadata_list.append(
                {
                    "id": sample_id,
                    "recording_id": rid,
                    "source_type": source_type,
                    "start_time": chunk_meta.get("start_time", 0.0),
                    "duration": sample_duration,
                    "n_channels": int(audio.shape[0]),
                    "motor_sample_rate": chunk_meta.get("motor_sample_rate", MOTOR_SAMPLE_RATE),
                    "rps_shape": list(rps.shape),
                    "is_real_recording": True,
                }
            )

    mpath = output_dir / "metadata.json"
    existing: dict = {}
    if mpath.exists():
        with open(mpath) as f:
            existing = json.load(f)
    existing["valid"] = metadata_list
    with open(mpath, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Created {len(metadata_list)} real-recording valid samples in {split_dir}")
    print(f"Metadata saved to {mpath}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Create DREGON-LibriMix dataset v2 with command RPS data"
    )
    parser.add_argument(
        "--speech_dir",
        type=Path,
        default=get_data_path("librispeech/LibriSpeech/train-clean-100"),
        help="Path to LibriSpeech directory",
    )
    parser.add_argument(
        "--dregon_dir",
        type=Path,
        default=get_data_path("DREGON"),
        help="Path to DREGON dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=get_datasets_path("DREGON-LM"),
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=6000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num_valid",
        type=int,
        default=600,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=SAMPLE_DURATION,
        help="Duration of each sample in seconds",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=SAMPLE_RATE,
        help="Target sample rate",
    )
    parser.add_argument(
        "--snr_min",
        type=float,
        default=-30.0,
        help="Minimum SNR in dB",
    )
    parser.add_argument(
        "--snr_max",
        type=float,
        default=0.0,
        help="Maximum SNR in dB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--motor_combo_fraction",
        type=float,
        default=0.2,
        help="Fraction of train samples from synthetic motor combos (0.0-1.0)",
    )
    parser.add_argument(
        "--multichannel",
        action="store_true",
        help="Produce full 8-channel samples (channel axis = minibatch) instead "
        "of per-channel-expanded mono samples",
    )
    parser.add_argument(
        "--speech_per_channel",
        choices=["independent", "shared"],
        default="independent",
        help="(multichannel only) different speech+SNR per channel, or the same",
    )
    parser.add_argument(
        "--min_motor_rps",
        type=float,
        default=0.0,
        help="Only sample from the in-flight window: the time range where all 4 "
        "rotors exceed this RPS threshold (after cleaning). Uses "
        "motors_measured when available (catches actual spindown), otherwise "
        "motors_command. Recommended: 30.0 to exclude takeoff/landing. "
        "0.0 disables (default, backward-compat).",
    )
    parser.add_argument(
        "--source_white_noise_prob",
        type=float,
        default=0.0,
        help="(multichannel) probability of using white noise AS the target source "
        "INSTEAD OF a LibriSpeech utterance (0.0 = always speech)",
    )
    parser.add_argument(
        "--white_noise_prob",
        type=float,
        default=0.0,
        help="Probability of adding white noise ON TOP OF speech source",
    )
    parser.add_argument(
        "--white_noise_snr",
        type=float,
        default=30.0,
        help="SNR of additive white noise relative to speech (dB, positive=quieter)",
    )
    parser.add_argument(
        "--real_valid",
        action="store_true",
        help="Create the valid split from real DREGON in_flight_source recordings "
        "(no mixing — mixture = raw recording). Uses --valid_duration and "
        "--valid_recording_ids. Incompatible with the mono pipeline.",
    )
    parser.add_argument(
        "--valid_duration",
        type=float,
        default=8.0,
        help="(--real_valid) clip duration in seconds (default 8.0)",
    )
    parser.add_argument(
        "--valid_recording_ids",
        type=str,
        default="",
        help="(--real_valid) comma-separated recording IDs to use; "
        f"defaults to: {','.join(REAL_VALID_RECORDING_IDS)}",
    )
    parser.add_argument(
        "--train_noise_sources",
        type=str,
        default="",
        help="(--multichannel) comma-separated noise-source specs composing the "
        "TRAIN noise pool. Each spec is one of 'dregon-split:<split>', "
        "'dregon-id:<recording_id>', or 'michaels:<id>' (id = 125 / FLY125 / all). "
        "Empty = default DREGON in_flight_noise. Example: "
        "'dregon-split:in_flight_noise,michaels:FLY125'.",
    )
    parser.add_argument(
        "--valid_noise_sources",
        type=str,
        default="",
        help="(--multichannel) comma-separated noise-source specs composing the "
        "VALID noise pool (same spec grammar as --train_noise_sources). When set, "
        "overrides --valid_recording_ids. Example: "
        "'dregon-id:free-flight_speech-low_room1,michaels:FLY124'.",
    )
    parser.add_argument(
        "--max_non_overlapping",
        action="store_true",
        help="(--real_valid) extract EVERY non-overlapping chunk from every "
        "recording (num_valid is ignored). Guarantees zero overlap between "
        "samples.",
    )

    args = parser.parse_args()

    if args.multichannel:
        # Compose the train/valid noise pools from source specs (when given).
        # Each pool is a plain list[td.Frame], so any aligned sources (DREGON
        # splits/recordings, Michael's, …) can be mixed and matched.
        michaels_root = args.dregon_dir.parent
        train_pool = None
        if args.train_noise_sources.strip():
            train_pool = load_noise_sources(
                [s for s in args.train_noise_sources.split(",")],
                dregon_dir=args.dregon_dir,
                sr=args.sample_rate,
                michaels_root=michaels_root,
            )
        valid_pool = None
        if args.valid_noise_sources.strip():
            valid_pool = load_noise_sources(
                [s for s in args.valid_noise_sources.split(",")],
                dregon_dir=args.dregon_dir,
                sr=args.sample_rate,
                michaels_root=michaels_root,
            )

        # --- Training split (synthesised mixtures) ---
        if args.num_train > 0:
            print("=" * 60)
            print("Creating TRAIN set (multichannel synthesised)...")
            print("=" * 60)
            create_dregon_librimix_multichannel(
                speech_dir=args.speech_dir,
                dregon_dir=args.dregon_dir,
                output_dir=args.output_dir,
                num_samples=args.num_train,
                sample_duration=args.duration,
                sample_rate=args.sample_rate,
                snr_range=(args.snr_min, args.snr_max),
                split="train",
                seed=args.seed,
                speech_per_channel=args.speech_per_channel,
                source_white_noise_prob=args.source_white_noise_prob,
                white_noise_prob=args.white_noise_prob,
                white_noise_snr=args.white_noise_snr,
                min_motor_rps=args.min_motor_rps,
                noise_records=train_pool,
            )

        # --- Validation split ---
        if args.num_valid > 0:
            if args.real_valid:
                # Passthrough: raw clips, no mixing. ``valid_pool`` (when set)
                # overrides --valid_recording_ids.
                rids = (
                    [r.strip() for r in args.valid_recording_ids.split(",") if r.strip()]
                    or None  # None → defaults to REAL_VALID_RECORDING_IDS
                )
                print("=" * 60)
                print("Creating VALID set (real recordings, no mixing)...")
                print("=" * 60)
                create_dregon_real_valid(
                    dregon_dir=args.dregon_dir,
                    output_dir=args.output_dir,
                    num_samples=args.num_valid,
                    sample_duration=args.valid_duration,
                    sample_rate=args.sample_rate,
                    recording_ids=rids,
                    seed=args.seed + 1,
                    min_motor_rps=args.min_motor_rps,
                    max_non_overlapping=args.max_non_overlapping,
                    records=valid_pool,
                )
            else:
                # Synthesised valid (same pipeline as train, different seed/records).
                print("=" * 60)
                print("Creating VALID set (multichannel synthesised)...")
                print("=" * 60)
                create_dregon_librimix_multichannel(
                    speech_dir=args.speech_dir,
                    dregon_dir=args.dregon_dir,
                    output_dir=args.output_dir,
                    num_samples=args.num_valid,
                    sample_duration=args.duration,
                    sample_rate=args.sample_rate,
                    snr_range=(args.snr_min, args.snr_max),
                    split="valid",
                    seed=args.seed + 1,
                    speech_per_channel=args.speech_per_channel,
                    source_white_noise_prob=args.source_white_noise_prob,
                    white_noise_prob=args.white_noise_prob,
                    white_noise_snr=args.white_noise_snr,
                    min_motor_rps=args.min_motor_rps,
                    noise_records=valid_pool,
                )

        print("\n" + "=" * 60)
        print("Dataset creation complete!")
        print("=" * 60)
        return

    # --- Create training set ---
    print("=" * 60)
    print("Creating TRAINING set...")
    print("=" * 60)
    create_dregon_librimix(
        speech_dir=args.speech_dir,
        dregon_dir=args.dregon_dir,
        output_dir=args.output_dir,
        num_samples=args.num_train,
        sample_duration=args.duration,
        sample_rate=args.sample_rate,
        snr_range=(args.snr_min, args.snr_max),
        split="train",
        seed=args.seed,
        motor_combo_fraction=args.motor_combo_fraction,
        white_noise_prob=args.white_noise_prob,
        white_noise_snr=args.white_noise_snr,
        min_motor_rps=args.min_motor_rps,
    )

    # --- Create validation set ---
    print()
    print("=" * 60)
    print("Creating VALIDATION set...")
    print("=" * 60)
    create_dregon_librimix(
        speech_dir=args.speech_dir,
        dregon_dir=args.dregon_dir,
        output_dir=args.output_dir,
        num_samples=args.num_valid,
        sample_duration=args.duration,
        sample_rate=args.sample_rate,
        snr_range=(args.snr_min, args.snr_max),
        split="valid",
        seed=args.seed + 1,
        motor_combo_fraction=0.0,
        white_noise_prob=args.white_noise_prob,
        white_noise_snr=args.white_noise_snr,
        min_motor_rps=args.min_motor_rps,
    )

    print()
    print("=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
