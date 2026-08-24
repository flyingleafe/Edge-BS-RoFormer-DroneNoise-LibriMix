"""Pure mixture-render cores — the one home of the project's mixing math.

Everything here is torch-free and disk-free: these functions render ONE sample
(or one clip) from already-loaded sources, and are shared by

- the dload derived-dataset generators (:mod:`data_processing.derivations`),
- the online-mixing stream (:mod:`data_processing.online_mixing`),
- the SE-valid builders (``generate_se_valid`` in ``derivations``).

Historically these lived in ``scripts/create_dregon_librimix.py`` /
``scripts/create_dataset.py`` (deleted) and were duplicated across
``online_mixing.py`` — three copies of the SNR math, two of the motor-track
resolution, two of the in-flight window. This module is the single copy.

RNG contract: the chunk extractors and renderers advance the global
``np.random`` / ``random`` generators in a fixed, documented order (the
derived-dataset generators seed them per split). New code should prefer
explicitly-passed ``np.random.Generator`` (the online stream does); the global
RNG entry points are kept for the byte-stable dataset recipes.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import librosa
import numpy as np
import soundfile as sf
import tdseries as td

from data_processing.frames import adapt_recording_frame, get_meta
from data_processing.sources.dregon import clean_command_spikes

SAMPLE_RATE = 16000
MOTOR_SAMPLE_RATE = 929.0  # Hz — default DREGON motor logging rate
NUM_ROTORS = 4

#: DREGON-LM train noise: all in_flight_noise recordings (published frames).
TRAIN_NOISE_SPLITS = ["in_flight_noise"]
#: DREGON-LM synthesized valid: these in_flight_source recordings.
VALID_NOISE_RECORDING_IDS = [
    "free-flight_whitenoise-low_room1",
    "free-flight_speech-low_room1",
]
#: DREGON-LM real-valid: raw clips of these co-recorded-source recordings.
REAL_VALID_RECORDING_IDS = [
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
]


# =============================================================================
# Audio I/O + SNR math (the single copy)
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


def adjust_length_mc(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or randomly crop a ``(C, T)`` array along the time axis."""
    current = audio.shape[-1]
    if current > target_length:
        start = np.random.randint(0, current - target_length + 1)
        return audio[:, start : start + target_length]
    if current < target_length:
        return np.pad(audio, ((0, 0), (0, target_length - current)), mode="constant")
    return audio


def calculate_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB (inf for silent noise)."""
    speech_power = np.sum(speech**2)
    noise_power = np.sum(noise**2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(speech_power / max(noise_power, 1e-10))


def scale_noise_to_snr(speech: np.ndarray, noise: np.ndarray, target_snr: float) -> np.ndarray:
    """``noise * scale`` so ``speech`` sits ``target_snr`` dB above it.

    The offline (LibriMix) convention: the SPEECH is the reference and the
    noise is scaled onto it — the mirror of :func:`scale_source_to_snr`, which
    the online/streaming path uses. A silent speech or noise draw is a no-op
    (there is no finite scale that hits the target).
    """
    speech_power = np.sum(speech**2)
    noise_power = np.sum(noise**2)
    if noise_power <= 0 or speech_power <= 0:
        return noise
    target_noise_power = speech_power / (10 ** (target_snr / 10))
    return noise * np.sqrt(target_noise_power / noise_power)


def mix_audio(
    speech: np.ndarray, noise: np.ndarray, target_snr: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The DN-LM mix: scale the noise to ``target_snr`` (if given), then add.

    Unlike :func:`mix_at_snr` (the DREGON-LM recipe) this applies **no**
    internal anti-clip scaling — the caller (``mix_dn_lm``) owns that step.
    Kept faithful to the Paper-1 recipe byte-for-byte.
    """
    if target_snr is not None:
        noise = scale_noise_to_snr(speech, noise, target_snr)
    return speech + noise, speech, noise


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The DREGON-LM mix: :func:`mix_audio` plus anti-clip scaling.

    Returns ``(mixture, scaled_speech, scaled_noise)``; anti-clipping scales
    all three down jointly if the mixture peaks above 0.95.
    """
    mixture, speech, noise = mix_audio(speech, noise, target_snr=target_snr)

    max_val = np.abs(mixture).max()
    if max_val > 0.95:
        scale_factor = 0.95 / max_val
        mixture = mixture * scale_factor
        speech = speech * scale_factor
        noise = noise * scale_factor

    return mixture.astype(np.float32), speech.astype(np.float32), noise.astype(np.float32)


def generate_white_noise(length: int, snr_db: float, speech: np.ndarray) -> np.ndarray:
    """White noise mixed with ``speech`` at ``snr_db`` (noise below speech)."""
    noise = np.random.randn(length).astype(np.float32)
    return (speech + scale_noise_to_snr(speech, noise, snr_db)).astype(np.float32)


# ─── Source-to-noise SNR (the online/streaming convention) ────────────────────
#
# Unlike ``mix_at_snr`` (which scales the noise to sit *below* the speech),
# these scale the SOURCE onto the noise reference — the convention the online
# mixer, the SE-valid builders, and ``streams.mix_frames`` share. At the
# project's ultra-low SNRs the source is scaled *down* onto a fixed-level
# noise bed; the scaled source is also the correct clean reference for
# speech-enhancement targets (SI-SDR is computed against exactly the speech
# component present in the mixture).


def scale_source_to_snr(
    source: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    per_channel: bool = False,
    ref_power_floor: float | None = None,
) -> np.ndarray:
    """Return ``source * scale`` — the clean source as it appears in the mixture.

    ``scale`` puts ``source * scale`` at ``snr_db`` relative to ``noise``
    (globally, or per channel when ``per_channel``). The mixture is then
    ``noise + scaled_source``.

    ``ref_power_floor`` (mean power, that is RMS squared) raises the reference
    noise power to ``max(noise_power, ref_power_floor)``. It exists because a
    near-silent noise chunk scales the source down with it: every zero-RPS
    training sample then becomes globally quiet, and the model learns that
    level shortcut instead of an off state. The floor keeps the sample at a
    usable level while the noise itself stays quiet. ``None`` (the default) is
    the unfloored behavior.
    """
    eps = 1e-12
    if per_channel:
        noise_power = np.mean(noise.astype(np.float64) ** 2, axis=1, keepdims=True)
        source_power = np.mean(source.astype(np.float64) ** 2, axis=1, keepdims=True)
    else:
        noise_power = np.array([[np.mean(noise.astype(np.float64) ** 2)]])
        source_power = np.array([[np.mean(source.astype(np.float64) ** 2)]])
    if ref_power_floor is not None:
        noise_power = np.maximum(noise_power, float(ref_power_floor))
    scale = np.sqrt((noise_power * (10.0 ** (float(snr_db) / 10.0))) / (source_power + eps))
    return (source * scale.astype(np.float32)).astype(np.float32)


#: A draw at or below this mean power is digital silence. Silent draws must
#: never reach :func:`scale_source_to_snr`: it scales the SOURCE by
#: sqrt(noise_power/...), so a silent NOISE draw yields scale == 0 -> the clean
#: target AND the mixture both become all-zeros. Measured on the drone pool,
#: `drone_audio` alone contributes ~5.8% silent draws; with an SI-SDR loss such
#: a sample returns a constant +80 dB with zero gradient, and with a magnitude
#: loss it actively teaches "output silence".
MIN_DRAW_POWER = 1e-12
MAX_DRAW_RETRIES = 8


def is_silent(x: np.ndarray) -> bool:
    """True when ``x`` carries no usable energy (digital silence)."""
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) <= MIN_DRAW_POWER


def mix_at_source_to_noise_snr(
    source: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    per_channel: bool = False,
    ref_power_floor: float | None = None,
) -> np.ndarray:
    """``noise + scale_source_to_snr(source, noise, snr_db)`` — the mixture.

    ``ref_power_floor`` is the source-scaling reference floor — see
    :func:`scale_source_to_snr`.
    """
    scaled = scale_source_to_snr(
        source, noise, snr_db, per_channel=per_channel, ref_power_floor=ref_power_floor
    )
    return (noise + scaled).astype(np.float32)


# =============================================================================
# Rotor-speed tracks + in-flight windows (the single copy)
# =============================================================================


def resolve_motor_tracks(tf: td.Frame) -> tuple[str, str, bool]:
    """Resolve the rotor-speed entry names of a noise ``td.Frame``.

    Returns ``(detect_key, rps_key, needs_cleaning)``:

    - ``detect_key``: entry used for in-flight window detection (real measured
      speeds when available — they capture spindown during landing).
    - ``rps_key``: entry read as rotor-speed ground truth.
    - ``needs_cleaning``: whether ``clean_command_spikes`` must be applied.

    Two conventions are supported so that any aligned ``td.Frame`` can serve as
    a noise source:

    - **Raw DREGON loads**: separate ``motors_measured`` (real, preferred for
      detection) and ``motors_command`` (cleaner, preferred as GT), carrying
      logging freezes → ``needs_cleaning=True``. (Published ``DREGON-frames``
      are pre-cleaned and are read through the generic path below.)
    - **Generic / published frames**: a single ``rps`` entry of aligned rotor
      speeds (rev/s), no cleaning.
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


def find_inflight_window(
    tf: td.Frame,
    motor_key: str,
    min_motor_rps: float,
    clean: bool = True,
) -> tuple[float, float]:
    """Return (t_start, t_end) of the in-flight window (absolute seconds).

    First and last absolute times where **all 4 rotors** exceed
    ``min_motor_rps``. For raw DREGON command telemetry ``clean=True`` strips
    the pre-takeoff logging artefact first; for already-clean tracks (generic
    ``rps``) pass ``clean=False``. Raises ``ValueError`` when no window exists.
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


def _valid_span(
    tf: td.Frame, duration_sec: float, min_motor_rps: float
) -> tuple[str, str, bool, float, float]:
    """The ``(detect_key, rps_key, needs_clean, valid_start, valid_end)`` span
    a chunk of ``duration_sec`` may be cut from (audio ∩ telemetry ∩ flight)."""
    detect_key, rps_key, needs_clean = resolve_motor_tracks(tf)
    audio_s = tf["audio"]
    detect_s = tf[detect_key]
    valid_start = max(audio_s.t_start, detect_s.t_start)
    valid_end = min(audio_s.t_end, detect_s.t_end)
    if min_motor_rps > 0.0:
        t_fl_start, t_fl_end = find_inflight_window(
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
    return detect_key, rps_key, needs_clean, valid_start, valid_end


def _cut_chunk(
    tf: td.Frame, rps_key: str, needs_clean: bool, start_sec: float, duration_sec: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cut ``[start_sec, start_sec+duration_sec]`` from a noise frame."""
    sliced = tf.time[start_sec : start_sec + duration_sec]
    audio_s = tf["audio"]

    audio_samples = np.asarray(sliced["audio"].data)
    if audio_samples.ndim == 1:
        audio_samples = audio_samples[np.newaxis, :]
    audio = audio_samples.astype(np.float32)  # (C, N)

    motor_sliced = sliced[rps_key]
    if motor_sliced.data is not None:
        vals = np.asarray(motor_sliced.data).copy()  # (4, M) — time-last
        rps = (clean_command_spikes(vals) if needs_clean else vals).astype(np.float32)
    else:
        rps = np.zeros((NUM_ROTORS, 0), dtype=np.float32)

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


def extract_multichannel_noise_chunk(
    tf: td.Frame,
    duration_sec: float,
    min_motor_rps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract a random ``(C, n_samples)`` noise chunk with aligned RPS.

    Returns ``(audio (C, N) float32, rps (4, M) float32, metadata)`` where
    metadata holds ``recording_id`` / ``start_time`` / ``motor_sample_rate`` /
    ``n_channels``. ``min_motor_rps`` > 0 restricts sampling to the in-flight
    window (set 30.0 to exclude takeoff/ramp-up).
    """
    _, rps_key, needs_clean, valid_start, valid_end = _valid_span(tf, duration_sec, min_motor_rps)
    start_sec = np.random.uniform(valid_start, valid_end - duration_sec)
    return _cut_chunk(tf, rps_key, needs_clean, start_sec, duration_sec)


def extract_non_overlapping_multichannel_chunks(
    tf: td.Frame,
    duration_sec: float,
    min_motor_rps: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """ALL non-overlapping ``(C, N)`` chunks spanning the in-flight window, in
    order (a trailing remainder shorter than ``duration_sec`` is dropped)."""
    _, rps_key, needs_clean, valid_start, valid_end = _valid_span(tf, duration_sec, min_motor_rps)
    n_chunks = int((valid_end - valid_start) // duration_sec)
    return [
        _cut_chunk(tf, rps_key, needs_clean, valid_start + i * duration_sec, duration_sec)
        for i in range(n_chunks)
    ]


# =============================================================================
# Composed noise pools from published frames datasets
# =============================================================================


def load_noise_source_frames(specs: list[dict[str, Any]], *, sample_rate: int) -> list[td.Frame]:
    """Compose a noise-source pool (``list[td.Frame]``) from declarative specs.

    Each spec selects recordings from one published ``tdframe-v1`` dataset::

        {"dataset": "DREGON-frames", "splits": ["in_flight_noise"],
         "exclude_recording_ids": ["free-flight_nosource_room1"]}
        {"dataset": "michaels-frames@<ver>", "recording_ids": ["FLY125"]}

    Keys: ``dataset`` (required, optional ``@version``), ``splits`` /
    ``split``, ``recording_ids``, ``exclude_recording_ids``, ``take``. Every
    selected frame is reduced to the canonical (audio + rps + meta) noise
    frame at ``sample_rate`` (:func:`frames.adapt_recording_frame`) — fixes
    are baked in at derivation time, so nothing is re-cleaned here.
    """
    from data_processing.streams import iter_published_frames

    frames: list[td.Frame] = []
    for spec in specs:
        dataset = str(spec["dataset"])
        name, _, version = dataset.partition("@")
        splits = spec.get("splits")
        if splits is None and spec.get("split") is not None:
            splits = [spec["split"]]
        wanted = {str(x) for x in spec["recording_ids"]} if spec.get("recording_ids") else None
        excluded = (
            {str(x) for x in spec["exclude_recording_ids"]}
            if spec.get("exclude_recording_ids")
            else None
        )
        kept = 0
        for frame in iter_published_frames(
            name, version or None, splits=[str(s) for s in splits] if splits else None
        ):
            rid = str(get_meta(frame, "recording_id", ""))
            if wanted is not None and rid not in wanted:
                continue
            if excluded is not None and rid in excluded:
                continue
            adapted = adapt_recording_frame(frame, sample_rate=sample_rate)
            if adapted is None:
                continue
            frames.append(adapted)
            kept += 1
            if spec.get("take") is not None and kept >= int(spec["take"]):
                break
    return frames


# =============================================================================
# Sample renderers (the DREGON-LM / DN-LM recipes)
# =============================================================================


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
    ``noise`` each ``(T, C)`` float32 plus ``rps`` ``(4, M)``, and ``meta`` is
    the per-sample metadata dict (without ``id``). Advances
    ``random``/``np.random`` in exactly the historical order (record pick →
    chunk extraction retries → per-channel source draw → per-channel SNR), so
    re-derivations seeded identically reproduce the published recipe. Keep
    that order stable.
    """
    record = random.choice(noise_records)
    noise = None
    rps = np.zeros((NUM_ROTORS, 0), dtype=np.float32)
    noise_meta: dict[str, Any] = {}
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
    if noise is None:
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


def infer_source_type(recording_id: str) -> str:
    """Infer the co-recorded source type from a recording id.

    Michael's recordings are pure drone noise (no co-recorded source), so they
    map to ``"nosource"`` like DREGON ``*_nosource_*`` recordings.
    """
    rid = recording_id.lower()
    if "speech" in rid:
        return "speech"
    if "whitenoise" in rid:
        return "whitenoise"
    if "nosource" in rid or "michaels" in rid or rid.startswith("fly"):
        return "nosource"
    return "unknown"


def iter_real_valid_clips(
    records: list[td.Frame],
    *,
    num_samples: int,
    sample_duration: float,
    sample_rate: int,
    seed: int,
    min_motor_rps: float = 0.0,
    max_non_overlapping: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """Yield ``(audio (C, T), rps (4, M), meta)`` raw recording clips — the
    DREGON-LM real-valid recipe (no mixing; ``mixture`` IS the raw recording).

    Two modes: random draws (``num_samples`` chunks at random positions, may
    overlap) or ``max_non_overlapping`` (every non-overlapping chunk of every
    recording's in-flight window, in order; ``num_samples`` ignored).
    """
    random.seed(seed)
    np.random.seed(seed)
    if not records:
        raise ValueError("no recordings given for real-valid clips")

    target_length = int(sample_duration * sample_rate)

    def _meta(rid: str, chunk_meta: dict) -> dict:
        return {
            "recording_id": rid,
            "source_type": infer_source_type(rid),
            "start_time": chunk_meta.get("start_time", 0.0),
            "duration": sample_duration,
            "n_channels": int(chunk_meta.get("n_channels", 0)),
            "motor_sample_rate": chunk_meta.get("motor_sample_rate", MOTOR_SAMPLE_RATE),
            "is_real_recording": True,
        }

    if max_non_overlapping:
        for tf in records:
            rid = str(get_meta(tf, "recording_id", "?"))
            try:
                chunks = extract_non_overlapping_multichannel_chunks(
                    tf, duration_sec=sample_duration, min_motor_rps=min_motor_rps
                )
            except ValueError as e:
                print(f"  Skipping {rid}: {e}")
                continue
            for audio, rps, chunk_meta in chunks:
                yield adjust_length_mc(audio, target_length), rps, _meta(rid, chunk_meta)
        return

    for _ in range(num_samples):
        record = random.choice(records)
        for _ in range(20):
            try:
                audio, rps, chunk_meta = extract_multichannel_noise_chunk(
                    record, duration_sec=sample_duration, min_motor_rps=min_motor_rps
                )
                break
            except ValueError:
                record = random.choice(records)
        else:
            raise ValueError("Could not find a valid chunk after 20 attempts")
        audio = adjust_length_mc(audio, target_length)  # (C, T)
        rid = chunk_meta["recording_id"]
        yield audio, rps, _meta(rid, chunk_meta)


# =============================================================================
# DN-LM (mono DroneNoise-LibriMix, Paper 1)
# =============================================================================


def apply_distance_attenuation(audio: np.ndarray, distance: float) -> np.ndarray:
    """Free-field inverse-distance attenuation (α = 1/d)."""
    attenuation = 1.0 / distance
    return audio * attenuation


def mix_dn_lm(
    speech: np.ndarray,
    noise: np.ndarray,
    *,
    target_length: int,
    speech_distance_range: tuple[float, float] = (5, 20),
    noise_distance: float = 0.5,
    target_snr_range: tuple[float, float] | None = (-30, 0),
) -> tuple[dict[str, np.ndarray], float, float]:
    """Core DN-LM mix of one already-loaded (speech, noise) pair — no I/O.

    Length adjustment, normalization, inverse-distance attenuation, SNR mixing
    and anti-clip scaling; returns ``(arrays, actual_snr, speech_distance)``
    with mono ``vocals``/``noise``/``mixture``. Advances
    ``np.random``/``random`` exactly as the historical loop body did — the
    caller performs the speech/noise *file* draws first; keep this call order
    stable.
    """
    speech = adjust_length(speech, target_length)
    noise = adjust_length(noise, target_length)

    speech = normalize_audio(speech)
    noise = normalize_audio(noise)

    speech_distance = random.uniform(*speech_distance_range)
    speech_attenuated = apply_distance_attenuation(speech, speech_distance)
    noise_attenuated = apply_distance_attenuation(noise, noise_distance)

    target_snr = random.uniform(*target_snr_range) if target_snr_range else None
    mixture, speech_final, noise_final = mix_audio(
        speech_attenuated, noise_attenuated, target_snr=target_snr
    )

    actual_snr = calculate_snr(speech_final, noise_final)

    max_val = max(np.abs(mixture).max(), np.abs(speech_final).max(), np.abs(noise_final).max())
    if max_val > 1.0:
        scale = 0.95 / max_val
        mixture = mixture * scale
        speech_final = speech_final * scale
        noise_final = noise_final * scale

    arrays = {"vocals": speech_final, "noise": noise_final, "mixture": mixture}
    return arrays, float(actual_snr), float(speech_distance)
