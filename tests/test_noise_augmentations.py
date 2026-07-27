"""Unit tests for the G6 strong noise-chunk augmentations.

Each augmentation is exercised on synthetic combs with known teeth, and the
measurements mirror the G6 acceptance criteria: freq_scale moves the comb and
the labels by alpha; spectral_recolor changes magnitudes but not peak bins;
tooth_dropout notches >10 dB at the targeted teeth and <1 dB elsewhere;
random_reverb preserves RMS (+-1 dB) and peak bins (+-1 bin); floor_inject
raises the inter-tooth floor while keeping tooth peaks within +-1 dB.
"""

from __future__ import annotations

import numpy as np
import tdseries as td

from data_processing.frames import make_recording_frame
from data_processing.noise_augmentations import (
    _floor_inject,
    _freq_scale,
    _random_reverb,
    _spec_mask,
    _spectral_recolor,
    _tooth_dropout,
    maybe_apply_noise_augmentation,
)

SR = 16000
T = 16000
LABEL_RATE = 100.0
L = int(np.ceil(T / SR * LABEL_RATE)) + 1


def _comb_audio(f0: float, n_harm: int = 20, channels: int = 2, amp: float = 0.05) -> np.ndarray:
    t = np.arange(T, dtype=np.float64) / SR
    x = sum(amp / k * np.sin(2 * np.pi * k * f0 * t + 0.3 * k) for k in range(1, n_harm + 1))
    return np.tile(np.asarray(x, dtype=np.float32), (channels, 1))


def _const_label(rps_per_rotor: list[float]) -> np.ndarray:
    return np.stack(
        [np.full(L, v, dtype=np.float32) for v in rps_per_rotor], axis=0
    )  # (R, L) @ 100 Hz


def _mag(audio_ch: np.ndarray) -> np.ndarray:
    """Hann-windowed |rfft| of one channel — 1 Hz bins over the 1 s chunk."""
    return np.abs(np.fft.rfft(audio_ch * np.hanning(audio_ch.shape[-1])))


def _band_energy(mag: np.ndarray, f_center: float, half_hz: float) -> float:
    freqs = np.fft.rfftfreq(T, 1.0 / SR)
    band = (freqs >= f_center - half_hz) & (freqs <= f_center + half_hz)
    return float(np.sum(mag[band] ** 2))


def _peak_bin(mag: np.ndarray, f_center: float, half_hz: float = 10.0) -> int:
    freqs = np.fft.rfftfreq(T, 1.0 / SR)
    idx = np.flatnonzero((freqs >= f_center - half_hz) & (freqs <= f_center + half_hz))
    return int(idx[np.argmax(mag[idx])])


def _db(x: float) -> float:
    return 10.0 * np.log10(x + 1e-30)


# ── freq_scale ──────────────────────────────────────────────────────────────


def test_freq_scale_moves_comb_and_labels():
    audio = _comb_audio(60.0)
    label = _const_label([60.0, 60.0, 60.0, 60.0])
    out, new_label = _freq_scale(
        audio,
        label,
        {"alpha_low": 1.2, "alpha_high": 1.2},
        np.random.default_rng(0),
        sample_rate=SR,
        label_rate_hz=LABEL_RATE,
    )
    # Natural scaled length — NO padding (the sourcing pipeline oversamples
    # the window instead; downstream extraction crops).
    assert out.shape == (audio.shape[0], int(round(T / 1.2))) and out.dtype == np.float32
    n = out.shape[-1]
    mag = np.abs(np.fft.rfft(out[0] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    for k in (1, 2, 3):
        band = (freqs > 72.0 * k - 2.0) & (freqs < 72.0 * k + 2.0)
        near = (freqs > 72.0 * k - 12.0) & (freqs < 72.0 * k + 12.0)
        assert mag[band].max() >= 0.9 * mag[near].max(), k
    # Labels: 72 everywhere on the scaled time base — no zero tail.
    assert new_label.shape[-1] == int(np.ceil(n / SR * LABEL_RATE))
    assert np.allclose(new_label, 72.0, atol=1e-3)


# ── spectral_recolor ────────────────────────────────────────────────────────


def test_spectral_recolor_changes_magnitude_not_peaks():
    audio = _comb_audio(60.0)
    label = _const_label([60.0] * 4)
    out, new_label = _spectral_recolor(
        audio, label, {}, np.random.default_rng(1), sample_rate=SR, label_rate_hz=LABEL_RATE
    )
    assert np.array_equal(new_label, label)
    mag_in, mag_out = _mag(audio[0]), _mag(out[0])
    ratios_db = []
    for k in range(1, 11):
        assert abs(_peak_bin(mag_out, 60.0 * k) - _peak_bin(mag_in, 60.0 * k)) == 0, k
        ratios_db.append(
            _db(_band_energy(mag_out, 60.0 * k, 3.0)) - _db(_band_energy(mag_in, 60.0 * k, 3.0))
        )
    # Magnitudes actually moved (some tooth recolored by > 2 dB)…
    assert max(abs(r) for r in ratios_db) > 2.0
    # …and not uniformly (it is a curve, not a global gain).
    assert max(ratios_db) - min(ratios_db) > 1.0


# ── random_reverb ───────────────────────────────────────────────────────────


def test_random_reverb_preserves_rms_and_peaks():
    audio = _comb_audio(60.0)
    label = _const_label([60.0] * 4)
    out, new_label = _random_reverb(
        audio,
        label,
        {"n_rirs": 16},
        np.random.default_rng(2),
        sample_rate=SR,
        label_rate_hz=LABEL_RATE,
    )
    assert np.array_equal(new_label, label)
    rms_in = np.sqrt(np.mean(audio**2))
    rms_out = np.sqrt(np.mean(out**2))
    assert abs(20.0 * np.log10(rms_out / rms_in)) < 1.0  # +-1 dB (renormalized)
    mag_in, mag_out = _mag(audio[0]), _mag(out[0])
    for k in (1, 2, 3, 5, 8):
        assert abs(_peak_bin(mag_out, 60.0 * k) - _peak_bin(mag_in, 60.0 * k)) <= 1, k


# ── tooth_dropout ───────────────────────────────────────────────────────────


def test_tooth_dropout_notches_targeted_teeth_only():
    # Audio contains ONLY rotor 0's comb (60 rev/s); rotors 1-3 carry a far-away
    # label so their teeth are silent — collateral is measured on rotor 0's
    # untargeted teeth.
    audio = _comb_audio(60.0)
    label = _const_label([60.0, 90.0, 90.0, 90.0])
    out, new_label = _tooth_dropout(
        audio,
        label,
        {"teeth": [[0, 3]]},
        np.random.default_rng(3),
        sample_rate=SR,
        label_rate_hz=LABEL_RATE,
    )
    assert np.array_equal(new_label, label)
    mag_in, mag_out = _mag(audio[0]), _mag(out[0])
    # Targeted tooth (180 Hz): > 10 dB drop.
    drop = _db(_band_energy(mag_in, 180.0, 5.0)) - _db(_band_energy(mag_out, 180.0, 5.0))
    assert drop > 10.0, drop
    # Untargeted teeth (their bands lie outside the +-2-bin notch): < 1 dB.
    for k in (1, 2, 4, 5, 8):
        delta = abs(
            _db(_band_energy(mag_out, 60.0 * k, 5.0)) - _db(_band_energy(mag_in, 60.0 * k, 5.0))
        )
        assert delta < 1.0, (k, delta)


def test_tooth_dropout_silent_rotor_is_noop():
    audio = _comb_audio(60.0)
    label = _const_label([60.0, 90.0, 90.0, 90.0])
    out, _ = _tooth_dropout(
        audio,
        label,
        {"teeth": [[1, 3]]},  # rotor 1 has no audio: 270 Hz band is empty
        np.random.default_rng(4),
        sample_rate=SR,
        label_rate_hz=LABEL_RATE,
    )
    mag_in, mag_out = _mag(audio[0]), _mag(out[0])
    for k in range(1, 9):
        delta = abs(
            _db(_band_energy(mag_out, 60.0 * k, 5.0)) - _db(_band_energy(mag_in, 60.0 * k, 5.0))
        )
        assert delta < 1.0, (k, delta)


# ── spec_mask ───────────────────────────────────────────────────────────────


def test_spec_mask_shapes_and_energy():
    audio = _comb_audio(60.0)
    label = _const_label([60.0] * 4)
    out, new_label = _spec_mask(
        audio, label, {}, np.random.default_rng(5), sample_rate=SR, label_rate_hz=LABEL_RATE
    )
    assert out.shape == audio.shape and out.dtype == np.float32
    assert np.array_equal(new_label, label)
    assert np.isfinite(out).all()
    assert not np.array_equal(out, audio)  # something was masked
    # Masking only removes energy (up to COLA reconstruction tolerance).
    assert float(np.sum(out**2)) <= float(np.sum(audio**2)) * 1.01


# ── floor_inject ────────────────────────────────────────────────────────────


def test_floor_inject_raises_floor_keeps_teeth():
    audio = _comb_audio(60.0)
    label = _const_label([60.0] * 4)
    out, new_label = _floor_inject(
        audio,
        label,
        {"tilt_low": 1.0, "tilt_high": 1.0, "level_low_db": -10.0, "level_high_db": -10.0},
        np.random.default_rng(6),
        sample_rate=SR,
        label_rate_hz=LABEL_RATE,
    )
    assert np.array_equal(new_label, label)
    mag_in, mag_out = _mag(audio[0]), _mag(out[0])
    freqs = np.fft.rfftfreq(T, 1.0 / SR)
    # Inter-tooth bins: 100..1300 Hz, > 10 Hz away from every 60 Hz multiple.
    off_tooth = (freqs > 100) & (freqs < 1300) & (np.abs((freqs + 30) % 60.0 - 30.0) > 10.0)
    assert np.median(mag_out[off_tooth]) > 3.0 * np.median(mag_in[off_tooth])
    for k in (1, 2, 3, 5):
        delta = abs(
            _db(_band_energy(mag_out, 60.0 * k, 3.0)) - _db(_band_energy(mag_in, 60.0 * k, 3.0))
        )
        assert delta < 1.0, (k, delta)


# ── policy entry point / Frame round trip ───────────────────────────────────


def _comb_frame(duration_s: float = 1.2) -> td.Frame:
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float64) / SR
    x = sum(0.05 / k * np.sin(2 * np.pi * k * 60.0 * t) for k in range(1, 15))
    audio = np.tile(np.asarray(x, dtype=np.float32), (2, 1))
    audio_us = td.uniform(audio, SR, dims=("mic", "time"), t_start=0.0)
    motor_t = np.linspace(0.0, duration_s - 0.01, 120, dtype=np.float64)
    rps = np.full((4, 120), 60.0, dtype=np.float32)
    rps_es = td.events(motor_t, rps, dims=("rotor", "time"), t_start=0.0, t_end=duration_s)
    return make_recording_frame({"audio": audio_us, "rps": rps_es}, meta={"recording_id": "synth"})


def test_maybe_apply_fires_and_rebuilds_frame():
    frame = _comb_frame()
    spec = {"probability": 1.0, "choices": [{"freq_scale": {"alpha_low": 1.2, "alpha_high": 1.2}}]}
    out = maybe_apply_noise_augmentation(
        frame, spec, np.random.default_rng(7), target_len=T, sample_rate=SR
    )
    assert out is not frame
    audio = np.asarray(out["audio"].data)
    # freq_scale emits the natural scaled length (T/alpha); the true
    # target_len crop happens downstream in the mixing pipeline.
    assert audio.shape == (2, int(round(T / 1.2))) and audio.dtype == np.float32
    rps = np.asarray(out["rps"].data)
    assert rps.shape[0] == 4
    assert abs(float(rps[:, : int(0.8 * LABEL_RATE)].mean()) - 72.0) < 0.5
    # The rebuilt frame slots into the downstream interpolation path.
    from data_processing.online_mixing import _resolve_motor_tracks, interpolate_rps_to_stft_grid

    assert _resolve_motor_tracks(out)[1] == "rps"
    grid = interpolate_rps_to_stft_grid(out, n_frames=32, hop_length=512)
    assert grid.shape == (4, 32) and np.isfinite(grid).all()


def test_maybe_apply_no_fire_returns_same_frame():
    frame = _comb_frame()
    assert (
        maybe_apply_noise_augmentation(
            frame, None, np.random.default_rng(8), target_len=T, sample_rate=SR
        )
        is frame
    )
    assert (
        maybe_apply_noise_augmentation(
            frame,
            {"probability": 0.0, "choices": ["spec_mask"]},
            np.random.default_rng(9),
            target_len=T,
            sample_rate=SR,
        )
        is frame
    )
