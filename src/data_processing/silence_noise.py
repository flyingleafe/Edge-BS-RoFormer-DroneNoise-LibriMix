"""Synthetic *rotors-off* noise — the zero-labeled arm of the honest base regime.

Motivation. A rotor-speed predictor that only ever sees rotating noise has no
reason to learn an **off** state. Two failure modes follow from that:

* **The level shortcut.** Real recordings contain quiet spans (rotors stopped)
  whose RMS is 0.001-0.004, while flight RMS is 0.07-0.08. If the only quiet
  chunks are also the only zero-RPS chunks, the model reads the level, not the
  comb. The mixer makes this worse: the speech source is scaled *onto* the
  noise power, so a near-silent noise chunk gives a globally quiet sample. The
  reference-power floor (``snr_ref_floor_rms``, ``mixing.scale_source_to_snr``)
  removes that half of the shortcut; this pool removes the other half.
* **Loud non-rotor noise.** A validation clip with stopped rotors peaked at
  41-50 Hz and was loud. A predictor trained only on combs reports a rotor
  speed for it.

``SilenceNoisePool`` (``kind: silence``) supplies chunks whose rotor-speed
label is **always zero** and whose level spans quiet room tone up to full
flight level. Each chunk draws one floor type:

* ``room_tone`` — colored noise, tilt alpha in U(0.5, 1.5), RMS log-uniform in
  [5e-4, 5e-3]. The measured level of a real quiet span.
* ``colored`` — tilt alpha in U(0.0, 2.0), RMS log-uniform in [5e-3, 8e-2].
  Loud audio that is not a rotor.
* ``lf_rumble`` — a weak colored base plus a dominant band-limited component,
  band low edge U(30, 60) Hz and width U(30, 120) Hz, RMS log-uniform in
  [1e-2, 8e-2]. The real failure mode above.

Synthesis is rfft shaping of white gaussian noise: one type, tilt and level
draw per chunk, one independent noise draw per channel (decorrelated mics).
Everything is numpy, so the pool is picklable and runs in the fork DataLoader
workers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tdseries as td

from data_processing.frames import make_recording_frame

#: The floor types and their default draw weights.
DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "room_tone": 0.3,
    "colored": 0.4,
    "lf_rumble": 0.3,
}

#: Per-type sampling ranges. ``rms`` is log-uniform, everything else uniform.
DEFAULT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "room_tone": {"rms": (5e-4, 5e-3), "alpha": (0.5, 1.5)},
    "colored": {"rms": (5e-3, 8e-2), "alpha": (0.0, 2.0)},
    "lf_rumble": {
        "rms": (1e-2, 8e-2),
        "band_low_hz": (30.0, 60.0),
        "band_width_hz": (30.0, 120.0),
    },
}

#: ``lf_rumble`` base component: colored, this many dB below the total.
LF_BASE_REL_DB = -20.0
#: ``lf_rumble`` base component tilt (power spectrum ~ f**-tilt).
LF_BASE_TILT = 1.0
#: Spectrum shaping clamps the frequency axis here (rule 9.1 of the tilt law:
#: f**-alpha diverges at DC).
MIN_SHAPING_HZ = 20.0


def _merge_ranges(override: Any) -> dict[str, dict[str, tuple[float, float]]]:
    """Return :data:`DEFAULT_RANGES` with ``override`` applied per type."""
    ranges = {t: dict(v) for t, v in DEFAULT_RANGES.items()}
    if not override:
        return ranges
    if not isinstance(override, dict):
        raise TypeError(f"silence ranges must be a mapping, got {type(override).__name__}")
    for name, params in override.items():
        if name not in ranges:
            raise ValueError(
                f"unknown silence floor type {name!r} in ranges; "
                f"known types: {sorted(DEFAULT_RANGES)}"
            )
        for key, val in dict(params).items():
            if key not in ranges[name]:
                raise ValueError(
                    f"unknown range key {key!r} for silence type {name!r}; "
                    f"known keys: {sorted(DEFAULT_RANGES[name])}"
                )
            ranges[name][key] = (float(val[0]), float(val[1]))
    return ranges


def _tilt_shape(freqs: np.ndarray, alpha: float) -> np.ndarray:
    """Amplitude shaping |H(f)| for a power spectrum ~ ``f**-alpha``.

    Frequency is clamped at :data:`MIN_SHAPING_HZ`, so the shaping stays finite
    at DC. Amplitude is the square root of power, hence the ``-alpha / 2``.
    """
    f = np.maximum(np.asarray(freqs, dtype=np.float64), MIN_SHAPING_HZ)
    shape = f ** (-float(alpha) / 2.0)
    shape[np.asarray(freqs) <= 0.0] = 0.0  # drop DC
    return shape


def _shaped_noise(rng: np.random.Generator, n: int, shape: np.ndarray, rms: float) -> np.ndarray:
    """White gaussian noise shaped by ``|H(f)| = shape``, scaled to ``rms``."""
    spec = np.fft.rfft(rng.standard_normal(n))
    out = np.fft.irfft(spec * shape, n=n)
    cur = float(np.sqrt(np.mean(out**2)))
    if cur <= 0.0:
        return np.zeros(n, dtype=np.float64)
    return out / cur * float(rms)


class SilenceNoisePool:
    """Rotors-off noise source (``kind: silence``).

    Same ``sample_timeframe(rng, duration_s) -> td.Frame`` interface as the
    other noise pools. The returned Frame carries ``(n_channels, T)`` audio and
    an all-zeros ``(n_rotors, M)`` rotor-speed track — the exact label.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        duration_s: float = 1.0,
        n_channels: int = 8,
        n_rotors: int = 4,
        types: dict[str, float] | None = None,
        ranges: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        self.sample_rate = int(sample_rate)
        self.chunk_s = float(duration_s)
        self.n_mics = int(n_channels)
        self.n_rotors = int(n_rotors)
        weights = (
            dict(DEFAULT_TYPE_WEIGHTS) if types is None else {k: float(v) for k, v in types.items()}
        )
        for name in weights:
            if name not in DEFAULT_TYPE_WEIGHTS:
                raise ValueError(
                    f"unknown silence floor type {name!r}; known types: "
                    f"{sorted(DEFAULT_TYPE_WEIGHTS)}"
                )
        weights = {k: v for k, v in weights.items() if v > 0.0}
        if not weights:
            raise ValueError("silence source needs at least one floor type with a positive weight")
        self.types: tuple[str, ...] = tuple(weights)
        w = np.array([weights[t] for t in self.types], dtype=np.float64)
        self.type_probs: np.ndarray = w / w.sum()
        self.ranges = _merge_ranges(ranges)
        self._base_seed = int(seed)
        # Placeholder geometry (there is no rotor here; carried for interface
        # parity with the other pools).
        self.mic_pos = np.zeros((self.n_mics, 3), dtype=np.float64)
        self.rotor_pos = np.zeros((self.n_rotors, 3), dtype=np.float64)

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> SilenceNoisePool:
        def g(key: str, default: Any = None) -> Any:
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        def _plain(value: Any) -> Any:
            if value is None or isinstance(value, dict):
                return value
            from data_processing.generated_noise import _to_plain

            return _to_plain(value)

        types = _plain(g("types"))
        return cls(
            sample_rate=sample_rate,
            duration_s=duration_s,
            n_channels=int(g("n_channels", 8)),
            n_rotors=int(g("n_rotors", 4)),
            types=dict(types) if types else None,
            ranges=_plain(g("ranges")),
            seed=int(g("seed", 0)),
        )

    def close(self) -> None:  # interface parity with the other pools
        return None

    def _draw_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """One type / tilt / level draw, shared by every channel of the chunk."""
        floor_type = str(rng.choice(np.asarray(self.types), p=self.type_probs))
        r = self.ranges[floor_type]
        lo, hi = r["rms"]
        params: dict[str, Any] = {
            "floor_type": floor_type,
            "rms": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        }
        if floor_type == "lf_rumble":
            params["band_low_hz"] = float(rng.uniform(*r["band_low_hz"]))
            params["band_width_hz"] = float(rng.uniform(*r["band_width_hz"]))
        else:
            params["alpha"] = float(rng.uniform(*r["alpha"]))
        return params

    def render(
        self, rng: np.random.Generator, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Render ``(audio (M, T), rps (R, T) all-zeros, draw parameters)``."""
        T = int(round(duration_s * self.sample_rate))
        params = self._draw_params(rng)
        freqs = np.fft.rfftfreq(T, d=1.0 / self.sample_rate)

        audio = np.empty((self.n_mics, T), dtype=np.float32)
        if params["floor_type"] == "lf_rumble":
            # A weak colored base plus a dominant band-limited component. The
            # two are independent, so their powers add: the base sits
            # LF_BASE_REL_DB below the total and the band carries the rest.
            base_rms = params["rms"] * 10.0 ** (LF_BASE_REL_DB / 20.0)
            band_rms = float(np.sqrt(max(params["rms"] ** 2 - base_rms**2, 0.0)))
            base_shape = _tilt_shape(freqs, LF_BASE_TILT)
            low = params["band_low_hz"]
            high = low + params["band_width_hz"]
            band_shape = ((freqs >= low) & (freqs <= high)).astype(np.float64)
            for m in range(self.n_mics):
                sig = _shaped_noise(rng, T, base_shape, base_rms)
                sig = sig + _shaped_noise(rng, T, band_shape, band_rms)
                audio[m] = sig.astype(np.float32)
        else:
            shape = _tilt_shape(freqs, params["alpha"])
            for m in range(self.n_mics):
                audio[m] = _shaped_noise(rng, T, shape, params["rms"]).astype(np.float32)

        rps = np.zeros((self.n_rotors, T), dtype=np.float32)
        return audio, rps, params

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        audio, rps, params = self.render(rng, duration_s)
        audio_us = td.uniform(
            np.ascontiguousarray(audio), self.sample_rate, dims=("mic", "time"), t_start=0.0
        )
        t = np.arange(audio.shape[-1], dtype=np.float64) / self.sample_rate
        rps_es = td.events(t, np.ascontiguousarray(rps), dims=("rotor", "time"), t_start=0.0)
        return make_recording_frame(
            {"audio": audio_us, "rps": rps_es},
            meta={"recording_id": "silence", "floor_type": params["floor_type"]},
            mic_pos=self.mic_pos,
            rotor_pos=self.rotor_pos,
        )
