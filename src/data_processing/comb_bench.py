"""A controlled static-comb benchmark, with rotor spread as an explicit axis.

WHY A NEW ONE. The comb-floor stream was the campaign's "static comb" task and
it cannot measure static-comb solving, for three reasons found by inspection:

* Its fixed validation set carries `flight_reuse: 32`, so 96 clips contain only
  12 distinct rotor trajectories — the audio differs per clip, the LABELS
  repeat eight times each.
* A third of those clips have all four rotors at literally identical speeds, a
  configuration the training stream never produces (0 of 40 sampled). On them
  there is no spread to resolve and a collapsed "fan" prediction is correct, so
  they reward the very degeneracy the campaign was trying to measure.
* Speech is mixed in at -30 to 0 dB SNR. That is the right task for speech
  enhancement and the wrong one for "given a static comb, restore the rotor
  speeds" — at the loud end the speech harmonics are comparable to the comb.

This module generates the task the question actually names: a pure static comb,
four rotors, nothing else but a small white floor. Rotor spread, centre rate and
excursion are explicit arguments, so results are reported as a FUNCTION of how
far apart the rotors are rather than as one number that hides it.

The rotors share ONE harmonic profile, as rotors of one airframe do. Drawing an
independent profile per rotor produces a 21.6 dB loudness spread that no real
aircraft has, and it was measured to dominate results — see the seeding notes in
`docs/experiments/synthetic-solvability-limits.md`.
"""

from __future__ import annotations

import numpy as np

from .rotor_spectral_model import ProfileRanges, _comb_waveform, sample_profile

__all__ = ["comb_clip", "REGIMES"]

#: Benchmark cells: (name, centre rev/s, spread rev/s, excursion rev/s).
#: `spread=0` keeps the degenerate case in view as a labelled cell rather than
#: letting it contaminate an aggregate.
REGIMES = (
    ("identical", 75.0, 0.0, 1.5),
    ("tight", 75.0, 2.0, 1.5),
    ("close", 75.0, 5.0, 1.5),
    ("typical", 75.0, 11.0, 1.5),
    ("wide", 75.0, 20.0, 1.5),
    ("typical-fast", 75.0, 11.0, 6.0),
    ("typical-idle", 40.0, 11.0, 1.5),
)


def comb_clip(
    seed: int, centre: float = 75.0, spread: float = 11.0, excursion: float = 1.5,
    n_rotors: int = 4, sr: int = 16000, dur_s: float = 8.0, hop: int = 512,
    n_harmonics: int = 100, noise_rms: float = 0.01,
):
    """One pure static-comb clip: ``(audio, rps, ft)``.

    ``audio`` is ``(n_samples,)``; ``rps`` is ``(n_rotors, n_frames)`` in rev/s
    on the frame grid ``ft`` (hop-aligned, matching the STFT contract).
    """
    rng = np.random.default_rng(seed)
    n_t = int(round(sr * dur_s))
    t = np.arange(n_t) / sr
    prof = sample_profile(rng, ProfileRanges(), n_harmonics=n_harmonics,
                          ref_rps=centre, sample_rate=sr)
    a_k = np.asarray(prof.a_k, dtype=np.float64)
    offs = (np.linspace(-0.5, 0.5, n_rotors) * spread) if n_rotors > 1 else np.zeros(1)
    audio = np.zeros(n_t)
    tracks = []
    for i in range(n_rotors):
        ph = rng.uniform(0.0, 2.0 * np.pi, 2)
        r = (centre + offs[i]
             + excursion * np.sin(2 * np.pi * 0.11 * t + ph[0])
             + 0.33 * excursion * np.sin(2 * np.pi * 0.37 * t + ph[1]))
        audio += _comb_waveform(r, a_k, sr, rng) * (r / 80.0) ** 2.5
        tracks.append(r)
    audio = audio + noise_rms * rng.standard_normal(n_t)
    ft = np.arange(n_t // hop + 1) * hop / sr
    rps = np.stack([np.interp(ft, t, r) for r in tracks])
    return audio, rps, ft
