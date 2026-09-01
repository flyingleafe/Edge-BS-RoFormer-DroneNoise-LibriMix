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

from . import rps_synthesis
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


#: Trajectory rate for the OU draw. RPS is slow; the audio-rate track is an
#: interpolation of this, exactly as the online-mix pools do it.
TRAJ_FS = 1000.0


def rotor_tracks_ou(rng, centre, spread, excursion, n_rotors, t):
    """Calibrated rotor trajectories at the audio grid ``t``: ``(n_rotors, len(t))``.

    WHY NOT SINUSOIDS. This benchmark originally drew each rotor as two fixed
    sinusoids (0.11 and 0.37 Hz). Measured against the real telemetry of the
    beat-VK protocol on the 32 ms frame grid, that made it 21x too smooth on
    average and 42x too smooth at the 99th percentile:

        source                          mean     p99      max   rev/s per frame
        two sinusoids (the old default) 0.028   0.069    0.070
        REAL telemetry, 15 windows      0.601   2.916   25.337

    Its HARDEST frame-to-frame change was smaller than real telemetry's MEAN, so
    every trajectory it produced sat far inside any plausible transition band and
    the benchmark had no power to test a temporal model at all. That is why a
    slew sweep on it moved nothing (2.820 / 2.821 / 2.835) while the same knob is
    badly mis-set on real data.

    `rps_synthesis` already solves this: four Ornstein-Uhlenbeck control modes
    (collective, roll, pitch, yaw) through the quadrotor mixer, with the OU
    parameters fitted to DREGON's 929 Hz in-flight telemetry. Taking its dynamics
    and imposing this benchmark's axes on top keeps both properties:

    * `centre` sets the collective mean, so the cell's speed band is unchanged.
    * `spread` rescales the DIFFERENTIAL part of the draw so the clip's mean
      rotor spread hits the cell's target, leaving the temporal structure of both
      the common and differential modes intact.
    * `excursion` maps to `aggressiveness` (1.5 -> 1.0, a typical free flight),
      the generator's own global multiplier on every mode's dynamic std.

    A consequence worth stating: rotors now WANDER and cross rather than holding
    fixed offsets, which is what real ones do and what the old benchmark could
    not represent.
    """
    dur = float(t[-1] - t[0]) + 1.0 / TRAJ_FS
    w = rps_synthesis.generate(dur, TRAJ_FS, aggressiveness=max(excursion / 1.5, 1e-3),
                               rng=rng)
    w = w[:n_rotors] if w.shape[0] >= n_rotors else np.repeat(w, n_rotors, 0)[:n_rotors]
    mid = w.mean(axis=0, keepdims=True)
    off = w - mid                                   # differential part
    # SPREAD IS THE SEPARATION OF THE ROTORS' MEANS, and the wander around those
    # means is kept at its calibrated amplitude regardless. Rescaling the
    # instantaneous offsets instead would make `spread=0` mean four LITERALLY
    # identical rotors, where the cell has always meant four rotors sharing a
    # mean speed and wandering through one another -- the interleaving case, and
    # the hardest one in the benchmark.
    wander = off - off.mean(axis=1, keepdims=True)
    means = (np.linspace(-0.5, 0.5, n_rotors) * float(spread))[:, None]
    track = (mid - mid.mean()) + float(centre) + means + wander
    tt = np.arange(w.shape[1]) / TRAJ_FS
    return np.stack([np.interp(t, tt, r) for r in track])


def comb_clip(
    seed: int, centre: float = 75.0, spread: float = 11.0, excursion: float = 1.5,
    n_rotors: int = 4, sr: int = 16000, dur_s: float = 8.0, hop: int = 512,
    n_harmonics: int = 100, noise_rms: float = 0.01, trajectory: str = "ou",
):
    """One pure static-comb clip: ``(audio, rps, ft)``.

    ``audio`` is ``(n_samples,)``; ``rps`` is ``(n_rotors, n_frames)`` in rev/s
    on the frame grid ``ft`` (hop-aligned, matching the STFT contract).

    ``trajectory="ou"`` (default) draws DREGON-calibrated dynamics — see
    `rotor_tracks_ou`. ``trajectory="sinusoid"`` restores the original
    two-sinusoid draw and exists only to reproduce numbers measured before
    2026-09-01; it is 42x too smooth and must not be used for new results.
    """
    rng = np.random.default_rng(seed)
    n_t = int(round(sr * dur_s))
    t = np.arange(n_t) / sr
    prof = sample_profile(rng, ProfileRanges(), n_harmonics=n_harmonics,
                          ref_rps=centre, sample_rate=sr)
    a_k = np.asarray(prof.a_k, dtype=np.float64)
    if trajectory == "ou":
        tracks = list(rotor_tracks_ou(rng, centre, spread, excursion, n_rotors, t))
    elif trajectory == "sinusoid":
        offs = (np.linspace(-0.5, 0.5, n_rotors) * spread) if n_rotors > 1 else np.zeros(1)
        tracks = []
        for i in range(n_rotors):
            ph = rng.uniform(0.0, 2.0 * np.pi, 2)
            tracks.append(centre + offs[i]
                          + excursion * np.sin(2 * np.pi * 0.11 * t + ph[0])
                          + 0.33 * excursion * np.sin(2 * np.pi * 0.37 * t + ph[1]))
    else:
        raise ValueError(f"unknown trajectory {trajectory!r}")
    audio = np.zeros(n_t)
    for r in tracks:
        audio += _comb_waveform(r, a_k, sr, rng) * (r / 80.0) ** 2.5
    audio = audio + noise_rms * rng.standard_normal(n_t)
    ft = np.arange(n_t // hop + 1) * hop / sr
    rps = np.stack([np.interp(ft, t, r) for r in tracks])
    return audio, rps, ft
