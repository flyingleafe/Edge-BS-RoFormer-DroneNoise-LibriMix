"""The stochastic-comb counterpart of `comb_bench`, on the same regime axes.

WHY IT IS BUILT THIS WAY. The two families must differ in ONE thing — how the
comb is rendered — or a difference in results cannot be attributed. So this
module reuses `comb_bench`'s rotor trajectories verbatim (same centre, spread,
excursion, same two-sinusoid wander, same seed) and changes only what is put at
`k * r(t)`:

* `comb_bench` lays down a deterministic sinusoid per harmonic with a constant
  amplitude — a delta-thin line whose power is exactly `a_k`.
* Here each harmonic is a LORENTZIAN of half width `gamma0 + slope * k` Hz whose
  power drifts in time as a Gaussian process, and the whole spectrum is realized
  by filtering white noise. Every bin is then an exponential random variable
  about its mean, so even a line bin flickers by about 5.2 dB frame to frame.

That second family is the one the campaign measured at 7.2-8.2 dB of
peak-to-bulk contrast against 5.6-6.3 dB for structureless noise and 15-25 dB
for the static comb (`synthetic-solvability-limits.md`). Whether a comb that
faint is recoverable at all is the question this benchmark exists to answer, so
the knobs that set the contrast are exposed rather than buried: `gamma0_hz` and
`gamma_slope_hz` (linewidth), `line_mode` (Rayleigh realization or coherent
tones), and `harm_gp_db` (how hard the amplitudes breathe).

The four rotors share one airframe timbre (`rotor_similarity` fixed high), for
the reason recorded in `comb_bench`: independent per-rotor profiles make the
rotors separable by loudness, which no real quadrotor offers.
"""

from __future__ import annotations

import numpy as np

from .stochastic_rotor_noise import StochasticRanges, sample_params, synthesize

__all__ = ["stoch_comb_clip", "rotor_tracks"]


def rotor_tracks(rng, centre: float, spread: float, excursion: float,
                 n_rotors: int, t: np.ndarray) -> np.ndarray:
    """The trajectory construction of `comb_bench.comb_clip`, shared verbatim."""
    offs = (np.linspace(-0.5, 0.5, n_rotors) * spread) if n_rotors > 1 else np.zeros(1)
    out = []
    for i in range(n_rotors):
        ph = rng.uniform(0.0, 2.0 * np.pi, 2)
        out.append(centre + offs[i]
                   + excursion * np.sin(2 * np.pi * 0.11 * t + ph[0])
                   + 0.33 * excursion * np.sin(2 * np.pi * 0.37 * t + ph[1]))
    return np.stack(out)


def stoch_comb_clip(
    seed: int, centre: float = 75.0, spread: float = 11.0, excursion: float = 1.5,
    n_rotors: int = 4, sr: int = 16000, dur_s: float = 8.0, hop: int = 512,
    n_harmonics: int = 100, line_mode: str = "stochastic",
    gamma0_hz: tuple[float, float] = (0.5, 4.0),
    gamma_slope_hz: tuple[float, float] = (0.05, 0.8),
    harm_gp_db: tuple[float, float] | None = None,
    floor_rel_db: tuple[float, float] | None = None,
):
    """One stochastic-comb clip: ``(audio, rps, ft)``, same shapes as `comb_clip`."""
    # The trajectories come from `comb_clip` ITSELF, not from a re-implementation
    # of its sampler. Its per-rotor phase draws are interleaved with waveform
    # synthesis, so any reconstruction of the draw order is fragile; calling it
    # and discarding its audio makes the two families share their labels exactly,
    # which is what lets a difference in results be attributed to the rendering.
    from .comb_bench import comb_clip
    _, rps_hop, ft = comb_clip(seed=seed, centre=centre, spread=spread,
                               excursion=excursion, n_rotors=n_rotors, sr=sr,
                               dur_s=dur_s, hop=hop, n_harmonics=n_harmonics)
    n_t = int(round(sr * dur_s))
    t = np.arange(n_t) / sr
    tracks = np.stack([np.interp(t, ft, r) for r in rps_hop])
    rng = np.random.default_rng(seed + 900_000)

    kw = dict(gamma0_hz=tuple(gamma0_hz), gamma_slope_hz=tuple(gamma_slope_hz),
              rotor_similarity=(0.9, 0.95))
    if harm_gp_db is not None:
        kw["harm_gp_db"] = tuple(harm_gp_db)
    if floor_rel_db is not None:
        kw["floor_rel_db"] = tuple(floor_rel_db)
    ranges = StochasticRanges(**{k: v for k, v in kw.items()
                                 if hasattr(StochasticRanges, k)
                                 or k in StochasticRanges.__dataclass_fields__})
    params = sample_params(rng, ranges, n_rotors=n_rotors, n_harmonics=n_harmonics,
                           sample_rate=sr)
    audio, _ = synthesize(params, tracks, rng=rng, n_mics=1, line_mode=line_mode,
                          normalize_rms=0.1)
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)[:n_t]
    return audio, rps_hop, ft
