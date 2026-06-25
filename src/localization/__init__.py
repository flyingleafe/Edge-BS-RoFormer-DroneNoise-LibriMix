"""
Acoustic source localization for drone rotors.

Given an N-channel recording of drone noise and the (approximate) microphone
array geometry, estimate the 3-D position of each rotor relative to the array.

The core engine is a **near-field** SRP-PHAT over a 3-D position grid
(:mod:`.srp_phat`) — far-field DOA is invalid here because the array aperture is
comparable to the source range.  :func:`localize_rotors` wraps it with two
modes: audio-only top-K peak picking, and RPS-aided per-rotor harmonic isolation
(uses rotor-speed telemetry to separate the coherent rotor sources).
"""

from .rotor_localization import (
    RotorLocalizationResult,
    default_search_bounds,
    harmonic_mask,
    localize_rotors,
    match_and_score,
)
from .srp_phat import (
    SPEED_OF_SOUND,
    Grid,
    extract_peaks,
    make_grid,
    phat_cross_spectrum,
    refine_peak,
    srp_power,
)

__all__ = [
    "localize_rotors",
    "RotorLocalizationResult",
    "match_and_score",
    "default_search_bounds",
    "harmonic_mask",
    "phat_cross_spectrum",
    "srp_power",
    "make_grid",
    "extract_peaks",
    "refine_peak",
    "Grid",
    "SPEED_OF_SOUND",
]
