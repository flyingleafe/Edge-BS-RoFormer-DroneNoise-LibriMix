"""Time-domain float comparisons.

Time values are float64 seconds. Audio sample rates push us toward
sub-microsecond resolution; Unix timestamps push us toward absolute magnitudes
~1.7e9, where ulp(t) ≈ 4e-7 s. We use a `numpy.isclose`-style criterion that
behaves correctly across both regimes.

The default tolerances are tight: `atol=1e-9` and `rtol=1e-12`. For uniform
series we tighten the *grid-alignment* check by deriving the tolerance from
the sample period (`1/sr`).
"""
from __future__ import annotations

# Absolute tolerance for time equality in seconds.
DEFAULT_ATOL: float = 1e-9
# Relative tolerance (scales with |t|).
DEFAULT_RTOL: float = 1e-12


# Float64 machine epsilon — used as an ulp-scaling factor for time magnitudes.
_FLOAT64_EPS: float = 2.220446049250313e-16


def tclose(a: float, b: float, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    """True iff `a` and `b` are within (atol + rtol*max(|a|,|b|))."""
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def t_atol_at(t_ref: float, base_atol: float = DEFAULT_ATOL) -> float:
    """Time-equality absolute tolerance scaled by the ulp at |t_ref|.

    Returns `base_atol` plus a few ulps of `|t_ref|`, which is the right scale
    for differences computed by additions/subtractions involving floats of
    magnitude `|t_ref|` (e.g. Unix timestamps).
    """
    return base_atol + 8.0 * abs(t_ref) * _FLOAT64_EPS


def grid_atol(sr: float, t_ref: float = 0.0, frac: float = 1e-3) -> float:
    """Absolute tolerance for sample-grid alignment at rate `sr` near time `t_ref`.

    Combines a fraction of one sample period (to absorb rounding through
    sample-rate arithmetic) with an ulp-scaled term (to absorb absolute
    timestamp magnitude). The latter dominates for Unix-magnitude anchors.
    """
    return frac / sr + t_atol_at(t_ref, base_atol=0.0)
