"""Quadrotor control-allocation constants shared by the tracking stack.

A quadrotor controls four degrees of freedom — *collective thrust*, *roll*,
*pitch*, *yaw* — through four motors related by the fixed linear mixer
:data:`MIXER` (``w = B @ m``).  The constants and the two projection helpers
here are the single source of truth for that mixer math.  They started life in
``data_processing.rps_synthesis`` (which re-exports them for backward
compatibility) and moved here so that ``tracking`` never imports
``data_processing``.
"""

from __future__ import annotations

import numpy as np

NUM_ROTORS = 4

# Quadrotor control-allocation mixer.  Columns = [common, roll, pitch, yaw];
# rows = rotors in the order [RFront, LFront, LBack, RBack] (matches
# ``data_processing.sources.michaels.ROTOR_ORDER``).  Entries are +/-1, so the columns
# are mutually orthogonal with squared norm 4 -> B^T B = 4 I and B^-1 = B^T / 4.
MIXER = np.array(
    [
        [1.0, +1.0, +1.0, +1.0],  # RFront
        [1.0, -1.0, +1.0, -1.0],  # LFront
        [1.0, -1.0, -1.0, +1.0],  # LBack
        [1.0, +1.0, -1.0, -1.0],  # RBack
    ]
)

MODE_NAMES = ("common", "roll", "pitch", "yaw")


def modes_from_rps(w: np.ndarray) -> np.ndarray:
    """Project rotor speeds onto control modes: ``m = B^T w / 4``.

    Args:
        w: ``(4, M)`` rotor speeds (rev/s).

    Returns:
        ``(4, M)`` mode coefficients in the order :data:`MODE_NAMES`.
    """
    return (MIXER.T @ w) / NUM_ROTORS


def rps_from_modes(m: np.ndarray) -> np.ndarray:
    """Recover rotor speeds from control modes: ``w = B m``."""
    return MIXER @ m
