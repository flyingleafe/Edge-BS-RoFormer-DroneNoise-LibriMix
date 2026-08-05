"""Frame conventions: which published entry carries the rotor speed.

The preference order is a *data* decision with two consumers — the noise-pool
adapter (:func:`data_processing.frames.adapt_recording_frame`) and the plotting
alias table (:data:`data_processing.canonical.ENTRY_ALIASES`) — so it is pinned
here once.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from data_processing.canonical import ENTRY_ALIASES
from data_processing.frames import PUBLISHED_RPS_KEYS, adapt_recording_frame

SR = 16000
N_MOTOR = 32


def _audio() -> td.Series:
    rng = np.random.default_rng(0)
    return td.uniform(
        (0.05 * rng.standard_normal((2, SR))).astype(np.float32),
        SR,
        dims=("mic", "time"),
        t_start=0.0,
    )


def _motors(base: float) -> td.Series:
    values = np.tile(np.arange(4, dtype=np.float32)[:, None] + base, (1, N_MOTOR))
    return td.events(
        np.linspace(0.0, 0.99, N_MOTOR), values, dims=("rotor", "time"), t_start=0.0, t_end=1.0
    )


def test_published_rps_key_preference_order():
    """``rps`` > ``motors_measured`` > ``motors_command``.

    ``motors_measured`` is the real tachometer and the track the beat-VK
    protocol pins as ground truth; ``motors_command`` is only what the flight
    controller asked for and is the fallback for the recordings (most of
    DREGON) that log no measured track.
    """
    assert PUBLISHED_RPS_KEYS == ("rps", "motors_measured", "motors_command")
    # The plotting aliases are the same order minus the canonical name itself,
    # so a frame plots under the track the pipeline reads.
    assert ENTRY_ALIASES["rps"][:2] == ("motors_measured", "motors_command")


@pytest.mark.parametrize(
    ("entries", "expected_base"),
    [
        ({"rps": 70.0, "motors_measured": 55.0, "motors_command": 60.0}, 70.0),
        ({"motors_measured": 55.0, "motors_command": 60.0}, 55.0),
        ({"motors_command": 60.0}, 60.0),
    ],
)
def test_adapt_recording_frame_picks_the_preferred_track(entries, expected_base):
    frame = td.Frame({"audio": _audio(), **{k: _motors(v) for k, v in entries.items()}})

    adapted = adapt_recording_frame(frame, sample_rate=SR)

    assert adapted is not None
    assert set(adapted.keys()) == {"audio", "rps"}
    np.testing.assert_array_equal(
        np.asarray(adapted["rps"].data), np.asarray(_motors(expected_base).data)
    )


def test_adapt_recording_frame_returns_none_without_a_rotor_track():
    assert adapt_recording_frame(td.Frame({"audio": _audio()}), sample_rate=SR) is None
