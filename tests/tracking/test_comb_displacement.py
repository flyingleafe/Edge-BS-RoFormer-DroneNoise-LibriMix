"""Tests for the interferer geometry of ``tracking.comb_displacement``.

``nearest_interloper_hz`` is the quantitative sibling of
``carrier_collision_mask``, so the tests pin the hand-computable distances and
the consistency of the two against each other.

Run:  pytest tests/tracking/test_comb_displacement.py
"""

import numpy as np

from tracking.comb_displacement import (
    DisplacementConfig,
    carrier_collision_mask,
    nearest_interloper_hz,
)

#: Four constant rotors: the carrier (80), a near twin (75), a slow one (60)
#: and a silent one (3 rev/s, below ``min_rate``).
RATES = np.array([[80.0, 80.0], [75.0, 75.0], [60.0, 60.0], [3.0, 3.0]])


def test_nearest_matches_hand_computed_distances() -> None:
    d = nearest_interloper_hz(RATES, RATES[0], 0, [1, 7, 70], f_max=6000.0)
    # k=1: 80 Hz; rotor 1's fundamental at 75 -> 5 Hz.
    # k=7: 560 Hz; rotor 2's 9th at 540 -> 20 Hz (rotor 1's 525/600 are farther).
    # k=70: 5600 Hz; rotor 2's 93rd at 5580 -> 20 Hz.
    assert np.allclose(d, [[5.0, 5.0], [20.0, 20.0], [20.0, 20.0]])


def test_own_rotor_and_silent_rotors_are_not_interferers() -> None:
    # Only the silent rotor is left: no interferer anywhere.
    rates = np.array([[80.0], [3.0]])
    d = nearest_interloper_hz(rates, rates[0], 0, [1, 2, 3], f_max=6000.0)
    assert np.all(np.isinf(d))


def test_f_max_excludes_high_interferers() -> None:
    ks = [70]
    near = nearest_interloper_hz(RATES, RATES[0], 0, ks, f_max=6000.0)
    far = nearest_interloper_hz(RATES, RATES[0], 0, ks, f_max=1000.0)
    assert float(near[0, 0]) == 20.0
    assert np.isinf(far[0, 0])  # every line near 5600 Hz is above the cut


def test_agrees_with_carrier_collision_mask() -> None:
    cfg = DisplacementConfig(f_max=6000.0)
    ks = list(range(1, 30))
    d = nearest_interloper_hz(RATES, RATES[0], 0, ks, f_max=cfg.f_max)
    mask = carrier_collision_mask(RATES, RATES[0], 0, ks, cfg=cfg)
    sep = np.array([cfg.collision_guard * cfg.search_hz(k) for k in ks])[:, None]
    assert np.array_equal(mask, d < sep)


def test_half_integer_carrier_shifts_the_geometry() -> None:
    integer = nearest_interloper_hz(RATES, RATES[0], 0, [10], f_max=6000.0)
    half = nearest_interloper_hz(RATES, RATES[0], 0, [10], half=True, f_max=6000.0)
    # k=10 -> 800 Hz, 20 Hz from rotor 2's 13th at 780. The half-integer carrier
    # sits at 840 Hz, which rotor 2's 14th line hits exactly.
    assert float(integer[0, 0]) == 20.0
    assert float(half[0, 0]) == 0.0


def test_offsets_are_the_same_geometry_signed(  # phase 6d
) -> None:
    """``interloper_offsets_hz`` must agree with the DISTANCE, and carry a sign.

    The ridge component excises these positions from its floor region instead of
    gating the cell out, so a sign error would excise the wrong side of the band
    and leave the interferer in the floor.
    """
    from tracking.comb_displacement import interloper_offsets_hz

    ks = [1, 7, 70]
    band = np.array([40.0, 40.0, 40.0])
    offs = interloper_offsets_hz(RATES[:, 0], 80.0, 0, ks, band_hz=band, f_max=6000.0)
    nearest = nearest_interloper_hz(RATES, RATES[0], 0, ks, f_max=6000.0)[:, 0]
    for i, o in enumerate(offs):
        assert o.size, f"k={ks[i]}: no interferer inside the band"
        assert np.isclose(np.min(np.abs(o)), nearest[i]), (
            f"k={ks[i]}: offsets {o} disagree with the distance {nearest[i]}"
        )
    # k=1: rotor 1's fundamental is BELOW the carrier's 80 Hz, so the offset is
    # negative; k=7: rotor 2's 9th at 540 is below 560; both signs must appear
    # somewhere in the set, or the excision is one-sided.
    assert float(offs[0].min()) < 0.0
    assert any(float(o.max()) > 0.0 for o in offs)


def test_offsets_respect_the_band_margin() -> None:
    from tracking.comb_displacement import interloper_offsets_hz

    wide = interloper_offsets_hz(RATES[:, 0], 80.0, 0, [7], band_hz=np.array([40.0]))
    narrow = interloper_offsets_hz(RATES[:, 0], 80.0, 0, [7], band_hz=np.array([2.0]))
    assert wide[0].size > narrow[0].size
    assert all(abs(v) <= 1.5 * 40.0 for v in wide[0])
