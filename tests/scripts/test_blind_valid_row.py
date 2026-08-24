"""The stitch, the zero convention and the acceptance gates of ``blind_valid_row``.

The annotation itself is the corpus campaign's worker and is not re-tested here.
What IS this driver's own is the seam between a per-window annotation and a
per-clip score: the midpoint cut, the rule that a refused or uncovered span
decodes to 0 rev/s, and the arm-dependent gates.
"""

from __future__ import annotations

import numpy as np
import pytest

blind_valid_row = pytest.importorskip("blind_valid_row")


def _window(uid: str, t0: float, rates: list[float], *, dur: float = 20.0, failed=None) -> dict:
    """One annotated window with a CONSTANT trajectory per rotor."""
    ft = np.arange(0.0, dur, 0.032)
    return {
        "uid": uid,
        "recording_id": "rec",
        "t0_s": t0,
        "dur_s": dur,
        "ft": ft,
        "rps": np.tile(np.asarray(rates, dtype=np.float64)[:, None], (1, ft.size)),
        "failed_gates": list(failed or []),
    }


def test_parent_recording_strips_the_rig_prefix() -> None:
    assert blind_valid_row.parent_recording("michaels_FLY124") == "FLY124"
    assert blind_valid_row.parent_recording("free-flight_nosource_room1") == (
        "free-flight_nosource_room1"
    )


def test_gates_accept_a_good_window_and_name_what_fails() -> None:
    good = {"fvk_ratio_double": 1.41, "spread_rev_s": 10.3, "pr_margin_half_min_db": -0.25}
    gates = {"g1", "g5", "pr"}
    assert blind_valid_row._gate_window(good, gates) == []

    assert blind_valid_row._gate_window({**good, "fvk_ratio_double": 1.05}, gates) == ["g1"]
    assert blind_valid_row._gate_window({**good, "spread_rev_s": 20.0}, gates) == ["g5"]
    assert blind_valid_row._gate_window({**good, "pr_margin_half_min_db": -3.1}, gates) == ["pr"]
    # A missing reading is not a pass: the instrument did not run.
    assert blind_valid_row._gate_window({"spread_rev_s": 1.0}, {"g1"}) == ["g1"]
    # No gate selected accepts everything.
    assert blind_valid_row._gate_window({}, set()) == []


def test_predict_cuts_at_the_midpoint_and_zeros_what_is_not_covered() -> None:
    a = _window("a", 0.0, [70.0, 71.0, 80.0, 81.0])
    b = _window("b", 16.0, [40.0, 41.0, 50.0, 51.0])
    # Centers are 10 s and 26 s, so the cut inside the 16-20 s overlap is 18 s.
    t = np.array([5.0, 17.0, 19.0, 30.0, 100.0])
    out = blind_valid_row.predict([a, b], t, gated=True)

    assert out.shape == (4, 5)
    np.testing.assert_allclose(out[:, 0], [70.0, 71.0, 80.0, 81.0])
    np.testing.assert_allclose(out[:, 1], [70.0, 71.0, 80.0, 81.0])
    np.testing.assert_allclose(out[:, 2], [40.0, 41.0, 50.0, 51.0])
    np.testing.assert_allclose(out[:, 3], [40.0, 41.0, 50.0, 51.0])
    # 100 s is past every window.
    np.testing.assert_allclose(out[:, 4], 0.0)


def test_a_refused_window_decodes_to_zero_only_under_the_gates() -> None:
    a = _window("a", 0.0, [70.0, 71.0, 80.0, 81.0], failed=["g1"])
    t = np.array([5.0])
    np.testing.assert_allclose(blind_valid_row.predict([a], t, gated=True)[:, 0], 0.0)
    np.testing.assert_allclose(
        blind_valid_row.predict([a], t, gated=False)[:, 0], [70.0, 71.0, 80.0, 81.0]
    )
    # No annotation at all is the same statement.
    np.testing.assert_allclose(blind_valid_row.predict([], t, gated=True), 0.0)


def test_g4_marks_both_members_of_a_disagreeing_pair() -> None:
    a = _window("a", 0.0, [75.0, 75.0, 85.0, 85.0])
    b = _window("b", 16.0, [75.1, 75.1, 85.1, 85.1])
    c = _window("c", 32.0, [61.0, 61.0, 73.0, 73.0])
    blind_valid_row._apply_g4([a, b, c])

    assert a["failed_gates"] == []
    assert b["failed_gates"] == ["g4"]
    assert c["failed_gates"] == ["g4"]
    assert b["g4_step_rev_s"] > blind_valid_row.G4_STEP_MAX_REV_S


def test_g4_reads_a_row_permutation_as_agreement() -> None:
    """A window names its rotors in its own order, so the step is PIT-aligned."""
    a = _window("a", 0.0, [75.0, 85.0, 75.0, 85.0])
    b = _window("b", 16.0, [85.0, 75.0, 85.0, 75.0])
    blind_valid_row._apply_g4([a, b])
    assert a["failed_gates"] == []
    assert b["failed_gates"] == []


def test_aggregate_pools_are_exact_sums_not_means_of_means() -> None:
    per_clip = [
        {"flight": {"sum_abs": 10.0, "sum_sq": 50.0, "n": 10}},
        {
            "flight": {"sum_abs": 2.0, "sum_sq": 2.0, "n": 2},
            "zero": {"sum_abs": 0.0, "sum_sq": 0.0, "n": 4},
        },
    ]
    got = blind_valid_row._aggregate(per_clip)
    assert got["flight"]["n"] == 12
    assert got["flight"]["mae"] == pytest.approx(12.0 / 12.0)
    assert got["flight"]["rmse"] == pytest.approx(np.sqrt(52.0 / 12.0))
    assert got["zero"]["mae"] == pytest.approx(0.0)
    assert got["all"]["n"] == 16
    assert got["all"]["mae"] == pytest.approx(12.0 / 16.0)
    # The eight microphones of a clip share one prediction, so the pooled
    # numbers are unchanged and only the equivalent count differs.
    assert got["all"]["n_channel_equivalent"] == 16 * 8
