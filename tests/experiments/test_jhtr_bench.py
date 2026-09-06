"""Behavioral risks in JHTR scientific gates; no network, checkpoint or renderer."""

import numpy as np
import pytest

from experiments.refiner_bench import (
    diagnostic_guesses,
    paired_group_bootstrap,
    precision_gates,
    recovery_intervals,
    trajectory_errors,
)


def _truth(n=2, t=64):
    return np.broadcast_to(np.array([40.0, 60.0, 80.0, 100.0])[None, :, None], (n, 4, t)).copy()


def test_microphones_and_speech_copies_do_not_create_independent_confidence_samples():
    reference = np.array([1.0, 4.0, 2.0])
    candidate = np.array([2.0, 1.0, 1.0])
    groups = np.array(["flight-a", "flight-b", "flight-c"])
    original = paired_group_bootstrap(reference, candidate, groups)
    copied = paired_group_bootstrap(
        np.repeat(reference, 16), np.repeat(candidate, 16), np.repeat(groups, 16)
    )
    assert original["n_groups"] == copied["n_groups"] == 3
    assert copied["improvement"] == pytest.approx(original["improvement"])
    assert copied["ci95"] == pytest.approx(original["ci95"])
    assert original["ci95"][0] < 0 < original["ci95"][1]


def test_unequal_groups_preserve_benchmark_sample_weighting():
    result = paired_group_bootstrap(
        np.array([10.0, 0.0, 0.0, 0.0]), np.zeros(4), np.array(["a", "b", "b", "b"])
    )
    assert result["improvement"] == 2.5  # not the equal-flight mean of 5
    assert result["ci95"] == pytest.approx([0.0, 10.0])
    one_flight = paired_group_bootstrap(np.ones(32), np.zeros(32), np.repeat("a", 32))
    assert one_flight["status"] == "unestablished"
    assert one_flight["ci95"] is None


def test_conditional_order_errors_are_not_forgiven_by_pit():
    truth = _truth()
    errors = trajectory_errors(truth[:, [1, 0, 2, 3]], truth)
    np.testing.assert_array_equal(errors["ordered_mae"], [10.0, 10.0])
    np.testing.assert_array_equal(errors["ordered_mse"], [200.0, 200.0])
    np.testing.assert_array_equal(errors["pit_mae_mae_assignment"], 0.0)
    np.testing.assert_array_equal(errors["pit_mae_mse_assignment"], 0.0)
    np.testing.assert_array_equal(errors["pit_mse"], 0.0)


def test_identity_is_not_a_precision_refiner_and_oracle_drift_cannot_hide():
    truth = _truth()
    observable = np.ones_like(truth, bool)
    oracle = np.repeat(truth[:, None], 7, axis=1)
    guesses = diagnostic_guesses(truth)
    identity_offsets = {
        name: (init, init) for name, (init, _) in guesses.items() if name.startswith("offset")
    }
    identity = precision_gates(oracle, identity_offsets, truth, observable)
    assert identity["preservation_pass"]
    assert not identity["correction_pass"]
    assert not identity["precision_pass"]
    corrected = {
        name: (init, truth + 0.5 * (init - truth))
        for name, (init, _) in guesses.items()
        if name.startswith("offset")
    }
    assert precision_gates(oracle, corrected, truth, observable)["precision_pass"]
    oracle[:, 3] += 0.3  # an intermediate walkout must fail even if final is exact
    result = precision_gates(oracle, corrected, truth, observable)
    assert result["correction_pass"]
    assert not result["preservation_pass"]
    assert not result["precision_pass"]


def test_missing_observability_or_signed_offset_coverage_cannot_pass_claim():
    truth = _truth()
    oracle = np.repeat(truth[:, None], 7, axis=1)
    assert precision_gates(oracle, {}, truth, None)["status"] == "unestablished"
    partial = {"offset+0.5": (truth + 0.5, truth)}
    result = precision_gates(oracle, partial, truth, np.ones_like(truth, bool))
    assert not result["precision_pass"]


def test_all_four_recovery_requires_physical_half_second_and_one_crop_assignment():
    truth = _truth(n=1)
    too_short = truth + 2.0
    too_short[:, :, 10:26] = truth[:, :, 10:26]  # 16 points span .480 seconds
    assert not recovery_intervals(too_short, truth)["success"][0]
    long_enough = too_short.copy()
    long_enough[:, :, 26] = truth[:, :, 26]
    result = recovery_intervals(long_enough[:, [1, 0, 3, 2]], truth)
    assert result["success"][0]
    assert result["first_interval_start_s"][0] == pytest.approx(0.320)
    assert result["longest_interval_s"][0] == pytest.approx(0.512)
    # Per-time PIT would make this perfect; no fixed whole-crop permutation can.
    changing_identity = truth.copy()
    changing_identity[:, :, ::2] = truth[:, [1, 0, 3, 2], ::2]
    assert not recovery_intervals(changing_identity, truth)["success"][0]


def test_capture_diagnostics_do_not_synthesize_missing_coverage():
    truth = np.zeros((2, 4, 32))
    truth[1] = 100.0
    original = truth.copy()
    guesses = diagnostic_guesses(truth)
    np.testing.assert_array_equal(truth, original)
    np.testing.assert_array_equal(guesses["double"][1], [False, False])
    np.testing.assert_array_equal(guesses["false_active"][1], [True, False])
    np.testing.assert_array_equal(guesses["missing_active"][1], [False, True])
    np.testing.assert_array_equal(guesses["oracle"][0], truth)
