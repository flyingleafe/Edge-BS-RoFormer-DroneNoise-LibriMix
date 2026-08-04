"""Unit tests for the importable core of ``scripts/se_eval.py``.

No GPU, no network, no dataset downloads: metrics on synthetic signals,
valid-spec parsing, dataset-name → loader resolution (stubbed), grouping.
"""

from __future__ import annotations

import numpy as np
import pytest
import se_eval


# ── compute_metrics ─────────────────────────────────────────────────────────
def test_metrics_identity_estimate() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(4000).astype(np.float32)
    m = se_eval.compute_metrics(ref, ref)
    assert set(m) == set(se_eval.METRICS)
    assert m["si_sdr"] > 40.0
    assert m["gain_db"] == pytest.approx(0.0, abs=1e-4)
    assert m["corr"] == pytest.approx(1.0, abs=1e-5)


def test_metrics_scaled_estimate_energy_vs_correlation() -> None:
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(4000).astype(np.float32)
    m = se_eval.compute_metrics(ref, 0.5 * ref)
    # half-amplitude: energy is down ~6 dB but the shape is intact
    assert m["gain_db"] == pytest.approx(-6.02, abs=0.05)
    assert m["corr"] == pytest.approx(1.0, abs=1e-5)
    # SI-SDR is scale-invariant, so it stays huge
    assert m["si_sdr"] > 40.0


def test_metrics_truncates_to_common_length_and_guards() -> None:
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(1000).astype(np.float32)
    est = np.concatenate([ref, np.zeros(500, np.float32)])
    m = se_eval.compute_metrics(ref, est)
    assert m["si_sdr"] > 40.0  # extra tail is cut, not scored
    # PESQ needs longer clips at 16 kHz — must degrade to NaN, not raise
    assert m["pesq"] != m["pesq"] or isinstance(m["pesq"], float)


# ── valid-spec parsing + loader resolution ──────────────────────────────────
def test_parse_valid_forms() -> None:
    assert se_eval.parse_valid("SE-valid-drone") == ("SE-valid-drone", None, None)
    assert se_eval.parse_valid("SE-valid-harmonic@abc123") == ("SE-valid-harmonic", "abc123", None)
    assert se_eval.parse_valid("SE-valid-harmonic#motors") == ("SE-valid-harmonic", None, "motors")
    assert se_eval.parse_valid("X@v1#drone") == ("X", "v1", "drone")


def test_parse_valid_rejects_empty() -> None:
    with pytest.raises(ValueError):
        se_eval.parse_valid("")
    with pytest.raises(ValueError):
        se_eval.parse_valid("@v1")


def test_load_valid_resolves_to_sevalid_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_processing.frame_datasets as fd

    calls: dict[str, object] = {}

    def stub(dataset: str, **kwargs: object) -> str:
        calls["dataset"] = dataset
        calls.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(fd, "SEValidFrameDataset", stub)
    assert se_eval.load_valid("SE-valid-drone@v9#drone", sample_rate=8000) == "sentinel"
    assert calls == {
        "dataset": "SE-valid-drone",
        "version": "v9",
        "category": "drone",
        "sample_rate": 8000,
    }


# ── grouping ────────────────────────────────────────────────────────────────
def test_group_rows_means_and_counts() -> None:
    rows = [
        {"method": "m", "valid": "v", "category": "drone", "input_snr": -15.0, "si_sdr": 1.0},
        {"method": "m", "valid": "v", "category": "drone", "input_snr": -15.0, "si_sdr": 3.0},
        {"method": "m", "valid": "v", "category": "motors", "input_snr": -15.0, "si_sdr": 7.0},
    ]
    out = se_eval.group_rows(rows, ["category", "input_snr"], metrics=["si_sdr"])
    assert len(out) == 2
    drone = next(r for r in out if r["category"] == "drone")
    assert drone["n"] == 2
    assert drone["si_sdr"] == pytest.approx(2.0)
    assert drone["method"] == "m"


def test_group_rows_nanmean_skips_missing() -> None:
    rows = [
        {"method": "m", "valid": "v", "input_snr": 0.0, "pesq": float("nan")},
        {"method": "m", "valid": "v", "input_snr": 0.0, "pesq": 2.0},
    ]
    out = se_eval.group_rows(rows, ["input_snr"], metrics=["pesq"])
    assert out[0]["pesq"] == pytest.approx(2.0)
