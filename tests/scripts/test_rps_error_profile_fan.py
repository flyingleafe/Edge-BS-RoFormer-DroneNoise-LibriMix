"""The fan statistic of ``scripts/rps_error_profile.py`` on a synthetic dump.

Two models see the same ground truth. The tracker predicts the label, so its
fan must equal the true fan in every bucket and its slope must be 1. The fixed
fan predicts the mean rotor speed plus a constant offset vector, so its fan
must be the same 6 rev/s in every bucket and its slope must be 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import rps_error_profile

# Four rotor speeds for each sample, constant over the time-frames. The three
# cruise samples fall into three different buckets of the true fan, and the
# last sample has every rotor stopped, so it has no cruise time-frame.
SAMPLES = [
    [70.0, 70.5, 71.0, 71.5],  # fan 1.5 -> bucket [0, 2)
    [65.0, 66.0, 67.0, 68.0],  # fan 3.0 -> bucket [2, 5)
    [60.0, 62.0, 64.0, 67.0],  # fan 7.0 -> bucket [5, 10)
    [0.0, 0.0, 0.0, 0.0],  # no cruise time-frame
]
OFFSETS = np.array([-3.0, -1.0, 1.0, 3.0])  # the fixed fan, 6 rev/s wide
T = 8
FAN_TRUE = [1.5, 3.0, 7.0, np.nan, np.nan]


def make_dump(root: Path) -> Path:
    """Write one set directory with a ground truth and two predictions."""
    d = root / "dump" / "s1"
    d.mkdir(parents=True)
    gt = np.stack([np.repeat(np.array(s)[:, None], T, axis=1) for s in SAMPLES])
    n_t = np.full(len(SAMPLES), T, dtype=np.int64)
    np.savez(d / "_gt.npz", rps=gt.astype(np.float32), n_t=n_t)
    (d / "_meta.json").write_text(
        json.dumps([{"recording_id": f"r{i}", "channel": 0} for i in range(len(SAMPLES))])
    )
    # the tracker: the label itself
    np.savez(
        d / "tracker.npz",
        pred=gt.astype(np.float32),
        n_t=n_t,
        metric=np.zeros(len(SAMPLES), dtype=np.float32),
    )
    # the fixed fan: the mean rotor speed of each time-frame, plus the offsets
    mean = gt.mean(axis=1, keepdims=True)
    fixed = mean + OFFSETS[None, :, None]
    np.savez(
        d / "fixedfan.npz",
        pred=fixed.astype(np.float32),
        n_t=n_t,
        metric=np.zeros(len(SAMPLES), dtype=np.float32),
    )
    return d.parent


@pytest.fixture
def profiled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the CLI over the synthetic dump and give back the output directory."""
    dump = make_dump(tmp_path)
    out = tmp_path / "profile"
    monkeypatch.setattr(
        sys,
        "argv",
        ["rps_error_profile.py", "--dump", str(dump), "--out", str(out)],
    )
    rps_error_profile.main()
    return out


def rows(df: pd.DataFrame, exp: str, by: str) -> pd.DataFrame:
    """The rows of one experiment, in a stable order."""
    return cast(pd.DataFrame, df[df["exp"] == exp]).sort_values(by=by)


def test_fan_buckets_follow_the_true_spread(profiled: Path) -> None:
    fan = pd.read_csv(profiled / "fan.csv")
    assert list(fan.columns) == [
        "exp",
        "set",
        "arch",
        "train",
        "speech",
        "objective",
        "spread_lo",
        "spread_hi",
        "n_frames",
        "fan_true",
        "fan_pred",
    ]
    for exp in ("tracker", "fixedfan"):
        g = rows(fan, exp, "spread_lo")
        assert list(g["spread_lo"]) == [0.0, 2.0, 5.0, 10.0, 20.0]
        assert list(g["n_frames"]) == [T, T, T, 0, 0]
        # the true fan does not depend on the model
        for got, want in zip(g["fan_true"], FAN_TRUE, strict=True):
            assert (np.isnan(got) and np.isnan(want)) or got == pytest.approx(want)

    tracker = rows(fan, "tracker", "spread_lo")
    # a model that tracks four lines follows the true fan
    for got, want in zip(tracker["fan_pred"], FAN_TRUE, strict=True):
        assert (np.isnan(got) and np.isnan(want)) or got == pytest.approx(want)

    fixed = rows(fan, "fixedfan", "spread_lo")
    # a fixed fan keeps the same width in every bucket
    for got, n in zip(fixed["fan_pred"], fixed["n_frames"], strict=True):
        assert np.isnan(got) if n == 0 else got == pytest.approx(6.0)


def test_summary_slope_separates_the_two_models(profiled: Path) -> None:
    s = pd.read_csv(profiled / "summary.csv").set_index("exp")
    all_true = np.mean([1.5, 3.0, 7.0])  # the buckets hold the same count
    assert s.loc["tracker", "fan_true"] == pytest.approx(all_true)
    assert s.loc["tracker", "fan_pred"] == pytest.approx(all_true)
    assert s.loc["tracker", "fan_slope"] == pytest.approx(1.0)
    assert s.loc["fixedfan", "fan_true"] == pytest.approx(all_true)
    assert s.loc["fixedfan", "fan_pred"] == pytest.approx(6.0)
    assert s.loc["fixedfan", "fan_slope"] == pytest.approx(0.0, abs=1e-12)


def test_frames_carry_the_per_sample_fan(profiled: Path) -> None:
    fr = pd.read_csv(profiled / "frames.csv")
    g = rows(fr, "fixedfan", "frame")
    assert list(g["fan_n"]) == [T, T, T, 0]
    assert list(g["fan_true"])[:3] == pytest.approx([1.5, 3.0, 7.0])
    assert list(g["fan_pred"])[:3] == pytest.approx([6.0, 6.0, 6.0])
    # the stopped sample has no cruise time-frame
    assert np.isnan(g["fan_true"].iloc[3])
    assert np.isnan(g["fan_pred"].iloc[3])
    # the columns the CLI already had stay in place
    assert {"mae", "cls", "gt_mean", "metric_monitored"} <= set(fr.columns)
