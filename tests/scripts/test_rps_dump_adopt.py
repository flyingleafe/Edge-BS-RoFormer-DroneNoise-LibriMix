"""The dump seam of ``scripts/rps_dump_adopt``: order, grid and metric.

The estimators themselves are the source drivers' and are not re-tested here.
What IS this CLI's own is the conversion: read the sample order out of a dump's
``_meta.json``, put each unit's prediction on the sample's own frame grid, give
one 8-channel unit to every microphone of its clip, and score with the same
statistic ``rps_dump`` stores. Each of those is checked against a direct
computation on a tiny synthetic dump — two clips, two microphones, and a stored
grid that is not the target grid.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

rps_dump_adopt = pytest.importorskip("rps_dump_adopt")

N_ROTORS = 4
#: clip 0 is 11 frames long, clip 1 is 7 — so the dump is NaN-padded to 11
FRAMES = (11, 7)
#: every source unit arrives on this grid, which matches neither clip
SOURCE_FRAMES = 5


def _gt_clip(clip: int, frames: int) -> np.ndarray:
    """A smooth, distinct ``(4, frames)`` label per clip."""
    t = np.linspace(0.0, 1.0, frames)
    return np.stack([40.0 + 10.0 * k + 20.0 * clip + 5.0 * t * (k + 1) for k in range(N_ROTORS)])


def _write_dump(tmp_path: Path) -> Path:
    """A two-clip, two-microphone dump directory: ``_gt.npz`` plus ``_meta.json``."""
    dump = tmp_path / "tiny"
    dump.mkdir()
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    width = max(FRAMES)
    gt = np.full((len(order), N_ROTORS, width), np.nan, dtype=np.float32)
    n_t = np.array([FRAMES[c] for c, _ in order], dtype=np.int64)
    for i, (clip, _channel) in enumerate(order):
        gt[i, :, : FRAMES[clip]] = _gt_clip(clip, FRAMES[clip])
    np.savez(dump / "_gt.npz", rps=gt, n_t=n_t)
    (dump / "_meta.json").write_text(
        json.dumps(
            [{"recording_id": f"sample_{clip:05d}", "channel": ch} for clip, ch in order],
        )
    )
    return dump


def _source_units() -> dict[tuple[int, int | None], np.ndarray]:
    """One distinct ``(4, 5)`` prediction per (clip, microphone)."""
    return {
        (clip, ch): np.stack(
            [
                np.linspace(30.0 + 7.0 * k + 3.0 * clip + 1.0 * ch, 90.0, SOURCE_FRAMES)
                for k in range(N_ROTORS)
            ]
        )
        for clip in (0, 1)
        for ch in (0, 1)
    }


def _direct_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    """PIT MAE by brute force: the MSE-optimal assignment, then the MAE on it.

    This is ``metrics.rps.rps_mae_frame`` written out — one Hungarian match per
    sample on the pairwise MSE, scored with the absolute error — and it is the
    statistic ``scripts/rps_dump.py`` stores for a model that emits ``rps_pred``.
    """
    mse = ((pred[:, None] - gt[None, :]) ** 2).mean(-1)
    best = min(permutations(range(N_ROTORS)), key=lambda p: sum(mse[k, p[k]] for k in range(4)))
    return float(np.mean([np.abs(pred[k] - gt[best[k]]).mean() for k in range(N_ROTORS)]))


def _direct_resample(src: np.ndarray, n: int) -> np.ndarray:
    """Linear resampling at ``align_corners=False`` sample positions, written out."""
    tg = src.shape[-1]
    pos = np.clip((np.arange(n) + 0.5) * tg / n - 0.5, 0, tg - 1)
    return np.stack([np.interp(pos, np.arange(tg), row) for row in src])


def test_read_dump_gives_the_sample_order_and_the_clip_ids(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path)
    gt, n_t, order, clip_ids = rps_dump_adopt.read_dump(dump)

    assert order == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert clip_ids == ["sample_00000", "sample_00001"]
    assert list(n_t) == [11, 11, 7, 7]
    assert gt.shape == (4, N_ROTORS, 11)


def test_assemble_puts_every_unit_at_its_own_sample(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path)
    gt, n_t, order, _clip_ids = rps_dump_adopt.read_dump(dump)
    units = _source_units()

    pred, lengths, metric, resampled = rps_dump_adopt.assemble(units, gt, n_t, order)

    assert resampled is True
    assert pred.shape == (4, N_ROTORS, 11)
    assert list(lengths) == [11, 11, 7, 7]
    for i, (clip, channel) in enumerate(order):
        want = _direct_resample(units[(clip, channel)], FRAMES[clip])
        np.testing.assert_allclose(pred[i, :, : FRAMES[clip]], want.astype(np.float32), rtol=1e-6)
    # The short clip keeps its tail NaN-padded, as the dump format asks.
    assert np.isnan(pred[2, :, 7:]).all()
    assert np.isfinite(pred[0]).all()


def test_assemble_scores_with_the_dump_metric(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path)
    gt, n_t, order, _clip_ids = rps_dump_adopt.read_dump(dump)
    units = _source_units()

    _pred, _lengths, metric, _resampled = rps_dump_adopt.assemble(units, gt, n_t, order)

    for i, (clip, channel) in enumerate(order):
        want = _direct_metric(
            _direct_resample(units[(clip, channel)], FRAMES[clip]),
            _gt_clip(clip, FRAMES[clip]).astype(np.float64),
        )
        # The dump keeps its label in float32, thus the tolerance is float32's.
        assert metric[i] == pytest.approx(want, rel=1e-6)
    # Four samples, four distinct predictions, thus four distinct scores.
    assert len(set(np.round(metric, 9))) == 4


def test_a_permuted_prediction_scores_as_the_label_itself(tmp_path: Path) -> None:
    """The metric is permutation invariant, which is what PIT means here."""
    dump = _write_dump(tmp_path)
    gt, n_t, order, _clip_ids = rps_dump_adopt.read_dump(dump)
    units = {
        (clip, ch): _gt_clip(clip, FRAMES[clip])[[2, 0, 3, 1]] for clip in (0, 1) for ch in (0, 1)
    }

    _pred, _lengths, metric, resampled = rps_dump_adopt.assemble(units, gt, n_t, order)

    assert resampled is False  # already on the dump grid
    np.testing.assert_allclose(metric, 0.0, atol=1e-5)  # float32 label round trip


def test_an_8_channel_unit_reaches_every_microphone_of_its_clip(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path)
    gt, n_t, order, _clip_ids = rps_dump_adopt.read_dump(dump)
    units: dict[tuple[int, int | None], np.ndarray] = {
        (clip, None): _source_units()[(clip, 0)] for clip in (0, 1)
    }

    pred, _lengths, metric, _resampled = rps_dump_adopt.assemble(units, gt, n_t, order)

    np.testing.assert_allclose(pred[0], pred[1], equal_nan=True)
    np.testing.assert_allclose(pred[2], pred[3], equal_nan=True)
    assert metric[0] == pytest.approx(metric[1])
    assert metric[2] == pytest.approx(metric[3])
    # A per-channel unit still wins over the 8-channel fallback.
    units[(1, 1)] = _source_units()[(1, 1)]
    pred2, _l2, _m2, _r2 = rps_dump_adopt.assemble(units, gt, n_t, order)
    assert not np.allclose(pred2[2], pred2[3], equal_nan=True)


def test_a_missing_unit_is_an_error_not_a_zero(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path)
    gt, n_t, order, _clip_ids = rps_dump_adopt.read_dump(dump)
    units = _source_units()
    del units[(1, 1)]

    with pytest.raises(SystemExit, match="no prediction for clip 1 channel 1"):
        rps_dump_adopt.assemble(units, gt, n_t, order)


def test_a_unit_json_without_a_prediction_is_not_adoptable(tmp_path: Path) -> None:
    """The layout the classical and OT drivers really stored: sums, no track."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "nmf__clip000_ch0.json").write_text(
        json.dumps({"method": "nmf", "clip": 0, "channel": 0, "pools": {"zero": {"n": 4}}})
    )

    with pytest.raises(rps_dump_adopt.NotAdoptable, match="no per-frame prediction"):
        rps_dump_adopt.read_pred_units(raw)

    (raw / "nmf__clip000_ch0.json").write_text(
        json.dumps({"clip": 0, "channel": 0, "pred": [[1.0, 2.0]] * N_ROTORS})
    )
    units = rps_dump_adopt.read_pred_units(raw)
    np.testing.assert_allclose(units[(0, 0)], np.array([[1.0, 2.0]] * N_ROTORS))


def test_clip_frames_reads_the_length_of_each_clip(tmp_path: Path) -> None:
    dump = _write_dump(tmp_path)
    _gt, n_t, order, clip_ids = rps_dump_adopt.read_dump(dump)
    assert rps_dump_adopt.clip_frames(order, n_t, len(clip_ids)) == [11, 7]
