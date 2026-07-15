"""Tests for ``notebooks/geom_calibration.py``.

The pure-math tests (synthetic geometry recovery, Procrustes, permutation
detection) need no dataset and run fast. Two integration smoke tests exercise
the full DREGON / Michael's pipelines and are skipped automatically when the
(git-ignored) recordings are not present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import geom_calibration as gc  # noqa: E402


def _synthetic_records(
    mic: np.ndarray, rotor: np.ndarray, freqs: np.ndarray
) -> list[gc.RotorBandRTF]:
    """Exact free-field records (coherence 1) for a known geometry."""
    recs: list[gc.RotorBandRTF] = []
    for r in range(rotor.shape[0]):
        d = np.linalg.norm(mic - rotor[r][None, :], axis=1)
        ref = int(np.argmin(d))
        ph = -2.0 * np.pi * freqs[None, :] * (d[:, None] - d[ref]) / gc.SPEED_OF_SOUND
        mag = (d[ref] / d)[:, None] * np.ones_like(ph)
        recs.append(
            gc.RotorBandRTF(
                rotor=r, ref=ref, freqs=freqs, meas_phase=ph, meas_mag=mag, coh=np.ones_like(ph)
            )
        )
    return recs


def test_procrustes_recovers_rigid_transform() -> None:
    rng = np.random.default_rng(1)
    src = rng.standard_normal((8, 3))
    theta = 0.7
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]]
    )
    dst = src @ rot.T + np.array([1.0, -2.0, 0.5])
    aligned, rmse = gc.procrustes_align(src, dst)
    assert rmse < 1e-6
    assert np.allclose(aligned, dst, atol=1e-9)


def test_synthetic_geometry_recovery() -> None:
    rng = np.random.default_rng(0)
    mic_true = rng.standard_normal((8, 3)) * 0.05
    rotor = np.array([[0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, 0.2, 0], [-0.2, -0.2, 0]], float)
    freqs = np.linspace(400.0, 800.0, 60)
    recs = _synthetic_records(mic_true, rotor, freqs)
    mic_init = mic_true + rng.standard_normal((8, 3)) * 0.01

    resid0 = gc.phase_residual_rms_deg(recs, mic_init, rotor)
    mic_opt, _ = gc.run_bundle_adjustment(recs, mic_init, rotor, lam=1e-4, iters=3000, lr=2e-3)
    resid1 = gc.phase_residual_rms_deg(recs, mic_opt, rotor)

    assert resid1 < 0.1  # drives the coherent-band phase residual to ~0
    assert resid1 < resid0
    _, proc_rmse_cm = gc.procrustes_align(mic_opt, mic_true)
    assert proc_rmse_cm < 1.0  # recovered up to a global rigid gauge, sub-cm


def test_permutation_detects_reflection() -> None:
    """A genuinely reflected measured TDOA must select a flip; identity must not."""
    rng = np.random.default_rng(3)
    theta = np.deg2rad(112.5 + 45.0 * np.arange(8))
    mic = np.stack([np.full(8, 0.3), 0.0825 * np.cos(theta), 0.33 + 0.0825 * np.sin(theta)], -1)
    rotor = np.array([[0.23, -0.23, 0], [0.23, 0.23, 0], [-0.23, 0.23, 0], [-0.23, -0.23, 0]])
    sr = 44100
    dist = gc.s0.distance_matrix(mic, rotor)
    meas = np.vstack([gc.s0.freefield_tdoa_row(dist[r], 0, sr) for r in range(4)])
    meas = meas + rng.standard_normal(meas.shape) * 0.05  # tiny noise

    perm_id = gc.detect_mic_permutation(meas, mic, rotor, sr)
    assert not perm_id.flip_selected
    assert perm_id.best_roll_score > 0.99

    reflected = meas[:, ::-1]
    perm_flip = gc.detect_mic_permutation(reflected, mic, rotor, sr)
    assert perm_flip.flip_selected
    assert perm_flip.best_flip_score > 0.99


def _dregon_available() -> bool:
    try:
        gc.s0.find_dregon_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _dregon_available(), reason="DREGON recordings not present")
def test_dregon_pipeline_smoke() -> None:
    res = gc.calibrate_dregon_positions(speeds=(70,), lam=50.0, iters=400)
    assert res.resid_after_deg <= res.resid_before_deg + 1e-6
    assert res.mic_delta_cm.max() < 10.0  # cm-scale, gauge is well-posed


def _michaels_available() -> bool:
    try:
        gc.find_data_root()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _michaels_available(), reason="Michael's recordings not present")
def test_michaels_pipeline_smoke() -> None:
    res, perm = gc.calibrate_michaels_positions(windows=(60.0,), lam=20.0, iters=400)
    assert res.resid_after_deg < res.resid_before_deg
    assert perm.best_roll_name in perm.table
    assert res.mic_delta_cm.max() < 10.0
