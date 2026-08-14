"""THE refactoring guard: the v3b joint solve, pinned array by array.

The acceptance test beside this one (``test_joint_decompose.py``) says the
alternation WORKS — it reads thresholds on an instrument. This one says the
alternation did not MOVE: it re-runs the shipped v3b configuration on a small
deterministic fixture and compares every product against a committed reference,
at 1e-10 relative. A refactor that keeps the arithmetic passes it; a refactor
that reorders one operation does not.

The reference was captured on the pre-refactor implementation (the direct
three-block loop). It is regenerated with::

    python tests/tracking/test_joint_regression.py --regen

Regenerating is how a DELIBERATE numerical change is recorded, and it must
never be done to make a red test green.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _joint_fixture import make_fixture

from tracking import joint_solve_window
from tracking.decompose import BandwidthSchedule, solve_config
from tracking.joint_decompose import JointConfig, JointResult

REF = Path(__file__).parent / "fixtures" / "joint_v3b_reference.npz"

SR = 8000
SECONDS = 3.0
N_ROT = 2
N_MIC = 2
K_MAX = 10
#: The shipped v3b arm: three rounds, the annealing ladder, ``psi`` from round
#: two, whitening on, bandwidth neutral, and the v2 linewidth-matched schedule.
#: Only the ladder's top rung is scaled, because the fixture stops at ``k`` 10.
SCHEDULE = "3,0,1.5,3"
LADDER = (3, 6, K_MAX)
#: The residual is pinned on a decimated grid: 3400 samples is as strong a
#: statement at 1e-10 as the whole array, and it keeps the reference small.
RESID_DECIMATE = 7


def _run() -> JointResult:
    """The v3b configuration on the fixture — the one call this file guards."""
    fx = make_fixture(seed=0, seconds=SECONDS, sr=SR, n_rot=N_ROT, n_mic=N_MIC, k_max=K_MAX)
    cfg = solve_config(K_MAX, sr=SR, mics=N_MIC, bw_rps=1.0, f_max=3000.0)
    return joint_solve_window(
        np.asarray(fx["audio"]),
        np.asarray(fx["r_hat"]),
        cfg,
        k_hi=K_MAX,
        mics=N_MIC,
        jcfg=JointConfig(iters=3, k_trust=LADDER, psd_n_fft=2048, profile_n_fft=2048),
        bw_schedule=BandwidthSchedule.parse(SCHEDULE),
        t_start_s=1.25,
    )


def _products(res: JointResult) -> dict[str, Any]:
    """Every array a consumer reads off one window, plus the diagnostics."""
    resid = np.asarray(res.residual, dtype=np.float64)
    return {
        "env_x": np.asarray(res.env.x, dtype=np.complex64),
        "env_phase": np.asarray(res.env.phase, dtype=np.float64)[:, ::RESID_DECIMATE],
        "env_bw_track": np.asarray(res.env.bw_track, dtype=np.float64),
        "env_valid": np.asarray(res.env.valid, dtype=bool),
        "theta_env": np.asarray(res.theta_env, dtype=np.float64),
        "psi": np.asarray(res.psi, dtype=np.float64),
        "track_energy": np.asarray(res.track_energy, dtype=np.float64),
        "residual": resid[:, ::RESID_DECIMATE],
        "residual_energy": (resid**2).sum(axis=-1),
        "psd_log_s": np.asarray(res.psd.log_s, dtype=np.float64),
        "psd_t_block": np.asarray(res.psd.t_block, dtype=np.float64),
        "iterations": np.array(json.dumps(res.iterations, sort_keys=True)),
    }


def _close(got: np.ndarray, want: np.ndarray, name: str) -> None:
    """Relative agreement at 1e-10, against the reference's own scale."""
    assert got.shape == want.shape, f"{name}: shape {got.shape} against {want.shape}"
    if want.dtype == bool:
        assert np.array_equal(got, want), f"{name}: {int((got != want).sum())} flips"
        return
    scale = float(np.max(np.abs(want))) if want.size else 0.0
    err = float(np.max(np.abs(got - want))) if want.size else 0.0
    assert err <= 1e-10 * max(scale, 1e-30), f"{name}: {err} against a scale of {scale}"


@pytest.mark.skipif(not REF.exists(), reason="reference not captured — run with --regen")
def test_v3b_joint_solve_reproduces_the_pinned_reference() -> None:
    got = _products(_run())
    with np.load(REF, allow_pickle=False) as ref:
        assert sorted(ref.files) == sorted(got), "the product set changed"
        for name in sorted(got):
            if name == "iterations":
                continue
            _close(np.asarray(got[name]), np.asarray(ref[name]), name)
        # The report numbers travel as JSON, so they are compared as text: a
        # rounded diagnostic that moves is a change in the report a consumer
        # reads, whatever the arrays did.
        assert json.loads(str(got["iterations"])) == json.loads(str(ref["iterations"]))


def _regen() -> None:
    REF.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF, allow_pickle=False, **_products(_run()))
    print(f"wrote {REF} ({REF.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    _regen()
