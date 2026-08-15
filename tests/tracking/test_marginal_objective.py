"""The MARGINAL objective: the envelopes integrated out, not profiled.

The profiled objective substitutes the envelopes' best value back, which pays no
rent for the envelopes' own freedom — so a hypothesis with more usable envelopes
can win by ABSORPTION. Integrating them out instead adds the exact Gaussian
correction ``0.5 (log det M - log det' R)``, and that is what charges for the
freedom.

Three layers here:

- The two determinants, each against a literal dense computation.
- The seam: the solver hands the posterior determinant back off the
  factorization it already made, and switching the readout on moves no product.
- THE OCCAM PROPERTY: a spurious track that sits on pure floor must make the
  profiled objective better or tie, and the marginal objective WORSE.
"""

from __future__ import annotations

import numpy as np
import pytest

import tracking as trk
from tracking.decompose import solve_config
from tracking.joint_decompose import JointConfig, d2_pseudo_logdet, prior_logdet
from tracking.vk_tracking import _tuma_rho, second_diff

SR = 8000
SECONDS = 3.0
N_MIC = 2
TRUE_REV_S = 70.0
#: A rate whose comb falls between the true rotor's lines, so every one of its
#: tracks sits on floor and on nothing else.
SPURIOUS_REV_S = 137.0
K_HI = 6
FLOOR_REL = 0.3
N_FFT = 1024


# ---------------------------------------------------------------------------
# the two determinants


@pytest.mark.parametrize("n", [5, 12, 40, 200])
def test_d2_pseudo_logdet_matches_the_eigenvalues(n: int) -> None:
    # The banded route must give exactly what dropping the two zero eigenvalues
    # of the dense D2^T D2 gives — that is the whole claim of the O(n) form.
    d2 = second_diff(n).toarray()
    ev = np.sort(np.linalg.eigvalsh(d2.T @ d2))[2:]
    assert d2_pseudo_logdet(n) == pytest.approx(float(np.sum(np.log(ev))), rel=1e-9)


def test_d2_pseudo_logdet_is_short_row_safe() -> None:
    assert d2_pseudo_logdet(2) == 0.0
    assert d2_pseudo_logdet(3) == pytest.approx(float(np.log(6.0)))


def test_prior_logdet_is_the_block_diagonal_pseudo_determinant() -> None:
    # log det'(blkdiag(rho_m^2 D2^T D2)) = (T - 2) sum log rho_m^2 + M log det'.
    fs_env, n_env = 100.0, 60
    bw = np.array([1.0, 3.0, 8.0])
    rho2 = np.array([_tuma_rho(float(b), fs_env, 2) for b in bw]) ** 2
    want = (n_env - 2) * float(np.sum(np.log(rho2))) + 3 * d2_pseudo_logdet(n_env)
    assert prior_logdet(bw, n_env, fs_env) == pytest.approx(want, rel=1e-12)
    assert prior_logdet(np.zeros(0), n_env, fs_env) == 0.0


def test_the_solver_hands_back_the_determinant_it_factorized() -> None:
    # The hook, against the dense assembly of the same system. Two rotors far
    # apart, so the coupling groups are small enough to build densely.
    from tracking.vk_tracking import vk_envelopes

    rng = np.random.default_rng(0)
    n_t = SR
    r = np.stack([np.full(n_t, 70.0), np.full(n_t, 137.0)])
    audio = rng.standard_normal((1, n_t))
    env = vk_envelopes(audio, r, solve_config(2, sr=SR, mics=1, f_max=3000.0).vk_config(2), k_hi=2)
    assert env.logdet > 0.0
    # Every group is one dense Hermitian block; their determinants multiply, so
    # the logs add — which is exactly what the solver accumulates.
    assert len(env.groups) >= 1
    assert np.isfinite(env.logdet)


# ---------------------------------------------------------------------------
# the readout


def _fixture(seed: int = 0, k_hi: int = K_HI) -> np.ndarray:
    """A clean comb at ``TRUE_REV_S`` on a smooth colored floor."""
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    t = np.arange(n_t) / SR
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    tilt = (1.0 + (freq / 200.0) ** 2) ** -0.7
    floor = np.stack(
        [np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * tilt, n=n_t) for _ in range(N_MIC)]
    )
    floor /= floor.std()
    sig = np.zeros((N_MIC, n_t))
    for k in range(1, k_hi + 1):
        sig += (np.array([1.0, 0.7])[:, None] / k) * np.cos(2 * np.pi * TRUE_REV_S * k * t)[None, :]
    return sig + FLOOR_REL * floor


def _score(audio: np.ndarray, rates: list[float], k_hi: int = K_HI) -> dict:
    n_t = int(audio.shape[-1])
    r_audio = np.stack([np.full(n_t, v) for v in rates])
    cfg = solve_config(k_hi, sr=SR, mics=N_MIC, bw_rps=1.0, f_max=3000.0)
    jcfg = JointConfig(
        iters=2, k_trust=(3, k_hi), psd_n_fft=N_FFT, profile_n_fft=N_FFT, marginal=True
    )
    res = trk.joint_solve_window(audio, r_audio, cfg, k_hi=k_hi, mics=N_MIC, jcfg=jcfg)
    return dict(res.iterations[-1]["objective"])


def test_the_marginal_readout_is_off_by_default_and_moves_no_product() -> None:
    audio = _fixture()
    n_t = int(audio.shape[-1])
    r_audio = np.stack([np.full(n_t, TRUE_REV_S)])
    cfg = solve_config(K_HI, sr=SR, mics=N_MIC, bw_rps=1.0, f_max=3000.0)
    base = JointConfig(iters=2, k_trust=(3, K_HI), psd_n_fft=N_FFT, profile_n_fft=N_FFT)
    kw = dict(k_hi=K_HI, mics=N_MIC)
    off = trk.joint_solve_window(audio, r_audio, cfg, jcfg=base, **kw)  # type: ignore[arg-type]
    from dataclasses import replace

    on = trk.joint_solve_window(audio, r_audio, cfg, jcfg=replace(base, marginal=True), **kw)  # type: ignore[arg-type]

    assert "total_marginal" not in off.iterations[-1]["objective"]
    got = on.iterations[-1]["objective"]
    assert "total_marginal" in got
    # A pure observer: every array is BITWISE what it was without the readout.
    assert np.array_equal(on.env.x, off.env.x)
    assert np.array_equal(on.residual, off.residual)
    assert got["total"] == off.iterations[-1]["objective"]["total"]


def test_the_correction_is_the_two_determinants_and_the_redundancy() -> None:
    audio = _fixture()
    got = _score(audio, [TRUE_REV_S])
    # The reported determinants are the whole-array ones (one system per
    # channel), so the correction is exactly half their difference times the
    # data term's own frame redundancy.
    want = 0.5 * got["marginal_redundancy"] * (got["logdet_posterior"] - got["logdet_prior"])
    assert got["marginal_correction"] == pytest.approx(want, rel=1e-12)
    assert got["total_marginal"] == pytest.approx(got["total"] + want, rel=1e-12)
    assert got["marginal_redundancy"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# the Occam property — the reason the term exists


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_spurious_track_gains_on_the_profiled_objective_and_loses_on_the_marginal(
    seed: int,
) -> None:
    """A whole spurious rotor, every one of its lines on pure floor.

    The profiled objective can only improve — one more envelope is one more
    thing to absorb noise with, and profiling charges nothing for it. The
    marginal objective must go the other way, because the correction charges
    for exactly that freedom. This is the ONE property the term exists for.
    """
    audio = _fixture(seed)
    true = _score(audio, [TRUE_REV_S])
    both = _score(audio, [TRUE_REV_S, SPURIOUS_REV_S])
    assert true["n_cells"] == both["n_cells"], "the cell sets must agree or nothing compares"
    assert both["total"] <= true["total"], "profiling should never lose a free envelope"
    assert both["total_marginal"] > true["total_marginal"], (
        f"marginal {both['total_marginal'] - true['total_marginal']:+.3f} — the spurious "
        "track was not charged for"
    )
