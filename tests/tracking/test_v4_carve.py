"""The CARVE: what the comb channel takes where there is no comb.

Gate 2 of ``docs/v4-unified-model-design.md``, in miniature. The v3 arm's bands
are capped at a fraction of the local line spacing, and opening them to the
physical linewidth law is what v4 does — but a wide band over a track with no
line under it is exactly how a decomposition carves a trench in the broadband
floor, which is the artifact the FLY rotor-0 band shows. The amplitude prior is
what makes the wide bands safe, and this file measures that it does.

The fixture gives ONE rotor a band it owns alone, and gives line power to the
LOWER half of its harmonics only. The upper half is pure floor sitting under
tracks the model is still solving for, so what those tracks take is the carve.

Three claims:

- **The comb channel takes almost nothing where there is no line.** Under a
  tenth of what the same wide bands take without the prior.
- **It still takes the lines it owns.** The prior is a ratio and not a gate, so
  the owned harmonics must be almost untouched — otherwise "takes nothing" would
  be satisfied by a decomposition that does nothing.
- **The identity survives it.** ``comb + broadband == audio``, exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking import joint_solve_window
from tracking.decompose import reconstruct, solve_config
from tracking.joint_decompose import JointConfig

SR = 8000
SECONDS = 8.0
RATE = 50.0
K_HI = 30
#: Harmonics that carry a line. Above this the band is pure floor — and the v3
#: mask has blanketed it long since, so the floor there is also where the v4
#: joint fit has to be right for the ridge to be right.
K_OWNED = 15
N_MIC = 2


def carve_fixture(seed: int = 0):
    """One comb, lines on the low harmonics only, over a smooth colored floor."""
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    psd = 2e-5 * (1.0 + (freq / 500.0) ** 2) ** -0.6
    for k in range(1, K_OWNED + 1):
        psd = psd + (1e-3 / k**0.8) / (1.0 + ((freq - k * RATE) / (0.6 * k)) ** 2)
    audio = np.stack(
        [
            np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * np.sqrt(psd * SR), n=n_t)
            * (0.8 + 0.2 * c)
            for c in range(N_MIC)
        ]
    )
    return audio, np.full((1, n_t), RATE)


@pytest.fixture(scope="module")
def arms():
    """The same v4 solve with the amplitude prior on, and with it switched off.

    ``v4_ridge_c0 = 0`` is exactly "the v4 bands with no prior": the physical
    linewidth law, no spacing cap, and nothing bounding an envelope's level —
    which is the configuration the v3 spacing cap existed to prevent.
    """
    audio, rates = carve_fixture()
    cfg = solve_config(K_HI, sr=SR, mics=N_MIC, bw_rps=1.0)
    with_ridge = joint_solve_window(
        audio,
        rates,
        cfg,
        k_hi=K_HI,
        mics=N_MIC,
        objective=False,
        jcfg=JointConfig(iters=2, v4=True),
    )
    no_ridge = joint_solve_window(
        audio,
        rates,
        cfg,
        k_hi=K_HI,
        mics=N_MIC,
        objective=False,
        jcfg=JointConfig(iters=2, v4=True, v4_ridge_c0=0.0),
    )
    return audio, with_ridge, no_ridge


def test_the_comb_channel_takes_almost_nothing_where_there_is_no_line(arms, capsys):
    audio, with_ridge, no_ridge = arms
    k = np.asarray(with_ridge.env.k)
    free = k > K_OWNED
    owned = (k >= 3) & (k <= K_OWNED)
    take = np.asarray(with_ridge.track_energy)
    bare = np.asarray(no_ridge.track_energy)

    carve = float(take[free].sum() / max(float(bare[free].sum()), 1e-300))
    kept = float(take[owned].sum() / max(float(bare[owned].sum()), 1e-300))
    with capsys.disabled():
        print("\n  carve: comb-channel take in the LINE-FREE part of an owned band")
        print(f"    with the amplitude prior   {take[free].sum():.4e}")
        print(f"    wide bands, no prior       {bare[free].sum():.4e}   ratio {carve:.4f}")
        print(f"    the owned harmonics keep   {kept:.4f} of the unridged take")
        print(
            f"    residual fraction: prior {with_ridge.iterations[-1]['residual_fraction']:.4f}"
            f"   no prior {no_ridge.iterations[-1]['residual_fraction']:.4f}"
        )
    assert carve < 0.10, f"the line-free tracks took {carve:.3f} of the unridged take"
    # And it is not "takes nothing anywhere": the lines it owns survive.
    assert kept > 0.5


def test_the_three_channel_identity_is_a_subtraction(arms):
    """``comb + broadband == audio`` to float roundoff, by construction."""
    audio, with_ridge, _ = arms
    env = with_ridge.env
    stride = int(round(float(env.t_env[1] - env.t_env[0]) * env.fs))
    comb, _ = reconstruct(env.x, env.k, env.rotor, env.phase, stride)
    got = float(np.abs(audio[:N_MIC] - (comb + with_ridge.residual)).max())
    assert got < 1e-6, f"identity broken by {got:.3e}"
    # The v4 arm runs no stochastic stage, so the residual IS the broadband
    # channel and there is no third array to reconcile.
    assert with_ridge.stochastic is None


def test_the_v4_arm_refuses_regime_three():
    audio, rates = carve_fixture()
    cfg = solve_config(K_HI, sr=SR, mics=N_MIC, bw_rps=1.0)
    with pytest.raises(ValueError, match="already carries the line flanks"):
        joint_solve_window(
            audio,
            rates,
            cfg,
            k_hi=K_HI,
            mics=N_MIC,
            jcfg=JointConfig(iters=1, v4=True),
            stochastic=True,
        )
