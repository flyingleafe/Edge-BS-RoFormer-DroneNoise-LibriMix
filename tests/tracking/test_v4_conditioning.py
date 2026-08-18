"""The v4 solve stays factorizable, or it fails loudly — never into SuperLU.

The field failure this file exists for: a full-scale DREGON spin-up window
(32 kHz, ``k_hi`` 83, 332 tracks in one coupling group) reported the banded
system non positive definite, fell back to ``_solve_group_splu``, and SuperLU
printed "Not enough memory to perform factorization" before the worker was
OOM-killed and the pool broke.

Two mechanisms, and only the first is a surprise:

- **The v4 bands make near-degenerate pairs, and nothing holds them up.** With
  ``b_A(k) = max(b0, 0.6 k)`` uncapped, two rotors whose lines nearly coincide
  have passbands that nearly coincide, so the difference direction of that pair
  has almost no data curvature. ``rho^2`` is small (a wide band) so the
  curvature prior does not hold it either, and ``beta = c0 S / H`` is small for
  exactly the STRONG lines — so the loudest comb in the window is the one that
  breaks the factorization. The assembler's absolute ``1e-8`` is all that is
  left, and it is not enough.
- **The fallback is the bomb.** SuperLU's fill-in on a 300-track group does not
  fit in memory. On a v2-sized group the fallback is right; under a proper
  amplitude prior it is not, so the v4 arm refuses it.

What is asserted here is the pair: the floored ridge factorizes a group that
fails without it, and where even the floor is not enough the failure is a clean
per-unit exception rather than a dead pool.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking.decompose import group_plan, solve_config
from tracking.joint_decompose import RIDGE_FLOOR_FRAC, v4_rho2_gain
from tracking.vk_tracking import env_stride, vk_envelopes

SR = 16000
SECONDS = 3.0
N_MIC = 2
K_HI = 30
#: The ridge a STRONG line asks for on its own — ``c0 S / H`` with the line 90 dB
#: over the floor. It is the case the floor exists for, so it is the case tested.
STRONG_BETA = 1e-9


def spinup_fixture(spread: float, base: float = 42.0, seed: int = 0):
    """Four rotors spinning up together, fanning out by ``spread`` and CROSSING.

    The field geometry: low rates (so the lines are densely spaced), all four
    rotors within a couple of rev/s of each other, and the fan reversing across
    the window so every pair coincides somewhere in it.
    """
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    t = np.arange(n_t) / SR
    off = np.array([-1.5, -0.5, 0.5, 1.5]) * spread
    rates = np.stack([base + 6.0 * t / SECONDS + o * (1.0 - 2.0 * t / SECONDS) for o in off])
    phase = 2.0 * np.pi * np.cumsum(rates, axis=-1) / SR
    audio = 0.001 * rng.standard_normal((N_MIC, n_t))
    for i in range(4):
        for k in range(1, K_HI + 1):
            for c, gain in enumerate((1.0, 0.8)):
                audio[c] += (
                    gain * (1.0 / k**0.5) * np.cos(k * phase[i] + 2.0 * np.pi * rng.random())
                )
    return audio, rates, n_t


def solve(spread: float, frac: float):
    audio, rates, n_t = spinup_fixture(spread)
    cfg = solve_config(K_HI, sr=SR, mics=N_MIC, bw_rps=1.0)
    vk = cfg.vk_config(K_HI)
    n_env = len(range(0, n_t, env_stride(vk)[0]))
    return vk_envelopes(
        audio,
        rates,
        vk,
        k_hi=K_HI,
        rho2_gain=v4_rho2_gain(rates, K_HI, cfg),
        ridge=np.full((4 * K_HI, n_env), STRONG_BETA),
        ridge_floor_frac=frac,
    )


def test_the_fixture_really_is_one_big_near_degenerate_group(capsys):
    """Guard the guard: if the rotors stop merging, the test below proves nothing."""
    _, rates, _ = spinup_fixture(2.0)
    cfg = solve_config(K_HI, sr=SR, mics=N_MIC, bw_rps=1.0)
    plan = group_plan(rates, K_HI, cfg)
    with capsys.disabled():
        print(f"\n  spin-up fixture: {plan}")
    assert plan["max_group"] == 4 * K_HI, "the four combs must merge into ONE coupled group"


def test_the_ridge_floor_is_what_makes_the_group_factorize(capsys):
    """THE fix, both directions: fails without the floor, factorizes with it."""
    with pytest.raises(MemoryError, match="non positive definite"):
        solve(2.0, 0.0)
    env = solve(2.0, RIDGE_FLOOR_FRAC)
    with capsys.disabled():
        print(
            f"  spread 2.0 rev/s: floor 0 -> non positive definite;"
            f" floor {RIDGE_FLOOR_FRAC:g} -> logdet {env.logdet:.1f}"
        )
    assert np.isfinite(env.logdet)
    assert np.isfinite(env.x).all()


def test_the_floor_has_a_reach_and_past_it_the_failure_is_clean(capsys):
    """Four rotors inside 1 rev/s are not four combs to a 3-second window.

    The floor that would factorize them costs a strong line 17 % of its
    amplitude, so it is not taken. What matters is that the window fails as ONE
    unit — a `.err` file from `gridrun` — instead of dragging SuperLU onto a
    300-track system and taking the worker, and the pool, with it.
    """
    with pytest.raises(MemoryError) as excinfo:
        solve(0.3, RIDGE_FLOOR_FRAC)
    msg = str(excinfo.value)
    with capsys.disabled():
        print(f"  spread 0.3 rev/s: {msg[:96]}...")
    # The message has to name the mechanism and the knob, or the .err file is
    # just a stack trace someone has to re-derive this whole file from.
    assert "non positive definite" in msg
    assert "ridge_floor_frac" in msg
    assert "splu" in msg
    assert str(4 * K_HI) in msg


#: A FLY-like rig: rotors TENS of hertz apart. The wide separations are the
#: whole point — they leave a high-``k`` track uncoupled, so the solver's own
#: per-group clamp never bites on it and its band is whatever ``bw_rps`` asks
#: for, which at ``k`` 60 is 60 Hz on a 100 Hz envelope grid.
FLY_RATES = (60.0, 75.0, 90.0, 105.0)


def fly_fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    t = np.arange(n_t) / SR
    rates = np.stack([np.full(n_t, v) + 2.0 * t / SECONDS for v in FLY_RATES])
    phase = 2.0 * np.pi * np.cumsum(rates, axis=-1) / SR
    audio = 0.001 * rng.standard_normal((N_MIC, n_t))
    for i in range(len(FLY_RATES)):
        for k in range(1, K_HI + 1):
            for c, gain in enumerate((1.0, 0.8)):
                audio[c] += (
                    gain * (1.0 / k**0.5) * np.cos(k * phase[i] + 2.0 * np.pi * rng.random())
                )
    return audio, rates, n_t


def test_the_fallback_band_law_does_not_inherit_the_solver_clamp(capsys):
    """The FLY failure, at the seam it happened: the RETRY's band derivation.

    ``state.rho2_gain`` is what a v2 schedule compiles to, and a v2 schedule is
    floored at the solver's own clamped band because it was only ever allowed to
    WIDEN ("a schedule never NARROWS a track below v1"). On a wide-separation rig
    that floor is 60 to 90 Hz for an isolated high-``k`` track, so the retry
    inherited a band the envelope grid cannot represent — measured below as the
    v3 contrast. ``bandwidth_neutral`` then widens a LOUD track further, the
    inverse Tuma relation saturates at ``fs_env`` exactly, and the objective's
    forward relation refuses to convert it back (``bw_hz=100.0 exceeds
    fs_env=100.0``, 13 of 21 FLY124 units).
    """
    from tracking.decompose import BandwidthSchedule, base_bandwidths, track_rho2_gain
    from tracking.joint_decompose import v4_fallback_rho2_gain
    from tracking.vk_tracking import _tuma_bw, _tuma_rho

    _, rates, _ = fly_fixture()
    cfg = solve_config(K_HI, sr=SR, mics=N_MIC, bw_rps=1.0)
    _, fs_env = env_stride(cfg.vk_config(K_HI))
    sched = BandwidthSchedule(3.0, 0.0, 1.5, 3.0)
    assert BandwidthSchedule.parse("3,0,1.5,3") == sched  # the CLI spelling of it
    base = np.asarray(base_bandwidths(rates, K_HI, cfg), dtype=np.float64)
    rho2_base = np.array([_tuma_rho(float(b), fs_env, 2) for b in base]) ** 2

    def achieved(gain):
        return _tuma_bw(np.sqrt(rho2_base * np.asarray(gain, dtype=np.float64)), fs_env, 2)

    v3 = achieved(track_rho2_gain(rates, K_HI, cfg, sched, 1.0))
    v4 = achieved(v4_fallback_rho2_gain(rates, K_HI, cfg, sched))
    none_sched = achieved(v4_fallback_rho2_gain(rates, K_HI, cfg, None))
    with capsys.disabled():
        print(
            f"\n  FLY geometry, achieved envelope band (schedule absmax 3 Hz):"
            f"\n    solver's own clamp     max {base.max():7.3f} Hz"
            f"\n    v2 schedule gain       max {v3.max():7.3f} Hz   <- the field failure"
            f"\n    v4 fallback law        max {v4.max():7.3f} Hz"
            f"\n    v4 fallback, NO schedule  max {none_sched.max():7.3f} Hz"
        )
    # The v3 gain is the recorded CONTRAST: it cannot pull the band below the
    # solver's own, so it leaves it far outside anything the grid supports.
    assert v3.max() > 20.0
    # The fallback law asks for the schedule and gets it.
    assert v4.max() <= sched.bw_abs_max + 1e-9
    # And it never depends on a driver having remembered to pass a schedule.
    assert none_sched.max() <= BandwidthSchedule().bw_abs_max + 1e-9
    assert v4.min() > 0.0


def test_the_readout_survives_a_band_at_the_envelope_rate():
    """The observer must not be the thing that raises.

    ``_tuma_bw`` saturates at ``fs_env`` for any gain that widens a band past
    the grid, and the forward relation refuses exactly that value. A readout
    that crashes on its own input turns a recoverable window into a dead unit,
    so it reads a saturated track at the widest band that IS convertible.
    """
    from tracking.joint_decompose import map_objective, masked_smooth_psd

    audio, rates, n_t = fly_fixture()
    k = np.tile(np.arange(1.0, 5.0), 1)
    n_env = 40
    psd = masked_smooth_psd(audio, SR, rates, 4, n_fft=1024, n_blocks=1)
    got = map_objective(
        audio,
        SR,
        psd,
        x=np.zeros((N_MIC, int(k.size), n_env), dtype=np.complex128),
        k=k,
        bw_track=np.full(int(k.size), 100.0),  # exactly fs_env: what the field saw
        theta=np.zeros((rates.shape[0], n_env)),
        psi=np.zeros((int(k.size), n_env)),
        fs_env=100.0,
        n_fft=1024,
    )
    assert np.isfinite(got["total"])
    assert np.isfinite(got["envelope_prior"])


def test_the_v3_path_keeps_its_splu_fallback():
    """No ridge means no v4, and the fallback that v2 needs is untouched."""
    rng = np.random.default_rng(0)
    n_t = 4000
    y = rng.standard_normal((1, n_t))
    r = np.full((1, n_t), 40.0)
    cfg = solve_config(4, sr=8000, mics=1, bw_rps=1.0)
    # Nothing here should raise: with ridge=None the solver is the v3 solver,
    # fallback and all.
    env = vk_envelopes(y, r, cfg.vk_config(4), k_hi=4)
    assert np.isfinite(env.x).all()
