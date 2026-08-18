"""``J_v4``: the marginal Whittle likelihood, and what it can tell apart.

The v3 measure needed three bolt-ons because its noise model did not contain the
comb: the envelope marginalization (profiling charges nothing for the envelopes'
freedom), the H-aware data term (no coherent envelope carries the line flanks),
and the adaptive floor (a block floor is constant over four seconds and a gust is
not). ``J_v4`` needs none of them, because the comb IS the noise model:

    J_v4 = sum [ P / M + log M ] + phase priors + lam_f ||D2 log S||^2 ,
    M = S + sum_l H_l L_l

Four claims, one file:

- **It is OFF by default and it is a pure observer.** Without the v4 arguments
  the objective dict is what it was, key for key.
- **It discriminates.** The true rates score better than rates shifted far
  enough that every line of the hypothesis misses every real one — and the
  discrimination survives each hypothesis fitting its OWN floor and powers,
  which is the only comparison a blind rescore can make.
- **It has no envelope term and no separate rent.** Those are deletions, so they
  are asserted as deletions.
- **It scores the ORIGINAL signal.** Handing it a residual would count the comb
  twice, so the state-level entry point refuses to run without the audio.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tracking.decompose import solve_config
from tracking.joint_decompose import (
    JointConfig,
    fit_floor_powers,
    floor_penalty,
    joint_objective,
    joint_state,
    map_objective,
    masked_smooth_psd,
    solve_block,
)

SR = 8000
SECONDS = 6.0
N_FFT = 2048
N_MIC = 2
K_HI = 24
RATES = (50.0, 61.0)
#: The hypothesis shift, in rev/s. At ``k`` 1 it moves a line by 5 Hz and at
#: ``k`` 24 by 120 Hz, so above the lowest harmonics the shifted comb's lines sit
#: nowhere near the real ones.
SHIFT = 5.0


def fixture(rates, seed: int = 0):
    """An incoherent comb of half width ``0.6 k`` on a colored floor."""
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    psd = 1e-5 * (1.0 + (freq / 300.0) ** 2) ** -0.7
    for rate in rates:
        for k in range(1, K_HI + 1):
            f0 = k * rate
            if f0 > 0.42 * SR:
                continue
            psd = psd + (4e-4 / k**0.8) / (1.0 + ((freq - f0) / (0.6 * k)) ** 2)
    audio = np.stack(
        [
            np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * np.sqrt(psd * SR), n=n_t)
            * (0.7 + 0.3 * c)
            for c in range(N_MIC)
        ]
    )
    return audio, np.stack([np.full(n_t, v) for v in rates])


def score(audio, carrier, k) -> dict:
    """``J_v4`` of one hypothesis, which fits its OWN floor and line powers.

    Everything else is held at zero — no corrections, no envelopes — so the
    whole difference between two hypotheses is the model each one can build of
    the SAME audio.
    """
    psd, hp = fit_floor_powers(audio, SR, carrier, K_HI, n_fft=N_FFT, n_blocks=2)
    n_env = 40
    return map_objective(
        audio,
        SR,
        psd,
        x=np.zeros((audio.shape[0], int(k.size), n_env), dtype=np.complex128),
        k=k,
        bw_track=np.full(int(k.size), 3.0),
        theta=np.zeros((carrier.shape[0], n_env)),
        psi=np.zeros((int(k.size), n_env)),
        fs_env=100.0,
        n_fft=N_FFT,
        v4_powers=hp,
        v4_carrier=carrier,
    )


@pytest.fixture(scope="module")
def comb():
    audio, rates = fixture(RATES)
    return audio, rates, np.tile(np.arange(1.0, K_HI + 1), len(RATES))


def test_the_v4_readout_is_off_by_default_and_adds_only_its_own_keys(comb):
    audio, rates, k = comb
    psd = masked_smooth_psd(audio, SR, rates, K_HI, n_fft=N_FFT, n_blocks=2)
    n_env = 40
    common: dict[str, Any] = dict(
        x=np.zeros((N_MIC, int(k.size), n_env), dtype=np.complex128),
        k=k,
        bw_track=np.full(int(k.size), 3.0),
        theta=np.zeros((len(RATES), n_env)),
        psi=np.zeros((int(k.size), n_env)),
        fs_env=100.0,
        n_fft=N_FFT,
    )
    off = map_objective(audio, SR, psd, **common)
    assert not [key for key in off if key.endswith("_v4") or key == "floor_penalty"]

    _, hp = fit_floor_powers(audio, SR, rates, K_HI, n_fft=N_FFT, n_blocks=2, warm=psd)
    on = map_objective(audio, SR, psd, **common, v4_powers=hp, v4_carrier=rates)
    # Every v3 term is untouched — it is an observer, and it observes the same
    # arrays it did before.
    for key, val in off.items():
        assert on[key] == val, key
    assert {"data_v4", "total_v4", "floor_penalty", "h_energy_v4"} <= set(on)


def test_the_v4_total_is_its_own_terms_and_carries_no_envelope_prior(comb):
    """The deletions, asserted as deletions."""
    audio, rates, k = comb
    got = score(audio, rates, k)
    assert got["total_v4"] == pytest.approx(
        got["data_v4"] + got["phase_priors"] + got["floor_penalty"]
    )
    # ``envelope_prior`` is still REPORTED (it is a v3 column) but it is not in
    # the v4 total, and the priors here are zero, so the total is the data term
    # plus the floor penalty and nothing else.
    assert got["total_v4"] == pytest.approx(got["data_v4"] + got["floor_penalty"])
    assert got["floor_penalty"] > 0.0


def test_the_true_rates_beat_a_shifted_hypothesis(comb, capsys):
    """THE discrimination, with each hypothesis fitting its own model."""
    audio, rates, k = comb
    true = score(audio, rates, k)
    shifted = score(audio, rates + SHIFT, k)
    n = max(int(true["n_cells"]), 1)
    with capsys.disabled():
        print(
            f"\n  J_v4 per cell: true {true['total_v4'] / n:.6f}"
            f"   shifted by {SHIFT:g} rev/s {shifted['total_v4'] / n:.6f}"
            f"   margin {(shifted['total_v4'] - true['total_v4']) / n:.6f}"
        )
        print(
            f"  the v3 profiled total for the same two: {true['total'] / n:.6f}"
            f" vs {shifted['total'] / n:.6f}"
        )
    assert true["total_v4"] < shifted["total_v4"]
    # And clearly, not by rounding: a whole nat per thousand cells is a wide
    # margin on a readout whose terms are of order one per cell.
    assert (shifted["total_v4"] - true["total_v4"]) / n > 1e-3


def test_the_floor_penalty_reads_the_fitted_surface(comb):
    """It is a reading of ``S``, so a stiffer floor must cost less curvature."""
    audio, rates, _ = comb
    psd, _ = fit_floor_powers(audio, SR, rates, K_HI, n_fft=N_FFT, n_blocks=2)
    stiff = floor_penalty(psd, 2000.0)
    loose = floor_penalty(psd, 200.0)
    # Same surface, different weight: the penalty is linear in lambda_f.
    assert stiff > loose > 0.0


def test_the_state_level_objective_refuses_to_score_the_residual(comb):
    """Scoring the residual would count the comb twice — once subtracted, once modelled."""
    audio, rates, _ = comb
    cfg = solve_config(K_HI, sr=SR, mics=N_MIC, bw_rps=1.0)
    jc = JointConfig(iters=1, v4=True)
    state = joint_state(rates, cfg, k_hi=K_HI, n_t=int(audio.shape[-1]), jcfg=jc)
    from tracking.joint_decompose import floor_block

    state = solve_block(floor_block(state, audio), audio)
    with pytest.raises(ValueError, match="MARGINAL"):
        joint_objective(state)
    got = joint_objective(state, audio)
    assert "total_v4" in got
