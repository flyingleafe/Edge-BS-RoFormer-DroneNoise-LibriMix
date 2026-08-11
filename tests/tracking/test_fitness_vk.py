"""F_VK: the profiled coupled-VK residual and its L-BFGS trajectory optimizer.

Five properties, all on synthetic combs with known truth and all sized for the
default suite (short windows, low sample rate, small ``k_max``):

1. the envelope-theorem gradient against central finite differences,
2. the score's ordering — truth beats scale, offset and smooth-noise corruptions,
3. what the ``k_max`` schedule actually buys on this landscape,
4. L-BFGS recovery from a constant-offset init,
5. the fixed-degrees-of-freedom discipline (identical cells across candidates).

Run:  pytest tests/tracking/test_fitness_vk.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracking.fitness_vk import (
    FVKConfig,
    FVKStage,
    alias_charge,
    fvk_loss,
    fvk_score,
    optimize_trajectory,
    solve_envelopes,
)

FS = 4000.0
DUR = 1.0
HOP = 0.032


def _comb(rates: list[np.ndarray], k_top: int, snr_db: float, seed: int) -> np.ndarray:
    """``(1, T)`` sum of harmonic combs (amps ``1/sqrt(k)``, random phases) + noise."""
    rng = np.random.default_rng(seed)
    sig = np.zeros_like(rates[0])
    for r in rates:
        phase = 2 * np.pi * np.cumsum(r) / FS
        for k in range(1, k_top + 1):
            sig += (1.0 / np.sqrt(k)) * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
    noise = rng.standard_normal(len(sig))
    noise *= np.sqrt(np.mean(sig**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return (sig + noise)[None, :]


def _twin_fixture(k_top: int = 8, snr_db: float = 10.0, seed: int = 11):
    """Two rotors, both wobbling, at 30 and 35 rev/s. Truth on the frame grid."""
    n = int(DUR * FS)
    t = np.arange(n) / FS
    r1 = 30.0 + 1.0 * np.sin(2 * np.pi * 0.5 * t)
    r2 = 35.0 - 0.5 * np.cos(2 * np.pi * 0.3 * t)
    audio = _comb([r1, r2], k_top, snr_db, seed)
    ft = np.arange(0, DUR, HOP)
    truth = np.stack([np.interp(ft, t, r1), np.interp(ft, t, r2)])
    return audio, ft, truth


def _cfg(k_max: int, bw_rps: float = 0.5, **kw) -> FVKConfig:
    return FVKConfig(
        sr=int(FS), fs_env=100.0, k_min=1, k_max=k_max, bw_rps=bw_rps, f_max=1800.0, **kw
    )


# ---------------------------------------------------------------------------
# 1. the envelope theorem


def test_envelope_theorem_gradient_matches_finite_differences():
    """``d/dtheta`` of the detached-``a*`` loss IS the profiled objective's.

    Two readings, because they check different things. The FIXED-``a*``
    difference isolates the carrier chain rule (``phi = 2 pi k cumsum(r) / sr``
    -> ``Re[a c]`` -> the weighted residual) and must be exact to rounding. The
    PROFILED difference re-solves the envelopes at every probe, so it is the
    envelope theorem itself; it holds only as well as ``a*`` is a stationary
    point of the objective, and the VK solve's decimation and band-limiting
    leave a few percent of the gradient scale behind.
    """
    fs, n = 2000.0, 2000
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)
    r_true = 25.0 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(r_true) / fs
    sig = sum((1 / np.sqrt(k)) * np.cos(k * phase + rng.uniform(0, 2 * np.pi)) for k in range(1, 5))
    noise = rng.standard_normal(n)
    noise *= np.sqrt(np.mean(sig**2) / 10 / np.mean(noise**2))
    audio = (sig + noise)[None, :]
    cfg = FVKConfig(sr=int(fs), fs_env=100.0, k_min=1, k_max=4, bw_rps=0.5, f_max=900.0)
    base = r_true[None, :]
    theta0, h = 0.07, 1e-3

    def evaluate(theta: float, env=None):
        th = torch.tensor(float(theta), dtype=torch.float64, requires_grad=True)
        r_t = torch.from_numpy(base) + th
        if env is None:
            with torch.no_grad():
                env = solve_envelopes(audio, r_t.detach().numpy(), cfg, k_hi=cfg.k_max)
        return fvk_loss(audio, r_t, cfg, env=env), th

    loss, th = evaluate(theta0)
    loss.backward()
    assert th.grad is not None
    grad = float(th.grad)

    env0 = solve_envelopes(audio, base + theta0, cfg, k_hi=cfg.k_max)
    fd_fixed = (float(evaluate(theta0 + h, env0)[0]) - float(evaluate(theta0 - h, env0)[0])) / (
        2 * h
    )
    assert fd_fixed == pytest.approx(grad, rel=1e-3)

    fd_profiled = (float(evaluate(theta0 + h)[0]) - float(evaluate(theta0 - h)[0])) / (2 * h)
    # Measured 3.2 % of the gradient at this window length (0.7 % at 2 s); the
    # residual is the solver's decimated/band-limited normal equations, not the
    # chain rule, which the fixed-a* reading above pins to 1e-5.
    assert abs(fd_profiled - grad) < 0.10 * abs(grad)


# ---------------------------------------------------------------------------
# 2. the ordering


@pytest.mark.parametrize(
    "label",
    ["scale_1.005", "offset_0.3", "offset_0.1", "smooth_noise"],
)
def test_truth_beats_within_basin_corruptions(label):
    """The global optimum sits at truth for every within-basin corruption."""
    audio, ft, truth = _twin_fixture()
    cfg = _cfg(8)
    t = np.arange(int(DUR * FS)) / FS
    smooth = np.stack(
        [np.interp(ft, t, 0.15 * np.sin(2 * np.pi * 0.9 * t + p)) for p in (0.3, 2.0)]
    )
    candidates = {
        "scale_1.005": truth * 1.005,
        "offset_0.3": truth + 0.3,
        "offset_0.1": truth + 0.1,
        "smooth_noise": truth + smooth,
    }
    best = fvk_score(audio, FS, truth, ft, cfg, reference=truth)
    other = fvk_score(audio, FS, candidates[label], ft, cfg, reference=truth)
    assert best["objective"] < other["objective"]
    assert best["r2"] > other["r2"]
    assert best["r2"] > 0.85  # 10 dB SNR: 0.909 of the energy is comb


def test_score_shape_and_alias_charge():
    """The reported profile, and the counter-term that breaks the sub-multiple.

    A half-rate comb CONTAINS the true comb as a subset, so its residual is
    almost as good — only the empty slots tell them apart, which is exactly what
    :func:`alias_charge` counts (design §1 Fact 2).
    """
    audio, ft, truth = _twin_fixture()
    cfg = _cfg(8, alias_penalty=1.0)
    at_truth = fvk_score(audio, FS, truth, ft, cfg, reference=truth)
    assert at_truth["k_index"] == list(range(1, 9))
    assert len(at_truth["k_energy"]) == 8
    assert at_truth["k_energy"][0] == max(at_truth["k_energy"])  # 1/sqrt(k) amps
    assert at_truth["alias_charge"] == pytest.approx(0.0, abs=1e-9)

    half = fvk_score(audio, FS, truth / 2, ft, cfg, reference=truth)
    assert half["alias_charge"] > 0.3  # half the comb's slots hold no line
    assert half["objective"] > at_truth["objective"]


def test_alias_charge_is_per_cell():
    """``alias_charge`` returns one charge per ``(channel, track)`` cell."""
    audio, ft, truth = _twin_fixture()
    cfg = _cfg(8)
    n_t = audio.shape[-1]
    t = np.arange(n_t) / FS
    r_audio = np.stack([np.interp(t, ft, row) for row in truth])
    env = solve_envelopes(audio, r_audio, cfg, k_hi=8)
    mean, per_cell = alias_charge(env, cfg)
    assert per_cell.shape == (1, 16)
    assert mean == pytest.approx(float(per_cell.mean()))


# ---------------------------------------------------------------------------
# 3. what the k schedule buys


def test_k_annealing_sharpens_the_well():
    """The coarse rung is shallow and unimodal; the fine rung is deep and narrow.

    Design §1 Fact 5 predicts the basin ``1/(K T)``. That law is for a COHERENT
    sum over harmonics at a fixed integration length. Here every harmonic gets
    its own VK envelope with a ``k``-scaled band, so the capture radius is
    ``bw_rps / 2`` rev/s at EVERY harmonic and ``k_max`` does not move it: at a
    0.5 rev/s constant error the gradient still points at truth at ``k_max`` 5
    AND at 80 (measured; no assert on the second — it is the finding, not the
    contract). What ``k_max`` moves is the DEPTH and the curvature of the well:
    on the 1 s / 18 rev/s single-rotor window the objective at truth falls
    0.587 -> 0.073 from ``k_max`` 5 to 80 while its ±0.1 rev/s neighbours barely
    move, i.e. the well narrows by an order of magnitude. That is the precision
    half of the same law, and it is why the schedule still has to start coarse.

    The basin knob here is therefore ``bw_rps``, and it has units. Opened to
    2.0 rev/s (no longer a capture band, an ill-conditioned one) the same
    ``k_max`` = 80 landscape breaks into 7 local minima inside ±1 rev/s, against
    2 at ``k_max`` = 5 — the classic picture, reached by the bandwidth axis.
    """
    audio, ft, truth = _twin_fixture(k_top=8)
    n_t = audio.shape[-1]
    t = np.arange(n_t) / FS
    err = 0.5

    def slope(k_hi: int) -> float:
        cfg = _cfg(k_hi, bw_rps=0.2)
        r_audio = np.stack([np.interp(t, ft, row) for row in truth]) + err
        th = torch.zeros((), dtype=torch.float64, requires_grad=True)
        loss = fvk_loss(audio, torch.from_numpy(r_audio) + th, cfg, k_hi=k_hi)
        loss.backward()
        assert th.grad is not None
        return float(th.grad)

    assert slope(5) > 0  # positive gradient = descending toward the lower truth

    def contrast(k_hi: int) -> float:
        cfg = _cfg(k_hi, bw_rps=0.2)
        at = fvk_score(audio, FS, truth, ft, cfg, reference=truth, k_hi=k_hi)
        off = fvk_score(audio, FS, truth + 0.15, ft, cfg, reference=truth, k_hi=k_hi)
        return off["objective"] / at["objective"]

    assert contrast(8) > contrast(2)  # the fine rung discriminates harder


# ---------------------------------------------------------------------------
# 4. the optimizer


def test_lbfgs_recovers_a_constant_offset():
    """0.3 rev/s constant-offset init -> truth, under the default schedule."""
    audio, ft, truth = _twin_fixture(k_top=12)
    cfg = _cfg(12)
    interior = (ft > 0.15) & (ft < DUR - 0.15)  # the taper span is not fitted

    r_out, diag = optimize_trajectory(
        audio, FS, truth + 0.3, ft, cfg, reference=truth, knot_s=0.25, smooth_lambda=1.0
    )
    err = np.abs(r_out - truth)[:, interior]
    assert err.max() < 0.05
    assert np.sqrt((err**2).mean()) < 0.03

    # Continuation validity: the schedule is capped and de-duplicated against
    # the harmonic cap, every rung improves its own loss, and the argmin path is
    # continuous (each rung moves less than the one before it).
    stages = diag["stages"]
    assert [s["k_max"] for s in stages] == [5, 10, 12]
    for s in stages:
        assert s["loss_end"] <= s["loss_start"]
    assert stages[0]["move_max"] > stages[-1]["move_max"]


def test_optimizer_leaves_truth_where_it_is():
    """Started AT truth, the refiner must not wander off the plateau."""
    audio, ft, truth = _twin_fixture(k_top=12)
    cfg = _cfg(12)
    interior = (ft > 0.15) & (ft < DUR - 0.15)
    r_out, _ = optimize_trajectory(
        audio,
        FS,
        truth,
        ft,
        cfg,
        reference=truth,
        schedule=(FVKStage(6, max_iter=10), FVKStage(12, max_iter=10)),
        smooth_lambda=1.0,
    )
    assert np.abs(r_out - truth)[:, interior].max() < 0.05


# ---------------------------------------------------------------------------
# 5. fixed degrees of freedom


def test_fixed_degrees_of_freedom_across_candidates():
    """Every candidate is scored on the identical harmonic/channel cell set.

    The VK validity mask is the one thing that would react to the candidate, so
    :meth:`FVKConfig.vk_config` disables it and the harmonic cap comes from the
    pinned reference instead. Without that, a slower candidate would model MORE
    harmonics and win on cell count rather than on fit.
    """
    audio, ft, truth = _twin_fixture()
    cfg = _cfg(8)
    ref = fvk_score(audio, FS, truth, ft, cfg, reference=truth)
    for cand in (truth * 1.05, truth * 0.9, truth + 2.0):
        other = fvk_score(audio, FS, cand, ft, cfg, reference=truth)
        assert other["n_cells"] == ref["n_cells"]
        assert other["n_tracks"] == ref["n_tracks"]
        assert other["k_hi"] == ref["k_hi"]
        assert other["n_channels"] == ref["n_channels"]


def test_permutation_invariance():
    """Swapping the rotor rows cannot change an acoustic fit.

    Same property :mod:`tracking.fitness` records: rotor identity is certified
    by the residual pairing, never by the fit.
    """
    audio, ft, truth = _twin_fixture()
    cfg = _cfg(8)
    direct = fvk_score(audio, FS, truth, ft, cfg, reference=truth)
    swapped = fvk_score(audio, FS, truth[::-1], ft, cfg, reference=truth)
    assert swapped["objective"] == pytest.approx(direct["objective"], rel=1e-7)


# ---------------------------------------------------------------------------
# the stages


def test_stages_round_trip_through_a_frame():
    """``fvk_stage`` scores without touching ``rps``; ``fvk_refine_stage`` refines."""
    import tracking as trk

    audio, ft, truth = _twin_fixture(k_top=8)
    frame = trk.tracking_frame(
        audio, int(FS), rps=truth + 0.2, frame_times=ft, rps_meas=truth, dtype=np.float64
    )
    cfg = _cfg(8)
    scored = trk.fvk_stage(cfg)(frame)
    r_seen, _ = trk.get_rps(scored)
    assert np.allclose(r_seen, truth + 0.2)
    entry = scored["meta"]["tracking"][-1]
    assert entry["stage"] == "fvk"
    assert entry["n_cells"] == 16

    refined = trk.fvk_refine_stage(
        cfg, schedule=(FVKStage(4, max_iter=8), FVKStage(8, max_iter=8))
    )(scored)
    r_new, _ = trk.get_rps(refined)
    interior = (ft > 0.15) & (ft < DUR - 0.15)
    assert np.abs(r_new - truth)[:, interior].max() < np.abs(truth + 0.2 - truth).max()
    assert refined["meta"]["tracking"][-1]["stage"] == "fvk_refine"
