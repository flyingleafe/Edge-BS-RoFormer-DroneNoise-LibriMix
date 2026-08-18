"""F2: the amplitude prior, and the band law it makes possible.

The v3 solve caps every envelope band at a fraction of the local LINE SPACING,
and that cap is not a statement about lines — it is what an IMPROPER prior needs
to stay identifiable. The VK prior charges for CURVATURE only, so nothing bounds
an envelope's level, and two overlapping passbands have a cancelling mode. Give
each track a proper Gaussian amplitude prior and the cap is unnecessary, which is
what lets the bands open to the measured linewidth law.

Four claims, one file:

- **The prior is inert when it is not asked for.** ``ridge=None`` is the v3
  arithmetic bit for bit, on both solver paths.
- **THE CALIBRATION of ``c0``, frozen.** A line of known power in known noise
  must come back at the Wiener target, and the target is measured rather than
  derived: solve the same configuration with the ridge OFF on the line alone and
  on the noise alone, and those two recon powers ARE the line and noise power the
  envelope band admits. The optimal shrinkage of that observation follows.
- **It shrinks the right way.** A track sitting on the floor is pulled toward
  zero; a line far above it keeps its amplitude. That is the ratio behaviour the
  carve depends on, and it is what a threshold could not do.
- **The band law is the linewidth law.** ``max(b0, 0.6 k)`` Hz with NO spacing
  cap, against the v2 schedule's cap.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking.decompose import base_bandwidths, reconstruct, solve_config
from tracking.joint_decompose import (
    LINEWIDTH_HZ_PER_K,
    RIDGE_FLOOR_FRAC,
    V4_RIDGE_C0,
    v4_rho2_gain,
)
from tracking.vk_tracking import env_stride, vk_envelopes

SR = 16000
SECONDS = 8.0
RATE = 50.0
K_HI = 60
#: The three harmonics the calibration is read at — the design's own set. They
#: span a factor 12 of linewidth (3 Hz to 36 Hz), which is the axis that could
#: have made one constant fail to serve the whole comb.
K_TEST = (5, 20, 60)
#: Flat two-sided floor power spectral density of the calibration fixture.
S0 = 1e-6
#: How far the ridged solve's power may sit from the Wiener target.
TOLERANCE = 0.20


def _shaped(rng: np.random.Generator, n_t: int, psd: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * np.sqrt(psd * SR), n=n_t)


def calibration_fixture(h_over_s: float, seed: int = 0):
    """``(line-only, noise-only, n_t)`` — Lorentzian lines of known peak power."""
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    line_psd = np.zeros_like(freq)
    for k in K_TEST:
        gamma = LINEWIDTH_HZ_PER_K * k
        line_psd += (h_over_s * S0) / (1.0 + ((freq - k * RATE) / gamma) ** 2)
    return (
        _shaped(rng, n_t, line_psd)[None, :],
        _shaped(rng, n_t, np.full_like(freq, S0))[None, :],
        n_t,
    )


def _recon_power(
    y, r_audio, cfg, gain, ridge, n_t, k_hi: int = K_HI, frac: float = RIDGE_FLOOR_FRAC
) -> np.ndarray:
    vk = cfg.vk_config(k_hi)
    env = vk_envelopes(
        y, r_audio, vk, k_hi=k_hi, rho2_gain=gain, ridge=ridge, ridge_floor_frac=frac
    )
    stride, _ = env_stride(vk)
    _, energy = reconstruct(env.x, env.k, env.rotor, env.phase, stride)
    return energy / n_t


def test_the_ridge_is_inert_when_it_is_not_asked_for():
    """``ridge=None`` and an all-zero ridge are the same solve; both are v3."""
    rng = np.random.default_rng(0)
    n_t = 8000
    y = rng.standard_normal((2, n_t))
    r = np.full((1, n_t), 40.0)
    cfg = solve_config(6, sr=8000, mics=2, bw_rps=1.0)
    base = vk_envelopes(y, r, cfg.vk_config(6), k_hi=6)
    zero = vk_envelopes(y, r, cfg.vk_config(6), k_hi=6, ridge=np.zeros(6))
    assert np.array_equal(base.x, zero.x)
    # And a positive ridge DOES move it, or the argument would be decorative.
    moved = vk_envelopes(y, r, cfg.vk_config(6), k_hi=6, ridge=np.full(6, 10.0))
    assert not np.allclose(base.x, moved.x)


def test_the_ridge_reaches_both_solver_paths_alike():
    """The banded solve and the splu reference must agree under the prior."""
    rng = np.random.default_rng(1)
    n_t = 4000
    y = rng.standard_normal((1, n_t))
    r = np.full((1, n_t), 40.0)
    cfg = solve_config(4, sr=8000, mics=1, bw_rps=1.0)
    from dataclasses import replace as dc_replace

    vk = cfg.vk_config(4)
    ridge = np.linspace(0.5, 4.0, 4)
    banded = vk_envelopes(y, r, vk, k_hi=4, ridge=ridge)
    splu = vk_envelopes(y, r, dc_replace(vk, solver="splu"), k_hi=4, ridge=ridge)
    assert np.allclose(banded.x, splu.x, rtol=1e-8, atol=1e-12)


def test_the_ridge_constant_hits_the_wiener_target(capsys):
    """THE calibration of :data:`V4_RIDGE_C0`, frozen at 1.

    It is 1 for a derivable reason, not a fitted one: the band is
    ``0.6 k`` Hz and the line's own half width is the same ``0.6 k``, so both
    the line and the noise contribute in proportion to the same noise-equivalent
    bandwidth and the ratio of the two POWERS the band admits is exactly the
    ratio of the two DENSITIES, ``S / H``. This measures that it is also true of
    the real solver, whose ridge competes with the curvature prior as well.

    It runs WITH :data:`RIDGE_FLOOR_FRAC` applied, which is what makes this the
    UPPER bound on that constant: a floor big enough to disturb these numbers is
    a floor that has stopped being a regularizer and started being the estimator.
    """
    cfg = solve_config(K_HI, sr=SR, mics=1, bw_rps=1.0)
    rows: dict[tuple[float, int], list[float]] = {}
    for h_over_s in (10.0, 3.0):
        for seed in (0, 1, 2):
            lines, noise, n_t = calibration_fixture(h_over_s, seed=seed)
            r_audio = np.full((1, n_t), RATE)
            gain = v4_rho2_gain(r_audio, K_HI, cfg)
            zero = np.zeros(len(gain))
            # The target, measured: what the band admits of each part alone.
            # The TARGET is what the band admits with no prior at all, so the
            # two reference solves carry neither the ridge nor its floor.
            p_line = _recon_power(lines, r_audio, cfg, gain, zero, n_t, frac=0.0)
            p_noise = _recon_power(noise, r_audio, cfg, gain, zero, n_t, frac=0.0)
            beta = np.full(len(gain), V4_RIDGE_C0 / h_over_s)  # c0 * S / H, u = 1
            p_got = _recon_power(lines + noise, r_audio, cfg, gain, beta, n_t)
            for k in K_TEST:
                m = k - 1
                target = p_line[m] ** 2 / (p_line[m] + p_noise[m])
                rows.setdefault((h_over_s, k), []).append(float(p_got[m] / target))

    with capsys.disabled():
        print(f"\n  ridge calibration, c0 = {V4_RIDGE_C0}: recon power over the Wiener target")
        for (h_over_s, k), vals in sorted(rows.items()):
            print(f"    H/S {h_over_s:4.1f}  k{k:02d}  {np.mean(vals):5.3f} +- {np.std(vals):.3f}")

    for (h_over_s, k), vals in rows.items():
        got = float(np.mean(vals))
        assert abs(got - 1.0) <= TOLERANCE, f"H/S {h_over_s} k{k}: {got:.3f} of the Wiener target"


def test_the_prior_shrinks_a_floor_level_track_and_spares_a_line(capsys):
    """The ratio behaviour the carve depends on — no threshold anywhere.

    Two tracks see the identical observation; only their fitted line power
    differs. The one whose ``H`` says "there is no line here" must lose almost
    all of its amplitude, and the one whose ``H`` is far above the floor must
    keep almost all of it.
    """
    rng = np.random.default_rng(2)
    n_t = 16000
    y = rng.standard_normal((1, n_t))
    r = np.full((1, n_t), 50.0)
    cfg = solve_config(4, sr=8000, mics=1, bw_rps=1.0)
    gain = v4_rho2_gain(r, 4, cfg)
    free = _recon_power(y, r, cfg, gain, np.zeros(4), n_t, k_hi=4, frac=0.0)
    # beta = c0 * S / H: a line 20 dB over the floor, and one AT the floor.
    strong = _recon_power(y, r, cfg, gain, np.full(4, V4_RIDGE_C0 / 100.0), n_t, k_hi=4)
    weak = _recon_power(y, r, cfg, gain, np.full(4, V4_RIDGE_C0 * 100.0), n_t, k_hi=4)
    with capsys.disabled():
        print(
            f"\n  shrinkage: H/S 100 keeps {float(np.mean(strong / free)):.3f}"
            f", H/S 0.01 keeps {float(np.mean(weak / free)):.4f} of the unridged power"
        )
    assert float(np.mean(strong / free)) > 0.9
    assert float(np.mean(weak / free)) < 0.05


def test_the_v4_band_law_is_the_linewidth_law_with_no_spacing_cap(capsys):
    """``max(b0, 0.6 k)`` Hz — and the v3 cap is what it is measured against."""
    n_t = 16000
    # Four rotors close together: a DENSE comb, where the v2 spacing cap bites
    # hardest and the lines of different rotors interleave.
    r = np.stack([np.full(n_t, v) for v in (70.0, 72.0, 74.0, 76.0)])
    cfg = solve_config(40, sr=16000, mics=1, bw_rps=1.0)
    vk = cfg.vk_config(40)
    _, fs_env = env_stride(vk)
    gain = v4_rho2_gain(r, 40, cfg)
    base = base_bandwidths(r, 40, cfg)
    from tracking.vk_tracking import _tuma_bw, _tuma_rho

    got = _tuma_bw(
        np.sqrt(np.array([_tuma_rho(float(b), fs_env, 2) for b in base]) ** 2 * gain), fs_env, 2
    )
    _, k = np.repeat(np.arange(4), 40), np.tile(np.arange(1, 41), 4)
    want = np.maximum(1.0, LINEWIDTH_HZ_PER_K * k)
    with capsys.disabled():
        print("\n  v4 band law against the solver's own clamped band, Hz")
        for kk in (1, 10, 25, 40):
            i = int(np.flatnonzero(k == kk)[0])
            print(f"    k{kk:02d}  want {want[i]:6.2f}  got {got[i]:6.2f}  v3 base {base[i]:6.2f}")
    assert np.allclose(got, want, rtol=1e-6)
    # The point of the exercise: at high k the law asks for far more than the
    # solver's own spacing-clamped band would ever have given.
    assert float(np.max(want / base)) > 10.0


def test_v4_rho2_gain_scales_with_rho_scale():
    n_t = 8000
    r = np.full((1, n_t), 50.0)
    cfg = solve_config(8, sr=8000, mics=1, bw_rps=1.0)
    one = v4_rho2_gain(r, 8, cfg)
    two = v4_rho2_gain(r, 8, cfg, rho_scale=2.0)
    assert two == pytest.approx(one * 4.0)
