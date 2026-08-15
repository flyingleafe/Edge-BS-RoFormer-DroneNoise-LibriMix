"""Regime 3: the STOCHASTIC comb channel (:func:`tracking.stochastic_split`).

Three claims, one file:

- **It captures what it is for.** A synthetic regime-3 comb — every line an
  INCOHERENT narrowband process of linewidth about ``0.6 k`` Hz, which is the
  measured shaft-wander law and is wider than any band a coherent envelope may
  have — plus a smooth colored floor. The channel must recover most of the comb
  and sweep in little of the floor.
- **The split is exact.** ``residual = stochastic + broadband`` to float
  roundoff, and the weighted overlap-add it is built on is an identity when the
  gain is one.
- **It is OFF by default.** The alternation with the stage absent is bit for bit
  the alternation without it, so every pinned regression stays pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking.joint_decompose import (
    LINEWIDTH_HZ_PER_K,
    _wola_plan,
    comb_lines,
    line_half_widths,
    stochastic_split,
)

SR = 8000
SECONDS = 4.0
RATES = (50.0, 61.0)
K_HI = 20
N_FFT = 1024
#: Amplitude decay of the injected comb, and the floor level as a fraction of
#: the comb's own root mean square.
FLOOR_REL = 0.15


def regime3_fixture(
    seed: int = 0, *, floor_rel: float = FLOOR_REL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(comb, floor, rates)`` — an INCOHERENT comb of linewidth ``0.6 k``.

    Each line is built in the frequency domain as a Gaussian band centered on
    ``k r`` whose full width at half maximum is exactly ``0.6 k`` Hz, so the
    linewidth is the fixture's input and not an emergent property of a phase
    model. That is what regime 3 means: there is no envelope with a bandwidth
    small against the line spacing that can carry this line.
    """
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    rates = np.stack([np.full(n_t, v) for v in RATES])
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    mic_gain = np.array([1.0, 0.7])
    comb = np.zeros((2, n_t))
    for rate in RATES:
        for k in range(1, K_HI + 1):
            sigma = LINEWIDTH_HZ_PER_K * k / 2.3548  # FWHM -> standard deviation
            shape = np.exp(-0.5 * ((freq - k * rate) / sigma) ** 2)
            v = np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * shape, n=n_t)
            comb += (mic_gain[:, None] * (1.0 / k**0.8)) * (v / v.std())[None, :]
    tilt = (1.0 + (freq / 150.0) ** 2) ** -0.7
    floor = np.stack(
        [np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * tilt, n=n_t) for _ in range(2)]
    )
    floor *= float(floor_rel) * float(np.sqrt(np.mean(comb**2))) / float(floor.std())
    return comb, floor, rates


# ---------------------------------------------------------------------------
# the band law


def test_comb_lines_tile_the_harmonics_per_rotor() -> None:
    lines, k = comb_lines([10.0, 20.0], 3)
    assert lines == pytest.approx([10.0, 20.0, 30.0, 20.0, 40.0, 60.0])
    assert k == pytest.approx([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_line_half_widths_are_capped_by_the_local_spacing() -> None:
    # Two lines 1 Hz apart at k 40: the linewidth law asks for 24 Hz and the
    # spacing allows 1 Hz. The cap is what keeps one band off its neighbour, so
    # the union below cannot take the neighbour's energy twice.
    half = line_half_widths(np.array([1000.0, 1001.0]), np.array([40.0, 40.0]))
    assert half == pytest.approx([1.0, 1.0])
    # Alone, a line gets the whole linewidth; the floor is the readout's bin.
    assert float(line_half_widths(np.array([1000.0]), np.array([40.0]))[0]) == pytest.approx(24.0)
    assert float(
        line_half_widths(np.array([1000.0]), np.array([1.0]), min_half_hz=5.0)[0]
    ) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# the identity


def test_the_overlap_add_round_trip_is_an_identity() -> None:
    """The analysis/synthesis pair, exercised at a gain of one, sample by sample.

    Every gain the split applies rides on this: the weighted overlap-add divides
    out its own window-square sum, so a unit gain rebuilds the input exactly at
    ANY window and any hop, with no COLA constant asked to be true.
    """
    rng = np.random.default_rng(3)
    n_t, n_fft, hop = 1000, 256, 64
    y = rng.standard_normal((2, n_t))
    pad, n_pad, starts = _wola_plan(n_t, n_fft, hop)
    assert pad == n_fft
    assert n_pad >= pad + n_t + pad
    assert int(starts[-1]) + n_fft == n_pad

    win = np.hanning(n_fft)
    yp = np.zeros((2, n_pad))
    yp[:, pad : pad + n_t] = y
    num = np.zeros((2, n_pad))
    den = np.zeros(n_pad)
    off = np.arange(n_fft)
    for s in starts:
        seg = yp[:, int(s) + off] * win
        back = np.fft.irfft(np.fft.rfft(seg, axis=-1), n=n_fft, axis=-1) * win
        num[:, int(s) : int(s) + n_fft] += back
        den[int(s) : int(s) + n_fft] += win**2
    assert float(den[pad : pad + n_t].min()) > 0.0
    got = (num / np.maximum(den, 1e-300))[:, pad : pad + n_t]
    assert float(np.abs(got - y).max()) <= 1e-6 * float(np.abs(y).max())


def test_a_negligible_floor_passes_the_bands_through() -> None:
    # With the floor far below the signal the Wiener gain is one, so the channel
    # IS the band-selected residual — the limit the algebra must reach.
    from tracking.joint_decompose import SmoothPSD

    comb, _, rates = regime3_fixture()
    n_t = int(comb.shape[-1])
    freq = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    zero_floor = SmoothPSD(
        freq=freq, t_block=np.array([0.5 * n_t / SR]), log_s=np.full((2, 1, freq.size), -80.0)
    )
    split = stochastic_split(comb, SR, rates, K_HI, psd=zero_floor, n_fft=N_FFT)
    # Not exactly one: the padded frames at the two ends hold almost no signal,
    # and a band with no power in it has nothing for the gain to hold on to.
    assert split.diag["mean_gain"] > 0.95
    assert split.diag["stochastic_fraction"] > 0.5


def test_the_three_channels_add_up_to_the_original() -> None:
    # coherent + stochastic + broadband = original. The coherent part is
    # whatever the alternation explained; here it is a stand-in, because the
    # claim is about the SPLIT of what is left and not about the solve.
    comb, floor, rates = regime3_fixture()
    coherent = 0.3 * comb
    original = comb + floor
    residual = original - coherent
    split = stochastic_split(residual, SR, rates, K_HI, n_fft=N_FFT)
    total = coherent + split.stochastic + split.broadband
    scale = float(np.abs(original).max())
    assert float(np.abs(original - total).max()) <= 1e-6 * scale
    assert split.diag["wola_min_weight"] > 0.0


# ---------------------------------------------------------------------------
# what it captures


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_channel_captures_the_incoherent_comb(seed: int) -> None:
    """At least 80 % of the injected comb, at most 20 % extra from the floor.

    ``capture`` is the projection of the channel onto the injected comb over the
    comb's own energy — the share of the excess it recovered. ``floor_taken`` is
    the projection onto the injected FLOOR over the same energy, which is what
    the channel swept in from between the lines; the two are read against one
    denominator so they are one statement about one split.
    """
    comb, floor, rates = regime3_fixture(seed)
    split = stochastic_split(comb + floor, SR, rates, K_HI, n_fft=N_FFT)
    e_comb = float((comb**2).sum())
    capture = float((split.stochastic * comb).sum()) / e_comb
    floor_taken = float((split.stochastic * floor).sum()) / e_comb
    assert capture >= 0.80, f"captured {capture:.3f} of the injected comb"
    assert floor_taken <= 0.20, f"swept in {floor_taken:.3f} of a comb's worth of floor"


def test_the_gate_reading_falls_when_the_channel_is_taken_out() -> None:
    """The order cell of the BROADBAND channel is what the campaign gates on.

    It is the reading the whole split exists to move, so it is read here on the
    same instrument the report carries: the absolute ``excess_db`` first, and
    the ``depth_db`` ratio beside it.
    """
    from tracking.joint_decompose import order_cell_bands

    comb, floor, rates = regime3_fixture()
    residual = comb + floor
    split = stochastic_split(residual, SR, rates, K_HI, n_fft=N_FFT)
    before = order_cell_bands(residual, SR, rates, k_max=K_HI, n_fft=4096)
    after = order_cell_bands(split.broadband, SR, rates, k_max=K_HI, n_fft=4096)
    for name in ("k1-9", "k10-24"):
        assert before[name]["excess_db"] - after[name]["excess_db"] > 10.0, name
        assert after[name]["depth_db"] < 0.6 * before[name]["depth_db"], name


def test_a_pure_floor_leaves_almost_nothing_in_the_channel() -> None:
    # The null: with no comb at all the Wiener gain has nothing to hold on to,
    # so the channel must be a small fraction of the residual and not the band's
    # whole content. This is what stops the split from being a band-pass filter.
    _, floor, rates = regime3_fixture()
    split = stochastic_split(floor, SR, rates, K_HI, n_fft=N_FFT)
    assert split.diag["stochastic_fraction"] < 0.05
    assert split.diag["mean_gain"] < 0.35


def test_overlapping_bands_are_one_band_and_one_gain() -> None:
    # Two rotors 0.02 rev/s apart put every pair of lines inside one bin, so the
    # 20 lines resolve into 10 UNIONS — the property that stops the energy under
    # a merged pair from being subtracted twice.
    rng = np.random.default_rng(0)
    twins = np.stack([np.full(SR, 50.0), np.full(SR, 50.02)])
    split = stochastic_split(rng.standard_normal((1, SR)), SR, twins, 10, n_fft=N_FFT)
    assert split.diag["n_bands_per_frame"] == pytest.approx(10.0, abs=0.5)
    apart = np.stack([np.full(SR, 50.0), np.full(SR, 133.0)])
    wide = stochastic_split(rng.standard_normal((1, SR)), SR, apart, 10, n_fft=N_FFT)
    assert wide.diag["n_bands_per_frame"] > split.diag["n_bands_per_frame"] + 5.0


# ---------------------------------------------------------------------------
# the default is OFF


def test_the_stage_is_off_by_default_and_changes_nothing_when_on() -> None:
    """The recipe without regime 3 is BITWISE the recipe with it switched off.

    And with it ON every product of the alternation is still bitwise the same —
    the split reads the finished residual and writes beside it.
    """
    from _joint_fixture import make_fixture

    import tracking as trk
    from tracking.decompose import solve_config
    from tracking.joint_decompose import JointConfig

    fx = make_fixture(seed=0, seconds=2.0, sr=SR, n_rot=2, n_mic=2, k_max=8)
    cfg = solve_config(8, sr=SR, mics=2, bw_rps=1.0, f_max=3000.0)
    jcfg = JointConfig(iters=2, k_trust=(3, 8), psd_n_fft=1024, profile_n_fft=1024)
    kw = dict(k_hi=8, mics=2, jcfg=jcfg, objective=False)
    off = trk.joint_solve_window(fx["audio"], fx["r_hat"], cfg, **kw)  # type: ignore[arg-type]
    on = trk.joint_solve_window(fx["audio"], fx["r_hat"], cfg, stochastic=True, **kw)  # type: ignore[arg-type]

    assert off.stochastic is None
    assert on.stochastic is not None
    for name in ("residual", "theta_env", "psi", "track_energy"):
        assert np.array_equal(getattr(on, name), getattr(off, name)), name
    assert np.array_equal(on.env.x, off.env.x)
    assert np.array_equal(on.psd.log_s, off.psd.log_s)
    assert on.iterations == off.iterations
    # And the seam is what the driver reads: residual - stochastic is broadband.
    broadband = np.asarray(on.residual) - np.asarray(on.stochastic)
    assert float(np.abs(broadband).max()) <= float(np.abs(on.residual).max()) * 1.5
