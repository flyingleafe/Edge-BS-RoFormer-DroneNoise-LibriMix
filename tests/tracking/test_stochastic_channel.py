"""Regime 3: the STOCHASTIC comb channel (:func:`tracking.stochastic_split`).

Four claims, one file:

- **It leaves the floor where the floor is.** The gate the campaign reads is the
  order-cell excess of the BROADBAND channel, and that instrument cannot tell a
  surviving line from a dent — both are structure at the line position. So the
  claim is two sided: inside the comb bands the broadband channel's mean power
  must land ON the true floor, neither above it (a line survived) nor below it
  (the subtraction over-shot). This is the assertion the flat per-band Wiener
  gain this file used to test fails in BOTH directions.
- **It shapes per bin.** Several lines of very different strengths inside ONE
  union band must be attenuated by very different amounts. One flat gain per
  band cannot do that, and the measured consequence was a comb PATTERN that
  survived at reduced amplitude.
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
    STOCHASTIC_WIDTH_FACTOR,
    _line_mask,
    _wola_plan,
    comb_lines,
    frame_starts,
    line_half_widths,
    stft_power,
    stochastic_half_widths,
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


def narrowband(rng: np.random.Generator, n_t: int, center: float, fwhm: float) -> np.ndarray:
    """One INCOHERENT line: white noise shaped by a Gaussian band, unit variance.

    The linewidth is the fixture's INPUT and not an emergent property of a phase
    model, which is what makes it a regime-3 line: no envelope with a bandwidth
    small against the line spacing can carry it.
    """
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    shape = np.exp(-0.5 * ((freq - center) / (fwhm / 2.3548)) ** 2)
    v = np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * shape, n=n_t)
    return v / v.std()


def regime3_fixture(
    seed: int = 0, *, floor_rel: float = FLOOR_REL, seconds: float = SECONDS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(comb, floor, rates)`` — an INCOHERENT comb of linewidth ``0.6 k``."""
    rng = np.random.default_rng(seed)
    n_t = int(round(seconds * SR))
    rates = np.stack([np.full(n_t, v) for v in RATES])
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    mic_gain = np.array([1.0, 0.7])
    comb = np.zeros((2, n_t))
    for rate in RATES:
        for k in range(1, K_HI + 1):
            v = narrowband(rng, n_t, k * rate, LINEWIDTH_HZ_PER_K * k)
            comb += (mic_gain[:, None] * (1.0 / k**0.8)) * v[None, :]
    tilt = (1.0 + (freq / 150.0) ** 2) ** -0.7
    floor = np.stack(
        [np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * tilt, n=n_t) for _ in range(2)]
    )
    floor *= float(floor_rel) * float(np.sqrt(np.mean(comb**2))) / float(floor.std())
    return comb, floor, rates


def band_mean_power(sig: np.ndarray, rates: np.ndarray, *, n_fft: int = N_FFT) -> float:
    """Mean short-time power over the bins the split's search regions cover.

    The one reading behind the two-sided floor claim: the SAME frame grid, the
    same band law and the same normalization the split itself uses, applied to
    whichever channel the caller hands over. Comparing two signals through it is
    comparing them bin for bin.
    """
    y = np.atleast_2d(np.asarray(sig, dtype=np.float64))
    starts = frame_starts(int(y.shape[-1]), n_fft, n_fft // 4)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    df = SR / n_fft
    total, count = 0.0, 0
    for sub, chunk in stft_power(y, starts, n_fft):
        for i, s in enumerate(sub):
            rate = rates[:, int(s) : int(s) + n_fft].mean(axis=-1)
            lines, ks = comb_lines(rate, K_HI)
            idx = _line_mask(freq, lines, stochastic_half_widths(ks, min_half_hz=df))
            total += float(chunk[:, i, idx].sum())
            count += int(idx.sum()) * int(y.shape[0])
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# the band law


def test_comb_lines_tile_the_harmonics_per_rotor() -> None:
    lines, k = comb_lines([10.0, 20.0], 3)
    assert lines == pytest.approx([10.0, 20.0, 30.0, 20.0, 40.0, 60.0])
    assert k == pytest.approx([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_line_half_widths_are_capped_by_the_local_spacing() -> None:
    # The COHERENT law, unchanged: two lines 1 Hz apart at k 40 ask for 24 Hz
    # and the spacing allows 1 Hz. Regime 3 does not use this, but the coherent
    # path does, so the cap must stay exactly where it was.
    half = line_half_widths(np.array([1000.0, 1001.0]), np.array([40.0, 40.0]))
    assert half == pytest.approx([1.0, 1.0])
    assert float(line_half_widths(np.array([1000.0]), np.array([40.0]))[0]) == pytest.approx(24.0)
    assert float(
        line_half_widths(np.array([1000.0]), np.array([1.0]), min_half_hz=5.0)[0]
    ) == pytest.approx(5.0)


def test_the_stochastic_half_widths_are_three_linewidths_and_are_not_capped() -> None:
    """Regime 3's own law: ``3 * 0.6 k``, with NO spacing cap.

    The cap protects coherent identifiability. A per-bin power split with
    unioned bands can neither double count nor lose identifiability, and the cap
    only creates gaps: at high ``k`` the flanks of interleaved combs merge into
    one continuous rotor-locked field between the nominal lines, and a
    spacing-capped band never reaches it.
    """
    assert float(stochastic_half_widths([40.0])[0]) == pytest.approx(72.0)
    # Two lines 1 Hz apart still get the FULL three linewidths each, where the
    # coherent law would have cut both down to 1 Hz.
    assert stochastic_half_widths([40.0, 40.0]) == pytest.approx([72.0, 72.0])
    assert float(stochastic_half_widths([1.0], min_half_hz=5.0)[0]) == pytest.approx(5.0)
    assert pytest.approx(3.0) == STOCHASTIC_WIDTH_FACTOR


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
    # With the floor far below the signal the amplitude gain is zero, so the
    # channel IS the band-selected residual — the limit the algebra must reach.
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
    assert split.diag["p_smooth_frames"] == 5
    assert split.diag["p_smooth_bins"] == 3


# ---------------------------------------------------------------------------
# what the broadband channel is left holding


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_broadband_channel_lands_on_the_floor_inside_the_bands(seed: int) -> None:
    """TWO SIDED: no surviving hump, and no dent either.

    The band-mean power of the broadband channel is compared with the band-mean
    power of the INJECTED floor, on the split's own frame grid and its own band
    law. Both failure modes of the flat per-band gain this replaces show up
    here, and both are DENTS on this reading: the shipped v3c gain leaves
    ``(S / P)^2 P`` in the band and the conditional mean leaves ``S^2 / P``,
    which measure -2.30 dB and -2.04 dB against the floor where the per-bin
    amplitude gain measures +0.87 dB.

    The fixture is run at ``floor_rel = 1.0``, which puts the comb about 5 dB
    over the floor INSIDE the bands. That is the regime the method is for — the
    measured DREGON residual sits 0.5 to 1.7 dB over its floor there — and not
    the 20 dB of the capture fixture, where the analysis-modify-synthesis error
    of the overlap-add, not the gain, is what limits the reading.
    """
    comb, floor, rates = regime3_fixture(seed, floor_rel=1.0)
    split = stochastic_split(comb + floor, SR, rates, K_HI, n_fft=N_FFT)
    got = band_mean_power(split.broadband, rates)
    want = band_mean_power(floor, rates)
    assert got == pytest.approx(want, rel=0.25), (
        f"broadband band power {10 * np.log10(got / want):+.2f} dB against the true floor"
    )


def test_the_gate_reading_falls_when_the_channel_is_taken_out() -> None:
    """The order cell of the BROADBAND channel is what the campaign gates on.

    It is the reading the whole split exists to move, so it is read here on the
    same instrument the report carries: the absolute ``excess_db`` is the claim,
    and the ``depth_db`` ratio is only asked not to RISE — it is a ratio, and
    the module docstring's own warning is that a ratio can hold or rise as the
    absolute comb power falls toward the floor.
    """
    from tracking.joint_decompose import order_cell_bands

    comb, floor, rates = regime3_fixture()
    residual = comb + floor
    split = stochastic_split(residual, SR, rates, K_HI, n_fft=N_FFT)
    before = order_cell_bands(residual, SR, rates, k_max=K_HI, n_fft=4096)
    after = order_cell_bands(split.broadband, SR, rates, k_max=K_HI, n_fft=4096)
    for name in ("k1-9", "k10-24"):
        assert before[name]["excess_db"] - after[name]["excess_db"] > 10.0, name
        assert after[name]["depth_db"] < before[name]["depth_db"], name


def test_a_pure_floor_is_returned_almost_untouched() -> None:
    """The null: with no comb at all the broadband channel IS the input.

    This is what stops the split from being a band-pass filter, and it is also
    where the one systematic cost of the estimator is visible. The gain is
    clipped at one — it may attenuate, never boost — so on pure floor it takes
    the positive chi-square fluctuations and never gives the negative ones back.
    That soft-threshold bias is what the smoothing widths are sized against, and
    at ``5 x 3`` it is a few percent of the energy, not tens.
    """
    _, floor, rates = regime3_fixture()
    split = stochastic_split(floor, SR, rates, K_HI, n_fft=N_FFT)
    assert split.diag["stochastic_fraction"] < 0.05
    assert split.diag["mean_gain"] < 0.35
    kept = band_mean_power(split.broadband, rates) / band_mean_power(floor, rates)
    assert kept > 0.75, f"pure floor lost {-10 * np.log10(kept):.2f} dB inside the bands"


def test_one_union_band_attenuates_its_lines_by_different_amounts() -> None:
    """The regression test for the FLAT-GAIN failure mode, and it is per bin.

    Three lines of very different strengths are put inside ONE union band. A
    single gain per band scales all three by the same factor and leaves the comb
    PATTERN standing (measured on DREGON: order-cell depth 0.386 -> 0.380 dB at
    k10-24). The per-bin gain reads the smoothed periodogram, which carries the
    line SHAPE, so the strong line's core must lose far more power than the weak
    lines and far more again than the floor bins between them.
    """
    rng = np.random.default_rng(7)
    n_t = int(round(SECONDS * SR))
    rate = 50.0
    ks = (21, 22, 23)
    # 2 * 0.6 * k is 25.2 / 26.4 / 27.6 Hz against a 50 Hz spacing, so the three
    # search regions touch and become one band. That is the fixture's point.
    half = stochastic_half_widths(np.asarray(ks, dtype=float))
    assert float(half.min()) > 0.5 * rate, "the three regions must union into one"

    strengths = (1.0, 0.08, 0.08)
    floor = rng.standard_normal((1, n_t)) * 0.05
    y = floor.copy()
    for k, amp in zip(ks, strengths, strict=True):
        y += amp * narrowband(rng, n_t, k * rate, LINEWIDTH_HZ_PER_K * k)[None, :]
    rates = np.full((1, n_t), rate)
    split = stochastic_split(y, SR, rates, max(ks), n_fft=N_FFT)

    freq = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    starts = frame_starts(n_t, N_FFT, N_FFT // 4)

    def spectrum(sig: np.ndarray) -> np.ndarray:
        acc = np.zeros(freq.size)
        n = 0
        for _, chunk in stft_power(sig, starts, N_FFT):
            acc += chunk[0].sum(axis=0)
            n += chunk.shape[1]
        return acc / n

    keep = spectrum(split.broadband) / spectrum(y)
    core = [int(np.argmin(np.abs(freq - k * rate))) for k in ks]
    between = int(np.argmin(np.abs(freq - (ks[0] + 0.5) * rate)))
    assert keep[core[0]] < 0.5 * keep[core[1]], "the strong line was not shaped out"
    assert keep[core[0]] < 0.5 * keep[core[2]], "the strong line was not shaped out"
    assert keep[between] > 2.0 * keep[core[0]], "the floor between the lines was dug out"


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
# the floor in TIME: the block staircase and its seams

#: The seam fixture: 8 s so the four floor blocks are 2 s each, and a floor whose
#: level ramps by five times across the window, so the block staircase has
#: something to step over. The ramp is LINEAR, so the share the split takes
#: declines gradually and any sharp move in it is the estimator's, not the
#: fixture's.
SEAM_SECONDS = 8.0
SEAM_BLOCKS = 4
SEAM_RAMP = (0.6, 2.4)
#: The readout grid: four groups per floor block, so a block boundary falls on a
#: group boundary and the smearing of one 1024-sample frame is small against a
#: group. Groups rather than frames because the share of a NOISE residual is
#: chi-square noisy frame by frame, and the seam is a step, not a spike.
SEAM_GROUPS = 16


def seam_fixture(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(residual, rates)`` whose true floor level ramps across the window."""
    comb, floor, rates = regime3_fixture(seed, floor_rel=1.0, seconds=SEAM_SECONDS)
    u = np.arange(comb.shape[-1], dtype=np.float64) / (comb.shape[-1] - 1)
    lo, span = SEAM_RAMP
    return comb + floor * (lo + span * u)[None, :], rates


def taken_share(residual: np.ndarray, rates: np.ndarray, **kw: object) -> np.ndarray:
    """Per-group energy the split TOOK, against a constant-floor control run.

    The numerator alone is too noisy to read a step out of: it is the energy of
    a noise signal, so it fluctuates group to group whatever the gain does. The
    control run is the SAME split on the SAME residual with a single floor
    block, so it carries the same fluctuation and none of the floor's time
    dependence — dividing by it leaves the time dependence of the floor and
    almost nothing else. That is the series a seam is visible in.
    """
    y = np.asarray(residual, dtype=np.float64)
    n_t = int(y.shape[-1])
    w = n_t // SEAM_GROUPS

    def groups(x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)[:, : SEAM_GROUPS * w]
        return (a**2).reshape(a.shape[0], SEAM_GROUPS, w).sum(axis=(0, 2))

    ref = stochastic_split(y, SR, rates, K_HI, n_fft=N_FFT, psd_blocks=1)
    got = stochastic_split(y, SR, rates, K_HI, n_fft=N_FFT, psd_blocks=SEAM_BLOCKS, **kw)  # type: ignore[arg-type]
    # The first and last group hold the weighted overlap-add's own edge, where
    # the padded frames carry almost no signal; they are not a reading of a gain.
    return (groups(got.stochastic) / groups(ref.stochastic))[1:-1]


def seam_ratio(share: np.ndarray) -> float:
    """Biggest jump AT a block boundary, over the median jump everywhere else."""
    d = np.abs(np.diff(share))
    at = np.zeros(d.size, dtype=bool)
    per = SEAM_GROUPS // SEAM_BLOCKS
    at[[b * per - 2 for b in range(1, SEAM_BLOCKS)]] = True
    return float(d[at].max() / np.median(d[~at]))


def test_the_block_floor_makes_seams_and_the_interpolated_one_does_not() -> None:
    """The block floor is a STEP in time, and the step is visible in the output.

    ``S`` is one spectrum per block — about four seconds — which is a modelling
    statement that the wash is stationary over that span. It is not: much of the
    comb band sits within about 1 dB of the floor, so the clip at ``a = 1``
    toggles whole bands between "nothing taken" and "something taken" across one
    boundary, and the demonstration spectrograms then carry rectangular patches
    whose vertical edges land exactly on the block grid (measured on FLY124 at
    59.5 s and 63.4 s, boundaries 15 and 16 of that run's 3.96 s grid).

    So the assertion is about WHERE the changes are, not about how large they
    are: with the block floor the share is flat inside a block and cliffs at the
    boundaries, and with the floor interpolated in time it declines with the
    ramp and the boundaries are not special at all.
    """
    residual, rates = seam_fixture()
    stepped = taken_share(residual, rates)
    smooth = taken_share(residual, rates, floor_time_interp=True)

    assert seam_ratio(stepped) > 8.0, "the block floor left no seam to fix"
    assert seam_ratio(smooth) <= 2.0, (
        f"the interpolated floor still steps at the block boundaries: {seam_ratio(smooth):.2f}x "
        "the median jump elsewhere"
    )
    # Both take the same energy overall — the flag moves WHEN it is taken, not
    # how much: this is a continuity fix and not a stronger or weaker gain.
    assert float(smooth.sum()) == pytest.approx(float(stepped.sum()), rel=0.2)


def test_the_interpolated_floor_keeps_the_split_exact() -> None:
    # The identity is the split's own contract and it must not depend on which
    # floor the gain was read off: the broadband channel is a SUBTRACTION.
    residual, rates = seam_fixture()
    split = stochastic_split(
        residual, SR, rates, K_HI, n_fft=N_FFT, psd_blocks=SEAM_BLOCKS, floor_time_interp=True
    )
    got = np.asarray(split.stochastic) + np.asarray(split.broadband)
    assert float(np.abs(residual - got).max()) <= 1e-12 * float(np.abs(residual).max())
    assert split.diag["floor_time_interp"] is True


def test_the_time_interpolation_is_off_by_default_and_bitwise() -> None:
    # Every published number was produced on the block floor, so the default
    # path must be the same arithmetic and not merely the same answer.
    residual, rates = seam_fixture()
    kw = dict(n_fft=N_FFT, psd_blocks=SEAM_BLOCKS)
    base = stochastic_split(residual, SR, rates, K_HI, **kw)  # type: ignore[arg-type]
    off = stochastic_split(residual, SR, rates, K_HI, floor_time_interp=False, **kw)  # type: ignore[arg-type]
    assert np.array_equal(base.stochastic, off.stochastic)
    assert base.diag["floor_time_interp"] is False
    on = stochastic_split(residual, SR, rates, K_HI, floor_time_interp=True, **kw)  # type: ignore[arg-type]
    assert not np.array_equal(base.stochastic, on.stochastic)


def test_one_block_is_the_same_floor_whichever_way_it_is_read() -> None:
    # With a single block there is nothing to interpolate between, so the flag
    # must be a no-op rather than a second code path with its own answer.
    residual, rates = seam_fixture()
    kw = dict(n_fft=N_FFT, psd_blocks=1)
    off = stochastic_split(residual, SR, rates, K_HI, **kw)  # type: ignore[arg-type]
    on = stochastic_split(residual, SR, rates, K_HI, floor_time_interp=True, **kw)  # type: ignore[arg-type]
    assert np.array_equal(off.stochastic, on.stochastic)


def test_an_unusable_t_block_falls_back_to_the_even_block_centers() -> None:
    """``SmoothPSD.t_block`` holds block CENTERS, and the fallback rebuilds them.

    ``masked_smooth_psd`` writes ``t_start_s`` plus the center of each block of
    an even grid, so subtracting ``t_start_s`` gives the window-relative
    centers. A floor that does not carry one usable center per block — an older
    or a hand-built ``SmoothPSD`` — must not silently place the interpolation
    somewhere else: the even grid is derived instead, and on a floor that came
    from block C the two are the same thing.
    """
    from dataclasses import replace

    from tracking.joint_decompose import masked_smooth_psd

    residual, rates = seam_fixture()
    psd = masked_smooth_psd(residual, SR, rates, K_HI, n_fft=N_FFT, n_blocks=SEAM_BLOCKS)
    assert psd.t_block.size == SEAM_BLOCKS
    n_t = residual.shape[-1]
    want = (np.arange(SEAM_BLOCKS) + 0.5) * (n_t / SEAM_BLOCKS) / SR
    np.testing.assert_allclose(psd.t_block, want)

    kw = dict(n_fft=N_FFT, floor_time_interp=True)
    good = stochastic_split(residual, SR, rates, K_HI, psd=psd, **kw)  # type: ignore[arg-type]
    blind = stochastic_split(
        residual,
        SR,
        rates,
        K_HI,
        psd=replace(psd, t_block=np.zeros(SEAM_BLOCKS)),
        **kw,  # type: ignore[arg-type]
    )
    assert np.array_equal(good.stochastic, blind.stochastic)


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
