"""The SHAPE-CONSTRAINED H: a hypothesis may only claim energy on its own comb.

The shape-free nuisance ``H = max(0, P~ - S)`` is fitted from the same data it
explains, so inside a hypothesis's search regions it explains ANY excess. That
is harmless while the regions are sparse — a wrong comb's regions then miss the
humps and pay for them in full, which is what
``tests/tracking/test_h_aware_objective.py`` measures. It stops being harmless
when the regions BLANKET the band: a region half width is ``3 x 0.6 k`` Hz, so
above ``k`` about 10 a multi-rotor comb's regions merge into one continuous
covering and every hypothesis's regions then sit on every hump. The term becomes
hypothesis independent and measures nothing. That is the one window of the
five-window rescore the H-aware term failed on (DREGON free flight, room 1).

``JointConfig.h_lorentzian`` constrains the SHAPE instead of the support: ``H``
is a NON-NEGATIVE mixture of Lorentzians pinned at the hypothesis's OWN line
positions, at the measured half width ``0.6 k`` Hz — the linewidth law itself,
never the region's 3.0 multiplier, which says where the fit may look and not how
wide a line is. A wrong comb's bumps land BETWEEN its lines, no non-negative
amplitude puts a Lorentzian peak there, and its amplitudes come back small.

Two claims, one file:

- **The fit is a fit.** A noiseless mixture of three Lorentzians at known
  positions is recovered amplitude by amplitude, and the same mixture read
  through misplaced lines is not.
- **THE ACCEPTANCE TEST, the DREGON failure in miniature.** In a regime where
  BOTH hypotheses' regions blanket the comb band and cover every hump, the
  shape-free term cannot separate the true rates from the shifted ones, and the
  Lorentzian term separates them by a wide margin.

**The fixture's one calibrated compromise, and it is geometric.** The shift
cannot be made five half widths. A hypothesis's own region reaches ``3 x 0.6 k``
Hz, so a shift of more than three half widths takes its regions OFF the humps
and the shape-free term starts discriminating again — the very thing this test
must hold flat. The shift here is two half widths, which is inside the region
(the blanket holds, measured below) and still far enough out on a Lorentzian
(``1 / (1 + 4)``, a fifth of the peak) for the constraint to bite. The other
compromise is the ROTOR COUNT: four close rates put so many lines in the band
that the shifted comb has a line within one half width of every hump and can
reproduce the whole surface, so the constraint measurably dies. Two well
separated combs are what leaves the shape any room at all, and that a truly
dense comb defeats the Lorentzian constraint as well is a property of the
method, not of this fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking.joint_decompose import (
    LINEWIDTH_HZ_PER_K,
    SmoothPSD,
    _line_mask,
    _lorentzian_design,
    comb_lines,
    map_objective,
    stochastic_half_widths,
)

SR = 2000
SECONDS = 8.0
N_MIC = 2
N_FFT = 1024
#: Two combs whose regions merge into a blanket above ``k`` about 8 — the rate
#: over ``3.6`` rule of :func:`stochastic_half_widths` — and whose lines are far
#: enough apart that a hump is still a hump.
RATES = (45.0, 64.0)
K_HI = 22
#: The harmonics that carry a hump. Below this the regions are narrow and
#: separated, so a shifted hypothesis would miss the humps for the ORDINARY
#: reason and the test would not be about the blanket at all.
K_LO = 8
#: Hypothesis B: every rate up by this many rev/s. Two half widths at every
#: harmonic (``1.2 / 0.6``), because the linewidth law and the shift are both
#: proportional to ``k``.
SHIFT = 1.2
#: Peak of a true hump over the floor, in power. Twenty is about 13 dB, which is
#: what DREGON's flanks stand over their own floor inside the regions.
HUMP = 20.0
LEVEL = 1.0


def flat_psd(n_ch: int = N_MIC) -> SmoothPSD:
    """The PINNED flat floor — the fixture's floor is known, never fitted."""
    freq = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    return SmoothPSD(
        freq=freq,
        t_block=np.array([0.5 * SECONDS]),
        log_s=np.full((n_ch, 1, freq.size), np.log(LEVEL)),
    )


def fixture(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(residual, true rates on the audio grid)`` — LORENTZIAN humps on a floor.

    The humps are put in as a power-spectral SHAPE and the noise is coloured
    through its square root, so the measured periodogram is the floor times that
    shape in expectation. They sit on the FIRST rotor's lines only: the second
    comb is there to make both hypotheses' regions blanket the band, which is
    the condition the test is about.
    """
    n_t = int(round(SECONDS * SR))
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n_t, d=1.0 / SR)
    shape = np.ones_like(f)
    for k in range(K_LO, K_HI + 1):
        shape += HUMP / (1.0 + ((f - k * RATES[0]) / (LINEWIDTH_HZ_PER_K * k)) ** 2)
    y = np.stack(
        [
            np.fft.irfft(
                np.fft.rfft(rng.standard_normal(n_t) * np.sqrt(SR * LEVEL)) * np.sqrt(shape),
                n=n_t,
            )
            for _ in range(N_MIC)
        ]
    )
    return y, np.stack([np.full(n_t, v) for v in RATES])


def score(y: np.ndarray, r: np.ndarray, *, lorentzian: bool) -> dict:
    """:func:`map_objective` at the pinned floor, every prior held at zero."""
    n_env = 20
    k = np.tile(np.arange(1.0, K_HI + 1), len(RATES))
    return map_objective(
        y,
        SR,
        flat_psd(int(y.shape[0])),
        x=np.zeros((int(y.shape[0]), k.size, n_env), dtype=np.complex128),
        k=k,
        bw_track=np.full(k.size, 3.0),
        theta=np.zeros((int(r.shape[0]), n_env)),
        psi=np.zeros((k.size, n_env)),
        fs_env=100.0,
        n_fft=N_FFT,
        h_carrier=r,
        h_lorentzian=lorentzian,
    )


# ---------------------------------------------------------------------------
# the fit is a fit


def test_the_nnls_recovers_a_known_lorentzian_mixture() -> None:
    """Three well separated lines, no noise: the amplitudes come back."""
    from scipy.optimize import nnls

    sr, n_fft = 2000, 16384
    freq = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lines, k = comb_lines(np.array([200.0]), 3)
    half = LINEWIDTH_HZ_PER_K * k
    bins = np.flatnonzero((freq >= 100.0) & (freq <= 700.0))
    design, kept = _lorentzian_design(freq, bins, lines, half)
    assert list(kept) == [0, 1, 2]

    want = np.array([3.0, 1.0, 0.4])
    f = freq[bins]
    y = sum(a / (1.0 + ((f - fl) / h) ** 2) for a, fl, h in zip(want, lines, half, strict=True))
    amp, _ = nnls(design, np.asarray(y))
    assert amp == pytest.approx(want, rel=0.01)
    assert float(np.sum((y - design @ amp) ** 2) / np.sum(np.asarray(y) ** 2)) < 0.01


def test_misplaced_lines_cannot_fit_the_same_mixture() -> None:
    # The other half of the claim: the recovery above is the POSITIONS being
    # right and not the basis being flexible. Three half widths off leaves most
    # of the mixture unexplained.
    from scipy.optimize import nnls

    sr, n_fft = 2000, 16384
    freq = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lines, k = comb_lines(np.array([200.0]), 3)
    half = LINEWIDTH_HZ_PER_K * k
    bins = np.flatnonzero((freq >= 100.0) & (freq <= 700.0))
    f = freq[bins]
    want = np.array([3.0, 1.0, 0.4])
    y = np.asarray(
        sum(a / (1.0 + ((f - fl) / h) ** 2) for a, fl, h in zip(want, lines, half, strict=True))
    )
    design, _ = _lorentzian_design(freq, bins, lines + 3.0 * half, half)
    amp, _ = nnls(design, y)
    assert float(np.sum((y - design @ amp) ** 2) / np.sum(y**2)) > 0.5


# ---------------------------------------------------------------------------
# the premise: both hypotheses blanket the band and cover every hump


def test_both_hypotheses_regions_blanket_the_band_and_cover_the_humps() -> None:
    """Measured, not assumed — the acceptance test below means nothing without it."""
    freq = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    df = SR / N_FFT
    regions = []
    for rates in (np.array(RATES), np.array(RATES) + SHIFT):
        lines, k = comb_lines(rates, K_HI)
        regions.append(_line_mask(freq, lines, stochastic_half_widths(k, min_half_hz=df)))
    lines, k = comb_lines(np.array(RATES), K_HI)
    humps = lines[(k >= K_LO) & (lines <= 0.5 * SR)]
    at = np.clip(np.searchsorted(freq, humps), 0, freq.size - 1)
    band = (freq >= K_LO * RATES[0]) & (freq <= 0.5 * SR)
    for region in regions:
        assert float(region[at].mean()) == pytest.approx(1.0)
        assert float(region[band].mean()) > 0.9


# ---------------------------------------------------------------------------
# THE acceptance test


def test_the_lorentzian_h_separates_what_the_shape_free_h_cannot() -> None:
    """The DREGON failure in miniature, and the fix, on one fixture.

    Both hypotheses' regions cover every hump (the test above), so the shape-free
    nuisance explains the humps for both and the data term barely moves between
    them. Pinning the nuisance to each hypothesis's OWN lines at the measured
    linewidth puts the true rates clearly ahead.
    """
    y, r_true = fixture()
    r_shift = r_true + SHIFT

    free_t, free_s = score(y, r_true, lorentzian=False), score(y, r_shift, lorentzian=False)
    lor_t, lor_s = score(y, r_true, lorentzian=True), score(y, r_shift, lorentzian=True)

    # The profiled term cannot separate them at all — the floor is pinned, so it
    # is the identical number for the two.
    assert free_s["total"] == pytest.approx(free_t["total"], rel=1e-12)
    # Nor does the shape of the two region sets differ: this is WHERE, not how
    # much spectrum a hypothesis claims.
    assert free_s["h_cells"] == pytest.approx(free_t["h_cells"], rel=0.1)

    free_margin = (free_s["data_h"] - free_t["data_h"]) / free_t["data_h"]
    lor_margin = (lor_s["data_h"] - lor_t["data_h"]) / lor_t["data_h"]
    # (a) the shape-free term is nearly blind here, which is the failure.
    assert free_margin < 0.05
    # (b) the shape-constrained one is not, and it is the same sign as truth.
    assert lor_margin > 0.10
    assert lor_margin > 2.5 * free_margin
    # (c) and the total inherits it, because data_h is the only term that moved.
    assert lor_s["total_h"] - lor_t["total_h"] > free_s["total_h"] - free_t["total_h"]


def test_the_fit_diagnostics_say_which_comb_the_energy_sits_on() -> None:
    # The mechanism, read straight off the diagnostics rather than through the
    # objective: the true comb's Lorentzians explain the humps, the shifted
    # comb's leave most of the excess on the table.
    y, r_true = fixture()
    true = score(y, r_true, lorentzian=True)["h_fit"]
    shift = score(y, r_true + SHIFT, lorentzian=True)["h_fit"]

    assert true["fit_residual_share"] < 0.15
    assert shift["fit_residual_share"] > 3.0 * true["fit_residual_share"]
    assert true["wall_s"] >= 0.0
    assert 0.0 < true["active_line_frac"] <= 1.0
