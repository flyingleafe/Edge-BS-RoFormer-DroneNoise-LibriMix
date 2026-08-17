"""The H-AWARE data term: the STOCHASTIC COMB written into the likelihood.

The coherent envelopes cannot carry the ``0.6 k`` Hz flanks of a line — that is
regime 3's whole reason to exist — so the profiled data term charges EVERY
hypothesis for the same flank energy and the true trajectory gains nothing by
sitting on it. The H-aware term gives the noise model a comb-shaped nuisance
``H = max(0, P~ - S)`` inside the hypothesis's OWN search regions, so a
trajectory whose regions sit on the humps stops paying for them while a
hypothesis whose regions land on empty floor gains nothing.

Three claims, one file:

- **It is a pure observer and it is OFF by default.** With ``h_aware`` off the
  objective dict is what it was, key for key and bit for bit, and switching it
  on moves no product of the alternation.
- **It explains a hump only where the regions sit.** On an INCOHERENT comb the
  true rates take the data term far below the profiled one; rates shifted so the
  regions miss the lines leave it where it was.
- **That IS the discrimination.** At a pinned floor the profiled data term
  cannot separate the two hypotheses at all, and the H-aware one separates them
  by a wide margin.

The fixture is the regime-3 one (``test_stochastic_channel.py``): lines whose
linewidth is an INPUT and not an emergent property of a phase model, which is
exactly the energy no coherent envelope can carry.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tracking as trk
from tracking.decompose import solve_config
from tracking.joint_decompose import (
    LINEWIDTH_HZ_PER_K,
    _line_mask,
    comb_lines,
    map_objective,
    masked_smooth_psd,
    stochastic_half_widths,
)
from tracking.joint_decompose import JointConfig as JC

SR = 8000
SECONDS = 4.0
N_MIC = 2
N_FFT = 2048
#: The hypothesis shift: five rev/s, which puts every region of the shifted comb
#: off every line of the true one at the harmonics tested here.
SHIFT = 5.0
#: Floor level as a fraction of the comb's own root mean square.
FLOOR_REL = 0.15


def narrowband(rng: np.random.Generator, n_t: int, center: float, fwhm: float) -> np.ndarray:
    """One INCOHERENT line: white noise shaped by a Gaussian band, unit variance."""
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    shape = np.exp(-0.5 * ((freq - center) / (fwhm / 2.3548)) ** 2)
    v = np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * shape, n=n_t)
    return v / v.std()


def fixture(
    rates: tuple[float, ...], k_hi: int, *, seed: int = 0, comb: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """``(audio, rates on the audio grid)`` — a broad-line comb on a colored floor."""
    rng = np.random.default_rng(seed)
    n_t = int(round(SECONDS * SR))
    freq = np.fft.rfftfreq(n_t, d=1.0 / SR)
    mic_gain = np.array([1.0, 0.7])
    sig = np.zeros((N_MIC, n_t))
    if comb:
        for rate in rates:
            for k in range(1, k_hi + 1):
                v = narrowband(rng, n_t, k * rate, LINEWIDTH_HZ_PER_K * k)
                sig += (mic_gain[:, None] * (1.0 / k**0.8)) * v[None, :]
    tilt = (1.0 + (freq / 150.0) ** 2) ** -0.7
    floor = np.stack(
        [np.fft.irfft(np.fft.rfft(rng.standard_normal(n_t)) * tilt, n=n_t) for _ in range(N_MIC)]
    )
    level = float(np.sqrt(np.mean(sig**2))) if comb else 1.0
    floor *= FLOOR_REL * level / float(floor.std())
    return sig + floor, np.stack([np.full(n_t, v) for v in rates])


def track_k(rates: tuple[float, ...], k_hi: int) -> np.ndarray:
    """The pinned track table's ``k`` — what the objective reads the comb from."""
    return np.tile(np.arange(1.0, k_hi + 1), len(rates))


def score(y: np.ndarray, r: np.ndarray, psd: object, k: np.ndarray, *, h: bool = True) -> dict:
    """:func:`map_objective` at a PINNED floor, with the priors held at zero.

    Everything but the carrier is identical between two calls — the same audio,
    the same floor, zero corrections and an all-zero envelope — so the whole
    difference between two hypotheses is the data term, which is what these
    tests are about.
    """
    n_env = 40
    return map_objective(
        y,
        SR,
        psd,  # type: ignore[arg-type]
        x=np.zeros((int(y.shape[0]), int(k.size), n_env), dtype=np.complex128),
        k=k,
        bw_track=np.full(int(k.size), 3.0),
        theta=np.zeros((int(r.shape[0]), n_env)),
        psi=np.zeros((int(k.size), n_env)),
        fs_env=100.0,
        n_fft=N_FFT,
        h_carrier=r if h else None,
    )


def line_overlap(r_hyp: np.ndarray, r_true: np.ndarray, k_hi: int) -> float:
    """Share of the TRUE line cores that the hypothesis's search regions cover."""
    freq = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    df = SR / N_FFT
    lh, kh = comb_lines(r_hyp[:, 0], k_hi)
    region = _line_mask(freq, lh, stochastic_half_widths(kh, min_half_hz=df))
    lt, _ = comb_lines(r_true[:, 0], k_hi)
    core = _line_mask(freq, lt, np.full(lt.size, df))
    return float((region & core).sum()) / max(float(core.sum()), 1.0)


# ---------------------------------------------------------------------------
# the readout is an observer


def test_h_aware_off_leaves_the_objective_dict_exactly_as_it_was() -> None:
    # Bitwise: the H-aware branch allocates and computes NOTHING unless it is
    # asked, and the terms it does not touch are equal by `==`, not by approx.
    y, r = fixture((50.0,), 8)
    k = track_k((50.0,), 8)
    psd = masked_smooth_psd(y, SR, r, 8, n_fft=N_FFT)
    off = score(y, r, psd, k, h=False)
    on = score(y, r, psd, k, h=True)

    assert set(on) - set(off) == {"data_h", "total_h", "h_cells", "h_energy"}
    for key, val in off.items():
        assert on[key] == val, f"{key} moved when the H-aware readout was switched on"
    assert on["total_h"] == on["total"] - on["data"] + on["data_h"]


def test_h_aware_is_off_by_default_and_moves_no_product() -> None:
    # The same statement one layer up: through JointConfig, on the alternation.
    y, r = fixture((50.0,), 6)
    cfg = solve_config(6, sr=SR, mics=N_MIC, bw_rps=1.0, f_max=3000.0)
    base = JC(iters=2, k_trust=(3, 6), psd_n_fft=N_FFT, profile_n_fft=N_FFT)
    kw = dict(k_hi=6, mics=N_MIC)
    off = trk.joint_solve_window(y, r, cfg, jcfg=base, **kw)  # type: ignore[arg-type]
    on = trk.joint_solve_window(y, r, cfg, jcfg=replace(base, h_aware=True), **kw)  # type: ignore[arg-type]

    assert "data_h" not in off.iterations[-1]["objective"]
    got = on.iterations[-1]["objective"]
    assert "data_h" in got and "total_h" in got
    # A pure observer: every array is BITWISE what it was without the readout.
    assert np.array_equal(on.env.x, off.env.x)
    assert np.array_equal(on.residual, off.residual)
    assert got["total"] == off.iterations[-1]["objective"]["total"]
    assert got["h_cells"] > 0


# ---------------------------------------------------------------------------
# the mechanism


@pytest.mark.parametrize("rates,k_hi", [((50.0,), 8), ((50.0, 61.0), 8)])
def test_the_hump_is_explained_only_where_the_regions_sit(
    rates: tuple[float, ...], k_hi: int
) -> None:
    """The true rates explain the comb; rates whose regions miss it explain nothing.

    The floor is PINNED (one fit, both hypotheses), so the profiled data term is
    the identical number for the two and every difference below is the H term's.
    """
    y, r_true = fixture(rates, k_hi)
    r_shift = r_true + SHIFT
    k = track_k(rates, k_hi)
    psd = masked_smooth_psd(y, SR, r_true, k_hi, n_fft=N_FFT)
    true = score(y, r_true, psd, k)
    shift = score(y, r_shift, psd, k)

    # The two hypotheses open regions of the same SIZE — this is a test of WHERE
    # they sit and not of how much spectrum they claim.
    assert shift["h_cells"] == pytest.approx(true["h_cells"], rel=0.15)
    # (a) the humps are explained: the data term collapses.
    assert true["data_h"] < 0.35 * true["data"]
    # (b) nothing is explained: the misplaced regions sit on floor.
    assert shift["data_h"] > 0.70 * shift["data"]
    # (c) and that IS the discrimination. The profiled term cannot separate the
    # two at all at a pinned floor; the H-aware term separates them widely.
    assert shift["total"] == pytest.approx(true["total"], rel=1e-12)
    assert shift["total_h"] - true["total_h"] > 0.2 * true["data"]


def test_the_shifted_regions_really_do_miss_the_lines() -> None:
    # The premise of the test above, measured rather than assumed: a five rev/s
    # shift puts the regions off the line cores at every harmonic tested.
    _, r_true = fixture((50.0,), 8)
    assert line_overlap(r_true + SHIFT, r_true, 8) < 0.10
    assert line_overlap(r_true, r_true, 8) == pytest.approx(1.0)


@pytest.mark.parametrize("shift", [0.0, SHIFT])
def test_on_pure_floor_the_h_term_charges_almost_nothing(shift: float) -> None:
    """No comb at all: ``H`` is the clip bias of a noisy ``P~ - S`` and no more.

    ``P~`` is a smoothed periodogram, so half of its floor bins sit above ``S``
    and clipping at zero leaves a small POSITIVE nuisance whatever the carrier
    is. It has to stay small, or the term would reward coverage by itself —
    which is the failure it exists to remove.
    """
    y, r_true = fixture((50.0,), 8, comb=False)
    r = r_true + shift
    k = track_k((50.0,), 8)
    psd = masked_smooth_psd(y, SR, r_true, 8, n_fft=N_FFT)
    got = score(y, r, psd, k)

    assert got["h_cells"] > 0, "the regions must exist, or the tolerance means nothing"
    assert abs(got["data_h"] - got["data"]) < 0.02 * got["data"]
