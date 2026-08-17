"""The TIME-ADAPTIVE floor: one profiled scale per (channel, frame).

Block C fits ONE smooth spectrum per time block — four blocks over a sixteen
second window — so the noise model is stationary inside four seconds. DREGON's
low band is not: the rotor wash under a gust rises by many decibels for a span
of frames, and every one of those frames then pays a Whittle misfit of ``P / S``
per cell that no comb hypothesis caused and no comb hypothesis can remove. Worse
for a MEASURE, the block floor and its rent both move with however a given
hypothesis's own solve happened to distribute that energy, so the term becomes a
lottery on a quantity the measure is not about.

``JointConfig.adaptive_floor`` gives the floor one profiled gain per (channel,
frame), ``S_eff = gamma(c, t) S``. Under the Whittle model ``P / S`` is Exp(1),
whose median is ``ln 2``, so the median ratio over frequency divided by ``ln 2``
is an unbiased scale on a floor that is already correct — and ``rent`` grows by
``n_freq log gamma``, which is what a loud frame is charged for being invoked.

Three claims, one file:

- **Flags off is bit for bit what it was.** The objective with both new flags off
  equals a plain Whittle sum reimplemented here, and the two new diagnostics do
  not appear.
- **A gust stops dominating the data term.** On a white residual with one span of
  frames at three times the power, the flag reads ``gamma`` about 3 in the gust,
  brings the whole window's data per cell back to about 1, and charges the rent
  the ``n_freq log 3`` per gusty frame that it should.
- **And the gust stops leaking into H.** With the H-aware term on, the nuisance
  the gust manufactures is what the adaptive floor removes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking.joint_decompose import SmoothPSD, frame_starts, map_objective

SR = 8000
SECONDS = 4.0
N_MIC = 2
N_FFT = 1024
HOP = N_FFT // 2
#: The gust span, in FRAMES of the readout's own grid. Frames overlap by half,
#: so the two frames on each side of the span are partly gusty; the smoother
#: (``P_SMOOTH_FRAMES``) blurs two more. Every tolerance below is loose enough
#: for both.
GUST_FRAMES = (20, 40)
#: Power ratio of the gust. Three, because the reading is then unambiguous: a
#: data term of 3 per cell and a floor gain of 3 are the same number.
GUST_POWER = 3.0
#: The floor level the fixture is built against, in the power-spectral-density
#: units of the readout (``1 / (sr * sum w^2)``). White noise of variance
#: ``sr * LEVEL`` has exactly this density in every bin.
LEVEL = 1.0


def flat_psd(n_ch: int = N_MIC, level: float = LEVEL) -> SmoothPSD:
    """A PINNED flat floor — one block, one level, no fit anywhere in the test.

    The whole point of the fixture is that the TRUE floor is known, so a data
    term that does not read 1 per cell is the estimator's doing and not the
    floor block's.
    """
    freq = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    return SmoothPSD(
        freq=freq,
        t_block=np.array([0.5 * SECONDS]),
        log_s=np.full((n_ch, 1, freq.size), np.log(level)),
    )


def gusty(seed: int = 0, power: float = GUST_POWER, span: tuple[int, int] = GUST_FRAMES):
    """White noise at the flat floor, with one contiguous span of frames louder."""
    n_t = int(round(SECONDS * SR))
    y = np.random.default_rng(seed).standard_normal((N_MIC, n_t)) * np.sqrt(SR * LEVEL)
    a, b = span[0] * HOP, (span[1] - 1) * HOP + N_FFT
    y[:, a:b] *= np.sqrt(float(power))
    return y


def whittle(y: np.ndarray, psd: SmoothPSD, frames: slice | None = None) -> tuple[float, float, int]:
    """``(data, rent, cells)`` of the plain Whittle pair — the local reference.

    Written out rather than called, because the point of the first test is that
    the shipped readout with both flags off IS this sum. It reads the same
    Hann normalization ``masked_smooth_psd`` fits on, which is what makes the
    flat floor above the true one.
    """
    st = frame_starts(int(y.shape[-1]), N_FFT, HOP)
    if frames is not None:
        st = st[frames]
    win = np.hanning(N_FFT)
    scale = 1.0 / (SR * float((win**2).sum()))
    seg = y[:, st[:, None] + np.arange(N_FFT)] * win
    power = np.abs(np.fft.rfft(seg, axis=-1)) ** 2 * scale
    s_lin = np.exp(np.asarray(psd.log_s)[:, 0, :])[:, None, :]
    return float(np.sum(power / s_lin)), float(np.sum(np.log(s_lin)) * st.size), int(power.size)


def score(y: np.ndarray, *, adaptive: bool, carrier: np.ndarray | None = None, k_hi: int = 1):
    """:func:`map_objective` at the pinned floor, with every prior held at zero."""
    n_env = 20
    k = np.arange(1.0, k_hi + 1)
    return map_objective(
        y,
        SR,
        flat_psd(int(y.shape[0])),
        x=np.zeros((int(y.shape[0]), k.size, n_env), dtype=np.complex128),
        k=k,
        bw_track=np.full(k.size, 3.0),
        theta=np.zeros((1, n_env)),
        psi=np.zeros((k.size, n_env)),
        fs_env=100.0,
        n_fft=N_FFT,
        adaptive_floor=adaptive,
        h_carrier=carrier,
    )


# ---------------------------------------------------------------------------
# the flags are inert when they are off


def test_both_flags_off_is_the_plain_whittle_sum() -> None:
    # The hard requirement of the two levers: switching neither on leaves the
    # readout the sum it always was, to the last bit the two summation orders
    # can agree on.
    y = gusty()
    psd = flat_psd()
    got = score(y, adaptive=False)
    data, rent, cells = whittle(y, psd)

    assert got["data"] == pytest.approx(data, rel=1e-12)
    assert got["rent"] == pytest.approx(rent, rel=1e-12)
    assert got["n_cells"] == cells
    assert got["total"] == pytest.approx(
        data + rent + got["phase_priors"] + got["envelope_prior"], rel=1e-12
    )
    # The diagnostics of the two levers exist only when they are asked for.
    assert "floor_gamma" not in got
    assert "h_fit" not in got


def test_the_adaptive_readout_adds_only_its_own_keys() -> None:
    y = gusty()
    off, on = score(y, adaptive=False), score(y, adaptive=True)
    assert set(on) - set(off) == {"floor_gamma"}
    # Nothing outside the noise model moves: the priors are the same arrays.
    for key in ("phase_priors", "envelope_prior", "theta_prior", "psi_prior", "n_cells"):
        assert on[key] == off[key]


# ---------------------------------------------------------------------------
# the mechanism


def test_a_gust_stops_dominating_the_data_term() -> None:
    """Three times the power for a span of frames, read three ways."""
    y = gusty()
    psd = flat_psd()
    off, on = score(y, adaptive=False), score(y, adaptive=True)
    n_f = int(off["n_freq"])
    n_gust = GUST_FRAMES[1] - GUST_FRAMES[0]

    # (a) with the flag off the gusty frames pay the full ratio, per cell.
    data_g, _, cells_g = whittle(y, psd, slice(*GUST_FRAMES))
    assert data_g / cells_g == pytest.approx(GUST_POWER, rel=0.1)
    # ... and the whole window is well above the 1 per cell a correct floor gives.
    assert off["data"] / off["n_cells"] > 1.5

    # (b) the gain reads the gust. The span is a sixth of the frames, so the
    # upper percentile of gamma sits inside it.
    gamma = on["floor_gamma"]
    assert gamma["p95"] == pytest.approx(GUST_POWER, rel=0.15)
    assert gamma["p50"] == pytest.approx(1.0, rel=0.15)

    # (c) and the data term comes back to one per cell over the whole window.
    assert on["data"] / on["n_cells"] == pytest.approx(1.0, rel=0.06)

    # (d) the rent pays for it: n_freq log gamma per gusty (channel, frame). The
    # boxcar smears the span's edges and the half-overlapped frames on each side
    # are partly gusty, so this is an order-of-magnitude statement and not an
    # identity.
    want = n_f * N_MIC * n_gust * float(np.log(GUST_POWER))
    assert (on["rent"] - off["rent"]) / want == pytest.approx(1.0, rel=0.35)


def test_the_gust_stops_leaking_into_the_h_nuisance() -> None:
    """With ``H`` on, a gust manufactures a nuisance the adaptive floor removes.

    ``H = max(0, P~ - S_eff)`` inside the comb regions. A gusty frame puts every
    region bin far above a floor fitted for the whole block, so the nuisance
    absorbs WIND — energy that has nothing to do with any comb — and the term
    then rewards whichever hypothesis happened to open regions over the gust.
    Under the adaptive floor ``S_eff`` follows the gust and the nuisance drops
    back to the clip bias of a noisy ``P~ - S``.
    """
    y = gusty(power=6.0, span=(15, 45))
    carrier = np.stack([np.full(int(y.shape[-1]), 80.0)])
    off = score(y, adaptive=False, carrier=carrier, k_hi=20)
    on = score(y, adaptive=True, carrier=carrier, k_hi=20)

    assert off["h_cells"] > 0, "the regions must exist, or the comparison means nothing"
    assert on["h_cells"] == off["h_cells"], "the regions are the carrier's, not the floor's"
    assert on["h_energy"] < 0.25 * off["h_energy"]
