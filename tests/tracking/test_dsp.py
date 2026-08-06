"""The tracking stack's DSP primitives — one implementation each.

:mod:`tracking.dsp` holds THE band-select kernel (:func:`zoom_bands`), THE
demodulation driver (:func:`demod`) and THE moving average (:func:`boxcar`).
There is no backend axis to compare any more, so what is pinned here is what
the consolidation could still have broken:

* the kernel's own identities — the probe is a bin shift of the same
  transform, a per-row band really is per row, a 1-D input keeps its shape
* the driver's two carrier sources (a phase matrix and the comb recursion)
  agree, and the flush size is a cache knob only
* the peel's tile size is a cache knob only

Everything runs on CPU on a 2 s synthetic comb, so the whole module is a few
seconds. The GPU leg is ``scripts/tracking_ref.py --self-check --device cuda``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tracking.dsp import (  # noqa: E402
    band_bins,
    boxcar,
    demod,
    dsp_config,
    padded_n_env,
    resolve,
    thread_pool,
    threads,
    zoom_bands,
)
from tracking.phase_increment_tracker import _demod_bank, pi_kalman_refine  # noqa: E402
from tracking.vk_tracking import (  # noqa: E402
    VKConfig,
    demodulate,
    ls_project_envelopes,
    vk_envelopes,
)

torch = pytest.importorskip("torch")

SR = 4000.0
STRIDE = 64
N_T = 8000
KS = list(range(1, 12))


def _clip(n_ch: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y32, r, phi)`` for a single slowly-varying rotor comb."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / SR
    r = 30.0 + 2.0 * np.sin(2.0 * np.pi * 0.3 * t)
    phi = 2.0 * np.pi * np.cumsum(r) / SR
    y = np.stack(
        [
            sum(np.cos(k * phi + 0.3 * k + ch) for k in KS) + 0.1 * rng.standard_normal(N_T)
            for ch in range(n_ch)
        ]
    ).astype(np.float32)
    return y, r, phi


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-30))


# ---------------------------------------------------------------------------
# selection


def test_resolve_defaults_and_override() -> None:
    assert resolve() == ("cpu", "exact")
    with dsp_config(device="cpu", pad="fast") as sel:
        assert sel == ("cpu", "fast")
        assert resolve() == ("cpu", "fast")
        assert resolve(pad="exact")[1] == "exact"  # explicit argument wins
    assert resolve() == ("cpu", "exact")


def test_resolve_rejects_unknown_pad() -> None:
    with pytest.raises(ValueError), dsp_config(pad="ish"):
        pass


def test_thread_pool_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one thread knob: the pool beats the env, the env beats the default."""
    from tracking.dsp import _cpu_budget

    budget = _cpu_budget()
    monkeypatch.delenv("TRACKING_FFT_WORKERS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert threads() == 1  # the Slurm-safe default is unchanged
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert threads() == min(2, budget)
    monkeypatch.setenv("TRACKING_FFT_WORKERS", "3")
    assert threads() == min(3, budget)
    monkeypatch.setenv("TRACKING_FFT_WORKERS", "auto")
    assert threads() == budget
    monkeypatch.delenv("TRACKING_FFT_WORKERS")
    with thread_pool(1):  # the in-process override beats every env var
        assert threads() == 1
        assert torch.get_num_threads() == 1
        with thread_pool(None):
            assert threads() == 1
    assert threads() == min(2, budget)
    with thread_pool(0):
        assert threads() == budget
    assert threads() == min(2, budget)


def test_threads_do_not_change_the_transform() -> None:
    """The thread count is a performance knob, never a numerical one."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096))).astype(np.complex64)
    with thread_pool(1):
        a, _ = zoom_bands(x, 64, 64, 6.0 / 16000)
    with thread_pool(4):
        b, _ = zoom_bands(x, 64, 64, 6.0 / 16000)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# the kernel


def test_zoom_bands_is_a_brickwall_lowpass() -> None:
    """The kept band is exactly the bins inside ``+-band_cyc``.

    Two tones, one inside the band and one far outside: the in-band tone
    survives at full amplitude on the decimated grid, the other is gone.
    """
    n_env = N_T // STRIDE
    t = np.arange(N_T) / SR
    inb = np.exp(2j * np.pi * 3.0 * t).astype(np.complex64)
    out = np.exp(2j * np.pi * 40.0 * t).astype(np.complex64)
    low, probe = zoom_bands((inb + out)[None, :], STRIDE, n_env, 6.0 / SR)
    assert probe is None
    assert low.shape == (1, n_env)
    assert low.dtype == np.complex64
    trim = slice(n_env // 5, -(n_env // 5))
    assert _rel(inb[::STRIDE][None, :][:, trim], low[:, trim]) < 5e-2
    assert float(np.abs(low[:, trim]).max()) < 1.1  # the far tone left nothing


def test_zoom_bands_per_row_band_and_probe() -> None:
    y, _, phi = _clip()
    x = np.repeat((y * np.exp(-1j * phi)).astype(np.complex64)[:, None, :], 3, axis=1)
    n_env = N_T // STRIDE
    bands = np.array([6.0, 8.0, 10.0]) / SR
    shifts = np.array([11.0, -12.0, 13.0]) / SR
    got, got_p = zoom_bands(x, STRIDE, n_env, bands, shifts)
    assert got_p is not None
    assert got.shape == got_p.shape == (y.shape[0], 3, n_env)
    # A wider band keeps strictly more energy — the per-row cutoff is live.
    assert np.abs(got[:, 2]).sum() > np.abs(got[:, 0]).sum()
    # Each row is that row's band around its own centre, so a per-row call
    # must agree with the same row asked for on its own.
    for a in range(3):
        one, one_p = zoom_bands(x[:, a], STRIDE, n_env, float(bands[a]), float(shifts[a]))
        assert one_p is not None
        assert _rel(got[:, a], one) < 1e-6
        assert _rel(got_p[:, a], one_p) < 1e-6


def test_zoom_bands_1d_input() -> None:
    y, _, phi = _clip(n_ch=1)
    x = (y[0] * np.exp(-1j * phi)).astype(np.complex64)
    n_env = N_T // STRIDE
    got, _ = zoom_bands(x, STRIDE, n_env, 6.0 / SR)
    row, _ = zoom_bands(x[None, :], STRIDE, n_env, 6.0 / SR)
    assert got.shape == (n_env,)
    assert np.array_equal(got, row[0])


def test_band_env_matches_band_cyc() -> None:
    """The two band parameterizations pick the same bins for the VK cutoff."""
    for stride in (16, 64, 160, 256):
        for n_env in (125, 1000, 1600):
            n_pad = stride * n_env
            by_env = band_bins(None, 0.45, n_pad, n_env)
            by_cyc = band_bins(0.45 / stride, None, n_pad, n_env)
            assert int(by_env[0]) == int(by_cyc[0]), (stride, n_env)


def test_vk_band_always_fits_the_decimated_nyquist() -> None:
    """Why the kernel needs no short-grid special case.

    ``floor(0.45 n) <= (n - 1) // 2`` for every ``n``, so the ``band_env =
    0.45`` band is inside the decimated Nyquist range on every grid and the
    zoom identity holds down to a handful of envelope samples.
    """
    for n in range(1, 64):
        assert int(np.floor(0.45 * n)) <= (n - 1) // 2


# ---------------------------------------------------------------------------
# the driver


def test_demod_carrier_sources_agree() -> None:
    """The comb recursion and an explicit phase matrix give the same bank."""
    y, _, phi = _clip()
    n_env = N_T // STRIDE
    comb, _ = demod(
        y,
        c1=np.exp(-1j * phi).astype(np.complex64)[None, :],
        rotor=np.zeros(len(KS), dtype=np.int64),
        k=np.asarray(KS, dtype=np.int64),
        stride=STRIDE,
        n_env=n_env,
        band_cyc=6.0 / SR,
    )
    explicit, _ = demod(
        y,
        phase=np.stack([k * phi for k in KS]),
        stride=STRIDE,
        n_env=n_env,
        band_cyc=6.0 / SR,
    )
    assert comb.shape == explicit.shape == (y.shape[0], len(KS), n_env)
    # The recursion drifts by ~k * eps(complex64) against a fresh exp.
    assert _rel(explicit, comb) < 1e-4


def test_demod_bank_is_the_one_rotor_naming() -> None:
    """``demod_bank`` is a call of ``demod``, not a second implementation."""
    y, _, phi = _clip()
    n_env = N_T // STRIDE
    t = np.arange(N_T) / SR
    bands = np.full(len(KS), 6.0 / SR)
    offs = np.full(len(KS), 11.0)
    on_b, off_b = _demod_bank(y, phi, t, KS, 11.0, STRIDE, n_env, 6.0 / SR, bands, offs, SR)
    on_d, off_d = demod(
        y,
        c1=np.exp(-1j * phi).astype(np.complex64)[None, :],
        rotor=np.zeros(len(KS), dtype=np.int64),
        k=np.asarray(KS, dtype=np.int64),
        stride=STRIDE,
        n_env=n_env,
        band_cyc=bands,
        shift_cyc=offs / SR,
    )
    assert off_b is not None and off_d is not None
    assert np.array_equal(on_b, on_d)
    assert np.array_equal(off_b, off_d)


def test_demod_chunking_is_transparent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-harmonic flush must give the same bank as an all-in-one flush."""
    y, _, phi = _clip()
    n_env = N_T // STRIDE
    c1 = np.exp(-1j * phi).astype(np.complex64)[None, :]
    ks = np.asarray(KS, dtype=np.int64)
    rot = np.zeros(len(KS), dtype=np.int64)
    monkeypatch.setenv("TRACKING_DEMOD_BUDGET_MB", "4096")
    big, _ = demod(y, c1=c1, rotor=rot, k=ks, stride=STRIDE, n_env=n_env, band_cyc=6.0 / SR)
    monkeypatch.setenv("TRACKING_DEMOD_BUDGET_MB", "1")
    small, _ = demod(y, c1=c1, rotor=rot, k=ks, stride=STRIDE, n_env=n_env, band_cyc=6.0 / SR)
    assert np.array_equal(big, small)


def test_demod_rejects_a_missing_carrier() -> None:
    y, _, _ = _clip(n_ch=1)
    with pytest.raises(ValueError):
        demod(y, stride=STRIDE, n_env=N_T // STRIDE, band_cyc=6.0 / SR)


# ---------------------------------------------------------------------------
# through the public entry points


def test_vk_envelopes_structured_demod_matches_the_general_one() -> None:
    """``vk_envelopes`` takes the comb path; ``demodulate`` the phase-matrix
    one. They are the same driver, so they must agree, and the solve on top
    of them must be deterministic."""
    y, r, _ = _clip(n_ch=2)
    cfg = VKConfig(fs=SR, fs_env=SR / STRIDE, bw_hz=1.0, k_max=8, f_min=20.0, f_max=1200.0)
    r_aud = np.vstack([r, r + 1.7])
    env = vk_envelopes(y.astype(np.float64), r_aud, cfg)
    again = vk_envelopes(y.astype(np.float64), r_aud, cfg)
    assert np.array_equal(env.z, again.z)
    assert np.array_equal(env.x, again.x)
    phase = env.k[:, None] * env.phase[env.rotor]
    assert _rel(env.z, demodulate(y.astype(np.float64), phase, cfg)) < 1e-4


def test_ls_project_removes_energy() -> None:
    """The peel's guarantee, on a mis-scaled and mis-phased envelope set: the
    sequential projection can only take energy out of the residual."""
    y, r, _ = _clip(n_ch=2)
    cfg = VKConfig(fs=SR, fs_env=SR / STRIDE, bw_hz=1.0, k_max=8, f_min=20.0, f_max=1200.0)
    y64 = y.astype(np.float64)
    env = vk_envelopes(y64, np.vstack([r, r + 1.7]), cfg)
    env = replace(env, x=env.x * (1.6 * np.exp(1j * 2.0)))
    got, diag = ls_project_envelopes(y64, env)
    assert diag["e_resid_ratio"] < 1.0
    assert diag["n_tracks_fitted"] > 0
    assert got.x.shape == env.x.shape
    again, diag2 = ls_project_envelopes(y64, env)
    assert np.array_equal(got.x, again.x)
    assert diag2 == diag


def test_ls_project_tiling_is_a_cache_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tile size must not change the fit: the blocks are independent."""
    import tracking.vk_tracking as vk

    y, r, _ = _clip(n_ch=2)
    cfg = VKConfig(fs=SR, fs_env=SR / STRIDE, bw_hz=1.0, k_max=8, f_min=20.0, f_max=1200.0)
    y64 = y.astype(np.float64)
    env = vk_envelopes(y64, np.vstack([r, r + 1.7]), cfg)
    monkeypatch.setattr(vk, "LS_TILE_BYTES", 1)  # one block per tile
    small, d_small = ls_project_envelopes(y64, env)
    monkeypatch.setattr(vk, "LS_TILE_BYTES", 1 << 30)  # the whole clip
    big, d_big = ls_project_envelopes(y64, env)
    assert _rel(big.x, small.x) < 1e-9
    assert d_big["clipped_frac"] == d_small["clipped_frac"]
    assert d_big["e_resid_ratio"] == pytest.approx(d_small["e_resid_ratio"], rel=1e-9)


def test_pi_kalman_refine_locks_on() -> None:
    y, r, _ = _clip(n_ch=2)
    ft = np.arange(0.0, N_T / SR - 0.02, 0.032)
    truth = np.interp(ft, np.arange(N_T) / SR, r)
    got = pi_kalman_refine(
        y.astype(np.float64),
        truth[None, :] + 0.4,
        ft,
        sr=int(SR),
        n_iter=2,
        fs_env=SR / STRIDE,
        k_max=10,
        f_max=1200.0,
        k_caps=(6, 10),
    )[0]
    assert float(np.abs(got[0] - truth).max()) < 0.1


# ---------------------------------------------------------------------------
# smooth-length padding


def test_padded_n_env_grows_to_a_fast_length() -> None:
    assert padded_n_env(1000, "exact") == 1000
    assert padded_n_env(1000, "fast") == 1000  # already 5-smooth
    n_bad = 1009  # prime
    assert padded_n_env(n_bad, "fast") > n_bad
    assert padded_n_env(n_bad, "exact") == n_bad
    for n in (1, 2, 7, 13, 97, 1009, 258304):
        m = padded_n_env(n, "fast")
        assert m >= n
        rest = m
        for f in (2, 3, 5):
            while rest % f == 0:
                rest //= f
        assert rest == 1


def test_fast_pad_is_an_approximation_not_a_reparameterization() -> None:
    """Smooth padding appends a zero tail, so the brickwall output really moves.

    The grid and the shape are preserved, the values are not: the extra
    zeros lengthen the circular convolution and the sinc ringing rides
    deep into the clip. This is why ``pad="fast"`` is opt-in — it buys a
    transform length, it does not come free.
    """
    y, _, phi = _clip()
    n_env = 101  # prime -> the padded length really differs
    stride = N_T // n_env
    x = (y * np.exp(-1j * phi)).astype(np.complex64)[:, : stride * n_env]
    with dsp_config(pad="exact"):
        ref, _ = zoom_bands(x, stride, n_env, 6.0 / SR)
    with dsp_config(pad="fast"):
        got, _ = zoom_bands(x, stride, n_env, 6.0 / SR)
    assert got.shape == ref.shape == (y.shape[0], n_env)
    assert np.isfinite(got).all()
    assert _rel(ref, got) > 1e-3  # NOT a bit-level reparameterization
    trim = n_env // 5
    assert _rel(ref[:, trim:-trim], got[:, trim:-trim]) < 0.2


# ---------------------------------------------------------------------------
# the one moving average


def test_boxcar_is_length_preserving_and_reflects() -> None:
    v = np.arange(10.0)
    assert np.array_equal(boxcar(v, 1), v)  # w <= 1 is the identity
    sm = boxcar(v, 3)
    assert sm.shape == v.shape
    # Reflect padding keeps a linear ramp a linear ramp everywhere but the
    # ends, where it holds the end value rather than pulling toward zero.
    assert np.allclose(sm[1:-1], v[1:-1])
    assert sm[0] > 0.0
    assert sm[-1] < v[-1]


def test_boxcar_smooths_the_last_axis_of_a_stack() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 64))
    sm = boxcar(x, 5)
    assert sm.shape == x.shape
    assert sm.std() < x.std()
    for i in range(3):
        assert np.allclose(sm[i], boxcar(x[i], 5))
