"""Backend equivalence for the tracking demodulation transforms.

The torch path of :mod:`tracking.demod_backend` is a *different summation
order* of the same complex64 zoom-IFFT, so every check here is a tolerance
check at the transform's own precision (~1e-6 relative), never a bit
comparison — the bit-identity guard is the scipy path against the frozen
reference (``scripts/tracking_ref.py --compare --exact``).

Everything runs on CPU on a 2 s synthetic comb, so the whole module is a
few seconds.
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

from tracking.demod_backend import (  # noqa: E402
    band_bins,
    demod_backend,
    demod_comb,
    padded_n_env,
    resolve,
    zoom_bands,
)
from tracking.phase_increment_tracker import _demod_bank, pi_kalman_refine  # noqa: E402
from tracking.vk_tracking import (  # noqa: E402
    VKConfig,
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
    assert resolve() == ("scipy", "cpu", "exact")
    with demod_backend(backend="torch", device="cpu", pad="fast") as sel:
        assert sel == ("torch", "cpu", "fast")
        assert resolve() == ("torch", "cpu", "fast")
        assert resolve(backend="scipy")[0] == "scipy"  # explicit argument wins
    assert resolve() == ("scipy", "cpu", "exact")


def test_resolve_rejects_unknown() -> None:
    with pytest.raises(ValueError), demod_backend(backend="cupy"):
        pass
    with pytest.raises(ValueError), demod_backend(pad="ish"):
        pass


# ---------------------------------------------------------------------------
# the kernel


def test_zoom_bands_scalar_band() -> None:
    y, _, phi = _clip()
    x = (y * np.exp(-1j * phi)).astype(np.complex64)
    n_env = N_T // STRIDE
    with demod_backend(backend="scipy"):
        ref, probe = zoom_bands(x, STRIDE, n_env, 6.0 / SR)
    assert probe is None
    with demod_backend(backend="torch", device="cpu"):
        got, _ = zoom_bands(x, STRIDE, n_env, 6.0 / SR)
    assert got.shape == ref.shape == (y.shape[0], n_env)
    assert got.dtype == np.complex64
    assert _rel(ref, got) < 1e-5


def test_zoom_bands_per_row_band_and_probe() -> None:
    y, _, phi = _clip()
    x = np.repeat((y * np.exp(-1j * phi)).astype(np.complex64)[:, None, :], 3, axis=1)
    n_env = N_T // STRIDE
    bands = np.array([6.0, 8.0, 10.0]) / SR
    shifts = np.array([11.0, -12.0, 13.0]) / SR
    with demod_backend(backend="scipy"):
        ref, ref_p = zoom_bands(x, STRIDE, n_env, bands, shifts)
    with demod_backend(backend="torch", device="cpu"):
        got, got_p = zoom_bands(x, STRIDE, n_env, bands, shifts)
    assert ref_p is not None and got_p is not None
    assert _rel(ref, got) < 1e-5
    assert _rel(ref_p, got_p) < 1e-5
    # A wider band keeps strictly more energy — the per-row cutoff is live.
    assert np.abs(ref[:, 2]).sum() > np.abs(ref[:, 0]).sum()


def test_zoom_bands_1d_input() -> None:
    y, _, phi = _clip(n_ch=1)
    x = (y[0] * np.exp(-1j * phi)).astype(np.complex64)
    n_env = N_T // STRIDE
    with demod_backend(backend="scipy"):
        ref, _ = zoom_bands(x, STRIDE, n_env, 6.0 / SR)
    with demod_backend(backend="torch", device="cpu"):
        got, _ = zoom_bands(x, STRIDE, n_env, 6.0 / SR)
    assert ref.shape == got.shape == (n_env,)
    assert _rel(ref, got) < 1e-5


def test_band_env_matches_band_cyc() -> None:
    """The two band parameterizations pick the same bins for the VK cutoff."""
    for stride in (16, 64, 160, 256):
        for n_env in (125, 1000, 1600):
            n_pad = stride * n_env
            by_env = band_bins(None, 0.45, n_pad, n_env)
            by_cyc = band_bins(0.45 / stride, None, n_pad, n_env)
            assert int(by_env[0]) == int(by_cyc[0]), (stride, n_env)


# ---------------------------------------------------------------------------
# the fused comb


def test_demod_comb_matches_demod_bank() -> None:
    y, _, phi = _clip()
    n_env = N_T // STRIDE
    t = np.arange(N_T) / SR
    bands = np.full(len(KS), 6.0 / SR)
    offs = np.full(len(KS), 11.0)
    with demod_backend(backend="scipy"):
        on_ref, off_ref = _demod_bank(y, phi, t, KS, 11.0, STRIDE, n_env, 6.0 / SR, bands, offs, SR)
    with demod_backend(backend="torch", device="cpu"):
        on_t, off_t = _demod_bank(y, phi, t, KS, 11.0, STRIDE, n_env, 6.0 / SR, bands, offs, SR)
    assert on_t.shape == on_ref.shape == (y.shape[0], len(KS), n_env)
    assert _rel(on_ref, on_t) < 1e-5
    assert _rel(off_ref, off_t) < 1e-5


def test_demod_comb_chunking_is_transparent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-harmonic flush must give the same bank as an all-in-one flush."""
    y, _, phi = _clip()
    n_env = N_T // STRIDE
    c1 = np.exp(-1j * phi).astype(np.complex64)[None, :]
    ks = np.asarray(KS, dtype=np.int64)
    rot = np.zeros(len(KS), dtype=np.int64)
    with demod_backend(backend="torch", device="cpu"):
        big, _ = demod_comb(y, c1, rot, ks, STRIDE, n_env, 6.0 / SR)
        monkeypatch.setenv("TRACKING_TORCH_BUDGET_MB", "1")
        small, _ = demod_comb(y, c1, rot, ks, STRIDE, n_env, 6.0 / SR)
    assert np.array_equal(big, small)


# ---------------------------------------------------------------------------
# through the public entry points


def test_vk_envelopes_backend_equivalence() -> None:
    y, r, _ = _clip(n_ch=2)
    cfg = VKConfig(fs=SR, fs_env=SR / STRIDE, bw_hz=1.0, k_max=8, f_min=20.0, f_max=1200.0)
    r_aud = np.vstack([r, r + 1.7])
    with demod_backend(backend="scipy"):
        ref = vk_envelopes(y.astype(np.float64), r_aud, cfg)
    with demod_backend(backend="torch", device="cpu"):
        got = vk_envelopes(y.astype(np.float64), r_aud, cfg)
    assert np.array_equal(ref.valid, got.valid)
    assert _rel(ref.z, got.z) < 1e-5
    assert _rel(ref.x, got.x) < 1e-4


def test_ls_project_backend_equivalence() -> None:
    """The peel has no transform in it, but it has two cores — and the torch
    one sums each block in a different order, so the gains must still agree."""
    y, r, _ = _clip(n_ch=2)
    cfg = VKConfig(fs=SR, fs_env=SR / STRIDE, bw_hz=1.0, k_max=8, f_min=20.0, f_max=1200.0)
    r_aud = np.vstack([r, r + 1.7])
    y64 = y.astype(np.float64)
    with demod_backend(backend="scipy"):
        env = vk_envelopes(y64, r_aud, cfg)
        # Mis-scale and mis-phase, so the fitted gains are far from 1 and a
        # disagreement between the cores would show.
        env = replace(env, x=env.x * (1.6 * np.exp(1j * 2.0)))
        ref, d_ref = ls_project_envelopes(y64, env)
    with demod_backend(backend="torch", device="cpu"):
        got, d_got = ls_project_envelopes(y64, env)
    assert _rel(ref.x, got.x) < 1e-6
    assert d_got["n_tracks_fitted"] == d_ref["n_tracks_fitted"]
    assert d_got["clipped_frac"] == pytest.approx(d_ref["clipped_frac"], abs=1e-4)
    assert d_got["e_resid_ratio"] == pytest.approx(d_ref["e_resid_ratio"], rel=1e-4)


def test_pi_kalman_backend_equivalence() -> None:
    y, r, _ = _clip(n_ch=2)
    ft = np.arange(0.0, N_T / SR - 0.02, 0.032)
    r0 = np.interp(ft, np.arange(N_T) / SR, r)[None, :] + 0.4

    def _run() -> np.ndarray:
        return pi_kalman_refine(
            y.astype(np.float64),
            r0,
            ft,
            sr=int(SR),
            n_iter=2,
            fs_env=SR / STRIDE,
            k_max=10,
            f_max=1200.0,
            k_caps=(6, 10),
        )[0]

    with demod_backend(backend="scipy"):
        ref = _run()
    with demod_backend(backend="torch", device="cpu"):
        got = _run()
    # Well below the 0.2 rev/s honest floor of the tracker.
    assert float(np.abs(ref - got).max()) < 1e-4


# ---------------------------------------------------------------------------
# smooth-length padding


def test_padded_n_env_grows_to_a_fast_length() -> None:
    assert padded_n_env(1000, "exact") == 1000
    assert padded_n_env(1000, "fast") == 1000  # already 5-smooth
    n_bad = 1009  # prime
    assert padded_n_env(n_bad, "fast") > n_bad
    assert padded_n_env(n_bad, "exact") == n_bad


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
    with demod_backend(backend="scipy", pad="exact"):
        ref, _ = zoom_bands(x, stride, n_env, 6.0 / SR)
    with demod_backend(backend="scipy", pad="fast"):
        got, _ = zoom_bands(x, stride, n_env, 6.0 / SR)
    assert got.shape == ref.shape == (y.shape[0], n_env)
    assert np.isfinite(got).all()
    assert _rel(ref, got) > 1e-3  # NOT a bit-level reparameterization
    trim = n_env // 5
    assert _rel(ref[:, trim:-trim], got[:, trim:-trim]) < 0.2


def test_fast_pad_backends_agree() -> None:
    """The two backends stay equivalent under smooth padding."""
    y, _, phi = _clip()
    n_env = 101
    stride = N_T // n_env
    x = (y * np.exp(-1j * phi)).astype(np.complex64)[:, : stride * n_env]
    with demod_backend(backend="scipy", pad="fast"):
        ref, _ = zoom_bands(x, stride, n_env, 6.0 / SR)
    with demod_backend(backend="torch", device="cpu", pad="fast"):
        got, _ = zoom_bands(x, stride, n_env, 6.0 / SR)
    assert _rel(ref, got) < 1e-5
