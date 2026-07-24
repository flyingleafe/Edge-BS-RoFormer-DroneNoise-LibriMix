"""Torch (GPU-capable) inference kernels for the coupled Vold-Kalman tracker.

Opt-in numeric backend for :mod:`data_processing.vk_tracking` (design
``docs/vk-order-tracking-design.md``; CPU fast paths in §8). Selected via
``VKConfig(backend="torch", device=..., torch_dtype=...)`` — orchestration,
annealing and config stay in the numpy driver; only the three hot numeric
kernels are routed here, each with a numpy-in / numpy-out signature so the
driver's state never changes representation:

1. :func:`demod_tracks` — batched demodulation: per-track conj-phasor
   multiply + FFT brickwall lowpass + decimate, mirroring ``lp_mode="fft"``
   (:func:`vk_tracking._fft_lp_decimate`) exactly: complex64 FFT stage,
   float64 phases (they reach ~1e7 rad), identical bin selection, complex128
   output. :func:`demod_cross` is the pair-phasor counterpart for the
   coupled-group cross terms.
2. :func:`solve_group` — the coupled-group Hermitian PD normal equations.
   torch has no banded Cholesky, so the time-major banded system of
   ``_solve_group_banded`` (same-time coupling at offsets ``1..g-1``, p=2
   prior at offsets ``g`` and ``2g``) is reformulated *block-tridiagonally*:
   states are taken in time-pairs (block size ``2g``), which absorbs the
   ``t-2`` prior band into the adjacent super-block — strictly
   block-tridiagonal by construction — and solved by block-Thomas (sequential
   over time-pairs, dense ``torch.linalg.cholesky`` / ``cholesky_solve`` per
   step, all channels solved at once). Numerically non-PD systems raise
   ``numpy.linalg.LinAlgError`` so the driver's splu fallback engages,
   exactly like the scipy banded path.
3. :func:`freq_update` — the Fisher-weighted phase-slope frequency update
   (periodogram no-comb gate, envelope-SNR shrinkage, weighted slope fusion)
   with all array reductions in torch. The closing ``(3, T_env)`` real SPD
   pentadiagonal smoothness solve stays on CPU scipy (``solveh_banded``): it
   is O(T_env) scalar-sequential work on a tiny system — a GPU launch per
   Thomas step would cost orders of magnitude more than the transfer of two
   length-``T_env`` vectors.

Precision: ``torch_dtype="complex128"`` (default) assembles and solves the
group systems in complex128, matching scipy; ``"complex64"`` runs the solve
in single precision (the demod FFT stage is complex64 in *both* backends, as
in the numpy fast path, and phases/frequency updates always stay float64).

Torch is imported lazily at module level — this module is only imported by
the driver when ``backend="torch"`` is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from scipy.linalg import solveh_banded

from .vk_tracking import _TINY, _second_diff, _stride, _vk_noise_bandwidth

if TYPE_CHECKING:
    from .vk_tracking import Envelopes, VKConfig

__all__ = ["demod_tracks", "demod_cross", "solve_group", "freq_update"]

# torch stubs do not re-export linalg.LinAlgError; it subclasses RuntimeError.
_TORCH_LINALG_ERROR: type[RuntimeError] = getattr(torch.linalg, "LinAlgError", RuntimeError)

_CHUNK_BYTES = 128e6  # bound on the complex64 audio-rate working set, as in
# vk_tracking.demodulate — one chunk of (C, m, T) phasor products at a time


def _device(cfg: VKConfig) -> torch.device:
    """Resolve ``cfg.device``: ``"auto"`` = cuda when available, else cpu."""
    if cfg.device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(cfg.device)


def _cdtype(cfg: VKConfig) -> torch.dtype:
    return torch.complex64 if cfg.torch_dtype == "complex64" else torch.complex128


def _fft_lp_decimate_t(x: torch.Tensor, stride: int, n_env: int) -> torch.Tensor:
    """Torch mirror of :func:`vk_tracking._fft_lp_decimate` (complex64 in,
    complex128 out): brickwall keep ``|f| <= 0.45 * fs_env`` via zoom-IFFT."""
    n_pad = stride * n_env
    spec = torch.fft.fft(x, n=n_pad, dim=-1)
    if n_env < 8:  # degenerate grids: exact-but-tiny full inverse transform
        f = torch.fft.fftfreq(n_pad, d=1.0, device=x.device)
        spec[..., f.abs() > 0.45 / stride] = 0.0
        full = torch.fft.ifft(spec, dim=-1)
        return full[..., ::stride].to(torch.complex128)
    b = min(int(np.floor(0.45 * n_env)), (n_env - 1) // 2)  # bins per side
    low = torch.zeros((*x.shape[:-1], n_env), dtype=torch.complex64, device=x.device)
    low[..., : b + 1] = spec[..., : b + 1]
    low[..., n_env - b :] = spec[..., n_pad - b :]
    dec = torch.fft.ifft(low, dim=-1)
    return (dec / stride).to(torch.complex128)


def demod_tracks(
    audio: np.ndarray,
    phase: np.ndarray,
    rotor: np.ndarray,
    k: np.ndarray,
    cfg: VKConfig,
) -> np.ndarray:
    """Torch counterpart of :func:`vk_tracking._demod_tracks_fft`.

    ``audio``: ``(C, T)`` or ``(T,)`` float64; ``phase``: ``(R, T)`` float64
    rotor fundamental phase. Returns ``(C, M, T_env)`` complex128. Per-track
    carriers are computed directly (``exp(-1j k phi)`` on float64 phases via
    ``torch.polar``, then complex64) instead of the CPU path's rotor-major
    recursion — same result up to complex64 rounding, better parallelism.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    stride, _ = _stride(cfg)
    n_ch, n_t = y.shape
    n_env = len(range(0, n_t, stride))
    n_tracks = len(rotor)
    out = np.empty((n_ch, n_tracks, n_env), dtype=np.complex128)
    if n_tracks == 0:
        return out
    dev = _device(cfg)
    y_t = torch.as_tensor(y, device=dev).to(torch.float32)
    ph_t = torch.as_tensor(phase, device=dev)  # (R, T) float64
    k_f = torch.as_tensor(np.asarray(k, dtype=np.float64), device=dev)
    rot_i = torch.as_tensor(np.asarray(rotor, dtype=np.int64), device=dev)
    chunk = max(1, int(_CHUNK_BYTES / (max(1, n_ch) * max(1, n_t) * 8)))
    for lo in range(0, n_tracks, chunk):
        hi = min(lo + chunk, n_tracks)
        ph_m = k_f[lo:hi, None] * ph_t[rot_i[lo:hi]]  # (m, T) float64
        carr = torch.polar(torch.ones_like(ph_m), -ph_m).to(torch.complex64)
        z = _fft_lp_decimate_t(y_t[:, None, :] * carr[None], stride, n_env)
        out[:, lo:hi] = z.cpu().numpy()
    return out


def demod_cross(
    phase: np.ndarray,
    rotor: np.ndarray,
    k: np.ndarray,
    pairs: list[tuple[int, int]],
    n_env: int,
    cfg: VKConfig,
) -> np.ndarray:
    """LP-decimated cross terms ``LP[conj(c_m) c_n]`` for track pairs.

    ``pairs``: global track index pairs ``(m, n)``. Returns ``(P, T_env)``
    complex128 — the torch counterpart of the pair-carrier loop in
    :func:`vk_tracking.vk_envelopes` (``lp_mode="fft"`` branch).
    """
    stride, _ = _stride(cfg)
    n_t = phase.shape[-1]
    out = np.empty((len(pairs), n_env), dtype=np.complex128)
    if not pairs:
        return out
    dev = _device(cfg)
    ph_t = torch.as_tensor(phase, device=dev)  # (R, T) float64
    k_arr = np.asarray(k, dtype=np.float64)
    rot_arr = np.asarray(rotor, dtype=np.int64)
    m_idx = np.array([m for m, _ in pairs], dtype=np.int64)
    n_idx = np.array([n for _, n in pairs], dtype=np.int64)
    chunk = max(1, int(_CHUNK_BYTES / (max(1, n_t) * 8)))
    for lo in range(0, len(pairs), chunk):
        hi = min(lo + chunk, len(pairs))
        k_m = torch.as_tensor(k_arr[m_idx[lo:hi], None], device=dev)
        k_n = torch.as_tensor(k_arr[n_idx[lo:hi], None], device=dev)
        dphi = k_n * ph_t[rot_arr[n_idx[lo:hi]]] - k_m * ph_t[rot_arr[m_idx[lo:hi]]]
        carr = torch.polar(torch.ones_like(dphi), dphi).to(torch.complex64)
        out[lo:hi] = _fft_lp_decimate_t(carr, stride, n_env).cpu().numpy()
    return out


def solve_group(
    d2td2_diags: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho: float,
    w: list[np.ndarray],
    cross: dict[tuple[int, int], np.ndarray],
    z_g: np.ndarray,
    cfg: VKConfig,
) -> np.ndarray:
    """Coupled-group solve as a block-tridiagonal Hermitian PD system.

    Same system as :func:`vk_tracking._solve_group_banded` (time-major
    interleave ``index = t * g + a``): dense-in-track same-time blocks
    ``D_t`` (validity + coupling), scalar-diagonal prior links ``E_t``
    (``t -> t+1``) and ``F_t`` (``t -> t+2``). Pairing consecutive time
    samples into ``2g`` super-states makes the system strictly
    block-tridiagonal — the ``t-2`` band lands inside the adjacent
    super-block — and it is solved by block-Thomas with a dense Cholesky per
    super-block (sequential over the ~``T_env/2`` time-pairs; all channels
    share each factorization). Odd ``T_env`` is padded with one decoupled
    identity state. Raises ``numpy.linalg.LinAlgError`` when a super-block
    Schur complement is numerically non-PD (driver falls back to splu).
    Returns ``(g, T_env, C)`` complex128.
    """
    g = len(w)
    n_ch, _, n_env = z_g.shape
    dev = _device(cfg)
    cdt = _cdtype(cfg)
    d0, d1, d2 = (torch.as_tensor(np.asarray(d, dtype=np.float64), device=dev) for d in d2td2_diags)
    rho2 = float(rho) ** 2
    w_t = torch.as_tensor(np.stack(w), device=dev)  # (g, T_env) float64
    ar = torch.arange(g, device=dev)

    # Same-time blocks D_t: diagonal = prior main diag + eps + validity mask,
    # off-diagonal = the LP'd cross-coupling terms (Hermitian).
    diag = (rho2 * d0)[:, None] + 1e-8 + w_t.T  # (T_env, g)
    blocks = torch.zeros((n_env, g, g), dtype=cdt, device=dev)
    blocks[:, ar, ar] = diag.to(cdt)
    for (a, b), g_mn in cross.items():  # a < b: upper triangle
        c_ab = (w_t[a] * w_t[b]) * torch.as_tensor(
            np.asarray(g_mn, dtype=np.complex128), device=dev
        )
        blocks[:, a, b] = c_ab.to(cdt)
        blocks[:, b, a] = c_ab.conj().to(cdt)

    # RHS 2 w z, time-major: (T_env, g, C).
    z_t = torch.as_tensor(z_g, device=dev)  # (C, g, T_env) complex128
    rhs = ((2.0 * w_t.T)[:, :, None] * z_t.permute(2, 1, 0)).to(cdt)

    # Pad to an even number of time samples with a decoupled identity state.
    n_pad = n_env + (n_env & 1)
    if n_pad != n_env:
        pad_block = torch.zeros((1, g, g), dtype=cdt, device=dev)
        pad_block[0, ar, ar] = 1.0
        blocks = torch.cat([blocks, pad_block], dim=0)
        rhs = torch.cat([rhs, torch.zeros((1, g, n_ch), dtype=cdt, device=dev)], dim=0)
    e_full = torch.zeros(max(n_pad - 1, 0), dtype=torch.float64, device=dev)
    e_full[: n_env - 1] = rho2 * d1
    f_full = torch.zeros(max(n_pad - 2, 0), dtype=torch.float64, device=dev)
    f_full[: n_env - 2] = rho2 * d2

    # Super-blocks over time-pairs: A_tau = [[D_2t, E_2t], [E_2t, D_2t+1]],
    # B_tau (tau -> tau+1) = [[F_2t, 0], [E_2t+1, F_2t+1]] (E/F real scalars).
    n2 = n_pad // 2
    a_sb = torch.zeros((n2, 2 * g, 2 * g), dtype=cdt, device=dev)
    a_sb[:, :g, :g] = blocks[0::2]
    a_sb[:, g:, g:] = blocks[1::2]
    e_in = e_full[0::2].to(cdt)  # (n2,) links inside each super-block
    a_sb[:, ar, g + ar] = e_in[:, None]
    a_sb[:, g + ar, ar] = e_in[:, None]
    if n2 > 1:
        b_sb = torch.zeros((n2 - 1, 2 * g, 2 * g), dtype=cdt, device=dev)
        b_sb[:, ar, ar] = f_full[0::2].to(cdt)[:, None]
        b_sb[:, g + ar, ar] = e_full[1::2].to(cdt)[:, None]
        b_sb[:, g + ar, g + ar] = f_full[1::2].to(cdt)[:, None]
    else:
        b_sb = torch.zeros((0, 2 * g, 2 * g), dtype=cdt, device=dev)

    y = rhs.reshape(n2, 2 * g, n_ch).clone()
    factors = torch.empty_like(a_sb)
    try:
        factors[0] = torch.linalg.cholesky(a_sb[0])
        for t in range(1, n2):
            b_prev = b_sb[t - 1]
            # One solve for both the Schur update and the RHS carry.
            sol = torch.cholesky_solve(torch.cat([b_prev, y[t - 1]], dim=1), factors[t - 1])
            bh = b_prev.mH
            y[t] = y[t] - bh @ sol[:, 2 * g :]
            factors[t] = torch.linalg.cholesky(a_sb[t] - bh @ sol[:, : 2 * g])
        x = torch.empty_like(y)
        x[n2 - 1] = torch.cholesky_solve(y[n2 - 1], factors[n2 - 1])
        for t in range(n2 - 2, -1, -1):
            x[t] = torch.cholesky_solve(y[t] - b_sb[t] @ x[t + 1], factors[t])
    except _TORCH_LINALG_ERROR as exc:  # numerically non-PD
        raise np.linalg.LinAlgError(str(exc)) from exc
    sol_np = x.reshape(n_pad, g, n_ch)[:n_env].permute(1, 0, 2).cpu().numpy()
    return np.ascontiguousarray(sol_np.astype(np.complex128))


def _median_np(t: torch.Tensor) -> torch.Tensor:
    """``np.median`` semantics along the last dim (mean of the two central
    order statistics for even lengths — ``torch.median`` takes the lower)."""
    s, _ = torch.sort(t, dim=-1)
    n = s.shape[-1]
    if n % 2:
        return s[..., n // 2]
    return 0.5 * (s[..., n // 2 - 1] + s[..., n // 2])


def freq_update(
    env: Envelopes,
    rotor_idx: int,
    lam: float,
    cfg: VKConfig,
    z_res: np.ndarray,
) -> np.ndarray | None:
    """Torch mirror of :func:`vk_tracking._freq_update` (same contract).

    Gate periodogram, envelope-SNR shrinkage and the Fisher-weighted slope
    fusion run as batched torch reductions; the closing pentadiagonal
    smoothness solve and grid interpolation stay on CPU (see module
    docstring). Float64/complex128 throughout — update precision is never
    downgraded by ``torch_dtype``.
    """
    sel = np.where(env.rotor == rotor_idx)[0]
    if len(sel) == 0:
        return None
    v_np = env.valid[sel]  # (m, T_env) bool
    if not v_np.any():
        return None
    dev = _device(cfg)
    x = torch.as_tensor(env.x[:, sel], device=dev)  # (C, m, T_env) complex128
    v_bool = torch.as_tensor(v_np, device=dev)
    v_f = v_bool.to(torch.float64)
    kf = torch.as_tensor(env.k[sel].astype(np.float64), device=dev)

    # No-comb gate: periodogram peak/median ratio over each track's demod band.
    n_env = env.z.shape[-1]
    window = torch.as_tensor(np.hanning(n_env), device=dev)
    z_sel = torch.as_tensor(env.z[:, sel], device=dev)  # (C, m, T_env)
    has_valid = v_bool.any(dim=1)  # (m,)
    pxx = torch.fft.fft(z_sel * (v_f * window)[None], dim=-1).abs() ** 2
    med = _median_np(pxx)  # (C, m)
    ratio = pxx.amax(dim=-1) / torch.clamp(med, min=_TINY)
    peak = float(ratio[:, has_valid].max()) if bool(has_valid.any()) else 0.0
    if peak < cfg.update_gate:
        return None

    # Per-track envelope SNR -> Wiener-style shrinkage weight.
    e_sig = ((0.5 * x).abs() ** 2 * v_f[None]).sum(dim=(0, 2))
    z_res_t = torch.as_tensor(z_res[:, sel], device=dev)
    e_res = (z_res_t.abs() ** 2 * v_f[None]).sum(dim=(0, 2))
    b_demod = 0.9 * env.fs_env
    nu_np = (
        np.array([_vk_noise_bandwidth(float(env.bw_track[m]), env.fs_env, cfg.p) for m in sel])
        / b_demod
    )
    nu = torch.as_tensor(nu_np, device=dev)
    snr = e_sig / torch.clamp(e_res * nu, min=_TINY)
    shrink = snr / (1.0 + snr)

    # Fisher-weighted phase slopes, fused over channels and harmonics.
    prod = x[..., 1:] * x[..., :-1].conj()  # (C, m, T_env - 1)
    vv = (v_bool[:, 1:] & v_bool[:, :-1]).to(torch.float64)[None]
    wgt = (kf[None, :, None] ** 2) * prod.abs() * vv * shrink[None, :, None]
    delta_hat = prod.angle() * (env.fs_env / (2.0 * np.pi)) / kf[None, :, None]
    num = (wgt * delta_hat).sum(dim=(0, 1))
    den = wgt.sum(dim=(0, 1))  # (T_env - 1,)
    pos = den > 0
    if not bool(pos.any()):
        return None

    scale = float(den[pos].mean())
    w_norm = (den / scale).cpu().numpy()
    fused = (num / torch.clamp(den, min=_TINY * scale)).cpu().numpy()

    # Tiny real SPD pentadiagonal smoothness solve — CPU scipy (see docstring).
    d2 = _second_diff(len(fused))
    d2td2 = (d2.T @ d2).tocsr()
    ab = np.zeros((3, len(fused)))
    ab[2] = w_norm + 1e-3 + lam * np.asarray(d2td2.diagonal(0))
    ab[1, 1:] = lam * np.asarray(d2td2.diagonal(1))
    ab[0, 2:] = lam * np.asarray(d2td2.diagonal(2))
    delta_mid = cast(np.ndarray, solveh_banded(ab, w_norm * fused, lower=False))
    t_mid = env.t_env[:-1] + 0.5 / env.fs_env
    return np.interp(env.t_env, t_mid, delta_mid)
