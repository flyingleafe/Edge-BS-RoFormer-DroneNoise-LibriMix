"""Device-agnostic backend for the tracking demodulation transforms.

Every hot transform of the tracker is the same kernel — the **zoom-IFFT
band select**: multiply the audio by a carrier, transform the whole clip,
keep a narrow band around one bin, inverse-transform at the decimated
length. :func:`tracking.vk_tracking._fft_lp_decimate` and
:func:`tracking.phase_increment_tracker._zoom_lp_decimate_bank` are two
parameterizations of it, and issue #16 measured that this kernel is ~55 %
of ``vk_envelopes`` and ~95 % of ``pi_kalman_refine``.

This module holds ONE implementation of that kernel per backend:

``"scipy"``
    The historical ``scipy.fft`` path, kept bit-identical (the frozen
    reference ``scripts/tracking_ref.py --compare --exact`` guards it).

``"torch"``
    A device-agnostic ``torch.fft`` path — the same arithmetic in
    complex64, on whatever ``device`` is selected. numpy goes in and numpy
    comes out at every seam, so no caller and no downstream numpy code
    changes. On CUDA the transform is cuFFT and the *carriers* are built on
    the device too (:func:`demod_comb`), so a whole rotor pass ships one
    ``(C, T)`` float32 clip instead of a ``(C, K, T)`` complex64 bank.

Both backends share the band arithmetic, so switching them changes only the
floating-point summation order of the transform, never the band the tracker
keeps.

Selection, first hit wins: the :func:`demod_backend` context manager, then
the environment (``TRACKING_BACKEND``, ``TRACKING_DEVICE``,
``TRACKING_PAD``), then the defaults ``("scipy", "cpu", "exact")``. The
default backend stays ``scipy`` because it is the measured CPU winner and
the bit-identical one; ``torch`` is the opt-in that a GPU (or a
torch-with-MKL build) earns.

Smooth-length padding (``pad="fast"``)
--------------------------------------
The transform length is ``n_pad = stride * n_env``. When ``n_pad`` carries
a large prime factor both pocketfft and cuFFT fall back to Bluestein and
cost an order of magnitude more. ``pad="fast"`` grows the envelope grid to
``scipy.fft.next_fast_len(n_env)`` (and truncates the result back), which
removes a bad factor **of n_env**. It cannot remove a bad factor of
``stride`` — ``n_pad`` is a multiple of ``stride`` by construction, so a
stride like ``round(44100 / 62.5) = 706 = 2 * 353`` poisons every admissible
length. The fix there is a smooth stride (an ``fs_env`` whose stride
factorizes), not padding. Because it moves the bin grid, ``pad="fast"`` is
NOT bit-identical and is off by default.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np

__all__ = [
    "BACKENDS",
    "PAD_MODES",
    "demod_backend",
    "demod_comb",
    "resolve",
    "torch_device",
    "zoom_bands",
]

BACKENDS = ("scipy", "torch")
PAD_MODES = ("exact", "fast")

#: Working set of one torch flush (bytes). The device holds the carrier
#: buffer, the transform's padded copy and its output at once, so the budget
#: divides by ``_WORKSPACE * n_ch * n_pad * 8``. Override with
#: ``TRACKING_TORCH_BUDGET_MB``; the CUDA default is deliberately small
#: enough for a 16 GB T4 alongside the audio and the carriers.
TORCH_BUDGET_BYTES = 512e6
_WORKSPACE = 3

_OVERRIDE: dict[str, str | None] = {"backend": None, "device": None, "pad": None}


# ---------------------------------------------------------------------------
# selection


def _env(name: str, choices: tuple[str, ...] | None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    val = raw.strip().lower()
    if choices is not None and val not in choices:
        raise ValueError(f"{name}={raw!r} is not one of {choices}")
    return val


def resolve(
    backend: str | None = None, device: str | None = None, pad: str | None = None
) -> tuple[str, str, str]:
    """The active ``(backend, device, pad)``; explicit arguments win."""
    b = backend or _OVERRIDE["backend"] or _env("TRACKING_BACKEND", BACKENDS) or "scipy"
    d = device or _OVERRIDE["device"] or _env("TRACKING_DEVICE", None) or "cpu"
    p = pad or _OVERRIDE["pad"] or _env("TRACKING_PAD", PAD_MODES) or "exact"
    if b not in BACKENDS:
        raise ValueError(f"unknown backend {b!r} (expected one of {BACKENDS})")
    if p not in PAD_MODES:
        raise ValueError(f"unknown pad mode {p!r} (expected one of {PAD_MODES})")
    return b, d, p


@contextmanager
def demod_backend(
    backend: str | None = None, device: str | None = None, pad: str | None = None
) -> Iterator[tuple[str, str, str]]:
    """Run the block with an explicit backend/device/pad (``None`` = no change).

    Process-wide (the tracking stack is single-threaded numpy/scipy) and
    restored on exit, exactly like
    :func:`tracking.vk_tracking.fft_worker_pool`.
    """
    prev = dict(_OVERRIDE)
    for key, val in (("backend", backend), ("device", device), ("pad", pad)):
        if val is not None:
            _OVERRIDE[key] = str(val).strip().lower()
    try:
        yield resolve()
    finally:
        _OVERRIDE.update(prev)


def torch_device(device: str | None = None) -> Any:
    """``torch.device`` for the active selection (imports torch lazily)."""
    import torch

    return torch.device(resolve(device=device)[1])


# ---------------------------------------------------------------------------
# band arithmetic (shared by both backends)


def padded_n_env(n_env: int, pad: str) -> int:
    """Transform-side envelope length: ``n_env`` (exact) or the next fast length."""
    if pad == "exact":
        return int(n_env)
    from scipy import fft as sfft

    return int(cast(Any, sfft.next_fast_len)(int(n_env)))


def band_bins(
    band_cyc: float | np.ndarray | None,
    band_env: float | None,
    n_pad: int,
    n_envp: int,
) -> np.ndarray:
    """Half-band in bins of the padded transform, clipped to the envelope Nyquist.

    ``band_env`` (a fraction of the *envelope* grid, the ``vk_tracking``
    parameterization) takes precedence over ``band_cyc`` (cycles/sample at
    the *audio* rate, the ``phase_increment_tracker`` one) — the two agree
    up to one bin of floating-point rounding, and each caller keeps the form
    its historical code used so the exact path stays bit-identical.
    """
    b_max = (n_envp - 1) // 2
    if band_env is not None:
        return np.array([min(int(np.floor(float(band_env) * n_envp)), b_max)], dtype=np.int64)
    bc = np.asarray(band_cyc, dtype=np.float64).reshape(-1)
    return np.minimum(np.floor(bc * n_pad).astype(np.int64), b_max)


def shift_bins(shift_cyc: float | np.ndarray | None, n_pad: int) -> np.ndarray | None:
    """Probe offset in bins (snapped to the grid), or ``None`` for no probe.

    A constant frequency offset is a pure bin shift, which is why the
    off-comb noise probe needs no transform of its own.
    """
    if shift_cyc is None:
        return None
    return np.rint(np.asarray(shift_cyc, dtype=np.float64).reshape(-1) * n_pad).astype(np.int64)


def _pick(arr: np.ndarray, a: int) -> int:
    return int(arr[0] if arr.size == 1 else arr[a])


# ---------------------------------------------------------------------------
# the scipy path


def _copy_band(dst: np.ndarray, spec: np.ndarray, b: int, shift: int) -> None:
    """Fill ``dst`` (last axis ``n_envp``) with the ``+-b`` bins of ``spec``
    centered on bin ``shift`` (``0`` = the band around DC). Indices wrap."""
    if shift == 0:
        dst[..., : b + 1] = spec[..., : b + 1]
        if b > 0:
            dst[..., -b:] = spec[..., -b:]
        return
    dst[..., : b + 1] = spec.take(np.arange(shift, shift + b + 1), axis=-1, mode="wrap")
    if b > 0:
        dst[..., -b:] = spec.take(np.arange(shift - b, shift), axis=-1, mode="wrap")


def _zoom_scipy(
    x: np.ndarray,
    stride: int,
    n_env: int,
    bb: np.ndarray,
    sb: np.ndarray | None,
    n_envp: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    from scipy import fft as sfft

    n_pad = stride * n_envp
    xc = np.asarray(x, dtype=np.complex64)
    spec = cast(np.ndarray, sfft.fft(xc, n=n_pad, axis=-1, workers=workers))
    shape = x.shape[:-1] + (n_envp,)
    low = np.zeros(shape, dtype=np.complex64)
    probe = None if sb is None else np.zeros(shape, dtype=np.complex64)
    if bb.size == 1 and (sb is None or sb.size == 1):
        _copy_band(low, spec, int(bb[0]), 0)
        if probe is not None and sb is not None:
            _copy_band(probe, spec, int(bb[0]), int(sb[0]))
    else:
        for a in range(shape[-2]):
            b = _pick(bb, a)
            _copy_band(low[..., a, :], spec[..., a, :], b, 0)
            if probe is not None and sb is not None:
                _copy_band(probe[..., a, :], spec[..., a, :], b, _pick(sb, a))

    def _inv(band: np.ndarray) -> np.ndarray:
        dec = cast(np.ndarray, sfft.ifft(band, axis=-1, workers=workers))
        out = dec / np.complex64(stride)  # complex64 in, complex64 out
        return out if n_envp == n_env else out[..., :n_env]

    return _inv(low), (None if probe is None else _inv(probe))


# ---------------------------------------------------------------------------
# the torch path


def _band_gather(spec: Any, bb: np.ndarray, sb: np.ndarray | None, n_envp: int, rows: int) -> Any:
    """Gather the ``+-b`` band (centered on ``shift``) out of ``spec``.

    One ``torch.gather`` with a precomputed ``(rows, n_envp)`` wrapped index
    plus a keep mask — the device form of :func:`_copy_band`, and the only
    place where the two backends could disagree about *which* bins survive
    (they do not: both call :func:`band_bins` / :func:`shift_bins`).
    """
    import torch

    n_pad = spec.shape[-1]
    idx = np.zeros((rows, n_envp), dtype=np.int64)
    keep = np.zeros((rows, n_envp), dtype=bool)
    for a in range(rows):
        b = _pick(bb, a)
        sh = 0 if sb is None else _pick(sb, a)
        idx[a, : b + 1] = np.arange(sh, sh + b + 1) % n_pad
        keep[a, : b + 1] = True
        if b > 0:
            idx[a, n_envp - b :] = np.arange(sh - b, sh) % n_pad
            keep[a, n_envp - b :] = True
    dev = spec.device
    shape = (1,) * (spec.ndim - 2) + (rows, n_envp)
    it = torch.from_numpy(idx).to(dev).reshape(shape).expand(*spec.shape[:-1], n_envp)
    kt = torch.from_numpy(keep).to(dev).reshape(shape)
    return spec.gather(-1, it) * kt


def _zoom_torch(
    x: Any,
    stride: int,
    n_env: int,
    bb: np.ndarray,
    sb: np.ndarray | None,
    n_envp: int,
    device: str,
) -> tuple[Any, Any | None]:
    """Zoom-IFFT band select of a device tensor; returns device tensors."""
    import torch

    n_pad = stride * n_envp
    rows = x.shape[-2] if x.ndim >= 2 else 1
    if bb.size == 1 and (sb is None or sb.size == 1):
        rows = 1
    spec = torch.fft.fft(x, n=n_pad, dim=-1)
    if rows == 1 and x.ndim < 2:  # (T,) input: give the gather a rows axis
        spec = spec.reshape(1, n_pad)

    def _inv(band: Any) -> Any:
        # Divide by the real stride: complex division by a real-valued
        # complex64 (what the scipy path writes) is componentwise anyway.
        dec = torch.fft.ifft(band, dim=-1).to(torch.complex64) / float(stride)
        return dec if n_envp == n_env else dec[..., :n_env]

    low = _inv(_band_gather(spec, bb, None, n_envp, rows))
    probe = None if sb is None else _inv(_band_gather(spec, bb, sb, n_envp, rows))
    if x.ndim < 2:
        low = low.reshape(-1)
        probe = None if probe is None else probe.reshape(-1)
    return low, probe


def _torch_budget() -> float:
    raw = os.environ.get("TRACKING_TORCH_BUDGET_MB")
    if raw:
        try:
            return max(1.0, float(raw)) * 1e6
        except ValueError:
            pass
    return TORCH_BUDGET_BYTES


def _torch_chunk(n_ch: int, n_pad: int) -> int:
    """Tracks per device flush under :func:`_torch_budget`."""
    return max(1, int(_torch_budget() / (_WORKSPACE * max(1, n_ch) * max(1, n_pad) * 8)))


# ---------------------------------------------------------------------------
# public kernels


def zoom_bands(
    x: np.ndarray,
    stride: int,
    n_env: int,
    band_cyc: float | np.ndarray | None = None,
    shift_cyc: float | np.ndarray | None = None,
    *,
    band_env: float | None = None,
    workers: int = 1,
    backend: str | None = None,
    device: str | None = None,
    pad: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Brickwall lowpass + decimate of ``x`` (complex, time last), plus a probe.

    Zero-pad to ``stride * n_env``, transform, keep the ``+-band`` bins around
    DC (``low``) and — when ``shift_cyc`` is given — the same width around a
    shifted center (``probe``), inverse-transform at the decimated length.

    ``band_cyc`` / ``shift_cyc`` are cycles/sample at the audio rate; either
    may be a ``(rows,)`` array addressing axis ``-2``. ``band_env`` gives the
    half-band as a fraction of the envelope grid instead (see
    :func:`band_bins`). Returns ``(low, probe | None)``, complex64 numpy,
    last axis ``n_env``.
    """
    bk, dev, pd = resolve(backend, device, pad)
    n_envp = padded_n_env(n_env, pd)
    n_pad = stride * n_envp
    bb = band_bins(band_cyc, band_env, n_pad, n_envp)
    sb = shift_bins(shift_cyc, n_pad)
    if bk == "scipy":
        return _zoom_scipy(x, stride, n_env, bb, sb, n_envp, workers)
    import torch

    xt = torch.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=np.complex64))).to(dev)
    low, probe = _zoom_torch(xt, stride, n_env, bb, sb, n_envp, dev)
    return low.cpu().numpy(), (None if probe is None else probe.cpu().numpy())


def demod_comb(
    y32: np.ndarray,
    c1: np.ndarray,
    rotor: np.ndarray,
    k: np.ndarray,
    stride: int,
    n_env: int,
    band_cyc: float | np.ndarray | None = None,
    shift_cyc: float | np.ndarray | None = None,
    *,
    band_env: float | None = None,
    device: str | None = None,
    pad: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fused carrier + demodulate + band select for a rotor-harmonic comb (torch).

    ``y32`` ``(C, T)`` float32 audio, ``c1`` ``(R, T)`` complex64 *conjugate*
    fundamental phasors (``exp(-i phi_r)``), ``rotor`` / ``k`` ``(M,)`` the
    track table. Returns ``(z_on, z_off | None)``, each ``(C, M, n_env)``
    complex64.

    The whole comb is built on the device: the clip and the ``R``
    fundamentals are the only host-to-device traffic, and every harmonic
    carrier comes from the same power recursion the numpy path uses
    (``c_k = c_{k-1} c_1``, complex64), so the two agree to the recursion's
    own ~``k * eps`` drift. Harmonics are flushed in chunks sized by
    :func:`_torch_chunk`, which is what keeps a ``(8, 40, 256000)`` complex64
    bank (655 MB) off a 16 GB card in one piece.
    """
    import torch

    _, dev, pd = resolve(device=device, pad=pad)
    n_envp = padded_n_env(n_env, pd)
    n_pad = stride * n_envp
    n_ch, n_t = y32.shape
    n_m = len(k)
    bb = band_bins(band_cyc, band_env, n_pad, n_envp)
    sb = shift_bins(shift_cyc, n_pad)

    yt = torch.from_numpy(np.ascontiguousarray(y32)).to(dev)
    c1t = torch.from_numpy(np.ascontiguousarray(np.asarray(c1, dtype=np.complex64))).to(dev)
    z_on = np.empty((n_ch, n_m, n_env), dtype=np.complex64)
    z_off = None if sb is None else np.empty_like(z_on)

    chunk = min(max(1, _torch_chunk(n_ch, n_pad)), max(1, n_m))
    buf = torch.empty((n_ch, chunk, n_t), dtype=torch.complex64, device=dev)
    order = sorted(range(n_m), key=lambda m: (int(rotor[m]), int(k[m])))
    cur: Any = None
    cur_rotor, cur_k = -1, 0
    idxs: list[int] = []

    def flush() -> None:
        rows = np.asarray(idxs)
        sub_b = bb if bb.size == 1 else bb[rows]
        sub_s = None if sb is None else (sb if sb.size == 1 else sb[rows])
        on, off = _zoom_torch(buf[:, : len(idxs)], stride, n_env, sub_b, sub_s, n_envp, dev)
        z_on[:, rows] = on.cpu().numpy()
        if z_off is not None and off is not None:
            z_off[:, rows] = off.cpu().numpy()
        idxs.clear()

    for m in order:
        rot, kk = int(rotor[m]), int(k[m])
        if rot != cur_rotor:
            cur = c1t[rot] if kk == 1 else c1t[rot] ** kk
            cur_rotor, cur_k = rot, kk
        elif kk != cur_k:
            if kk - cur_k > 2:  # rare gaps: one pow instead of many multiplies
                cur = cur * c1t[rot] ** (kk - cur_k)
            else:
                for _ in range(kk - cur_k):
                    cur = cur * c1t[rot]
            cur_k = kk
        torch.mul(yt, cur, out=buf[:, len(idxs)])
        idxs.append(m)
        if len(idxs) == chunk:
            flush()
    if idxs:
        flush()
    return z_on, z_off
