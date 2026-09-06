"""The signal-processing primitives of the tracking stack — one each.

Every hot transform of the tracker is the same kernel — the **zoom-IFFT band
select**: multiply the audio by a carrier, transform the whole clip, keep a
narrow band around one bin, inverse-transform at the decimated length. Issue
#16 measured that this kernel is ~55 % of ``vk_envelopes`` and ~95 % of
``pi_kalman_refine``, and the 2026-08 consolidation reduced it to ONE
implementation:

:func:`zoom_bands`
    The kernel itself: complex signal in, band-selected + decimated signal
    out. Numpy at the seam, ``torch.fft`` inside, on whatever ``device`` is
    selected.
:func:`demod`
    The one demodulation driver: carrier synthesis (a per-track phase matrix
    *or* the harmonic power recursion of a rotor comb), the chunked flush that
    bounds the working set, and :func:`zoom_bands` on each flush.
:func:`boxcar`
    The one moving average.

There is no backend axis any more. The scipy/pocketfft path and the numpy peel
core were deleted in the consolidation: torch runs the same arithmetic on CPU
and on CUDA, so a second implementation bought a summation order and nothing
else. What is left is ``device`` (where the arithmetic runs) and ``pad`` (how
the transform length is chosen), selected, first hit wins, by the
:func:`dsp_config` context manager, then the environment (``TRACKING_DEVICE``,
``TRACKING_PAD``), then the defaults ``("cpu", "exact")``.

Thread count
------------
Torch owns its own CPU thread pool. :func:`threads` reads the historical
``TRACKING_FFT_WORKERS`` knob (then ``OMP_NUM_THREADS``, then 1) and
:func:`thread_pool` applies it for a block, so an offline caller still opts in
to threads exactly as it did and a Slurm allocation is not oversubscribed by
default.

Smooth-length padding (``pad="fast"``)
--------------------------------------
The transform length is ``n_pad = stride * n_env``. When ``n_pad`` carries a
large prime factor the FFT falls back to Bluestein and costs an order of
magnitude more. ``pad="fast"`` grows the envelope grid to the next 5-smooth
length (and truncates the result back), which removes a bad factor **of
n_env**. It cannot remove a bad factor of ``stride`` — ``n_pad`` is a multiple
of ``stride`` by construction, so a stride like ``round(44100 / 62.5) = 706 =
2 * 353`` poisons every admissible length. The fix there is a smooth stride (an
``fs_env`` whose stride factorizes), not padding. Because it moves the bin
grid, ``pad="fast"`` is NOT value-preserving and is off by default.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import numpy as np

__all__ = [
    "PAD_MODES",
    "band_bins",
    "boxcar",
    "demod",
    "dsp_config",
    "padded_n_env",
    "resolve",
    "shift_bins",
    "thread_pool",
    "threads",
    "torch_device",
    "zoom_bands",
]

PAD_MODES = ("exact", "fast")

#: Working set of one demodulation flush (bytes) on a CPU device. This is a
#: **cache** knob, not a memory-headroom knob: channels are already batched
#: jointly, so a bigger flush amortizes nothing and only leaves cache. The
#: 64 MB default was measured on the frozen 16 s clip (8 mics, ``n_pad`` =
#: 256000): the whole-bank flush costs ~1.7x the one-harmonic flush, and the
#: optimum sits at a ~50 MB working set on 1, 2 and 8 channels alike.
DEMOD_BUDGET_BYTES = 64e6

#: The same budget on a non-CPU device, where it stops being a cache knob and
#: becomes the device-memory bound (the full ``(8, 40, 256000)`` complex64
#: bank is 655 MB, which must not land on a 16 GB card in one piece).
DEVICE_BUDGET_BYTES = 512e6

#: Large arrays alive during one flush: the carrier buffer, the transform's
#: padded copy, and its output — all ``(n_ch, chunk, n_pad)`` complex64.
_WORKSPACE = 3

_OVERRIDE: dict[str, str | None] = {"device": None, "pad": None}
_THREADS_OVERRIDE: int | None = None


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


def resolve(device: str | None = None, pad: str | None = None) -> tuple[str, str]:
    """The active ``(device, pad)``; explicit arguments win."""
    d = device or _OVERRIDE["device"] or _env("TRACKING_DEVICE", None) or "cpu"
    p = pad or _OVERRIDE["pad"] or _env("TRACKING_PAD", PAD_MODES) or "exact"
    if p not in PAD_MODES:
        raise ValueError(f"unknown pad mode {p!r} (expected one of {PAD_MODES})")
    return d, p


@contextmanager
def dsp_config(device: str | None = None, pad: str | None = None) -> Iterator[tuple[str, str]]:
    """Run the block with an explicit device/pad (``None`` = no change).

    Process-wide (the tracking stack is single-threaded on the host side) and
    restored on exit, exactly like :func:`thread_pool`.
    """
    prev = dict(_OVERRIDE)
    for key, val in (("device", device), ("pad", pad)):
        if val is not None:
            _OVERRIDE[key] = str(val).strip().lower()
    try:
        yield resolve()
    finally:
        _OVERRIDE.update(prev)


def torch_device(device: str | None = None) -> Any:
    """``torch.device`` for the active selection (imports torch lazily)."""
    import torch

    return torch.device(resolve(device=device)[0])


def _cpu_budget() -> int:
    """CPUs actually available to this process (cgroup/affinity aware)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def threads() -> int:
    """Torch CPU threads, clamped to the CPUs available to this process.

    Precedence, first hit wins: the :func:`thread_pool` override, then
    ``TRACKING_FFT_WORKERS`` (``"auto"`` takes the whole budget — the name is
    historical, it is the tracking stack's one thread knob), then
    ``OMP_NUM_THREADS``, then 1. The default stays 1 on purpose:
    oversubscribing on a restricted Slurm allocation thrashes, so threads must
    be asked for.
    """
    avail = _cpu_budget()
    if _THREADS_OVERRIDE is not None:
        return max(1, min(_THREADS_OVERRIDE, avail))
    env = os.environ.get("TRACKING_FFT_WORKERS")
    if env is not None:
        if env.strip().lower() == "auto":
            return avail
        try:
            return max(1, min(int(env), avail))
        except ValueError:
            pass
    try:
        return max(1, min(int(os.environ.get("OMP_NUM_THREADS", "1")), avail))
    except ValueError:
        return 1


@contextmanager
def thread_pool(n: int | None) -> Iterator[int]:
    """Run the block with an explicit torch CPU thread count.

    ``None`` = no change (the environment still decides), ``n <= 0`` = the
    whole CPU budget. ``torch.set_num_threads`` is process-wide and cannot be
    scoped, so the previous count is restored on exit.
    """
    global _THREADS_OVERRIDE
    if n is None:
        yield threads()
        return
    prev = _THREADS_OVERRIDE
    _THREADS_OVERRIDE = _cpu_budget() if n <= 0 else int(n)
    want = threads()
    try:
        import torch

        had = torch.get_num_threads()
        torch.set_num_threads(want)
    except Exception:  # torch absent or already parallel-initialized
        had = None
    try:
        yield want
    finally:
        _THREADS_OVERRIDE = prev
        if had is not None:
            import torch

            torch.set_num_threads(had)


# ---------------------------------------------------------------------------
# band arithmetic


def padded_n_env(n_env: int, pad: str) -> int:
    """Transform-side envelope length: ``n_env`` (exact) or the next fast length."""
    if pad == "exact":
        return int(n_env)
    return _next_fast_len(int(n_env))


@lru_cache(maxsize=64)
def _next_fast_len(n: int) -> int:
    """Smallest 5-smooth integer ``>= n`` (the FFT-friendly lengths)."""
    if n <= 1:
        return max(1, n)
    best = 2 ** int(np.ceil(np.log2(n)))
    p5 = 1
    while p5 < best:
        p3 = p5
        while p3 < best:
            p2 = p3
            while p2 < n:
                p2 *= 2
            best = min(best, p2)
            p3 *= 3
        p5 *= 5
    return int(best)


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
    its historical code used.
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


def _collapse(arr: np.ndarray | None) -> np.ndarray | None:
    """A constant per-row band/shift is a SCALAR one.

    Callers naturally hand over one entry per track even when every entry is
    the same (``demod_bank``'s fixed probe offset is the common case), and the
    scalar form takes the cheap slice path instead of a per-row gather. The
    bins are identical either way — this only picks the kernel.
    """
    if arr is None or arr.size <= 1:
        return arr
    return arr[:1] if bool((arr == arr[0]).all()) else arr


# ---------------------------------------------------------------------------
# the kernel


@lru_cache(maxsize=32)
def _wrap_index(start: int, count: int, n_pad: int, dev: str) -> Any:
    """Cached ``(count,)`` int64 index ``[start, start + count) mod n_pad``.

    Building this on the host and shipping it dominates a small demodulation,
    and the tracker asks for the same few bands thousands of times per pass —
    hence the cache, keyed by the device string so a CUDA run keeps its own.
    """
    import torch

    idx = (np.arange(start, start + count, dtype=np.int64)) % n_pad
    return torch.from_numpy(idx).to(dev)


def _take_band(spec: Any, b: int, sh: int, n_envp: int, dev: str) -> Any:
    """The ``+-b`` bins of ``spec`` around bin ``sh``, laid out on an
    ``n_envp``-long DC-centred grid (positive frequencies first, negatives at
    the tail). Indices wrap; everything outside the band stays zero."""
    import torch

    n_pad = spec.shape[-1]
    out = torch.zeros(spec.shape[:-1] + (n_envp,), dtype=spec.dtype, device=spec.device)
    if sh == 0:  # contiguous by construction: two narrow copies, no index
        out[..., : b + 1] = spec[..., : b + 1]
        if b > 0:
            out[..., -b:] = spec[..., -b:]
        return out
    out[..., : b + 1] = spec.index_select(-1, _wrap_index(sh, b + 1, n_pad, dev))
    if b > 0:
        out[..., -b:] = spec.index_select(-1, _wrap_index(sh - b, b, n_pad, dev))
    return out


def _gather_bands(spec: Any, bb: np.ndarray, sb: np.ndarray | None, n_envp: int, rows: int) -> Any:
    """Per-row bands (each row its own half-width and shift), by one gather.

    The general form of :func:`_take_band`, used only when the bands really
    differ row by row (the ``k_scaled`` / ``clean``-probe paths).
    """
    n_pad = spec.shape[-1]
    idx, keep = _band_index(
        bb.tobytes(), None if sb is None else sb.tobytes(), n_envp, rows, n_pad, str(spec.device)
    )
    shape = (1,) * (spec.ndim - 2) + (rows, n_envp)
    it = idx.reshape(shape).expand(*spec.shape[:-1], n_envp)
    return spec.gather(-1, it) * keep.reshape(shape)


@lru_cache(maxsize=16)
def _band_index(
    bb_bytes: bytes, sb_bytes: bytes | None, n_envp: int, rows: int, n_pad: int, dev: str
) -> tuple[Any, Any]:
    import torch

    bb = np.frombuffer(bb_bytes, dtype=np.int64)
    sb = None if sb_bytes is None else np.frombuffer(sb_bytes, dtype=np.int64)
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
    return torch.from_numpy(idx).to(dev), torch.from_numpy(keep).to(dev)


def _zoom_torch(
    x: Any,
    stride: int,
    n_env: int,
    bb: np.ndarray,
    sb: np.ndarray | None,
    n_envp: int,
    dev: str,
) -> tuple[Any, Any | None]:
    """Zoom-IFFT band select of a device tensor; returns device tensors."""
    import torch

    n_pad = stride * n_envp
    uniform = bb.size == 1 and (sb is None or sb.size == 1)
    rows = 1 if uniform else (x.shape[-2] if x.ndim >= 2 else 1)
    spec = torch.fft.fft(x, n=n_pad, dim=-1)
    if not uniform and x.ndim < 2:  # (T,) input: give the gather a rows axis
        spec = spec.reshape(1, n_pad)

    def _inv(band: Any) -> Any:
        # Divide by the real stride: complex division by a real-valued
        # complex64 is componentwise anyway.
        dec = torch.fft.ifft(band, dim=-1).to(torch.complex64) / float(stride)
        return dec if n_envp == n_env else dec[..., :n_env]

    if uniform:
        b = int(bb[0])
        low = _inv(_take_band(spec, b, 0, n_envp, dev))
        probe = None if sb is None else _inv(_take_band(spec, b, int(sb[0]), n_envp, dev))
    else:
        low = _inv(_gather_bands(spec, bb, None, n_envp, rows))
        probe = None if sb is None else _inv(_gather_bands(spec, bb, sb, n_envp, rows))
        if x.ndim < 2:  # the gather's rows axis was borrowed, not asked for
            low = low.reshape(-1)
            probe = None if probe is None else probe.reshape(-1)
    return low, probe


def zoom_bands(
    x: np.ndarray,
    stride: int,
    n_env: int,
    band_cyc: float | np.ndarray | None = None,
    shift_cyc: float | np.ndarray | None = None,
    *,
    band_env: float | None = None,
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
    import torch

    dev, pd = resolve(device, pad)
    n_envp = padded_n_env(n_env, pd)
    n_pad = stride * n_envp
    bb = _collapse(band_bins(band_cyc, band_env, n_pad, n_envp))
    sb = _collapse(shift_bins(shift_cyc, n_pad))
    assert bb is not None
    xt = torch.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=np.complex64))).to(dev)
    low, probe = _zoom_torch(xt, stride, n_env, bb, sb, n_envp, dev)
    return low.cpu().numpy(), (None if probe is None else probe.cpu().numpy())


# ---------------------------------------------------------------------------
# the one demodulation driver


def demod_chunk(n_ch: int, n_pad: int, device: str = "cpu") -> int:
    """Tracks per flush under the working-set budget of :data:`DEMOD_BUDGET_BYTES`.

    The flush transforms ``(n_ch, chunk, n_pad)`` complex64 at once, so the
    budget divides by ``_WORKSPACE * n_ch * n_pad * 8`` bytes.
    ``TRACKING_DEMOD_BUDGET_MB`` overrides both defaults.
    """
    default = DEMOD_BUDGET_BYTES if device.split(":")[0] == "cpu" else DEVICE_BUDGET_BYTES
    raw = os.environ.get("TRACKING_DEMOD_BUDGET_MB")
    budget = default
    if raw:
        with contextlib.suppress(ValueError):
            budget = max(1.0, float(raw)) * 1e6
    return max(1, int(budget / (_WORKSPACE * max(1, n_ch) * max(1, n_pad) * 8)))


def demod(
    y: np.ndarray,
    *,
    phase: np.ndarray | None = None,
    c1: np.ndarray | None = None,
    rotor: np.ndarray | None = None,
    k: np.ndarray | None = None,
    stride: int,
    n_env: int,
    band_cyc: float | np.ndarray | None = None,
    shift_cyc: float | np.ndarray | None = None,
    band_env: float | None = None,
    device: str | None = None,
    pad: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """THE demodulation of the tracking stack: carrier, band select, decimate.

    ``y`` is ``(T,)`` or ``(C, T)`` real audio. The carrier is given one of
    two ways, and they are the only two the stack needs:

    ``phase`` ``(M, T)`` float64
        An arbitrary per-track instantaneous phase in radians; the carrier is
        ``exp(-i phase_m)``, one complex exponential per track.
    ``c1`` ``(R, T)`` complex64 + ``rotor`` / ``k`` ``(M,)``
        A rotor-harmonic COMB. ``c1`` holds the *conjugate* fundamental
        phasors ``exp(-i phi_r)`` and every harmonic carrier comes from the
        power recursion ``c_k = c_{k-1} c_1`` — one exp per rotor instead of
        one per track (an exp of a 1e7-rad float64 phase costs ~10x a
        complex64 multiply). The recursion's drift is ~``k * eps(c64)``.

    Returns ``(z_on, z_off | None)``, each ``(C, M, n_env)`` complex64:
    the demodulated audio brickwall-lowpassed to ``+-band`` and decimated by
    ``stride``, plus — when ``shift_cyc`` is given — the same band around a
    constant frequency offset (the off-comb noise probe), sliced out of the
    SAME forward transform because a constant offset is a pure bin shift.

    Everything runs on the selected device: the clip and the ``R``
    fundamentals (or one chunk of phase rows) are the only traffic across the
    seam, instead of a ``(C, chunk, T)`` complex64 buffer per flush.
    Harmonics are flushed in chunks sized by :func:`demod_chunk`.
    """
    import torch

    dev, pd = resolve(device, pad)
    n_envp = padded_n_env(n_env, pd)
    n_pad = stride * n_envp
    y32 = np.atleast_2d(np.asarray(y, dtype=np.float32))
    n_ch, n_t = y32.shape

    if phase is not None:
        ph = np.atleast_2d(np.asarray(phase, dtype=np.float64))
        n_m = ph.shape[0]
        if ph.shape[-1] != n_t:
            raise ValueError(f"phase length {ph.shape[-1]} != audio length {n_t}")
        order: list[int] = list(range(n_m))
    else:
        if c1 is None or rotor is None or k is None:
            raise ValueError("demod needs either phase= or (c1=, rotor=, k=)")
        c1a = np.atleast_2d(np.asarray(c1, dtype=np.complex64))
        rot_a = np.asarray(rotor, dtype=np.int64).reshape(-1)
        k_a = np.asarray(k, dtype=np.int64).reshape(-1)
        n_m = len(k_a)
        if len(rot_a) != n_m:
            raise ValueError(f"rotor has {len(rot_a)} entries, k has {n_m}")
        # Rotor-major, harmonic ascending: the order the recursion needs.
        order = sorted(range(n_m), key=lambda m: (int(rot_a[m]), int(k_a[m])))

    bb = _collapse(band_bins(band_cyc, band_env, n_pad, n_envp))
    sb = _collapse(shift_bins(shift_cyc, n_pad))
    assert bb is not None
    z_on = np.empty((n_ch, n_m, n_env), dtype=np.complex64)
    z_off = None if sb is None else np.empty_like(z_on)
    if n_m == 0:
        return z_on, z_off

    yt = torch.from_numpy(np.ascontiguousarray(y32)).to(dev)
    chunk = min(max(1, demod_chunk(n_ch, n_pad, dev)), n_m)
    buf = torch.empty((n_ch, chunk, n_t), dtype=torch.complex64, device=dev)
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

    if phase is not None:
        for m in order:
            row = torch.from_numpy(np.ascontiguousarray(ph[m])).to(dev)
            carr = torch.polar(torch.ones_like(row), -row).to(torch.complex64)
            torch.mul(yt, carr, out=buf[:, len(idxs)])
            idxs.append(m)
            if len(idxs) == chunk:
                flush()
    else:
        c1t = torch.from_numpy(np.ascontiguousarray(c1a)).to(dev)
        cur: Any = None
        cur_rotor, cur_k = -1, 0
        for m in order:
            rot, kk = int(rot_a[m]), int(k_a[m])
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


# ---------------------------------------------------------------------------
# the one moving average


def boxcar(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average of width ``w`` along the last axis, length preserving.

    Reflect padding at both ends, so the ends are smoothed rather than pulled
    toward zero. ``w <= 1`` (or a series too short to average) is the identity,
    so turning the smoothing off needs no second code path.
    """
    a = np.asarray(x)
    n = a.shape[-1]
    if w <= 1 or n < 2:
        return a
    w = min(int(w), n)
    pad_l, pad_r = w // 2, w - 1 - w // 2
    pad = [(0, 0)] * (a.ndim - 1) + [(pad_l, pad_r)]
    ap = np.pad(a, pad, mode="reflect")
    kern = np.ones(w) / w
    if a.ndim == 1:
        return np.convolve(ap, kern, mode="valid")
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="valid"), -1, ap)
