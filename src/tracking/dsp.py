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
    "analytic_signal_tensor",
    "band_bins",
    "boxcar",
    "demod",
    "demodulate_trajectories",
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
# differentiable trajectory reads (tensor seams, no NumPy driver round trip)


def analytic_signal_tensor(audio: Any, *, pad_samples: int = 8000) -> Any:
    """Zero-pad real ``(B, N)`` audio, then form its complex64 analytic signal.

    Padding precedes the Hilbert transform: its imaginary part is generally
    nonzero over the pad. The returned length is exactly ``N + 2 * pad_samples``.
    This tensor-only seam preserves gradients to the original waveform, even
    under neural AMP; the transform itself always runs in float32/complex64.
    """
    import torch
    import torch.nn.functional as F

    if audio.ndim != 2 or audio.shape[-1] == 0 or audio.is_complex():
        raise ValueError("audio must be a nonempty real (B, N) tensor")
    if not isinstance(pad_samples, int) or pad_samples < 0:
        raise ValueError("pad_samples must be a nonnegative integer")
    padded = F.pad(audio.to(torch.float32), (pad_samples, pad_samples))
    n = padded.shape[-1]
    bins = torch.arange(n, device=audio.device)
    # DC and (for even lengths) Nyquist are singletons; strictly positive
    # frequencies double, and negative frequencies disappear.
    weight = torch.where(
        (bins == 0) | ((n % 2 == 0) & (bins == n // 2)),
        1.0,
        torch.where((bins > 0) & (bins <= (n - 1) // 2), 2.0, 0.0),
    ).to(torch.float32)
    return torch.fft.ifft(torch.fft.fft(padded, dim=-1) * weight, dim=-1)


def _rates_at_samples(rates: Any, samples: Any, hop_length: int) -> Any:
    """Linear interpolation at physical audio-sample positions, ends held."""
    position = (samples.to(dtype=rates.dtype) / hop_length).clamp(0, rates.shape[-1] - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(rates.shape[-1] - 1)
    weight = position - left
    a = rates.index_select(-1, left)
    return a + weight * (rates.index_select(-1, right) - a)


def demodulate_trajectories(
    analytic: Any,
    rates: Any,
    orders: Any,
    *,
    n_samples: int,
    sample_rate: int = 16000,
    hop_length: int = 512,
    envelope_rate: int = 500,
    half_bandwidths: tuple[float, ...] = (8.0, 32.0, 128.0),
    pad_samples: int = 8000,
    harmonic_chunk: int = 4,
) -> tuple[Any, Any]:
    """Read differentiable complex basebands around ``(B, R, T)`` shaft rates.

    Rates are at ``j * hop_length / sample_rate`` with endpoint holding, not
    stretched to the waveform ends. Output ``(B, R, K, W, M)`` is sampled at
    ``m / envelope_rate``, including the final grid point when the crop length
    is an exact multiple of the stride. ``orders`` is a 1-D tensor.

    The bool mask has the same shape and excludes the first/last 0.25 s of the
    crop, stopped candidates (<= 0.5 rev/s), nonpositive orders, and bands
    touching DC or Nyquist. It describes geometry, NOT energy or confidence.
    Callers must additionally mask silent crops and unavailable lag partners.
    Invalid envelopes are retained rather than clamped onto valid frequencies.

    Shaft phase is trapezoid-integrated in float64 with phase zero at original
    sample zero; harmonic phase is reduced modulo 2pi before complex64 carrier
    synthesis. Each harmonic chunk's demodulated FFT serves every bandwidth.
    Zoom-IFFTs share the existing band-selection primitive and stride scaling.
    The zero pad limits, but does not eliminate, brickwall boundary leakage.
    """
    import math

    import torch

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(pad_samples, int) or pad_samples < 0:
        raise ValueError("pad_samples must be a nonnegative integer")
    if sample_rate <= 0 or envelope_rate <= 0 or sample_rate % envelope_rate:
        raise ValueError("sample_rate must be a positive integer multiple of envelope_rate")
    if hop_length <= 0 or not isinstance(harmonic_chunk, int) or harmonic_chunk < 1:
        raise ValueError("hop_length and harmonic_chunk must be positive")
    if (
        analytic.ndim != 2
        or not analytic.is_complex()
        or analytic.shape[-1] != n_samples + 2 * pad_samples
    ):
        raise ValueError("analytic must have shape (B, n_samples + 2 * pad_samples)")
    if rates.ndim != 3 or rates.shape[0] != analytic.shape[0] or rates.shape[-1] == 0:
        raise ValueError("rates must have shape (B, R, T) with T >= 1")
    if orders.ndim != 1 or orders.numel() == 0:
        raise ValueError("orders must be a nonempty one-dimensional tensor")
    if rates.device != analytic.device or orders.device != analytic.device:
        raise ValueError("analytic, rates and orders must share a device")
    if not half_bandwidths or any(
        not math.isfinite(b) or b <= 0 or b >= envelope_rate / 2 for b in half_bandwidths
    ):
        raise ValueError("half_bandwidths must lie strictly between zero and envelope Nyquist")

    stride = int(sample_rate // envelope_rate)
    n_padded = analytic.shape[-1]
    # A multiple of stride makes the zoom grid exactly envelope_rate, including
    # for arbitrary crop lengths. This extra transform pad is less than stride.
    n_envp = (n_padded + stride - 1) // stride
    n_fft = n_envp * stride
    m = n_samples // stride + 1
    dev = analytic.device
    sample_positions = torch.arange(n_padded, device=dev) - pad_samples
    rates64 = rates.to(torch.float64)
    r_audio = _rates_at_samples(rates64, sample_positions, hop_length)
    increments = (r_audio[..., 1:] + r_audio[..., :-1]) * (math.pi / sample_rate)
    phase = torch.cat((torch.zeros_like(r_audio[..., :1]), increments.cumsum(-1)), dim=-1)
    phase = phase - phase[..., pad_samples : pad_samples + 1]

    env_samples = torch.arange(m, device=dev) * stride
    r_env = _rates_at_samples(rates64, env_samples, hop_length)
    interior = (env_samples >= sample_rate * 0.25) & (env_samples <= n_samples - sample_rate * 0.25)
    bandwidths = torch.tensor(half_bandwidths, dtype=torch.float64, device=dev)
    # _take_band assumes both +/- bins fit, with no envelope-Nyquist bin.
    bins = [min(math.floor(b * n_fft / sample_rate), (n_envp - 1) // 2) for b in half_bandwidths]
    crop_indices = (torch.arange(m, device=dev) + pad_samples // stride) % n_envp
    fractional_shift = pad_samples % stride
    if fractional_shift:
        # Move the zoom grid by the remainder so even non-stride-aligned pads
        # read original sample zero, not the nearest envelope-grid neighbour.
        freq = torch.fft.fftfreq(n_envp, d=1.0 / envelope_rate, device=dev)
        angle = freq * (2 * math.pi * fractional_shift / sample_rate)
        grid_shift = torch.polar(torch.ones_like(angle), angle)
    else:
        grid_shift = None

    chunks = []
    masks = []
    analytic64 = analytic.to(torch.complex64)
    for start in range(0, orders.numel(), harmonic_chunk):
        order = orders[start : start + harmonic_chunk].to(torch.float64)
        angle = torch.remainder(phase.unsqueeze(2) * order[None, None, :, None], 2 * math.pi)
        angle = angle.to(torch.float32)
        carrier = torch.polar(torch.ones_like(angle), -angle)
        spec = torch.fft.fft(analytic64[:, None, None, :] * carrier, n=n_fft, dim=-1)
        bands = []
        for b in bins:
            band = _take_band(spec, b, 0, n_envp, str(dev))
            if grid_shift is not None:
                band = band * grid_shift
            envelope = torch.fft.ifft(band, dim=-1) / float(stride)
            bands.append(envelope.index_select(-1, crop_indices))
        chunks.append(torch.stack(bands, dim=3))
        center = r_env.unsqueeze(2) * order[None, None, :, None]
        width = bandwidths[None, None, None, :, None]
        masks.append(
            (center.unsqueeze(3) > width)
            & (center.unsqueeze(3) + width < sample_rate / 2)
            & (r_env[:, :, None, None, :] > 0.5)
            & (order[None, None, :, None, None] > 0)
            & interior
        )
    return torch.cat(chunks, dim=2), torch.cat(masks, dim=2)


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
