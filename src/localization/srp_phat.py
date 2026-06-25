"""
Near-field SRP-PHAT source localization on a 3-D position grid.

Why near-field
--------------
Classical SRP-PHAT (and ``pyroomacoustics.doa.SRP``) assumes plane-wave
propagation and searches over *direction* (azimuth/elevation): the steering
delay between mics ``i`` and ``j`` is ``(m_j - m_i)·u / c`` for a unit look
direction ``u``.  That model breaks for drone rotors: the mic array aperture
(~0.17 m on DREGON) is the same order as the source range (~0.3 m), so the
wavefront curvature carries the actual *position* information.

This module therefore searches over a 3-D grid of candidate *positions* ``p``
and uses the true propagation delay to each microphone,

    d_i(p) = ||p - m_i|| / c ,

so the steered response peaks at the source location (range included), not
just its direction.

Method
------
For an N-channel STFT ``X[i, f, t]`` we PHAT-whiten each bin,

    Xhat[i, f, t] = X[i, f, t] / |X[i, f, t]| ,

and form the time-averaged whitened cross-spectrum

    R[i, j, f] = sum_t Xhat[i, f, t] * conj(Xhat[j, f, t]) .

The PHAT steered-response power at a grid point ``p`` is

    P(p) = sum_f sum_{i,j} R[i, j, f]
                 * exp(+j 2 pi f d_i(p)) * exp(-j 2 pi f d_j(p)) ,

which is (up to a constant) the sum of GCC-PHAT cross-correlations of every mic
pair evaluated at the position-dependent TDOA ``d_i(p) - d_j(p)``.  ``P(p)`` is
real; we take its real part.

Everything runs in torch (CPU or CUDA).  The cost is dominated by the grid
einsum ``O(G * C^2 * F)``; grid points are processed in chunks to bound memory,
and a coarse-to-fine search avoids ever materialising a fine grid over the whole
volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SPEED_OF_SOUND = 343.0  # m/s, dry air ~20 C


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A regular 3-D grid of candidate positions.

    ``points`` is ``(G, 3)`` flattened in C-order over ``(nx, ny, nz)``; the
    per-axis coordinate vectors are kept so a flat SRP map can be reshaped back
    to ``(nx, ny, nz)`` for plotting / peak analysis.
    """

    points: torch.Tensor  # (G, 3) float
    xs: torch.Tensor  # (nx,)
    ys: torch.Tensor  # (ny,)
    zs: torch.Tensor  # (nz,)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.xs), len(self.ys), len(self.zs))

    def reshape_map(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(self.shape)


def make_grid(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    step: float | tuple[float, float, float],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Grid:
    """Build a regular grid over an axis-aligned box.

    Parameters
    ----------
    bounds : ((x0, x1), (y0, y1), (z0, z1))
        Inclusive box limits in metres, in the microphone coordinate frame.
    step : float or (sx, sy, sz)
        Grid spacing in metres (scalar = isotropic).
    """
    if isinstance(step, (int, float)):
        steps = (float(step), float(step), float(step))
    else:
        steps = tuple(float(s) for s in step)  # type: ignore[assignment]

    axes = []
    for (lo, hi), s in zip(bounds, steps):
        n = max(1, int(round((hi - lo) / s)) + 1)
        axes.append(torch.linspace(lo, hi, n, device=device, dtype=dtype))
    xs, ys, zs = axes

    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    return Grid(points=points, xs=xs, ys=ys, zs=zs)


# ---------------------------------------------------------------------------
# Whitened cross-spectrum
# ---------------------------------------------------------------------------


def _as_tensor(x, *, dtype, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def phat_cross_spectrum(
    audio: torch.Tensor | np.ndarray,
    sr: float,
    *,
    n_fft: int = 4096,
    hop: int | None = None,
    fmin: float = 150.0,
    fmax: float = 4000.0,
    eps: float = 1e-8,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """PHAT-whitened, time-averaged cross-spectrum of a multichannel signal.

    Parameters
    ----------
    audio : (C, N)
        Multichannel waveform, channel-first.
    sr : float
        Sample rate.
    n_fft, hop : int
        STFT parameters (``hop`` defaults to ``n_fft // 4``).
    fmin, fmax : float
        Frequency band retained for localization.  Rotor harmonics dominate the
        low/mid band; whitening removes magnitude so the band just selects where
        there is usable phase coherence and excludes DC / ultrasonic noise.

    Returns
    -------
    R : (C, C, F_band) complex
        ``R[i, j, f] = sum_t Xhat[i, f, t] conj(Xhat[j, f, t])``.
    freqs : (F_band,) float
        Centre frequency (Hz) of each retained bin.
    """
    x = _as_tensor(audio, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError(f"audio must be (C, N) or (N,), got shape {tuple(x.shape)}")

    hop = n_fft // 4 if hop is None else hop
    window = torch.hann_window(n_fft, device=x.device)
    # (C, F, T) complex
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )

    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(x.device)
    band = (freqs >= fmin) & (freqs <= fmax)
    X = X[:, band, :]
    freqs = freqs[band]

    # PHAT whitening: unit magnitude, keep phase.
    Xhat = X / (X.abs() + eps)

    # R[i, j, f] = sum_t Xhat[i, f, t] conj(Xhat[j, f, t])
    R = torch.einsum("ift,jft->ijf", Xhat, Xhat.conj())
    return R, freqs


# ---------------------------------------------------------------------------
# Steered response power
# ---------------------------------------------------------------------------


def srp_power(
    R: torch.Tensor,
    freqs: torch.Tensor,
    mic_positions: torch.Tensor | np.ndarray,
    grid: Grid,
    *,
    c: float = SPEED_OF_SOUND,
    chunk: int = 4096,
) -> torch.Tensor:
    """Evaluate the PHAT steered-response power over a grid.

    Parameters
    ----------
    R : (C, C, F) complex
        Whitened cross-spectrum from :func:`phat_cross_spectrum`.
    freqs : (F,)
        Bin centre frequencies (Hz).
    mic_positions : (C, 3)
        Microphone coordinates (metres), same frame as ``grid``.
    grid : Grid
        Candidate positions.
    chunk : int
        Number of grid points processed per batch (memory control).

    Returns
    -------
    power : (G,) float
        Real steered-response power at each grid point.
    """
    device = R.device
    rdtype = torch.float64 if R.dtype == torch.complex128 else torch.float32
    mics = _as_tensor(mic_positions, dtype=rdtype, device=device)  # (C, 3)
    pts = grid.points.to(device=device, dtype=rdtype)  # (G, 3)
    f = freqs.to(device=device, dtype=rdtype)  # (F,)

    C = mics.shape[0]
    if R.shape[0] != C:
        raise ValueError(f"R has {R.shape[0]} channels but mic_positions has {C}")

    two_pi_over_c = 2.0 * np.pi / c
    out = torch.empty(pts.shape[0], device=device, dtype=rdtype)

    for g0 in range(0, pts.shape[0], chunk):
        gp = pts[g0 : g0 + chunk]  # (g, 3)
        # distance from each candidate point to each mic: (g, C)
        dist = torch.cdist(gp, mics)
        # steering phase A[g, i, f] = exp(+j 2pi f d_i / c)
        phase = two_pi_over_c * dist.unsqueeze(-1) * f.view(1, 1, -1)  # (g, C, F)
        A = torch.polar(torch.ones_like(phase), phase)  # complex (g, C, F)
        # B[g, i, f] = sum_j R[i, j, f] conj(A[g, j, f])
        B = torch.einsum("ijf,gjf->gif", R, A.conj())
        # P(g) = Re sum_{i,f} A[g,i,f] B[g,i,f]
        out[g0 : g0 + chunk] = torch.einsum("gif,gif->g", A, B).real

    return out


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------


def extract_peaks(
    power: torch.Tensor,
    grid: Grid,
    n_peaks: int,
    *,
    suppression_radius: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy non-max suppression: pick the ``n_peaks`` strongest separated maxima.

    Returns
    -------
    positions : (n_peaks, 3)
    values : (n_peaks,)
    """
    pts = grid.points.to(power.device)
    p = power.clone()
    chosen_idx: list[int] = []
    for _ in range(n_peaks):
        i = int(torch.argmax(p).item())
        chosen_idx.append(i)
        d = torch.linalg.norm(pts - pts[i], dim=-1)
        p[d < suppression_radius] = -torch.inf
        if torch.isneginf(p).all():
            break
    idx = torch.tensor(chosen_idx, device=power.device)
    return pts[idx], power[idx]


def refine_peak(
    R: torch.Tensor,
    freqs: torch.Tensor,
    mic_positions: torch.Tensor | np.ndarray,
    center: torch.Tensor,
    *,
    half_width: float = 0.04,
    step: float = 0.005,
    c: float = SPEED_OF_SOUND,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, float]:
    """Refine a single estimate by a fine local grid search around ``center``.

    Returns the argmax position ``(3,)`` and its power.
    """
    device = R.device if device is None else device
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    bounds = (
        (cx - half_width, cx + half_width),
        (cy - half_width, cy + half_width),
        (cz - half_width, cz + half_width),
    )
    fine = make_grid(bounds, step, device=device, dtype=R.real.dtype if R.is_complex() else R.dtype)
    power = srp_power(R, freqs, mic_positions, fine, c=c)
    i = int(torch.argmax(power).item())
    return fine.points[i], float(power[i].item())
