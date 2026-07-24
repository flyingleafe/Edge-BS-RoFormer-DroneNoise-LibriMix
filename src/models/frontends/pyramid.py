"""Multi-resolution STFT pyramid front-end with per-band IF (G8a, C1 of the
hierarchical front-end design — docs/g8-hierarchical-frontend-design.md).

The single-window STFT is caught in a resolution conundrum that lands
differently per harmonic: fundamentals (30-120 Hz) need fine FREQUENCY
resolution (7.8 Hz bins are catastrophically coarse there) but tolerate slow
time; high harmonics (1-2 kHz, k≈10-25) need fine TIME resolution and phase
stability while IF supplies sub-bin frequency. Constant-Q allocates exactly
backwards (G2a refuted). This front-end runs four parallel STFTs, each used
ONLY in its band — the wavelet/MuReNN allocation without CQT's high-k
coarseness:

| band (Hz) | n_fft | Δf (Hz) | window | serves |
|---|---|---|---|---|
| 30-250    | 8192 | 1.95 | 512 ms | fundamentals + k≤2 |
| 250-1000  | 4096 | 3.9  | 256 ms | mid harmonics |
| 1000-2000 | 2048 | 7.8  | 128 ms | k≈10-25: fine time, IF sub-bin |
| 2000-4000 | 1024 | 15.6 | 64 ms  | headroom band |

Each band contributes log1p-magnitude + IF-deviation channels (the proven
G2b estimator, per band; every band uses hop = n_fft/4 so its IF scaling
follows ITS OWN n_fft/hop relation), cropped to its frequency range and
resampled onto:

- a common LOG-FREQUENCY axis, geometric from 30 Hz to 4 kHz with density
  set by the 8192-band's 1.95 Hz resolution at the bottom (~48 bins/octave
  → 340 rows over ~7.06 octaves). Log-f makes comb patterns
  shift-equivariant in f0 (a comb at 80 rev/s is a translate of one at 60)
  — the substrate C2's harmonic stacking exploits. Resampling is fixed
  linear interpolation with index/weight buffers precomputed in
  ``__init__`` (the G4 gather machinery — that part of G4 worked);
- the standard hop-512 time grid: each band is computed at its natural hop
  (n_fft/4) and linearly interpolated in time. Band channels are zero
  outside their own rows, so the 8192-band's 512 ms time smear stays
  confined to its own (slow-moving, k≤2) rows and cannot leak into the
  high bands.

Output ``(B, 8, 340, T)`` — channel order
``[mag_b0, if_b0, mag_b1, if_b1, mag_b2, if_b2, mag_b3, if_b3]`` (b0 = the
30-250 Hz / n_fft 8192 band), ``T = n_samples // hop_length + 1``. Zero
trainable parameters; all torch, on-device (``torch.remainder`` IF wrap, no
numpy unwrap sync).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from . import SpectralFrontEnd, register_frontend


@register_frontend
class PyramidIFFrontEnd(SpectralFrontEnd):
    """Octave-banded multi-window STFT pyramid + per-band IF (key ``pyramid_if``)."""

    key = "pyramid_if"
    out_channels = 8

    def __init__(
        self,
        hop_length: int = 512,
        sample_rate: int = 16000,
        n_ffts: tuple[int, ...] = (8192, 4096, 2048, 1024),
        band_edges: tuple[float, ...] = (30.0, 250.0, 1000.0, 2000.0, 4000.0),
        bins_per_octave: float = 48.0,
    ):
        super().__init__()
        if len(band_edges) != len(n_ffts) + 1:
            raise ValueError("band_edges must have len(n_ffts) + 1 entries")
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_ffts = tuple(n_ffts)
        self.band_edges = tuple(band_edges)
        self.n_bands = len(n_ffts)
        self.out_channels = 2 * self.n_bands

        f_min, f_max = band_edges[0], band_edges[-1]
        n_rows = int(round(math.log2(f_max / f_min) * bins_per_octave)) + 1
        self.n_rows = n_rows
        # Geometric grid with exact endpoints: f[i] = f_min * (f_max/f_min)^(i/(n-1)).
        i = torch.arange(n_rows, dtype=torch.float64)
        f_log = f_min * (f_max / f_min) ** (i / (n_rows - 1))
        self.f_log: Tensor
        self.register_buffer("f_log", f_log.to(torch.float32))

        # Per-band buffers: window, IF bin advance, log-grid gather, row mask.
        for b, n_fft in enumerate(self.n_ffts):
            hop_b = n_fft // 4
            n_bins = n_fft // 2 + 1
            bin_hz = sample_rate / n_fft
            self.register_buffer(f"window_{b}", torch.hann_window(n_fft))
            k_bin = torch.arange(n_bins, dtype=torch.float32)
            self.register_buffer(f"advance_{b}", 2.0 * math.pi * hop_b / n_fft * k_bin)

            pos = f_log / bin_hz  # fractional source bin per log row (float64)
            j = pos.floor().long().clamp(0, n_bins - 2)
            frac = (pos - j.to(torch.float64)).clamp(0.0, 1.0).to(torch.float32)
            lo, hi = band_edges[b], band_edges[b + 1]
            in_band = (f_log >= lo) & (
                (f_log < hi) if b < self.n_bands - 1 else (f_log <= hi + 1e-6)
            )
            self.register_buffer(f"idx_lo_{b}", j)
            self.register_buffer(f"idx_hi_{b}", j + 1)
            self.register_buffer(f"frac_{b}", frac)
            self.register_buffer(f"row_mask_{b}", in_band.to(torch.float32))

    def num_frames(self, n_samples: int) -> int:
        return n_samples // self.hop_length + 1

    # ── pieces ───────────────────────────────────────────────────────────

    def _band_mag_if(self, audio: Tensor, b: int) -> tuple[Tensor, Tensor]:
        """Band ``b``'s log1p magnitude + IF deviation on its native grids.

        Returns ``(mag, if_dev)`` each ``(B, F_b, T_b)`` with
        ``F_b = n_fft_b // 2 + 1`` and ``T_b = n // (n_fft_b // 4) + 1``.
        The IF deviation is in fractional bins of THIS band's grid (its own
        n_fft/hop relation); with hop = n_fft/4 it is bounded to [-2, 2).
        """
        n_fft = self.n_ffts[b]
        hop_b = n_fft // 4
        X = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_b,
            window=self.get_buffer(f"window_{b}"),
            return_complex=True,
            normalized=True,
        )
        mag = torch.log1p(X.abs())
        phase = torch.angle(X)
        dphi = phase[..., 1:] - phase[..., :-1]
        dev = dphi - self.get_buffer(f"advance_{b}")[None, :, None]
        dev = torch.remainder(dev + math.pi, 2.0 * math.pi) - math.pi
        if_dev = dev * (n_fft / (2.0 * math.pi * hop_b))
        return mag, F.pad(if_dev, (1, 0))

    def _gather_rows(self, x: Tensor, b: int) -> Tensor:
        """Fixed linear interp of ``x (B, F_b, T_b)`` onto the log grid → (B, R, T_b)."""
        lo = x[:, self.get_buffer(f"idx_lo_{b}"), :]
        hi = x[:, self.get_buffer(f"idx_hi_{b}"), :]
        w = self.get_buffer(f"frac_{b}")[None, :, None]
        return lo * (1.0 - w) + hi * w

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        t_out = audio.shape[-1] // self.hop_length + 1

        outs = []
        for b in range(self.n_bands):
            mag, if_dev = self._band_mag_if(audio, b)
            x = torch.stack([self._gather_rows(mag, b), self._gather_rows(if_dev, b)], dim=1)
            if x.shape[-1] != t_out:  # band's natural hop → hop-512 grid
                bsz, c, r, t_b = x.shape
                x = F.interpolate(
                    x.reshape(bsz, c * r, t_b), size=t_out, mode="linear", align_corners=False
                ).reshape(bsz, c, r, t_out)
            outs.append(x * self.get_buffer(f"row_mask_{b}")[None, None, :, None])
        return torch.cat(outs, dim=1)  # (B, 2*n_bands, R, T)
