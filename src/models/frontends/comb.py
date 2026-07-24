"""Comb matched-filter front-end on the linear STFT grid (G4 VK-parity arm).

The trainable analogue of the blind VK tracker's whitened comb scan
(``data_processing.vk_blind_seeding``: ``whitened_logmag`` + ``comb_scan`` /
``_tooth_values`` / the stage-guard tooth statistic), composed with the IF
(instantaneous-frequency) machinery from ``frontends.stft.STFTMagIF``.

Motivation (docs/experiments/g1-vk-parity.md § Phase G4): G2 showed that
phase/IF evidence helps but constant-Q harmonic stacking hurts — harmonic
AGGREGATION is the missing ingredient and it must live on the LINEAR
frequency grid. This front-end aggregates whitened-magnitude and IF evidence
along harmonic combs for a dense grid of candidate f0s, so the trunk operates
in f0-space where each rotor is a ridge.

Per candidate f0 row and time frame, three channels:

1. **comb score** — mean over teeth ``k`` of the whitened log-magnitude
   ``W`` sampled at ``k*f0`` (linear interpolation between adjacent bins);
   the scan statistic of ``vk_blind_seeding.comb_scan``. ``W`` = log10
   magnitude minus a running median over frequency (``whiten_hz`` window),
   exactly mirroring ``whitened_logmag`` (relative log eps included).
2. **frequency consensus** — each tooth's IF deviation converted to a base
   deviation in rev/s (``IF_bins * bin_hz / k``), combined across teeth by a
   magnitude-weighted mean with Fisher ``k^2`` weighting (a tooth at harmonic
   ``k`` measures f0 ``k``× more precisely, variance ∝ 1/k²) — the same
   weighting shape as the VK information-filter update. Clamped to
   ``±consensus_clamp`` rev/s to stay a bounded input.
3. **occupancy** — fraction of teeth whose ``W`` exceeds the spectrum's
   median over frequency at that frame (the stage-guard tooth statistic of
   ``vk_blind_seeding``, with the robust-sigma offset dropped to keep the
   channel a smooth 0..1 fraction).

The tooth gather is precomputed in ``__init__`` as index + weight buffers
(one fused advanced-indexing gather + weighted sums per forward — no
per-row Python loops). Teeth are capped at ``k*f0 <= max_harmonic_hz``
(1200 Hz — the band the VK scan itself proved out), so small f0 rows get
more teeth, mirroring the scan's mean-over-teeth normalisation. All torch,
on-device; the IF wrap uses ``torch.remainder`` (no numpy unwrap sync).

Output ``(B, 3, n_rows, T)`` on the model's standard hop time grid
(``T = n_samples // hop_length + 1``): default f0 grid 30..120 rev/s step
0.25 → 361 rows.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from . import SpectralFrontEnd, register_frontend


@register_frontend
class CombIFFrontEnd(SpectralFrontEnd):
    """Whitened comb matched-filter + IF-consensus front-end (key ``comb_if``)."""

    key = "comb_if"
    out_channels = 3

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
        f0_min: float = 30.0,
        f0_max: float = 120.0,
        f0_step: float = 0.25,
        max_harmonic_hz: float = 1200.0,
        whiten_hz: float = 150.0,
        consensus_clamp: float = 2.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.consensus_clamp = consensus_clamp

        bin_hz = sample_rate / n_fft
        self.bin_hz = bin_hz
        n_bins = n_fft // 2 + 1

        self.window: Tensor
        self.register_buffer("window", torch.hann_window(n_fft))
        # Expected per-hop phase advance per bin (same as STFTMagIF).
        k_bin = torch.arange(n_bins, dtype=torch.float32)
        self.bin_advance: Tensor
        self.register_buffer("bin_advance", 2.0 * math.pi * hop_length / n_fft * k_bin)

        # Whitening window in bins (odd, like whitened_logmag's `| 1`).
        self.whiten_bins = int(round(whiten_hz / bin_hz)) | 1

        # ── f0 grid + precomputed tooth gather (index/weight buffers) ──────
        n_rows = int(round((f0_max - f0_min) / f0_step)) + 1
        f0 = f0_min + f0_step * torch.arange(n_rows, dtype=torch.float32)
        self.n_rows = n_rows
        self.f0_grid: Tensor
        self.register_buffer("f0_grid", f0)

        k_max = int(max_harmonic_hz // f0_min)
        harm = torch.arange(1, k_max + 1, dtype=torch.float32)  # (K,)
        tooth_hz = f0[:, None] * harm[None, :]  # (R, K)
        valid = (tooth_hz <= max_harmonic_hz) & (tooth_hz <= (n_bins - 1) * bin_hz)

        pos = tooth_hz / bin_hz  # fractional bin position
        j = pos.floor().long().clamp(0, n_bins - 2)
        frac = (pos - j.float()).clamp(0.0, 1.0)

        mask = valid.float()
        self.idx_lo: Tensor
        self.idx_hi: Tensor
        self.frac: Tensor
        self.mask: Tensor
        self.inv_count: Tensor
        self.harm: Tensor
        self.register_buffer("idx_lo", j)
        self.register_buffer("idx_hi", j + 1)
        self.register_buffer("frac", frac)
        self.register_buffer("mask", mask)
        self.register_buffer("inv_count", 1.0 / mask.sum(dim=1).clamp(min=1.0))
        self.register_buffer("harm", harm)

    def num_frames(self, n_samples: int) -> int:
        return n_samples // self.hop_length + 1

    # ── pieces ───────────────────────────────────────────────────────────

    def _whitened_logmag(self, mag: Tensor) -> Tensor:
        """log10 magnitude minus running median over frequency (per frame).

        Mirrors ``vk_blind_seeding.whitened_logmag`` (including the relative
        log eps from ``rps_refinement.compute_logmag`` — keeps the log floor
        tied to signal scale so silent bins do not dominate).
        """
        eps = 1e-3 * mag.flatten(1).median(dim=1).values.clamp(min=1e-12)
        logmag = torch.log10(mag + eps[:, None, None])
        pad = self.whiten_bins // 2
        lm_pad = F.pad(logmag, (0, 0, pad, pad), mode="replicate")  # pad freq dim
        env = lm_pad.unfold(1, self.whiten_bins, 1).median(dim=-1).values
        return logmag - env

    def _if_deviation_bins(self, X: Tensor) -> Tensor:
        """IF deviation from bin center in fractional bins (as STFTMagIF)."""
        phase = torch.angle(X)
        dphi = phase[..., 1:] - phase[..., :-1]
        dev = dphi - self.bin_advance[None, :, None]
        dev = torch.remainder(dev + math.pi, 2.0 * math.pi) - math.pi
        if_dev = dev * (self.n_fft / (2.0 * math.pi * self.hop_length))
        return F.pad(if_dev, (1, 0))  # first frame has no predecessor

    def _gather_teeth(self, x: Tensor) -> Tensor:
        """Sample ``x (B, F, T)`` at all tooth positions → ``(B, R, K, T)``."""
        lo = x[:, self.idx_lo, :]  # (B, R, K, T)
        hi = x[:, self.idx_hi, :]
        w = self.frac[None, :, :, None]
        return lo * (1.0 - w) + hi * w

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        X = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=True,
        )
        mag = X.abs()  # (B, F, T)
        white = self._whitened_logmag(mag)
        if_bins = self._if_deviation_bins(X)

        teeth_w = self._gather_teeth(white)  # (B, R, K, T)
        teeth_if = self._gather_teeth(if_bins)

        m = self.mask[None, :, :, None]
        inv_n = self.inv_count[None, :, None]

        # 1. comb score: mean whitened log-mag over the row's valid teeth.
        comb = (teeth_w * m).sum(dim=2) * inv_n  # (B, R, T)

        # 2. frequency consensus: magnitude*k^2-weighted mean of per-tooth
        #    base-frequency deviations (rev/s), clamped to stay bounded.
        harm = self.harm[None, None, :, None]  # (1, 1, K, 1)
        dev_rev = teeth_if * self.bin_hz / harm  # (B, R, K, T), rev/s
        wgt = torch.exp(teeth_w).clamp(max=1e4) * harm**2 * m
        consensus = (wgt * dev_rev).sum(dim=2) / (wgt.sum(dim=2) + 1e-8)
        consensus = consensus.clamp(-self.consensus_clamp, self.consensus_clamp)

        # 3. occupancy: fraction of teeth above the frame's spectrum median
        #    (stage-guard tooth statistic, sans robust-sigma offset).
        thr = white.median(dim=1, keepdim=True).values.unsqueeze(1)  # (B, 1, 1, T)
        occ = ((teeth_w > thr).float() * m).sum(dim=2) * inv_n

        return torch.stack([comb, consensus, occ], dim=1)  # (B, 3, R, T)
