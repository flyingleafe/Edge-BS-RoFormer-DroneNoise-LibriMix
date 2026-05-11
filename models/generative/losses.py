"""
Multi-scale STFT loss (DDSP-style) ported from drone_audition.losses.

Implemented directly on torch.stft to avoid an extra Asteroid filterbanks
dependency at this layer (Asteroid is already a transitive dep elsewhere in
the repo, but for a generator-only training loop a small standalone module
is preferable).
"""
from __future__ import annotations

import torch
from torch import nn


def _mean_diff(a: torch.Tensor, b: torch.Tensor, loss_type: str = "L1") -> torch.Tensor:
    if loss_type == "L1":
        return (a - b).abs().mean()
    if loss_type == "L2":
        return ((a - b) ** 2).mean()
    raise ValueError(f"Unknown loss_type {loss_type}")


class MultiScaleSTFT(nn.Module):
    """Multi-scale spectral loss (linear-magnitude + log-magnitude).

    Args:
        n_ffts: list of FFT/window sizes.
        hop_sizes: list of hop sizes (defaults to n_fft // 4 each).
        log_weight: weight on the log-magnitude component.
        loss_type: 'L1' or 'L2'.
    """

    def __init__(
        self,
        n_ffts: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        log_weight: float = 1.0,
        loss_type: str = "L1",
        eps: float = 1e-8,
    ):
        super().__init__()
        if n_ffts is None:
            n_ffts = [2048, 1024, 512, 256, 128, 64]
        if hop_sizes is None:
            hop_sizes = [n // 4 for n in n_ffts]
        assert len(n_ffts) == len(hop_sizes)
        self.n_ffts = n_ffts
        self.hop_sizes = hop_sizes
        self.log_weight = log_weight
        self.loss_type = loss_type
        self.eps = eps
        # Buffers for windows (registered so they move with `.to(device)`)
        for nf in n_ffts:
            self.register_buffer(f"_win_{nf}", torch.hann_window(nf), persistent=False)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        window = getattr(self, f"_win_{n_fft}")
        spec = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, return_complex=True, center=True,
        )
        return spec.abs()

    def forward(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert est.shape == target.shape, f"shape mismatch: {est.shape} vs {target.shape}"
        loss = est.new_zeros(())
        for n_fft, hop in zip(self.n_ffts, self.hop_sizes):
            s_est = self._stft_mag(est, n_fft, hop)
            s_tgt = self._stft_mag(target, n_fft, hop)
            lin = _mean_diff(s_est, s_tgt, self.loss_type)
            log = _mean_diff(
                torch.log(s_est + self.eps), torch.log(s_tgt + self.eps), self.loss_type
            )
            loss = loss + lin + self.log_weight * log
        return loss
