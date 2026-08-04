"""Spectral losses: multi-scale STFT (DDSP-style) and the auraloss MRSTFT wrapper.

Ported from ``src/models/generative/losses.py`` (``MultiScaleSTFT``) and the
``auraloss.freq.MultiResolutionSTFTLoss`` usage inlined in ``train.py``
(``choice_loss`` / ``multistft_loss``).
"""

from __future__ import annotations

from typing import Any

import auraloss
import tdseries as td
import torch
from torch import nn

from framespec import FrameSpec
from losses._common import AUDIO_RATE, audio_series_spec, get_tensor

# ─── Pure tensor functions / nn.Modules ─────────────────────────────────────


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
        lin_weight: weight on the linear-magnitude component. The linear term is
            scale-SENSITIVE (raw magnitudes), so on quiet ultra-low-SNR targets
            it dominates the gradient and pulls a masking model toward
            attenuation-toward-silence (see docs/experiments/f1-se-blind-baselines.md
            DCUNet loss diagnosis). Set to 0.0 for a scale-robust log-only term.
        loss_type: 'L1' or 'L2'.
    """

    def __init__(
        self,
        n_ffts: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        log_weight: float = 1.0,
        lin_weight: float = 1.0,
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
        self.lin_weight = lin_weight
        self.loss_type = loss_type
        self.eps = eps
        # Buffers for windows (registered so they move with `.to(device)`)
        for nf in n_ffts:
            self.register_buffer(f"_win_{nf}", torch.hann_window(nf), persistent=False)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        window = getattr(self, f"_win_{n_fft}")
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
        )
        return spec.abs()

    def forward(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert est.shape == target.shape, f"shape mismatch: {est.shape} vs {target.shape}"
        loss = est.new_zeros(())
        for n_fft, hop in zip(self.n_ffts, self.hop_sizes, strict=True):
            s_est = self._stft_mag(est, n_fft, hop)
            s_tgt = self._stft_mag(target, n_fft, hop)
            lin = _mean_diff(s_est, s_tgt, self.loss_type)
            log = _mean_diff(
                torch.log(s_est + self.eps), torch.log(s_tgt + self.eps), self.loss_type
            )
            loss = loss + self.lin_weight * lin + self.log_weight * log
        return loss


def multistft_reshape(y_: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a stem/channel axis into the batch axis for MultiResolutionSTFTLoss.

    ``auraloss`` STFT losses expect ``(batch, channel, time)``. Ported from
    ``train.py::multistft_loss``, which supported 4D ``(stem, batch, channel,
    time)`` tensors (reshaped to 3D) and already-3D tensors (pass-through).
    Also accepts 2D ``(batch, time)`` and per-sample 1D ``(time,)`` mono
    Frame audio (not present in the original, which never saw un-channeled
    tensors), inserting singleton batch/channel axes as needed.
    """
    if y_.dim() == 1:
        return y_.view(1, 1, -1), y.view(1, 1, -1)
    if y_.dim() == 4:
        y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
        y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
        return y1_, y1
    if y_.dim() == 3:
        return y_, y
    if y_.dim() == 2:
        return y_.unsqueeze(1), y.unsqueeze(1)
    raise ValueError(f"Invalid shape for predicted array: {y_.shape}. Expected 1-4 dimensions.")


def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Collapse every leading axis into one batch axis: ``(*, time) -> (B, time)``.

    ``torch.stft`` only accepts 1D/2D input, so :class:`MultiScaleSTFT` (called
    directly on ``(time,)``/``(batch, time)`` tensors in its original
    single-channel usage) needs this to handle multi-mic Frame audio
    ``(batch, mic, time)``. Returns the flattened tensor and the original
    leading shape (unused here, kept for symmetry/debugging).
    """
    if x.dim() <= 2:
        return x, x.shape[:-1]
    leading = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), leading


# ─── Frame adapters ──────────────────────────────────────────────────────────


class MultiScaleSTFTLoss(nn.Module):
    """Frame adapter around :class:`MultiScaleSTFT`.

    Compares ``pred[pred_key]`` (default ``"enhanced"``) against
    ``target[target_key]`` (default ``"target"``, the clean-speech ground
    truth — see module docstring).
    """

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        pred_key: str = "enhanced",
        target_key: str = "target",
        **stft_kwargs: Any,
    ) -> None:
        super().__init__()
        self.core = MultiScaleSTFT(**stft_kwargs)
        self.pred_key = pred_key
        self.target_key = target_key
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def forward(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        est, _ = _flatten_to_2d(get_tensor(pred, self.pred_key))
        tgt, _ = _flatten_to_2d(get_tensor(target, self.target_key))
        return self.core(est, tgt)


class AuraMRSTFTLoss(nn.Module):
    """Frame adapter around ``auraloss.freq.MultiResolutionSTFTLoss``.

    Config'd exactly as it was inlined in ``train.py::choice_loss`` (arbitrary
    kwargs forwarded to auraloss, e.g. ``fft_sizes``/``hop_sizes``/``scale``).
    """

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        pred_key: str = "enhanced",
        target_key: str = "target",
        **mrstft_kwargs: Any,
    ) -> None:
        super().__init__()
        self.core = auraloss.freq.MultiResolutionSTFTLoss(**mrstft_kwargs)
        self.pred_key = pred_key
        self.target_key = target_key
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def forward(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        est = get_tensor(pred, self.pred_key)
        tgt = get_tensor(target, self.target_key)
        est, tgt = multistft_reshape(est, tgt)
        return self.core(est, tgt)


__all__ = [
    "MultiScaleSTFT",
    "multistft_reshape",
    "MultiScaleSTFTLoss",
    "AuraMRSTFTLoss",
]
