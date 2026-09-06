"""Scale-invariant SDR loss for speech enhancement.

Negative SI-SDR (Le Roux et al., ICASSP 2019) — the standard, metric-aligned SE
objective. Unlike masked MSE (which at ultra-low SNR rewards attenuation toward
silence, improving SDR but not SI-SDR/intelligibility), this directly optimizes
the reported scale-invariant SDR. Matches ``metrics.separation.si_sdr``: the
target projection ``scale = <est,ref>/||ref||^2`` is applied per sample, and the
ratio ``||s_target||^2 / ||e_noise||^2`` is reduced over every non-batch axis.
Typically composed with a small multi-resolution STFT term (which supplies a
stable early gradient) — see ``conf/loss/si_sdr_mrstft.yaml``.
"""

from __future__ import annotations

import tdseries as td
import torch

from framespec import FrameSpec
from losses._common import AUDIO_RATE, audio_series_spec, get_tensor


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Mean negative SI-SDR (dB) over the batch. Reduces over all non-batch axes.

    ``est``/``ref``: ``(B, ..., T)`` (mono ``(B, T)`` or multichannel
    ``(B, C, T)``). Differentiable; lower is better (maximizes SI-SDR).
    """
    dims = tuple(range(1, est.dim())) if est.dim() > 1 else (0,)
    dot = (est * ref).sum(dim=dims, keepdim=True)
    ref_energy = (ref * ref).sum(dim=dims, keepdim=True) + eps
    s_target = (dot / ref_energy) * ref
    e_noise = est - s_target
    ratio = (s_target**2).sum(dim=dims) / ((e_noise**2).sum(dim=dims) + eps)
    si_sdr = 10.0 * torch.log10(ratio + eps)
    return -si_sdr.mean()


class SISDRLoss:
    """Frame adapter around :func:`si_sdr_loss` for the ``speech_enhancement`` task."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        pred_key: str = "enhanced",
        target_key: str = "target",
    ) -> None:
        self.pred_key = pred_key
        self.target_key = target_key
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        est = get_tensor(pred, self.pred_key)
        tgt = get_tensor(target, self.target_key)
        if est.dim() == 1:  # per-sample mono (T,) -> (1, T)
            est = est.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
        return si_sdr_loss(est, tgt)


__all__ = ["si_sdr_loss", "SISDRLoss"]
