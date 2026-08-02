"""Spatial (cross-microphone) likelihood — the objective that can see wind.

Why this exists
---------------
:mod:`losses.spectral_likelihood` folds the microphone axis into the batch, so
it is a product of **per-microphone marginals**. That makes it comparable across
channel counts, and it fixed the far larger defect (fitting a stochastic
component to one realization; see that module). But it is structurally blind to
the one thing that distinguishes the wind-wake channel from the generator's own
broadband branch: **wind is incoherent across microphones, the propagated rotor
field is correlated across them**. Coherence lives in the cross-spectrum, which
a marginal likelihood never looks at.

Measured consequence: trained under the marginal likelihood the wind channel
stays inert, carrying 9.9e-4 of the predicted variance on DREGON and 4.4e-6 on
Michael's — the flexible learned broadband filter explains the residual first,
and nothing rewards the only thing wind can uniquely do.

The model
---------
Drop the mean entirely and marginalize the unknown global phase into the
covariance (the same phase-marginalization argument as the Rice likelihood, now
applied per bin to the whole array). The observed spatial vector is then
zero-mean complex Gaussian::

    x(f,t) ~ CN(0, R(f,t))
    R      = sum_r  d_r d_r^H P_r(f,t)  +  diag(w(f,t))

- ``d_r(f) = [ (ref/dist_{m,r}) * exp(-j 2 pi f tau_{m,r}) ]_m`` is the steering
  vector of rotor ``r``. It is **known exactly from geometry** — the same 1/r and
  delay the generator already applies in :func:`~models.generative.propagate` —
  so it costs no parameters.
- ``P_r`` is rotor ``r``'s emitted source power (harmonic bank + its broadband
  residual). Correlated across microphones by construction.
- ``w`` is the per-microphone wind power. **Diagonal by construction.**

So everything from the rotors is a rank-``R`` structured term and the wind is the
only diagonal one: they are separable, which is exactly what the marginal
likelihood could not do.

This is the standard array-processing / Whittle form, and the negative
log-likelihood of one bin is::

    NLL = log det R + x^H R^{-1} x

Both terms are evaluated with the matrix determinant lemma and Woodbury, so the
cost is a batched ``R x R`` (4x4) solve rather than an ``M x M`` inverse::

    C        = P^{1/2} D^H W^{-1} D P^{1/2}          (R x R)
    log det R = log det W + log det(I + C)
    R^{-1} x  = W^{-1} x - W^{-1} D P^{1/2} (I + C)^{-1} P^{1/2} D^H W^{-1} x

The wind power ``w`` therefore appears in ``log det W`` and in every ``W^{-1}``,
i.e. it is load-bearing in both terms — unlike the marginal case, where it could
be crowded out by the rotor branch explaining the same per-mic power.
"""

from __future__ import annotations

from typing import Any

import tdseries as td
import torch
from torch import nn

from losses._common import AUDIO_RATE, audio_series_spec, get_tensor
from tasks.spec import FrameSpec

__all__ = ["spatial_whittle_nll", "steering_vectors", "SpatialLikelihood", "SpatialLikelihoodLoss"]

SPEED_OF_SOUND = 343.0


def steering_vectors(
    rel_pos: torch.Tensor,
    freqs: torch.Tensor,
    *,
    ref_distance: float = 1.0,
    c: float = SPEED_OF_SOUND,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-rotor array steering vectors ``[B, M, R, F]`` (complex).

    ``d_{m,r}(f) = (ref/dist_{m,r}) * exp(-j 2 pi f dist_{m,r}/c)`` — the exact
    1/r attenuation and propagation delay the generator itself applies, so the
    likelihood and the synthesizer describe the same field.

    Args:
        rel_pos: ``[B, M, R, 3]`` rotor->microphone vectors (metres).
        freqs: ``[F]`` bin centre frequencies (Hz).
    """
    dist = torch.linalg.vector_norm(rel_pos, dim=-1).clamp_min(eps)  # [B, M, R]
    amp = (ref_distance / dist).unsqueeze(-1)  # [B, M, R, 1]
    phase = -2.0 * torch.pi * freqs.reshape(1, 1, 1, -1) * (dist / c).unsqueeze(-1)
    return amp * torch.exp(1j * phase.to(torch.float32))


def spatial_whittle_nll(
    x: torch.Tensor,
    steering: torch.Tensor,
    source_psd: torch.Tensor,
    wind_psd: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Negative log-likelihood of a multichannel bin under ``CN(0, R)``.

    ``R = D diag(source_psd) D^H + diag(wind_psd)``, evaluated by Woodbury so the
    only linear algebra is an ``R x R`` solve.

    Args:
        x: ``[B, M, F, T]`` complex observed STFT.
        steering: ``[B, M, R, F]`` complex steering vectors.
        source_psd: ``[B, R, F, T]`` per-rotor emitted power (>= 0).
        wind_psd: ``[B, M, F, T]`` per-microphone incoherent power (> 0).

    Returns:
        ``[B, F, T]`` elementwise NLL.
    """
    # -> [B, F, T, M] / [B, F, T, M, R] / [B, F, T, R]
    xv = x.permute(0, 2, 3, 1).unsqueeze(-1)  # [B, F, T, M, 1]
    d = steering.permute(0, 3, 1, 2).unsqueeze(2)  # [B, F, 1, M, R]
    p = source_psd.permute(0, 2, 3, 1).clamp_min(0.0)  # [B, F, T, R]
    w = wind_psd.permute(0, 2, 3, 1).clamp_min(eps)  # [B, F, T, M]

    sqrt_p = p.clamp_min(eps).sqrt().to(d.dtype)  # [B, F, T, R]
    dp = d * sqrt_p.unsqueeze(-2)  # [B, F, T, M, R]  (= D P^{1/2})
    winv = (1.0 / w).to(d.dtype).unsqueeze(-1)  # [B, F, T, M, 1]

    dpw = dp.conj().transpose(-2, -1) * winv.transpose(-2, -1)  # [B,F,T,R,M] = P^H D^H W^-1
    cmat = dpw @ dp  # [B, F, T, R, R]
    r = cmat.shape[-1]
    ident = torch.eye(r, dtype=cmat.dtype, device=cmat.device)
    a = ident + cmat

    # log det R = log det W + log det(I + C)
    logdet = w.clamp_min(eps).log().sum(-1) + torch.log(
        torch.linalg.det(a).abs().clamp_min(eps)
    )

    # x^H R^-1 x  =  x^H W^-1 x  -  (D P^{1/2 H} W^-1 x)^H (I+C)^-1 (same)
    wx = winv * xv  # [B, F, T, M, 1]
    quad_diag = (xv.conj() * wx).sum(dim=(-2, -1)).real  # [B, F, T]
    y = dpw @ xv  # [B, F, T, R, 1]
    sol = torch.linalg.solve(a, y)  # [B, F, T, R, 1]
    quad_corr = (y.conj() * sol).sum(dim=(-2, -1)).real
    quad = (quad_diag - quad_corr).clamp_min(0.0)

    return logdet + quad


class SpatialLikelihood(nn.Module):
    """Tensor-level spatial Whittle likelihood over one or more STFT scales.

    Args:
        n_ffts: STFT sizes to score at.
        hop_ratio: hop as a fraction of each ``n_fft``.
        sample_rate: audio rate, for the steering-vector frequencies.
        floor_rel: wind-power floor as a fraction of the clip's mean observed bin
            power (detached). As in the marginal likelihood, a learned variance
            is unbounded below without a floor, and an absolute floor is inert on
            loud clips and dominant on quiet ones.
        ref_distance: reference distance of the generator's 1/r law.
    """

    def __init__(
        self,
        n_ffts: tuple[int, ...] | list[int] = (1024,),
        hop_ratio: float = 0.25,
        sample_rate: int = 16000,
        floor_rel: float = 1e-4,
        ref_distance: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_ffts = tuple(int(n) for n in n_ffts)
        self.hop_ratio = float(hop_ratio)
        self.sample_rate = int(sample_rate)
        self.floor_rel = float(floor_rel)
        self.ref_distance = float(ref_distance)
        for n_fft in self.n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def _window(self, n_fft: int) -> torch.Tensor:
        return getattr(self, f"window_{n_fft}")

    def _stft(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        """``[B, M, T]`` -> complex ``[B, M, F, N]``."""
        b, m, t = x.shape
        window = self._window(n_fft).to(device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x.reshape(b * m, t),
            n_fft=n_fft,
            hop_length=max(1, int(n_fft * self.hop_ratio)),
            window=window,
            return_complex=True,
            center=True,
        )
        return spec.reshape(b, m, spec.shape[-2], spec.shape[-1])

    @staticmethod
    def _resample(grid: torch.Tensor, freqs: int, frames: int) -> torch.Tensor:
        """``[B, K, n_env, n_grid]`` -> ``[B, K, freqs, frames]``."""
        b, k = grid.shape[0], grid.shape[1]
        out = torch.nn.functional.interpolate(
            grid.reshape(b * k, 1, grid.shape[-2], grid.shape[-1]),
            size=(frames, freqs),
            mode="bilinear",
            align_corners=True,
        )
        return out.reshape(b, k, frames, freqs).transpose(-1, -2)

    def forward(
        self,
        target: torch.Tensor,
        rel_pos: torch.Tensor,
        source_psd: torch.Tensor,
        wind_psd: torch.Tensor,
    ) -> torch.Tensor:
        """Mean spatial NLL of ``target``.

        Args:
            target: ``[B, M, T]`` recorded multichannel audio.
            rel_pos: ``[B, M, R, 3]`` rotor->mic vectors.
            source_psd: ``[B, R, n_env, n_grid]`` per-rotor emitted power.
            wind_psd: ``[B, M, n_env, n_grid]`` per-microphone incoherent power.
        """
        total = target.new_zeros(())
        for n_fft in self.n_ffts:
            spec = self._stft(target, n_fft)  # [B, M, F, N]
            n_freqs, n_frames = spec.shape[-2], spec.shape[-1]
            freqs = torch.linspace(
                0.0, self.sample_rate / 2, n_freqs, device=target.device, dtype=target.dtype
            )
            d = steering_vectors(rel_pos, freqs, ref_distance=self.ref_distance)
            window_power = self._window(n_fft).to(target.dtype).pow(2).sum()
            src = self._resample(source_psd, n_freqs, n_frames) * window_power
            wind = self._resample(wind_psd, n_freqs, n_frames) * window_power
            floor = self.floor_rel * spec.detach().abs().pow(2).mean()
            total = total + spatial_whittle_nll(spec, d, src, wind + floor).mean()
        return total / len(self.n_ffts)


class SpatialLikelihoodLoss(nn.Module):
    """Frame adapter around :class:`SpatialLikelihood`.

    Needs the generator's **per-rotor** source power and **per-microphone** wind
    power separately — summing them (as the marginal path does) destroys exactly
    the structure that makes them identifiable.
    """

    def __init__(
        self,
        *,
        n_channels: int | None = 1,
        sr: tuple[int, int] = AUDIO_RATE,
        target_key: str = "audio",
        source_key: str = "source_psd",
        wind_key: str = "wind_psd",
        rel_pos_key: str = "rel_pos",
        **core_kwargs: Any,
    ) -> None:
        super().__init__()
        self.core = SpatialLikelihood(**core_kwargs)
        self.target_key = target_key
        self.source_key = source_key
        self.wind_key = wind_key
        self.rel_pos_key = rel_pos_key
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({source_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def forward(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        return self.core(
            get_tensor(target, self.target_key),
            get_tensor(pred, self.rel_pos_key),
            get_tensor(pred, self.source_key),
            get_tensor(pred, self.wind_key),
        )
