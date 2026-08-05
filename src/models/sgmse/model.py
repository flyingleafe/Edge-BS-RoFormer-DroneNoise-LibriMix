"""SGMSE+ (Richter et al., IEEE TASLP 2023) — score-based diffusion speech
enhancement, faithful pure-PyTorch port trained from scratch at native 16 kHz.

Integration mirrors ``models.diffusion_buffer.DiffusionBufferModel`` exactly so
the task-generic training loop needs no changes: ``forward(mix, target=None)``
returns the **denoising-score-matching (sigma^2-weighted) scalar loss** when a
clean ``target`` is given (training), and runs the **predictor-corrector reverse
SDE sampler** to return the enhanced waveform when ``target`` is None (eval).
Built through ``models.registry.LEGACY_MODEL_BUILDERS`` via the legacy
``model_type: sgmse`` registry entry (same dispatch as ``diffusion_buffer``).

Pieces:
  * complex STFT (n_fft=510, hop=128, Hann, center) + magnitude compression
    ``S = c |X|^alpha e^{i angle(X)}`` (c=0.15, alpha=0.5), applied to clean and
    noisy identically;
  * OUVE SDE (:mod:`.sde`), NCSN++ backbone (:mod:`.ncsnpp`), PC sampler
    (:mod:`.sampling`);
  * network input = cat([x_t, y], dim=1) (2 complex ch = 4 real ch in, 2 real /
    1 complex ch score out); score = -dnn(input, t);
  * parameter EMA (decay 0.999): EMA weights are swapped in on ``.eval()`` and
    swapped back out on ``.train()`` (mirrors the reference ``ScoreModel.train``),
    and the EMA is updated once per training ``forward``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .ncsnpp import NCSNpp
from .sampling import get_pc_sampler
from .sde import OUVESDE


class _EMA:
    """Minimal parameter EMA (torch_ema-free): shadow = decay*shadow + (1-decay)*param."""

    def __init__(self, parameters, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in parameters]
        self.collected: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, parameters) -> None:
        for s, p in zip(self.shadow, parameters):
            s.sub_((1.0 - self.decay) * (s - p.detach()))

    @torch.no_grad()
    def store(self, parameters) -> None:
        self.collected = [p.detach().clone() for p in parameters]

    @torch.no_grad()
    def copy_to(self, parameters) -> None:
        for s, p in zip(self.shadow, parameters):
            p.data.copy_(s.data)

    @torch.no_grad()
    def restore(self, parameters) -> None:
        if self.collected is None:
            return
        for c, p in zip(self.collected, parameters):
            p.data.copy_(c.data)

    def to(self, device) -> None:
        self.shadow = [s.to(device) for s in self.shadow]
        if self.collected is not None:
            self.collected = [c.to(device) for c in self.collected]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class SGMSEModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        audio_cfg = config.audio
        stft_cfg = _cfg_get(audio_cfg, "stft", {})
        self.sample_rate = int(_cfg_get(audio_cfg, "sample_rate", 16000))
        self.n_fft = int(_cfg_get(stft_cfg, "n_fft", _cfg_get(audio_cfg, "n_fft", 510)))
        self.win_length = int(_cfg_get(stft_cfg, "win_length", self.n_fft))
        self.hop_length = int(
            _cfg_get(stft_cfg, "hop_length", _cfg_get(audio_cfg, "hop_length", 128))
        )

        comp_cfg = _cfg_get(_cfg_get(config, "preprocessing", {}), "magnitude_compression", {})
        self.comp_beta = float(_cfg_get(comp_cfg, "beta", 0.15))  # spec_factor c
        self.comp_alpha = float(_cfg_get(comp_cfg, "alpha", 0.5))  # spec_abs_exponent

        sde_cfg = _cfg_get(config, "sde", {})
        self.t_eps = float(_cfg_get(sde_cfg, "t_eps", 0.03))
        self.sde = OUVESDE(
            theta=float(_cfg_get(sde_cfg, "theta", 1.5)),
            sigma_min=float(_cfg_get(sde_cfg, "sigma_min", 0.05)),
            sigma_max=float(_cfg_get(sde_cfg, "sigma_max", 0.5)),
            N=int(_cfg_get(sde_cfg, "N", 30)),
        )

        model_cfg = _cfg_get(config, "model", {})
        ch_mult = _cfg_get(model_cfg, "ch_mult", (1, 1, 2, 2, 2, 2, 2))
        ch_mult = tuple(int(c) for c in ch_mult)
        attn_res = _cfg_get(model_cfg, "attn_resolutions", (16,))
        attn_res = tuple(int(a) for a in attn_res)
        self.dnn = NCSNpp(
            scale_by_sigma=bool(_cfg_get(model_cfg, "scale_by_sigma", True)),
            nf=int(_cfg_get(model_cfg, "nf", 128)),
            ch_mult=ch_mult,
            num_res_blocks=int(_cfg_get(model_cfg, "num_res_blocks", 2)),
            attn_resolutions=attn_res,
            fourier_scale=int(_cfg_get(model_cfg, "fourier_scale", 16)),
            image_size=int(_cfg_get(model_cfg, "image_size", 256)),
            embedding_type=str(_cfg_get(model_cfg, "embedding_type", "fourier")),
            dropout=float(_cfg_get(model_cfg, "dropout", 0.0)),
        )
        # U-Net downsamples by 2^(num_resolutions-1); spatial dims must be divisible.
        self.pad_mult = 2 ** (len(ch_mult) - 1)

        sampling_cfg = _cfg_get(config, "sampling", {})
        self.sampler_snr = float(_cfg_get(sampling_cfg, "snr", 0.5))
        self.sampler_corrector_steps = int(_cfg_get(sampling_cfg, "corrector_steps", 1))
        self.sampler_denoise = bool(_cfg_get(sampling_cfg, "denoise", True))

        window = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("stft_window", window)

        self.ema = _EMA(self.dnn.parameters(), float(_cfg_get(model_cfg, "ema_decay", 0.999)))
        self._ema_active = False

    # ─── EMA-aware train/eval + device transfer ──────────────────────────────

    def train(self, mode: bool = True):
        if mode and self._ema_active:
            self.ema.restore(self.dnn.parameters())
            self._ema_active = False
        elif (not mode) and (not self._ema_active):
            self.ema.store(self.dnn.parameters())
            self.ema.copy_to(self.dnn.parameters())
            self._ema_active = True
        return super().train(mode)

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        self.ema.to(next(self.dnn.parameters()).device)
        return res

    # ─── STFT + magnitude compression ────────────────────────────────────────

    def _window(self, device: torch.device) -> torch.Tensor:
        win = self.stft_window
        assert isinstance(win, torch.Tensor)
        return win.to(device)

    def _stft(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window(wave.device),
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window(spec.device),
            center=True,
            length=length,
        )

    def _compress(self, spec: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        mag_c = self.comp_beta * torch.pow(mag + 1e-12, self.comp_alpha)
        return torch.polar(mag_c, phase)

    def _decompress(self, spec: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        mag = torch.pow(torch.clamp(mag, min=1e-12) / self.comp_beta, 1.0 / self.comp_alpha)
        return torch.polar(mag, phase)

    def _pad_spec(self, spec: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad a complex ``[B, F, T]`` spectrogram's F and T up to ``pad_mult``.

        Returns the padded spec and the original ``(F, T)`` for later cropping.
        """
        f, t = spec.shape[-2], spec.shape[-1]
        pad_f = (self.pad_mult - f % self.pad_mult) % self.pad_mult
        pad_t = (self.pad_mult - t % self.pad_mult) % self.pad_mult
        if pad_f or pad_t:
            spec = torch.nn.functional.pad(spec, (0, pad_t, 0, pad_f))
        return spec, f, t

    def _prepare_mono(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3:
            return audio.mean(dim=1)
        return audio

    def _score(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score s_theta(x_t, y, t) = -dnn(cat([x_t, y], 1), t) (complex [B,1,F,T])."""
        dnn_input = torch.cat([x_t, y], dim=1)
        return -self.dnn(dnn_input, t)

    # ─── forward: DSM loss (train) / PC-sampler enhancement (eval) ────────────

    def forward(self, mix: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        if target is not None:
            return self._forward_train(mix, target)
        return self._forward_infer(mix)

    def _forward_train(self, mix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mix_mono = self._prepare_mono(mix)
        target_mono = self._prepare_mono(target)
        norm = mix_mono.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        mix_mono = mix_mono / norm
        target_mono = target_mono / norm

        y = self._compress(self._stft(mix_mono))
        x0 = self._compress(self._stft(target_mono))
        y, _, _ = self._pad_spec(y)
        x0, _, _ = self._pad_spec(x0)
        y = y.unsqueeze(1)  # [B, 1, F, T]
        x0 = x0.unsqueeze(1)

        t = torch.rand(x0.shape[0], device=x0.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x0, y, t)
        z = torch.randn_like(x0)  # complex normal (per-component var 0.5)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z

        score = self._score(x_t, y, t)
        losses = torch.square(torch.abs(score * sigma + z))  # sigma^2-weighted DSM, Eq. (7)
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        if self.training:
            self.ema.update(self.dnn.parameters())
        return loss

    def _forward_infer(self, mix: torch.Tensor) -> torch.Tensor:
        length = mix.shape[-1]
        channels = mix.shape[1] if mix.ndim == 3 else 1
        mix_mono = self._prepare_mono(mix)
        norm = mix_mono.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        mix_mono = mix_mono / norm

        y = self._compress(self._stft(mix_mono))
        y, f0, t0 = self._pad_spec(y)
        y = y.unsqueeze(1)  # [B, 1, F, T]

        sampler = get_pc_sampler(
            self.sde,
            self._score,
            y,
            denoise=self.sampler_denoise,
            eps=self.t_eps,
            snr=self.sampler_snr,
            corrector_steps=self.sampler_corrector_steps,
        )
        spec = sampler()  # [B, 1, F, T] complex
        spec = spec[:, 0, :f0, :t0]  # crop padding
        spec = self._decompress(spec)
        wave = self._istft(spec, length=length)
        wave = wave * norm
        wave = wave.unsqueeze(1)  # [B, 1, L]
        if channels > 1:
            wave = wave.repeat(1, channels, 1)
        return wave
