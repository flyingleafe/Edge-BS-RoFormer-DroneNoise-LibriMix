import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.real, x.imag], dim=1)


def _channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def _complex_randn(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.complex(torch.randn(shape, device=device), torch.randn(shape, device=device))


class DiffusionScoreNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionBufferModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        audio_cfg = config.audio
        stft_cfg = audio_cfg.get("stft", {})
        self.sample_rate = int(audio_cfg.get("sample_rate", 16000))
        self.n_fft = int(stft_cfg.get("n_fft", audio_cfg.get("n_fft", 510)))
        self.win_length = int(stft_cfg.get("win_length", self.n_fft))
        self.hop_length = int(stft_cfg.get("hop_length", audio_cfg.get("hop_length", 256)))
        self.window_type = stft_cfg.get("window", "hann_periodic")

        preprocess_cfg = config.get("preprocessing", {}).get("magnitude_compression", {})
        self.comp_beta = float(preprocess_cfg.get("beta", 0.15))
        self.comp_alpha = float(preprocess_cfg.get("alpha", 0.5))

        model_cfg = config.get("model", {})
        base_ch = int(model_cfg.get("base_channels", 64))

        diffusion_cfg = config.get("diffusion_buffer", {})
        self.buffer_size = int(diffusion_cfg.get("buffer_size", 20))
        self.t_eps = float(diffusion_cfg.get("t_eps", 0.03))
        self.t_max = float(diffusion_cfg.get("t_max", 0.8))

        sde_cfg = config.get("sde", {})
        diffusion_sde_cfg = sde_cfg.get("diffusion", {})
        self.sde_c = float(diffusion_sde_cfg.get("c", 0.08))
        self.sde_k = float(diffusion_sde_cfg.get("k", 2.6))

        self.score_net = DiffusionScoreNet(in_ch=5, base_ch=base_ch)

        window = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("stft_window", window)
        self._build_sigma_table()

    def _build_sigma_table(self, steps: int = 2048) -> None:
        t_grid = torch.linspace(0.0, self.t_max, steps)
        g = self.sde_c * (self.sde_k ** t_grid)
        denom = (1.0 - t_grid).clamp(min=1e-6)
        integrand = (g ** 2) / (denom ** 2)
        integral = torch.cumsum((integrand[:-1] + integrand[1:]) * (t_grid[1:] - t_grid[:-1]) * 0.5, dim=0)
        integral = torch.cat([torch.zeros(1), integral], dim=0)
        sigma_sq = ((1.0 - t_grid) ** 2) * integral
        sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-12))
        self.register_buffer("sigma_t_grid", t_grid)
        self.register_buffer("sigma_table", sigma)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, self.t_max)
        t_flat = t.reshape(-1)
        idx = torch.searchsorted(self.sigma_t_grid, t_flat, right=False)
        idx = torch.clamp(idx, 1, self.sigma_t_grid.numel() - 1)
        t0 = self.sigma_t_grid[idx - 1]
        t1 = self.sigma_t_grid[idx]
        s0 = self.sigma_table[idx - 1]
        s1 = self.sigma_table[idx]
        w = (t_flat - t0) / (t1 - t0 + 1e-12)
        sigma = s0 + w * (s1 - s0)
        return sigma.reshape(t.shape).to(t.device)

    def _g(self, t: torch.Tensor) -> torch.Tensor:
        return self.sde_c * (self.sde_k ** t)

    def _stft(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(wave.device),
            center=False,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(spec.device),
            center=False,
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

    def _make_time_map(self, t_vals: torch.Tensor, frames: int) -> torch.Tensor:
        batch, bsz = t_vals.shape
        t_map = torch.zeros((batch, 1, 1, frames), device=t_vals.device)
        t_map[..., -bsz:] = t_vals.view(batch, 1, 1, bsz)
        return t_map

    def _score(self, v: torch.Tensor, y: torch.Tensor, t_map: torch.Tensor) -> torch.Tensor:
        v_ch = _complex_to_channels(v)
        y_ch = _complex_to_channels(y)
        x = torch.cat([v_ch, y_ch, t_map], dim=1)
        return self.score_net(x)

    def _prepare_mono(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2:
            return audio
        return audio.mean(dim=1)

    def forward(self, mix: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        if target is not None:
            return self._forward_train(mix, target)
        return self._forward_infer(mix)

    def _forward_train(self, mix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mix_mono = self._prepare_mono(mix)
        target_mono = self._prepare_mono(target)

        y = self._compress(self._stft(mix_mono))
        s = self._compress(self._stft(target_mono))

        batch, _, frames = y.shape
        bsz = min(self.buffer_size, frames)
        if bsz < 1:
            raise ValueError("Buffer size must be at least 1.")

        t_rand = torch.rand((batch, bsz - 1), device=y.device) * (self.t_max - self.t_eps) + self.t_eps
        t_rand, _ = torch.sort(t_rand, dim=1)
        t_vals = torch.cat([torch.full((batch, 1), self.t_eps, device=y.device), t_rand], dim=1)

        sigma = self._sigma(t_vals).view(batch, 1, 1, bsz)
        z = _complex_randn((batch, y.shape[1], bsz), device=y.device)

        y_last = y[..., -bsz:]
        s_last = s[..., -bsz:]
        t_vals_broadcast = t_vals.view(batch, 1, 1, bsz)
        mu = (1.0 - t_vals_broadcast) * s_last + t_vals_broadcast * y_last

        v = s.clone()
        v[..., -bsz:] = mu + sigma * z

        t_map = self._make_time_map(t_vals, frames)
        score = self._score(v, y, t_map)
        score_last = score[..., -bsz:]

        target_score = -z / (sigma + 1e-12)
        target_score = _complex_to_channels(target_score)

        return F.mse_loss(score_last, target_score)

    def _forward_infer(self, mix: torch.Tensor) -> torch.Tensor:
        batch, channels, length = mix.shape
        mix_mono = self._prepare_mono(mix)

        y = self._compress(self._stft(mix_mono))
        frames = y.shape[-1]
        bsz = min(self.buffer_size, frames)
        if bsz < 1:
            raise ValueError("Buffer size must be at least 1.")

        t_schedule = torch.linspace(self.t_eps, self.t_max, bsz, device=y.device).unsqueeze(0).repeat(batch, 1)
        sigma_schedule = self._sigma(t_schedule).view(batch, 1, 1, bsz)
        t_map = self._make_time_map(t_schedule, frames)

        v = torch.zeros_like(y)
        output = y.clone()

        t_vals = t_schedule.view(batch, 1, 1, bsz)
        g_vals = self._g(t_vals)
        dt = torch.zeros(bsz, device=y.device)
        dt[0] = -t_schedule[:, 0].mean()
        dt[1:] = (t_schedule[:, :-1] - t_schedule[:, 1:]).mean(dim=0)
        dt = dt.view(1, 1, 1, bsz)

        for k in range(frames):
            v = torch.roll(v, shifts=-1, dims=-1)
            noise = _complex_randn((batch, y.shape[1]), device=y.device)
            v[..., -1] = y[..., k] + sigma_schedule[..., -1].squeeze(-1) * noise

            score = self._score(v, y, t_map)
            score_last = score[..., -bsz:]
            score_last_c = _channels_to_complex(score_last)

            v_last = v[..., -bsz:]
            y_last = y[..., -bsz:]
            f = (y_last - v_last) / torch.clamp(1.0 - t_vals, min=1e-6)
            v_last = v_last + (-f + (g_vals ** 2) * score_last_c) * dt
            v[..., -bsz:] = v_last

            out_index = k - bsz + 1
            if out_index >= 0:
                output[..., out_index] = v[..., -bsz]

        output = self._decompress(output)
        wave = self._istft(output, length=length)
        wave = wave.unsqueeze(1)

        if channels > 1:
            wave = wave.repeat(1, channels, 1)
        return wave
