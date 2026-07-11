"""
Noise-modeller architectures ported from `drone_audition.models.harmonic_gen_new`.

These predict per-harmonic amplitudes and filtered-noise magnitudes from motor
speeds (RPS) and synthesise audio via an oscillator bank + time-varying filter.
The VP-transform machinery the source kept in the same file now lives in the
sibling `harmonic_transform` module.

Self-contained — no `env.settings` dependency; sample rate is an explicit
constructor argument (default 16 kHz to match this repo).

Classes
-------
- `HarmonicNoiseGenNew`        : end-to-end RPS -> audio noise generator
- `JointAmplitudePredictor`    : CNN predicting harmonic + noise amplitudes (+ optional F0s)
- `ConstantAmplitudePredictor` : learnable constant harmonic/noise distributions
- `DirectionalOutputHead`      : z-conditioned output head (positional bias + input correction)
- `SpeedsPostprocessingWrapper`: post-processes predicted F0s after the main net
- `LearnableTimeShift`         : soft time shift via a learned narrow Gaussian kernel
- `BetterAmplitudePredictor`   : placeholder (no forward) kept for parity
- `SimpleHarmonicNoiseGen`     : random-phase harmonic synthesiser (DEPRECATED)
- `PropellerAmplitudePredictor`: per-propeller amplitude CNN (DEPRECATED)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio as TA
from einops import rearrange, repeat
from torch import nn

from .dsp import (
    frequency_filter,
    harmonic_freq_series,
    harmonic_oscillator_bank,
    oscillator_bank,
    upsample_with_windows,
)
from .math_utils import exp_sigmoid
from .nn import CausalConv1dBlock

# ---------------------------------------------------------------------------
# Noise modellers
# ---------------------------------------------------------------------------


class HarmonicNoiseGenNew(nn.Module):
    """Generates harmonic noise from motor speeds.

    A network predicts harmonic amplitudes and filtered-noise magnitudes; the
    harmonics are synthesised via an oscillator bank and (optionally) a
    filtered-noise residual is added.
    """

    def __init__(
        self,
        net=None,
        n_harmonics=100,
        use_diff_noise=True,
        use_random_phases=False,
        use_z=False,
        rps_jitter_sigma: float = 0.0,
        rps_jitter_tau: float = 0.05,
        stft_window_len=2048,
        stft_hop_len=512,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__()

        if net is None:
            net = JointAmplitudePredictor(n_harmonics=n_harmonics, **kwargs)

        self.net = net
        self.n_harmonics = n_harmonics
        self.use_diff_noise = use_diff_noise
        self.use_random_phases = use_random_phases
        self.use_z = use_z
        self.rps_jitter_sigma = float(rps_jitter_sigma)
        self.rps_jitter_tau = float(rps_jitter_tau)
        self.sample_rate = sample_rate

        if use_random_phases:
            self.stft = TA.transforms.Spectrogram(
                n_fft=stft_window_len, hop_length=stft_hop_len, power=None
            )
            self.istft = TA.transforms.InverseSpectrogram(
                n_fft=stft_window_len, hop_length=stft_hop_len
            )

    def _apply_rps_jitter(self, f0s: torch.Tensor) -> torch.Tensor:
        """Add a stochastic Ornstein-Uhlenbeck perturbation to the fundamental.

        Real rotor speeds carry a fast zero-mean jitter that telemetry-conditioned
        generation cannot know; it broadens the ``k``-th harmonic by ``+/- k *
        sigma`` Hz. Adding ``delta(t)`` to the fundamental ``f0s`` (which is then
        multiplied by the harmonic index in :func:`harmonic_freq_series`) makes
        every harmonic of a rotor share one *coherent*, frequency-proportional
        broadening -- exactly the physical effect. See
        ``scripts/calibrate_rps_jitter.py`` for how ``sigma``/``tau`` were fit.

        The OU sample obeys the Euler-Maruyama recursion::

            delta[n+1] = delta[n] * (1 - dt/tau) + sigma * sqrt(2*dt/tau) * N(0,1)
            delta[0]   ~ N(0, sigma)

        Cost note: ``f0s`` arrives at audio rate (``sample_rate``), but the jitter
        bandwidth is ``~1/(2*pi*tau)`` (a few Hz), so an audio-rate scan would be
        wasteful and numerically pointless. We instead run the recursion on a
        coarse control grid (``dt ~ tau/10``, so Euler stays accurate) and linearly
        interpolate up to audio rate. ``delta`` carries no gradient (pure noise);
        adding it leaves ``f0s``'s autograd graph intact.

        Injecting on ``f0s`` (not on the raw ``ms`` fed to the amplitude net) keeps
        the predicted ``harm_amps``/``noise_amps`` -- and therefore the Stage-2
        smoothness regularisers -- unaffected.
        """
        b, o, t = f0s.shape
        tau = self.rps_jitter_tau
        sigma = self.rps_jitter_sigma
        duration = t / self.sample_rate

        # control grid: dt ~ tau/10 (Euler-accurate), but never coarser than 50 Hz.
        ctrl_dt = min(tau / 10.0, 1.0 / 50.0)
        n_intervals = max(1, int(np.ceil(duration / ctrl_dt)))
        n_ctrl = n_intervals + 1
        dt = duration / n_intervals  # exact grid spanning the clip
        alpha = 1.0 - dt / tau
        noise_scale = sigma * float(np.sqrt(2.0 * dt / tau))

        bo = b * o
        eps = torch.randn(bo, n_intervals, device=f0s.device, dtype=f0s.dtype)
        d = sigma * torch.randn(bo, device=f0s.device, dtype=f0s.dtype)
        cols = [d]
        for n in range(n_intervals):
            d = d * alpha + noise_scale * eps[:, n]
            cols.append(d)
        delta = torch.stack(cols, dim=1).reshape(bo, 1, n_ctrl)  # [B*O, 1, n_ctrl]
        delta = F.interpolate(delta, size=t, mode="linear", align_corners=True)
        return f0s + delta.reshape(b, o, t)

    def gen_harm_noise(self, freqs, amps, initial_phases=None):
        harm_noise = oscillator_bank(
            freqs, amps, initial_phases=initial_phases, sr=self.sample_rate
        )

        if self.use_random_phases:
            mag = self.stft(harm_noise).abs()
            ang = torch.exp(1j * (torch.randn_like(mag) * torch.pi))

            S = mag * ang
            harm_noise = self.istft(S)

        return harm_noise

    def forward(self, ms, z=None, initial_phases=None, return_dict=False, rps_jitter=None):
        """Synthesise audio from motor speeds.

        Initial harmonic phases (``[*freqs.shape[:-1]]`` = per (batch, oscillator,
        harmonic)):
        * if ``initial_phases`` is given, it is used verbatim (train or eval) —
          the optional inference override;
        * else, in **training** mode a fresh random phase per harmonic is sampled
          each call (phase augmentation: the amplitude/magnitude model must not
          rely on a fixed phase alignment);
        * else, in **eval** mode phases are **zero** (deterministic, reproducible
          synthesis).

        RPS jitter (``rps_jitter_sigma > 0``) mirrors the same train/eval
        convention: a fresh OU perturbation is added to the fundamental in
        **training** mode and **off** at eval, unless ``rps_jitter`` overrides it
        (``True``/``False``). See :meth:`_apply_rps_jitter`.
        """
        b, o, t = ms.shape

        # Motor speeds are not necessarily processed
        if self.use_z:
            harm_amps, noise_amps, f0s = self.net(ms, z)
        else:
            harm_amps, noise_amps, f0s = self.net(ms)

        do_jitter = self.training if rps_jitter is None else rps_jitter
        if self.rps_jitter_sigma > 0.0 and do_jitter:
            f0s = self._apply_rps_jitter(f0s)

        freqs = harmonic_freq_series(f0s, self.n_harmonics)

        amps_t = harm_amps.shape[-1]
        up_factor = t // amps_t

        if up_factor > 1:
            amps = rearrange(harm_amps, "b o c t -> (b o c) t 1")
            amps_up = upsample_with_windows(amps, up_factor * amps_t)
            amps_up = rearrange(amps_up, "(b o c) t 1 -> b o c t", b=b, o=o)

            if up_factor * amps_t < t:
                amps_up = F.pad(amps_up, (0, t - up_factor * amps_t))
        else:
            amps_up = harm_amps

        assert amps_up.shape == freqs.shape

        if initial_phases is None and self.training:
            # Phase augmentation during training only; eval => None => zero phase.
            initial_phases = (
                torch.rand(freqs.shape[:-1], device=freqs.device, dtype=freqs.dtype) * 2 * torch.pi
            )

        harm_noise = self.gen_harm_noise(freqs, amps_up, initial_phases)

        if self.use_diff_noise:
            norm_noise = torch.randn((b, t), dtype=ms.dtype, device=ms.device)
            noise_amps = torch.cat([noise_amps, noise_amps[:, :, -1:]], dim=-1)
            diff_noise = frequency_filter(norm_noise, noise_amps.transpose(-1, -2))

            audio = harm_noise.sum(-2) + diff_noise
        else:
            audio = harm_noise.sum(-2)
            diff_noise = None

        if not return_dict:
            return audio
        else:
            return {
                "harm_amps": harm_amps,
                "noise_amps": noise_amps,
                "f0s": f0s,
                "harm_noise": harm_noise,
                "diff_noise": diff_noise,
                "audio": audio,
            }


class SpeedsPostprocessingWrapper(nn.Module):
    """Modifies f0s additionally after the main module returned the amplitudes."""

    def __init__(self, main_net: nn.Module, postprocessor: nn.Module):
        super().__init__()
        self.main_net = main_net
        self.postprocessor = postprocessor

    def forward(self, ms):
        harm_amps, noise_amps, f0s = self.main_net(ms)
        f0s = self.postprocessor(f0s)
        return harm_amps, noise_amps, f0s


class LearnableTimeShift(nn.Module):
    """Learns a soft time shift via a narrow, heavily-centered Gaussian kernel."""

    def __init__(self, width=2049):
        super().__init__()
        self.width = width
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(width / 20.0))

    def forward(self, ms):
        rng = torch.arange(-self.width / 2, self.width / 2, device=ms.device, dtype=ms.dtype)
        pdf = torch.exp(-0.5 * ((rng - self.mu) / self.sigma) ** 2) / (
            self.sigma * np.sqrt(2 * torch.pi)
        )
        pdf = rearrange(pdf, "k -> 1 1 k")

        b, o, t = ms.shape
        ms = rearrange(ms, "b o t -> (b o) 1 t")
        ms_pad = F.pad(ms, (self.width // 2, self.width // 2), mode="replicate")
        shifted = F.conv1d(ms_pad, pdf, padding="valid")
        shifted = rearrange(shifted, "(b o) 1 t -> b o t", b=b, o=o)
        return shifted


class ConstantAmplitudePredictor(nn.Module):
    """Constant harmonic distribution per channel and constant noise distribution."""

    def __init__(
        self, n_oscillators=4, n_harmonics=100, noise_mags=60, harm_hop_size=1, noise_hop_size=512
    ):
        super().__init__()
        self.harmonic_dist = nn.Parameter(torch.randn((n_oscillators, n_harmonics)))
        self.noise_dist = nn.Parameter(torch.randn(noise_mags))
        self.harm_hop_size = harm_hop_size
        self.noise_hop_size = noise_hop_size

    def forward(self, ms):
        b, o, t = ms.shape
        t_harm = t // self.harm_hop_size + 1
        t_noise = t // self.noise_hop_size + 1

        harm_amps = repeat(exp_sigmoid(self.harmonic_dist), "o k -> b o k t", b=b, t=t_harm)
        noise_amps = repeat(exp_sigmoid(self.noise_dist), "k -> b k t", b=b, t=t_noise)

        return harm_amps, noise_amps, ms


class DirectionalOutputHead(nn.Module):
    """Double-linear layer with biases derived from `z` and per-input corrections."""

    def __init__(self, in_ch, out_ch, z_dim):
        super().__init__()
        self.positional_bias = nn.Linear(z_dim, out_ch)
        self.inp_corrections = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x, z):
        A = self.positional_bias(z)
        Ad = self.inp_corrections(x)
        return A.unsqueeze(-2) + Ad


class JointAmplitudePredictor(nn.Module):
    """Predict amplitudes of harmonics and filtered noise from motor speeds."""

    def __init__(
        self,
        n_oscillators=4,
        n_harmonics=100,
        noise_amps=60,
        entry_window=2048,
        entry_hop=512,
        n_blocks=3,
        dilated=False,
        predict_f0s=False,
        z_dim=0,
        film=False,
        film_spectral_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_oscillators = n_oscillators
        self.noise_amps = noise_amps
        self.z_dim = z_dim
        # FiLM conditioning modulates the backbone features by z (per-drone
        # spectral envelope AND dynamics); the alternative (z_dim>0, film=False)
        # only adds a z-derived bias to the output heads (static envelope).
        self.film = bool(film) and z_dim > 0

        self.entry = CausalConv1dBlock(
            n_oscillators, 32, entry_window, entry_hop, padding_mode="reflect"
        )

        self.meat = nn.Sequential(
            *[
                CausalConv1dBlock(
                    32 * (2**i), 32 * (2 ** (i + 1)), 3, 1, dilation=2**i if dilated else 1
                )
                for i in range(n_blocks)
            ]
        )

        last_ch = 32 * (2**n_blocks)
        self.last_ch = last_ch

        if self.z_dim == 0:
            self.harm_out = nn.Linear(last_ch, n_harmonics * n_oscillators)
            self.noise_out = nn.Linear(last_ch, noise_amps)
        elif self.film:
            # z -> (gamma, beta) over the backbone channels. Init near identity
            # (gamma~1, beta~0) for stable starts, but with a small *nonzero*
            # weight so the embedding receives gradient from step 1 (a zeroed
            # weight makes d(out)/dz = 0 and the embedding never moves).
            self.film_gen = nn.Linear(self.z_dim, 2 * last_ch)
            with torch.no_grad():
                self.film_gen.weight.mul_(0.1)
                self.film_gen.bias[:last_ch].fill_(1.0)
                self.film_gen.bias[last_ch:].zero_()
            if film_spectral_norm:
                # Bound the Lipschitz constant of the z -> (gamma, beta) map so
                # the decoder stays smooth around each conditioning code (vicinal
                # sampling / interpolation regulariser). NOTE: the spectral-norm
                # parametrization changes state-dict keys (weight ->
                # parametrizations.weight.original + power-iteration buffers), so
                # this is a NEW-training-only option — plain checkpoints do not
                # load into it and vice versa.
                self.film_gen = torch.nn.utils.parametrizations.spectral_norm(self.film_gen)
            self.harm_out = nn.Linear(last_ch, n_harmonics * n_oscillators)
            self.noise_out = nn.Linear(last_ch, noise_amps)
        else:
            self.harm_out = DirectionalOutputHead(last_ch, n_harmonics * n_oscillators, self.z_dim)
            self.noise_out = DirectionalOutputHead(last_ch, noise_amps, self.z_dim)

        if predict_f0s:
            self.f0s_out = nn.Linear(last_ch, n_oscillators)
        else:
            self.f0s_out = None

    def forward(self, ms, z=None):
        x = self.entry(ms)
        x = self.meat(x)
        x = rearrange(x, "b c t -> b t c")

        if self.z_dim > 0:
            assert z is not None and z.shape[-1] == self.z_dim
            if self.film:
                gamma, beta = self.film_gen(z).split(self.last_ch, dim=-1)  # each [B, C]
                x = gamma.unsqueeze(-2) * x + beta.unsqueeze(-2)  # modulate [B, t, C]
                harms = self.harm_out(x)
                noise_mags = self.noise_out(x)
            else:
                harms = self.harm_out(x, z)
                noise_mags = self.noise_out(x, z)
        else:
            harms = self.harm_out(x)
            noise_mags = self.noise_out(x)

        if self.f0s_out:
            f0s = torch.sigmoid(self.f0s_out(x)) * 180.0 + 20.0
            f0s = rearrange(f0s, "b t o -> b o t")

            if ms.shape[-1] > f0s.shape[-1]:
                f0s = F.interpolate(f0s, size=ms.shape[-1], align_corners=True, mode="linear")
        else:
            f0s = ms

        harms = rearrange(harms, "b t (o h) -> b o h t", o=self.n_oscillators)
        noise_mags = rearrange(noise_mags, "b t n -> b n t")

        return exp_sigmoid(harms), exp_sigmoid(noise_mags), f0s


class BetterAmplitudePredictor(nn.Module):
    """Predict amplitudes and correct motor speeds (placeholder — no forward yet)."""

    def __init__(
        self,
        n_oscillators=4,
        n_harmonics=100,
        noise_amps=60,
        entry_window=2048,
        entry_hop=512,
        **kwargs,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_oscillators = n_oscillators
        self.noise_amps = noise_amps


# ---------------------------------------------------------------------------
# Simpler stuff (DEPRECATED)
# ---------------------------------------------------------------------------


class SimpleHarmonicNoiseGen(nn.Module):
    """Generates harmonic noise from one propeller with random phases (DEPRECATED)."""

    def __init__(
        self,
        amp_predictor: nn.Module | None = None,
        n_props: int = 4,
        n_harmonics: int = 100,
        n_fft=2048,
        hop_length=512,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.n_props = n_props
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate

        self.harmonic_dist = nn.Parameter(torch.randn((n_props, n_harmonics)))
        self.stft = TA.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.istft = TA.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        self.amp_predictor = amp_predictor

    def gen(self, ms):
        b, o, t = ms.shape

        if self.amp_predictor is None:
            amps = repeat(exp_sigmoid(self.harmonic_dist), "o k -> b o k t", b=b, t=t)
        else:
            A, C_corrections = self.amp_predictor(ms)
            A_up = upsample_with_windows(A.transpose(-1, -2), t).transpose(-1, -2)
            C_corr_up = upsample_with_windows(C_corrections.transpose(-1, -2), t).transpose(-1, -2)

            amps = A_up.unsqueeze(-2) * (exp_sigmoid(self.harmonic_dist).unsqueeze(-1) + C_corr_up)

        # scales amplitudes down so that it's easier to output sensible amplitudes
        return harmonic_oscillator_bank(ms, 0.1 * amps, sr=self.sample_rate)

    def gen_magnitude(self, ms):
        wav = self.gen(ms)
        return self.stft(wav).abs()

    def forward(self, ms, return_dict=False):
        mag = self.gen_magnitude(ms)
        ang = torch.exp(1j * (torch.randn_like(mag) * torch.pi))

        S = mag * ang
        wav_components = self.istft(S)
        wav = wav_components.sum(-2)

        if return_dict:
            return {
                "audio": wav,
                "components": wav_components,
                "mags": mag,
            }
        else:
            return wav


class PropellerAmplitudePredictor(nn.Module):
    """Per-propeller amplitude/correction CNN (DEPRECATED)."""

    def __init__(
        self, n_harmonics=100, total_stride=512, entry_filter_dim=128, entry_filter_stride=32
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.total_stride = total_stride

        self.entry = CausalConv1dBlock(
            1, 16, entry_filter_dim, entry_filter_stride, padding_mode="reflect"
        )
        self.meat = nn.Sequential(
            *[
                CausalConv1dBlock(16 * (2**i), 16 * (2 ** (i + 1)), 3, 1, dilation=2**i)
                for i in range(3)
            ]
        )

        self.A_out = nn.Linear(128, 1)
        self.C_out = nn.Linear(128, n_harmonics)

    def forward(self, ms):
        o = ms.shape[-2]
        ms = rearrange(ms, "b o t -> (b o) 1 t")

        x = self.entry(ms)
        x = self.meat(x)

        x = rearrange(x, "... c t -> ... t c")
        A = exp_sigmoid(self.A_out(x))
        C = exp_sigmoid(self.C_out(x))

        A = rearrange(A, "(b o) t 1 -> b o t", o=o)
        C = rearrange(C, "(b o) t k -> b o k t", o=o)

        return A, C
