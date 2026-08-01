"""Spectral likelihood for generators with a *stochastic* component.

Why this exists
---------------
The multi-scale magnitude loss (:mod:`losses.spectral`) compares **one
realization** of the generator against **one realization** of the recording.
That is the correct thing to do for a deterministic signal and the wrong thing
to do for a random one. A rotor recording is both: a coherent harmonic comb
(deterministic given the shaft phase) plus broadband aerodynamic noise and wind
gusts (a random process, of which the recording shows a single draw nobody can
reproduce).

Asking a random component to match a particular draw sample-by-sample has two
consequences, both measured in ``tests/losses/test_spectral_likelihood.py``:

1. **The minimizer is biased.** For a bin whose content is circular complex
   Gaussian, ``|X|`` is Rayleigh, and the minimizer of an L1 magnitude loss is
   the *median* of that Rayleigh rather than its RMS. The fitted power is too
   low by a factor ``ln 2`` — a systematic **-1.6 dB** on any purely stochastic
   component, independent of training length or capacity.
2. **The gradient is noise.** Each draw pulls the parameters a different way;
   only the (biased) mean survives averaging, so the stochastic branch learns
   far more slowly than the deterministic one it competes with.

The fix is to stop predicting a realization and start predicting a
*distribution*: the coherent branch supplies a **mean**, the stochastic
branches supply a **variance**, and the objective is the likelihood of the
observed spectrum under that model. Nothing is ever sampled during training.

The model
---------
Per STFT bin, with the coherent field's complex amplitude ``mu`` and the total
incoherent power ``sigma2``::

    X = mu * exp(j theta) + CN(0, sigma2)

The absolute phase ``theta`` is **not identifiable**: the generator does not
know the recording's initial rotor phases, and per-harmonic phase noise makes
the phase decohere anyway (see the coupled-VK draft, sec. "Demodulation and the
lock statistic": the phase of harmonic ``k`` is ``k*phi(t) + b_k(t) + psi_k``
with ``b_k`` Brownian). So ``theta`` is marginalized out under a uniform prior,
which turns the complex-Gaussian likelihood into a **Rice** likelihood on the
magnitude ``r = |X|``::

    p(r) = (2r/sigma2) exp(-(r^2 + a^2)/sigma2) I0(2 r a / sigma2),   a = |mu|

Dropping the parameter-free ``log(2r)``, the negative log-likelihood is::

    NLL = log(sigma2) + (r - a)^2 / sigma2 - log I0e(2 r a / sigma2)

written here in the algebraically equivalent form that is numerically stable at
both ends (``(r^2 + a^2 - 2ra) = (r - a)^2`` absorbs the Bessel argument's
growth, and ``I0e(z) = I0(z) e^{-z}`` never overflows).

Both limits are the textbook estimator, which is what makes this trustworthy:

- ``a = 0`` (bin is pure noise): the Bessel term vanishes and the NLL becomes
  ``log sigma2 + r^2/sigma2`` — the **Whittle** likelihood of a Gaussian process
  with power spectral density ``sigma2``, whose minimizer is ``sigma2 = E|X|^2``.
  Unbiased, unlike the magnitude loss it replaces.
- ``sigma2 -> 0`` (bin is pure tone): the NLL is dominated by ``(r - a)^2``,
  i.e. ordinary magnitude matching, so nothing is lost where the old loss was
  already right.

Phase noise (partial coherence)
-------------------------------
A harmonic that decoheres over the analysis window is neither fully coherent nor
fully random. With phase-diffusion coherence ``gamma = |E[e^{j b}]| in [0, 1]``
the mean is damped to ``gamma*a`` and the power it loses, ``(1 - gamma^2) a^2``,
reappears as variance. :func:`split_coherence` applies that transfer, so one
scalar per harmonic band moves a component continuously between the two limits
above rather than forcing a binary choice.
"""

from __future__ import annotations

from typing import Any

import tdseries as td
import torch
from torch import nn

from losses._common import AUDIO_RATE, audio_series_spec, get_tensor
from tasks.spec import FrameSpec

__all__ = [
    "rice_nll",
    "split_coherence",
    "SpectralLikelihood",
    "SpectralLikelihoodLoss",
]


def rice_nll(
    r: torch.Tensor,
    a: torch.Tensor,
    sigma2: torch.Tensor,
    *,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Phase-marginalized negative log-likelihood of magnitude ``r``.

    The exact NLL of ``|mu e^{j theta} + CN(0, sigma2)|`` with ``theta`` uniform
    and ``a = |mu|``, dropping the parameter-free ``log(2r)`` term.

    Args:
        r: observed magnitudes (any shape), ``>= 0``.
        a: predicted coherent magnitudes, broadcastable to ``r``, ``>= 0``.
        sigma2: predicted incoherent power, broadcastable to ``r``, ``> 0``.
        eps: floor on ``sigma2`` (an exactly-zero variance is an infinitely
            confident claim and would make the loss unbounded below).

    Returns:
        Elementwise NLL, same broadcast shape.
    """
    sigma2 = sigma2.clamp_min(eps)
    z = (2.0 * r * a / sigma2).clamp_min(0.0)
    # log I0(z) = log I0e(z) + z, and (r^2 + a^2)/sigma2 - z = (r - a)^2/sigma2.
    log_i0e = torch.special.i0e(z).clamp_min(eps).log()
    return sigma2.log() + (r - a).pow(2) / sigma2 - log_i0e


def split_coherence(
    a: torch.Tensor, sigma2: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move partially-decohered power from the mean into the variance.

    A component of amplitude ``a`` whose phase has coherence ``gamma`` over the
    analysis window contributes a *mean* of ``gamma * a`` and an extra *variance*
    of ``(1 - gamma^2) a^2``: total power ``a^2`` is conserved, and the split
    between "predictable" and "random" is exactly ``gamma``. ``gamma = 1``
    leaves the component fully coherent, ``gamma = 0`` makes it pure noise.

    Args:
        a: coherent magnitudes.
        sigma2: incoherent power to add to.
        gamma: per-component coherence in ``[0, 1]``, broadcastable to ``a``.

    Returns:
        ``(a_eff, sigma2_eff)``.
    """
    gamma = gamma.clamp(0.0, 1.0)
    return gamma * a, sigma2 + (1.0 - gamma.pow(2)) * a.pow(2)


class SpectralLikelihood(nn.Module):
    """Tensor-level Rice/Whittle spectral likelihood over one or more STFT scales.

    The generator supplies, per microphone:

    - ``coherent``: the deterministic waveform (harmonic bank, propagated) —
      its ``|STFT|`` is the Rice mean ``a``;
    - ``noise_psd``: the **power** spectral envelope of the *stochastic*
      branches on a uniform ``0..Nyquist`` grid, ``[..., n_env, n_grid]``, never
      sampled. Power rather than magnitude so that no ``sqrt``/square round trip
      sits in the graph: it is the identity analytically, but autograd evaluates
      it stepwise and ``inf * 0 = NaN`` in every bin the model predicts silent.

    ``noise_psd`` is resolution-agnostic on purpose: the loss resamples it onto
    whatever STFT grid each scale uses, so the model never has to commit to the
    analysis resolution. The conversion to expected bin power is
    ``sigma2 = noise_psd * ||w||^2`` (verified against a Monte-Carlo estimate
    in the tests) — ``||w||^2`` being the analysis window's power, the standard
    periodogram normalization for unit-variance white excitation.

    Multi-scale note: averaging the NLL over several ``n_ffts`` is an *ensemble*
    of likelihoods, not a joint likelihood of the data (the scales are not
    independent looks). It is kept because harmonic structure needs fine
    resolution while the noise envelope is better conditioned coarse; set a
    single ``n_ffts`` entry for a strictly proper likelihood.

    **Why the variance needs a floor.** A likelihood with a *learned* variance is
    unbounded below: wherever the mean fits well, ``log sigma2 -> -inf`` faster
    than ``(r - a)^2 / sigma2`` grows, so the optimizer can drive the loss to
    ``-inf`` by claiming infinite confidence on a handful of bins. In practice
    this diverges to NaN within a couple of epochs (observed on both the wind
    and no-wind arms). The floor is expressed **relative to the clip's own mean
    observed power**, not as an absolute number: audio level varies by orders of
    magnitude across clips and rigs, so an absolute floor is either inert on
    loud clips or dominant on quiet ones. ``floor_rel = 1e-6`` is 60 dB below
    the clip mean — well under any real noise floor, while still bounding the
    objective.

    Args:
        n_ffts: STFT sizes to score at.
        hop_ratio: hop as a fraction of each ``n_fft``.
        floor_rel: variance floor as a fraction of the clip's mean observed bin
            power (detached, so the floor is not itself a training target).
        noise_floor: additional absolute power floor, guarding the degenerate
            all-silent clip where the relative floor would be zero.
        beta: beta-NLL gradient balancing (Seitzer et al., "On the Pitfalls of
            Heteroscedastic Uncertainty Estimation"). Each bin's term is scaled
            by a **detached** ``sigma2**beta``, which leaves the optimum where it
            was but stops low-variance bins from dominating the gradient. ``0``
            recovers the plain NLL; ``0.5`` makes the mean's gradient scale like
            ``(r - a)/sigma`` instead of ``(r - a)/sigma^2``, which is what keeps
            the objective usable when the mean is still far off. **Set ``0`` for
            scoring**: the weight rescales the loss *value*, so a nonzero beta is
            no longer a proper scoring rule and shifts the argmin whenever sigma
            is shared across bins rather than predicted per bin.
        gamma_init: initial coherence if ``learn_coherence`` is set.
        learn_coherence: fit one global coherence scalar transferring harmonic
            power into the variance (see :func:`split_coherence`). Off by
            default so the loss stays a pure objective with no state.
    """

    def __init__(
        self,
        n_ffts: tuple[int, ...] | list[int] = (2048, 512),
        hop_ratio: float = 0.25,
        floor_rel: float = 1e-4,
        noise_floor: float = 1e-12,
        beta: float = 0.5,
        gamma_init: float = 0.9,
        learn_coherence: bool = False,
    ) -> None:
        super().__init__()
        self.n_ffts = tuple(int(n) for n in n_ffts)
        self.hop_ratio = float(hop_ratio)
        self.floor_rel = float(floor_rel)
        self.noise_floor = float(noise_floor)
        self.beta = float(beta)
        for n_fft in self.n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)
        self.raw_gamma: nn.Parameter | None = None
        if learn_coherence:
            g = torch.tensor(float(gamma_init)).clamp(1e-4, 1 - 1e-4)
            self.raw_gamma = nn.Parameter(torch.logit(g))

    def _window(self, n_fft: int) -> torch.Tensor:
        return getattr(self, f"window_{n_fft}")

    def _stft_mag(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        """``|STFT|`` of ``[B, T]`` -> ``[B, F, N]``."""
        window = self._window(n_fft).to(device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=max(1, int(n_fft * self.hop_ratio)),
            window=window,
            return_complex=True,
            center=True,
        )
        return spec.abs()

    def _resample_noise(self, noise_psd: torch.Tensor, n_fft: int, n_frames: int) -> torch.Tensor:
        """``[B, n_env, n_grid]`` magnitude response -> ``[B, F, n_frames]``.

        Bilinear over (frame, frequency); both axes are uniform grids over the
        same physical spans (clip duration, ``0..Nyquist``), so a plain resize
        is the correct resampling.
        """
        b = noise_psd.shape[0]
        grid = noise_psd.unsqueeze(1)  # [B, 1, n_env, n_grid]
        out = torch.nn.functional.interpolate(
            grid, size=(n_frames, n_fft // 2 + 1), mode="bilinear", align_corners=True
        )
        return out.reshape(b, n_frames, n_fft // 2 + 1).transpose(1, 2)

    def forward(
        self,
        target: torch.Tensor,
        coherent: torch.Tensor,
        noise_psd: torch.Tensor,
    ) -> torch.Tensor:
        """Mean NLL of ``target`` under the predicted (mean, variance) spectrum.

        Args:
            target: ``[B, T]`` recorded audio.
            coherent: ``[B, T]`` the generator's deterministic component.
            noise_psd: ``[B, n_env, n_grid]`` stochastic power envelope.

        Returns:
            Scalar loss.
        """
        total = target.new_zeros(())
        for n_fft in self.n_ffts:
            r = self._stft_mag(target, n_fft)
            a = self._stft_mag(coherent, n_fft)
            window_power = self._window(n_fft).to(target.dtype).pow(2).sum()
            psd = self._resample_noise(noise_psd, n_fft, r.shape[-1])
            # Per-clip relative floor — see the class docstring. Detached, so it
            # tracks the data's scale without being something the model can move.
            floor = self.floor_rel * r.detach().pow(2).mean(dim=(-2, -1), keepdim=True)
            sigma2 = psd.clamp_min(0.0) * window_power + floor + self.noise_floor
            if self.raw_gamma is not None:
                a, sigma2 = split_coherence(a, sigma2, torch.sigmoid(self.raw_gamma))
            nll = rice_nll(r, a, sigma2)
            if self.beta:
                nll = nll * sigma2.detach().pow(self.beta)
            total = total + nll.mean()
        return total / len(self.n_ffts)


class SpectralLikelihoodLoss(nn.Module):
    """Frame adapter around :class:`SpectralLikelihood`.

    Reads the recorded audio from ``target[target_key]`` and the generator's
    *distributional* prediction from two ``pred`` entries: ``coherent_key`` (the
    deterministic waveform) and ``noise_key`` (the stochastic power
    envelope). A generator that cannot supply those two entries cannot be
    trained with this loss — which is the point: predicting a distribution is a
    model capability, not a loss-side trick.
    """

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        target_key: str = "audio",
        coherent_key: str = "coherent",
        noise_key: str = "noise_psd",
        **core_kwargs: Any,
    ) -> None:
        super().__init__()
        self.core = SpectralLikelihood(**core_kwargs)
        self.target_key = target_key
        self.coherent_key = coherent_key
        self.noise_key = noise_key
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({coherent_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def forward(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        tgt = get_tensor(target, self.target_key)
        coh = get_tensor(pred, self.coherent_key)
        psd = get_tensor(pred, self.noise_key)
        # Fold any microphone axis into the batch; noise_psd carries the same
        # leading axes plus (n_env, n_grid).
        tgt2 = tgt.reshape(-1, tgt.shape[-1])
        coh2 = coh.reshape(-1, coh.shape[-1])
        psd3 = psd.reshape(-1, psd.shape[-2], psd.shape[-1])
        return self.core(tgt2, coh2, psd3)
