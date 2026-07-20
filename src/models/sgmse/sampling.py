"""Predictor-Corrector reverse-SDE sampler — faithful port of sgmse/sampling.

Reverse-diffusion predictor + annealed Langevin dynamics (ALD) corrector, the
SGMSE+ enhancement defaults (predictor='reverse_diffusion', corrector='ald',
snr=0.5, 1 corrector step, denoise=True). Per PC step the corrector updates
FIRST, then the predictor; the final ``x_mean`` (one-step-denoised) is returned.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def reverse_diffusion_predictor(sde, score_fn: ScoreFn, x, y, t, stepsize):
    """One reverse-diffusion (ancestral) predictor step (sgmse ReverseDiffusionPredictor)."""
    dt = stepsize
    drift, diffusion = sde.sde(x, y, t)
    f = drift * dt
    G = diffusion * torch.sqrt(dt)
    score = score_fn(x, y, t)
    rev_f = f - G[:, None, None, None] ** 2 * score
    z = torch.randn_like(x)
    x_mean = x - rev_f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


def annealed_langevin_corrector(sde, score_fn: ScoreFn, x, y, t, snr, n_steps):
    """Annealed Langevin dynamics corrector (sgmse AnnealedLangevinDynamics)."""
    std = sde.marginal_prob(x, y, t)[1]
    x_mean = x
    for _ in range(n_steps):
        grad = score_fn(x, y, t)
        noise = torch.randn_like(x)
        step_size = (snr * std) ** 2 * 2
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]
    return x, x_mean


def get_pc_sampler(
    sde, score_fn: ScoreFn, y, *, denoise=True, eps=3e-2, snr=0.5, corrector_steps=1
):
    """Build the PC sampler function (sgmse ``get_pc_sampler``, OUVE defaults)."""

    def pc_sampler():
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            for i in range(sde.N):
                t = timesteps[i]
                stepsize = t - timesteps[i + 1] if i != len(timesteps) - 1 else timesteps[-1]
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, _ = annealed_langevin_corrector(
                    sde, score_fn, xt, y, vec_t, snr, corrector_steps
                )
                xt, xt_mean = reverse_diffusion_predictor(sde, score_fn, xt, y, vec_t, stepsize)
            return xt_mean if denoise else xt

    return pc_sampler
