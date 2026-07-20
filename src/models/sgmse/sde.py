"""OUVE SDE — faithful port of sgmse/sdes.py ``OUVESDE`` (Richter et al. 2023).

Ornstein-Uhlenbeck Variance-Exploding SDE, complex STFT domain:

    dx = theta (y - x) dt + sigma(t) dw,
    sigma(t) = sigma_min (sigma_max/sigma_min)^t sqrt(2 log(sigma_max/sigma_min)).

Closed-form perturbation kernel p(x_t | x0, y) = N(mu, std^2 I) with
mu = e^{-theta t} x0 + (1 - e^{-theta t}) y and std given by ``_std``. Default
theta=1.5, sigma_min=0.05, sigma_max=0.5, N=30, T=1.
"""

from __future__ import annotations

import numpy as np
import torch


class OUVESDE:
    def __init__(self, theta=1.5, sigma_min=0.05, sigma_max=0.5, N=30, **ignored):
        self.theta = float(theta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.logsig = float(np.log(self.sigma_max / self.sigma_min))
        self.N = int(N)

    @property
    def T(self) -> float:
        return 1.0

    def copy(self) -> OUVESDE:
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    def sde(self, x, y, t):
        drift = self.theta * (y - x)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, y, t):
        exp_interp = torch.exp(-self.theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            / (theta + logsig)
        )

    def marginal_prob(self, x0, y, t):
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y):
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T
