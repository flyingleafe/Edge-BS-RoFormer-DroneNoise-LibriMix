"""Inference core of the per-drone egonoise GP (checkpoint load + posterior query).

Relocated verbatim from ``experiments.gp_rotor_noise.train_egonoise_gp`` so
that :mod:`data_processing.gp_noise` (the ``kind: gp`` online-mix source) no
longer imports from the ``experiments`` sandbox (import-linter contract
"nothing imports experiments"). The training side (``fit``, dataset streaming,
evaluation, CLI) stays in ``experiments.gp_rotor_noise.train_egonoise_gp``,
whose ``EgonoiseGPModel`` subclasses this one — experiments importing
data_processing is the permitted direction.

Contents: the ``FS`` render rate, the :class:`EgonoiseGPConfig` dataclass,
the batched Matern-5/2 GP constructor (:func:`_make_gp`) and
:class:`EgonoiseGPModel` — posterior-mean coefficient prediction, broadband
synthesis and (de)serialisation. Checkpoints (``best.pt``) are plain
``torch.save`` dicts, so files written by either class load in both.
"""

# pyright: reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOptionalSubscript=false, reportOptionalCall=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportCallIssue=false
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

FS = 44100.0


def _make_gp(train_x: torch.Tensor, train_y: torch.Tensor):
    """One independent ARD Matern-5/2 GP per Fourier coefficient (batched).

    ``train_x`` (N, D) standardized; ``train_y`` (N, T).  Same construction as
    ``jasa_gp._make_gp`` but with ``D`` input dims (here D=4: xyz + rps).
    """
    import gpytorch

    n_tasks = train_y.shape[-1]
    n_dims = train_x.shape[-1]
    batch = torch.Size([n_tasks])

    class BatchGP(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_dims, batch_shape=batch),
                batch_shape=batch,
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch)
    tx = train_x.unsqueeze(0).expand(n_tasks, *train_x.shape).contiguous()
    ty = train_y.t().contiguous()
    model = BatchGP(tx, ty, likelihood)
    return model, likelihood, tx, ty


@dataclass
class EgonoiseGPConfig:
    drone: str = "dregon"
    n_harm: int = 60
    iters: int = 400
    lr: float = 0.05
    train_seed: int = 0
    holdout_rps: tuple[float, ...] = (60.0,)
    n_blades: int = 2  # overwritten from meta at fit time
    verbose: bool = True
    extra: dict = field(default_factory=dict)


class EgonoiseGPModel:
    """Per-drone GP: predict Fourier coefficients at (mic xyz, rps).

    This base class carries the fitted state and everything needed at
    consumption time (``load``/``save``/``predict_coeffs``/``_broadband``).
    Training (``fit``) and time-domain rendering (``synthesize``) live in the
    ``experiments.gp_rotor_noise.train_egonoise_gp`` subclass.
    """

    def __init__(self, cfg: EgonoiseGPConfig | None = None):
        self.cfg = cfg or EgonoiseGPConfig()
        self.model = None
        self.likelihood = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        # broadband model: sigma_b(rps) table + global colored magnitude shape
        self.bb_rps_grid: np.ndarray | None = None
        self.bb_sigma: np.ndarray | None = None
        self.bb_freqs: np.ndarray | None = None
        self.bb_mag: np.ndarray | None = None
        self._tx: torch.Tensor | None = None
        self._ty: torch.Tensor | None = None

    # ── prediction ──────────────────────────────────────────────────────────
    def predict_coeffs(self, mics: np.ndarray, rps: float) -> np.ndarray:
        """Posterior-mean Fourier coefficients at each mic for one rps -> (M, 2H+1)."""
        import gpytorch

        assert self.model is not None
        M = mics.shape[0]
        xq = np.concatenate([np.asarray(mics, np.float64), np.full((M, 1), rps)], axis=1)
        xqs = torch.tensor((xq - self.x_mean) / self.x_std, dtype=torch.float32)
        n_tasks = self.y_mean.shape[0]
        xqb = xqs.unsqueeze(0).expand(n_tasks, M, xqs.shape[-1]).contiguous()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mu = self.model(xqb).mean.numpy()  # (T, M)
        return mu.T * self.y_std + self.y_mean

    def _broadband(self, n_mics: int, n: int, rps: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sigma = float(np.interp(rps, self.bb_rps_grid, self.bb_sigma))
        f_new = np.fft.rfftfreq(n, 1.0 / FS)
        mag = np.interp(f_new, self.bb_freqs, self.bb_mag)
        out = np.empty((n_mics, n))
        for m in range(n_mics):
            spec = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, len(mag)))
            sig = np.fft.irfft(spec, n)
            out[m] = sig / (np.sqrt(np.mean(sig**2)) + 1e-12) * sigma
        return out

    # ── (de)serialisation ───────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "model": self.model.state_dict(),
                "likelihood": self.likelihood.state_dict(),
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "bb_rps_grid": self.bb_rps_grid,
                "bb_sigma": self.bb_sigma,
                "bb_freqs": self.bb_freqs,
                "bb_mag": self.bb_mag,
                "train_x": self._tx,
                "train_y": self._ty,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> EgonoiseGPModel:
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["cfg"]
        cfg["holdout_rps"] = tuple(cfg.get("holdout_rps", ()))
        m = cls(EgonoiseGPConfig(**cfg))
        for k in (
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "bb_rps_grid",
            "bb_sigma",
            "bb_freqs",
            "bb_mag",
        ):
            setattr(m, k, state[k])
        tx, ty = state["train_x"], state["train_y"]
        model, likelihood, _, _ = _make_gp(
            torch.zeros(tx.shape[1], tx.shape[2]), torch.zeros(tx.shape[1], ty.shape[0])
        )
        model.set_train_data(inputs=tx, targets=ty, strict=False)
        model.load_state_dict(state["model"])
        likelihood.load_state_dict(state["likelihood"])
        model.eval()
        likelihood.eval()
        m.model, m.likelihood, m._tx, m._ty = model, likelihood, tx, ty
        return m
