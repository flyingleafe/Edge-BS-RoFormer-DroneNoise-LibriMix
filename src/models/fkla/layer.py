"""Vendored from flyingleafe/kla-loglinear@11e5a39 (``src/fkla/layer.py``).

FLAT-KLA SUBSET ONLY: ``FenwickKLALayer`` slimmed to its exact flat path
(``use_levels=False, use_fold=True``, no Triton — the plain-torch ``flat_kla``
op) and renamed ``FlatKLALayer``; ``FenwickKLABlock`` kept verbatim as
``FlatKLABlock``. Deleted relative to upstream: Fenwick level readout
(``w_proj`` reduced to the single fold column), all Triton/truncated/adaptive
routing, ``readout_mode``/``uncertainty="iso"``/E5-ablation knobs, and the LM
wrappers. Deviation: ``p_init`` is an argument (upstream hardcodes the KLA
default 0.01) so the config can mirror the CKLA ``pnoise`` gain fix
(``p_init=1.0`` — see ``conf/model/simple_conv_v2_ckla_pnoise.yaml``).

Why vendored: kla-loglinear is a private GitHub repo — cluster nodes cannot
fetch private git dependencies, so a pip/uv git-dep is not deployable there.

Scaffolding follows KLA (arXiv:2602.10743 Fig. 7 / Appendix A): causal
conv1d(k=4) + SiLU on the mixer input, QK L2-norm, expansion 1, gated
residual, RMSNorm. Dynamics A, P, Delta are learnable time-invariant (N, D)
parameters, OU-discretised (their §4.1/4.4; Delta init in [0.001, 0.1],
decay a log-spaced S4D-style). Lambda_v positivity: softplus.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.fkla.torch_op import flat_kla


class FlatKLALayer(nn.Module):
    """Flat-KLA sequence mixer — upstream ``FenwickKLALayer`` with
    ``use_levels=False, use_fold=True, use_triton=False`` baked in.

    Set ``layer.capture = []`` to record the post-cast scan inputs each
    forward (opt-in analysis/test tap, mirrors upstream's)."""

    def __init__(
        self,
        d_model: int,
        n_state: int = 16,
        conv_kernel: int = 4,
        p_init: float = 0.01,
    ):
        super().__init__()
        self.d_model, self.n_state = d_model, n_state

        self.conv = nn.Conv1d(
            d_model, d_model, conv_kernel, groups=d_model, padding=conv_kernel - 1, bias=True
        )
        self.k_proj = nn.Linear(d_model, n_state, bias=False)
        self.q_proj = nn.Linear(d_model, n_state, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.lamv_proj = nn.Linear(d_model, d_model, bias=True)
        # Upstream: w_proj = Linear(d_model, max_levels + 1); the flat path
        # reads only the last (fold) column — kept as a 1-wide projection.
        self.w_proj = nn.Linear(d_model, 1, bias=True)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

        # OU params: a log-spaced (S4D-style), p = p_init (upstream 0.01),
        # Delta in [0.001, 0.1] (KLA Table 10); stored in softplus-inverse form.
        a0 = torch.logspace(math.log10(0.5), math.log10(8.0), n_state)
        a0 = a0.unsqueeze(-1).expand(n_state, d_model).contiguous()
        self.a_param = nn.Parameter(torch.log(torch.expm1(a0)))
        self.p_param = nn.Parameter(torch.log(torch.expm1(torch.full((n_state, d_model), p_init))))
        dt0 = torch.exp(
            torch.rand(n_state, d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.dt_param = nn.Parameter(torch.log(torch.expm1(dt0)))

        self.capture: list[dict[str, Tensor]] | None = None

    def ou_discretise(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = F.softplus(self.a_param)
        p = F.softplus(self.p_param)
        dt = F.softplus(self.dt_param)
        abar = torch.exp(-a * dt)
        pbar = (p**2 / (2 * a)) * (1 - torch.exp(-2 * a * dt))
        return abar, pbar

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        _, T, _ = x.shape
        h = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        h = F.silu(h)

        k = F.normalize(self.k_proj(h), dim=-1)  # QK-norm
        q = F.normalize(self.q_proj(h), dim=-1)
        v = self.v_proj(h)
        lam_v = F.softplus(self.lamv_proj(h)) + 1e-4
        fold_w = F.softplus(self.w_proj(h))[..., 0]  # (B, T) — upstream wl[..., -1]

        abar, pbar = self.ou_discretise()
        # fp32 discipline for the scan-element algebra: under bf16 autocast
        # the projections emit bf16, but half-precision Moebius composition
        # chains are numerically unacceptable (see torch_op.flat_kla).
        if k.dtype not in (torch.float32, torch.float64):
            k, q, v, lam_v = k.float(), q.float(), v.float(), lam_v.float()
            fold_w = fold_w.float()
        if self.capture is not None:
            # opt-in analysis tap: the exact tensors the scan op consumes,
            # post-cast. `layer.capture = []` to enable.
            self.capture.append(
                {
                    "abar": abar.detach().cpu(),
                    "pbar": pbar.detach().cpu(),
                    "k": k.detach().cpu(),
                    "lam_v": lam_v.detach().cpu(),
                    "v": v.detach().cpu(),
                    "q": q.detach().cpu(),
                    "fold_w": fold_w.detach().cpu(),
                }
            )
        y = flat_kla(abar, pbar, k, v, lam_v, q, fold_weight=fold_w)
        y = self.norm(y) * F.silu(self.gate_proj(x))
        return self.out_proj(y)


class FlatKLABlock(nn.Module):
    """Pre-norm residual block: x + mixer(norm(x)), + MLP sub-block —
    upstream ``FenwickKLABlock`` verbatim, with the flat mixer."""

    def __init__(self, d_model: int, mlp_ratio: int = 4, **mixer_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = FlatKLALayer(d_model, **mixer_kwargs)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.SiLU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))
