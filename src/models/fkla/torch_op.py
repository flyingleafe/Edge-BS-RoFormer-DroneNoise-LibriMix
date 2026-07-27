"""Vendored from flyingleafe/kla-loglinear@11e5a39 (``src/fkla/torch_op.py``).

FLAT-KLA SUBSET ONLY: the exact flat readout (``flat_kla`` — the fold branch
of the dyadic tree, O(T) parallel prefix products, no sequential loop) and the
helpers it needs (``_compose``/``_lift``/``_element_tree``/``_prefix_messages``).
Deleted relative to upstream: the Fenwick multi-level readout (sentinel/dyadic
buckets, ``_transport_message``, ``_dead_zone_mu``, ``_prefix_products``,
``fenwick_kla`` itself), all Triton kernels, and the truncated/adaptive paths.
Code kept literal otherwise.

Why vendored: kla-loglinear is a private GitHub repo — cluster nodes cannot
fetch private git dependencies, so a pip/uv git-dep is not deployable there.

Conventions (0-indexed positions):

- dynamics abar, pbar: TIME-INVARIANT per-(N, D) parameters (KLA §4.4)
- per-token evidence phi[t,n,d] = k[t,n]^2 * lam_v[t,d],
  kappa[t,n,d] = k[t,n] * lam_v[t,d] * v[t,d]
- fold at t: ordered product (newest left) of the per-token scan elements,
  applied to the flat prior — the exact flat-KLA posterior (merge exactness).

Readout: y[t] = q[t] . mu[t], mu = eta / clamp(lam, eps) — KLA Algorithm 1.
"""

from __future__ import annotations

import torch


def _compose(newer: torch.Tensor, older: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) @ (..., 3, 3), renormalised per cell (projective)."""
    out = newer @ older
    scale = out.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    return out / scale


def _lift(abar, pbar, phi, kappa) -> torch.Tensor:
    """Per-token elements, shape (B, T, N, D, 3, 3)."""
    B, T, N, D = phi.shape
    a = abar.expand(B, T, N, D) if abar.dim() == 2 else abar
    p = pbar.expand(B, T, N, D) if pbar.dim() == 2 else pbar
    L = phi.new_zeros(B, T, N, D, 3, 3)
    a2 = a * a
    L[..., 0, 0] = a
    L[..., 0, 1] = kappa * p
    L[..., 0, 2] = kappa * a2
    L[..., 1, 1] = 1.0 + p * phi
    L[..., 1, 2] = a2 * phi
    L[..., 2, 1] = p
    L[..., 2, 2] = a2
    return L


def _element_tree(L: torch.Tensor) -> list[torch.Tensor]:
    """Dyadic tree: levels[j] has shape (B, T // 2^j, N, D, 3, 3), node m
    covering [m * 2^j, (m+1) * 2^j)."""
    levels = [L]
    cur = L
    while cur.shape[1] >= 2:
        M = cur.shape[1] // 2
        pairs = cur[:, : 2 * M]
        older = pairs[:, 0::2]
        newer = pairs[:, 1::2]
        cur = _compose(newer, older)
        levels.append(cur)
    return levels


def _prefix_messages(levels: list[torch.Tensor]) -> torch.Tensor:
    """Inclusive-prefix fold MESSAGES (e, x, w) at every position: the
    element applied to the flat prior (0, 0, 1) — i.e. the last column of
    the inclusive prefix product. Avoids materialising the final full-T
    3x3 compose: (L @ pre)[..., :, 2] == L @ pre[..., :, 2], a matrix-
    vector product (3x less work/memory for the largest op in the fold)."""
    J = len(levels) - 1
    if J == 0:
        msg = levels[0][..., :, 2]
        return msg / msg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    # exclusive prefixes down to level 1, then one level-0 half-step
    B = levels[0].shape[0]
    grid = levels[0].shape[2:-2]
    eye = torch.eye(3, dtype=levels[0].dtype, device=levels[0].device)
    pre = eye.expand(B, levels[J].shape[1], *grid, 3, 3)
    for j in range(J - 1, 0, -1):
        M = levels[j].shape[1]
        Mp = levels[j + 1].shape[1]
        even = pre
        odd = _compose(levels[j][:, 0 : 2 * Mp : 2], pre)
        out = torch.stack([even, odd], dim=2).flatten(1, 2)
        if 2 * Mp < M:
            tail = _compose(levels[j + 1][:, Mp - 1 : Mp], pre[:, Mp - 1 : Mp])
            out = torch.cat([out, tail], dim=1)
        pre = out
    # level 0 expansion, on 3-vectors: pre-column then apply L_t
    M = levels[0].shape[1]
    Mp = levels[1].shape[1]
    pre_col = pre[..., :, 2]  # (B, Mp, ..., 3)
    pre_col = pre_col / pre_col.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    even_col = pre_col  # nodes 0,2,...
    odd_col = torch.einsum("...ij,...j->...i", levels[0][:, 0 : 2 * Mp : 2], pre_col)
    odd_col = odd_col / odd_col.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    col = torch.stack([even_col, odd_col], dim=2).flatten(1, 2)  # (B, 2Mp, ..., 3)
    if 2 * Mp < M:
        tail_col = torch.einsum(
            "...ij,...j->...i", levels[1][:, Mp - 1 : Mp], pre_col[:, Mp - 1 : Mp]
        )
        tail_col = tail_col / tail_col.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        col = torch.cat([col, tail_col], dim=1)
    msg = torch.einsum("...ij,...j->...i", levels[0], col)  # inclusive at t
    return msg / msg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)


def flat_kla(
    abar: torch.Tensor,  # (N, D)
    pbar: torch.Tensor,  # (N, D)
    k: torch.Tensor,  # (B, T, N)
    v: torch.Tensor,  # (B, T, D)
    lam_v: torch.Tensor,  # (B, T, D), positive
    q: torch.Tensor,  # (B, T, N)
    fold_weight: torch.Tensor | None = None,  # (B, T) or None (= all-ones)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exact flat KLA (Algorithm 1 readout y_t = q_t . mu_t) via the fold
    branch of the tree — O(T) parallel prefix products, no sequential loop.

    Upstream's ``flat_kla`` calls ``fenwick_kla(..., level_weights=None,
    fold_weight=ones)``; this is that call path inlined (the fold branch of
    upstream ``fenwick_kla``, verbatim), with the layer's learned fold weight
    folded in (upstream's flat layer path applies it right after the op).

    Returns y: (B, T, D). Keep inputs fp32/fp64 — half-precision Möbius
    composition chains are numerically unacceptable (measured upstream: bf16
    p99 rel err 12% at T=100).
    """
    element_dtype = k.dtype
    read_dtype = torch.float64 if element_dtype == torch.float64 else torch.float32

    phi = k.pow(2).unsqueeze(-1) * lam_v.unsqueeze(-2)  # (B,T,N,D)
    kappa = k.unsqueeze(-1) * (lam_v * v).unsqueeze(-2)  # (B,T,N,D)

    abar_e, pbar_e = abar.to(element_dtype), pbar.to(element_dtype)
    L = _lift(abar_e, pbar_e, phi.to(element_dtype), kappa.to(element_dtype))
    tree = _element_tree(L)

    # fold level: exact flat posterior (Blelloch prefix products)
    fold_msg = _prefix_messages(tree).to(read_dtype)  # (B,T,N,D,3)
    lam_f = fold_msg[..., 1] / fold_msg[..., 2].clamp_min(1e-30)
    eta_f = fold_msg[..., 0] / fold_msg[..., 2].clamp_min(1e-30)
    mu_f = eta_f / lam_f.clamp_min(eps)
    y = torch.einsum("btn,btnd->btd", q, mu_f)
    if fold_weight is not None:
        y = fold_weight.unsqueeze(-1) * y
    return y
