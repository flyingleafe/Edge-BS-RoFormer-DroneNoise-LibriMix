"""CKLA — Complex Kalman Linear Attention for RPS prediction.

Design of record: ``docs/ckla-design.md``. Substrate semantics: KLA (Kalman
Linear Attention, arXiv 2602.10743) as implemented in the ``fkla`` project's
``reference.py``/``layer.py`` — we vendor the *flat* information-form
recursion only (no Fenwick tree: our sequences are ~126–250 STFT frames,
where a sequential fp32 scan over T is cheap and exact) and generalize the
per-cell latent from a real Gaussian information pair to a **complex** one.
The sequential loop is the reference (it is kernel-launch-bound on GPU —
T × ~15 tiny kernels); on CUDA the fused Triton op (``models.ckla_triton``,
same stepwise arithmetic, one launch) is used by default (opt out via
``CKLA_NO_TRITON=1``). The log-depth associative scan
``complex_kla_scan_parallel`` is opt-in only (``CKLA_PARALLEL_SCAN=1``): its
span-gain products can blow up where the stepwise recurrences stay finite —
see the note at its ``use_parallel_scan`` attribute.

State grid G = (N, D): N state-expansion slots × D value channels. Per (n, d)
cell the latent is (η ∈ ℂ, λ ∈ ℝ≥0). The only change vs the real KLA flat
recursion is that the OU transition becomes complex, ā = e^{−γΔ + iω_t}:

    den_t = |ā_t|² + p̄_t · λ_{t−1}
    λ_t   = λ_{t−1} / den_t + φ_t              (real — identical to KLA)
    η_t   = ā_t · η_{t−1} / den_t + κ_t        (complex — rotation acts here)

This is exact, not an approximation: for a complex-Gaussian latent with
transition z_t = ā z_{t−1} + ε, ε ~ CN(0, p̄), the predicted covariance is
|ā|²Σ + p̄ — rotation is unitary and drops out of second moments, so the
proven real precision algebra is untouched and the rotation multiplies only
the information vector as a unit-modulus scalar. Input-dependent ω_t breaks
none of this because λ never sees ω.

Why: the rotation frequency ω_t[n] = ω0[n] + s[n]·(W_ω h_t)[n] is produced
from the layer input, so a stack of these layers can *close the loop* on
frequency tracking — block 1 forms a coarse frequency belief from mag+IF
evidence, block 2's rotation is conditioned on block 1's output. That is the
closed-loop ingredient the open-loop Kalman harmonic tracker (K2) lacked.
W_ω is zero-initialized so training starts LTI (a complex-OU LRU with
uncertainty) and input dependence grows only where gradients ask for it.

Implementation notes (design §1):

- η is stored as an explicit (re, im) pair of real tensors — no torch
  complex dtype, avoiding complex-autograd slow paths and keeping the op
  trivially portable.
- **fp32 mandatory** for all scan math below any autocast: the layer casts
  its projections to float when autocast produced half precision (same cast
  discipline as ``FenwickKLALayer.forward``), and the op body uses only
  elementwise mul/div/sum — none of which autocast downcasts — so the scan
  stays fp32 even inside an autocast region.
- No dependency on the kla-loglinear checkout at runtime (design §7): the
  op is self-contained in this repo.

Model wiring (design §3): ``SimpleConvV2CKLA`` = the ``SimpleConvV2Transformer``
trunk (front-end → 6× ResidualConvBlock2d → FrequencyAttentionPool →
(B, 128, T)) with the temporal transformer head replaced by
``TemporalCKLAHead``. Default front-end is ``stft_mag_if`` (G2b — the only
front-end that beat baseline). Registry keys ``simple_conv_v2_ckla`` /
``simple_conv_v2_ckla_mag`` in ``models.registry``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Literal, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.rps_predictor import (
    FrequencyAttentionPool,
    ResidualConvBlock2d,
    _remap_legacy_state_dict,
)

_TritonScan = Callable[..., tuple[Tensor, Tensor]]
_triton_scan_cache: _TritonScan | Literal[False] | None = None


def _get_triton_scan() -> _TritonScan | None:
    """Lazy accessor for the fused Triton op (``models.ckla_triton``).

    Deferred so that CPU-only imports of this module never pay the triton
    import cost; returns None when triton is unavailable.
    """
    global _triton_scan_cache
    if _triton_scan_cache is None:
        try:
            from models.ckla_triton import HAS_TRITON, complex_kla_scan_triton

            _triton_scan_cache = complex_kla_scan_triton if HAS_TRITON else False
        except ImportError:  # pragma: no cover - ckla_triton ships with this repo
            _triton_scan_cache = False
    return _triton_scan_cache if _triton_scan_cache else None


@overload
def complex_kla_scan(
    abar_mag: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    pbar: Tensor,
    k: Tensor,
    v: Tensor,
    lam_v: Tensor,
    q: Tensor,
    eps: float = ...,
    return_state: Literal[False] = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def complex_kla_scan(
    abar_mag: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    pbar: Tensor,
    k: Tensor,
    v: Tensor,
    lam_v: Tensor,
    q: Tensor,
    eps: float = ...,
    *,
    return_state: Literal[True],
) -> tuple[Tensor, Tensor, dict[str, Tensor]]: ...


def complex_kla_scan(
    abar_mag: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    pbar: Tensor,
    k: Tensor,
    v: Tensor,
    lam_v: Tensor,
    q: Tensor,
    eps: float = 1e-6,
    return_state: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Flat complex-KLA scan (design §1). Sequential loop over T, fp32/fp64.

    The complex transition is split as ā_t = abar_mag · (cos ω_t + i sin ω_t):
    the *magnitude* e^{−γΔ} is time-invariant per (slot, channel) — exactly
    the real-KLA discretised decay — while the rotation phase ω_t is
    per-step, per-slot (broadcast over channels). Evidence is formed inside
    the op from (k, v, λv): φ = k²·λv and κ = k·λv·v with κ_im = 0 (real
    value path — the state is still complex, rotation mixes the quadratures).

    Parameters
    ----------
    abar_mag : (N, D)
        Time-invariant decay magnitude e^{−γΔ}, in (0, 1].
    cos_w, sin_w : (B, T, N)
        Cosine/sine of the per-step rotation phase ω_t per slot.
    pbar : (N, D)
        Time-invariant discretised process noise, ≥ 0.
    k, q : (B, T, N)
        Key/query features per slot.
    v, lam_v : (B, T, D)
        Value and evidence precision per channel; lam_v > 0.
    eps : float
        Readout dead-zone: μ = η / max(λ, eps).
    return_state : bool
        When True, additionally return a dict of per-step stacked internals
        (§6 diagnostics): ``lam``, ``eta_re``, ``eta_im`` — each (B, T, N, D)
        post-update — and ``contrib`` (B, T, N), the per-slot readout
        contribution q_t[n] · ‖μ_t[n, :]‖ (L2 over channels of the complex
        posterior mean). The default path is byte-identical to
        ``return_state=False`` — the extra tensors are only stacked, never
        enter the y computation.

    Returns
    -------
    (y_re, y_im) : each (B, T, D)
        Real/imaginary parts of the readout y_t[d] = Σ_n q_t[n] · μ_t[n, d].
        With ``return_state=True`` a third element, the state dict above.
    """
    B, T, _ = k.shape
    D = v.shape[-1]
    N = abar_mag.shape[0]
    dtype, device = q.dtype, q.device

    a_mag2 = (abar_mag * abar_mag).unsqueeze(0)  # (1, N, D)
    pbar_b = pbar.unsqueeze(0)  # (1, N, D)

    lam = torch.zeros(B, N, D, dtype=dtype, device=device)
    eta_re = torch.zeros(B, N, D, dtype=dtype, device=device)
    eta_im = torch.zeros(B, N, D, dtype=dtype, device=device)

    ys_re: list[Tensor] = []
    ys_im: list[Tensor] = []
    st_lam: list[Tensor] = []
    st_eta_re: list[Tensor] = []
    st_eta_im: list[Tensor] = []
    st_contrib: list[Tensor] = []
    for t in range(T):
        k_t = k[:, t, :, None]  # (B, N, 1)
        q_t = q[:, t, :, None]
        v_t = v[:, t, None, :]  # (B, 1, D)
        lv_t = lam_v[:, t, None, :]
        phi = k_t * k_t * lv_t  # (B, N, D)
        kappa_re = k_t * lv_t * v_t  # κ_im = 0 (real value path)

        # den uses λ_{t−1}; rotation never enters (|ā|² only).
        den = a_mag2 + pbar_b * lam
        a_re = abar_mag * cos_w[:, t, :, None]  # (B, N, D)
        a_im = abar_mag * sin_w[:, t, :, None]
        new_eta_re = (a_re * eta_re - a_im * eta_im) / den + kappa_re
        new_eta_im = (a_re * eta_im + a_im * eta_re) / den
        lam = lam / den + phi
        eta_re, eta_im = new_eta_re, new_eta_im

        lam_safe = torch.clamp(lam, min=eps)
        ys_re.append((q_t * (eta_re / lam_safe)).sum(dim=1))  # (B, D)
        ys_im.append((q_t * (eta_im / lam_safe)).sum(dim=1))

        if return_state:
            # Diagnostics only — never feeds back into the y path above.
            st_lam.append(lam)
            st_eta_re.append(eta_re)
            st_eta_im.append(eta_im)
            mu_re = eta_re / lam_safe
            mu_im = eta_im / lam_safe
            mu_norm = torch.sqrt((mu_re * mu_re + mu_im * mu_im).sum(dim=-1))  # (B, N)
            st_contrib.append(q[:, t] * mu_norm)

    y_re, y_im = torch.stack(ys_re, dim=1), torch.stack(ys_im, dim=1)
    if not return_state:
        return y_re, y_im
    state = {
        "lam": torch.stack(st_lam, dim=1),  # (B, T, N, D)
        "eta_re": torch.stack(st_eta_re, dim=1),
        "eta_im": torch.stack(st_eta_im, dim=1),
        "contrib": torch.stack(st_contrib, dim=1),  # (B, T, N)
    }
    return y_re, y_im, state


_Elems = tuple[Tensor, ...]


def _assoc_scan(combine: Callable[[_Elems, _Elems], _Elems], elems: _Elems) -> _Elems:
    """Inclusive associative scan along dim 1 (time), log-depth pairwise style.

    ``combine(earlier, later)`` composes two scan elements ("apply *earlier*
    first, then *later*") and must be associative. Work is O(T) combines
    (T/2 + T/4 + … down-sweep plus the same up), depth is ⌈log₂ T⌉ — each
    level a handful of batched tensor ops over the whole remaining sequence,
    instead of the sequential loop's ~15 tiny kernels *per step*.
    """
    T = elems[0].shape[1]
    if T == 1:
        return elems
    even = tuple(e[:, 0::2] for e in elems)  # x₀, x₂, …  (⌈T/2⌉)
    odd = tuple(e[:, 1::2] for e in elems)  # x₁, x₃, …  (⌊T/2⌋)
    n_pairs = odd[0].shape[1]
    # Down-sweep: pair-reduce, then scan the half-length sequence.
    reduced = combine(tuple(e[:, :n_pairs] for e in even), odd)
    odd_scan = _assoc_scan(combine, reduced)  # inclusive results at 1, 3, 5, …
    # Up-sweep: even positions. r₀ = x₀; r_{2i} = x_{2i} ∘ r_{2i−1}.
    n_even = even[0].shape[1]
    if n_even > 1:
        tail = combine(tuple(s[:, : n_even - 1] for s in odd_scan), tuple(e[:, 1:] for e in even))
        even_res = tuple(torch.cat([e[:, :1], t], dim=1) for e, t in zip(even, tail))
    else:
        even_res = even  # T == 2: only r₀ = x₀ on the even side
    # Interleave: even results at 0, 2, …; odd results at 1, 3, …
    out: list[Tensor] = []
    for ev, od in zip(even_res, odd_scan):
        if ev.shape[1] == od.shape[1]:
            merged = torch.stack([ev, od], dim=2).flatten(1, 2)
        else:  # T odd — even side has one trailing element
            merged = torch.stack([ev[:, :-1], od], dim=2).flatten(1, 2)
            merged = torch.cat([merged, ev[:, -1:]], dim=1)
        out.append(merged)
    return tuple(out)


def _mobius_combine(earlier: _Elems, later: _Elems) -> _Elems:
    """Compose two 2×2 Möbius matrices (λ pass): M = M_later · M_earlier.

    Möbius maps are projective — M and c·M act identically — so after every
    combine the matrix is renormalized by the max-abs of its four entries
    (clamped, detached: the output is exactly scale-invariant, so detaching
    the normalizer changes no gradient). This keeps entries O(1) across the
    scan and prevents fp32 overflow/underflow over hundreds of steps.
    """
    a1, b1, c1, d1 = earlier
    a2, b2, c2, d2 = later
    a = a2 * a1 + b2 * c1
    b = a2 * b1 + b2 * d1
    c = c2 * a1 + d2 * c1
    d = c2 * b1 + d2 * d1
    norm = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.maximum(c.abs(), d.abs()))
    norm = norm.clamp(min=1e-30).detach()
    return a / norm, b / norm, c / norm, d / norm


def _linrec_combine(earlier: _Elems, later: _Elems) -> _Elems:
    """Compose two steps of the complex linear recurrence η ← g·η + κ:
    (g, κ)_later ∘ (g, κ)_earlier = (g_l·g_e, g_l·κ_e + κ_l), in explicit
    (re, im) pairs — no torch complex dtypes (autocast/fp32 discipline)."""
    g1r, g1i, k1r, k1i = earlier
    g2r, g2i, k2r, k2i = later
    gr = g2r * g1r - g2i * g1i
    gi = g2r * g1i + g2i * g1r
    kr = g2r * k1r - g2i * k1i + k2r
    ki = g2r * k1i + g2i * k1r + k2i
    # Span overflow guard. |g| = ā/den exceeds 1 while λ is still small
    # (den < ā); over a long such stretch the *span product* — and the κ
    # accumulator it multiplies, which compounds across scan levels — can
    # overflow fp32, though the sequential loop (which never materializes span
    # products) stays finite (NaN observed at ā≤0.5, k~1e-15, T=250). Cap both
    # composed magnitudes at 1e10: all downstream products/sums then stay
    # ≤~1e21 ≪ fp32 max, and the cap is exactly inactive (scale = 1) in every
    # non-diverged regime — trained-net η/κ are O(1–100). Where active, the
    # true value is astronomically diverged; we only keep it finite. Detached:
    # no gradient through the rescale.
    gmag = torch.maximum(gr.abs(), gi.abs())
    gscale = (1e10 / gmag.clamp(min=1e10)).detach()
    kmag = torch.maximum(kr.abs(), ki.abs())
    kscale = (1e10 / kmag.clamp(min=1e10)).detach()
    return gr * gscale, gi * gscale, kr * kscale, ki * kscale


@overload
def complex_kla_scan_parallel(
    abar_mag: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    pbar: Tensor,
    k: Tensor,
    v: Tensor,
    lam_v: Tensor,
    q: Tensor,
    eps: float = ...,
    return_state: Literal[False] = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def complex_kla_scan_parallel(
    abar_mag: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    pbar: Tensor,
    k: Tensor,
    v: Tensor,
    lam_v: Tensor,
    q: Tensor,
    eps: float = ...,
    *,
    return_state: Literal[True],
) -> tuple[Tensor, Tensor, dict[str, Tensor]]: ...


def complex_kla_scan_parallel(
    abar_mag: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    pbar: Tensor,
    k: Tensor,
    v: Tensor,
    lam_v: Tensor,
    q: Tensor,
    eps: float = 1e-6,
    return_state: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Parallel (associative-scan) version of :func:`complex_kla_scan`.

    Same signature, same math, same fp32/fp64 discipline — but log-depth in T
    instead of a Python loop, so it is not kernel-launch-bound on GPU (the
    sequential loop is T × ~15 tiny elementwise kernels; this is ~⌈log₂ T⌉
    levels of batched ops over the whole sequence). Two passes:

    **Pass 1 (λ):** λ_t = λ_{t−1}/(a² + p̄λ_{t−1}) + φ_t is a Möbius
    (linear-fractional) map of λ_{t−1}, i.e. the 2×2 matrix
    ``[[1 + φ_t p̄, φ_t a²], [p̄, a²]]`` acting projectively; composition is
    matrix product, scanned with :func:`_assoc_scan` (max-abs renormalization
    per combine — projectively free). λ_t is read off the cumulative matrix
    applied to λ₀ = 0, then re-derived by one exact recurrence step from
    λ_{t−1} so the readout matches the sequential rounding profile.

    **Pass 2 (η):** with λ_{t−1} known, den_t is explicit and the complex
    recurrence η_t = (ā rot_t/den_t)·η_{t−1} + κ_t is a first-order linear
    scan with the standard (g, κ) combine.

    ``return_state=True`` falls back to the sequential reference (diagnostics
    path — perf-irrelevant, keeps the stacked-internals contract exact).
    """
    if return_state:
        return complex_kla_scan(
            abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q, eps, return_state=True
        )
    B, T, N = k.shape
    D = v.shape[-1]

    a2 = (abar_mag * abar_mag)[None, None]  # (1, 1, N, D)
    p = pbar[None, None]  # (1, 1, N, D)
    k4 = k.unsqueeze(-1)  # (B, T, N, 1)
    lv4 = lam_v.unsqueeze(2)  # (B, T, 1, D)
    v4 = v.unsqueeze(2)
    phi = k4 * k4 * lv4  # (B, T, N, D)
    kappa_re = k4 * lv4 * v4  # κ_im = 0 (real value path)

    # Pass 1: Möbius scan for λ. Leaf matrices M_t = [[1+φp̄, φa²], [p̄, a²]].
    m_a = 1.0 + phi * p
    m_b = phi * a2
    m_c = p.expand(B, T, N, D)
    m_d = a2.expand(B, T, N, D)
    _, cum_b, _, cum_d = _assoc_scan(_mobius_combine, (m_a, m_b, m_c, m_d))
    # λ_t = (A·λ₀ + B)/(C·λ₀ + D) at λ₀ = 0 → B/D. D > 0 mathematically
    # (all entries ≥ 0, D ≥ Π a² > 0); the clamp only guards fp underflow.
    lam_scan = cum_b / cum_d.clamp(min=1e-30)
    lam_prev = F.pad(lam_scan[:, :-1], (0, 0, 0, 0, 1, 0))  # λ_{t−1}, λ₀ = 0
    den = a2 + p * lam_prev
    lam = lam_prev / den + phi  # one exact sequential step from λ_{t−1}

    # Pass 2: linear complex scan for η with per-step gain g_t = ā·rot_t/den_t.
    g_re = abar_mag * cos_w.unsqueeze(-1) / den  # (B, T, N, D)
    g_im = abar_mag * sin_w.unsqueeze(-1) / den
    _, _, eta_re, eta_im = _assoc_scan(
        _linrec_combine, (g_re, g_im, kappa_re, torch.zeros_like(kappa_re))
    )

    lam_safe = lam.clamp(min=eps)
    y_re = torch.einsum("btn,btnd->btd", q, eta_re / lam_safe)
    y_im = torch.einsum("btn,btnd->btd", q, eta_im / lam_safe)
    return y_re, y_im


def phase_diff_features(y_re: Tensor, y_im: Tensor) -> tuple[Tensor, Tensor]:
    """Angle-first-differential features of a complex readout sequence.

    Per token (dim=1) and per complex channel, the one-step phase
    differential of y is d_t = y_t · conj(y_{t−1}) with y_{−1} := y_0, so
    arg d_0 = 0. If the state phasor tracks a rotor, arg d_t is its angular
    velocity per frame — arg(d_t)·frame_rate/2π is the tracked instantaneous
    frequency — which the plain [Re, Im] readout discards (it passes the
    ω-oscillation through as feature noise instead).

    Parameters
    ----------
    y_re, y_im : (B, T, D)
        Real/imaginary parts of the readout sequence.

    Returns
    -------
    (arg_d, mag) : each (B, T, D)
        ``arg_d`` = atan2(Im d, Re d) ∈ (−π, π] (exactly 0 at t = 0) and
        ``mag`` = |y| computed as sqrt(|y|² + 1e−12) for gradient safety
        at y = 0. Where |d| is (near-)zero the angle is undefined and
        atan2's *backward* is NaN even if the forward value is masked out —
        so the inputs themselves are replaced by the constant (1, 0) there
        (no grad path through the dead branch), yielding arg 0.
    """
    prev_re = torch.cat([y_re[:, :1], y_re[:, :-1]], dim=1)
    prev_im = torch.cat([y_im[:, :1], y_im[:, :-1]], dim=1)
    d_re = y_re * prev_re + y_im * prev_im
    d_im = y_im * prev_re - y_re * prev_im
    alive = (d_re * d_re + d_im * d_im) > 1e-24
    d_re = torch.where(alive, d_re, torch.ones_like(d_re))
    d_im = torch.where(alive, d_im, torch.zeros_like(d_im))
    arg_d = torch.atan2(d_im, d_re)
    mag = torch.sqrt(y_re * y_re + y_im * y_im + 1e-12)
    return arg_d, mag


# mix-layer input width per readout mode, in units of d_model.
def phase_unit_features(y_re: Tensor, y_im: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Continuous variant of :func:`phase_diff_features`: the *unit* phase
    differential d̂ = d/|d| instead of arg d. Carries the same information
    (a pure phase-increment phasor, carrier-free) with no ±π wrap
    discontinuity and smooth gradients everywhere — atan2's wrap produced
    gradient spikes that NaN'd phase_only training on some seeds.

    Returns (Re d̂, Im d̂, |y|), each (B, T, D); d̂ = (1, 0) where |d| ≈ 0.
    """
    prev_re = torch.cat([y_re[:, :1], y_re[:, :-1]], dim=1)
    prev_im = torch.cat([y_im[:, :1], y_im[:, :-1]], dim=1)
    d_re = y_re * prev_re + y_im * prev_im
    d_im = y_im * prev_re - y_re * prev_im
    mag_d = torch.sqrt(d_re * d_re + d_im * d_im + 1e-24)
    alive = mag_d > 1e-12
    u_re = torch.where(alive, d_re / mag_d, torch.ones_like(d_re))
    u_im = torch.where(alive, d_im / mag_d, torch.zeros_like(d_im))
    mag = torch.sqrt(y_re * y_re + y_im * y_im + 1e-12)
    return u_re, u_im, mag


_READOUT_FEATURES: dict[str, int] = {
    "complex_mean": 2,  # [Re y, Im y]
    "phase_diff": 4,  # [Re y, Im y, arg d, |y|]
    "phase_only": 2,  # [|y|, arg d]
    "phase_unit": 3,  # [Re d̂, Im d̂, |y|] — continuous, no wrap discontinuity
}


class ComplexKLALayer(nn.Module):
    """Complex-KLA sequence mixer (design §2).

    Scaffolding copied from ``FenwickKLALayer`` (causal depthwise conv1d(k=4)
    + SiLU, QK L2-norm, λv softplus + 1e−4, gated residual, RMSNorm,
    out_proj), minus everything Fenwick (levels, fold, buckets). New pieces:
    per-slot rotation ω_t[n] = ω0[n] + s[n]·(W_ω h_t)[n] with W_ω
    zero-initialized (LTI start) and ω0 linearly spaced over [0, π]
    (LRU-style ring init); complex readout mixed back to d_model by a
    Linear(2·d_model, d_model).

    OU dynamics (γ, p, Δ) are learnable time-invariant (n_state, d_model)
    parameters stored in softplus-inverse form; γ log-spaced S4D-style,
    p init 0.01, Δ init log-uniform [0.001, 0.1] — exactly the KLA layer's
    storage and init.

    Set ``layer.capture = []`` to record the post-cast scan inputs each
    forward (opt-in analysis/test tap, mirrors the fkla layer's). Setting
    ``layer.capture_state = True`` alongside additionally records the scan
    internals (``lam``/``eta_re``/``eta_im`` (B, T, N, D), per-slot readout
    contributions ``contrib`` (B, T, N)) plus the pre-cos/sin rotation phase
    ``omega`` (B, T, N) in the same captured dict — the §6 activation-analysis
    tap. ``capture_state`` has no effect while ``capture`` is None.

    ``rotation=False`` removes the complex path entirely at TRAIN time (no
    ω0/s/W_ω parameters; the scan runs with cos ω = 1, sin ω = 0, i.e. the
    exact real-KLA flat recursion) — the design §5 ladder-item-1 control
    that decides whether the complex rotation is load-bearing (the P0b
    eval-time ablation measured a null delta on the trained P0 model).

    ``readout`` selects what the mix layer sees of the complex readout y:

    - ``"complex_mean"`` (default): [Re y, Im y] → Linear(2·d_model, d_model)
      — the original path, byte-identical.
    - ``"phase_diff"``: [Re y, Im y, arg d, |y|] with d_t = y_t·conj(y_{t−1})
      (see ``phase_diff_features``) → Linear(4·d_model, d_model). Exposes the
      state phasor's angular velocity — the tracked instantaneous frequency —
      alongside the raw quadratures.
    - ``"phase_only"``: [|y|, arg d] → Linear(2·d_model, d_model). Drops the
      raw quadratures entirely: the readout is rotation-invariant.
    """

    def __init__(
        self,
        d_model: int,
        n_state: int = 16,
        conv_kernel: int = 4,
        rotation: bool = True,
        p_init: float = 0.01,
        readout: str = "complex_mean",
    ):
        super().__init__()
        if readout not in _READOUT_FEATURES:
            raise ValueError(f"readout must be one of {sorted(_READOUT_FEATURES)}, got {readout!r}")
        self.d_model, self.n_state = d_model, n_state
        self.rotation = rotation
        self.readout = readout

        self.conv = nn.Conv1d(
            d_model, d_model, conv_kernel, groups=d_model, padding=conv_kernel - 1, bias=True
        )
        self.k_proj = nn.Linear(d_model, n_state, bias=False)
        self.q_proj = nn.Linear(d_model, n_state, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.lamv_proj = nn.Linear(d_model, d_model, bias=True)
        if rotation:
            # Rotation projection: zero-init weight AND bias so ω_t = ω0 at start.
            self.omega_proj = nn.Linear(d_model, n_state, bias=True)
            nn.init.zeros_(self.omega_proj.weight)
            nn.init.zeros_(self.omega_proj.bias)
            # Per-slot gate on the rotation excursion (init small) + ring init ω0.
            self.s = nn.Parameter(torch.full((n_state,), 0.1))
            self.omega0 = nn.Parameter(torch.linspace(0.0, math.pi, n_state))

        self.mix = nn.Linear(_READOUT_FEATURES[readout] * d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

        # OU params: γ log-spaced (S4D-style), p init = ``p_init``, Δ
        # log-uniform in [0.001, 0.1] (KLA Table 10); stored in
        # softplus-inverse form. The KLA default p_init=0.01 gives a
        # near-zero discretised process noise p̄ ~ 1e-6..1e-4, under which λ
        # accumulates all clip (gain φ/λ → 1e-6, the measured P1 pathology —
        # docs/experiments/ckla-activation-analysis.md §A2: the head
        # degenerates to clip-scale accumulators with no tracking
        # bandwidth). Larger p_init bounds λ at a steady state and keeps the
        # Kalman gain alive — the ckla_p1_pnoise arm.
        a0 = torch.logspace(math.log10(0.5), math.log10(8.0), n_state)
        a0 = a0.unsqueeze(-1).expand(n_state, d_model).contiguous()
        self.a_param = nn.Parameter(torch.log(torch.expm1(a0)))
        self.p_param = nn.Parameter(torch.log(torch.expm1(torch.full((n_state, d_model), p_init))))
        dt0 = torch.exp(
            torch.rand(n_state, d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.dt_param = nn.Parameter(torch.log(torch.expm1(dt0)))

        self.capture: list[dict[str, Tensor]] | None = None
        self.capture_state: bool = False
        # Parallel (associative-scan) path is OPT-IN (attr or env
        # CKLA_PARALLEL_SCAN=1). It materializes span-gain products the
        # sequential loop never forms; in transient regimes (λ still small,
        # |g|>1) their magnitude — and exponentially-amplified fp32 rounding —
        # produced forward NaNs on real 8 s training batches where the
        # sequential loop (and the Triton kernel, which replicates it
        # stepwise) stayed finite. Safe for short-T/CPU experimentation only
        # until a log-space-gain rewrite.
        self.use_parallel_scan: bool = os.environ.get("CKLA_PARALLEL_SCAN", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        # Fused Triton scan (models/ckla_triton.py) on CUDA by default — one
        # kernel per forward/backward instead of the parallel scan's ~log T
        # levels. Opt out per module (attribute) or globally via env
        # CKLA_NO_TRITON=1. CPU inputs always fall through to the torch paths.
        self.use_triton_scan: bool = os.environ.get("CKLA_NO_TRITON", "0").lower() not in (
            "1",
            "true",
            "yes",
        )

    def ou_discretise(self) -> tuple[Tensor, Tensor]:
        """(abar_mag, pbar), both (n_state, d_model): e^{−γΔ} and the OU
        process-noise integral (p²/2γ)(1 − e^{−2γΔ})."""
        a = F.softplus(self.a_param)
        p = F.softplus(self.p_param)
        dt = F.softplus(self.dt_param)
        abar_mag = torch.exp(-a * dt)
        pbar = (p**2 / (2 * a)) * (1 - torch.exp(-2 * a * dt))
        return abar_mag, pbar

    def forward(self, x: Tensor) -> Tensor:  # (B, T, D) → (B, T, D)
        _, T, _ = x.shape
        h = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        h = F.silu(h)

        k = F.normalize(self.k_proj(h), dim=-1)  # QK-norm
        q = F.normalize(self.q_proj(h), dim=-1)
        v = self.v_proj(h)
        lam_v = F.softplus(self.lamv_proj(h)) + 1e-4
        if self.rotation:
            omega = self.omega0 + self.s * self.omega_proj(h)  # (B, T, N)
        else:
            omega = torch.zeros(*h.shape[:2], self.n_state, dtype=h.dtype, device=h.device)

        abar_mag, pbar = self.ou_discretise()
        # fp32 discipline for the scan algebra: under bf16/fp16 autocast the
        # projections emit half precision, but half-precision information-form
        # recursions are numerically unacceptable (same cast points as
        # FenwickKLALayer.forward). The op body is elementwise-only, so
        # autocast cannot re-downcast it.
        if k.dtype not in (torch.float32, torch.float64):
            k, q, v, lam_v = k.float(), q.float(), v.float(), lam_v.float()
            omega = omega.float()
        cos_w, sin_w = torch.cos(omega), torch.sin(omega)

        if self.capture is not None:
            # opt-in analysis tap: the exact tensors the scan consumes,
            # post-cast (used by the autocast test and §6 diagnostics).
            self.capture.append(
                {
                    "abar_mag": abar_mag.detach().cpu(),
                    "pbar": pbar.detach().cpu(),
                    "k": k.detach().cpu(),
                    "q": q.detach().cpu(),
                    "v": v.detach().cpu(),
                    "lam_v": lam_v.detach().cpu(),
                    "cos_w": cos_w.detach().cpu(),
                    "sin_w": sin_w.detach().cpu(),
                }
            )

        if self.capture is not None and self.capture_state:
            y_re, y_im, state = complex_kla_scan(
                abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q, return_state=True
            )
            rec = self.capture[-1]
            rec["omega"] = omega.detach().cpu()
            for name, tens in state.items():
                rec[name] = tens.detach().cpu()
        else:
            triton_scan = _get_triton_scan() if (self.use_triton_scan and k.is_cuda) else None
            if triton_scan is not None:
                y_re, y_im = triton_scan(abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q)
            else:
                scan = complex_kla_scan_parallel if self.use_parallel_scan else complex_kla_scan
                y_re, y_im = scan(abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q)
        if self.readout == "complex_mean":
            feats = torch.cat([y_re, y_im], dim=-1)
        elif self.readout == "phase_diff":
            arg_d, mag = phase_diff_features(y_re, y_im)
            feats = torch.cat([y_re, y_im, arg_d, mag], dim=-1)
        elif self.readout == "phase_unit":
            u_re, u_im, mag = phase_unit_features(y_re, y_im)
            feats = torch.cat([u_re, u_im, mag], dim=-1)
        else:  # phase_only
            arg_d, mag = phase_diff_features(y_re, y_im)
            feats = torch.cat([mag, arg_d], dim=-1)
        y = self.mix(feats)
        y = self.norm(y) * F.silu(self.gate_proj(x))
        return self.out_proj(y)


class CKLABlock(nn.Module):
    """Pre-norm residual block: x + mixer(norm(x)), + MLP sub-block —
    identical shape to the fkla ``FenwickKLABlock``."""

    def __init__(self, d_model: int, mlp_ratio: int = 4, **mixer_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = ComplexKLALayer(d_model, **mixer_kwargs)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.SiLU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TemporalCKLAHead(nn.Module):
    """Temporal head over pooled trunk features: (B, in_ch, T) → (B, R, T).

    Linear in-projection → ``n_layers`` CKLABlocks over time → RMSNorm →
    linear read-out. No output activation or scaling — matches
    ``TemporalTransformerHead`` (raw linear per-frame RPS predictions), so
    training configs transfer unchanged. Depth is load-bearing (design §3):
    block 2's rotation is conditioned on block 1's output — the closed-loop
    capture→refine structure.
    """

    def __init__(
        self,
        in_ch: int = 128,
        d_model: int = 128,
        num_rotors: int = 4,
        n_layers: int = 2,
        n_state: int = 16,
        rotation: bool = True,
        p_init: float = 0.01,
        readout: str = "complex_mean",
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, d_model)
        self.blocks = nn.ModuleList(
            CKLABlock(d_model, n_state=n_state, rotation=rotation, p_init=p_init, readout=readout)
            for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, num_rotors)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, T) → (B, num_rotors, T)."""
        h = self.in_proj(x.transpose(1, 2))  # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        return self.proj(self.norm(h)).transpose(1, 2)


class SimpleConvV2CKLA(nn.Module):
    """``SimpleConvV2Transformer`` trunk with a ``TemporalCKLAHead`` (design §3).

    Trunk verbatim from ``SimpleConvV2Transformer``: front-end → 6×
    ``ResidualConvBlock2d`` encoder → ``FrequencyAttentionPool`` →
    (B, 128, T). Default front-end is ``stft_mag_if`` (G2b — the only
    front-end that beat baseline; ``frontend_key`` exists for the
    ``simple_conv_v2_ckla_mag`` registry variant / P1 ablation 5).

    Defaults d_model=128, n_layers=2, n_state=16 keep the head parameter
    budget ≈ the 2-layer transformer head it replaces (exact counts asserted
    in ``tests/models/test_ckla.py``).
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        frontend: nn.Module | None = None,
        frontend_key: str = "stft_mag_if",
        d_model: int = 128,
        n_layers: int = 2,
        n_state: int = 16,
        rotation: bool = True,
        p_init: float = 0.01,
        readout: str = "complex_mean",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend(frontend_key, n_fft=n_fft, hop_length=hop_length)
        self.frontend: nn.Module = frontend

        # First block adapts to the front-end's channel count (2 for the
        # default stft_mag_if).
        in_ch = getattr(frontend, "out_channels", 1)
        enc_spec = [
            (in_ch, 64, (7, 5), (2, 1), (3, 2)),
            (64, 128, (7, 5), (2, 1), (3, 2)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
            (128, 128, (5, 3), (2, 1), (2, 1)),
        ]
        self.encoder = nn.ModuleList()
        for ic, oc, kern, s, p in enc_spec:
            self.encoder.append(ResidualConvBlock2d(ic, oc, kern, s, p, use_se=True))

        self.freq_pool = FrequencyAttentionPool(128, num_heads=4)
        self.head = TemporalCKLAHead(
            in_ch=128,
            d_model=d_model,
            num_rotors=num_rotors,
            n_layers=n_layers,
            n_state=n_state,
            rotation=rotation,
            p_init=p_init,
            readout=readout,
        )

    def forward(self, audio: Tensor) -> Tensor:
        x = self.frontend(audio)  # (B, C, F, T)

        h = x
        for block in self.encoder:
            h = block(h)

        h = self.freq_pool(h)  # (B, 128, T)
        return self.head(h)  # (B, num_rotors, T)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with legacy checkpoint remap (SimpleConv family
        convention — harmless for CKLA checkpoints, which are all post-refactor)."""
        state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


class SimpleConvV2CKLACond(SimpleConvV2CKLA):
    """Conditional RPS **refiner**: ``forward(audio, cond) -> refined track``.

    The phase-only CKLA backbone extended with a corrupted-track conditioning
    input (registry key ``simple_conv_v2_ckla_phaseonly_cond``). Campaign
    rationale: classical refinement (VK refine / stage D) cannot pull a track
    with 0.5–2 rev/s error toward the truth on real audio although the comb
    information is present at high precision, so this model LEARNS the pull:
    train on ``(audio, corrupt(GT)) -> GT``
    (``data_processing.rps_corruption``), infer with a real coarse track
    (blind-VK Viterbi ~0.7 err / neural predictor 1–2.5 err) as ``cond``.

    Design choices (vs the plain CKLA model):

    - **Conditioning fusion**: the ``(B, R, F)`` conditioning track is
      normalised by ``cond_norm`` (rev/s scale, default 100 — keeps the MLP
      input O(1)), embedded per frame by a small 2-layer MLP to
      ``cond_embed_dim`` channels, linearly resampled onto the trunk's frame
      grid when the counts differ, and **concatenated along the feature dim**
      with the pooled trunk features before the temporal head (the head's
      ``in_ch`` grows to ``128 + cond_embed_dim``) — mirroring how existing
      fusion concatenates along channels before a trunk.
    - **Residual output**: the head predicts a bounded correction,
      ``out = cond + max_delta · tanh(head(...))`` (``max_delta`` ~3 rev/s).
      Residual over absolute for two reasons: (a) the refinement regime is
      *local* by construction — corruption ≤ ~2.5 rev/s offset + OU ≤ 1.5σ —
      and bounding the output to a tube around the conditioning makes
      "ignore the conditioning and re-predict from audio" impossible, which
      is exactly the failure mode that would collapse this back into the
      (weaker) unconditional predictor; (b) identity is the zero-weight
      solution, so early training cannot be worse than the coarse track.
    - **Rotor identity**: output row ``i`` corresponds to conditioning row
      ``i`` (the residual ties them structurally), so training uses a plain
      non-PIT MSE (``losses.RPSMSELoss``) against the GT aligned to the
      conditioning order (the corruption seam emits it that way).
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        frontend: nn.Module | None = None,
        frontend_key: str = "stft_mag_if",
        d_model: int = 128,
        n_layers: int = 2,
        n_state: int = 16,
        rotation: bool = True,
        p_init: float = 0.01,
        readout: str = "complex_mean",
        cond_embed_dim: int = 32,
        cond_norm: float = 100.0,
        max_delta: float = 3.0,
    ):
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            num_rotors=num_rotors,
            frontend=frontend,
            frontend_key=frontend_key,
            d_model=d_model,
            n_layers=n_layers,
            n_state=n_state,
            rotation=rotation,
            p_init=p_init,
            readout=readout,
        )
        self.cond_embed_dim = int(cond_embed_dim)
        self.cond_norm = float(cond_norm)
        self.max_delta = float(max_delta)
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_rotors, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        # Rebuild the head with the widened input (parent built it at 128).
        self.head = TemporalCKLAHead(
            in_ch=128 + self.cond_embed_dim,
            d_model=d_model,
            num_rotors=num_rotors,
            n_layers=n_layers,
            n_state=n_state,
            rotation=rotation,
            p_init=p_init,
            readout=readout,
        )

    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:  # type: ignore[override]
        """``audio (B, T_samples)`` + ``cond (B, R, F_cond)`` → ``(B, R, T)``.

        ``cond`` is linearly resampled to the trunk's frame count ``T`` when
        ``F_cond != T`` (endpoint-aligned — the dataset target and the STFT
        grid share phase 0, and off-by-one frame counts only arise from
        centering conventions).
        """
        x = self.frontend(audio)  # (B, C, F, T)
        h = x
        for block in self.encoder:
            h = block(h)
        h = self.freq_pool(h)  # (B, 128, T)
        t_frames = h.shape[-1]

        if cond.dim() != 3 or cond.shape[1] != self.num_rotors:
            raise ValueError(f"cond must be (B, {self.num_rotors}, F), got {tuple(cond.shape)}")
        c = cond.to(h.dtype)
        if c.shape[-1] != t_frames:
            c = F.interpolate(c, size=t_frames, mode="linear", align_corners=True)
        emb = self.cond_mlp((c / self.cond_norm).transpose(1, 2)).transpose(1, 2)  # (B, E, T)
        delta = self.head(torch.cat([h, emb], dim=1))  # (B, R, T)
        return c + self.max_delta * torch.tanh(delta)
