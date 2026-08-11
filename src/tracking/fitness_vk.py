"""F_VK — the profiled coupled Vold–Kalman residual, as an objective over TRAJECTORIES.

``docs/trajectory-fitness-design.md`` §1 Fact 3 and §2. The VK cost is quadratic
in the envelopes at a fixed trajectory, so substituting the closed-form envelope
solution back leaves a function of the trajectory alone — the *profiled residual*

.. math::

    F(\\phi) = \\min_a \\; \\|y - \\sum_m \\mathrm{Re}[a_m c_m(\\phi)]\\|_w^2
              + \\tfrac{s}{2} \\sum_m \\rho_m^2 \\|\\Delta^2 a_m\\|^2 ,
    \\qquad c_m(t) = \\exp(j 2\\pi k_m \\textstyle\\int r_{i(m)}) .

``w`` is the solver's own edge taper and ``s`` the decimation stride. Both are
part of the objective the solver actually minimizes, not decoration — see
:func:`_prior_term` and :func:`_data_weight`, and the numbers below.

Without the smoothness term this is the exact harmonic-subspace projector
``y^H P(phi) y`` — simultaneously the ML estimator and (after amplitude
marginalization) the Bayesian MAP statistic. With it, the off-diagonal coupling
blocks ``conj(c_m) c_n`` make near-equal rotors compete for the same energy,
which is the only published mechanism that breaks twin-rotor ambiguity. The VK
literature takes the speed as tachometer-given and never optimizes this over the
trajectory; that continuous step is what this module implements.

The envelope theorem
--------------------
:func:`fvk_loss` solves the envelopes with the EXISTING numpy/scipy solver
(:func:`tracking.vk_envelopes`) under ``no_grad``, then rebuilds the carriers in
torch as differentiable functions of the trajectory and evaluates the same
objective with ``a*`` **detached**. Because ``a*`` is a stationary point of the
objective in ``a``, the envelope theorem gives

.. math::  \\frac{dF}{d\\phi} = \\frac{\\partial L}{\\partial \\phi}(\\phi, a^*)
           + \\underbrace{\\frac{\\partial L}{\\partial a}(\\phi, a^*)}_{=\\,0}
             \\frac{da^*}{d\\phi} ,

so the gradient of the detached-``a*`` loss IS the gradient of the profiled
objective — no differentiation through the banded Cholesky is needed.

That equality only holds while ``dL/da`` really does vanish at ``a*``, which is
why the two details above are load-bearing. Measured on the 1 s single-rotor
window of ``tests/tracking/test_fitness_vk.py``, against central differences:

=========================================  ==================
objective the loss states                  gradient mismatch
=========================================  ==================
``rho^2`` prior, unweighted residual       39 % (65 % at 2 s)
``(s/2) rho^2`` prior, unweighted          7 %
``(s/2) rho^2`` prior, tapered residual    **3.2 %** (0.7 % at 2 s)
=========================================  ==================

What is left is the solver's remaining approximations — the normal equations
live on the decimated envelope grid, the cross terms are band-limited, and the
sum-frequency terms of ``Re[a c]`` are dropped. The test measures the mismatch
rather than assuming it away, and separately pins the carrier chain rule (the
part that must be exact) at 1.7e-5 by holding ``a*`` fixed.

Fixed degrees of freedom
------------------------
Same discipline as :mod:`tracking.fitness`: the harmonic set and the channel set
must not move between candidates, or the comparison measures the cell count and
not the trajectory. Here that is achieved by construction — the VK validity mask
is disabled (``f_min = 0``, ``f_max = inf``, ``min_rps = 0``) and the harmonic
cap is derived from a pinned REFERENCE trajectory instead
(:func:`k_cap`), so every candidate is scored on the identical
``(channel, rotor, harmonic)`` cell set. The score reports ``n_cells``.

What is in here
---------------
:func:`fvk_score`
    numpy-facing scorer: profiled residual, R^2, per-harmonic captured energy.
:func:`fvk_loss`
    the same objective as a differentiable torch scalar (envelope theorem).
:func:`alias_charge`
    the optional order/alias counter-term (Fact 2) — model harmonics whose line
    energy sits below their local floor, read through
    :func:`tracking.fitness.line_power`. Off by default.
:func:`optimize_trajectory`
    L-BFGS over a coarse cubic-spline parameterization of the trajectory under a
    ``k_max`` annealing schedule (Fact 5), with a convex log-domain smoothness
    prior (Fact 6).

Which knob is the basin
-----------------------
Fact 5 names ``K`` as the resolution axis because a coherent harmonic sum has
basin ``1/(K T)``. That is NOT this landscape: here every harmonic carries its
own VK envelope inside a ``k``-scaled band, so the capture radius is
``bw_rps / 2`` rev/s at every harmonic and raising ``k_max`` does not shrink it —
measured, the gradient at a 0.5 rev/s constant error still points at truth at
``k_max`` = 80. What ``k_max`` moves is the DEPTH and the curvature of the well
(objective at truth 0.587 -> 0.073 from ``k_max`` 5 to 80 on a 1 s window, its
±0.1 rev/s neighbours barely moving), which is the precision half of the same
law and why the schedule still starts coarse. **``bw_rps`` is the basin knob,
and it has units.** Opened to a non-capture 2.0 rev/s, ``k_max`` = 80 does break
into 7 local minima inside ±1 rev/s against 2 at ``k_max`` = 5.

The two Stages (:func:`tracking.fvk_stage`, :func:`tracking.fvk_refine_stage`)
live in :mod:`tracking.top` with every other stage.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from scipy.interpolate import BSpline

from tracking.dsp import torch_device
from tracking.fitness import line_power
from tracking.vk_tracking import (
    Envelopes,
    VKConfig,
    _tuma_rho,  # THE Tuma bandwidth<->selectivity relation; one implementation
    edge_taper,
    env_stride,
    vk_envelopes,
)

__all__ = [
    "DEFAULT_SCHEDULE",
    "FVKConfig",
    "FVKStage",
    "alias_charge",
    "fvk_loss",
    "fvk_score",
    "k_cap",
    "optimize_trajectory",
    "solve_envelopes",
]

_TINY = 1e-30

#: Median-to-mean factor of an exponential (periodogram) bin distribution — the
#: same unbiasing :mod:`tracking.fitness` applies to the ridge floor.
_LN2 = math.log(2.0)


@dataclass(frozen=True)
class FVKConfig:
    """Geometry of one F_VK measurement. Pinned per window, not per candidate."""

    sr: int = 16000
    fs_env: float = 100.0
    k_min: int = 1
    k_max: int = 40
    #: k-scaled per-harmonic −3 dB bandwidth: harmonic ``k`` is filtered with a
    #: ``k * bw_rps`` Hz band, so the CAPTURE RADIUS is ``bw_rps / 2`` rev/s at
    #: every harmonic (a rate error ``dr`` displaces harmonic ``k`` by ``k dr``
    #: Hz). This is the ``rho`` axis of the continuation schedule, in units.
    bw_rps: float = 1.0
    #: Harmonics whose line would sit above this (or above ``0.45 * sr``) are
    #: not modelled. Enforced through the harmonic CAP, never through a
    #: per-candidate mask — see the module docstring.
    f_max: float = 6000.0
    #: Safety factor on the reference rate when the cap is derived, so a
    #: candidate a little faster than the reference keeps the same cell set.
    cap_margin: float = 1.05
    max_channels: int = 8
    couple_hz: float | None = None
    solver: str = "banded"
    prune_far_pairs: bool = True
    #: Weight of the alias/order counter-term (Fact 2). ``0`` = off, which is
    #: the default: the term is a comparison charge with no gradient, so it
    #: belongs to scoring and to stage selection, not to the descent direction.
    alias_penalty: float = 0.0
    alias_dc_revs: float = 0.10
    alias_dc_bins: float = 2.0
    alias_annulus: tuple[float, float] = (3.0, 8.0)
    #: Torch device (``None`` = the tracking stack's own selection).
    device: str | None = None

    def __post_init__(self) -> None:
        if self.k_min < 1 or self.k_max < self.k_min:
            raise ValueError(f"need 1 <= k_min <= k_max, got {self.k_min}, {self.k_max}")
        if self.bw_rps <= 0:
            raise ValueError(f"bw_rps must be positive, got {self.bw_rps}")

    @property
    def stride(self) -> int:
        """Decimation stride of the envelope grid — the solver's own."""
        return env_stride(self.vk_config(self.k_min))[0]

    def vk_config(self, k_hi: int) -> VKConfig:
        """The :class:`VKConfig` one solve at harmonic cap ``k_hi`` runs under.

        ``f_min``/``f_max``/``min_rps`` are opened up on purpose: the validity
        mask is the one part of :func:`tracking.vk_envelopes` that would react
        to the candidate, and a cell set that moves with the candidate is not a
        comparison. The harmonic cap does that job instead.
        """
        return VKConfig(
            fs=float(self.sr),
            fs_env=self.fs_env,
            k_min=self.k_min,
            k_max=int(k_hi),
            f_min=0.0,
            f_max=float("inf"),
            min_rps=0.0,
            bw_rps=self.bw_rps,
            couple_hz=self.couple_hz,
            solver=self.solver,
            prune_far_pairs=self.prune_far_pairs,
            n_outer=1,
            k_schedule="fixed",
        )

    def echo(self) -> dict[str, Any]:
        """The config as plain scalars, for the score's ``config`` field."""
        return {
            "sr": int(self.sr),
            "fs_env": float(self.fs_env),
            "k_min": int(self.k_min),
            "k_max": int(self.k_max),
            "bw_rps": float(self.bw_rps),
            "f_max": float(self.f_max),
            "alias_penalty": float(self.alias_penalty),
        }


@dataclass(frozen=True)
class FVKStage:
    """One rung of the continuation schedule (design §2, Fact 5).

    ``k_max`` is the resolution axis — it sets how DEEP and how sharp the well
    is, not how wide (see "Which knob is the basin" in the module docstring).
    ``rho_scale`` multiplies the VK selectivity ``rho``: below 1 it widens every
    passband, which is the axis that does move the basin. ``max_iter`` is the
    L-BFGS budget of the rung.
    """

    k_max: int
    rho_scale: float = 1.0
    max_iter: int = 20


#: Geometric in ``k_max``: each rung roughly halves the width of the well its
#: predecessor left the argmin in, and every rung starts inside that argmin.
DEFAULT_SCHEDULE: tuple[FVKStage, ...] = (
    FVKStage(5),
    FVKStage(10),
    FVKStage(20),
    FVKStage(40),
    FVKStage(80),
)


# ---------------------------------------------------------------------------
# the fixed cell set


def k_cap(cfg: FVKConfig, reference: np.ndarray) -> int:
    """Highest harmonic that fits under ``f_max`` / Nyquist at the reference rate.

    The ONE place the harmonic set is decided. It reads the pinned reference
    trajectory (not the candidate), so every candidate of a window is scored on
    the same cells.
    """
    rate = float(np.max(np.abs(np.asarray(reference, dtype=np.float64))))
    top = min(float(cfg.f_max), 0.45 * float(cfg.sr))
    if rate <= 0:
        return int(cfg.k_max)
    return int(max(cfg.k_min, min(cfg.k_max, int(np.floor(top / (rate * cfg.cap_margin))))))


def _to_audio_grid(r: np.ndarray, frame_times: np.ndarray, n_t: int, sr: float) -> np.ndarray:
    """``(R, N)`` trajectories on ``frame_times`` -> ``(R, T)`` at audio rate."""
    r2 = np.atleast_2d(np.asarray(r, dtype=np.float64))
    ft = np.asarray(frame_times, dtype=np.float64)
    if r2.shape[-1] != len(ft):
        raise ValueError(f"rps has {r2.shape[-1]} frames but frame_times has {len(ft)}")
    t_audio = np.arange(n_t, dtype=np.float64) / float(sr)
    return np.stack([np.interp(t_audio, ft, row) for row in r2])


def solve_envelopes(
    audio: np.ndarray,
    r_audio: np.ndarray,
    cfg: FVKConfig,
    *,
    k_hi: int,
    rho_scale: float = 1.0,
) -> Envelopes:
    """One coupled VK envelope solve at harmonic cap ``k_hi`` — the thin wrapper.

    Everything numerical is :func:`tracking.vk_envelopes`; this only builds the
    config (validity mask off, cap instead) and turns ``rho_scale`` into the
    solver's own per-track ``rho2_gain``.
    """
    vk = cfg.vk_config(k_hi)
    n_tracks = int(np.atleast_2d(r_audio).shape[0]) * (int(k_hi) - cfg.k_min + 1)
    gain = None if rho_scale == 1.0 else np.full(n_tracks, float(rho_scale) ** 2)
    return vk_envelopes(audio, r_audio, vk, rho2_gain=gain)


def _rho2(env: Envelopes) -> np.ndarray:
    """``(M,)`` squared selectivity each track was actually solved with."""
    return np.array([_tuma_rho(float(b), env.fs_env, 2) for b in env.bw_track]) ** 2


# ---------------------------------------------------------------------------
# the objective


def _envelope_parts(env: Envelopes, device: Any) -> tuple[Any, Any, Any, Any]:
    """``(re, im, d_re, d_im)`` of the envelopes and their forward differences.

    The last difference column is ZERO, which is the "hold constant beyond the
    last knot" rule of :func:`tracking.vk_reconstruct` expressed without a
    special case.
    """
    re = torch.from_numpy(np.ascontiguousarray(env.x.real)).to(device, torch.float64)
    im = torch.from_numpy(np.ascontiguousarray(env.x.imag)).to(device, torch.float64)
    d_re, d_im = torch.zeros_like(re), torch.zeros_like(im)
    d_re[..., :-1] = re[..., 1:] - re[..., :-1]
    d_im[..., :-1] = im[..., 1:] - im[..., :-1]
    return re, im, d_re, d_im


def _upsample_index(n_t: int, n_env: int, stride: int, device: Any) -> tuple[Any, Any]:
    """Knot index and interpolation fraction of every audio sample."""
    idx = torch.arange(n_t, device=device)
    j = torch.clamp(idx // stride, max=max(n_env - 1, 0))
    frac = (idx % stride).to(torch.float64) / float(stride)
    return j, frac


def _model_signal(
    r_t: Any,
    env: Envelopes,
    cfg: FVKConfig,
    *,
    n_t: int,
    per_track: bool = False,
    weight: Any = None,
) -> tuple[Any, np.ndarray | None]:
    """``sum_m Re[a*_m c_m(phi)]`` as a ``(C, T)`` DIFFERENTIABLE tensor.

    ``r_t`` is the ``(R, T)`` audio-rate trajectory (the only tensor carrying a
    gradient); the envelopes enter detached, which is what makes the gradient
    the profiled objective's — see the module docstring. Tracks are accumulated
    one at a time: the whole ``(C, M, T)`` bank would be tens of gigabytes at a
    realistic ``k_max`` and buys nothing, since the sum is the only thing used.
    """
    device = r_t.device
    stride = cfg.stride
    n_env = int(env.x.shape[-1])
    re, im, d_re, d_im = _envelope_parts(env, device)
    j, frac = _upsample_index(n_t, n_env, stride, device)
    phase = 2.0 * math.pi * torch.cumsum(r_t, dim=-1) / float(cfg.sr)  # (R, T)

    model = torch.zeros((env.x.shape[0], n_t), dtype=torch.float64, device=device)
    energies = np.zeros(len(env.k)) if per_track else None
    for m in range(len(env.k)):
        ph = float(env.k[m]) * phase[int(env.rotor[m])]
        a_re = re[:, m][:, j] + d_re[:, m][:, j] * frac
        a_im = im[:, m][:, j] + d_im[:, m][:, j] * frac
        comp = a_re * torch.cos(ph) - a_im * torch.sin(ph)
        model = model + comp
        if energies is not None:
            sq = comp.detach() ** 2
            energies[m] = float(sq.sum() if weight is None else (weight * sq).sum())
    return model, energies


def _prior_term(env: Envelopes, cfg: FVKConfig) -> float:
    """``sum_m (stride/2) rho_m^2 ||D2 a_m||^2`` — the VK smoothness term.

    The ``stride / 2`` is what puts the prior on the AUDIO-rate data term's
    scale, and it is not cosmetic: get it wrong and ``a*`` is the argmin of a
    different objective, which is exactly the ``dL/da != 0`` that breaks the
    envelope theorem. The VK normal equations live on the decimated grid, where
    the data term reads ``x^H (I + C) x - 4 Re<x, w z>``. At audio rate the same
    quadratic form is ``sum_t (sum_m Re[a_m c_m])^2``, whose coefficient is
    ``stride / 2`` times the decimated one — ``stride`` audio samples per
    envelope sample, and ``Re[a c]^2`` averages to ``|a|^2 / 2`` over the
    carrier. So the prior that pairs with an audio-rate residual is
    ``(stride / 2) rho^2``, not ``rho^2``.

    Constant in the trajectory (``a*`` is detached), so it contributes no
    gradient; it is carried because F_VK is the profiled objective's VALUE and
    candidates are compared by it.
    """
    d2 = env.x[..., 2:] - 2.0 * env.x[..., 1:-1] + env.x[..., :-2]
    if d2.size == 0:
        return 0.0
    scale = 0.5 * cfg.stride
    return float(scale * (_rho2(env)[None, :] * (np.abs(d2) ** 2).sum(axis=-1)).sum())


def _data_weight(env: Envelopes, cfg: FVKConfig, n_t: int, device: Any) -> Any:
    """``(T,)`` audio-rate copy of the VK data weight ``w(t)``.

    The solver fades the data term at both window ends
    (:func:`tracking.edge_taper`), so the objective it minimises is a WEIGHTED
    residual. Profiling anything else leaves ``dL/da != 0`` on the tapered span
    and the envelope-theorem gradient picks up an error there — measured at
    13 % on a noiseless single-rotor window, 1 % once the taper is carried.
    The tracks all share one weight because the validity mask is disabled by
    :meth:`FVKConfig.vk_config`.
    """
    w = np.repeat(edge_taper(int(env.x.shape[-1])), cfg.stride)[:n_t]
    return torch.from_numpy(np.ascontiguousarray(w)).to(device, torch.float64)


def _prepare(audio: np.ndarray, cfg: FVKConfig, device: Any) -> tuple[np.ndarray, Any]:
    """``(C, T)`` float64 audio, capped at ``max_channels``, plus its tensor."""
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))[: cfg.max_channels]
    return y, torch.from_numpy(np.ascontiguousarray(y)).to(device, torch.float64)


def _terms(
    y_t: Any,
    r_t: Any,
    env: Envelopes,
    cfg: FVKConfig,
    *,
    per_track: bool = False,
) -> tuple[Any, float, np.ndarray | None]:
    """``(weighted data term, weighted signal energy, per-track energies)``.

    THE residual of this module — :func:`fvk_loss` and :func:`fvk_score` both
    read through it, so the differentiable objective and the reported score are
    the same number by construction.
    """
    n_t = int(y_t.shape[-1])
    w = _data_weight(env, cfg, n_t, y_t.device)
    model, energies = _model_signal(
        r_t, env, cfg, n_t=n_t, per_track=per_track, weight=w if per_track else None
    )
    data = (w * (y_t[: model.shape[0]] - model) ** 2).sum()
    return data, float((w * y_t**2).sum()), energies


def fvk_loss(
    audio: np.ndarray | Any,
    r_t: Any,
    cfg: FVKConfig,
    *,
    env: Envelopes | None = None,
    k_hi: int | None = None,
    rho_scale: float = 1.0,
) -> Any:
    """The profiled objective as a differentiable torch scalar.

    ``r_t``: ``(R, T)`` float64 tensor, the trajectory at AUDIO rate, carrying
    the gradient. ``env``: the envelopes to profile at — solved here from
    ``r_t.detach()`` when omitted. The returned scalar is

        ``sum_c ||y_c - sum_m Re[a*_m c_m(phi)]||_w^2
          + (stride/2) sum_m rho_m^2 ||D2 a*_m||^2``

    (plus ``alias_penalty * ||y||_w^2 * charge`` when the counter-term is on),
    where ``||.||_w`` carries the solver's own edge taper and ``a*`` is
    DETACHED. ``loss.backward()`` then gives the gradient of the profiled
    objective by the envelope theorem — see the module docstring for why the
    weight and the ``stride/2`` are load-bearing rather than cosmetic.
    """
    device = r_t.device
    y, y_t = _prepare(audio, cfg, device) if not torch.is_tensor(audio) else (None, audio)
    if y_t.dim() == 1:
        y_t = y_t[None, :]
    if env is None:
        with torch.no_grad():
            src = y if y is not None else y_t.detach().cpu().numpy()
            env = solve_envelopes(
                src,
                r_t.detach().cpu().numpy(),
                cfg,
                k_hi=int(k_hi if k_hi is not None else cfg.k_max),
                rho_scale=rho_scale,
            )
    data, e_w, _ = _terms(y_t, r_t, env, cfg)
    loss = data + _prior_term(env, cfg)
    if cfg.alias_penalty > 0:
        charge, _ = alias_charge(env, cfg)
        loss = loss + cfg.alias_penalty * e_w * charge
    return loss


def fvk_score(
    audio: np.ndarray,
    sr: float,
    r: np.ndarray,
    frame_times: np.ndarray,
    cfg: FVKConfig | None = None,
    *,
    reference: np.ndarray | None = None,
    k_hi: int | None = None,
    rho_scale: float = 1.0,
) -> dict[str, Any]:
    """Score one candidate trajectory against one window's audio.

    ``r``: ``(R, N)`` rev/s on ``frame_times`` (seconds, relative to the start
    of ``audio``). ``reference`` pins the harmonic cap — hand over the window's
    reference trajectory so every candidate keeps the same cells; it defaults to
    the candidate itself, which is only correct when there is one candidate.

    Returns a plain dict:

    ``residual``
        the profiled objective ``F(phi)`` — data term plus VK smoothness term.
    ``data_term`` / ``prior_term``
        its two halves.
    ``r2``
        fraction of the window's energy captured by the harmonic subspace,
        ``1 - ||y - model||_w^2 / ||y||_w^2``. Both norms carry the solver's
        edge taper, so ``r2`` reads on the span the objective is defined on.
    ``objective``
        ``residual / ||y||_w^2`` plus the alias charge — the scale-free number
        to compare candidates by (lower is better).
    ``k_energy``
        ``(K,)`` captured energy per harmonic index, summed over rotors and
        channels: the profile the continuation schedule is read against.
    ``n_cells`` / ``n_channels`` / ``n_tracks`` / ``k_hi``
        the FIXED degrees of freedom — identical across candidates by
        construction, and reported so a caller can check.
    """
    use = replace(cfg or FVKConfig(), sr=int(round(sr)))
    device = torch_device(use.device)
    y, y_t = _prepare(audio, use, device)
    n_t = int(y.shape[-1])
    r_audio = _to_audio_grid(r, frame_times, n_t, use.sr)
    ref = r_audio if reference is None else _to_audio_grid(reference, frame_times, n_t, use.sr)
    cap = int(k_hi) if k_hi is not None else k_cap(use, ref)

    env = solve_envelopes(y, r_audio, use, k_hi=cap, rho_scale=rho_scale)
    with torch.no_grad():
        r_t = torch.from_numpy(np.ascontiguousarray(r_audio)).to(device, torch.float64)
        data_t, e_y, energies = _terms(y_t, r_t, env, use, per_track=True)
        data = float(data_t)
    prior = _prior_term(env, use)

    ks = np.arange(use.k_min, cap + 1)
    assert energies is not None
    k_energy = np.zeros(len(ks))
    for m, kk in enumerate(env.k):
        k_energy[int(kk) - use.k_min] += energies[m]

    charge = None
    if use.alias_penalty > 0:
        charge = alias_charge(env, use)[0]
    out: dict[str, Any] = {
        "residual": data + prior,
        "data_term": data,
        "prior_term": prior,
        "r2": 1.0 - data / max(e_y, _TINY),
        "objective": (data + prior) / max(e_y, _TINY)
        + (0.0 if charge is None else use.alias_penalty * charge),
        "alias_charge": charge,
        "k_index": [int(v) for v in ks],
        "k_energy": [float(v) for v in k_energy],
        "energy": e_y,
        "k_hi": cap,
        "n_channels": int(y.shape[0]),
        "n_tracks": int(len(env.k)),
        "n_cells": int(y.shape[0] * len(env.k)),
        "config": use.echo(),
    }
    return out


# ---------------------------------------------------------------------------
# the alias / order counter-term (Fact 2)


def alias_charge(env: Envelopes, cfg: FVKConfig) -> tuple[float, np.ndarray]:
    """Charge every model harmonic whose line sits BELOW its local floor.

    A sub-multiple comb contains the true comb as a subset, so under any energy
    sum its residual is as good as truth (design §1 Fact 2) — the only way to
    break the degeneracy is to charge the *empty* slots. Each track's
    demodulated envelope is read at DC against the annulus around it by
    :func:`tracking.fitness.line_power` (the project's ONE "how much line is
    here" estimator; there is deliberately no peak search), and the charge is
    ``clip(1 - line / floor_expectation, 0, 1)``: 1 for a slot with no line at
    all, 0 for a slot the comb explains.

    Returns ``(mean charge, (C, M) per-cell charges)``.
    """
    n_env = int(env.z.shape[-1])
    if n_env < 4:
        return 0.0, np.zeros(env.z.shape[:2])
    freqs = np.fft.fftshift(np.fft.fftfreq(n_env, d=1.0 / env.fs_env))
    spec = np.fft.fftshift(np.fft.fft(env.z, axis=-1), axes=-1)
    power = (np.abs(spec) ** 2) / n_env
    res_hz = env.fs_env / n_env
    charge = np.zeros(env.z.shape[:2])
    for m in range(len(env.k)):
        half = max(cfg.alias_dc_revs * float(env.k[m]), cfg.alias_dc_bins * res_hz)
        lp = line_power(power[:, m], freqs, 0.0, half, annulus=cfg.alias_annulus, floor_scale=_LN2)
        denom = np.maximum(np.asarray(lp.floor) * max(lp.n_bins, 1), _TINY)
        charge[:, m] = np.clip(1.0 - np.asarray(lp.raw) / denom, 0.0, 1.0)
    return float(charge.mean()), charge


# ---------------------------------------------------------------------------
# the optimizer


def _spline_basis(n_t: int, sr: float, knot_s: float, degree: int = 3) -> np.ndarray:
    """``(T, B)`` cubic B-spline design matrix on a uniform knot grid.

    A coarse basis is the point: it caps the trajectory's own degrees of freedom
    at ``duration / knot_s + degree`` per rotor, which is what keeps the search
    off the near-flat plateau of near-optimal wiggly trajectories (Fact 6).
    """
    t = np.arange(n_t, dtype=np.float64) / float(sr)
    t0, t1 = float(t[0]), float(t[-1])
    span = max(t1 - t0, 1e-9)
    n_int = max(1, int(round(span / max(knot_s, 1e-6))))
    inner = np.linspace(t0, t1, n_int + 1)
    knots = np.concatenate([np.full(degree, t0), inner, np.full(degree, t1)])
    # design_matrix is closed on the left only; nudge the last sample inside.
    tq = np.clip(t, t0, t1 - span * 1e-12)
    return np.asarray(BSpline.design_matrix(tq, knots, degree, extrapolate=False).todense())


def _smoothness(r_t: Any, stride: int, floor: float = 1e-3) -> Any:
    """``sum (Delta log r)^2`` on the decimated grid — convex, log-domain.

    Kaldi's transition cost, not RAPT's: an octave branch would install a second
    basin exactly where Fact 2 says the landscape is already degenerate.
    """
    lr = torch.log(torch.clamp(r_t[:, ::stride], min=floor))
    return ((lr[:, 1:] - lr[:, :-1]) ** 2).sum()


def _run_rung(
    st: FVKStage,
    *,
    params: list[Any],
    trajectory: Any,
    audio: np.ndarray,
    audio_t: Any,
    cfg: FVKConfig,
    e_y: float,
    stride: int,
    smooth_lambda: float,
    lr: float,
) -> dict[str, Any]:
    """One rung of the continuation schedule: L-BFGS at a fixed ``(k, rho)``.

    The envelopes are re-solved inside the closure, so every line-search probe
    profiles its own trajectory — descending a stale ``a*`` is exactly the
    approximation the envelope theorem makes unnecessary.
    """
    with torch.no_grad():
        r_before = trajectory().detach().cpu().numpy()
    opt = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=int(st.max_iter),
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    trace: list[float] = []
    tic = time.perf_counter()

    def closure() -> Any:
        opt.zero_grad()
        r_t = trajectory()
        with torch.no_grad():
            env = solve_envelopes(
                audio,
                r_t.detach().cpu().numpy(),
                cfg,
                k_hi=st.k_max,
                rho_scale=st.rho_scale,
            )
        loss = fvk_loss(audio_t, r_t, cfg, env=env) / e_y
        loss = loss + smooth_lambda * _smoothness(r_t, stride)
        loss.backward()
        trace.append(float(loss.detach()))
        return loss

    opt.step(closure)
    with torch.no_grad():
        move = np.abs(trajectory().detach().cpu().numpy() - r_before)
    return {
        "k_max": int(st.k_max),
        "rho_scale": float(st.rho_scale),
        "n_evals": len(trace),
        "loss_start": trace[0] if trace else None,
        "loss_end": trace[-1] if trace else None,
        "move_max": float(move.max()) if move.size else 0.0,
        "move_rms": float(np.sqrt((move**2).mean())) if move.size else 0.0,
        "wall_s": round(time.perf_counter() - tic, 3),
        "loss_trace": [round(v, 9) for v in trace],
    }


def optimize_trajectory(
    audio: np.ndarray,
    sr: float,
    r_init: np.ndarray,
    frame_times: np.ndarray,
    cfg: FVKConfig | None = None,
    *,
    schedule: tuple[FVKStage, ...] | None = None,
    knot_s: float = 0.25,
    smooth_lambda: float = 1.0,
    reference: np.ndarray | None = None,
    lr: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine whole trajectories by L-BFGS on F_VK under a k-annealing schedule.

    Each rotor's trajectory is ``r_init + offset + B w`` with ``B`` a cubic
    B-spline basis on ``knot_s``-spaced knots and ``offset`` a per-rotor
    constant (the offset is representable by the basis too — it is kept because
    a constant rate error is the canonical corruption and its own parameter
    makes the diagnostics readable). The envelopes are RE-SOLVED inside every
    L-BFGS closure, so what is descended is the profiled objective and not a
    stale linearization.

    What is descended is ``F_VK / ||y||^2 + smooth_lambda * sum (Delta log r)^2``
    on the envelope grid. The normalization is what makes ``smooth_lambda``
    portable: the data term is then order 1 whatever the recording's level. The
    prior is convex and log-domain (Kaldi's transition cost, not RAPT's octave
    branch) and is deliberately weak by default — the coarse spline basis already
    caps the trajectory's degrees of freedom, and Fact 6 warns that a whole
    trajectory otherwise wanders along a near-flat plateau.

    Returns ``(r_refined on frame_times, diagnostics)``. The diagnostics carry
    one record per schedule rung — loss trace and argmin movement — which is the
    continuation-validity check of design §3: the argmin path must be
    continuous and the coarse rung must hold a single basin.
    """
    use = replace(cfg or FVKConfig(), sr=int(round(sr)))
    device = torch_device(use.device)
    y, y_t = _prepare(audio, use, device)
    n_t = int(y.shape[-1])
    r0 = _to_audio_grid(r_init, frame_times, n_t, use.sr)
    ref = r0 if reference is None else _to_audio_grid(reference, frame_times, n_t, use.sr)
    cap = k_cap(use, ref)
    stride = use.stride

    rungs = [
        replace(st, k_max=min(st.k_max, cap))
        for st in (schedule if schedule is not None else DEFAULT_SCHEDULE)
    ]
    seen: set[tuple[int, float]] = set()
    plan: list[FVKStage] = []
    for st in rungs:  # a schedule capped into duplicates is one rung, not five
        key = (st.k_max, st.rho_scale)
        if key not in seen:
            seen.add(key)
            plan.append(st)
    if not plan:
        plan = [FVKStage(cap)]

    r0_t = torch.from_numpy(np.ascontiguousarray(r0)).to(device, torch.float64)
    basis = torch.from_numpy(_spline_basis(n_t, use.sr, knot_s)).to(device, torch.float64)
    n_rotors, n_basis = r0.shape[0], int(basis.shape[1])
    w = torch.zeros((n_rotors, n_basis), dtype=torch.float64, device=device, requires_grad=True)
    off = torch.zeros((n_rotors, 1), dtype=torch.float64, device=device, requires_grad=True)
    e_y = max(float((y**2).sum()), _TINY)

    def trajectory() -> Any:
        return r0_t + off + w @ basis.T

    diag_stages = [
        _run_rung(
            st,
            params=[w, off],
            trajectory=trajectory,
            audio=y,
            audio_t=y_t,
            cfg=use,
            e_y=e_y,
            stride=stride,
            smooth_lambda=smooth_lambda,
            lr=lr,
        )
        for st in plan
    ]

    with torch.no_grad():
        r_final = trajectory().detach().cpu().numpy()
    ft = np.asarray(frame_times, dtype=np.float64)
    t_audio = np.arange(n_t, dtype=np.float64) / float(use.sr)
    r_out = np.stack([np.interp(ft, t_audio, row) for row in r_final])
    diag: dict[str, Any] = {
        "k_cap": cap,
        "n_basis": n_basis,
        "knot_s": float(knot_s),
        "smooth_lambda": float(smooth_lambda),
        "stages": diag_stages,
        "move_total_max": float(np.abs(r_final - r0).max()),
        "config": use.echo(),
    }
    return r_out, diag
