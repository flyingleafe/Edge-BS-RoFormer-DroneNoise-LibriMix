"""Non-negative fit of ``R(f) = sum_r P_r g_r g_r^H + diag(D)``.

Two estimators, deliberately both kept:

``fit_offdiag`` (**primary**)
    NNLS for ``P`` on the off-diagonal (cross-microphone) rows only, then
    ``D_m := max(R_mm - sum_r P_r |g_mr|^2, 0)``. This is the honest estimator:
    per :mod:`.design`, the diagonal carries no information about ``P``, so
    letting it enter the regression can only smuggle a per-mic level difference
    into a rotor share.

``fit_joint``
    NNLS for ``[P; D]`` on all ``M^2`` rows at once. Provably equal to the
    primary estimator whenever the implied ``D`` is non-negative (the diagonal
    block is then solved exactly for any ``P``), so a disagreement flags bins
    where a ``D_m`` was clipped at zero — i.e. bins where the model over-
    predicts a microphone's own level.

Weighting
---------
Each equation is divided by the scale of its own entry:
``sqrt(R_ii R_jj)`` for the ``(i,j)`` cross terms and ``R_ii`` for the diagonal.
Under a complex-Wishart CSD estimate those are, up to a constant, the entries'
standard deviations, so this is GLS-in-spirit and makes the fit scale-free
across frequency (loud low bands do not dominate quiet high ones).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from .design import IndexPlan, hermitian_to_real, rotor_columns

__all__ = ["Attribution", "fit_offdiag", "fit_joint", "bootstrap_shares", "model_matrix"]


@dataclass
class Attribution:
    """Per-bin fit result."""

    freqs: np.ndarray  # (F,)
    p_rotor: np.ndarray  # (F, R)  source PSD at 1 m
    d_mic: np.ndarray  # (F, M)  per-mic incoherent PSD
    recv_rotor: np.ndarray  # (F, R)  power received at the array from each rotor
    recv_diag: np.ndarray  # (F,)    power in the incoherent term, summed over mics
    off_explained: np.ndarray  # (F,) fraction of weighted off-diagonal energy explained
    d_clipped: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def shares(self) -> np.ndarray:
        """``(F, R+1)`` share of received residual power: rotors then diagonal."""
        tot = self.recv_rotor.sum(axis=1) + self.recv_diag
        tot = np.where(tot > 0, tot, np.nan)
        return np.concatenate([self.recv_rotor, self.recv_diag[:, None]], axis=1) / tot[:, None]


def _weights(diag: np.ndarray, plan: IndexPlan) -> tuple[np.ndarray, np.ndarray]:
    """Per-row inverse scales ``(w_diag (F, M), w_off (F, 2P))``."""
    d = np.maximum(diag, 1e-300)
    w_diag = 1.0 / d
    pair = 1.0 / np.sqrt(d[:, plan.iu] * d[:, plan.ju])
    return w_diag, np.concatenate([pair, pair], axis=1)


def _received(p: np.ndarray, a_diag: np.ndarray) -> np.ndarray:
    """``(F, R)`` power each rotor contributes summed over microphones."""
    return p * a_diag.sum(axis=1)


def fit_offdiag(R: np.ndarray, g: np.ndarray, plan: IndexPlan) -> Attribution:
    """Primary estimator. ``R`` ``(F, M, M)``, ``g`` ``(F, M, R)``."""
    diag, off = hermitian_to_real(R, plan)
    a_diag, a_off = rotor_columns(g, plan)
    w_diag, w_off = _weights(diag, plan)

    n_freq, n_rot = a_off.shape[0], a_off.shape[2]
    p = np.zeros((n_freq, n_rot))
    expl = np.zeros(n_freq)
    for i in range(n_freq):
        w = w_off[i][:, None]
        a = a_off[i] * w
        b = off[i] * w_off[i]
        sol, _ = nnls(a, b)
        p[i] = sol
        denom = float(b @ b)
        r = b - a @ sol
        expl[i] = 1.0 - float(r @ r) / denom if denom > 0 else np.nan

    model_diag = np.einsum("fmr,fr->fm", a_diag, p)
    d_raw = diag - model_diag
    d = np.maximum(d_raw, 0.0)
    return Attribution(
        freqs=np.arange(n_freq, dtype=np.float64),
        p_rotor=p,
        d_mic=d,
        recv_rotor=_received(p, a_diag),
        recv_diag=d.sum(axis=1),
        off_explained=expl,
        d_clipped=(d_raw < 0).any(axis=1),
    )


def fit_joint(R: np.ndarray, g: np.ndarray, plan: IndexPlan) -> Attribution:
    """Joint NNLS over ``[P; D]`` on all ``M^2`` rows (cross-check estimator)."""
    diag, off = hermitian_to_real(R, plan)
    a_diag, a_off = rotor_columns(g, plan)
    w_diag, w_off = _weights(diag, plan)
    n_freq, n_mic, n_rot = a_diag.shape[0], plan.n_mic, a_diag.shape[2]

    eye = np.eye(n_mic)
    zeros = np.zeros((plan.n_off_rows, n_mic))
    p = np.zeros((n_freq, n_rot))
    d = np.zeros((n_freq, n_mic))
    expl = np.zeros(n_freq)
    for i in range(n_freq):
        top = np.hstack([a_diag[i], eye]) * w_diag[i][:, None]
        bot = np.hstack([a_off[i], zeros]) * w_off[i][:, None]
        a = np.vstack([top, bot])
        b = np.concatenate([diag[i] * w_diag[i], off[i] * w_off[i]])
        sol, _ = nnls(a, b)
        p[i] = sol[:n_rot]
        d[i] = sol[n_rot:]
        bo = off[i] * w_off[i]
        r = bo - bot[:, :n_rot] @ p[i]
        denom = float(bo @ bo)
        expl[i] = 1.0 - float(r @ r) / denom if denom > 0 else np.nan

    return Attribution(
        freqs=np.arange(n_freq, dtype=np.float64),
        p_rotor=p,
        d_mic=d,
        recv_rotor=_received(p, a_diag),
        recv_diag=d.sum(axis=1),
        off_explained=expl,
        d_clipped=np.zeros(n_freq, dtype=bool),
    )


def bootstrap_shares(
    csd_obj,
    g: np.ndarray,
    plan: IndexPlan,
    band_masks: Sequence[np.ndarray],
    *,
    n_boot: int = 32,
    seed: int = 0,
) -> list[np.ndarray]:
    """Segment bootstrap of the band-integrated shares.

    Resamples Welch segments with replacement, refits **once per draw**, and
    integrates the result over every band, returning one ``(n_boot, R+1)``
    array per entry of ``band_masks``. Segment resampling is the right unit: it
    is the independent replication in a Welch estimate.
    """
    rng = np.random.default_rng(seed)
    n_seg = csd_obj.n_seg
    acc: list[list[np.ndarray]] = [[] for _ in band_masks]
    for _ in range(n_boot):
        idx = rng.integers(0, n_seg, size=n_seg)
        att = fit_offdiag(csd_obj.matrix(idx), g, plan)
        recv_all = np.concatenate([att.recv_rotor, att.recv_diag[:, None]], axis=1)
        for bi, m in enumerate(band_masks):
            recv = recv_all[m].sum(axis=0)
            acc[bi].append(recv / max(recv.sum(), 1e-300))
    return [np.asarray(a) for a in acc]


def model_matrix(p_rotor: np.ndarray, d_mic: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Rebuild ``(F, M, M)`` ``sum_r P_r g_r g_r^H + diag(D)`` from a fit."""
    out = np.einsum("fr,fmr,fnr->fmn", p_rotor.astype(np.complex128), g, np.conj(g))
    idx = np.arange(g.shape[1])
    out[:, idx, idx] += d_mic
    return out
