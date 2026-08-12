"""The linear design behind per-rotor attribution, and its identifiability.

Model, per frequency bin, for the array cross-spectral density (CSD) matrix of
the broadband residual::

    R(f) = sum_r  P_r(f) * g_r(f) g_r(f)^H   +   diag(D_1(f) .. D_M(f))

with ``P_r >= 0`` the per-rotor source PSD at 1 m and ``D_m >= 0`` the per-mic
incoherent term (flow noise on the diaphragm, electronics, and anything else
that does not obey the array's propagation law).

Real vectorisation
------------------
A Hermitian ``M x M`` matrix has ``M`` real diagonal + ``M(M-1)/2`` complex
off-diagonal degrees of freedom, i.e. ``M^2`` real numbers. For ``M = 8`` that
is ``8 + 56 = 64`` real equations against ``R + M = 12`` unknowns, so the system
is heavily over-determined. That is **not** the same as being identifiable.

The structural fact that governs everything here
------------------------------------------------
The ``M`` diagonal unknowns ``D_m`` span the whole diagonal subspace on their
own. So the ``M`` diagonal equations can be solved exactly for *any* value of
``P``, and, whenever the implied ``D`` stays non-negative, they carry **zero**
information about ``P``. Per-rotor power is identified by the **off-diagonal**
(cross-microphone) block alone.

This is the wind-channel lesson in matrix form: a component distinguished only
by its spatial law is invisible to per-microphone marginals. Consequently the
primary estimator in :mod:`.fit` regresses on the off-diagonal block only and
then *defines* ``D`` as the leftover diagonal; the joint fit is kept as a
cross-check and provably coincides with it away from the ``D >= 0`` boundary.

Identifiability diagnostics returned by :func:`identifiability`
---------------------------------------------------------------
All are computed on the off-diagonal design ``A_off`` (``2*M(M-1)/2`` rows,
``R`` columns), because that is the block that carries the information:

``cond``
    Condition number of ``A_off`` with **unit-norm columns** — the scale-free
    collinearity number. 1 = orthogonal rotor signatures, large = the rotors
    look alike and their powers trade off against one another.
``max_cos``
    Largest absolute cosine between two rotor columns. The pair achieving it is
    reported too: this is where an ambiguity, if any, lives.
``vif``
    Per-rotor variance-inflation factor ``1 / (1 - R^2_r)`` from regressing
    rotor ``r``'s column on the other three. This is the per-rotor number: a
    VIF of 1 means rotor ``r`` has a signature nothing else can imitate; VIF of
    100 means its power is 10x noisier than an orthogonal design would give.
``noise_gain``
    ``||e_r^T A_off^+||_2 * ||a_r||_2`` — the factor by which a relative error
    on the measured cross-spectra becomes a relative error on ``P_r``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IndexPlan",
    "index_plan",
    "hermitian_to_real",
    "rotor_columns",
    "diagonal_columns",
    "identifiability",
]


@dataclass(frozen=True)
class IndexPlan:
    """Row layout of the real vectorisation of a Hermitian ``M x M`` matrix."""

    n_mic: int
    iu: np.ndarray  # (P,) upper-triangle row indices, k=1
    ju: np.ndarray  # (P,) upper-triangle col indices, k=1

    @property
    def n_pairs(self) -> int:
        return int(self.iu.size)

    @property
    def n_diag_rows(self) -> int:
        return self.n_mic

    @property
    def n_off_rows(self) -> int:
        return 2 * self.n_pairs


def index_plan(n_mic: int) -> IndexPlan:
    iu, ju = np.triu_indices(n_mic, k=1)
    return IndexPlan(n_mic=n_mic, iu=iu, ju=ju)


def hermitian_to_real(R: np.ndarray, plan: IndexPlan) -> tuple[np.ndarray, np.ndarray]:
    """Split ``R`` ``(..., M, M)`` into ``(diag (..., M), off (..., 2P))``.

    The off-diagonal block is ``[Re(R_ij) for (i<j)] ++ [Im(R_ij) for (i<j)]``.
    """
    diag = np.real(np.einsum("...mm->...m", R))
    up = R[..., plan.iu, plan.ju]
    off = np.concatenate([np.real(up), np.imag(up)], axis=-1)
    return diag, off


def rotor_columns(g: np.ndarray, plan: IndexPlan) -> tuple[np.ndarray, np.ndarray]:
    """Design columns of the rotor terms.

    Args:
        g: ``(F, M, R)`` steering matrix.
        plan: row layout.

    Returns:
        ``(A_diag (F, M, R), A_off (F, 2P, R))`` — the diagonal and
        off-diagonal parts of ``vec(g_r g_r^H)``.
    """
    a_diag = np.abs(g) ** 2  # (F, M, R)
    up = g[:, plan.iu, :] * np.conj(g[:, plan.ju, :])  # (F, P, R)
    a_off = np.concatenate([np.real(up), np.imag(up)], axis=1)  # (F, 2P, R)
    return a_diag, a_off


def diagonal_columns(n_freq: int, plan: IndexPlan) -> tuple[np.ndarray, np.ndarray]:
    """Design columns of the per-mic diagonal terms: identity on the diagonal
    rows, exactly zero on every off-diagonal row (hence the structural fact in
    the module docstring)."""
    eye = np.broadcast_to(np.eye(plan.n_mic), (n_freq, plan.n_mic, plan.n_mic))
    zeros = np.zeros((n_freq, plan.n_off_rows, plan.n_mic))
    return eye, zeros


def _vif_and_gain(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column VIF and noise gain of a real design ``a`` ``(N, R)``."""
    n_col = a.shape[1]
    vif = np.full(n_col, np.nan)
    for r in range(n_col):
        y = a[:, r]
        others = np.delete(a, r, axis=1)
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ coef
        ss_tot = float(y @ y)
        r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 1.0
        vif[r] = 1.0 / max(1.0 - r2, 1e-15)
    pinv = np.linalg.pinv(a)  # (R, N)
    gain = np.linalg.norm(pinv, axis=1) * np.linalg.norm(a, axis=0)
    return vif, gain


def identifiability(g: np.ndarray, plan: IndexPlan) -> dict[str, np.ndarray]:
    """Per-frequency identifiability diagnostics of the off-diagonal design.

    Args:
        g: ``(F, M, R)`` steering matrix.
        plan: row layout.

    Returns:
        dict of ``(F,)`` or ``(F, R)`` arrays — see the module docstring.
        ``cond_with_diag`` additionally gives the condition number of the FULL
        ``(M^2, R+M)`` joint design, to show that the joint problem is not
        rank-deficient even though ``P`` is informed only by the off-diagonals.
    """
    _, a_off = rotor_columns(g, plan)  # (F, 2P, R)
    n_freq, _, n_rot = a_off.shape

    out: dict[str, np.ndarray] = {
        "cond": np.zeros(n_freq),
        "max_cos": np.zeros(n_freq),
        "max_cos_pair": np.zeros((n_freq, 2), dtype=int),
        "vif": np.zeros((n_freq, n_rot)),
        "noise_gain": np.zeros((n_freq, n_rot)),
        "cond_with_diag": np.zeros(n_freq),
    }

    a_diag, _ = rotor_columns(g, plan)
    eye, off_zero = diagonal_columns(n_freq, plan)

    iu, ju = np.triu_indices(n_rot, k=1)
    for i in range(n_freq):
        a = a_off[i]
        norm = np.linalg.norm(a, axis=0)
        norm = np.where(norm > 0, norm, 1.0)
        an = a / norm
        out["cond"][i] = np.linalg.cond(an)
        gram = np.abs(an.T @ an)
        cos = gram[iu, ju]
        j = int(np.argmax(cos))
        out["max_cos"][i] = cos[j]
        out["max_cos_pair"][i] = (iu[j], ju[j])
        vif, gain = _vif_and_gain(an)
        out["vif"][i] = vif
        out["noise_gain"][i] = gain
        full = np.block(
            [
                [a_diag[i], eye[i]],
                [a_off[i], off_zero[i]],
            ]
        )
        fnorm = np.linalg.norm(full, axis=0)
        fnorm = np.where(fnorm > 0, fnorm, 1.0)
        out["cond_with_diag"][i] = np.linalg.cond(full / fnorm)
    return out
