"""Numerical solvers for the inverse-harmonic-clustering multi-pitch estimator.

Reference: A. Björkman and F. Elvander, "Inverse Harmonic Clustering for
Multi-Pitch Estimation: An Optimal Transport Approach", IEEE TSP 2026
(arXiv:2508.02471), Sec. VI-VII. Equation numbers below are that paper's.

The stochastic estimator, eq (27), is

    min_{v >= 0}  (1/T)||r - A v||^2 + beta * 1^T v + zeta * S_c(v)

    S_c(v) = min_{M >= 0, M 1_G = v}  <C, M> + eps*D(M) + eta*||M||_{inf,1}

with ``D`` the entropic term and ``||M||_{inf,1} = sum_g max_f M[f,g]`` the
group-sparsity proxy over pitch candidates. It is solved by the Bregman (KL)
proximal gradient scheme of Prop. 1, eqs (32)-(34):

    v^{j+1} = v^{j} * exp(-gamma*u - gamma*zeta*lam),   gamma = 1/L,
    L = 2||A||^2 / T,   u = grad[(1/T)||r - A v||^2 + beta*1^T v]

whose inner dual problem eq (35) is solved by the block-coordinate scheme of
Prop. 2:

    lam^{k+1} = eps/(1 + gamma*zeta*eps) * (log v0 - log xi^k)
    Psi^{k+1} = argmin_{||Psi||_{1,inf} <= eta} <(v^{k+1} 1^T) * K, exp(Psi/eps)>

with ``v0 = v^{j} * exp(-gamma*u)``, ``K = exp(-C/eps)``,
``xi^k = exp(-C/eps + Psi^k/eps) 1_G`` and ``v = exp(lam/eps)``.

Numerics. The paper's ``eps`` is 1e-5..1e-6, so ``K = exp(-C/eps)`` underflows
almost everywhere (``C/eps`` reaches 2.5e4..2.5e5). *Everything* here is
therefore carried in the log domain: the code's state variables are
``log v``, ``log_v_dual = lam/eps`` and ``psi_q = Psi/eps``, and the only
exponentials taken are inside a max-shifted log-sum-exp.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Quadratic",
    "SolveResult",
    "build_quadratic",
    "solve_debiased",
    "solve_stochastic",
    "water_fill_log",
]


def _logsumexp(a: NDArray, axis: int) -> NDArray:
    mx = np.max(a, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.squeeze(mx, axis=axis) + np.log(np.sum(np.exp(a - mx), axis=axis))


# --------------------------------------------------------------------------
# the |.|_{1,inf}-constrained sub-problem (Prop. 2's Psi step)
# --------------------------------------------------------------------------


def water_fill_log(
    log_b: NDArray, budget: float, max_active: int = 64, out: NDArray | None = None
) -> NDArray:
    """Solve, independently along the last axis, the Psi step of Prop. 2.

    For one pitch candidate the problem is

        min_{s >= 0, sum_f s_f <= budget}  sum_f exp(log_b[f] - s[f])

    which is what ``argmin_{||Psi||_{1,inf} <= eta} <v 1^T * K, exp(Psi/eps)>``
    becomes after the substitutions ``log_b = log v + log K``, ``s = -Psi/eps``
    and ``budget = eta/eps``. The dual norm ``||Psi||_{1,inf} = max_g sum_f
    |Psi[f,g]|`` bounds *each* pitch candidate's l1 mass by ``eta`` on its own,
    and the objective decreases in every ``-Psi``, so the budget is spent in
    full and ``Psi <= 0``.

    The KKT conditions give a *water-filling* solution: on the active set
    ``exp(log_b[f] - s[f])`` equals a common level ``tau``, so
    ``s[f] = (log_b[f] - log_tau)_+`` with ``log_tau`` fixed by the budget.
    Sorting ``log_b`` descending and scanning the prefix sums finds it — the
    "only a sorting operation" of the paper's [26, Thm 2]. The scan is
    identical in form to Euclidean projection onto the simplex.

    Because ``budget = eta/eps`` is enormous in absolute terms (1e3..1e4) yet
    tiny against the spread of ``log_b`` (``C/eps`` spans up to 2.5e5), the
    active set is small — a few entries per candidate, the near-tooth
    frequencies. Only the top ``max_active`` entries are sorted; the window is
    grown and the solve retried if any row saturates it.

    Parameters
    ----------
    log_b : (..., F) array
        ``log v[f] + log K[f, g]``, i.e. ``log v[f] - C[f, g]/eps``, laid out
        pitch-major so that the water-filled axis is contiguous.
    budget : float
        ``eta / eps``, the per-candidate l1 budget in log units.
    max_active : int
        Initial size of the sorted window; grown by 4x on saturation.
    out : array, optional
        Destination buffer for the result.

    Returns
    -------
    s : same shape as ``log_b``, ``>= 0``. ``Psi = -eps * s``.
    """
    log_b = np.asarray(log_b, dtype=np.float64)
    if budget <= 0:
        if out is None:
            return np.zeros_like(log_b)
        out.fill(0.0)
        return out
    n_freq = log_b.shape[-1]
    window = int(min(n_freq, max(1, max_active)))

    while True:
        if window >= n_freq:
            top = -np.sort(-log_b, axis=-1)
        else:
            part = np.partition(log_b, n_freq - window, axis=-1)[..., n_freq - window :]
            top = -np.sort(-part, axis=-1)
        csum = np.cumsum(top, axis=-1)
        counts = np.arange(1, top.shape[-1] + 1, dtype=np.float64)
        log_tau = (csum - budget) / counts
        # `top > log_tau` holds exactly on a prefix (simplex-projection lemma).
        n_active = np.count_nonzero(top > log_tau, axis=-1)
        if window < n_freq and np.any(n_active >= window):
            window = int(min(n_freq, window * 4))
            continue
        break

    n_active = np.maximum(n_active, 1)
    level = np.take_along_axis(log_tau, n_active[..., None] - 1, axis=-1)
    res = np.subtract(log_b, level, out=out)
    return np.maximum(res, 0.0, out=res)


# --------------------------------------------------------------------------
# the smooth data-fit part
# --------------------------------------------------------------------------


@dataclass
class Quadratic:
    """The smooth term ``(1/T)||r - A v||^2`` in Gram form, plus its step size.

    ``A`` has columns ``a(w_f) = [1, e^{i w_f}, ..., e^{i w_f (T-1)}]^T``. For
    real ``v`` the real-stacked least-squares problem of the paper's footnote 3
    has Gram ``Re(A^H A)`` and correlation ``Re(A^H r)``, which is all the
    solver ever needs — the dictionary itself is never materialized.

    On a *uniform* frequency grid the Gram is symmetric Toeplitz (see
    :func:`_gram_row`), so it is stored as its first row and applied by FFT
    through a circulant embedding: ``O(F log F)`` and a few hundred kB of
    traffic per iteration instead of ``O(F^2)`` and tens of MB. The dense
    matrix is kept only as the fallback for a non-uniform grid.
    """

    corr: NDArray  # (F,) = Re(A^H r)
    n_lags: int  # T
    op_norm_sq: float  # ||A||^2 (largest eigenvalue of the Gram)
    freqs_rad: NDArray = field(repr=False)  # (F,) analysis grid, rad/sample
    r0: float = 1.0  # r_hat(0), the frame's total in-band power
    gram: NDArray | None = field(default=None, repr=False)  # dense fallback
    kernel: NDArray | None = field(default=None, repr=False)  # FFT of the embedding
    fft_size: int = 0

    @property
    def n_freq(self) -> int:
        return self.corr.size

    @property
    def step(self) -> float:
        """``gamma = 1/L`` with ``L = 2||A||^2 / T`` (Sec. VII-A)."""
        return self.n_lags / (2.0 * self.op_norm_sq)

    def matvec(self, nu: NDArray) -> NDArray:
        """``Re(A^H A) v``."""
        if self.kernel is None:
            assert self.gram is not None
            return self.gram @ nu
        pad = np.zeros(self.fft_size, dtype=np.float64)
        pad[: nu.size] = nu
        return np.fft.irfft(np.fft.rfft(pad) * self.kernel, self.fft_size)[: nu.size]

    def grad(self, nu: NDArray, beta: float = 0.0) -> NDArray:
        """``grad[(1/T)||r - A v||^2 + beta 1^T v]`` at ``v = nu``."""
        return (2.0 / self.n_lags) * (self.matvec(nu) - self.corr) + beta

    def residual_sq(self, nu: NDArray, r_energy: float) -> float:
        """``||r - A v||^2`` from the Gram, given ``||r||^2``."""
        return float(r_energy - 2.0 * nu @ self.corr + nu @ self.matvec(nu))

    def with_corr(self, corr: NDArray, r0: float) -> Quadratic:
        """Same dictionary and step size, a different frame's covariance."""
        return replace(self, corr=np.asarray(corr, dtype=np.float64), r0=float(r0))


def _dirichlet(x: NDArray, n: int) -> NDArray:
    """``sum_{t=0}^{n-1} exp(i x t)``, stable at ``x -> 0 (mod 2 pi)``."""
    half = 0.5 * x
    sin_half = np.sin(half)
    out = np.empty(x.shape, dtype=np.complex128)
    small = np.abs(sin_half) < 1e-12
    out[small] = n
    xs = x[~small]
    out[~small] = np.exp(0.5j * xs * (n - 1)) * (np.sin(0.5 * xs * n) / np.sin(0.5 * xs))
    return out


def _is_uniform(freqs_rad: NDArray) -> bool:
    diffs = np.diff(freqs_rad)
    return bool(freqs_rad.size > 2 and np.allclose(diffs, diffs[0], rtol=1e-9, atol=1e-15))


def _gram_row(freqs_rad: NDArray, n_lags: int) -> NDArray:
    """First row of the symmetric Toeplitz ``Re(A^H A)`` on a uniform grid.

    ``(A^H A)[f, f'] = sum_t exp(i (w_{f'} - w_f) t)`` is a Dirichlet kernel of
    the frequency *difference*, and on a uniform grid that difference depends
    only on ``f' - f``. ``Re D`` is even in the lag, so the real Gram is
    symmetric Toeplitz and this one ``O(F)`` row determines it.
    """
    lags = np.diff(freqs_rad)[0] * np.arange(freqs_rad.size, dtype=np.float64)
    return np.real(_dirichlet(lags, n_lags))


def _dense_gram(freqs_rad: NDArray, n_lags: int) -> NDArray:
    """``Re(A^H A)`` for an arbitrary (possibly non-uniform) grid."""
    if _is_uniform(freqs_rad):
        from scipy.linalg import toeplitz

        return toeplitz(_gram_row(freqs_rad, n_lags))
    tau = np.arange(n_lags, dtype=np.float64)
    dic = np.exp(1j * np.outer(tau, freqs_rad))
    return np.real(dic.conj().T @ dic)


def _circulant_kernel(row: NDArray) -> tuple[NDArray, int]:
    """FFT of the circulant embedding of a symmetric Toeplitz matrix."""
    n = row.size
    size = int(2 ** np.ceil(np.log2(2 * n)))
    emb = np.zeros(size, dtype=np.float64)
    emb[:n] = row
    emb[size - n + 1 :] = row[:0:-1]
    return np.fft.rfft(emb), size


def _correlate(r_hat: NDArray, freqs_rad: NDArray, chunk: int = 512) -> NDArray:
    """``Re(A^H r)``, chunked over frequency to bound peak memory."""
    tau = np.arange(r_hat.size, dtype=np.float64)
    out = np.empty(freqs_rad.size, dtype=np.float64)
    for start in range(0, freqs_rad.size, chunk):
        stop = min(start + chunk, freqs_rad.size)
        phase = np.exp(-1j * np.outer(freqs_rad[start:stop], tau))
        out[start:stop] = np.real(phase @ r_hat)
    return out


def _largest_eigenvalue(matvec: Callable[[NDArray], NDArray], n: int) -> float:
    """Largest eigenvalue of a symmetric PSD operator, by Lanczos.

    This fixes ``L = 2||A||^2/T`` and hence the step size ``gamma = 1/L``, so
    an *under*-estimate would make the outer iteration diverge; the power
    iteration fallback therefore carries a safety factor.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    class _SymOperator(LinearOperator):
        def __init__(self) -> None:
            super().__init__(dtype=np.dtype(np.float64), shape=(n, n))

        def _matvec(self, x: NDArray) -> NDArray:
            return matvec(np.asarray(x, dtype=np.float64).ravel())

        def _rmatvec(self, x: NDArray) -> NDArray:
            return self._matvec(x)

    try:
        val = float(eigsh(_SymOperator(), k=1, which="LA", return_eigenvectors=False)[0])
        if np.isfinite(val) and val > 0:
            return val
    except Exception:  # pragma: no cover - ARPACK convergence fallback
        pass
    vec = np.ones(n, dtype=np.float64)
    val = 0.0
    for _ in range(500):
        vec = matvec(vec)
        val = float(np.linalg.norm(vec))
        if val <= 0:
            return 1.0
        vec /= val
    return val * 1.02  # guard against a power-iteration underestimate


def build_quadratic(r_hat: NDArray, freqs_rad: NDArray) -> Quadratic:
    """Assemble the Gram-form data-fit term for one covariance vector."""
    r_hat = np.asarray(r_hat, dtype=np.complex128)
    freqs_rad = np.asarray(freqs_rad, dtype=np.float64)
    n_lags = r_hat.size
    n_freq = freqs_rad.size
    corr = _correlate(r_hat, freqs_rad)
    r0 = float(np.real(r_hat[0])) if n_lags else 1.0
    if _is_uniform(freqs_rad):
        kernel, size = _circulant_kernel(_gram_row(freqs_rad, n_lags))
        quad = Quadratic(
            corr=corr,
            n_lags=n_lags,
            op_norm_sq=1.0,
            freqs_rad=freqs_rad,
            r0=r0,
            kernel=kernel,
            fft_size=size,
        )
    else:
        quad = Quadratic(
            corr=corr,
            n_lags=n_lags,
            op_norm_sq=1.0,
            freqs_rad=freqs_rad,
            r0=r0,
            gram=_dense_gram(freqs_rad, n_lags),
        )
    return replace(quad, op_norm_sq=_largest_eigenvalue(quad.matvec, n_freq))


# --------------------------------------------------------------------------
# the outer Bregman proximal-gradient loop
# --------------------------------------------------------------------------


@dataclass
class SolveResult:
    """Output of one call to :func:`solve_stochastic` / :func:`solve_debiased`."""

    nu: NDArray  # (F,) spectral estimate
    pitch_mass: NDArray  # (G,) M^T 1_F, the distribution over pitch candidates
    n_iter: int
    converged: bool
    log_v: NDArray | None = None  # (F,) lam/eps, the frequency dual
    psi_q: NDArray | None = None  # (G, F) Psi/eps, the group-sparsity dual


class _InnerDual:
    """Block-coordinate descent on eq (35) (Prop. 2), all in the log domain.

    State is kept **pitch-major**, ``(G, F)``: both reductions the sweep needs
    — the log-sum-exp over pitch candidates and the water-filling over
    frequencies — then run over a contiguous axis, and the buffers can be
    reused across outer iterations instead of reallocating ``F * G`` doubles
    every sweep.

    ``cost_qt = (C/eps).T``; ``psi_q = Psi/eps`` (warm-started, ``<= 0``);
    ``alpha = 1/(1 + gamma*zeta*eps)``.
    """

    def __init__(self, cost_qt: NDArray, budget: float, alpha: float, max_active: int) -> None:
        self.cost_qt = cost_qt
        self.budget = budget
        self.alpha = alpha
        self.max_active = max_active
        self.psi_q = np.zeros_like(cost_qt)
        self._work = np.empty_like(cost_qt)
        self.log_v = np.zeros(cost_qt.shape[1], dtype=np.float64)
        self._primed = False

    def _lambda_step(self, log_nu0: NDArray) -> None:
        # log xi = logsumexp_g (Psi - C)/eps ; lam/eps = alpha (log nu0 - log xi)
        np.subtract(self.psi_q, self.cost_qt, out=self._work)
        mx = self._work.max(axis=0)
        self._work -= mx
        np.exp(self._work, out=self._work)
        log_xi = mx + np.log(self._work.sum(axis=0))
        np.subtract(log_nu0, log_xi, out=self.log_v)
        self.log_v *= self.alpha

    def _psi_step(self) -> None:
        np.subtract(self.log_v, self.cost_qt, out=self._work)  # log b, (G, F)
        water_fill_log(self._work, self.budget, self.max_active, out=self.psi_q)
        np.negative(self.psi_q, out=self.psi_q)

    def sweep(self, log_nu0: NDArray, n_iter: int) -> NDArray:
        """Run ``n_iter`` block-coordinate sweeps; return ``lam/eps``.

        Each sweep is Psi-then-lambda, so the lambda step is always the last
        thing done and ``M 1_G = v`` holds exactly at the point the outer
        iteration reads it. ``Psi`` and ``lambda`` are both warm-started from
        the previous outer iteration; only the very first call needs a
        priming lambda step, since ``log_v`` has no meaningful value yet.
        """
        if not self._primed:
            self._lambda_step(log_nu0)
            self._primed = True
        for _ in range(max(1, n_iter)):
            self._psi_step()
            self._lambda_step(log_nu0)
        return self.log_v

    def log_pitch_mass(self) -> NDArray:
        """``log(M^T 1_F)`` — the eq-(34) mass on each pitch candidate."""
        np.add(self.log_v, self.psi_q, out=self._work)
        self._work -= self.cost_qt
        return _logsumexp(self._work, axis=1)


def solve_stochastic(
    quad: Quadratic,
    cost: NDArray,
    *,
    eta: float,
    zeta: float,
    eps: float,
    beta: float,
    max_iter: int = 300,
    tol: float = 1e-6,
    inner_iters: int = 5,
    max_active: int = 64,
    nu_init: NDArray | float | None = None,
) -> SolveResult:
    """Solve eq (27) by the Bregman proximal-gradient scheme of Prop. 1.

    Parameters
    ----------
    quad : Quadratic
        The data-fit term for this frame.
    cost : (F, G) array
        The eq-(18) ground cost.
    eta, zeta, eps, beta : float
        Paper hyper-parameters (Tables I / II, ``Proposed_c`` column).
    max_iter, tol : int, float
        Outer budget; ``tol`` is on the relative l1 change of ``nu``.
    inner_iters : int
        Block-coordinate sweeps of Prop. 2 per outer step. The dual is
        warm-started across outer steps, so a handful suffices.
    nu_init : array or float, optional
        Starting spectrum. Multiplicative updates cannot revive an exact zero,
        so this must be strictly positive; default is the non-negative part of
        the matched filter ``Re(A^H r)/T``, floored well below its peak. The
        paper does not state an initialization; a flat one also converges but
        needs several times more iterations.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_freq, _ = cost.shape
    if quad.n_freq != n_freq:
        raise ValueError(f"cost has F={n_freq}, quadratic has F={quad.n_freq}")

    nu = _initial_nu(quad, n_freq, nu_init)

    gamma = quad.step
    inner = _InnerDual(
        cost_qt=np.ascontiguousarray(cost.T / eps),
        budget=eta / eps,
        alpha=1.0 / (1.0 + gamma * zeta * eps),
        max_active=max_active,
    )

    log_nu = np.log(nu)
    converged = False
    it = 0
    for step in range(1, max_iter + 1):
        it = step
        nu = np.exp(log_nu)
        log_nu0 = log_nu - gamma * quad.grad(nu, beta)
        log_v = inner.sweep(log_nu0, inner_iters)
        # v^{j+1} = v0 * exp(-gamma*zeta*lam) with lam = eps * log_v.
        log_nu_new = log_nu0 - gamma * zeta * eps * log_v
        nu_new = np.exp(log_nu_new)
        denom = float(np.abs(nu).sum()) + 1e-300
        rel = float(np.abs(nu_new - nu).sum()) / denom
        log_nu = log_nu_new
        if rel < tol:
            converged = True
            break

    # M^T 1_F is a partition of nu over the pitch candidates, so its entries
    # are powers on the same scale as nu and exponentiate safely.
    log_mass = inner.log_pitch_mass()
    return SolveResult(
        nu=np.exp(log_nu),
        pitch_mass=np.exp(log_mass),
        n_iter=it,
        converged=converged,
        log_v=inner.log_v.copy(),
        psi_q=inner.psi_q,
    )


def _initial_nu(quad: Quadratic, n_freq: int, nu_init: NDArray | float | None) -> NDArray:
    if nu_init is not None:
        nu = np.broadcast_to(np.asarray(nu_init, dtype=np.float64), (n_freq,)).astype(np.float64)
        if np.any(nu <= 0):
            raise ValueError("nu_init must be strictly positive")
        return nu
    matched = np.maximum(quad.corr, 0.0)
    total = float(matched.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(n_freq, max(quad.r0, 1e-12) / n_freq)
    # Scale so that sum_f nu_f = r(0): at tau = 0 the model gives exactly that,
    # which puts the start on the correct power scale instead of the matched
    # filter's T-fold over-estimate.
    nu = matched * (max(quad.r0, 1e-12) / total)
    return np.maximum(nu, 1e-8 * nu.max())


def solve_debiased(
    quad: Quadratic,
    cost_min: NDArray,
    *,
    zeta: float,
    beta: float,
    max_iter: int = 300,
    tol: float = 1e-6,
    nu_init: NDArray | float | None = None,
) -> SolveResult:
    """Solve the debiasing program eq (37).

        min_{v >= 0} (1/T)||r - A v||^2 + beta * 1^T v + zeta * c_min^T v

    with ``[c_min]_f = min_g C[f, g]`` over the *retained* pitch candidates.
    This is the ``eta = 0``, ``eps -> 0`` limit of eq (27) restricted to the
    active pitch grid (Sec. VII-C): the inner problem then converges in one
    step with ``Psi = 0``, and Prop. 1's update degenerates to plain
    exponentiated gradient descent on a non-negatively weighted LASSO.
    """
    cost_min = np.asarray(cost_min, dtype=np.float64)
    n_freq = cost_min.size
    nu = _initial_nu(quad, n_freq, nu_init)
    gamma = quad.step
    linear = beta + zeta * cost_min
    log_nu = np.log(nu)
    converged = False
    it = 0
    for step in range(1, max_iter + 1):
        it = step
        nu = np.exp(log_nu)
        log_nu_new = log_nu - gamma * quad.grad(nu, 0.0) - gamma * linear
        nu_new = np.exp(log_nu_new)
        denom = float(np.abs(nu).sum()) + 1e-300
        rel = float(np.abs(nu_new - nu).sum()) / denom
        log_nu = log_nu_new
        if rel < tol:
            converged = True
            break
    return SolveResult(
        nu=np.exp(log_nu),
        pitch_mass=np.zeros(0),
        n_iter=it,
        converged=converged,
    )
