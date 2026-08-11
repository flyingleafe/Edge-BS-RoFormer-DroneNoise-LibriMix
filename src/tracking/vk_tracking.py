"""Coupled Vold–Kalman (VK) order tracking for rotor-speed refinement.

Implements the design in ``docs/vk-order-tracking-design.md``: a 2nd-generation
Vold–Kalman order filter (Vold & Leuridan 1993; Tuma 2005 bandwidth relation)
with three deviations from the textbook formulation that make it tractable and
turn it into a *tracker* rather than a filter with known frequencies:

1. **Demodulate + decimate** — envelopes are narrowband by construction, so
   each track is demodulated by its phasor (``y * conj(c_m)``), zero-phase
   low-passed and decimated to an envelope grid ``fs_env`` (default 100 Hz);
   the VK-2 normal equations are solved on the decimated grid.
2. **Sparse coupling groups** — the cross terms ``conj(c_m) c_n`` that make
   tracks compete for spectral energy (explaining-away) survive decimation
   iff the instantaneous frequency separation is below ~``fs_env / 2``.
   Tracks are partitioned by union-find on that predicate and each group
   solves its own coupled system.
3. **Outer frequency loop** — VK gives envelopes given frequencies; the
   trajectory itself is refined by the phase-slope of the envelopes
   (stage D of :mod:`tracking.rps_refinement`, Fisher weights
   ``k^2 |x|^2``), fused across channels and harmonics by a
   smoothness-regularised weighted 1-D solve, with an annealing schedule on
   ``k_max`` (and the demod/VK bandwidth) for basin capture.

CPU-inference fast paths (2026-07-20, see the vk_bench profiling study): the
coupled-group normal equations are solved as a time-major-interleaved
Hermitian **banded** Cholesky system (``VKConfig.solver``, splu kept as the
reference/fallback), and cross-coupling terms are only demodulated for pairs
that can actually beat inside the lowpass band (``VKConfig.prune_far_pairs``).

Conventions match ``rps_refinement.py``: trajectories in rev/s with harmonics
at ``k * r`` Hz, arrays time-last, mono-or-multichannel audio plus a
trajectory sampled on an arbitrary time grid. Every demodulation goes through
:mod:`tracking.dsp`; the linear algebra stays numpy/scipy float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded, solveh_banded
from scipy.sparse.linalg import splu

from tracking.dsp import demod, torch_device, zoom_bands

__all__ = [
    "VKConfig",
    "Envelopes",
    "VKResult",
    "demodulate",
    "edge_taper",
    "ls_project_envelopes",
    "vk_envelopes",
    "vk_track",
    "vk_reconstruct",
]

_TINY = 1e-30
_MAX_CHANNELS = 8  # design §3: cap multichannel fusion at 8 channels


@dataclass(frozen=True)
class VKConfig:
    """Knobs for the coupled VK tracker (defaults tuned for 16 kHz drones)."""

    fs: float = 16000.0
    fs_env: float = 100.0  # envelope-grid rate (actual rate = fs / stride)
    p: int = 2  # difference-prior order; only 2 supported (design §3)
    bw_hz: float = 1.0  # envelope −3 dB bandwidth (Tuma relation, at fs_env)
    k_min: int = 1
    k_max: int = 40
    f_min: float = 60.0  # ignore harmonics below (rumble / DC)
    f_max: float = 6000.0  # and above (weak, broadband-dominated)
    couple_hz: float | None = None  # coupling predicate; None → fs_env / 2
    n_outer: int = 6  # outer frequency-loop rounds
    traj_lambda: float = 1e4  # trajectory-smoothness weight (D2 on env grid)
    max_step: float = 0.5  # per-round clip on |delta r| (rev/s)
    k_schedule: str = "grow"  # "grow" (anneal k_max + bandwidth) | "fixed"
    min_rps: float = 5.0  # rotors below carry no comb (near-silence)
    # --- capture / robustness knobs (beyond the paper formulation) ---
    bw_start_hz: float | None = None  # initial annealed bandwidth; None → 0.2*fs_env
    split_eps: float = 0.05  # rev/s: rotor pairs closer than this everywhere...
    split_nudge: float = 0.1  # ...get a deterministic ±nudge (symmetry breaking)
    update_gate: float = 30.0  # skip a rotor's freq update when no track's
    # demodulated periodogram shows a peak/median ratio above this: a comb
    # line anywhere in the demod band (locked or detuned) peaks at >~70 even
    # at −10 dB band SNR, white noise tops out near ~15 — no peak means no
    # comb, and a rotor without a comb must not drift (design §3 / test 5)
    conf_window_s: float = 1.0  # window for the reported confidence
    sep_bw_factor: float = 6.0  # per-group bandwidth clamp: a coupled group's
    # VK bandwidth is capped at this multiple of its minimum track separation
    # (a passband much wider than the separation cannot attribute the lines —
    # the near-degenerate solve blows up in a cancelling mode whose phase
    # slope points at the pair mean, i.e. twin collapse)
    solver: str = "banded"  # coupled-group linear solver: "banded" =
    # time-major-interleaved Hermitian banded Cholesky (_solve_group_banded;
    # ~5-10x faster than splu with zero fill-in, falls back to splu per
    # group if the assembled system is numerically non-PD); "splu" = the
    # original track-major sparse LU (reference implementation).
    prune_far_pairs: bool = True  # skip cross-coupling demod for track pairs
    # whose instantaneous frequency separation never enters the demod lowpass
    # band (min separation >= 1.25x the 0.45*fs_env cutoff): their LP'd
    # coupling coefficient is pure spectral leakage (~1e-3 relative), and
    # skipping them removes the dominant share of full-length pair demods in
    # richly-coupled configs. Pairs with no co-valid samples are always
    # skipped (their contribution is exactly zero).
    bw_adapt: bool = False  # IAVKF-style per-track bandwidth adaptation
    # (Jiang, Chen & Wang, IEEE TII 2024, eqs 25-27): after each outer-round
    # envelope solve, each track's selectivity rho_m^2 is multiplied by the
    # orthogonal factor |<z_m, a_m>| / <a_m, a_m> computed in the demodulated
    # domain (a_m = x_m / 2, the envelope on z's scale, over the round's valid
    # samples). The factor is >= ~1 while the envelope is still over-smoothed
    # relative to the in-band observation and -> 1 at lock, so the band
    # narrows early and stabilises. The cumulative gain is a persistent
    # per-(rotor, harmonic) state applied ON TOP of the annealing schedule's
    # per-round bw_hz — annealing and adaptation compose, they do not fight.
    bw_adapt_clamp: float = 4.0  # cumulative rho_m^2 adaptation is clamped to
    # [rho0^2 / clamp^2, rho0^2 * clamp^2] (rho0 = the schedule's selectivity)
    bw_rps: float | None = None  # k-scaled per-track bandwidth: when set, each
    # track's target −3 dB bandwidth is k * bw_rps Hz (k = the track's harmonic
    # index), so the capture radius is bw_rps rev/s at EVERY harmonic. This
    # REPLACES the scalar bw_hz / anneal schedule as the band source (the
    # anneal still applies to k_max via k_schedule); bw_adapt gains still
    # compose multiplicatively on rho².
    freq_weight: str = "k2_amp"  # _freq_update fusion weight: "k2_amp" =
    # k^2 |x[t+1] conj(x[t])| (the original Fisher form); "k_beta" =
    # k^freq_weight_beta with NO per-sample amplitude factor — the measured
    # per-harmonic variance law (WP18) is 1/v_k ∝ k^beta with no |p_k|.
    freq_weight_beta: float = 2.0  # the beta of the "k_beta" weight

    def __post_init__(self) -> None:
        if self.solver not in ("banded", "splu"):
            raise ValueError(f"unknown solver {self.solver!r} (expected 'banded' or 'splu')")
        if self.bw_adapt_clamp < 1.0:
            raise ValueError(f"bw_adapt_clamp must be >= 1, got {self.bw_adapt_clamp}")
        if self.bw_rps is not None and self.bw_rps <= 0:
            raise ValueError(f"bw_rps must be positive when set, got {self.bw_rps}")
        if self.freq_weight not in ("k2_amp", "k_beta"):
            raise ValueError(
                f"unknown freq_weight {self.freq_weight!r} (expected 'k2_amp' or 'k_beta')"
            )


@dataclass
class Envelopes:
    """Complex VK envelopes on the decimated grid plus their bookkeeping."""

    x: np.ndarray  # (C, M, T_env) complex128 envelopes
    z: np.ndarray  # (C, M, T_env) demodulated + decimated observations
    rotor: np.ndarray  # (M,) int — rotor index of each track
    k: np.ndarray  # (M,) int — harmonic index of each track
    valid: np.ndarray  # (M, T_env) bool — track validity mask
    t_env: np.ndarray  # (T_env,) seconds
    fs_env: float  # actual decimated rate (fs / stride)
    fs: float  # audio rate
    phase: np.ndarray  # (R, T_audio) rotor fundamental phase (rad)
    groups: list[list[int]] = field(default_factory=list)  # coupling groups
    bw_track: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (M,) Hz —
    # the VK bandwidth each track was actually solved with (after the
    # per-group separation clamp); feeds the noise-equivalent-bandwidth
    # normalisation of the envelope SNR


@dataclass
class VKResult:
    """Output of :func:`vk_track`."""

    r_refined: np.ndarray  # (R, N) rev/s on the input ``frame_times`` grid
    frame_times: np.ndarray  # (N,)
    r_env: np.ndarray  # (R, T_env) rev/s on the dense envelope grid
    t_env: np.ndarray  # (T_env,)
    envelopes: Envelopes  # final-round envelopes
    residual_ratios: list[float] = field(default_factory=list)  # per round
    max_deltas: list[float] = field(default_factory=list)  # per round
    confidence: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (R, W)
    conf_times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (W,)
    groups_log: list[list[list[tuple[int, int]]]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# building blocks


def _tuma_rho(bw_hz: float, fs_env: float, p: int) -> float:
    """Selectivity ``rho`` from the −3 dB bandwidth (Tuma 2005), at ``fs_env``.

    Derivation: the VK-2 transfer is ``H = 1 / (1 + rho^2 (2 sin(w/2))^(2p))``
    for the p-th difference prior; ``|H| = 1/sqrt(2)`` at the half-band edge
    ``w = pi * bw / fs_env`` gives ``rho = sqrt(sqrt(2) - 1) / (2 sin(w/2))^p``.
    """
    if p != 2:
        raise ValueError(f"only p = 2 is supported (higher orders ill-conditioned), got {p}")
    if bw_hz <= 0:
        raise ValueError(f"bw_hz must be positive, got {bw_hz}")
    phi = np.pi * bw_hz / fs_env  # half-bandwidth in rad/sample
    if phi >= np.pi:
        raise ValueError(f"bw_hz={bw_hz} exceeds the envelope rate fs_env={fs_env}")
    denom = (2.0 * np.sin(phi / 2.0)) ** p
    rho = float(np.sqrt(np.sqrt(2.0) - 1.0) / denom)
    rho_max = 1.0 / np.sqrt(np.finfo(np.float64).eps)
    if not np.isfinite(rho) or rho > rho_max:
        arg = 0.5 * (np.sqrt(np.sqrt(2.0) - 1.0) / rho_max) ** (1.0 / p)
        bw_min = 2.0 * fs_env / np.pi * float(np.arcsin(arg))
        raise ValueError(
            f"bandwidth too small: bw_hz={bw_hz} gives rho={rho:.3e} "
            f"(ill-conditioned); minimum usable bandwidth at fs_env={fs_env} "
            f"is ~{bw_min:.3g} Hz"
        )
    return rho


def _tuma_bw_min(fs_env: float, p: int) -> float:
    """Minimum usable −3 dB bandwidth (Hz): the band where rho hits ``rho_max``.

    The same limit :func:`_tuma_rho` raises at, with a small safety margin so
    a band clipped to this floor never trips the ill-conditioning check.
    """
    rho_max = 1.0 / np.sqrt(np.finfo(np.float64).eps)
    arg = 0.5 * (np.sqrt(np.sqrt(2.0) - 1.0) / rho_max) ** (1.0 / p)
    return 2.0 * fs_env / np.pi * float(np.arcsin(arg)) * (1.0 + 1e-6)


def _tuma_bw(rho: np.ndarray, fs_env: float, p: int) -> np.ndarray:
    """Inverse of :func:`_tuma_rho`: −3 dB bandwidth (Hz) from selectivity.

    Vectorised over ``rho``; used to report the *effective* per-track
    bandwidth when ``bw_adapt`` rescales rho away from the schedule's value
    (feeds the noise-equivalent-bandwidth normalisation of the envelope SNR).
    """
    arg = 0.5 * (np.sqrt(np.sqrt(2.0) - 1.0) / np.asarray(rho, dtype=np.float64)) ** (1.0 / p)
    return 2.0 * fs_env / np.pi * np.arcsin(np.minimum(arg, 1.0))


def _vk_noise_bandwidth(bw_hz: float, fs_env: float, p: int) -> float:
    """Two-sided noise-equivalent bandwidth (Hz) of the VK-2 envelope filter.

    ``B_neq = fs_env * mean(|H(w)|^2)`` over the digital band, with
    ``H = 1 / (1 + rho^2 (2 sin(w/2))^(2p))``. White noise through the VK
    solve retains exactly this much of the demodulated band's power — the
    reference floor for the envelope SNR.
    """
    rho = _tuma_rho(bw_hz, fs_env, p)
    w = np.linspace(0.0, np.pi, 2049)
    h = 1.0 / (1.0 + rho**2 * (2.0 * np.sin(w / 2.0)) ** (2 * p))
    return fs_env * float(np.mean(h**2))


def second_diff(n: int) -> sparse.csr_array:
    """``(n-2, n)`` second-difference operator ``[1, -2, 1]``."""
    if n < 3:
        raise ValueError(f"need at least 3 samples for a second difference, got {n}")
    d2 = (
        sparse.eye_array(n - 2, n, k=0)
        - 2.0 * sparse.eye_array(n - 2, n, k=1)
        + sparse.eye_array(n - 2, n, k=2)
    )
    return d2.tocsr()


def edge_taper(n_env: int) -> np.ndarray:
    """``(n_env,)`` fade of the VK data weight at both ends of the window.

    The zero-phase LP's first/last few envelope samples are padding-guess
    transients; fitting them flares the envelopes at the boundaries. The weight
    ramps over the filter settling span (~3 cutoff periods = ``3 / 0.45``
    envelope samples) and the smoothness prior extrapolates instead. Part of the
    OBJECTIVE, not of the solve, which is why anything that profiles the VK cost
    (:mod:`tracking.fitness_vk`) reads the same taper from here.
    """
    n_edge = min(8, max(1, n_env // 4))
    taper = np.ones(n_env)
    ramp = (np.arange(n_edge) + 1.0) / (n_edge + 1.0)
    taper[:n_edge] = ramp
    taper[-n_edge:] = ramp[::-1]
    return taper


def env_stride(cfg: VKConfig) -> tuple[int, float]:
    """Decimation stride and the *actual* envelope rate ``fs / stride``."""
    stride = max(1, int(round(cfg.fs / cfg.fs_env)))
    return stride, cfg.fs / stride


def _lp_decimate(x: np.ndarray, stride: int, n_env: int) -> np.ndarray:
    """FFT brickwall lowpass + decimate: keep ``|f| <= 0.45 * fs_env``.

    Zoom-IFFT: zero-pad ``x`` to ``stride * n_env``, complex-FFT along the
    last axis, retain only the low band (positive *and* negative bins — the
    input is complex, the spectrum is not conjugate-symmetric), and inverse
    FFT at length ``n_env`` directly. Because the retained band fits inside
    the decimated Nyquist range, ``ifft(retained) / stride`` *is* the
    brickwall-filtered signal sampled at every ``stride``-th point — no
    audio-rate inverse transform needed.

    The transform is :func:`tracking.dsp.zoom_bands`, THE band-select kernel;
    ``band_env = 0.45`` gives the half-band as a fraction of the envelope
    grid, which is what makes the kept band independent of how the padded
    length is expressed. The FFT runs in complex64 (envelope precision is
    ~1e-7 relative, far below the VK solve's noise floor); phase/cumsum
    precision is the *caller's* concern and must stay float64 upstream.

    The retained band always fits inside the decimated Nyquist range (for
    ``band_env = 0.45``, ``floor(0.45 n_env) <= (n_env - 1) // 2`` for every
    ``n_env``), so the zoom identity holds on degenerate grids too and no
    short-clip special case is needed.
    """
    low, _ = zoom_bands(x, stride, n_env, band_env=0.45)
    return low.astype(np.complex128)


def _track_carriers(phase: np.ndarray, rotor: np.ndarray, k: np.ndarray, tracks, sign: float):
    """Yield ``(m, exp(sign * 1j * k[m] * phase[rotor[m]]))`` as complex64.

    Rotor-major recursion: one complex exp per rotor fundamental, then
    ``c_k = c_{k-1} * c_1`` per harmonic step instead of an exp per track
    (exp of a 1e5-sample float64 phase costs ~10x a complex64 multiply).
    ``phase`` must be float64 — it reaches ~1e7 rad and float32 drifts by
    radians — the exp is taken *before* the complex64 cast; the recursion's
    accumulated phase drift is ~k * eps(c64), negligible for k <= ~40.
    Yielded arrays must not be mutated (equal-k tracks share one array).
    """
    order = sorted(tracks, key=lambda m: (int(rotor[m]), int(k[m])))
    cur_rotor, cur_k = -1, 0
    c1 = cur = np.empty(0, dtype=np.complex64)
    for m in order:
        rot, kk = int(rotor[m]), int(k[m])
        if rot != cur_rotor:
            c1 = np.exp(sign * 1j * phase[rot]).astype(np.complex64)
            cur = c1**kk if kk != 1 else c1
            cur_rotor, cur_k = rot, kk
        elif kk != cur_k:
            if kk - cur_k > 2:  # rare gaps: one pow instead of many multiplies
                cur = cur * c1 ** (kk - cur_k)
            else:
                for _ in range(kk - cur_k):
                    cur = cur * c1
            cur_k = kk
        yield m, cur


def _track_table(n_rotors: int, k_lo: int, k_hi: int) -> tuple[np.ndarray, np.ndarray]:
    """``(M,)`` rotor and harmonic index per track, rotor-major."""
    ks = np.arange(k_lo, k_hi + 1, dtype=np.int64)
    rotor = np.repeat(np.arange(n_rotors, dtype=np.int64), len(ks))
    k = np.tile(ks, n_rotors)
    return rotor, k


def _coupling_groups(f: np.ndarray, valid: np.ndarray, couple_hz: float) -> list[list[int]]:
    """Union-find partition of tracks by ``min_t |f_m - f_n| < couple_hz``.

    ``f``: (M, T_env) instantaneous track frequencies; the minimum is taken
    over samples where *both* tracks are valid. Tracks with no valid samples
    join no group (they are excluded from the solve, envelope stays 0).
    """
    parent = np.arange(f.shape[0])

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    idx = np.where(valid.any(axis=1))[0]
    for ai in range(len(idx)):
        m = int(idx[ai])
        for bi in range(ai + 1, len(idx)):
            n = int(idx[bi])
            both = valid[m] & valid[n]
            if both.any() and float(np.min(np.abs(f[m, both] - f[n, both]))) < couple_hz:
                ra, rb = find(m), find(n)
                if ra != rb:
                    parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for m in idx:
        groups.setdefault(find(int(m)), []).append(int(m))
    return sorted(groups.values())


# ---------------------------------------------------------------------------
# coupled-group solvers


def _solve_group_splu(
    d2td2: sparse.csr_array,
    eye: Any,
    rho2: float | np.ndarray,
    w: list[np.ndarray],
    cross: dict[tuple[int, int], np.ndarray],
    z_g: np.ndarray,
) -> np.ndarray:
    """Reference coupled-group solve: track-major sparse LU (SuperLU).

    ``z_g``: (C, g, T_env) demodulated observations of the group's tracks;
    ``rho2``: squared selectivity, scalar or per-track ``(g,)`` (bandwidth
    adaptation); returns ``(g, T_env, C)`` envelopes. Verbatim the original
    formulation — kept as the A/B reference (``cfg.solver == "splu"``) and as
    the fallback when the banded Cholesky reports a numerically non-PD system.
    """
    g = len(w)
    n_ch, _, n_env = z_g.shape
    r2 = np.broadcast_to(np.asarray(rho2, dtype=np.float64), (g,))
    blocks: list[list[Any]] = [[None] * g for _ in range(g)]
    for a in range(g):
        reg = float(r2[a]) * d2td2 + 1e-8 * eye  # eps keeps masked spans well-posed
        blocks[a][a] = (reg + sparse.diags_array(w[a])).astype(np.complex128)
    for (a, b), g_mn in cross.items():
        blocks[a][b] = sparse.diags_array(w[a] * w[b] * g_mn)
        blocks[b][a] = sparse.diags_array(w[a] * w[b] * np.conj(g_mn))
    mat = sparse.bmat(blocks, format="csc", dtype=np.complex128)
    rhs = np.concatenate(
        [2.0 * w[a][None, :] * z_g[:, a] for a in range(g)], axis=-1
    ).T  # (g * T_env, C)
    return splu(mat).solve(np.ascontiguousarray(rhs)).reshape(g, n_env, n_ch)


def _solve_group_banded(
    d2td2_diags: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho2: float | np.ndarray,
    w: list[np.ndarray],
    cross: dict[tuple[int, int], np.ndarray],
    z_g: np.ndarray,
) -> np.ndarray:
    """Coupled-group solve as one Hermitian positive-definite *banded* system.

    With time-major interleaved unknowns (all tracks of one time sample
    adjacent: index ``t * g + a``) the normal equations are banded: the
    per-track second-difference prior couples only ``|t - t'| <= p`` (band
    offsets ``g`` and ``2g``), and cross-track coupling is same-time only
    (offsets ``1 .. g-1``) — total bandwidth ``p * g`` with *zero* fill-in,
    vs SuperLU's fill-heavy factorization of the track-major layout.
    Assembled directly in LAPACK upper-banded storage; factorized by
    ``cholesky_banded`` (zpbtrf) and solved by ``cho_solve_banded`` (zpbtrs)
    for all channels at once — O(T g^3) flops. The VK functional is PD by
    construction; should decimation artefacts ever make the assembled system
    numerically non-PD, ``cholesky_banded`` raises ``LinAlgError`` and the
    caller falls back to :func:`_solve_group_splu`.
    """
    g = len(w)
    n_ch, _, n_env = z_g.shape
    d0, d1, d2 = d2td2_diags
    u = 2 * g  # p = 2 time-blocks of superdiagonals
    n = g * n_env
    ab = np.zeros((u + 1, n), dtype=np.complex128)  # ab[u + i - j, j] = A[i, j]
    # Per-track squared selectivity (scalar rho2 broadcasts; the products
    # below are bitwise identical to the former scalar formulation).
    r2 = np.broadcast_to(np.asarray(rho2, dtype=np.float64), (g,))
    diag = d0[:, None] * r2[None, :] + 1e-8 + np.stack(w, axis=-1)  # (T_env, g)
    ab[u] = diag.reshape(-1)
    ab[u - g, g:] = np.repeat(d1, g) * np.tile(r2, n_env - 1)
    ab[0, 2 * g :] = np.repeat(d2, g) * np.tile(r2, n_env - 2)
    for (a, b), g_mn in cross.items():  # a < b: upper triangle, offset b - a
        ab[u - (b - a), b::g] = w[a] * w[b] * g_mn
    cb = cholesky_banded(ab, lower=False)  # raises LinAlgError if not PD
    rhs = np.empty((n_env, g, n_ch), dtype=np.complex128)
    for a in range(g):
        rhs[:, a] = (2.0 * w[a][:, None]) * z_g[:, a].T
    sol = cast(np.ndarray, cho_solve_banded((cb, False), rhs.reshape(n, n_ch)))
    return sol.reshape(n_env, g, n_ch).transpose(1, 0, 2)


# ---------------------------------------------------------------------------
# public API


def demodulate(audio: np.ndarray, phase: np.ndarray, cfg: VKConfig) -> np.ndarray:
    """Demodulate ``(C, T)`` (or ``(T,)``) audio by per-track phases.

    ``phase``: (M, T) instantaneous track phase in radians at audio rate
    (i.e. ``k_m * phi_rotor``). Returns ``z`` of shape ``(C, M, T_env)``:
    ``LP[y * exp(-j phase_m)]`` brickwall-lowpassed to ``0.45 * fs_env`` and
    decimated by the stride to the envelope grid.

    The general form — an arbitrary phase per track. When the tracks ARE a
    rotor-harmonic comb, :func:`_demod_tracks_fft` says so and the driver
    takes one exp per rotor instead of one per track.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    phase = np.atleast_2d(np.asarray(phase, dtype=np.float64))
    if phase.shape[-1] != y.shape[-1]:
        raise ValueError(f"phase length {phase.shape[-1]} != audio length {y.shape[-1]}")
    stride, _ = env_stride(cfg)
    n_env = len(range(0, y.shape[-1], stride))
    on, _ = demod(y, phase=phase, stride=stride, n_env=n_env, band_env=0.45)
    return on.astype(np.complex128)


def _demod_tracks_fft(
    audio: np.ndarray, phase: np.ndarray, rotor: np.ndarray, k: np.ndarray, cfg: VKConfig
) -> np.ndarray:
    """Structured form of :func:`demodulate` for rotor-harmonic tracks.

    Same result as ``demodulate(audio, k[:, None] * phase[rotor], cfg)``, but
    without materialising the ``(M, T)`` phase matrix or taking an exp per
    track: the driver gets the ``R`` conjugate fundamentals and builds every
    harmonic carrier by the power recursion ``c_k = c_{k-1} c_1``.
    """
    y32 = np.asarray(np.atleast_2d(audio), dtype=np.float32)
    stride, _ = env_stride(cfg)
    n_env = len(range(0, y32.shape[-1], stride))
    if len(rotor) == 0:
        return np.empty((y32.shape[0], 0, n_env), dtype=np.complex128)
    c1 = np.exp(-1j * np.atleast_2d(phase)).astype(np.complex64)
    on, _ = demod(y32, c1=c1, rotor=rotor, k=k, stride=stride, n_env=n_env, band_env=0.45)
    return on.astype(np.complex128)


def vk_envelopes(
    audio: np.ndarray,
    r: np.ndarray,
    cfg: VKConfig,
    *,
    k_hi: int | None = None,
    bw_hz: float | None = None,
    rho2_gain: np.ndarray | None = None,
) -> Envelopes:
    """One coupled VK-2 envelope solve (all coupling groups) given trajectories.

    ``audio``: ``(T,)`` or ``(C, T)``; ``r``: ``(R, T)`` rev/s at *audio* rate.
    ``k_hi`` / ``bw_hz`` override ``cfg.k_max`` / ``cfg.bw_hz`` (used by the
    outer loop's annealing schedule); ``rho2_gain`` is an optional ``(M,)``
    per-track multiplicative gain on the squared selectivity ``rho^2`` (the
    outer loop's ``bw_adapt`` state — applied on top of whatever bandwidth
    the schedule sets, and reflected in ``bw_track`` via the inverse Tuma
    relation). Solves, per coupling group and channel,

        (rho^2 D2' D2 + diag(w) + coupling) x = 2 w z

    on the decimated grid, where ``w`` is the 0/1 track-validity mask,
    coupling blocks are diagonal with entries ``w_m w_n LP[conj(c_m) c_n]``,
    and ``rho`` comes from the Tuma bandwidth relation at ``fs_env``. The
    factor 2 restores the real-signal envelope scale (``z ≈ a / 2`` for
    ``y = Re[a c]``). The linear solve is dispatched by ``cfg.solver``
    (time-major Hermitian banded Cholesky by default, track-major splu as
    reference/fallback); coupling pairs that cannot beat inside the lowpass
    band are skipped per ``cfg.prune_far_pairs``.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))[:_MAX_CHANNELS]
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    n_t = y.shape[-1]
    if r.shape[-1] != n_t:
        raise ValueError(f"r length {r.shape[-1]} != audio length {n_t} (audio-rate expected)")
    stride, fs_env = env_stride(cfg)
    t_env = np.arange(0, n_t, stride, dtype=np.float64) / cfg.fs
    n_env = len(t_env)
    k_top = cfg.k_max if k_hi is None else min(int(k_hi), cfg.k_max)
    bw = cfg.bw_hz if bw_hz is None else float(bw_hz)
    couple_hz = fs_env / 2.0 if cfg.couple_hz is None else cfg.couple_hz

    rotor, k = _track_table(r.shape[0], cfg.k_min, k_top)
    if rho2_gain is not None and len(rho2_gain) != len(rotor):
        raise ValueError(f"rho2_gain has {len(rho2_gain)} entries, expected {len(rotor)} tracks")
    phase = 2.0 * np.pi * np.cumsum(r, axis=-1) / cfg.fs  # (R, T) fundamental phase

    # Track validity mask on the envelope grid (design §3).
    r_dec = r[:, ::stride]
    f = k[:, None].astype(np.float64) * r_dec[rotor]  # (M, T_env)
    f_hi = min(cfg.f_max, 0.45 * cfg.fs)
    valid = (f >= cfg.f_min) & (f <= f_hi) & (r_dec[rotor] >= cfg.min_rps)

    z = _demod_tracks_fft(y, phase, rotor, k, cfg)
    groups = _coupling_groups(f, valid, couple_hz)

    d2 = second_diff(n_env)
    d2td2 = (d2.T @ d2).tocsr()  # banded, bandwidth 2p+1
    d2td2_diags = (
        np.asarray(d2td2.diagonal(0)),
        np.asarray(d2td2.diagonal(1)),
        np.asarray(d2td2.diagonal(2)),
    )
    eye = sparse.eye_array(n_env)

    x = np.zeros_like(z)
    bw_track = np.full(len(rotor), bw)
    taper = edge_taper(n_env)  # data-term edge fade; see the function's docstring
    # Coupling pairs separated by at least this never beat inside the demod
    # lowpass (cutoff 0.45 * fs_env): their LP'd cross term is pure spectral
    # leakage, ~1e-3 relative.
    prune_hz = 1.25 * 0.45 * fs_env
    for group in groups:
        g = len(group)
        w = [valid[m].astype(np.float64) * taper for m in group]
        # Per-group bandwidth clamp (see VKConfig.sep_bw_factor) + pair
        # separations, reused for the far-pair pruning below. Pairs with no
        # co-valid samples contribute exactly nothing (w_a * w_b == 0
        # everywhere) and are always skipped.
        bw_g = bw
        sep_cap = np.inf  # the per-group separation clamp, applied to any band source
        pair_sep: dict[tuple[int, int], float] = {}
        for a in range(g):
            for b in range(a + 1, g):
                m, n = group[a], group[b]
                both = valid[m] & valid[n]
                if both.any():
                    sep = float(np.min(np.abs(f[m, both] - f[n, both])))
                    pair_sep[(a, b)] = sep
                    sep_cap = min(sep_cap, max(cfg.bw_hz, cfg.sep_bw_factor * sep))
                    bw_g = min(bw_g, max(cfg.bw_hz, cfg.sep_bw_factor * sep))
        rho2: float | np.ndarray
        if cfg.bw_rps is not None:
            # k-scaled per-track band: b_m = clip(k_m * bw_rps, b_lo, b_hi),
            # b_lo the smallest Tuma-usable band, b_hi inside the ~0.9*fs_env
            # two-sided demod lowpass. The per-group separation clamp applies
            # AFTER, with the same floor semantics as the scalar path.
            b_lo = _tuma_bw_min(fs_env, cfg.p)
            b_hi = 0.9 * fs_env
            b_m = np.clip(k[group].astype(np.float64) * cfg.bw_rps, b_lo, b_hi)
            b_m = np.minimum(b_m, max(sep_cap, b_lo))
            bw_track[group] = b_m
            rho2 = np.array([_tuma_rho(float(b), fs_env, cfg.p) for b in b_m]) ** 2
        else:
            bw_track[group] = bw_g
            rho = _tuma_rho(bw_g, fs_env, cfg.p)
            rho2 = rho**2
        if rho2_gain is not None:
            rho2 = rho2 * np.asarray(rho2_gain, dtype=np.float64)[group]
            bw_track[group] = _tuma_bw(np.sqrt(rho2), fs_env, cfg.p)
        pair_list = [
            pair for pair, sep in pair_sep.items() if not (cfg.prune_far_pairs and sep >= prune_hz)
        ]
        # LP-decimated cross terms conj(c_m) c_n on the envelope grid:
        # per-track carriers via the shared recursion (one exp per rotor),
        # cross phasors as complex64 products, decimated in memory-bounded
        # batches by THE band-select kernel.
        cross: dict[tuple[int, int], np.ndarray] = {}
        if pair_list:
            carr = dict(_track_carriers(phase, rotor, k, group, sign=1.0))
            chunk_p = max(1, int(128e6 / (max(1, n_t) * 8)))
            cs = np.empty((min(chunk_p, len(pair_list)), n_t), dtype=np.complex64)
            for lo in range(0, len(pair_list), chunk_p):
                sub = pair_list[lo : lo + chunk_p]
                for i, (a, b) in enumerate(sub):
                    np.multiply(carr[group[b]], np.conj(carr[group[a]]), out=cs[i])
                gd = _lp_decimate(cs[: len(sub)], stride, n_env)
                for i, pair in enumerate(sub):
                    cross[pair] = gd[i]
        z_g = z[:, group]  # (C, g, T_env)
        sol: np.ndarray | None = None
        if cfg.solver == "banded":
            try:
                sol = _solve_group_banded(d2td2_diags, rho2, w, cross, z_g)
            except np.linalg.LinAlgError:
                sol = None  # numerically non-PD: use the reference path below
        if sol is None:
            sol = _solve_group_splu(d2td2, eye, rho2, w, cross, z_g)
        for a in range(g):
            x[:, group[a]] = sol[a].T

    return Envelopes(
        x=x,
        z=z,
        rotor=rotor,
        k=k,
        valid=valid,
        t_env=t_env,
        fs_env=fs_env,
        fs=cfg.fs,
        phase=phase,
        groups=groups,
        bw_track=bw_track,
    )


def vk_reconstruct(env: Envelopes, n_samples: int | None = None) -> np.ndarray:
    """Reconstruct the ``(C, T)`` audio-rate waveform ``sum_m Re[x_m c_m]``.

    Envelopes are linearly interpolated from the envelope grid back to audio
    rate (they are narrowband by construction, so linear interp is exact
    enough for diagnostics/overlays and the residual-ratio metric). The
    upsample exploits the uniform grid (every ``stride``-th audio sample):
    each inter-knot block is ``x[j] + (x[j+1] - x[j]) * ramp``, written
    sequentially — no per-sample index gathers — and carriers come from the
    :func:`_track_carriers` recursion (one exp per rotor). Beyond the last
    knot the envelope is held constant (``np.interp``'s clamp behaviour).
    """
    n_ch, n_tracks, n_env = env.x.shape
    n_t = env.phase.shape[-1] if n_samples is None else int(n_samples)
    if n_tracks == 0 or n_env == 0 or n_t == 0:
        return np.zeros((n_ch, n_t), dtype=np.float64)
    stride = int(round(float(env.t_env[1] - env.t_env[0]) * env.fs)) if n_env > 1 else 1
    n_full = min(n_t, (n_env - 1) * stride) if n_env > 1 else 0
    # float32 accumulation: carriers are complex64 already and the sum of
    # <=~160 unit-scale tracks keeps ~6 significant digits — plenty for the
    # residual-ratio / envelope-SNR diagnostics this feeds.
    recon = np.zeros((n_ch, n_t), dtype=np.float32)
    ramp = np.arange(stride, dtype=np.float32) / np.float32(stride)
    phase = env.phase[:, :n_t]
    for m, carr in _track_carriers(phase, env.rotor, env.k, range(n_tracks), sign=1.0):
        xm = env.x[:, m]  # (C, n_env)
        if not xm.any():  # masked / never-solved tracks stay zero
            continue
        # Re[x_up * c] = Re(x_up) Re(c) - Im(x_up) Im(c), accumulated in place.
        cr, ci = np.real(carr), np.imag(carr)
        xr = np.real(xm).astype(np.float32)
        xi = np.imag(xm).astype(np.float32)
        if n_full:
            up_r = (xr[:, :-1, None] + np.diff(xr, axis=-1)[:, :, None] * ramp).reshape(n_ch, -1)[
                :, :n_full
            ]
            up_i = (xi[:, :-1, None] + np.diff(xi, axis=-1)[:, :, None] * ramp).reshape(n_ch, -1)[
                :, :n_full
            ]
            up_r *= cr[:n_full]
            up_i *= ci[:n_full]
            recon[:, :n_full] += up_r
            recon[:, :n_full] -= up_i
        if n_full < n_t:
            recon[:, n_full:] += xr[:, -1:] * cr[n_full:]
            recon[:, n_full:] -= xi[:, -1:] * ci[n_full:]
    return recon.astype(np.float64)


# ---------------------------------------------------------------------------
# least-squares re-projection of the envelopes (peel subtraction)

#: Default gain-fit block. The fit spends 2 real parameters per (channel,
#: harmonic, block), so the block must be long enough to stay overdetermined
#: and to resolve one harmonic from its neighbours (0.25 s -> 4 Hz at a
#: harmonic spacing of one rev/s = 40-100 Hz), and short enough to stay inside
#: the envelope's coherence time (~1 s at the 1 Hz peel bandwidth) so a
#: drifting gain is still tracked.
LS_BLOCK_S = 0.25

#: Radial clamp on the fitted gain. The LS solution shrunk toward zero along
#: its own ray still cannot increase the residual (the residual is a convex
#: quadratic along that segment, equal to ||y||^2 at g = 0), so clamping is
#: safe; |g| >> 1 means the block is fitting something other than the modelled
#: component and must not be amplified into the peel.
LS_GAIN_MAX = 4.0


#: Target working set of one gain-fit tile, in bytes, on a CPU device. The fit
#: streams five ``(C, tile)`` float64 products per tile and the tile is re-read
#: six times, so the whole tile must stay in cache — with clip-long arrays
#: every one of those passes is a trip to DRAM, which is what made this stage
#: cost 7-11 s. A quarter of a megabyte keeps eight channels of a 0.25 s block
#: resident; below that the per-call dispatch starts to show, which is why the
#: tile is at least one block. A non-CPU device has no cache to block for and
#: pays for kernel launches instead, so there the tile is the whole clip.
LS_TILE_BYTES = 256 * 1024


def _ls_upsample(
    knot: Any, dknot: Any, ramp: Any, stride: int, s0: int, nc: int, n_env: int
) -> Any:
    """``(C, nc)`` linear envelope upsample of samples ``s0 ... s0 + nc``.

    ``knot`` is the complex64 envelope on its own grid and ``dknot`` its
    forward difference with a **zero last column**, so the "hold constant
    beyond the last knot" rule of :func:`vk_reconstruct` needs no special
    case: the tail knot contributes ``knot[-1] + 0 * ramp``. ``ramp`` is the
    ``stride``-long ramp ``[0, 1/stride, ...]``.

    The aligned case (the tile starts on a knot and spans whole knots) is a
    pure broadcast over a ``(C, n_knots, stride)`` view — no gather, which is
    what makes the basis cheap; anything else falls back to an index take.
    """
    import torch

    n_ch = knot.shape[0]
    if stride > 0 and s0 % stride == 0 and nc % stride == 0 and (s0 + nc) // stride <= n_env:
        nk, j0 = nc // stride, s0 // stride
        sl_k = knot[:, j0 : j0 + nk, None]
        sl_d = dknot[:, j0 : j0 + nk, None]
        return (sl_k + sl_d * ramp).reshape(n_ch, nc)
    idx = torch.arange(s0, s0 + nc, device=knot.device)
    ki = torch.clamp(idx // stride, max=n_env - 1)
    fr = (idx % stride).to(torch.float32) / float(stride)
    return knot[:, ki] + dknot[:, ki] * fr


def _ls_project(
    y: np.ndarray,
    env: Envelopes,
    *,
    stride: int,
    block: int,
    starts: np.ndarray,
    blk_env: np.ndarray,
    gain_max: float,
) -> tuple[np.ndarray, np.ndarray, int, list[np.ndarray]]:
    """THE gain-fit core of :func:`ls_project_envelopes`.

    Blocks are independent and channels are independent, so the per-track work
    is a sweep over tiles of :data:`LS_TILE_BYTES` (one whole-clip tile off
    CPU): the basis is built into the tile, and all five block sums plus the
    residual update happen tile-local. The loop over tracks stays — it is the
    sequential matching pursuit itself. The carrier recursion runs on the
    device too, so the only traffic across the seam is the clip, the envelopes
    and the ``R`` fundamental phases, and everything that would force a host
    sync inside the track loop (the "track is all zero" test, the clip
    counter) is either precomputed or accumulated on the device.

    Returns ``(x_new, resid, n_clipped, per-track |g|)``.
    """
    import torch

    dev = torch_device()
    n_ch, n_t = y.shape
    n_blocks = len(starts)
    n_pad = n_blocks * block
    n_tracks = int(env.x.shape[1])
    n_env = int(env.x.shape[-1])

    # Tiles: whole blocks, sized to keep one float64 working set in cache.
    per_block = max(1, n_ch * block * 8)
    tile_blocks = n_blocks
    if torch.device(dev).type == "cpu":
        tile_blocks = max(1, min(n_blocks, LS_TILE_BYTES // per_block))
    tiles = [
        (b0, min(b0 + tile_blocks, n_blocks), b0 * block, min((b0 + tile_blocks) * block, n_t))
        for b0 in range(0, n_blocks, tile_blocks)
    ]
    n_tile = tile_blocks * block

    # The pad tail of p/q stays zero for good: only the real samples are ever
    # written, so a short trailing block sums exactly the samples that exist.
    pbuf = torch.zeros((n_ch, n_tile), dtype=torch.float64, device=dev)
    qbuf = torch.zeros((n_ch, n_tile), dtype=torch.float64, device=dev)
    resid = torch.zeros((n_ch, n_pad), dtype=torch.float64, device=dev)
    resid[:, :n_t] = torch.from_numpy(np.ascontiguousarray(y)).to(dev)

    ramp = torch.arange(stride, device=dev, dtype=torch.float32) / float(stride)
    phase = torch.from_numpy(np.ascontiguousarray(env.phase[:, :n_t])).to(dev)
    fund = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
    xs = torch.from_numpy(np.ascontiguousarray(env.x)).to(dev)
    x_new = xs.clone()
    blk = torch.from_numpy(np.ascontiguousarray(blk_env)).to(dev)
    n_clip = torch.zeros((), dtype=torch.int64, device=dev)

    active = [m for m in range(n_tracks) if env.x[:, m].any()]
    order = sorted(active, key=lambda m: (int(env.rotor[m]), int(env.k[m])))
    cur_rotor, cur_k, carr = -1, 0, fund[0]
    mags: list[Any] = []
    for m in order:
        rot, kk = int(env.rotor[m]), int(env.k[m])
        if rot != cur_rotor:
            carr = fund[rot] if kk == 1 else fund[rot] ** kk
            cur_rotor, cur_k = rot, kk
        elif kk != cur_k:
            carr = carr * (fund[rot] ** (kk - cur_k))
            cur_k = kk

        knot = xs[:, m].to(torch.complex64)
        dknot = torch.zeros_like(knot)
        dknot[:, :-1] = knot[:, 1:] - knot[:, :-1]
        g_re = torch.empty((n_ch, n_blocks), dtype=torch.float64, device=dev)
        g_im = torch.empty_like(g_re)
        mag_all = torch.empty_like(g_re)

        for b0, b1, s0, s1 in tiles:
            nb, nc = b1 - b0, s1 - s0
            u = _ls_upsample(knot, dknot, ramp, stride, s0, nc, n_env) * carr[s0:s1]
            pbuf[:, :nc] = u.real.to(torch.float64)
            qbuf[:, :nc] = -u.imag.to(torch.float64)
            if nc < nb * block:  # short trailing tile: the pad tail must be 0
                pbuf[:, nc : nb * block] = 0.0
                qbuf[:, nc : nb * block] = 0.0
            pv = pbuf[:, : nb * block].view(n_ch, nb, block)
            qv = qbuf[:, : nb * block].view(n_ch, nb, block)
            rv = resid[:, b0 * block : b1 * block].view(n_ch, nb, block)

            app = (pv * pv).sum(-1)
            aqq = (qv * qv).sum(-1)
            apq = (pv * qv).sum(-1)
            bp = (rv * pv).sum(-1)
            bq = (rv * qv).sum(-1)
            det = app * aqq - apq * apq
            # Degenerate blocks (the track is invalid / silent there) keep
            # g = 1: the component is zero anyway, so the gain cannot matter.
            ok = det > _TINY * torch.clamp(app * aqq, min=_TINY)
            ones = torch.ones_like(det)
            safe = torch.where(ok, det, ones)
            a = torch.where(ok, (aqq * bp - apq * bq) / safe, ones)
            b = torch.where(ok, (app * bq - apq * bp) / safe, torch.zeros_like(det))
            mag = torch.hypot(a, b)
            clip = mag > gain_max
            n_clip += clip.sum()
            shrink = torch.where(clip, gain_max / torch.clamp(mag, min=_TINY), ones)
            a, b = a * shrink, b * shrink

            rv -= a.unsqueeze(-1) * pv + b.unsqueeze(-1) * qv
            g_re[:, b0:b1], g_im[:, b0:b1] = a, b
            mag_all[:, b0:b1] = torch.clamp(mag, max=gain_max)  # the gain applied

        x_new[:, m] = xs[:, m] * torch.complex(g_re, g_im)[:, blk]
        mags.append(mag_all)

    gains = [g.cpu().numpy() for g in mags]
    return (
        x_new.cpu().numpy().astype(np.complex128),
        resid[:, :n_t].cpu().numpy(),
        int(n_clip.item()),
        gains,
    )


def ls_project_envelopes(
    audio: np.ndarray,
    env: Envelopes,
    *,
    block_s: float = LS_BLOCK_S,
    gain_max: float = LS_GAIN_MAX,
) -> tuple[Envelopes, dict[str, Any]]:
    """Re-fit every envelope onto ``audio`` by least squares, per time block.

    Each track ``m`` contributes the waveform ``s_m = Re[x_m c_m]``. The VK
    solve fixes ``x_m`` from the *modelled* comb; when the trajectory is
    slightly off, ``s_m`` is mis-scaled and mis-phased, and subtracting it
    open-loop can add energy instead of removing it. This function replaces
    ``x_m`` by ``g x_m`` with the complex gain ``g`` that minimises
    ``||resid - Re[g x_m c_m]||^2`` over each block of ``block_s`` seconds and
    each channel — the exact 2-parameter (in-phase / quadrature) projection of
    the residual onto that one harmonic's 2-D subspace.

    Tracks are fitted **sequentially against a running residual** (rotor-major,
    harmonic ascending), not independently against the clip. That ordering is
    what makes the guarantee hold for the *sum*: each projection removes energy
    from the residual, so ``||y - sum_m Re[g_m x_m c_m]||^2 <= ||y||^2``, and no
    two overlapping harmonics can both claim the same energy. Fitting each
    harmonic against the clip instead lets overlapping harmonics double-count
    and over-subtract — measured on ``free-flight_nosource_room1`` (4 s, 8 mic,
    telemetry track), full-comb residual ratio: 0.691 open-loop / 0.709
    independent / **0.602** sequential on the cruise window w01, and 265.1 /
    — / **0.892** on the takeoff window w00, where the open-loop peel injects
    two orders of magnitude more energy than the clip holds.

    The core is :func:`_ls_project` — one implementation, cache-blocked on a
    CPU device and clip-long on any other.

    Returns ``(envelopes, diag)``; the input is not mutated.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))[: env.x.shape[0]]
    n_ch = y.shape[0]
    n_t = min(int(y.shape[-1]), int(env.phase.shape[-1]))
    n_env = env.x.shape[-1]
    if n_ch != env.x.shape[0]:
        raise ValueError(f"audio has {y.shape[0]} channels, envelopes have {env.x.shape[0]}")
    stride = int(round(float(env.t_env[1] - env.t_env[0]) * env.fs)) if n_env > 1 else 1
    block = max(stride, int(round(block_s * env.fs)))
    starts = np.arange(0, n_t, block)
    n_blocks = len(starts)
    # Sample -> block index. The gain is piecewise constant in time, so what is
    # subtracted inside a block is exactly what was fitted there.
    blk_env = np.minimum(np.arange(n_env) * stride // block, n_blocks - 1)

    x_new, resid, n_clipped, gains = _ls_project(
        y[:, :n_t],
        env,
        stride=stride,
        block=block,
        starts=starts,
        blk_env=blk_env,
        gain_max=gain_max,
    )

    all_mag = np.concatenate([g.ravel() for g in gains]) if gains else np.zeros(0)
    e_in = float(np.mean(y[:, :n_t] ** 2))
    diag: dict[str, Any] = {
        "block_s": block / env.fs,
        "block_samples": int(block),
        "n_blocks": int(n_blocks),
        "n_tracks_fitted": len(gains),
        "gain_abs_mean": round(float(np.mean(all_mag)), 4) if all_mag.size else None,
        "gain_abs_p95": round(float(np.percentile(all_mag, 95)), 4) if all_mag.size else None,
        "clipped_frac": round(n_clipped / max(all_mag.size, 1), 5),
        # The guarantee, measured: the full-comb residual over the clip energy.
        "e_resid_ratio": round(float(np.mean(resid**2)) / max(e_in, _TINY), 5),
    }
    return replace(env, x=x_new), diag


# ---------------------------------------------------------------------------
# outer frequency loop


def _demod_residual(resid: np.ndarray, env: Envelopes, cfg: VKConfig) -> np.ndarray:
    """Demodulate a residual into every track's band."""
    return _demod_tracks_fft(resid, env.phase, env.rotor, env.k, cfg)


def k_schedule(cfg: VKConfig) -> list[int]:
    """Per-round ``k_max``: geometric growth for capture, else fixed."""
    if cfg.k_schedule == "fixed" or cfg.n_outer <= 1 or cfg.k_max <= cfg.k_min:
        return [cfg.k_max] * cfg.n_outer
    if cfg.k_schedule != "grow":
        raise ValueError(f"unknown k_schedule {cfg.k_schedule!r}")
    k_start = max(cfg.k_min, min(8, cfg.k_max))
    ratio = cfg.k_max / k_start
    return [int(round(k_start * ratio ** (rd / (cfg.n_outer - 1)))) for rd in range(cfg.n_outer)]


def bw_schedule(cfg: VKConfig, fs_env: float) -> list[float]:
    """Per-round VK/demod bandwidth: wide → narrow alongside the ``k`` schedule.

    Capture requires the passband to admit the initial detuning ``k * delta0``
    — with a 1 Hz band a biased init leaves every envelope noise-dominated and
    the Fisher-weighted slopes stall. Annealed together with ``k_max``.
    """
    if cfg.k_schedule == "fixed" or cfg.n_outer <= 1:
        return [cfg.bw_hz] * cfg.n_outer
    bw_start = 0.2 * fs_env if cfg.bw_start_hz is None else cfg.bw_start_hz
    bw_start = max(bw_start, cfg.bw_hz)
    return list(np.geomspace(bw_start, cfg.bw_hz, cfg.n_outer))


def _break_symmetry(r_env: np.ndarray, cfg: VKConfig) -> None:
    """Nudge rotor pairs with everywhere-identical trajectories apart, in place.

    An exactly symmetric init (both rotors of a twin pair at the pair mean) is
    a saddle point of the coupled objective — identical phasors give identical
    envelopes and identical updates forever. A deterministic ±``split_nudge``
    (lower rotor index down) breaks the tie; the coupled solve's
    explaining-away then pulls each track to its own line.
    """
    n_rotors = r_env.shape[0]
    nudge = np.zeros(n_rotors)
    for i in range(n_rotors):
        for j in range(i + 1, n_rotors):
            if float(np.max(np.abs(r_env[i] - r_env[j]))) < cfg.split_eps:
                nudge[i] -= cfg.split_nudge
                nudge[j] += cfg.split_nudge
    r_env += nudge[:, None]


def _freq_update(
    env: Envelopes, rotor_idx: int, lam: float, cfg: VKConfig, z_res: np.ndarray
) -> np.ndarray | None:
    """Phase-slope frequency update for one rotor, fused over channels + harmonics.

    Per envelope sample the slope ``angle(x[t+1] conj(x[t])) * fs_env / (2 pi k)``
    estimates the trajectory error in rev/s; estimates are fused with Fisher
    weights ``k^2 |x[t+1] conj(x[t])|`` (the geometric mean of ``k^2 |x|^2`` at
    the two samples), shrunk per track by its envelope-vs-residual SNR
    (``z_res``: the demodulated full-reconstruction residual, so energy
    explained by *other* tracks does not count against a track), then a
    weighted smoothness solve ``(diag(W) + lam D2' D2) delta = W delta_hat``
    gives the per-sample correction. Returns ``None`` (no update) when no
    track of this rotor shows a spectral peak in its demodulated band
    (``cfg.update_gate`` on the periodogram peak/median ratio) — no comb,
    and a rotor without a comb must not drift (design §3 / test 5).
    """
    sel = np.where(env.rotor == rotor_idx)[0]
    if len(sel) == 0:
        return None
    x = env.x[:, sel]  # (C, m, T_env)
    v = env.valid[sel]  # (m, T_env)
    if not v.any():
        return None
    kf = env.k[sel].astype(np.float64)

    # No-comb gate: a comb line anywhere in a track's demod band — locked or
    # detuned — concentrates in the periodogram of z; white noise does not.
    window = np.hanning(env.z.shape[-1])
    peak = 0.0
    for m in sel:
        if not env.valid[m].any():
            continue
        seg = env.z[:, m] * (env.valid[m] * window)[None, :]
        pxx = np.abs(np.fft.fft(seg, axis=-1)) ** 2
        med = np.median(pxx, axis=-1)
        peak = max(peak, float(np.max(pxx.max(axis=-1) / np.maximum(med, _TINY))))
    if peak < cfg.update_gate:
        return None

    # Per-track envelope SNR: |x/2|^2 vs the residual energy scaled from the
    # demod band down to the track's own VK band (noise-equivalent widths).
    # A captured or locked line sits far above the ~O(1) noise floor; an
    # out-of-band (stalled) line falls below. Used as a soft per-track
    # weight only — the hard no-comb gate is the slope-coherence test below.
    v_f = v.astype(np.float64)[None, :, :]
    e_sig = np.sum(np.abs(0.5 * x) ** 2 * v_f, axis=(0, 2))
    e_res = np.sum(np.abs(z_res[:, sel]) ** 2 * v_f, axis=(0, 2))
    b_demod = 0.9 * env.fs_env  # two-sided noise-equiv width of the demod LP
    nu = (
        np.array([_vk_noise_bandwidth(float(env.bw_track[m]), env.fs_env, cfg.p) for m in sel])
        / b_demod
    )
    snr = e_sig / np.maximum(e_res * nu, _TINY)
    shrink = snr / (1.0 + snr)  # Wiener-style per-track weight

    prod = x[..., 1:] * np.conj(x[..., :-1])  # (C, m, T_env - 1)
    vv = (v[:, 1:] & v[:, :-1]).astype(np.float64)[None, :, :]
    if cfg.freq_weight == "k_beta":
        # WP18: the measured per-harmonic variance law is 1/v_k ∝ k^beta with
        # no amplitude factor — drop the per-sample |prod| term.
        w = (kf[None, :, None] ** cfg.freq_weight_beta) * vv * shrink[None, :, None]
    else:
        w = (kf[None, :, None] ** 2) * np.abs(prod) * vv * shrink[None, :, None]
    delta_hat = np.angle(prod) * env.fs_env / (2.0 * np.pi * kf[None, :, None])
    num = np.sum(w * delta_hat, axis=(0, 1))
    den = np.sum(w, axis=(0, 1))  # (T_env - 1,)
    if not (den > 0).any():
        return None

    scale = float(den[den > 0].mean())
    w_norm = den / scale
    fused = num / np.maximum(den, _TINY * scale)
    d2 = second_diff(len(fused))
    d2td2 = (d2.T @ d2).tocsr()
    # Small anchor keeps delta -> 0 where no track carries evidence. The
    # system is real SPD pentadiagonal — solved directly in banded storage.
    ab = np.zeros((3, len(fused)))
    ab[2] = w_norm + 1e-3 + lam * np.asarray(d2td2.diagonal(0))
    ab[1, 1:] = lam * np.asarray(d2td2.diagonal(1))
    ab[0, 2:] = lam * np.asarray(d2td2.diagonal(2))
    delta_mid = cast(np.ndarray, solveh_banded(ab, w_norm * fused, lower=False))
    t_mid = env.t_env[:-1] + 0.5 / env.fs_env
    return np.interp(env.t_env, t_mid, delta_mid)


def _confidence(env: Envelopes, cfg: VKConfig, z_res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Windowed per-rotor confidence: track envelope energy vs in-band residual.

    ``conf = sum |x/2|^2 / sum |z_res|^2`` over channels, harmonics and the
    window (valid samples only), with ``z_res`` the demodulated residual of
    the *full* reconstruction (design §3). A locked comb gives ``conf >> 1``;
    pure noise gives ``conf ≈ 0`` — no comb, do not trust the track there.
    """
    n_rotors = int(env.rotor.max()) + 1 if len(env.rotor) else 0
    win = max(1, int(round(cfg.conf_window_s * env.fs_env)))
    n_env = env.x.shape[-1]
    n_win = max(1, n_env // win)
    conf = np.zeros((n_rotors, n_win))
    centers = np.zeros(n_win)
    for wi in range(n_win):
        sl = slice(wi * win, (wi + 1) * win if wi < n_win - 1 else n_env)
        centers[wi] = float(env.t_env[sl].mean())
        for i in range(n_rotors):
            sel = env.rotor == i
            v = env.valid[sel][:, sl].astype(np.float64)[None]
            num = float(np.sum(np.abs(0.5 * env.x[:, sel][..., sl]) ** 2 * v))
            den = float(np.sum(np.abs(z_res[:, sel][..., sl]) ** 2 * v))
            conf[i, wi] = num / max(den, _TINY)
    return conf, centers


def _bw_adapt_factors(env: Envelopes) -> np.ndarray:
    """IAVKF orthogonal factor per track, in the demodulated domain.

    ``factor_m = |<z_m, a_m>| / <a_m, a_m>`` with ``a_m = x_m / 2`` (the
    solved envelope on the observation scale — ``z ≈ a / 2`` at lock; cf.
    Jiang, Chen & Wang 2024 eqs 25–27, where the reconstructed component is
    correlated against the analytic signal), inner products taken over
    channels and the track's *valid* envelope samples of this round. The
    factor is > 1 while the solve still over-smooths the in-band observation
    (the smoother's eigenvalues are in [0, 1], so ``<z, Sz> >= <Sz, Sz>``)
    and → 1 at lock. NaN marks skipped tracks: no valid samples, degenerate
    ``<a, a> ≈ 0``, or non-finite inner products.
    """
    n_tracks = env.x.shape[1]
    fac = np.full(n_tracks, np.nan)
    for m in range(n_tracks):
        v = env.valid[m]
        if not v.any():
            continue
        a = 0.5 * env.x[:, m][:, v]  # (C, n_valid) envelope on z's scale
        z = env.z[:, m][:, v]
        den = float(np.sum(np.abs(a) ** 2))
        if not np.isfinite(den) or den <= _TINY:
            continue
        num = float(np.abs(np.vdot(a, z)))  # |<z, a>|, summed over channels
        if not np.isfinite(num):
            continue
        fac[m] = num / den
    return fac


def vk_track(
    audio: np.ndarray,
    r_init: np.ndarray,
    frame_times: np.ndarray,
    cfg: VKConfig,
) -> VKResult:
    """Coupled VK order tracking: alternate envelope solves and frequency updates.

    ``audio``: ``(T,)`` or ``(C, T)`` at ``cfg.fs``; ``r_init``: ``(R, N)``
    rev/s on the ``frame_times`` grid (any grid — telemetry, STFT frames, or a
    predictor's output). Runs ``cfg.n_outer`` rounds of (1) coupled envelope
    solve given the current trajectories, (2) Fisher-weighted phase-slope
    frequency update (clipped to ``cfg.max_step`` per round), with the
    ``k_max`` / bandwidth annealing schedule for basin capture. With
    ``cfg.bw_adapt`` each round additionally rescales every track's squared
    selectivity by its IAVKF orthogonal factor (:func:`_bw_adapt_factors`),
    accumulated in a persistent per-(rotor, harmonic) gain (clamped to
    ``[clamp^-2, clamp^2]``) that composes multiplicatively with the annealed
    per-round bandwidth. The result carries the refined trajectories on both
    the input grid and the dense envelope grid, the final envelopes,
    per-round residual ratios ``||y - y_hat||^2 / ||y||^2``, the windowed
    confidence, and the coupling-group log (plus, under ``bw_adapt``, the
    per-round factors and final gain in ``extras``).
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))[:_MAX_CHANNELS]
    r_init = np.atleast_2d(np.asarray(r_init, dtype=np.float64))
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if r_init.shape[-1] != len(frame_times):
        raise ValueError(f"r_init has {r_init.shape[-1]} frames, frame_times {len(frame_times)}")
    n_t = y.shape[-1]
    t_aud = np.arange(n_t, dtype=np.float64) / cfg.fs
    stride, fs_env = env_stride(cfg)
    t_env = np.arange(0, n_t, stride, dtype=np.float64) / cfg.fs
    r_env = np.stack([np.interp(t_env, frame_times, r_init[i]) for i in range(r_init.shape[0])])

    ks = k_schedule(cfg)
    bws = bw_schedule(cfg, fs_env)
    lams = (
        list(np.geomspace(10.0 * cfg.traj_lambda, cfg.traj_lambda, cfg.n_outer))
        if cfg.n_outer > 1
        else [cfg.traj_lambda]
    )

    y_energy = float(np.sum(y**2))
    residual_ratios: list[float] = []
    max_deltas: list[float] = []
    groups_log: list[list[list[tuple[int, int]]]] = []
    # IAVKF bandwidth-adaptation state: persistent per-(rotor, harmonic)
    # multiplicative gain on rho^2, applied on top of the annealed bandwidth.
    n_k_full = cfg.k_max - cfg.k_min + 1
    bw_gain = np.ones((r_env.shape[0], n_k_full)) if cfg.bw_adapt else None
    gain_lo, gain_hi = cfg.bw_adapt_clamp**-2, cfg.bw_adapt_clamp**2
    adapt_factors: list[np.ndarray] = []
    env: Envelopes | None = None
    for rd in range(cfg.n_outer):
        _break_symmetry(r_env, cfg)
        r_aud = np.stack([np.interp(t_aud, t_env, r_env[i]) for i in range(r_env.shape[0])])
        gain_vec = None
        if bw_gain is not None:
            rot_rd, k_rd = _track_table(r_env.shape[0], cfg.k_min, min(int(ks[rd]), cfg.k_max))
            gain_vec = bw_gain[rot_rd, k_rd - cfg.k_min]
        env = vk_envelopes(y, r_aud, cfg, k_hi=ks[rd], bw_hz=bws[rd], rho2_gain=gain_vec)
        if bw_gain is not None:
            fac = _bw_adapt_factors(env)
            ok = np.isfinite(fac) & (fac > 0)
            idx_r, idx_k = env.rotor[ok], env.k[ok] - cfg.k_min
            bw_gain[idx_r, idx_k] = np.clip(bw_gain[idx_r, idx_k] * fac[ok], gain_lo, gain_hi)
            full = np.full((r_env.shape[0], n_k_full), np.nan)
            full[env.rotor, env.k - cfg.k_min] = fac
            adapt_factors.append(full)
        recon = vk_reconstruct(env, n_samples=n_t)
        residual_ratios.append(float(np.sum((y - recon) ** 2)) / max(y_energy, _TINY))
        # Residual demodulated into each track's band: the reference against
        # which envelope SNR (gate, weights, confidence) is measured.
        z_res = _demod_residual(y - recon, env, cfg)
        groups_log.append(
            [
                [(int(env.rotor[m]), int(env.k[m])) for m in grp]
                for grp in env.groups
                if len(grp) > 1
            ]
        )
        max_d = 0.0
        for i in range(r_env.shape[0]):
            delta = _freq_update(env, i, float(lams[rd]), cfg, z_res)
            if delta is None:
                continue
            delta = np.clip(delta, -cfg.max_step, cfg.max_step)
            r_env[i] = np.maximum(r_env[i] + delta, 0.0)
            max_d = max(max_d, float(np.max(np.abs(delta))))
        max_deltas.append(max_d)

    assert env is not None  # n_outer >= 1
    # Confidence against the residual of the *final* trajectories.
    recon = vk_reconstruct(env, n_samples=n_t)
    z_res = _demod_residual(y - recon, env, cfg)
    conf, conf_times = _confidence(env, cfg, z_res)
    r_refined = np.stack([np.interp(frame_times, t_env, r_env[i]) for i in range(r_env.shape[0])])
    return VKResult(
        r_refined=r_refined,
        frame_times=frame_times,
        r_env=r_env,
        t_env=t_env,
        envelopes=env,
        residual_ratios=residual_ratios,
        max_deltas=max_deltas,
        confidence=conf,
        conf_times=conf_times,
        groups_log=groups_log,
        extras={
            "k_schedule": ks,
            "bw_schedule": bws,
            "lambda_schedule": lams,
            **(
                {"bw_adapt_factors": adapt_factors, "bw_gain": bw_gain}
                if bw_gain is not None
                else {}
            ),
        },
    )
