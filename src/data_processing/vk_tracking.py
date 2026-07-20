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
   (stage D of :mod:`data_processing.rps_refinement`, Fisher weights
   ``k^2 |x|^2``), fused across channels and harmonics by a
   smoothness-regularised weighted 1-D solve, with an annealing schedule on
   ``k_max`` (and the demod/VK bandwidth) for basin capture.

CPU-inference fast paths (2026-07-20, see the vk_bench profiling study): the
coupled-group normal equations are solved as a time-major-interleaved
Hermitian **banded** Cholesky system (``VKConfig.solver``, splu kept as the
reference/fallback), cross-coupling terms are only demodulated for pairs that
can actually beat inside the lowpass band (``VKConfig.prune_far_pairs``), and
the demod lowpass is selectable between the batched FFT brickwall and a
two-stage linear-phase FIR polyphase decimator (``VKConfig.lp_mode``).

Conventions match ``rps_refinement.py``: trajectories in rev/s with harmonics
at ``k * r`` Hz, arrays time-last, mono-or-multichannel audio plus a
trajectory sampled on an arbitrary time grid. The core is numpy + scipy only
(float64 solver, no torch) — it is an offline annotation tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded, solveh_banded
from scipy.signal import butter, firwin, kaiserord, resample_poly, sosfiltfilt
from scipy.sparse.linalg import splu

__all__ = [
    "VKConfig",
    "Envelopes",
    "VKResult",
    "demodulate",
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
    lp_mode: str = "fft"  # anti-alias lowpass for demod/decimate: "fft" =
    # batched FFT brickwall + zoom-IFFT (fastest measured — pocketfft's SIMD
    # beats scipy's polyphase kernel; complex64 FFT stage); "fir" = two-stage
    # linear-phase FIR polyphase decimation (see _fir_stages; exactly
    # delay-compensated, kept behind this flag for A/B); "iir" = the original
    # per-track zero-phase butter-4 sosfiltfilt (reference implementation).
    # fft/fir agree to ~5e-4 relative L2 on in-band content; both agree with
    # iir to ~1% relative L2 away from the segment edges (edge handling and
    # filter shape differ only near the boundaries / transition band).
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

    def __post_init__(self) -> None:
        if self.lp_mode not in ("fft", "fir", "iir"):
            raise ValueError(f"unknown lp_mode {self.lp_mode!r} (expected 'fft', 'fir' or 'iir')")
        if self.solver not in ("banded", "splu"):
            raise ValueError(f"unknown solver {self.solver!r} (expected 'banded' or 'splu')")


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


def _second_diff(n: int) -> sparse.csr_array:
    """``(n-2, n)`` second-difference operator ``[1, -2, 1]``."""
    if n < 3:
        raise ValueError(f"need at least 3 samples for a second difference, got {n}")
    d2 = (
        sparse.eye_array(n - 2, n, k=0)
        - 2.0 * sparse.eye_array(n - 2, n, k=1)
        + sparse.eye_array(n - 2, n, k=2)
    )
    return d2.tocsr()


def _stride(cfg: VKConfig) -> tuple[int, float]:
    """Decimation stride and the *actual* envelope rate ``fs / stride``."""
    stride = max(1, int(round(cfg.fs / cfg.fs_env)))
    return stride, cfg.fs / stride


def _lowpass_sos(cfg: VKConfig, fs_env: float) -> np.ndarray:
    """Zero-phase anti-alias lowpass for the demodulated envelopes.

    Butterworth order 4, cutoff ``0.45 * fs_env`` (design §2.1), realised as
    second-order sections: at cutoffs this far below the audio Nyquist the
    ``(b, a)`` form of a 4th-order butter is numerically fragile; sos is not.
    """
    return cast(np.ndarray, butter(4, 0.45 * fs_env, fs=cfg.fs, output="sos"))


def _lp_decimate(x: np.ndarray, sos: np.ndarray, stride: int) -> np.ndarray:
    """Zero-phase lowpass then decimate by ``stride`` along the last axis.

    ``padlen`` is stretched to ~3 cutoff periods (the filter's settling time):
    sosfiltfilt's default of a few samples is hopelessly short for a cutoff
    this far below the audio rate and leaves a large startup transient.
    """
    padlen = min(x.shape[-1] - 1, 8 * stride)  # cutoff 0.45/stride of fs
    return sosfiltfilt(sos, x, axis=-1, padlen=padlen)[..., ::stride]


def _fft_workers() -> int:
    """FFT worker threads: follow ``OMP_NUM_THREADS`` (the same knob callers
    already use to cap BLAS), clamped to the CPUs actually available to this
    process (cgroup/affinity — oversubscribing pocketfft workers on a
    restricted Slurm allocation thrashes), defaulting to 1."""
    try:
        avail = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        avail = os.cpu_count() or 1
    try:
        return max(1, min(int(os.environ.get("OMP_NUM_THREADS", "1")), avail))
    except ValueError:
        return 1


def _fft_lp_decimate(x: np.ndarray, stride: int, n_env: int) -> np.ndarray:
    """FFT brickwall lowpass + decimate: keep ``|f| <= 0.45 * fs_env``.

    Zoom-IFFT: zero-pad ``x`` to ``stride * n_env``, complex-FFT along the
    last axis, retain only the low band (positive *and* negative bins — the
    input is complex, the spectrum is not conjugate-symmetric), and inverse
    FFT at length ``n_env`` directly. Because the retained band fits inside
    the decimated Nyquist range, ``ifft(retained) / stride`` *is* the
    brickwall-filtered signal sampled at every ``stride``-th point — no
    audio-rate inverse transform needed. Equivalent to ``_lp_decimate`` up
    to the filter shape (brickwall vs butter-4) and edge handling (circular
    zero-pad vs reflect), i.e. everywhere except near the segment edges.

    The FFT runs in complex64 (envelope-precision is ~1e-7 relative, far
    below the VK solve's noise floor; ``scipy.fft`` — unlike ``np.fft`` —
    computes single-precision natively); phase/cumsum precision is the
    *caller's* concern and must stay float64 upstream. Batched transforms
    use ``OMP_NUM_THREADS`` workers (the process's declared CPU budget).
    """
    from scipy import fft as sfft

    w = _fft_workers()
    n_pad = stride * n_env
    xc = np.asarray(x, dtype=np.complex64)
    spec = cast(np.ndarray, sfft.fft(xc, n=n_pad, axis=-1, workers=w))
    if n_env < 8:  # degenerate grids: exact-but-tiny full inverse transform
        f = cast(np.ndarray, sfft.fftfreq(n_pad, d=1.0))  # cycles/sample at audio rate
        spec[..., np.abs(f) > 0.45 / stride] = 0.0
        full = cast(np.ndarray, sfft.ifft(spec, axis=-1, workers=w))
        return full[..., ::stride].astype(np.complex128)
    b = min(int(np.floor(0.45 * n_env)), (n_env - 1) // 2)  # bins per side
    low = np.zeros(x.shape[:-1] + (n_env,), dtype=np.complex64)
    low[..., : b + 1] = spec[..., : b + 1]
    low[..., n_env - b :] = spec[..., n_pad - b :]
    dec = cast(np.ndarray, sfft.ifft(low, axis=-1, workers=w))
    return (dec / np.complex64(stride)).astype(np.complex128)


@lru_cache(maxsize=8)
def _fir_stages(stride: int) -> tuple[tuple[int, np.ndarray], ...]:
    """Polyphase FIR decimator cascade for ``lp_mode="fir"`` (cached per stride).

    Band spec, expressed at the final envelope rate: passband edge
    ``0.45 * fs_env`` (the brickwall cutoff of :func:`_fft_lp_decimate`),
    stopband edge ``0.55 * fs_env`` — everything that could fold onto the
    retained ``|f| <= 0.45 * fs_env`` band is attenuated. Kaiser design at
    70 dB: aliases/leakage <= 3e-4 and passband ripple <= ~1e-3 — an order
    of magnitude below the tracker's 1e-3 rev/s regression tolerance. A
    single sharp filter at audio rate would need ~0.1*fs_env-wide transition
    (thousands of taps), so for composite strides the decimation is
    two-stage: a cheap wide-transition stage down to ``s2 * fs_env``
    (``s2`` = smallest divisor of ``stride`` >= 4, so the intermediate rate
    keeps the transition >= ~3 * fs_env wide), then the sharp stage at the
    low rate — ~2M tap-multiplies per 20 s signal instead of ~14M.

    All taps are odd-length symmetric (type-I linear phase): the group delay
    is an integer number of input samples and ``resample_poly`` compensates
    it exactly, so decimated sample ``n`` corresponds to audio sample
    ``n * stride`` with zero phase shift — the phase-slope frequency update
    stays unbiased (validated: < 1e-6 rad phase error on in-band tones).
    """

    def kaiser_lp(pass_c: float, stop_c: float) -> np.ndarray:
        # cutoffs in cycles/sample of the stage input; kaiserord takes the
        # transition width relative to Nyquist (0.5 cycles/sample)
        numtaps, beta = kaiserord(70.0, 2.0 * (stop_c - pass_c))
        window: Any = ("kaiser", beta)  # firwin stub over-narrows window to str
        h = firwin(int(numtaps) | 1, pass_c + stop_c, window=window, fs=2.0)
        return np.asarray(h, dtype=np.float32)

    if stride <= 1:
        return ()
    s2 = next((d for d in range(4, stride) if stride % d == 0), None)
    if s2 is None:  # small or prime stride: one sharp stage
        return ((stride, kaiser_lp(0.45 / stride, 0.55 / stride)),)
    s1 = stride // s2
    return (
        # stage 1 stopband: protect |f| <= 0.55*fs_env from folds around fs/s1
        (s1, kaiser_lp(0.45 / stride, 1.0 / s1 - 0.55 / stride)),
        (s2, kaiser_lp(0.45 / s2, 0.55 / s2)),
    )


def _fir_lp_decimate(x: np.ndarray, stride: int, n_env: int) -> np.ndarray:
    """FIR-polyphase counterpart of :func:`_fft_lp_decimate` (``lp_mode="fir"``).

    Same passband (``0.45 * fs_env``) and alignment (output sample ``n`` <->
    input sample ``n * stride``, exactly delay-compensated); differs in the
    Kaiser transition band 0.45–0.55 ``fs_env`` (vs brickwall) and in edge
    handling (zero pad vs circular). Real and imaginary parts are filtered
    as two stacked float32 rows: ``upfirdn`` promotes the taps to the signal
    dtype, so feeding complex input directly would double the multiplies.
    """
    stages = _fir_stages(stride)
    if not stages:  # stride 1: nothing to decimate, apply the brickwall only
        return _fft_lp_decimate(x, stride, n_env)
    y = np.empty((2, *x.shape), dtype=np.float32)
    y[0] = x.real
    y[1] = x.imag
    for q, h in stages:
        y = np.asarray(resample_poly(y, 1, q, axis=-1, window=h, padtype="constant"))
    y = y[..., :n_env]
    return (y[0] + 1j * y[1]).astype(np.complex128)


def _lp_decimate_fast(x: np.ndarray, stride: int, n_env: int, lp_mode: str) -> np.ndarray:
    """Dispatch the batched lowpass+decimate: FFT brickwall or FIR polyphase."""
    if lp_mode == "fir":
        return _fir_lp_decimate(x, stride, n_env)
    return _fft_lp_decimate(x, stride, n_env)


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
    rho: float,
    w: list[np.ndarray],
    cross: dict[tuple[int, int], np.ndarray],
    z_g: np.ndarray,
) -> np.ndarray:
    """Reference coupled-group solve: track-major sparse LU (SuperLU).

    ``z_g``: (C, g, T_env) demodulated observations of the group's tracks;
    returns ``(g, T_env, C)`` envelopes. Verbatim the original formulation —
    kept as the A/B reference (``cfg.solver == "splu"``) and as the fallback
    when the banded Cholesky reports a numerically non-PD system.
    """
    g = len(w)
    n_ch, _, n_env = z_g.shape
    reg = (rho**2) * d2td2 + 1e-8 * eye  # eps keeps masked spans well-posed
    blocks: list[list[Any]] = [[None] * g for _ in range(g)]
    for a in range(g):
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
    rho: float,
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
    diag = (rho**2) * d0[:, None] + 1e-8 + np.stack(w, axis=-1)  # (T_env, g)
    ab[u] = diag.reshape(-1)
    ab[u - g, g:] = np.repeat((rho**2) * d1, g)
    ab[0, 2 * g :] = np.repeat((rho**2) * d2, g)
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
    ``LP[y * exp(-j phase_m)]`` low-passed (per ``cfg.lp_mode``) and
    decimated by the stride to the envelope grid.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    phase = np.atleast_2d(np.asarray(phase, dtype=np.float64))
    if phase.shape[-1] != y.shape[-1]:
        raise ValueError(f"phase length {phase.shape[-1]} != audio length {y.shape[-1]}")
    stride, fs_env = _stride(cfg)
    n_ch, n_t = y.shape
    n_env = len(range(0, n_t, stride))
    n_tracks = phase.shape[0]
    z = np.empty((n_ch, n_tracks, n_env), dtype=np.complex128)
    if cfg.lp_mode in ("fft", "fir"):
        # Batched fast path: chunk tracks to bound the (C, m, T) complex64
        # working set at ~128 MB. Phases are float64 (they reach ~1e7 rad —
        # float32 drifts by radians); only the *demodulated* product drops to
        # complex64 for the filtering stage.
        y32 = y.astype(np.float32)
        chunk = max(1, int(128e6 / (max(1, n_ch) * max(1, n_t) * 8)))
        for lo in range(0, n_tracks, chunk):
            hi = min(lo + chunk, n_tracks)
            phasor = np.exp(-1j * phase[lo:hi]).astype(np.complex64)  # (m, T)
            z[:, lo:hi] = _lp_decimate_fast(
                y32[:, None, :] * phasor[None], stride, n_env, cfg.lp_mode
            )
        return z
    sos = _lowpass_sos(cfg, fs_env)
    for m in range(n_tracks):
        z[:, m] = _lp_decimate(y * np.exp(-1j * phase[m])[None, :], sos, stride)
    return z


def _demod_tracks_fft(
    audio: np.ndarray, phase: np.ndarray, rotor: np.ndarray, k: np.ndarray, cfg: VKConfig
) -> np.ndarray:
    """Structured fast path of :func:`demodulate` for rotor-harmonic tracks.

    Same result as ``demodulate(audio, k[:, None] * phase[rotor], cfg)`` with
    ``lp_mode="fft"`` / ``"fir"``, but without materialising the ``(M, T)``
    phase matrix or taking an exp per track: per-track conj-phasors come from
    the :func:`_track_carriers` recursion and the demodulated products are
    lowpass-decimated in memory-bounded batches.
    """
    y32 = np.asarray(np.atleast_2d(audio), dtype=np.float32)
    stride, _ = _stride(cfg)
    n_ch, n_t = y32.shape
    n_env = len(range(0, n_t, stride))
    n_tracks = len(rotor)
    z = np.empty((n_ch, n_tracks, n_env), dtype=np.complex128)
    if n_tracks == 0:
        return z
    chunk = max(1, int(128e6 / (max(1, n_ch) * max(1, n_t) * 8)))
    buf = np.empty((n_ch, min(chunk, n_tracks), n_t), dtype=np.complex64)
    idxs: list[int] = []
    for m, carr in _track_carriers(phase, rotor, k, range(n_tracks), sign=-1.0):
        np.multiply(y32, carr, out=buf[:, len(idxs)])
        idxs.append(m)
        if len(idxs) == buf.shape[1]:
            z[:, idxs] = _lp_decimate_fast(buf, stride, n_env, cfg.lp_mode)
            idxs = []
    if idxs:
        z[:, idxs] = _lp_decimate_fast(buf[:, : len(idxs)], stride, n_env, cfg.lp_mode)
    return z


def vk_envelopes(
    audio: np.ndarray,
    r: np.ndarray,
    cfg: VKConfig,
    *,
    k_hi: int | None = None,
    bw_hz: float | None = None,
) -> Envelopes:
    """One coupled VK-2 envelope solve (all coupling groups) given trajectories.

    ``audio``: ``(T,)`` or ``(C, T)``; ``r``: ``(R, T)`` rev/s at *audio* rate.
    ``k_hi`` / ``bw_hz`` override ``cfg.k_max`` / ``cfg.bw_hz`` (used by the
    outer loop's annealing schedule). Solves, per coupling group and channel,

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
    stride, fs_env = _stride(cfg)
    t_env = np.arange(0, n_t, stride, dtype=np.float64) / cfg.fs
    n_env = len(t_env)
    k_top = cfg.k_max if k_hi is None else min(int(k_hi), cfg.k_max)
    bw = cfg.bw_hz if bw_hz is None else float(bw_hz)
    couple_hz = fs_env / 2.0 if cfg.couple_hz is None else cfg.couple_hz

    rotor, k = _track_table(r.shape[0], cfg.k_min, k_top)
    phase = 2.0 * np.pi * np.cumsum(r, axis=-1) / cfg.fs  # (R, T) fundamental phase

    # Track validity mask on the envelope grid (design §3).
    r_dec = r[:, ::stride]
    f = k[:, None].astype(np.float64) * r_dec[rotor]  # (M, T_env)
    f_hi = min(cfg.f_max, 0.45 * cfg.fs)
    valid = (f >= cfg.f_min) & (f <= f_hi) & (r_dec[rotor] >= cfg.min_rps)

    if cfg.lp_mode in ("fft", "fir"):
        z = _demod_tracks_fft(y, phase, rotor, k, cfg)
    else:
        z = demodulate(y, k[:, None] * phase[rotor], cfg)
    groups = _coupling_groups(f, valid, couple_hz)

    d2 = _second_diff(n_env)
    d2td2 = (d2.T @ d2).tocsr()  # banded, bandwidth 2p+1
    d2td2_diags = (
        np.asarray(d2td2.diagonal(0)),
        np.asarray(d2td2.diagonal(1)),
        np.asarray(d2td2.diagonal(2)),
    )
    eye = sparse.eye_array(n_env)
    sos = _lowpass_sos(cfg, fs_env) if cfg.lp_mode == "iir" else None

    x = np.zeros_like(z)
    bw_track = np.full(len(rotor), bw)
    # Edge taper on the data term: the zero-phase LP's first/last few envelope
    # samples are padding-guess transients; fitting them flares the envelopes
    # at the boundaries. Fade the data weight over the filter settling span
    # (~3 cutoff periods = 3 / 0.45 envelope samples) and let the smoothness
    # prior extrapolate instead.
    n_edge = min(8, max(1, n_env // 4))
    taper = np.ones(n_env)
    ramp = (np.arange(n_edge) + 1.0) / (n_edge + 1.0)
    taper[:n_edge] = ramp
    taper[-n_edge:] = ramp[::-1]
    # Coupling pairs separated by at least this never beat inside the demod
    # lowpass (cutoff 0.45 * fs_env, FIR stopband edge 0.55 * fs_env): their
    # LP'd cross term is pure spectral leakage, ~1e-3 relative.
    prune_hz = 1.25 * 0.45 * fs_env
    for group in groups:
        g = len(group)
        w = [valid[m].astype(np.float64) * taper for m in group]
        # Per-group bandwidth clamp (see VKConfig.sep_bw_factor) + pair
        # separations, reused for the far-pair pruning below. Pairs with no
        # co-valid samples contribute exactly nothing (w_a * w_b == 0
        # everywhere) and are always skipped.
        bw_g = bw
        pair_sep: dict[tuple[int, int], float] = {}
        for a in range(g):
            for b in range(a + 1, g):
                m, n = group[a], group[b]
                both = valid[m] & valid[n]
                if both.any():
                    sep = float(np.min(np.abs(f[m, both] - f[n, both])))
                    pair_sep[(a, b)] = sep
                    bw_g = min(bw_g, max(cfg.bw_hz, cfg.sep_bw_factor * sep))
        bw_track[group] = bw_g
        rho = _tuma_rho(bw_g, fs_env, cfg.p)
        pair_list = [
            pair for pair, sep in pair_sep.items() if not (cfg.prune_far_pairs and sep >= prune_hz)
        ]
        # LP-decimated cross terms conj(c_m) c_n on the envelope grid. Fast
        # modes: per-track carriers via the shared recursion (one exp per
        # rotor), cross phasors as complex64 products, decimated in
        # memory-bounded batches — the per-pair exp + audio-rate filter of
        # the reference path dominated the whole solve for coupled groups.
        cross: dict[tuple[int, int], np.ndarray] = {}
        if pair_list and sos is None:  # lp_mode "fft" / "fir"
            carr = dict(_track_carriers(phase, rotor, k, group, sign=1.0))
            chunk_p = max(1, int(128e6 / (max(1, n_t) * 8)))
            cs = np.empty((min(chunk_p, len(pair_list)), n_t), dtype=np.complex64)
            for lo in range(0, len(pair_list), chunk_p):
                sub = pair_list[lo : lo + chunk_p]
                for i, (a, b) in enumerate(sub):
                    np.multiply(carr[group[b]], np.conj(carr[group[a]]), out=cs[i])
                gd = _lp_decimate_fast(cs[: len(sub)], stride, n_env, cfg.lp_mode)
                for i, pair in enumerate(sub):
                    cross[pair] = gd[i]
        else:
            for a, b in pair_list:
                m, n = group[a], group[b]
                dphi = k[n] * phase[rotor[n]] - k[m] * phase[rotor[m]]
                cross[(a, b)] = _lp_decimate(np.exp(1j * dphi), cast(np.ndarray, sos), stride)
        z_g = z[:, group]  # (C, g, T_env)
        sol: np.ndarray | None = None
        if cfg.solver == "banded":
            try:
                sol = _solve_group_banded(d2td2_diags, rho, w, cross, z_g)
            except np.linalg.LinAlgError:
                sol = None  # numerically non-PD: use the reference path below
        if sol is None:
            sol = _solve_group_splu(d2td2, eye, rho, w, cross, z_g)
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
# outer frequency loop


def _demod_residual(resid: np.ndarray, env: Envelopes, cfg: VKConfig) -> np.ndarray:
    """Demodulate a residual into every track's band (mode-dispatched)."""
    if cfg.lp_mode in ("fft", "fir"):
        return _demod_tracks_fft(resid, env.phase, env.rotor, env.k, cfg)
    return demodulate(resid, env.k[:, None] * env.phase[env.rotor], cfg)


def _k_schedule(cfg: VKConfig) -> list[int]:
    """Per-round ``k_max``: geometric growth for capture, else fixed."""
    if cfg.k_schedule == "fixed" or cfg.n_outer <= 1 or cfg.k_max <= cfg.k_min:
        return [cfg.k_max] * cfg.n_outer
    if cfg.k_schedule != "grow":
        raise ValueError(f"unknown k_schedule {cfg.k_schedule!r}")
    k_start = max(cfg.k_min, min(8, cfg.k_max))
    ratio = cfg.k_max / k_start
    return [int(round(k_start * ratio ** (rd / (cfg.n_outer - 1)))) for rd in range(cfg.n_outer)]


def _bw_schedule(cfg: VKConfig, fs_env: float) -> list[float]:
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
    w = (kf[None, :, None] ** 2) * np.abs(prod) * vv * shrink[None, :, None]
    delta_hat = np.angle(prod) * env.fs_env / (2.0 * np.pi * kf[None, :, None])
    num = np.sum(w * delta_hat, axis=(0, 1))
    den = np.sum(w, axis=(0, 1))  # (T_env - 1,)
    if not (den > 0).any():
        return None

    scale = float(den[den > 0].mean())
    w_norm = den / scale
    fused = num / np.maximum(den, _TINY * scale)
    d2 = _second_diff(len(fused))
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
    ``k_max`` / bandwidth annealing schedule for basin capture. The result
    carries the refined trajectories on both the input grid and the dense
    envelope grid, the final envelopes, per-round residual ratios
    ``||y - y_hat||^2 / ||y||^2``, the windowed confidence, and the
    coupling-group log.
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))[:_MAX_CHANNELS]
    r_init = np.atleast_2d(np.asarray(r_init, dtype=np.float64))
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if r_init.shape[-1] != len(frame_times):
        raise ValueError(f"r_init has {r_init.shape[-1]} frames, frame_times {len(frame_times)}")
    n_t = y.shape[-1]
    t_aud = np.arange(n_t, dtype=np.float64) / cfg.fs
    stride, fs_env = _stride(cfg)
    t_env = np.arange(0, n_t, stride, dtype=np.float64) / cfg.fs
    r_env = np.stack([np.interp(t_env, frame_times, r_init[i]) for i in range(r_init.shape[0])])

    ks = _k_schedule(cfg)
    bws = _bw_schedule(cfg, fs_env)
    lams = (
        list(np.geomspace(10.0 * cfg.traj_lambda, cfg.traj_lambda, cfg.n_outer))
        if cfg.n_outer > 1
        else [cfg.traj_lambda]
    )

    y_energy = float(np.sum(y**2))
    residual_ratios: list[float] = []
    max_deltas: list[float] = []
    groups_log: list[list[list[tuple[int, int]]]] = []
    env: Envelopes | None = None
    for rd in range(cfg.n_outer):
        _break_symmetry(r_env, cfg)
        r_aud = np.stack([np.interp(t_aud, t_env, r_env[i]) for i in range(r_env.shape[0])])
        env = vk_envelopes(y, r_aud, cfg, k_hi=ks[rd], bw_hz=bws[rd])
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
        extras={"k_schedule": ks, "bw_schedule": bws, "lambda_schedule": lams},
    )
