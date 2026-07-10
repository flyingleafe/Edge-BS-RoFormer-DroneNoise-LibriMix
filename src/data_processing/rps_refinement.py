"""RPS trajectory refinement by comb-spectral alignment.

Rotor-speed (RPS) labels from telemetry carry errors — clock offset against
the audio, command-vs-actual discrepancy, low sampling rate (Michael's 29 Hz),
interpolation — and a label error ``delta`` displaces the k-th rotor harmonic
by ``k * delta`` Hz: negligible at the fundamental, several STFT bins at
mid frequencies. This module *refines* trajectories against the audio itself,
initialised close to the truth (telemetry, or a trained predictor on
unlabeled data), by maximising summed log-magnitude along the harmonic comb.

Structure (separable nonlinear least squares; phases/amplitudes are linear
given frequencies, so the nonlinear search is only over low-dimensional
trajectory corrections):

- :func:`compute_logmag` — zoomed multichannel log-magnitude STFT front-end.
- :func:`comb_score` — differentiable k-summed track interpolation objective.
- :func:`estimate_clock_offset` — stage A: per-recording audio/telemetry
  clock offset by 1-D scan.
- :func:`coarse_delta` — stage B: windowed constant-per-window ``delta`` grid
  search with parabolic refinement (basin capture).
- :func:`refine_trajectories` — stage C: joint gradient refinement of
  per-rotor spline corrections ``delta_i(t)`` (torch, Adam), smoothness- and
  magnitude-regularised.
- :func:`comb_confidence` — per-window comb-contrast score (fit quality vs.
  off-comb reference shifts); gate for accepting refined labels.
- :func:`harmonic_lsq_residual` — coherent linear least-squares fit of
  per-harmonic complex envelopes by heterodyne demodulation at the (possibly
  refined) trajectories; the residual-energy ratio is the fit metric that
  improves when trajectories get closer to the truth.

Conventions: trajectories are in rev/s with harmonics at ``k * r`` Hz
(k = 1..K; blade-pass harmonics are simply the strong even/blade multiples),
arrays are time-last, and everything operates on a mono-or-multichannel audio
array plus a trajectory sampled on the STFT frame grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = [
    "RefineConfig",
    "LogMagSpec",
    "compute_logmag",
    "comb_score",
    "estimate_clock_offset",
    "coarse_delta",
    "refine_trajectories",
    "comb_confidence",
    "harmonic_lsq_residual",
    "RefinementResult",
]


@dataclass(frozen=True)
class RefineConfig:
    """Knobs for the whole refinement stack (defaults tuned for 16 kHz)."""

    sample_rate: int = 16000
    n_fft: int = 8192  # 1.95 Hz/bin at 16 kHz
    hop_length: int = 512  # 32 ms frame hop
    f_min: float = 60.0  # ignore harmonics below (rumble / DC)
    f_max: float = 6000.0  # and above (weak, broadband-dominated)
    k_max: int = 80  # hard cap on harmonic index
    # Stage B (coarse, constant delta per window)
    window_s: float = 2.0
    window_hop_s: float = 1.0
    delta_max: float = 3.0  # rev/s search half-range
    delta_step: float = 0.05  # rev/s grid step
    coarse_k_max: int = 20  # low harmonics only: wide basins
    # Stage C (spline refinement)
    knot_spacing_s: float = 0.25
    smooth_weight: float = 30.0  # on squared 2nd differences of knots (rev/s)^2
    anchor_weight: float = 0.02  # small pull toward the (coarse-corrected) init
    iters: int = 300
    lr: float = 0.03  # rev/s-scale Adam step
    # Confidence
    contrast_shifts: tuple[float, ...] = (-2.0, -1.35, -0.75, 0.75, 1.35, 2.0)
    device: str = "cpu"


@dataclass
class LogMagSpec:
    """Multichannel log-magnitude spectrogram with its grid metadata."""

    logmag: torch.Tensor  # (C, F, N) float32
    bin_hz: float
    frame_times: np.ndarray  # (N,) seconds, relative to the audio slice start
    sample_rate: int

    @property
    def n_frames(self) -> int:
        return int(self.logmag.shape[-1])

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0


def compute_logmag(audio: np.ndarray, cfg: RefineConfig) -> LogMagSpec:
    """Zoomed log-magnitude STFT of ``(T,)`` or ``(C, T)`` audio."""
    x = torch.as_tensor(np.atleast_2d(audio), dtype=torch.float32, device=cfg.device)
    window = torch.hann_window(cfg.n_fft, device=cfg.device)
    spec = torch.stft(
        x,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs()
    # Relative eps: keeps the log floor tied to signal scale, so silent bins
    # do not dominate dynamic range (lesson from the noise-gen loss diagnostic).
    eps = 1e-3 * mag.median()
    logmag = torch.log10(mag + eps)
    n = logmag.shape[-1]
    frame_times = np.arange(n) * cfg.hop_length / cfg.sample_rate
    return LogMagSpec(
        logmag=logmag,
        bin_hz=cfg.sample_rate / cfg.n_fft,
        frame_times=frame_times,
        sample_rate=cfg.sample_rate,
    )


def _interp_freq(logmag: torch.Tensor, freqs_hz: torch.Tensor, bin_hz: float) -> torch.Tensor:
    """Linearly interpolate ``(C, F, N)`` along F at per-frame frequencies.

    ``freqs_hz``: (..., N) target frequencies; returns ``(C, ..., N)`` values.
    Differentiable w.r.t. ``freqs_hz``.
    """
    pos = (freqs_hz / bin_hz).clamp(min=0.0)
    f_max_idx = logmag.shape[1] - 1
    pos = pos.clamp(max=float(f_max_idx) - 1e-3)
    lo = pos.floor().long()
    frac = pos - lo.to(pos.dtype)
    lo_flat = lo.reshape(-1, lo.shape[-1])  # (B, N)
    frac_flat = frac.reshape(-1, frac.shape[-1])
    n_idx = torch.arange(logmag.shape[-1], device=logmag.device)
    # gather per frame: logmag[c, lo[b, n], n]
    v_lo = logmag[:, lo_flat, n_idx]  # (C, B, N)
    v_hi = logmag[:, (lo_flat + 1).clamp(max=f_max_idx), n_idx]
    out = v_lo + (v_hi - v_lo) * frac_flat.unsqueeze(0)
    return out.reshape(logmag.shape[0], *lo.shape)


def _harmonic_freqs(
    r: torch.Tensor, cfg: RefineConfig, spec: LogMagSpec, k_max: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(K, ...)`` harmonic frequencies ``k * r`` and their validity mask."""
    kk = int(min(k_max or cfg.k_max, cfg.k_max))
    k = torch.arange(1, kk + 1, dtype=r.dtype, device=r.device)
    freqs = k.reshape(-1, *([1] * r.ndim)) * r.unsqueeze(0)  # (K, ..., N)
    valid = (freqs >= cfg.f_min) & (freqs <= min(cfg.f_max, 0.95 * spec.nyquist))
    return freqs, valid


def comb_score(
    spec: LogMagSpec,
    r: torch.Tensor,
    cfg: RefineConfig,
    *,
    k_max: int | None = None,
    per_frame: bool = False,
) -> torch.Tensor:
    """Mean log-magnitude along the harmonic comb of trajectories ``r``.

    ``r``: (..., N) rev/s on the frame grid (any leading batch shape, e.g.
    (R, N) rotors or (G, R, N) grid × rotors). Averages over channels,
    harmonics (valid ones), and — unless ``per_frame`` — frames.
    """
    freqs, valid = _harmonic_freqs(r, cfg, spec, k_max)
    vals = _interp_freq(spec.logmag, freqs, spec.bin_hz)  # (C, K, ..., N)
    w = valid.to(vals.dtype).unsqueeze(0)
    num = (vals * w).sum(dim=(0, 1))
    den = w.sum(dim=(0, 1)).clamp(min=1.0)
    frame_score = num / den  # (..., N)
    return frame_score if per_frame else frame_score.mean(dim=-1)


def _traj_on_frames(
    times: np.ndarray, values: np.ndarray, frame_times: np.ndarray, tau: float = 0.0
) -> np.ndarray:
    """Interpolate telemetry ``(R, M)`` @ ``times`` onto STFT frames (shifted by tau)."""
    out = np.stack([np.interp(frame_times + tau, times, values[i]) for i in range(values.shape[0])])
    return out.astype(np.float64)


def estimate_clock_offset(
    spec: LogMagSpec,
    motor_times: np.ndarray,
    motor_values: np.ndarray,
    cfg: RefineConfig,
    *,
    tau_range: float = 0.5,
    tau_step: float = 0.005,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Stage A: audio↔telemetry clock offset by scanning the comb score.

    Returns ``(tau_best, taus, scores)`` where positive ``tau`` means the
    telemetry lags the audio (trajectory evaluated at ``t + tau``).
    """
    taus = np.arange(-tau_range, tau_range + tau_step / 2, tau_step)
    trajs = np.stack(
        [_traj_on_frames(motor_times, motor_values, spec.frame_times, float(t)) for t in taus]
    )
    r = torch.as_tensor(trajs, dtype=torch.float32, device=spec.logmag.device)
    with torch.no_grad():
        scores = comb_score(spec, r, cfg, k_max=cfg.coarse_k_max).mean(dim=-1).cpu().numpy()
    i = int(np.argmax(scores))
    tau_best = float(taus[i])
    if 0 < i < len(taus) - 1:  # parabolic sub-step refinement
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            tau_best += float(0.5 * (a - c) / denom) * tau_step
    return tau_best, taus, scores


def _window_slices(n_frames: int, cfg: RefineConfig) -> list[slice]:
    hop_f = max(1, int(round(cfg.window_hop_s * cfg.sample_rate / cfg.hop_length)))
    win_f = max(2, int(round(cfg.window_s * cfg.sample_rate / cfg.hop_length)))
    starts = list(range(0, max(1, n_frames - win_f + 1), hop_f))
    if starts and starts[-1] + win_f < n_frames:
        starts.append(n_frames - win_f)
    return [slice(s, min(s + win_f, n_frames)) for s in starts]


def coarse_delta(
    spec: LogMagSpec,
    r_init: np.ndarray,
    cfg: RefineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage B: constant-per-window ``delta`` grid search per rotor.

    ``r_init``: (R, N) rev/s on the frame grid. Returns ``(delta, win_centers,
    win_scores)`` with ``delta`` (R, W) rev/s (parabolic-refined argmax),
    ``win_centers`` (W,) seconds, ``win_scores`` (R, W) the best comb score.
    """
    deltas = np.arange(-cfg.delta_max, cfg.delta_max + cfg.delta_step / 2, cfg.delta_step)
    r0 = torch.as_tensor(r_init, dtype=torch.float32, device=spec.logmag.device)
    d = torch.as_tensor(deltas, dtype=torch.float32, device=spec.logmag.device)
    slices = _window_slices(spec.n_frames, cfg)
    out = np.zeros((r_init.shape[0], len(slices)))
    out_scores = np.zeros_like(out)
    centers = np.array([spec.frame_times[sl].mean() for sl in slices])
    for w, sl in enumerate(slices):
        sub = LogMagSpec(spec.logmag[..., sl], spec.bin_hz, spec.frame_times[sl], spec.sample_rate)
        # (G, R, n) trajectories: every delta applied to every rotor
        r = r0[None, :, sl] + d[:, None, None]
        with torch.no_grad():
            scores = comb_score(sub, r, cfg, k_max=cfg.coarse_k_max).cpu().numpy()  # (G, R)
        for i in range(r_init.shape[0]):
            g = int(np.argmax(scores[:, i]))
            best = float(deltas[g])
            if 0 < g < len(deltas) - 1:
                a, b, c = scores[g - 1, i], scores[g, i], scores[g + 1, i]
                denom = a - 2 * b + c
                if abs(denom) > 1e-12:
                    best += float(0.5 * (a - c) / denom) * cfg.delta_step
            out[i, w] = best
            out_scores[i, w] = float(scores[g, i])
    return out, centers, out_scores


def _delta_windows_to_frames(
    delta: np.ndarray, centers: np.ndarray, frame_times: np.ndarray
) -> np.ndarray:
    """Interpolate per-window constant deltas onto the frame grid."""
    if delta.shape[1] == 1:
        return np.repeat(delta, len(frame_times), axis=1)
    return np.stack([np.interp(frame_times, centers, delta[i]) for i in range(delta.shape[0])])


@dataclass
class RefinementResult:
    r_refined: np.ndarray  # (R, N) rev/s on the frame grid
    r_coarse: np.ndarray  # (R, N) after stage B only
    frame_times: np.ndarray  # (N,)
    knots: np.ndarray  # (R, J) final knot deltas (rev/s, relative to coarse)
    knot_times: np.ndarray  # (J,)
    confidence: np.ndarray  # (R, W) per-window comb contrast
    conf_centers: np.ndarray  # (W,)
    history: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def refine_trajectories(
    spec: LogMagSpec,
    r_init: np.ndarray,
    cfg: RefineConfig,
    *,
    skip_coarse: bool = False,
) -> RefinementResult:
    """Stage B + C: coarse basin capture, then joint spline refinement.

    ``r_init``: (R, N) rev/s on the frame grid (telemetry- or
    predictor-derived). Returns refined trajectories plus diagnostics.
    """
    if not skip_coarse:
        delta_w, centers, _ = coarse_delta(spec, r_init, cfg)
        r_coarse = r_init + _delta_windows_to_frames(delta_w, centers, spec.frame_times)
    else:
        r_coarse = r_init.copy()

    device = spec.logmag.device
    n = spec.n_frames
    duration = float(spec.frame_times[-1] - spec.frame_times[0]) if n > 1 else cfg.knot_spacing_s
    n_knots = max(2, int(np.ceil(duration / cfg.knot_spacing_s)) + 1)
    knot_times = np.linspace(spec.frame_times[0], spec.frame_times[-1], n_knots)

    # Linear interpolation matrix knots -> frames (fixed, differentiable pass-through).
    A = np.zeros((n, n_knots), dtype=np.float32)
    idx = np.clip(np.searchsorted(knot_times, spec.frame_times, side="right") - 1, 0, n_knots - 2)
    tloc = (spec.frame_times - knot_times[idx]) / (knot_times[idx + 1] - knot_times[idx])
    A[np.arange(n), idx] = 1.0 - tloc
    A[np.arange(n), idx + 1] = tloc
    A_t = torch.as_tensor(A, device=device)

    r0 = torch.as_tensor(r_coarse, dtype=torch.float32, device=device)
    knots = torch.zeros((r_init.shape[0], n_knots), device=device, requires_grad=True)
    opt = torch.optim.Adam([knots], lr=cfg.lr)
    history: list[float] = []
    for _ in range(cfg.iters):
        opt.zero_grad()
        delta = knots @ A_t.T  # (R, N)
        score = comb_score(spec, r0 + delta, cfg).mean()
        smooth = (knots[:, 2:] - 2 * knots[:, 1:-1] + knots[:, :-2]).pow(2).mean()
        anchor = knots.pow(2).mean()
        loss = -score + cfg.smooth_weight * smooth + cfg.anchor_weight * anchor
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    with torch.no_grad():
        delta = (knots @ A_t.T).cpu().numpy()
    r_refined = r_coarse + delta

    conf, conf_centers = comb_confidence(spec, r_refined, cfg)
    return RefinementResult(
        r_refined=r_refined,
        r_coarse=r_coarse,
        frame_times=spec.frame_times,
        knots=knots.detach().cpu().numpy(),
        knot_times=knot_times,
        confidence=conf,
        conf_centers=conf_centers,
        history=history,
    )


def comb_confidence(
    spec: LogMagSpec, r: np.ndarray, cfg: RefineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window comb contrast: score(r) − median(score(r + shifts)).

    High contrast ⇒ a real comb is locked; ~0 ⇒ no harmonic structure (do not
    trust the refined label there).
    """
    rt = torch.as_tensor(r, dtype=torch.float32, device=spec.logmag.device)
    shifts = torch.as_tensor(cfg.contrast_shifts, dtype=torch.float32, device=rt.device)
    slices = _window_slices(spec.n_frames, cfg)
    out = np.zeros((r.shape[0], len(slices)))
    centers = np.array([spec.frame_times[sl].mean() for sl in slices])
    with torch.no_grad():
        for w, sl in enumerate(slices):
            sub = LogMagSpec(
                spec.logmag[..., sl], spec.bin_hz, spec.frame_times[sl], spec.sample_rate
            )
            on = comb_score(sub, rt[:, sl], cfg)  # (R,)
            off = comb_score(sub, rt[None, :, sl] + shifts[:, None, None], cfg)  # (S, R)
            out[:, w] = (on - off.median(dim=0).values).cpu().numpy()
    return out, centers


def harmonic_lsq_residual(
    audio: np.ndarray,
    r_frames: np.ndarray,
    frame_times: np.ndarray,
    cfg: RefineConfig,
    *,
    k_max: int | None = None,
    block_s: float = 0.25,
) -> dict[str, Any]:
    """Joint block-wise linear least-squares harmonic fit; returns residual ratio.

    Given trajectories ``r_frames`` (R, N) on the frame grid, build per-block
    design matrices of cos/sin columns along every harmonic track
    (phase = 2 pi k * integral of r) and solve for all track amplitudes
    **jointly** — this is the linear (variable-projection) half of the
    separable problem, and it handles overlapping/crossing harmonics
    correctly where independent per-track demodulation double-counts.
    Amplitudes are piecewise-constant per block (``block_s`` seconds).
    ``residual_energy / signal_energy`` drops as trajectories approach truth.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x[0]
    sr = cfg.sample_rate
    t = np.arange(len(x)) / sr
    kk = int(min(k_max or cfg.k_max, cfg.k_max))
    f_hi = min(cfg.f_max, 0.95 * sr / 2)

    # Continuous phase per rotor (global cumsum keeps blocks phase-coherent).
    phases = []
    k_valid: list[list[int]] = []
    for i in range(r_frames.shape[0]):
        r_t = np.interp(t, frame_times, r_frames[i])
        phases.append(2 * np.pi * np.cumsum(r_t) / sr)
        f_med = float(np.median(r_frames[i]))
        k_valid.append([k for k in range(1, kk + 1) if cfg.f_min <= k * f_med <= f_hi])

    block = max(16, int(round(block_s * sr)))
    resynth = np.zeros_like(x)
    n_tracks = sum(len(ks) for ks in k_valid)
    for s in range(0, len(x), block):
        sl = slice(s, min(s + block, len(x)))
        cols = []
        for i, ks in enumerate(k_valid):
            ph = phases[i][sl]
            for k in ks:
                cols.append(np.cos(k * ph))
                cols.append(np.sin(k * ph))
        if not cols:
            continue
        a_mat = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(a_mat, x[sl], rcond=1e-8)
        resynth[sl] = a_mat @ coef
    residual = x - resynth
    return {
        "residual_ratio": float(np.sum(residual**2) / max(np.sum(x**2), 1e-12)),
        "harmonic_energy_ratio": float(np.sum(resynth**2) / max(np.sum(x**2), 1e-12)),
        "n_tracks": n_tracks,
    }
