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
- :func:`harmonic_lsq_residual` — joint linear least-squares harmonic fit
  (VP-transform primitives); the residual-energy ratio is the fit metric that
  improves when trajectories get closer to the truth.
- :func:`refine_coherent` — stage D: phase-slope refinement by narrowband
  harmonic demodulation. Reads *phase*, not magnitude ridges, so it stays
  unbiased where the magnitude stages fail (tightly-paired rotors whose
  low/mid harmonics merge — e.g. DREGON's ~0.65 rev/s pairs). Needs a good
  init (telemetry or stages A–C); precision ≪ 0.1 rev/s.

Conventions: trajectories are in rev/s with harmonics at ``k * r`` Hz
(k = 1..K; blade-pass harmonics are simply the strong even/blade multiples),
arrays are time-last, and everything operates on a mono-or-multichannel audio
array plus a trajectory sampled on the STFT frame grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

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
    "refine_coherent",
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
    window_len: int = 2048,
    hop_len: int = 512,
) -> dict[str, Any]:
    """Joint per-frame least-squares harmonic fit; returns the residual ratio.

    Thin wrapper over the project's VP-transform primitives
    (``models.generative.harmonic_transform.lstsq_VP_transform`` /
    ``inverse_VP_transform``, cf. ``experiments/kalman_harmonic/phase0.py``):
    build harmonic frequency series ``k * r_i(t)`` for all rotors, solve the
    windowed I/Q amplitudes of **all rotors and harmonics jointly** per frame
    (the linear, variable-projection half of the problem — overlapping and
    crossing rotor harmonics are attributed correctly), reconstruct by
    overlap-add and report ``residual_energy / signal_energy``. This is the
    same harmonic basis the generative models synthesise in, so the metric is
    directly comparable across the project. Drops as trajectories approach
    the truth.
    """
    from models.generative.dsp import harmonic_freq_series
    from models.generative.harmonic_transform import (
        inverse_VP_transform,
        lstsq_VP_transform,
    )

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    t = np.arange(len(x)) / cfg.sample_rate
    r_t = np.stack(
        [np.interp(t, frame_times, r_frames[i]) for i in range(r_frames.shape[0])]
    ).astype(np.float32)
    kk = int(min(k_max or cfg.k_max, cfg.k_max))

    wav = torch.as_tensor(x)
    freqs = harmonic_freq_series(torch.as_tensor(r_t), kk)  # (R, K, T)
    with torch.no_grad():
        v = lstsq_VP_transform(
            freqs,
            wav,
            window_len=window_len,
            hop_len=hop_len,
            sr=cfg.sample_rate,
            method="gelsy",
        )
        recon = inverse_VP_transform(
            freqs, v, window_len=window_len, hop_len=hop_len, sr=cfg.sample_rate
        )
    if recon.dim() > wav.dim():
        recon = recon.sum(0)  # sum per-rotor components
    n = min(recon.shape[-1], wav.shape[-1])
    resid = wav[..., :n] - recon[..., :n]
    x_energy = float(torch.sum(wav[..., :n] ** 2))
    return {
        "residual_ratio": float(torch.sum(resid**2)) / max(x_energy, 1e-12),
        "harmonic_energy_ratio": float(torch.sum(recon[..., :n] ** 2)) / max(x_energy, 1e-12),
        "n_tracks": int(freqs.shape[0] * freqs.shape[1]),
    }


def refine_coherent(
    audio: np.ndarray,
    r_frames: np.ndarray,
    frame_times: np.ndarray,
    cfg: RefineConfig,
    *,
    k_min: int = 6,
    k_max: int | None = None,
    bandwidth_hz: float = 3.0,
    n_iter: int = 4,
    smooth_s: float = 0.25,
    max_step: float = 0.5,
) -> np.ndarray:
    """Stage D: coherent phase-slope refinement by harmonic demodulation.

    For each rotor and harmonic ``k``, demodulate the (mono) signal by the
    current track phase and low-pass to a narrow band (``bandwidth_hz``); if
    the track is off by ``delta`` rev/s the complex envelope rotates at
    ``k * delta`` Hz, so its phase slope is a direct, *local* estimate of the
    trajectory error — precision grows with ``k`` (Fisher weights
    ``k^2 |z|^2``), and the narrow band structurally rejects the neighbouring
    rotor's comb for ``k >= bandwidth / pair_separation`` — the twin-capture
    failure mode of magnitude-comb refinement on tightly paired quadrotors
    (measured on DREGON: rotor pairs ~0.65 rev/s apart) cannot occur.

    Returns the refined ``(R, N)`` trajectory on the frame grid. Iterative
    (default 4 rounds), each round clipped to ``max_step`` rev/s and smoothed
    over ``smooth_s`` seconds. Unlike the magnitude stages this reads *phase*,
    so it needs a good init (within ~``bandwidth_hz / k_max`` of the truth —
    run after stages A–C or from near-truth telemetry).
    """
    from scipy.signal import butter, filtfilt

    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x[0]
    sr = cfg.sample_rate
    t_s = np.arange(len(x)) / sr
    kk = int(min(k_max or 40, cfg.k_max))
    ba = cast("tuple[np.ndarray, np.ndarray]", butter(2, bandwidth_hz / (sr / 2)))
    b, a = ba
    smooth_n = max(1, int(round(smooth_s * sr)))
    kernel = np.ones(smooth_n) / smooth_n
    f_hi = min(cfg.f_max, 0.95 * sr / 2)

    r = r_frames.astype(np.float64).copy()
    for _ in range(n_iter):
        for i in range(r.shape[0]):
            r_t = np.interp(t_s, frame_times, r[i])
            phase1 = 2 * np.pi * np.cumsum(r_t) / sr
            num = np.zeros(len(x) - 1)
            den = np.zeros(len(x) - 1)
            f_med = float(np.median(r[i]))
            for k in range(k_min, kk + 1):
                if not (cfg.f_min <= k * f_med <= f_hi):
                    continue
                z = filtfilt(b, a, x * np.exp(-1j * k * phase1))
                dphi = np.diff(np.unwrap(np.angle(z)))
                inst_err_hz = dphi * sr / (2 * np.pi)
                w = (np.abs(z[1:]) ** 2) * k * k
                # per-harmonic rev/s error estimate = inst envelope rotation / k
                num += w * (inst_err_hz / k)
                den += w
            delta = num / np.maximum(den, 1e-18)
            delta = np.convolve(delta, kernel, mode="same")
            delta = np.clip(delta, -max_step, max_step)
            r[i] += np.interp(frame_times, t_s[1:], delta)
    return r
