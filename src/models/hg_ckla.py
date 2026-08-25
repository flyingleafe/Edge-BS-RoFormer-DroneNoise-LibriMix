"""HG-CKLA — the harmonic-gather CKLA cell (CKLA as a true ``pi_kalman`` pass).

Design of record: ``docs/pikalman-ckla-design.md``. Companion to
``docs/ckla-design.md`` (``models/ckla.py`` — the scan machinery this reuses)
and ``src/tracking/phase_increment_tracker.py`` (the classical algorithm this
mirrors). This module is stage A of the design: the conditional **refiner**
only (design §4 "Seeding and modes", first bullet). The end-to-end mode
(seed head + stack, PIT at the seed) is not here.

Why the cell exists (design §1): ``pi_kalman`` reads the spectrogram *at the
harmonic positions its own current estimate predicts*, and the tried CKLA head
cannot, because its frequency pool collapses the spectral axis before the
recurrence sees it. HG-CKLA moves the measurement inside the recurrence, as a
differentiable gather at state-predicted harmonic positions.

One cell = one ``pi_kalman`` outer iteration:

1. Positions ``p_{r,k}(t) = k f_r(t) / df`` from the state (rev/s, rotation
   harmonics), :func:`harmonic_positions`.
2. Soft gather with a narrow Gaussian window over frequency bins — the
   analog of the demodulation band, and the term that keeps the gradient
   local, :func:`soft_gather`.
3. Innovation phasor ``u = X_hat(t) conj(X_hat(t-1)) e^{-i 2 pi k f_r H/fs}``,
   whose angle is the per-harmonic frequency error, :func:`innovation_phasors`.
   Both frames of the pair go through ONE gather (the same weights), so the
   window contributes no phase of its own. The cell is fed the UNIT phasors
   and ``log1p`` magnitudes, never an explicit angle.
4. A linear-physics estimate of the rate error from those angles with the
   WP18 weight law ``w_k ~ k^2`` (learnable), :func:`physics_rate_error`, plus
   an MLP correction, plus one CKLA block over time (the random-walk
   smoother), and a sigmoid gain that decides how much of the estimate enters
   the state (the Kalman gain).

Annealing is the classical ``k_caps`` schedule: cell ``j`` gathers harmonics
``1..k_caps[j]`` only. Twin collisions are down-weighted by a soft gate over
the state harmonics, :func:`twin_gate`; the classical joint two-phasor pair
mode is out of scope (design §7).

Deviations from the design document, and why:

- **The linear-physics path uses the phasor ANGLE, not** ``Im(u/|u|)``
  (design §4 step 4). ``Im`` is the small-angle rendering of the same
  quantity, and the angle is not small in the regime this cell must work in:
  at ``k = 10``, ``df = 1`` rev/s and hop 512 at 16 kHz, the true angle is
  2.0 rad, where ``sin`` under-reads it by 55 %. The classical algorithm
  measures ``arg`` with a wrap guard, which is what this does — a guarded
  ``atan2`` on the UNIT phasor, with the dead-branch replacement of
  ``ckla.phase_unit_features`` so the backward pass is finite at zero
  magnitude, plus a mask that drops increments near ``pi``. The MLP still
  sees unit phasors only.
- **The voicing gate (design §4 "Voicing for free") is not implemented.**
  v1 stays focused on the rate state; the gate is an output-side head that
  can be added without touching the cell.
- **The complex STFT is computed inside the model** (a plain ``torch.stft``,
  same parameters as ``frontends.stft.STFTMag``, ``normalized=True``) instead
  of an added front-end output (design §7 "Complex STFT input"). v1 keeps the
  ``SpectralFrontEnd`` contract untouched; sharing one STFT with a trunk
  front-end is an optimization for the end-to-end mode, where a trunk exists.

Cost note: the gather materializes ``2 (B, R, K, W, T)`` values per cell —
the current and the delayed spectrogram — with ``W = 2 ceil(trunc sigma) + 1``
(13 at the defaults). The anneal schedule is therefore also a memory
schedule: cell 0 gathers 10 harmonics, not 40.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.ckla import CKLABlock

__all__ = [
    "HGCKLACell",
    "HGCKLARefiner",
    "harmonic_positions",
    "innovation_phasors",
    "physics_rate_error",
    "soft_gather",
    "twin_gate",
]

_TINY = 1e-12


def harmonic_positions(f: Tensor, n_harmonics: int, n_fft: int, sample_rate: float) -> Tensor:
    """Fractional STFT-bin positions of the rotation harmonics.

    Parameters
    ----------
    f : (B, R, T)
        Rotor rate state in rev/s (= Hz of harmonic 1).
    n_harmonics : int
        Number of harmonics K; harmonic k sits at ``k f`` Hz.
    n_fft, sample_rate :
        STFT grid: one bin is ``sample_rate / n_fft`` Hz.

    Returns
    -------
    (B, R, K, T)
        ``p = k f / df`` in fractional bins.
    """
    k = torch.arange(1, n_harmonics + 1, device=f.device, dtype=f.dtype)
    return f.unsqueeze(2) * k.view(1, 1, -1, 1) * (n_fft / sample_rate)


def soft_gather(
    x_re: Tensor,
    x_im: Tensor,
    pos: Tensor,
    sigma: float = 1.5,
    trunc: float = 4.0,
    prev_re: Tensor | None = None,
    prev_im: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gaussian-window gather of a complex STFT at fractional bin positions.

    ``X_hat(p) = sum_f w(f - p) X(f)`` with ``w`` a Gaussian of width *sigma*
    bins, truncated at ``+-trunc sigma`` and renormalized to unit sum. The
    window is the analog of the demodulation band: a point interpolation sees
    two bins, and its gradient dies as soon as the state leaves the line.

    The weight is **complex**: ``w(d) = gauss(d) exp(i pi d)``. ``torch.stft``
    refers the phase of every bin to the start of the frame, not to the center
    of the analysis window, which puts a factor ``exp(-i pi d)`` on the bin at
    offset ``d`` from a line. A real Gaussian therefore sums the main-lobe
    bins with alternating signs and they cancel: measured on a 10-tooth comb,
    the gathered magnitude varies over 24 dB with the fractional offset, and
    at the worst offsets the phasor dies. With the alignment the same comb
    gathers a magnitude that is flat over all ten teeth.

    One vectorized gather covers all (rotor, harmonic, frame) positions.

    Parameters
    ----------
    x_re, x_im : (B, F, T)
        Real and imaginary parts of the complex STFT.
    pos : (B, R, K, T)
        Fractional bin positions, from :func:`harmonic_positions`.
    sigma : float
        Gaussian width in bins.
    trunc : float
        Window half-width in units of *sigma*.
    prev_re, prev_im : (B, F, T), optional
        A second spectrogram to gather with the SAME weights (the
        one-frame-delayed copy — see :func:`innovation_phasors`). When given,
        the returned ``g_re``/``g_im`` are stacked along a leading axis of
        size 2: ``[current, delayed]``.

    Returns
    -------
    (g_re, g_im, valid)
        ``g_*`` are (B, R, K, T), or (2, B, R, K, T) with *prev_re* given.
        ``valid`` is a (B, R, K, T) 0/1 mask, 0 where the position leaves the
        usable band (DC bin or above Nyquist).
    """
    b, n_bins, t = x_re.shape
    r, k = pos.shape[1], pos.shape[2]
    half = int(math.ceil(trunc * sigma))
    off = torch.arange(-half, half + 1, device=pos.device, dtype=pos.dtype)
    width = off.numel()

    # round() has zero derivative, so the integer grid carries no gradient;
    # the window weight below does, through -pos.
    base = torch.round(pos.detach())
    idx = base.unsqueeze(-1) + off  # (B, R, K, T, W)
    dist = idx - pos.unsqueeze(-1)
    w = torch.exp(-0.5 * (dist / sigma) ** 2)
    inside = (idx >= 0) & (idx <= n_bins - 1)
    w = w * inside
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    dist_p = dist.permute(0, 1, 2, 4, 3)  # (B, R, K, W, T)
    w_p = w.permute(0, 1, 2, 4, 3)
    w_re = w_p * torch.cos(math.pi * dist_p)
    w_im = w_p * torch.sin(math.pi * dist_p)

    flat_idx = idx.clamp(0, n_bins - 1).long().permute(0, 1, 2, 4, 3).reshape(b, -1, t)
    src_re = x_re if prev_re is None else torch.stack([x_re, prev_re])
    src_im = x_im if prev_im is None else torch.stack([x_im, prev_im])
    shape = (b, r, k, width, t) if prev_re is None else (2, b, r, k, width, t)
    fi = flat_idx if prev_re is None else flat_idx.expand(2, -1, -1, -1)
    sel_re = torch.gather(src_re, -2, fi).view(shape)
    sel_im = torch.gather(src_im, -2, fi).view(shape)
    g_re = (sel_re * w_re - sel_im * w_im).sum(dim=-2)
    g_im = (sel_re * w_im + sel_im * w_re).sum(dim=-2)

    valid = ((pos >= 1.0) & (pos <= n_bins - 2.0)).to(x_re.dtype)
    return g_re, g_im, valid


def innovation_phasors(
    g_re: Tensor,
    g_im: Tensor,
    p_re: Tensor,
    p_im: Tensor,
    f: Tensor,
    hop_length: int,
    sample_rate: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-harmonic innovation phasors of a gathered comb.

    ``u(t) = X_hat(t) conj(X_hat(t-1)) exp(-i 2 pi k f(t) H / fs)``: the
    consecutive-frame product, de-rotated by the phase advance the state
    predicts. For a stationary line the gathered value is a fixed linear
    combination of bins that all advance at the line frequency, so
    ``arg u = 2 pi k (f_true - f) H / fs`` exactly — the per-harmonic rate
    error, free of the carrier.

    Both frames of the pair MUST be gathered with the same weights, that is
    at the position the state predicts for frame ``t`` (pass the delayed
    spectrogram to :func:`soft_gather` as ``prev_re``/``prev_im``). The
    window's own contribution to the phase then divides out of the product
    exactly, and a moving state adds no bias — one filter, two consecutive
    outputs, exactly as the classical demodulation reads its envelope.

    Parameters
    ----------
    g_re, g_im : (B, R, K, T)
        Gathered complex values at frame ``t``.
    p_re, p_im : (B, R, K, T)
        The same gather applied to the one-frame-delayed spectrogram.
    f : (B, R, T)
        The state the gather used, rev/s.
    hop_length, sample_rate :
        Frame advance ``dt = hop_length / sample_rate``.

    Returns
    -------
    (u_re, u_im, arg_u, log_mag, valid_t)
        Unit phasor components, its guarded angle, ``log1p`` of the gathered
        magnitude, and a 0/1 mask that is 0 in frame 0 (no predecessor).
        Where the product magnitude underflows, the phasor is replaced by
        ``(1, 0)`` with no gradient path — the ``ckla.phase_unit_features``
        discipline, so ``atan2``'s backward stays finite.
    """
    n_harm = g_re.shape[2]
    k = torch.arange(1, n_harm + 1, device=f.device, dtype=f.dtype).view(1, 1, -1, 1)
    dt = hop_length / sample_rate

    d_re = g_re * p_re + g_im * p_im
    d_im = g_im * p_re - g_re * p_im

    # Reduce the predicted advance modulo one turn before cos/sin: k f dt runs
    # to ~130 turns at k=40, and fp32 loses the fraction that matters.
    turns = torch.remainder(k * f.unsqueeze(2) * dt, 1.0)
    theta = 2.0 * math.pi * turns
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    r_re = d_re * cos_t + d_im * sin_t
    r_im = d_im * cos_t - d_re * sin_t

    mag_d = torch.sqrt(r_re * r_re + r_im * r_im + 1e-24)
    alive = mag_d > _TINY
    u_re = torch.where(alive, r_re / mag_d, torch.ones_like(r_re))
    u_im = torch.where(alive, r_im / mag_d, torch.zeros_like(r_im))
    arg_u = torch.atan2(u_im, u_re)

    log_mag = torch.log1p(torch.sqrt(g_re * g_re + g_im * g_im + 1e-24))
    valid_t = torch.ones_like(log_mag)
    valid_t[..., 0] = 0.0
    return u_re, u_im, arg_u, log_mag, valid_t


def physics_rate_error(
    arg_u: Tensor, weights: Tensor, hop_length: int, sample_rate: float
) -> Tensor:
    """Weighted linear-physics estimate of the rate error, rev/s.

    ``df = sum_k a_k arg(u_k) / (2 pi k dt)`` with ``a_k`` the normalized
    *weights* (WP18: ``1/v_k ~ k^2``). Every per-harmonic term is an unbiased
    estimate of the same ``df``, so the weighting only sets the variance.

    Parameters
    ----------
    arg_u : (B, R, K, T)
        Innovation angles, from :func:`innovation_phasors`.
    weights : (B, R, K, T)
        Non-negative per-measurement weights (already gated/masked).
    """
    n_harm = arg_u.shape[2]
    k = torch.arange(1, n_harm + 1, device=arg_u.device, dtype=arg_u.dtype)
    dt = hop_length / sample_rate
    per_k = arg_u / (2.0 * math.pi * k.view(1, 1, -1, 1) * dt)
    num = (weights * per_k).sum(dim=2)
    den = weights.sum(dim=2)
    return num / den.clamp(min=_TINY)


def twin_gate(f: Tensor, n_harmonics: int, band_hz: Tensor | float, tau_hz: Tensor) -> Tensor:
    """Soft twin gate: down-weight harmonics collided with another rotor.

    ``g = sigmoid((d - band) / tau)`` where ``d`` is the distance in Hz from
    harmonic ``k`` of rotor ``r`` to the NEAREST harmonic of any other rotor.
    The nearest one is closed-form — ``k' = round(k f_r / f_j)`` clamped into
    the harmonic range — so the gate costs ``O(R^2 K T)`` and never
    materializes the full harmonic-pair grid.

    Parameters
    ----------
    f : (B, R, T)
        Rotor rate state, rev/s.
    n_harmonics : int
        Harmonic count of both the gated and the interfering combs.
    band_hz : float or scalar Tensor
        Collision half-width (the demodulation band).
    tau_hz : Tensor
        Positive gate temperature in Hz.

    Returns
    -------
    (B, R, K, T) in (0, 1)
    """
    _, n_rotor, _ = f.shape
    k = torch.arange(1, n_harmonics + 1, device=f.device, dtype=f.dtype).view(1, 1, -1, 1)
    fk = f.unsqueeze(2) * k  # (B, R, K, T) Hz
    dist = torch.full_like(fk, 1e6)
    for j in range(n_rotor):
        fj = f[:, j, :].unsqueeze(1).unsqueeze(1).clamp(min=1e-3)  # (B, 1, 1, T)
        kj = torch.round((fk / fj).detach()).clamp(1.0, float(n_harmonics))
        d = (fk - kj * fj).abs()
        d[:, j] = 1e6  # a rotor never gates itself
        dist = torch.minimum(dist, d)
    return torch.sigmoid((dist - band_hz) / tau_hz.clamp(min=1e-3))


class HGCKLACell(nn.Module):
    """One neural ``pi_kalman`` iteration (design §4).

    ``forward(x_re, x_im, f) -> df``: gather at the state, measure the
    innovation angles, fuse them into a rate-error estimate, smooth over time
    with one CKLA block, and emit the gated state increment.

    The output head starts at (near) zero weight with a positive gain bias, so
    an UNTRAINED cell already applies ``sigmoid(1) = 0.73`` of the classical
    linear-physics correction — the analog of ``SimpleConvV2CKLACond``'s
    identity start, one step better.

    Parameters
    ----------
    k_cap : int
        Harmonic cap of this cell (the anneal schedule); the cell gathers
        ``1..k_cap`` only.
    d_model, n_state, p_init, rotation, readout :
        The CKLA block (``ckla.CKLABlock`` / ``ckla.ComplexKLALayer``).
    sigma, trunc :
        Gather window (bins, and its half-width in sigmas).
    band_bins :
        Twin-collision half-width, in STFT bins.
    max_step :
        Saturation of the per-cell increment, rev/s.
    wrap_guard :
        Drop innovation angles above ``wrap_guard * pi`` (the classical wrap
        guard: near ``pi`` the increment is ambiguous).
    """

    def __init__(
        self,
        k_cap: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: float = 16000.0,
        d_model: int = 64,
        n_state: int = 16,
        sigma: float = 1.5,
        trunc: float = 4.0,
        band_bins: float = 3.0,
        max_step: float = 3.0,
        wrap_guard: float = 0.8,
        p_init: float = 1.0,
        rotation: bool = True,
        readout: str = "phase_unit",
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.k_cap = int(k_cap)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = float(sample_rate)
        self.sigma = float(sigma)
        self.trunc = float(trunc)
        self.max_step = float(max_step)
        self.wrap_guard = float(wrap_guard)
        self.band_hz = float(band_bins) * self.sample_rate / self.n_fft

        # WP18 weight law: 1/v_k ~ k^2, normalized to mean 1 and kept positive
        # by softplus. Learnable — the law is the init, not a constraint.
        k = torch.arange(1, self.k_cap + 1, dtype=torch.float32)
        w0 = k**2 / (k**2).mean()
        self.w_param = nn.Parameter(torch.log(torch.expm1(w0)))
        # Twin-gate temperature (softplus, init = half the band).
        self.tau_param = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(max(self.band_hz / 2.0, 1e-2))))
        )

        n_feat = 3 * self.k_cap + 2  # [Re u, Im u, log|X|] per harmonic + 2 scalars
        self.in_proj = nn.Linear(n_feat, d_model)
        self.block = CKLABlock(
            d_model,
            mlp_ratio=mlp_ratio,
            n_state=n_state,
            rotation=rotation,
            p_init=p_init,
            readout=readout,
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 2)  # [gain logit, correction]
        # Near-zero weight + positive gain bias: the cell starts as the
        # classical pass (gain sigmoid(1) = 0.73 of the physics estimate, no
        # learned correction), and the small non-zero weight keeps the whole
        # stack in the gradient from the first step.
        nn.init.normal_(self.head.weight, std=1e-3)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.0, 0.0]))

    def measure(self, x_re: Tensor, x_im: Tensor, f: Tensor) -> dict[str, Tensor]:
        """The measurement operator: gather, innovate, weight, fuse.

        Returns a dict with ``feats`` (B, R, 3K+2, T), ``df_phys`` (B, R, T)
        and the intermediate ``mask``/``arg_u`` (diagnostics and tests).
        """
        pos = harmonic_positions(f, self.k_cap, self.n_fft, self.sample_rate)
        # The delayed spectrogram rides through the SAME gather, so both
        # frames of the innovation pair come out of one filter.
        prev_re = F.pad(x_re[..., :-1], (1, 0))
        prev_im = F.pad(x_im[..., :-1], (1, 0))
        g, gi, valid = soft_gather(x_re, x_im, pos, self.sigma, self.trunc, prev_re, prev_im)
        u_re, u_im, arg_u, log_mag, valid_t = innovation_phasors(
            g[0], gi[0], g[1], gi[1], f, self.hop_length, self.sample_rate
        )
        gate = twin_gate(f, self.k_cap, self.band_hz, F.softplus(self.tau_param))
        wrap_ok = (arg_u.abs() <= self.wrap_guard * math.pi).to(arg_u.dtype)
        mask = valid * valid_t * wrap_ok * gate

        weights = F.softplus(self.w_param).view(1, 1, -1, 1) * mask
        df_phys = physics_rate_error(arg_u, weights, self.hop_length, self.sample_rate)
        df_phys = df_phys.clamp(-self.max_step, self.max_step)

        scalars = torch.stack([df_phys / self.max_step, mask.mean(dim=2)], dim=2)
        feats = torch.cat([u_re * mask, u_im * mask, log_mag * mask, scalars], dim=2)
        return {"feats": feats, "df_phys": df_phys, "mask": mask, "arg_u": arg_u}

    def forward(self, x_re: Tensor, x_im: Tensor, f: Tensor) -> Tensor:
        """``(B, F, T)`` STFT + ``(B, R, T)`` state -> ``(B, R, T)`` increment."""
        m = self.measure(x_re, x_im, f)
        b, n_rotor, _, t = m["feats"].shape
        # Rotors fold into the batch: the cell is shared across rotors, and
        # the twin gate is the only cross-rotor coupling.
        h = m["feats"].permute(0, 1, 3, 2).reshape(b * n_rotor, t, -1)
        h = self.block(self.in_proj(h))
        out = self.head(self.norm(h)).view(b, n_rotor, t, 2).float()
        gain = torch.sigmoid(out[..., 0])
        corr = self.max_step * torch.tanh(out[..., 1])
        return gain * (m["df_phys"] + corr)


class HGCKLARefiner(nn.Module):
    """Stage-A HG-CKLA conditional refiner: ``forward(audio, cond) -> track``.

    Contract of ``ckla.SimpleConvV2CKLACond`` (registry key
    ``hg_ckla_refiner``, task ``rps_prediction`` with ``use_cond=true``):
    the ``(B, R, F)`` conditioning track is a coarse/corrupted RPS estimate,
    the output is a BOUNDED residual on it,
    ``cond + max_delta tanh(sum_j df_j / max_delta)``, and output row ``i``
    belongs to conditioning row ``i`` — so training uses the plain non-PIT
    ``losses.RPSMSELoss`` on ``(audio, corrupt(GT)) -> GT`` pairs from
    ``data_processing/rps_corruption.py``.

    What is different from the conv-trunk refiner: there is no trunk. The
    conditioning is the STATE, the model reads the complex STFT *at the
    harmonic positions that state predicts*, and each cell re-gathers at the
    updated state. The residual is bounded after every cell too, so the
    gathers never leave the tube around the conditioning.

    ``k_caps`` is the classical coarse-to-fine schedule (one cell per entry);
    the harmonic count is ``max(k_caps)``.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        sample_rate: float = 16000.0,
        k_caps: Sequence[int] = (10, 25, 40),
        d_model: int = 64,
        n_state: int = 16,
        sigma: float = 1.5,
        trunc: float = 4.0,
        band_bins: float = 3.0,
        max_delta: float = 5.0,
        max_step: float = 3.0,
        wrap_guard: float = 0.8,
        p_init: float = 1.0,
        rotation: bool = True,
        readout: str = "phase_unit",
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.num_rotors = int(num_rotors)
        self.sample_rate = float(sample_rate)
        self.max_delta = float(max_delta)
        self.k_caps = tuple(int(c) for c in k_caps)
        if not self.k_caps or min(self.k_caps) < 1:
            raise ValueError(
                f"k_caps must be a non-empty schedule of positive ints, got {k_caps!r}"
            )
        self.window: Tensor
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.cells = nn.ModuleList(
            HGCKLACell(
                k_cap=cap,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
                d_model=d_model,
                n_state=n_state,
                sigma=sigma,
                trunc=trunc,
                band_bins=band_bins,
                max_step=max_step,
                wrap_guard=wrap_guard,
                p_init=p_init,
                rotation=rotation,
                readout=readout,
            )
            for cap in self.k_caps
        )

    def num_frames(self, n_samples: int) -> int:
        return n_samples // self.hop_length + 1

    def stft(self, audio: Tensor) -> Tensor:
        """``(B, N)`` or ``(B, 1, N)`` -> complex ``(B, F, T)``.

        Same grid and normalization as ``frontends.stft.STFTMag``; fp32 by
        construction, because every phase measurement below depends on it.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return torch.stft(
            audio.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.float(),
            return_complex=True,
            normalized=True,
        )

    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:
        """``audio (B, N)`` + ``cond (B, R, F_cond)`` -> ``(B, R, T)`` rev/s.

        ``cond`` is linearly resampled onto the STFT frame grid when the frame
        counts differ (the ``SimpleConvV2CKLACond`` convention).
        """
        if cond.dim() != 3 or cond.shape[1] != self.num_rotors:
            raise ValueError(f"cond must be (B, {self.num_rotors}, F), got {tuple(cond.shape)}")
        spec = self.stft(audio)
        x_re, x_im = spec.real.contiguous(), spec.imag.contiguous()
        t_frames = x_re.shape[-1]

        c = cond.float()
        if c.shape[-1] != t_frames:
            c = F.interpolate(c, size=t_frames, mode="linear", align_corners=True)

        f = c
        acc = torch.zeros_like(c)
        for cell in self.cells:
            acc = acc + cell(x_re, x_im, f)
            f = c + self.max_delta * torch.tanh(acc / self.max_delta)
        return f
