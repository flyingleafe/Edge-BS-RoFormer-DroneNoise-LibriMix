"""
Position-aware drone-noise generator.

Extends :class:`HarmonicNoiseGenNew` from a *mono* synthesiser into one that
renders the noise **at an observation point** given the geometry of the rotors.

Physical model (a deliberate first approximation)
-------------------------------------------------
Each rotor is treated as an isotropic **point source**. Its emitted waveform
(harmonic oscillator bank + its own filtered broadband residual) is synthesised
once, *as radiated at the rotor's own location*, and is independent of where the
observer is. Placing that source at a position and rendering it at an
observation point is then pure free-field propagation:

- **Spherical spreading** (pressure, not intensity): amplitude scales as
  ``ref_distance / r`` where ``r`` is the rotor->observer distance. The absolute
  constant is irrelevant (the emitter's learned gains absorb it); only the
  *relative* attenuation between rotors matters.
- **Propagation delay**: ``tau = r / c`` seconds (``c = 343`` m/s). At 16 kHz one
  sample spans 2.14 cm and rotor ranges are ~0.2-0.4 m, so the sub-sample part of
  the delay is significant. We therefore apply an **exact fractional delay** with
  a Fourier phase ramp (differentiable in both the signal *and* the position, so
  the same model can later drive gradient-based position fitting).

The observed signal is the sum over rotors::

    y[t] = sum_i (ref_distance / r_i) * x_i(t - r_i / c)

The geometry is assumed **static** over a clip (DREGON is a hover rig), so each
rotor->observer delay is a single constant, not a time-varying track.

Design
------
The "emit" and "propagate" stages are decoupled. ``emit`` runs a *single-rotor*
:class:`HarmonicNoiseGenNew` (``n_oscillators=1``) with the rotor axis folded into
the batch, so the four rotors are modelled **independently** (no cross-rotor
source coupling -- a simplification) and a clip's sources are synthesised once
and can be rendered to any number of observation points cheaply.

Isotropic radiation means only the distance ``||rel_pos||`` is used today; the
full 3-vector is accepted and carried so directivity (via the emitter's
``DirectionalOutputHead`` / ``z`` hook) can be added later without an interface
change.
"""

from __future__ import annotations

import torch
from torch import nn

from .harmonic_gen_new import HarmonicNoiseGenNew

SPEED_OF_SOUND = 343.0


# ---------------------------------------------------------------------------
# Propagation primitives
# ---------------------------------------------------------------------------


def fractional_delay(
    signal: torch.Tensor,
    delay_seconds: torch.Tensor,
    sample_rate: float,
) -> torch.Tensor:
    """Delay ``signal`` along its last axis by a (possibly fractional) amount.

    Implemented as a linear-phase ramp in the rfft domain:
    ``X(f) -> X(f) * exp(-j 2 pi f * delay)``. Exact for band-limited signals,
    differentiable w.r.t. both ``signal`` and ``delay_seconds``.

    Args:
        signal: ``[..., T]`` real signal.
        delay_seconds: ``[...]`` delay per leading-dim element, broadcastable
            against ``signal[..., 0]``. Positive = later in time.
        sample_rate: samples per second.

    Returns:
        ``[..., T]`` delayed signal (same shape as ``signal``).

    Note:
        The phase ramp is circular, so energy delayed past the end wraps to the
        start. For the small delays here (a handful of samples over a clip of
        thousands) the wrapped energy is negligible and the STFT magnitude loss
        is computed with centre-padding that further suppresses edge effects.
    """
    n = signal.shape[-1]
    spec = torch.fft.rfft(signal, dim=-1)  # [..., F]
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate, device=signal.device, dtype=signal.dtype)
    # angle: [..., F] = -2 pi f tau
    angle = -2.0 * torch.pi * freqs * delay_seconds.unsqueeze(-1)
    spec = spec * torch.exp(1j * angle.to(spec.real.dtype))
    return torch.fft.irfft(spec, n=n, dim=-1)


def propagate(
    sources: torch.Tensor,
    rel_pos: torch.Tensor,
    *,
    sample_rate: float,
    c: float = SPEED_OF_SOUND,
    ref_distance: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Render per-rotor source waveforms at one or more observation points.

    Native multi-observer. Because propagation is linear, the per-rotor delay,
    attenuation **and the sum over rotors** are all done in the rfft domain
    before a single inverse transform per observer::

        X_r(f) = rfft(source_r)
        Y_m(f) = sum_r  a_{m,r} * X_r(f) * exp(-j 2 pi f tau_{m,r})
        y_m    = irfft(Y_m)

    so ``M`` observers cost ``R`` forward + ``M`` inverse transforms (not
    ``M*R``); the source transforms are shared across all observers. The whole
    thing stays differentiable w.r.t. ``sources`` and ``rel_pos`` (positions).

    Args:
        sources: ``[B, R, T]`` per-rotor source waveform (radiated at the rotor).
        rel_pos: vector(s) from each rotor to the observation point(s):
            - ``[B, R, 3]``        -> single observation point -> returns ``[B, T]``
            - ``[B, M, R, 3]``     -> ``M`` observers          -> returns ``[B, M, T]``
        sample_rate: audio sample rate.
        c: speed of sound (m/s).
        ref_distance: reference distance for the ``1/r`` law (metres).
        eps: distance floor to avoid division by zero at a coincident point.

    Returns:
        ``[B, T]`` (single observer) or ``[B, M, T]`` (``M`` observers).
    """
    if rel_pos.shape[-1] != 3:
        raise ValueError(f"rel_pos last dim must be 3 (xyz), got {tuple(rel_pos.shape)}")

    # Promote a single observer to the M-axis so there is one code path.
    squeeze_m = rel_pos.dim() == 3
    if squeeze_m:
        rel_pos = rel_pos.unsqueeze(1)  # [B, 1, R, 3]
    elif rel_pos.dim() != 4:
        raise ValueError(f"rel_pos must be [B, R, 3] or [B, M, R, 3], got {tuple(rel_pos.shape)}")

    b, r = rel_pos.shape[0], rel_pos.shape[2]
    if sources.shape[:2] != (b, r):
        raise ValueError(
            f"sources must be [B={b}, R={r}, T] to match rel_pos, got {tuple(sources.shape)}"
        )
    t = sources.shape[-1]

    dist = torch.linalg.vector_norm(rel_pos, dim=-1).clamp_min(eps)  # [B, M, R]
    tau = dist / c  # seconds
    amp = ref_distance / dist  # [B, M, R]

    spec = torch.fft.rfft(sources, dim=-1)  # [B, R, F]  -- R forward transforms
    freqs = torch.fft.rfftfreq(t, d=1.0 / sample_rate, device=sources.device, dtype=sources.dtype)

    # Accumulate the rotor sum in the frequency domain. Looping over the (small)
    # rotor axis avoids materialising the full [B, M, R, F] weight tensor; the
    # per-rotor term is only [B, M, F].
    obs = spec.new_zeros((spec.shape[0], rel_pos.shape[1], spec.shape[-1]))  # [B, M, F]
    for ri in range(r):
        # a_{m,r} * exp(-j 2 pi f tau_{m,r}), shape [B, M, F]
        phase = torch.exp((-2j * torch.pi) * freqs * tau[:, :, ri].unsqueeze(-1))
        obs = obs + amp[:, :, ri].unsqueeze(-1).to(phase.dtype) * spec[:, ri].unsqueeze(1) * phase

    out = torch.fft.irfft(obs, n=t, dim=-1)  # [B, M, T]  -- M inverse transforms
    return out.squeeze(1) if squeeze_m else out


# ---------------------------------------------------------------------------
# Position-aware generator
# ---------------------------------------------------------------------------


class PositionalHarmonicNoiseGen(nn.Module):
    """RPS + geometry -> drone noise at an observation point.

    Args:
        emitter: a single-rotor :class:`HarmonicNoiseGenNew`. If ``None``, one is
            built with ``n_oscillators=1`` (the rotor axis is folded into the
            batch in :meth:`emit`). Any extra ``**kwargs`` are forwarded to the
            default emitter / its :class:`JointAmplitudePredictor`.
        n_harmonics: harmonics per rotor (passed to the default emitter).
        sample_rate: audio sample rate.
        speed_of_sound: ``c`` for the delay law (m/s).
        ref_distance: reference distance for the ``1/r`` attenuation (metres).
        eps: distance floor for the attenuation/delay.
    """

    def __init__(
        self,
        emitter: HarmonicNoiseGenNew | None = None,
        *,
        n_harmonics: int = 100,
        sample_rate: int = 16000,
        speed_of_sound: float = SPEED_OF_SOUND,
        ref_distance: float = 1.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        if emitter is None:
            emitter = HarmonicNoiseGenNew(
                n_harmonics=n_harmonics,
                sample_rate=sample_rate,
                n_oscillators=1,
                **kwargs,
            )
        self.emitter = emitter
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self.ref_distance = ref_distance
        self.eps = eps

    def emit(self, rps: torch.Tensor) -> torch.Tensor:
        """Synthesise each rotor's source waveform (radiated at the rotor).

        Args:
            rps: ``[B, R, T]`` per-rotor speed at audio rate (Hz).

        Returns:
            ``[B, R, T]`` per-rotor source waveforms.
        """
        if rps.dim() != 3:
            raise ValueError(f"rps must be [B, R, T], got {tuple(rps.shape)}")
        b, r, t = rps.shape
        folded = rps.reshape(b * r, 1, t)  # rotor axis -> batch, single-rotor net
        src = self.emitter(folded)  # [B*R, T]
        return src.reshape(b, r, t)

    def forward(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        *,
        return_dict: bool = False,
    ):
        """Render drone noise at the observation point(s).

        Args:
            rps: ``[B, R, T]`` per-rotor speed at audio rate (Hz).
            rel_pos: ``[B, R, 3]`` (single point) or ``[B, M, R, 3]`` (M points):
                vector from each rotor to the observation point(s), metres.
            return_dict: if True, also return the per-rotor ``sources``.

        Returns:
            ``[B, T]`` / ``[B, M, T]`` observed signal, or a dict with
            ``{"audio", "sources"}``.
        """
        sources = self.emit(rps)  # [B, R, T]
        audio = propagate(
            sources,
            rel_pos,
            sample_rate=self.sample_rate,
            c=self.speed_of_sound,
            ref_distance=self.ref_distance,
            eps=self.eps,
        )
        if return_dict:
            return {"audio": audio, "sources": sources}
        return audio
