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

from typing import Literal, overload

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

    Per-drone conditioning is **external**: the model takes a conditioning code
    ``z`` ``(B, d)`` as an *input* (alongside ``rps`` and the geometry), exactly
    as it takes positions. It does not own a per-drone table — the ``id -> z``
    map lives in a separate :class:`tasks.noise_generation.DroneCodebook`. This
    keeps the model's parameter shape fixed regardless of how many drones exist,
    and enables few-shot adaptation to an unseen drone by freezing the model and
    optimising only a fresh code. ``d`` *is* architectural (it sizes the FiLM
    generator), so the model owns it; the number of drones ``K`` is not.

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
        cond_dim: conditioning-code dimension ``d``. ``0`` disables per-drone
            conditioning (single-drone); ``> 0`` FiLM-conditions the emitter on
            the externally supplied ``z`` ``(B, cond_dim)``.
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
        cond_dim: int = 0,
        silence_fade_rps: float = 10.0,
        **kwargs,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        # Below `silence_fade_rps` the per-rotor emitted source is smoothly faded
        # to zero (reaching exact silence at rps=0), so a stopped rotor makes no
        # sound. Without this the harmonic bank collapses to a DC pedestal at
        # rps=0 (frozen phase => sin(phi)=const), which is unphysical and taught
        # RPS predictors that silence != zero RPS. 0 disables the fade.
        self.silence_fade_rps = float(silence_fade_rps)
        if emitter is None:
            emitter_kwargs: dict = dict(
                n_harmonics=n_harmonics, sample_rate=sample_rate, n_oscillators=1, **kwargs
            )
            if cond_dim > 0:
                # FiLM-condition the emitter on the external code z (B, cond_dim).
                emitter_kwargs.update(use_z=True, z_dim=cond_dim, film=True)
            emitter = HarmonicNoiseGenNew(**emitter_kwargs)
        self.emitter = emitter
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self.ref_distance = ref_distance
        self.eps = eps

    @overload
    def emit(
        self,
        rps: torch.Tensor,
        z: torch.Tensor | None = ...,
        *,
        initial_phases: torch.Tensor | None = ...,
        return_dict: Literal[False] = ...,
        rps_jitter: bool | None = ...,
        rps_jitter_sigma: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    @overload
    def emit(
        self,
        rps: torch.Tensor,
        z: torch.Tensor | None = ...,
        *,
        initial_phases: torch.Tensor | None = ...,
        return_dict: Literal[True],
        rps_jitter: bool | None = ...,
        rps_jitter_sigma: torch.Tensor | None = ...,
    ) -> dict[str, torch.Tensor]: ...
    def emit(
        self,
        rps: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        initial_phases: torch.Tensor | None = None,
        return_dict: bool = False,
        rps_jitter: bool | None = None,
        rps_jitter_sigma: torch.Tensor | None = None,
    ):
        """Synthesise each rotor's source waveform (radiated at the rotor).

        Args:
            rps: ``[B, R, T]`` per-rotor speed at audio rate (Hz).
            z: ``[B, d]`` optional per-clip conditioning (drone embedding). The
                same vector conditions every rotor of a clip, so it is repeated
                across the folded rotor axis.
            initial_phases: optional ``[B, R, H]`` per-rotor per-harmonic initial
                phase offsets (radians). ``None`` (default) => the emitter samples
                random phases while training and uses zero phases at eval. Provide
                a tensor to pin phases (e.g. reproducible inference).
            return_dict: if True, also return the emitter's per-rotor control
                curves (``harm_amps``, ``noise_amps``) with the rotor axis
                restored — these are what the smoothness regularisers act on.

        Returns:
            ``[B, R, T]`` per-rotor source waveforms, or (if ``return_dict``) a
            dict ``{"sources", "harm_amps", "noise_amps"}`` where
            ``harm_amps`` is ``[B, R, O, H, t_a]`` and ``noise_amps`` is
            ``[B, R, F, t_n]``.
        """
        if rps.dim() != 3:
            raise ValueError(f"rps must be [B, R, T], got {tuple(rps.shape)}")
        b, r, t = rps.shape
        folded = rps.reshape(b * r, 1, t)  # rotor axis -> batch, single-rotor net
        # z may be per-clip ``[B, d]`` (same code for every rotor — the default) or
        # per-rotor ``[B, R, d]`` (per-rotor sub-embeddings ``z_r = z_drone + δz_r``):
        # the former is broadcast across the folded rotor axis, the latter folds
        # its own axis in directly so each rotor's emitter sees its own code.
        if z is None:
            z_folded = None
        elif z.dim() == 3:
            if z.shape[:2] != (b, r):
                raise ValueError(f"per-rotor z must be [B={b}, R={r}, d], got {tuple(z.shape)}")
            z_folded = z.reshape(b * r, z.shape[-1])  # [B*R, d]
        else:
            z_folded = z.repeat_interleave(r, dim=0)  # [B, d] -> [B*R, d]
        # A per-clip (per-drone) sigma broadcasts to every rotor of the clip, so
        # it folds into the batch exactly like z: [B] -> [B*R].
        sigma_folded = (
            rps_jitter_sigma.repeat_interleave(r, dim=0) if rps_jitter_sigma is not None else None
        )
        # Fold rotor into batch and add the single-oscillator axis: [B,R,H] ->
        # [B*R, 1, H], matching the emitter's freqs [B*R, O=1, H, t].
        ip_folded = None
        if initial_phases is not None:
            if initial_phases.shape[:2] != (b, r):
                raise ValueError(
                    f"initial_phases must be [B={b}, R={r}, H], got {tuple(initial_phases.shape)}"
                )
            ip_folded = initial_phases.reshape(b * r, 1, initial_phases.shape[-1])
        if not return_dict:
            src = self.emitter(
                folded,
                z=z_folded,
                initial_phases=ip_folded,
                rps_jitter=rps_jitter,
                rps_jitter_sigma=sigma_folded,
            )  # [B*R, T]
            return src.reshape(b, r, t)
        out = self.emitter(
            folded,
            z=z_folded,
            initial_phases=ip_folded,
            return_dict=True,
            rps_jitter=rps_jitter,
            rps_jitter_sigma=sigma_folded,
        )
        harm_amps = out["harm_amps"]  # [B*R, O, H, t_a]
        noise_amps = out["noise_amps"]  # [B*R, F, t_n]
        return {
            "sources": out["audio"].reshape(b, r, t),
            # The deterministic half of each rotor's source, kept separate so
            # `spectral_stats` can propagate it as the distribution's mean.
            "coherent": out["coherent"].reshape(b, r, t),
            "harm_amps": harm_amps.reshape(b, r, *harm_amps.shape[1:]),
            "noise_amps": noise_amps.reshape(b, r, *noise_amps.shape[1:]),
        }

    def forward(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        initial_phases: torch.Tensor | None = None,
        return_dict: bool = False,
        rps_jitter: bool | None = None,
        rps_jitter_sigma: torch.Tensor | None = None,
    ):
        """Render drone noise at the observation point(s).

        Args:
            rps: ``[B, R, T]`` per-rotor speed at audio rate (Hz).
            rel_pos: ``[B, R, 3]`` (single point) or ``[B, M, R, 3]`` (M points):
                vector from each rotor to the observation point(s), metres.
            z: ``[B, cond_dim]`` external per-drone conditioning code (from a
                :class:`tasks.noise_generation.DroneCodebook`). Required iff the
                model was built with ``cond_dim > 0``; ignored otherwise.
            initial_phases: optional ``[B, R, H]`` per-rotor per-harmonic initial
                phases (radians). ``None`` (default) => random phases while
                training, zero phases at eval. See :meth:`emit`.
            return_dict: if True, also return the per-rotor ``sources`` and the
                emitter control curves (``harm_amps``, ``noise_amps``) that the
                smoothness regularisers act on.

        Returns:
            ``[B, T]`` / ``[B, M, T]`` observed signal, or a dict with
            ``{"audio", "sources", "harm_amps", "noise_amps"}``.
        """
        if self.cond_dim > 0:
            if z is None:
                raise ValueError("model built with cond_dim>0 requires a conditioning code z")
            if z.shape[-1] != self.cond_dim:
                raise ValueError(
                    f"z last dim must be cond_dim={self.cond_dim}, got {tuple(z.shape)}"
                )
        else:
            z = None  # unconditioned: ignore any code passed in

        if return_dict:
            emitted = self.emit(
                rps,
                z=z,
                initial_phases=initial_phases,
                return_dict=True,
                rps_jitter=rps_jitter,
                rps_jitter_sigma=rps_jitter_sigma,
            )
            sources = emitted["sources"]  # [B, R, T]
        else:
            sources = self.emit(
                rps,
                z=z,
                initial_phases=initial_phases,
                rps_jitter=rps_jitter,
                rps_jitter_sigma=rps_jitter_sigma,
            )  # [B, R, T]
        # Smooth silence fade: per-rotor source -> 0 as rps -> 0, full amplitude
        # by `silence_fade_rps` (smoothstep). Emitter-level, so every component
        # (harmonic bank + broadband residual) vanishes when a rotor is stopped.
        if self.silence_fade_rps > 0.0:
            g = (rps / self.silence_fade_rps).clamp(0.0, 1.0)
            gate = g * g * (3.0 - 2.0 * g)  # smoothstep(0,1): C1, 0 at 0, 1 at fade_rps
            sources = sources * gate
        audio = propagate(
            sources,
            rel_pos,
            sample_rate=self.sample_rate,
            c=self.speed_of_sound,
            ref_distance=self.ref_distance,
            eps=self.eps,
        )
        if return_dict:
            return {
                "audio": audio,
                "sources": sources,
                "harm_amps": emitted["harm_amps"],
                "noise_amps": emitted["noise_amps"],
            }
        return audio

    def spectral_stats(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        initial_phases: torch.Tensor | None = None,
        rps_jitter: bool | None = None,
        rps_jitter_sigma: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict the observed field as a *distribution*, not a realization.

        Returns the two quantities :mod:`losses.spectral_likelihood` needs:

        - ``coherent`` ``[B, M, T]`` — the harmonic bank propagated to each
          microphone. Deterministic given the rotor phases, so it is the mean of
          the complex spectrum.
        - ``noise_psd`` ``[B, M, t_n, F]`` — the **power** spectral envelope of
          the *stochastic* broadband branch at each microphone. Never sampled.

        Power, not magnitude, is the interface on purpose. The loss needs
        ``sigma2``, so returning a magnitude would mean ``sqrt`` here and a
        square there — analytically the identity, but autograd evaluates it
        stepwise and ``d(sqrt)/dx`` is infinite at ``x = 0`` while
        ``d(x^2)/dx`` is zero, giving ``inf * 0 = NaN`` in every bin where the
        branch predicts exact silence (which it does, in ~12% of them).

        The broadband residual is drawn independently per rotor, so propagation
        adds **powers** rather than amplitudes: with the ``1/r`` weights
        ``a_{m,r}`` of :func:`propagate`, the observed noise power at microphone
        ``m`` is ``sum_r a_{m,r}^2 * noise_amps_r(f)^2``. The per-rotor delays
        are irrelevant here — they rotate phase, and phase carries no power.
        """
        if self.cond_dim > 0:
            if z is None:
                raise ValueError("model built with cond_dim>0 requires a conditioning code z")
        else:
            z = None

        emitted = self.emit(
            rps,
            z=z,
            initial_phases=initial_phases,
            return_dict=True,
            rps_jitter=rps_jitter,
            rps_jitter_sigma=rps_jitter_sigma,
        )
        coherent_src = emitted["coherent"]  # [B, R, T]
        noise_amps = emitted["noise_amps"]  # [B, R, F, t_n]

        if self.silence_fade_rps > 0.0:
            g = (rps / self.silence_fade_rps).clamp(0.0, 1.0)
            gate = g * g * (3.0 - 2.0 * g)  # [B, R, T]
            coherent_src = coherent_src * gate
            # The same gate multiplies the broadband branch; at the noise-frame
            # rate it scales POWER by gate^2, so pool the gate then square it.
            gate_n = torch.nn.functional.adaptive_avg_pool1d(gate, noise_amps.shape[-1])
            noise_amps = noise_amps * gate_n.unsqueeze(-2)

        coherent = propagate(
            coherent_src,
            rel_pos,
            sample_rate=self.sample_rate,
            c=self.speed_of_sound,
            ref_distance=self.ref_distance,
            eps=self.eps,
        )

        single = rel_pos.dim() == 3
        rp = rel_pos.unsqueeze(1) if single else rel_pos  # [B, M, R, 3]
        dist = torch.linalg.vector_norm(rp, dim=-1).clamp_min(self.eps)  # [B, M, R]
        amp2 = (self.ref_distance / dist).pow(2)  # [B, M, R]
        # [B, M, R] x [B, R, F, t] -> [B, M, F, t] power sum over rotors.
        power = torch.einsum("bmr,brft->bmft", amp2, noise_amps.pow(2))
        noise_psd = power.clamp_min(0.0).transpose(-1, -2)  # [B, M, t_n, F]
        if single:
            coherent = coherent if coherent.dim() == 3 else coherent.unsqueeze(1)
        return {"coherent": coherent, "noise_psd": noise_psd}

    def spatial_stats(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        n_fft: int = 1024,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Per-rotor **emitted** power, for the spatial (cross-mic) likelihood.

        :meth:`spectral_stats` sums the rotors' contributions at each microphone,
        which is right for a per-microphone marginal likelihood and destroys
        exactly the structure :mod:`losses.spatial_likelihood` needs: there, the
        rotors enter through *known* steering vectors as a rank-R term and the
        wind as a diagonal one, and that is what makes them separable.

        The returned power is the source **at the rotor**, before propagation —
        the loss applies the 1/r and delay itself via the steering vectors, so
        the two must not both apply it.

        The deterministic half is taken as ``|STFT|^2`` of the harmonic bank
        (exact, and it carries the comb structure a coarse envelope would lose);
        the stochastic half is the analytic broadband envelope. Mixing the two
        this way keeps the single-realization problem out of the random part
        while staying exact on the part that has no randomness.

        Returns ``{"source_psd": [B, R, t, F], "rel_pos": [B, M, R, 3]}``.
        """
        if self.cond_dim == 0:
            z = None
        emitted = self.emit(rps, z=z, return_dict=True, **kwargs)
        coherent_src = emitted["coherent"]  # [B, R, T]
        noise_amps = emitted["noise_amps"]  # [B, R, F_g, t_n]

        if self.silence_fade_rps > 0.0:
            g = (rps / self.silence_fade_rps).clamp(0.0, 1.0)
            gate = g * g * (3.0 - 2.0 * g)
            coherent_src = coherent_src * gate
            gate_n = torch.nn.functional.adaptive_avg_pool1d(gate, noise_amps.shape[-1])
            noise_amps = noise_amps * gate_n.unsqueeze(-2)

        b, r, t = coherent_src.shape
        window = torch.hann_window(n_fft, device=coherent_src.device, dtype=coherent_src.dtype)
        spec = torch.stft(
            coherent_src.reshape(b * r, t),
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
            center=True,
        )
        harm_psd = spec.abs().pow(2) / window.pow(2).sum()  # [B*R, F, N]
        harm_psd = harm_psd.reshape(b, r, harm_psd.shape[-2], harm_psd.shape[-1])

        # Put the analytic broadband envelope on the same (frame, freq) grid.
        broadband = torch.nn.functional.interpolate(
            noise_amps.pow(2).reshape(b * r, 1, noise_amps.shape[-2], noise_amps.shape[-1]),
            size=(harm_psd.shape[-2], harm_psd.shape[-1]),
            mode="bilinear",
            align_corners=True,
        ).reshape(b, r, harm_psd.shape[-2], harm_psd.shape[-1])

        source = (harm_psd + broadband).clamp_min(0.0).transpose(-1, -2)  # [B, R, N, F]
        single = rel_pos.dim() == 3
        rp = rel_pos.unsqueeze(1) if single else rel_pos
        # A superset of `spectral_stats`: the separated powers the spatial loss
        # needs, PLUS the mic-level mean/variance so the audio-based metrics
        # still have something to score without a second forward pass.
        out = {"source_psd": source, "rel_pos": rp}
        out.update(self.spectral_stats(rps, rel_pos, z=z, **kwargs))
        return out

