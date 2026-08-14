"""Amplitude-only propagation: geometry gains times a learnable per-mic EQ.

The amplitude-target objective never sees phase, so propagation from a rotor to
a microphone is an exact MULTIPLICATION — no delays, no summation::

    A_obs[r, k, c](t) = A_src[r, k](t) * g_{r,c}(f_k(t))
    g_{r,c}(f)        = (1 / dist_{r,c}) * EQ_c(f)

``1 / dist`` is fixed by the rig geometry
(:func:`models.generative.positional_harmonic_gen.amplitude_gains`). ``EQ_c`` is
this module: one smooth, low-order, LEARNABLE magnitude response per microphone,
shared across rotors, per rig.

Why an EQ, and why it is shared across rotors
---------------------------------------------
A room's reverberant transfer is frequency-structured. A rate-swept harmonic
SAMPLES that structure — the ``k``-th line of a rotor walks over a wide
frequency span as the shaft speeds up and slows down — so the measured envelope
carries an rps-correlated ripple that a plain ``1/r`` law cannot express. The
v1/v2 arms gave each ``(drone, microphone)`` pair one frequency-FLAT scalar,
which is the zeroth order of exactly this curve.

The curve does not know which rotor excited it: room response and capsule
sensitivity are properties of the receiver, so one EQ per (rig, microphone) is
shared across all rotors. It is per RIG because DREGON and Michael's are
different arrays in different rooms.

Delays are absent for the same reason the rotor sum is: a delay only rotates
phase, and each ``(rotor, harmonic)`` line is demodulated on its own, so no
coherent summation exists to model.

The parameterization
--------------------
``n_knots`` (<= 16) control points, equally spaced in **log frequency** between
``f_min`` and ``f_max``, holding the LOG gain; the response between knots is
linear in log-f. Low order is the smoothness prior: 16 knots over 20 Hz..8 kHz
is about 2.6 knots per octave, which cannot chase a single harmonic. Zero init
is unity gain, so an untrained EQ is the plain ``1/r`` law and the arm starts
where the v2 arms did.

The knot curve is EMITTED as a prediction entry (``mic_eq``), so a curvature
penalty is an ordinary composite-loss term (``losses.SmoothnessPenalty`` on the
knot axis) instead of a second regularization mechanism inside the model.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["MicEQ"]


class MicEQ(nn.Module):
    """Learnable smooth per-(rig, microphone) magnitude response ``EQ_c(f)``.

    Args:
        rigs: rig names — the SAME keys the ``DroneCodebook`` uses
            (``dregon`` / ``michaels``), because a batch already carries a rig id
            per sample and the propagation head is selected by it.
        n_mics: microphones per rig (8 on both arrays). A batch with fewer
            observers takes the leading slice, exactly as the per-mic gains do.
        n_knots: control points, spaced equally in log-f. Keep it low — this is
            the smoothness prior.
        f_min / f_max: the log-f knot span, in Hz. Frequencies outside it are
            clamped to the end knots (the response is held, never extrapolated).
    """

    def __init__(
        self,
        rigs: list[str] | tuple[str, ...],
        n_mics: int,
        *,
        n_knots: int = 16,
        f_min: float = 20.0,
        f_max: float = 8000.0,
    ) -> None:
        super().__init__()
        if int(n_knots) < 2:
            raise ValueError(f"n_knots must be at least 2 (a curve needs two ends), got {n_knots}")
        if not (0.0 < float(f_min) < float(f_max)):
            raise ValueError(f"need 0 < f_min < f_max, got {f_min} and {f_max}")
        rig_list = [self._key(r) for r in rigs]
        if not rig_list:
            raise ValueError("MicEQ needs at least one rig name")
        self.n_mics = int(n_mics)
        self.n_knots = int(n_knots)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self._log_f_min = math.log(self.f_min)
        self._log_f_span = math.log(self.f_max) - self._log_f_min
        # Zero = unity gain: an untrained EQ is the plain 1/r law.
        self.log_eq = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(self.n_mics, self.n_knots)) for k in rig_list}
        )

    @staticmethod
    def _key(name: str) -> str:
        key = str(name)
        if "." in key:
            raise ValueError(f"rig name must not contain '.': {name!r}")
        if not key:
            raise ValueError("rig name must be a non-empty string")
        return key

    def names(self) -> list[str]:
        return list(self.log_eq.keys())

    def knot_freqs(self) -> torch.Tensor:
        """``[n_knots]`` knot frequencies in Hz (log-spaced)."""
        u = torch.arange(self.n_knots, dtype=torch.float32) / (self.n_knots - 1)
        return torch.exp(self._log_f_min + u * self._log_f_span)

    def curve(self, rigs: list[str] | tuple[str, ...], n_mics: int | None = None) -> torch.Tensor:
        """``[B, M, n_knots]`` LOG-gain knots, one row per sample's rig."""
        missing = [r for r in rigs if self._key(r) not in self.log_eq]
        if missing:
            raise KeyError(f"unknown rig(s) {sorted(set(missing))}; known: {self.names()}")
        m = self.n_mics if n_mics is None else int(n_mics)
        if m > self.n_mics:
            raise ValueError(f"MicEQ was built for {self.n_mics} microphones, batch has {m}")
        return torch.stack([self.log_eq[self._key(r)][:m] for r in rigs], dim=0)

    def log_gain(self, freq: torch.Tensor, rigs: list[str] | tuple[str, ...], **kw) -> torch.Tensor:
        """``[B, ...]`` frequencies (Hz) -> ``[B, M, ...]`` log gain.

        The leading axis of ``freq`` is the batch; every other axis is carried
        through untouched, so ``[B, R, H, t]`` harmonic frequencies give
        ``[B, M, R, H, t]`` gains — the shape the amplitude bank needs.
        """
        knots = self.curve(rigs, **kw)  # [B, M, K]
        if freq.shape[0] != knots.shape[0]:
            raise ValueError(
                f"freq batch {freq.shape[0]} disagrees with {knots.shape[0]} rig names"
            )
        b, m, k = knots.shape
        flat = freq.reshape(b, -1).to(knots.dtype)
        # Clamp INTO the knot span first: f=0 (a stopped rotor) has no log, and
        # the response is held rather than extrapolated outside [f_min, f_max].
        u = torch.log(flat.clamp(self.f_min, self.f_max)) - self._log_f_min
        u = (u / self._log_f_span) * (k - 1)
        i0 = u.floor().clamp(0.0, float(k - 2)).long()  # [B, N]
        w = (u - i0.to(u.dtype)).unsqueeze(1)  # [B, 1, N]
        idx = i0.unsqueeze(1).expand(-1, m, -1)  # [B, M, N]
        g0 = knots.gather(2, idx)
        g1 = knots.gather(2, idx + 1)
        out = g0 + (g1 - g0) * w
        return out.reshape(b, m, *freq.shape[1:])

    def gain(self, freq: torch.Tensor, rigs: list[str] | tuple[str, ...], **kw) -> torch.Tensor:
        """``exp`` of :meth:`log_gain` — the multiplicative amplitude weight."""
        return torch.exp(self.log_gain(freq, rigs, **kw))

    def filter_audio(
        self, audio: torch.Tensor, rigs: list[str] | tuple[str, ...], sample_rate: float
    ) -> torch.Tensor:
        """Apply the same EQ to a rendered ``[B, M, T]`` waveform.

        The amplitude path multiplies each line by ``EQ_c(f_k(t))``; the
        RENDERING path must apply the identical response, or a checkpoint would
        render at a different spectrum than it was fit to. A zero-phase
        magnitude multiply in the rfft domain is that response — it is exactly
        what the amplitude path does to every line, evaluated on the analysis
        grid instead of on the lines.
        """
        if audio.dim() != 3:
            raise ValueError(f"audio must be [B, M, T], got {tuple(audio.shape)}")
        b, m, t = audio.shape
        freqs = torch.fft.rfftfreq(t, d=1.0 / float(sample_rate), device=audio.device)
        # log_gain wants a batched frequency tensor; the grid is shared.
        grid = freqs.unsqueeze(0).expand(b, -1)  # [B, F]
        gain = self.gain(grid, rigs, n_mics=m).to(audio.dtype)  # [B, M, F]
        spec = torch.fft.rfft(audio, dim=-1) * gain
        return torch.fft.irfft(spec, n=t, dim=-1)
