"""
Variable-Phasor (VP) transform — full port of the VP machinery from
`drone_audition.models.harmonic_gen_new`.

The VP-transform projects audio onto a basis of windowed harmonic phasors
(whose instantaneous freqs are derived from a fundamental-frequency series).
There are two projection flavours:

- `VP_transform`         : per-frame dot product onto the (normalised) phasors.
- `lstsq_VP_transform`   : least-squares solution onto the I/Q phasor plane.

A learnable network can operate on the projections; `inverse_VP_transform`
rebuilds audio from processed projections.

Note: the inverse is NOT a true inverse — the variable-phasor matrix is not
square, so reconstruction matches the original only when the signal lies
entirely within the phasors' hyperplane.

Self-contained — no dependency on `env.settings`; sample rate is an explicit
argument (default 16 kHz to match this repo).
"""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def harmonic_freq_series(freq: torch.Tensor, n_harmonics: int):
    """Make N harmonics from a fundamental-frequency series.

    Args:
        freq: [..., T] instantaneous fundamental frequency (Hz)
        n_harmonics: number of harmonics including the fundamental
    Returns:
        [..., N, T] frequencies of each harmonic per time step
    """
    coeffs = torch.arange(1, n_harmonics + 1, device=freq.device, dtype=freq.dtype)
    return torch.matmul(coeffs.unsqueeze(-1), freq.unsqueeze(-2))


def remove_above_nyquist(freqs, amps, sr: int):
    """Zero amplitudes at time steps where frequency exceeds Nyquist."""
    assert freqs.shape == amps.shape
    return torch.where(freqs > sr / 2, torch.zeros_like(amps), amps)


def freqs_to_phasors(freq: torch.Tensor, sr: int):
    """Convert frequency series to rotating complex phasors (cumprod form).

    ``torch.polar(1, phase_diff)`` builds each per-sample unit phasor
    ``cos(phi) + i sin(phi)`` directly; it is **bit-identical** to the former
    ``torch.exp(1j * phase_diff)`` (``exp`` of a purely imaginary number is that
    same cos/sin pair) but ~2x cheaper on CPU/GPU, and it feeds the *same*
    ``cumprod`` so every downstream consumer (oscillator bank **and** the VP
    transform below) is numerically unchanged.
    """
    phase_diff = freq * 2 * torch.pi / sr
    complex_diffs = torch.polar(torch.ones_like(phase_diff), phase_diff)
    return torch.cumprod(complex_diffs, -1)


def overlap_and_add(signal: torch.Tensor, frame_step: int):
    """Reimplementation of tf.signal.overlap_and_add using F.fold."""
    frames, frame_length = signal.shape[-2:]
    batch_dims = signal.shape[:-2]
    batch_dim_d = {f"b{i}": d for i, d in enumerate(batch_dims)}
    batch_dim_s = " ".join(batch_dim_d.keys())

    output_size = (frames - 1) * frame_step + frame_length

    result = F.fold(
        rearrange(signal, f"{batch_dim_s} t f -> ({batch_dim_s}) f t"),
        (output_size, 1),
        kernel_size=(frame_length, 1),
        stride=(frame_step, 1),
    )
    return rearrange(result, f"({batch_dim_s}) 1 t 1 -> {batch_dim_s} t", **batch_dim_d)


Solution = namedtuple("Solution", "solution")


# ---------------------------------------------------------------------------
# Least-squares helpers
# ---------------------------------------------------------------------------


def iterative_lstsq_minimize(A, B, method="l-bfgs"):
    """Solve LSTSQ on GPU via L-BFGS (correct but slow and memory-hungry).

    Requires the optional `pytorch-minimize` (`torchmin`) package — imported
    lazily so the rest of the module works without it.
    """
    from torchmin import minimize

    def f(X):
        return torch.linalg.matrix_norm(A @ X - B).mean()

    result = minimize(f, A.adjoint() @ B, method=method)
    return Solution(result.x)


def good_lstsq(A, B, retry=0):
    """LSTSQ that picks a driver that actually works on the given device.

    `gelsd` is robust but CPU-only; on CUDA we fall back to `gels` (with a
    single retry on transient failures).
    """
    assert A.device == B.device

    if A.device.type == "cpu":
        return torch.linalg.lstsq(A, B, driver="gelsd")
    else:
        # gelsd does not work on CUDA; gels is the usable driver there.
        try:
            return torch.linalg.lstsq(A, B, driver="gels")
        except Exception:
            if retry > 0:
                raise
            return good_lstsq(A, B, retry=retry + 1)


# ---------------------------------------------------------------------------
# Core VP primitives
# ---------------------------------------------------------------------------


def _framed_phasors(freqs, frame_len, hop_len, antialias=True, center=False, sr: int = 16000):
    phasors = freqs_to_phasors(freqs, sr=sr)
    if center:
        phasors = F.pad(phasors, (frame_len // 2, frame_len // 2))

    phasors_unfold = phasors.unfold(-1, frame_len, hop_len)

    window = torch.hann_window(frame_len, device=phasors_unfold.device)
    phasors_unfold = phasors_unfold * window

    # normalize (guard against all-zero frames -> division by zero / NaN)
    ph_norm = torch.linalg.vector_norm(phasors_unfold, dim=-1, keepdim=True)
    ph_norm = torch.where(ph_norm == 0, torch.ones_like(ph_norm), ph_norm)
    phasors_framed = phasors_unfold / ph_norm

    if antialias:
        if center:
            freqs = F.pad(freqs, (frame_len // 2, frame_len // 2))
        phasors_framed = remove_above_nyquist(
            freqs.unfold(-1, frame_len, hop_len), phasors_framed, sr=sr
        )

    return phasors_framed


def _apply_phasors_to_wav(framed_phasors, wav, window_len, hop_len):
    wav_unfold = rearrange(wav.unfold(-1, window_len, hop_len), "... t k -> ... 1 t k")
    return (framed_phasors * wav_unfold).sum(-1)


def _solve_phasors_lstsq(framed_phasors, wav, window_len, hop_len, method: str = "lstsq"):
    c = framed_phasors.shape[-4]
    sincos = torch.view_as_real(framed_phasors)

    sincos_rearranged = rearrange(sincos, "... c h t w k -> ... t w (c h k)")
    win = torch.hann_window(window_len, device=wav.device)
    windowed_audio = wav.unfold(-1, window_len, hop_len) * win

    audio_rearranged = rearrange(windowed_audio, "... t w -> ... t w 1")

    if method in ("gelsy", "normal"):
        # Pivoted-QR driver: ~100x faster than gelsd on CPU for the typical
        # (frames, window, tracks) batch AND robust to rank deficiency
        # (duplicate / near-coincident harmonic tracks), unlike the previous
        # ridge normal-equations path ("normal" is kept as a deprecated alias)
        # whose torch.linalg.solve crashed on exactly-collinear tracks.
        if sincos_rearranged.device.type == "cpu":
            V = torch.linalg.lstsq(sincos_rearranged, audio_rearranged, driver="gelsy").solution
        else:
            V = good_lstsq(sincos_rearranged, audio_rearranged).solution
    elif method == "lstsq":
        V = good_lstsq(sincos_rearranged, audio_rearranged).solution
    else:
        raise ValueError(f"unknown lstsq method: {method!r}")
    V = torch.nan_to_num(V, 0.0)
    V = torch.view_as_complex(rearrange(V, "... t (c h k) 1 -> ... c h t k", c=c, k=2))
    return V / 1.5


def _reconstruct_wav_from_phasors(framed_phasors, V, window_len, hop_len, center=False):
    wav_unfold = (V.unsqueeze(-1) * framed_phasors.conj()).sum(-3).real
    wav_added = overlap_and_add(wav_unfold, hop_len)

    # The scale is the area under the Hann window: (hop_len * 3) / window_len.
    wav = wav_added * (hop_len * 3) / window_len
    if center:
        pd = window_len // 2
        wav = wav[..., pd:-pd]

    return wav


# ---------------------------------------------------------------------------
# Public VP transforms
# ---------------------------------------------------------------------------


def VP_transform(
    freqs, wav, window_len=2048, hop_len=512, center=False, antialias=True, sr: int = 16000
):
    """Convolve the wav with variable harmonic phasors (per-frame dot product)."""
    assert freqs.shape[-1] == wav.shape[-1]

    if center:
        pd = (window_len // 2, window_len // 2)
        wav = F.pad(wav, pd, "constant", 0)

    phasors_framed = _framed_phasors(
        freqs, window_len, hop_len, center=center, antialias=antialias, sr=sr
    )
    return _apply_phasors_to_wav(phasors_framed, wav, window_len, hop_len)


def lstsq_VP_transform(
    freqs,
    wav,
    window_len=2048,
    hop_len=512,
    center=False,
    antialias=True,
    sr: int = 16000,
    method: str = "lstsq",
):
    """Project the wav onto the variable I/Q phasor planes via least squares.

    ``method="gelsy"`` uses the pivoted-QR driver instead of the default SVD
    ones — much faster on CPU and robust to duplicate/collinear tracks; use
    for metrics/analysis loops. (``"normal"`` is a deprecated alias.)
    """
    if center:
        pd = (window_len // 2, window_len // 2)
        wav = F.pad(wav, pd, "constant", 0)

    phasors_framed = _framed_phasors(
        freqs, window_len, hop_len, center=center, antialias=antialias, sr=sr
    )
    return _solve_phasors_lstsq(phasors_framed, wav, window_len, hop_len, method=method)


def inverse_VP_transform(
    freqs, V, window_len=2048, hop_len=512, center=False, antialias=True, sr: int = 16000
):
    """Reconstruct audio from phasor projections.

    NOT a true inverse — see module docstring. Reconstructs the oscillator
    banks with the given (complex) phase corrections.
    """
    phasors_framed = _framed_phasors(
        freqs, window_len, hop_len, center=center, antialias=antialias, sr=sr
    )
    wav = _reconstruct_wav_from_phasors(phasors_framed, V, window_len, hop_len)
    if center:
        wav = wav[..., window_len // 2 : -(window_len // 2)]
    return wav


def harmonic_VP_transform(
    ms, wav, n_harmonics=100, neighboring_freqs=0, window_len=2048, sr: int = 16000, **kwargs
):
    freqs = harmonic_freq_series(ms, n_harmonics)
    if neighboring_freqs > 0:
        freq_step = sr / window_len
        freqs_new = torch.stack(
            [freqs + i * freq_step for i in range(-neighboring_freqs, neighboring_freqs + 1)],
            -1,
        )
        freqs = rearrange(freqs_new, "... o h t k -> ... o (h k) t")

    return VP_transform(freqs, wav, window_len=window_len, sr=sr, **kwargs)


def harmonic_lstsq_VP_transform(ms, wav, n_harmonics=100, **kwargs):
    freqs = harmonic_freq_series(ms, n_harmonics)
    return lstsq_VP_transform(freqs, wav, **kwargs)


def inverse_harmonic_VP_transform(ms, V, n_harmonics=100, **kwargs):
    freqs = harmonic_freq_series(ms, n_harmonics)
    return inverse_VP_transform(freqs, V, **kwargs)


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------


class HarmonicTransformModule(nn.Module):
    """Audio + f0 series -> VP-transform -> net -> inverse VP -> audio.

    Motor speeds + audio -> VP transform -> projections -> deep network ->
    projections -> inverse VP -> audio. Useful for harmonics-aware filtering /
    enhancement.

    The wrapped `net` is called with `return_dict=True` and must return a dict
    containing a `'V'` key holding the processed projections.
    """

    def __init__(
        self,
        net: nn.Module,
        use_lstsq: bool = False,
        antialias: bool = True,
        center: bool = False,
        provide_phasors_ms: bool = False,
        n_harmonics: int = 100,
        window_len: int = 2048,
        hop_len: int = 512,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.window_len = window_len
        self.hop_len = hop_len
        self.sample_rate = sample_rate
        self.use_lstsq = use_lstsq
        self.antialias = antialias
        self.center = center
        self.provide_phasors_ms = provide_phasors_ms

        self.net = net

    def forward(self, audio, f0s, return_dict=False):
        freqs = harmonic_freq_series(f0s, self.n_harmonics)

        phasors_framed = _framed_phasors(
            freqs,
            self.window_len,
            self.hop_len,
            sr=self.sample_rate,
            center=self.center,
            antialias=self.antialias,
        )

        if self.center:
            pd = (self.window_len // 2, self.window_len // 2)
            audio = F.pad(audio, pd, "constant", 0)

        if self.use_lstsq:
            V = _solve_phasors_lstsq(phasors_framed, audio, self.window_len, self.hop_len)
        else:
            V = _apply_phasors_to_wav(phasors_framed, audio, self.window_len, self.hop_len)

        if self.provide_phasors_ms:
            results = self.net(V, f0s, phasors_framed, return_dict=True)
        else:
            results = self.net(V, return_dict=True)

        V_processed = results["V"]

        components = _reconstruct_wav_from_phasors(
            phasors_framed, V_processed, self.window_len, self.hop_len, center=self.center
        )

        wav = components.sum(-2)

        if not return_dict:
            return wav

        return {
            **results,
            "components": components,
            "wav": wav,
        }
