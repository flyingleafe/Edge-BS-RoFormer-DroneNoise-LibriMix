"""
Variable-Phasor (VP) transform module.

Ported from drone_audition.models.harmonic_gen_new. The VP-transform projects
audio onto a basis of windowed harmonic phasors (whose instantaneous freqs
are derived from a fundamental-frequency series). A learnable network operates
on those projections; the inverse rebuilds audio from processed projections.

Note: the inverse is NOT a true inverse — it reconstructs the projection of the
signal onto the harmonic hyperplane. Useful for analysis/synthesis pipelines.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .dsp import freqs_to_phasors, harmonic_freq_series, remove_above_nyquist
from .math_utils import overlap_and_add


def _windowed_phasors(freqs, window_len: int, hop_len: int, sr: int):
    phasors = freqs_to_phasors(freqs, sr=sr)
    # Mask harmonics above Nyquist; phasors are complex but the helper takes any tensor.
    phasors = remove_above_nyquist(freqs, phasors, sr=sr)
    phasors_unfold = F.unfold(
        phasors.unsqueeze(-1), kernel_size=(window_len, 1), stride=(hop_len, 1)
    )
    c = phasors.shape[-2]
    phasors_unfold = rearrange(phasors_unfold, "b (c k) t -> b c t k", c=c)
    window = torch.hann_window(window_len, device=phasors_unfold.device)
    phasors_windowed = phasors_unfold * rearrange(window, "k -> 1 1 1 k")
    phasors_windowed = phasors_windowed / torch.linalg.vector_norm(
        phasors_windowed, dim=-1, keepdim=True
    )
    return phasors_windowed


def _apply_phasors_to_wav(phasors_windowed, wav, window_len, hop_len):
    wav_unfold = F.unfold(
        rearrange(wav, "b t -> b 1 t 1"), kernel_size=(window_len, 1), stride=(hop_len, 1)
    )
    wav_unfold = rearrange(wav_unfold, "b (c k) t -> b c t k", c=1)
    return (phasors_windowed * wav_unfold).sum(-1)


def _reconstruct_wav_from_phasors(phasors_windowed, V, window_len, hop_len):
    wav_unfold = (V.unsqueeze(-1) * phasors_windowed.conj()).sum(-3).real
    wav_added = overlap_and_add(wav_unfold, hop_len)
    return wav_added * 3 / (window_len / hop_len)


def VP_transform(freqs, wav, window_len=2048, hop_len=512, center=False, sr: int = 16000):
    assert freqs.shape[-1] == wav.shape[-1]
    if center:
        pd = (window_len // 2, window_len // 2)
        freqs = F.pad(freqs, pd, "constant", 0)
        wav = F.pad(wav, pd, "constant", 0)
    phasors_windowed = _windowed_phasors(freqs, window_len, hop_len, sr)
    return _apply_phasors_to_wav(phasors_windowed, wav, window_len, hop_len)


def inverse_VP_transform(freqs, V, window_len=2048, hop_len=512, center=False, sr: int = 16000):
    if center:
        pd = (window_len // 2, window_len // 2)
        freqs = F.pad(freqs, pd, "constant", 0)
    phasors_windowed = _windowed_phasors(freqs, window_len, hop_len, sr)
    return _reconstruct_wav_from_phasors(phasors_windowed, V, window_len, hop_len)


class HarmonicTransformModule(nn.Module):
    """Audio + f0 series -> VP-transform -> Net -> inverse VP -> audio.

    Useful for harmonics-aware filtering / enhancement.
    """

    def __init__(
        self,
        net: nn.Module,
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
        self.net = net

    def forward(self, audio, f0s):
        freqs = harmonic_freq_series(f0s, self.n_harmonics)
        phasors_windowed = _windowed_phasors(freqs, self.window_len, self.hop_len, self.sample_rate)
        V = _apply_phasors_to_wav(phasors_windowed, audio, self.window_len, self.hop_len)
        V_processed = self.net(V)
        return _reconstruct_wav_from_phasors(phasors_windowed, V_processed, self.window_len, self.hop_len)
