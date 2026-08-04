"""HTDemucs fine-tuned from the official Meta music-separation checkpoint.

Wraps the in-repo :class:`models.demucs4ht.HTDemucs` (ZFTurbo lineage, an
exact architectural copy of ``demucs.htdemucs.HTDemucs`` v4) so the official
pretrained model (``htdemucs``, snapshot ``955717e8-8726e21a.th``, trained on
MUSDB18 + 800 extra songs at 44.1 kHz stereo, 4 stems) becomes a
speech-enhancement baseline for the F1 program (``f1_htdemucs_{a,b}``).

Three adaptations, in order of application:

1. **Sample rate — resample, do not retune.** Our data is 16 kHz; every
   pretrained STFT/conv weight assumes 44.1 kHz framing (nfft 4096 == 93 ms
   at 44.1 kHz). The wrapper resamples 16 k -> 44.1 k at the input and back
   at the output (``julius.ResampleFrac``, differentiable, kernel registered
   as a buffer). This preserves the pretrained feature alignment — the
   memory-endorsed route for pretrained weights ("native-16khz-baselines":
   resample only *because of* pretrained weights). All losses and metrics
   see 16 kHz waveforms; the 44.1 kHz path is internal to ``forward``.

2. **Channels — duplicate mono to stereo, average back.** The checkpoint's
   first convolutions expect 2 audio channels. Duplicating the mono input
   keeps both channel filters on-distribution (each sees a valid waveform)
   and touches zero weights; adapting the input conv to mono would either
   discard or sum pretrained filters and would also cascade into the
   source-grouped output layout. The two output channels are averaged.

3. **Head 4 stems -> 2 sources.** ``sources = ["speech", "noise"]``. The
   only source-dependent parameters in HTDemucs are the final transposed
   convolutions of the two branches (there are no per-source embeddings):

   - ``decoder.3.conv_tr.{weight,bias}`` — freq branch, out-channel axis
     grouped as (source, audio_channel*2 complex-as-channels) = 4 blocks of 4;
   - ``tdecoder.3.conv_tr.{weight,bias}`` — time branch, out-channel axis
     grouped as (source, audio_channel) = 4 blocks of 2.

   The 2-source head is warm-started by slicing source blocks:
   **speech <- VOCALS** (stem index 3, semantically closest to speech) and
   **noise <- OTHER** (stem index 2, the broadband instrumental residual).
   Every other tensor (533-key state dict minus these 4) loads verbatim
   with ``strict=True`` — the load fails loudly on any drift.

The wrapper's contract matches :class:`tasks.codecs.SpeechEnhancementCodec`:
``forward((B, 1, T) @ 16 kHz) -> (B, T)`` enhanced speech. Only the speech
output is returned (the in-repo core already selects ``target_instrument``),
so gradients reach the trunk and the speech block of the head; the noise
block rides along frozen-by-irrelevance.

``use_train_segment=False`` (weight-free flag): the model accepts any input
length instead of pinning eval inputs to the train segment length, so 1 s
training chunks and 2 s validation clips both pass through unchanged.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F
from julius.resample import ResampleFrac
from torch import nn

from models.demucs4ht import HTDemucs

__all__ = ["HTDemucsFineTune", "build_htdemucs_ft", "OFFICIAL_HTDEMUCS_KWARGS"]

# Constructor kwargs stored inside the official checkpoint package
# (``pkg["kwargs"]`` of ``hybrid_transformer/955717e8-8726e21a.th``), verified
# 2026-08-04. Hardcoded so the module is constructible without the checkpoint
# (tests, config validation); when a checkpoint is given, the stored kwargs
# are asserted against this dict so silent drift is impossible.
OFFICIAL_HTDEMUCS_KWARGS: dict[str, Any] = {
    "audio_channels": 2,
    "bottom_channels": 512,
    "cac": True,
    "channels": 48,
    "channels_time": None,
    "context": 1,
    "context_enc": 0,
    "dconv_comp": 8,
    "dconv_depth": 2,
    "dconv_init": 0.001,
    "dconv_mode": 3,
    "depth": 4,
    "emb_scale": 10,
    "emb_smooth": True,
    "end_iters": 0,
    "freq_emb": 0.2,
    "growth": 2,
    "kernel_size": 8,
    "multi_freqs": [],
    "multi_freqs_depth": 3,
    "nfft": 4096,
    "norm_groups": 4,
    "norm_starts": 4,
    "rescale": 0.1,
    "rewrite": True,
    "samplerate": 44100,
    "segment": Fraction(39, 5),
    "stride": 4,
    "t_auto_sparsity": False,
    "t_cape_augment": True,
    "t_cape_glob_loc_scale": [5000.0, 1.0, 1.4],
    "t_cape_mean_normalize": True,
    "t_cross_first": False,
    "t_dropout": 0.02,
    "t_emb": "sin",
    "t_gelu": True,
    "t_global_window": 100,
    "t_group_norm": False,
    "t_heads": 8,
    "t_hidden_scale": 4.0,
    "t_layer_scale": True,
    "t_layers": 5,
    "t_lr": None,
    "t_mask_random_seed": 42,
    "t_mask_type": "diag",
    "t_max_period": 10000.0,
    "t_max_positions": 10000,
    "t_norm_first": True,
    "t_norm_in": True,
    "t_norm_in_group": False,
    "t_norm_out": True,
    "t_sin_random_shift": 0,
    "t_sparse_attn_window": 400,
    "t_sparse_cross_attn": False,
    "t_sparse_self_attn": False,
    "t_sparsity": 0.95,
    "t_weight_decay": 0.0,
    "t_weight_pos_embed": 1.0,
    "time_stride": 2,
    "wiener_iters": 0,
    "wiener_residual": False,
}

# Official stem order in the checkpoint.
_STEMS = ("drums", "bass", "other", "vocals")
# Our 2-source head <- which pretrained stem block warm-starts it.
_SOURCE_INIT = {"speech": "vocals", "noise": "other"}
# The four source-dependent state-dict tensors: name -> out-channel axis.
_HEAD_TENSORS = {
    "decoder.3.conv_tr.weight": 1,  # ConvTranspose2d (in, out, kH, kW)
    "decoder.3.conv_tr.bias": 0,
    "tdecoder.3.conv_tr.weight": 1,  # ConvTranspose1d (in, out, k)
    "tdecoder.3.conv_tr.bias": 0,
}


def _remap_head(state: dict[str, torch.Tensor], sources: list[str]) -> dict[str, torch.Tensor]:
    """Slice the 4-stem output tensors into the 2-source layout.

    The out-channel axis of each head tensor is grouped as ``len(_STEMS)``
    equal per-source blocks (see module docstring). For each new source, the
    block of its ``_SOURCE_INIT`` donor stem is taken; blocks are then
    concatenated in the new source order.
    """
    out = dict(state)
    for name, axis in _HEAD_TENSORS.items():
        t = state[name]
        n = t.shape[axis]
        block = n // len(_STEMS)
        if block * len(_STEMS) != n:
            raise ValueError(f"{name}: out-channel size {n} not divisible by {len(_STEMS)} stems")
        pieces = []
        for src in sources:
            donor = _SOURCE_INIT[src]
            i = _STEMS.index(donor)
            pieces.append(t.narrow(axis, i * block, block))
        out[name] = torch.cat(pieces, dim=axis)
    return out


class HTDemucsFineTune(nn.Module):
    """Mono 16 kHz SE front around the pretrained 44.1 kHz stereo HTDemucs."""

    def __init__(
        self,
        checkpoint: str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        sources = ["speech", "noise"]
        self.core = HTDemucs(
            sources=sources,
            target_instrument="speech",
            use_train_segment=False,
            **OFFICIAL_HTDEMUCS_KWARGS,
        )
        model_sr = int(OFFICIAL_HTDEMUCS_KWARGS["samplerate"])
        self.model_sr = model_sr
        # julius reduces nothing itself — pass the reduced ratio (160:441 for
        # 16000:44100) so the sinc kernels stay small.
        g = math.gcd(self.sample_rate, model_sr)
        self.upsample = ResampleFrac(self.sample_rate // g, model_sr // g)
        self.downsample = ResampleFrac(model_sr // g, self.sample_rate // g)
        if checkpoint is not None:
            self._load_pretrained(checkpoint, sources)

    def _load_pretrained(self, checkpoint: str, sources: list[str]) -> None:
        from training.artifacts import resolve_checkpoint_uri

        path = resolve_checkpoint_uri(checkpoint)
        pkg = torch.load(path, map_location="cpu", weights_only=False)
        kw = pkg["kwargs"]
        if list(kw["sources"]) != list(_STEMS):
            raise ValueError(f"unexpected pretrained stem order: {kw['sources']}")
        mismatch = {
            k: (kw[k], v) for k, v in OFFICIAL_HTDEMUCS_KWARGS.items() if k in kw and kw[k] != v
        }
        if mismatch:
            raise ValueError(f"checkpoint kwargs differ from OFFICIAL_HTDEMUCS_KWARGS: {mismatch}")
        state = _remap_head(pkg["state"], sources)
        # Everything except the 4 remapped head tensors must match verbatim.
        self.core.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, T) mono @ 16 kHz -> (B, T) enhanced speech @ 16 kHz."""
        if x.dim() == 2:  # tolerate (B, T)
            x = x.unsqueeze(1)
        n = x.shape[-1]
        x44 = self.upsample(x)
        x44 = x44.expand(-1, 2, -1)  # mono -> stereo (duplicate)
        y44 = self.core(x44)  # (B, 1, 2, T44) — speech source only
        y44 = y44.squeeze(1).mean(dim=1, keepdim=True)  # (B, 1, T44)
        y = self.downsample(y44)
        # Exact-length guarantee against fractional-resampling rounding.
        if y.shape[-1] < n:
            y = F.pad(y, (0, n - y.shape[-1]))
        return y[..., :n].squeeze(1)


def build_htdemucs_ft(**params: Any) -> nn.Module:
    """Factory for the ``_target_`` config path: ``build_htdemucs_ft(**params)``."""
    return HTDemucsFineTune(**params)
