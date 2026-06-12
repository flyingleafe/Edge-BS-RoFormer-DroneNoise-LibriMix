"""PyTorch port of Basic Pitch's note-transcription model.

Faithful re-implementation of ``basic_pitch.models.model`` (Spotify Basic
Pitch, ICASSP 2022).  Produces three frame-level posteriorgrams:

    contour : (B, time, 264)   3 bins/semitone pitch contour ("Yp")
    note    : (B, time, 88)    1 bin/semitone note activations ("Yn")
    onset   : (B, time, 88)    1 bin/semitone note onsets       ("Yo")

The architecture, kernel sizes, strides, padding ("same") and the per-output
sigmoids match the TF original exactly, so the released pretrained weights can
be ported directly (see :meth:`BasicPitch.from_pretrained_tf`).

Reference: Bittner et al., "A Lightweight Instrument-Agnostic Model for
Polyphonic Note Transcription and Multipitch Estimation", ICASSP 2022.
https://github.com/spotify/basic-pitch
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cqt import CONTOURS_BINS_PER_SEMITONE, N_FREQ_BINS_CONTOURS, CQTFrontEnd
from .nn import HarmonicStacking, flatten_freq_ch

# kernel sizes / strides / filter counts (basic_pitch.models)
CONTOUR_KERNEL_SIZE_2 = (3, 39)
CONTOUR_KERNEL_SIZE_3 = (5, 5)
CONTOUR_FILTERS_2 = 8

NOTES_KERNEL_SIZE_1 = (7, 7)
NOTES_STRIDES_1 = (1, 3)
NOTES_KERNEL_SIZE_2 = (7, 3)

ONSET_KERNEL_SIZE_1 = (5, 5)
ONSET_STRIDES_1 = (1, 3)
ONSET_KERNEL_SIZE_2 = (3, 3)

BN_EPSILON = 1e-3  # Keras BatchNormalization default (PyTorch default is 1e-5)


def _same_pad(size: int, k: int, s: int) -> tuple[int, int]:
    """TensorFlow 'SAME' padding (before, after) for one spatial dim."""
    out = math.ceil(size / s)
    pad = max((out - 1) * s + k - size, 0)
    return pad // 2, pad - pad // 2


class Conv2dSame(nn.Conv2d):
    """``nn.Conv2d`` with TensorFlow-style 'SAME' padding (supports strides)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride=(1, 1), bias: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph0, ph1 = _same_pad(x.shape[-2], kh, sh)
        pw0, pw1 = _same_pad(x.shape[-1], kw, sw)
        x = F.pad(x, (pw0, pw1, ph0, ph1))
        return super().forward(x)


class BasicPitch(nn.Module):
    def __init__(
        self,
        n_harmonics: int = 8,
        n_filters_contour: int = 32,
        n_filters_onsets: int = 32,
        n_filters_notes: int = 32,
        no_contours: bool = False,
        sr: int = 22050,
    ):
        super().__init__()
        self.no_contours = no_contours

        # --- input representation -------------------------------------------
        self.cqt = CQTFrontEnd(n_harmonics, sr=sr)
        self.cqt_bn = nn.BatchNorm2d(1, eps=BN_EPSILON)

        if n_harmonics > 1:
            harmonics: list[float] = [0.5] + list(range(1, n_harmonics))
        else:
            harmonics = [1]
        self.harmonic_stacking = HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE, harmonics, N_FREQ_BINS_CONTOURS
        )
        n_h = len(harmonics)

        # --- contour branch --------------------------------------------------
        self.contour_conv1 = Conv2dSame(n_h, CONTOUR_FILTERS_2, CONTOUR_KERNEL_SIZE_2)
        self.contour_bn = nn.BatchNorm2d(CONTOUR_FILTERS_2, eps=BN_EPSILON)
        if not no_contours:
            self.contour_out = Conv2dSame(CONTOUR_FILTERS_2, 1, CONTOUR_KERNEL_SIZE_3)

        # --- note branch -----------------------------------------------------
        notes_in = 1 if not no_contours else CONTOUR_FILTERS_2
        self.notes_conv1 = Conv2dSame(
            notes_in, n_filters_notes, NOTES_KERNEL_SIZE_1, stride=NOTES_STRIDES_1
        )
        self.notes_out = Conv2dSame(n_filters_notes, 1, NOTES_KERNEL_SIZE_2)

        # --- onset branch ----------------------------------------------------
        self.onset_conv1 = Conv2dSame(
            n_h, n_filters_onsets, ONSET_KERNEL_SIZE_1, stride=ONSET_STRIDES_1
        )
        self.onset_bn = nn.BatchNorm2d(n_filters_onsets, eps=BN_EPSILON)
        self.onset_out = Conv2dSame(n_filters_onsets + 1, 1, ONSET_KERNEL_SIZE_2)

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        # audio: (B, n_samples) at 22050 Hz
        x = self.cqt(audio)  # (B, 1, time, n_bins)
        x = self.cqt_bn(x)
        x = self.harmonic_stacking(x)  # (B, n_h, time, 264)

        # contour branch
        xc = F.relu(self.contour_bn(self.contour_conv1(x)))
        if not self.no_contours:
            xc = torch.sigmoid(self.contour_out(xc))  # (B, 1, time, 264)
            contour = flatten_freq_ch(xc)  # (B, time, 264)
            x_contours_reduced = contour.unsqueeze(1)  # (B, 1, time, 264)
        else:
            contour = None
            x_contours_reduced = xc

        # note branch
        xn = F.relu(self.notes_conv1(x_contours_reduced))
        x_notes_pre = torch.sigmoid(self.notes_out(xn))  # (B, 1, time, 88)
        note = flatten_freq_ch(x_notes_pre)  # (B, time, 88)

        # onset branch
        xo = F.relu(self.onset_bn(self.onset_conv1(x)))  # (B, n_filters, time, 88)
        xo = torch.cat([x_notes_pre, xo], dim=1)  # (B, n_filters+1, time, 88)
        xo = torch.sigmoid(self.onset_out(xo))  # (B, 1, time, 88)
        onset = flatten_freq_ch(xo)  # (B, time, 88)

        out = {"onset": onset, "note": note}
        if not self.no_contours and contour is not None:
            out["contour"] = contour
        return out

    def contour_logits(self, audio: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid contour posteriorgram, ``(B, time, 264)``.

        Runs only the CQT → harmonic-stacking → contour branch (skipping the
        note/onset heads and the final sigmoid), for use as raw salience logits
        with ``BCEWithLogitsLoss``. Requires ``no_contours=False``.
        """
        if self.no_contours:
            raise ValueError("contour_logits requires no_contours=False")
        x = self.cqt(audio)  # (B, 1, time, n_bins)
        x = self.cqt_bn(x)
        x = self.harmonic_stacking(x)  # (B, n_h, time, 264)
        xc = F.relu(self.contour_bn(self.contour_conv1(x)))
        logits = self.contour_out(xc)  # (B, 1, time, 264) — no sigmoid
        return flatten_freq_ch(logits)  # (B, time, 264)

    # ------------------------------------------------------------------ weights
    @torch.no_grad()
    def load_tf_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Load weights extracted from the released TF checkpoint.

        ``weights`` maps ``layer_with_weights-<n>/<param>`` (the checkpoint
        variable names, with the ``/.ATTRIBUTES/VARIABLE_VALUE`` suffix
        stripped) to numpy arrays.  Layers are matched by parameter shape,
        which is unambiguous for this architecture.
        """
        convs = {}  # (kh,kw,in,out) -> (kernel, bias)
        bns = {}  # channels -> dict(gamma, beta, moving_mean, moving_variance)
        for name, arr in weights.items():
            base = name.split("/")[0]  # layer_with_weights-N
            param = name.split("/")[1]
            if param == "kernel":
                convs.setdefault(base, {})["kernel"] = arr
            elif param == "bias":
                convs.setdefault(base, {})["bias"] = arr
            elif param in ("gamma", "beta", "moving_mean", "moving_variance"):
                bns.setdefault(base, {})[param] = arr

        conv_by_shape = {tuple(v["kernel"].shape): v for v in convs.values()}
        bn_by_ch = {v["gamma"].shape[0]: v for v in bns.values()}

        def set_conv(layer: nn.Conv2d, tf_shape):
            v = conv_by_shape[tf_shape]
            k = np.transpose(v["kernel"], (3, 2, 0, 1))  # (kh,kw,in,out)->(out,in,kh,kw)
            layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(k)))
            layer.bias.copy_(torch.from_numpy(np.ascontiguousarray(v["bias"])))

        def set_bn(layer: nn.BatchNorm2d, ch: int):
            v = bn_by_ch[ch]
            layer.weight.copy_(torch.from_numpy(v["gamma"].copy()))
            layer.bias.copy_(torch.from_numpy(v["beta"].copy()))
            layer.running_mean.copy_(torch.from_numpy(v["moving_mean"].copy()))
            layer.running_var.copy_(torch.from_numpy(v["moving_variance"].copy()))

        set_bn(self.cqt_bn, 1)
        set_conv(self.contour_conv1, (3, 39, 8, 8))
        set_bn(self.contour_bn, 8)
        set_conv(self.contour_out, (5, 5, 8, 1))
        set_conv(self.notes_conv1, (7, 7, 1, 32))
        set_conv(self.notes_out, (7, 3, 32, 1))
        set_conv(self.onset_conv1, (5, 5, 8, 32))
        set_bn(self.onset_bn, 32)
        set_conv(self.onset_out, (3, 3, 33, 1))

    @classmethod
    def from_pretrained(cls, **kwargs) -> BasicPitch:
        """Load the released ICASSP-2022 weights (converted to a torch
        state_dict committed alongside this package).  No TF dependency."""
        import os

        ckpt = os.path.join(os.path.dirname(__file__), "weights", "icassp_2022.pt")
        model = cls(**kwargs)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        return model

    @classmethod
    def from_pretrained_tf(cls, weights_npz: str, **kwargs) -> BasicPitch:
        data = np.load(weights_npz)
        # Checkpoint var names like ``layer_with_weights-1/kernel/.ATTRIBUTES/
        # VARIABLE_VALUE`` were stored with '/' -> '__'.  Keep only the weight
        # layers and re-key as ``<layer>/<param>``.
        weights = {}
        for k in data.files:
            if not k.startswith("layer_with_weights-"):
                continue
            parts = k.split("__")  # [layer, param, .ATTRIBUTES, VARIABLE_VALUE]
            weights[f"{parts[0]}/{parts[1]}"] = data[k]
        model = cls(**kwargs)
        model.load_tf_weights(weights)
        model.eval()
        return model
