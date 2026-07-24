"""G8a/G8a2 multi-resolution pyramid front-end (C1 of the hierarchical
front-end design): shape/finiteness for both the dense (collapse_bands=True,
G8a2 default) and concat (False, dead G8a) layouts, band partition, cross-band
comb alignment on the shared log-f axis, dense-sum exactness, per-band IF
sub-bin recovery, time-grid contract, and the SimpleConvV2TransformerPyramid
model smoke.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.frontends import build_frontend
from models.frontends.pyramid import PyramidIFFrontEnd
from models.registry import build_model
from models.rps_predictor import ResidualConvBlock2d, SimpleConvV2TransformerPyramid

SR = 16000


def _pyramid(collapse: bool = True) -> PyramidIFFrontEnd:
    fe = build_frontend("pyramid_if", collapse_bands=collapse)
    assert isinstance(fe, PyramidIFFrontEnd)
    return fe


@pytest.mark.parametrize("collapse,n_ch", [(True, 2), (False, 8)])
def test_pyramid_shape_time_grid_and_finiteness(collapse, n_ch):
    fe = _pyramid(collapse)
    assert fe.out_channels == n_ch  # dense mag+IF, or 2 channels x 4 bands
    assert fe.n_rows == 340  # ~48 bins/octave over log2(4000/30) octaves
    audio = torch.randn(2, SR)
    out = fe(audio)
    assert out.shape == (2, n_ch, 340, SR // 512 + 1)  # hop-512 grid for 1 s
    assert torch.isfinite(out).all()
    assert fe.num_frames(SR) == SR // 512 + 1
    # 1.5 s: time grid still n // 512 + 1
    n = SR + SR // 2
    assert fe(torch.randn(1, n)).shape[-1] == n // 512 + 1 == fe.num_frames(n)
    # zero trainable parameters (the G8 overfitting constraint)
    assert sum(p.numel() for p in fe.parameters()) == 0


def test_pyramid_band_partition_and_crop():
    """Every log-f row belongs to exactly one band (coverage == 1 — the
    condition that makes the G8a2 dense sum exact); in the concat layout,
    band channels are zero outside their own rows (the 8192-band's smear
    cannot leak upward)."""
    fe = _pyramid(collapse=False)
    masks = torch.stack([fe.get_buffer(f"row_mask_{b}") for b in range(4)])  # (4, R)
    assert torch.all(masks.sum(dim=0) == 1.0)
    out = fe(torch.randn(1, SR))
    for b in range(4):
        off_rows = fe.get_buffer(f"row_mask_{b}") == 0
        assert out[0, 2 * b : 2 * b + 2, off_rows].abs().max() == 0.0


def test_dense_output_has_no_dead_rows():
    """G8a2 fix target: the dense mag channel is populated on every row (the
    G8a concat left 6 of 8 channels exactly zero at each row)."""
    torch.manual_seed(0)
    fe = _pyramid(collapse=True)
    out = fe(torch.randn(1, SR))
    assert (out[0, 0].abs().sum(dim=-1) > 0).all()  # every log-f row has mag


def _comb_60(dur_s: float = 2.0) -> torch.Tensor:
    t = torch.arange(int(SR * dur_s), dtype=torch.float64) / SR
    sig = torch.zeros_like(t)
    for k in range(1, int(4000 / 60.0) + 1):
        sig += (1.0 / k**0.5) * torch.sin(2 * math.pi * k * 60.0 * t + 0.7 * k)
    sig = (sig / sig.abs().max()).to(torch.float32)
    return sig + 0.005 * torch.randn_like(sig)


@pytest.mark.parametrize("collapse", [True, False])
def test_cross_band_comb_alignment_on_log_axis(collapse):
    """A 60 rev/s comb's k=2 tooth (120 Hz, 8192 band) and k=20 tooth
    (1200 Hz, 2048 band) land on the correct rows of the SHARED log-f grid —
    in both the dense and per-band channel layouts."""
    torch.manual_seed(0)
    fe = _pyramid(collapse)
    out = fe(_comb_60().unsqueeze(0))
    f_log = fe.get_buffer("f_log")
    # (tooth Hz, mag channel: dense ch 0, else 2*band)
    for tooth_hz, band in [(120.0, 0), (1200.0, 2)]:
        ch = 0 if collapse else 2 * band
        r_exp = int(torch.argmin((f_log - tooth_hz).abs()))
        mag = out[0, ch, :, 5:-5].mean(dim=-1)
        lo = max(0, r_exp - 2)  # +-2 rows ~ +-3% in f: excludes adjacent teeth
        r_meas = lo + int(mag[lo : r_exp + 3].argmax())
        assert abs(r_meas - r_exp) <= 1, (tooth_hz, r_exp, r_meas)


def test_dense_equals_owning_band_at_tooth_rows():
    """G8a2-specific: the row-mask partition makes the dense sum exact — at
    the k=2 (120 Hz, band 0) and k=20 (1200 Hz, band 2) tooth rows, the dense
    mag/IF values equal the owning band's channels from the concat layout."""
    torch.manual_seed(0)
    sig = _comb_60().unsqueeze(0)
    dense = _pyramid(collapse=True)(sig)  # (1, 2, R, T)
    concat = _pyramid(collapse=False)(sig)  # (1, 8, R, T)
    f_log = _pyramid().get_buffer("f_log")
    for tooth_hz, band in [(120.0, 0), (1200.0, 2)]:
        r = int(torch.argmin((f_log - tooth_hz).abs()))
        assert torch.allclose(dense[0, 0, r], concat[0, 2 * band, r], atol=1e-6)
        assert torch.allclose(dense[0, 1, r], concat[0, 2 * band + 1, r], atol=1e-6)
    # and globally: dense == sum over bands of the masked concat channels
    assert torch.allclose(dense[0, 0], concat[0, 0::2].sum(dim=0), atol=1e-6)
    assert torch.allclose(dense[0, 1], concat[0, 1::2].sum(dim=0), atol=1e-6)


def test_per_band_if_recovers_sub_bin_offset():
    """A tone offset +0.3 of a bin in each band's own terms reads ~+0.3 in
    that band's IF channel (native grid, before log-f resampling)."""
    fe = _pyramid()
    for b, n_fft in enumerate(fe.n_ffts):
        bin_hz = SR / n_fft
        lo_hz, hi_hz = fe.band_edges[b], fe.band_edges[b + 1]
        k0 = int(((lo_hz + hi_hz) / 2) / bin_hz)
        f = (k0 + 0.3) * bin_hz
        t = torch.arange(SR * 2, dtype=torch.float64) / SR
        tone = torch.sin(2 * math.pi * f * t).to(torch.float32).unsqueeze(0)
        _, if_dev = fe._band_mag_if(tone, b)
        mid = if_dev[0, k0, 4:-4]
        assert (mid.mean() - 0.3).abs() < 1e-3, (b, float(mid.mean()))


def test_registry_builds_pyramid_model():
    model = build_model(
        "simple_conv_v2_transformer_pyramid", n_fft=2048, hop_length=512, num_rotors=4
    )
    assert isinstance(model, SimpleConvV2TransformerPyramid)


def test_pyramid_transformer_forward_and_step():
    torch.manual_seed(0)
    model = SimpleConvV2TransformerPyramid()  # dense default (G8a2)
    block = model.encoder[0]
    assert isinstance(block, ResidualConvBlock2d)
    assert block.conv.in_channels == 2
    # A/B: collapse_bands=False reproduces the 8-channel G8a model
    block8 = SimpleConvV2TransformerPyramid(collapse_bands=False).encoder[0]
    assert isinstance(block8, ResidualConvBlock2d)
    assert block8.conv.in_channels == 8
    audio = torch.randn(2, SR)
    out = model(audio)
    assert out.shape == (2, 4, SR // 512 + 1)
    assert torch.isfinite(out).all()

    model.train()
    target = torch.rand(out.shape) * 50.0 + 40.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = F.mse_loss(model(audio), target)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    assert torch.isfinite(F.mse_loss(model(audio), target))
