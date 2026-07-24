"""G8a multi-resolution pyramid front-end (C1 of the hierarchical front-end
design): per-band shape/finiteness, cross-band comb alignment on the shared
log-f axis, per-band IF sub-bin recovery, time-grid contract, and the
SimpleConvV2TransformerPyramid model smoke.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.frontends import build_frontend
from models.frontends.pyramid import PyramidIFFrontEnd
from models.registry import build_model
from models.rps_predictor import ResidualConvBlock2d, SimpleConvV2TransformerPyramid

SR = 16000


def _pyramid() -> PyramidIFFrontEnd:
    fe = build_frontend("pyramid_if")
    assert isinstance(fe, PyramidIFFrontEnd)
    return fe


def test_pyramid_shape_time_grid_and_finiteness():
    fe = _pyramid()
    assert fe.out_channels == 8  # 2 channels x 4 bands
    assert fe.n_rows == 340  # ~48 bins/octave over log2(4000/30) octaves
    audio = torch.randn(2, SR)
    out = fe(audio)
    assert out.shape == (2, 8, 340, SR // 512 + 1)  # hop-512 time grid for 1 s
    assert torch.isfinite(out).all()
    assert fe.num_frames(SR) == SR // 512 + 1
    # 1.5 s: time grid still n // 512 + 1
    n = SR + SR // 2
    assert fe(torch.randn(1, n)).shape[-1] == n // 512 + 1 == fe.num_frames(n)
    # zero trainable parameters (the G8 overfitting constraint)
    assert sum(p.numel() for p in fe.parameters()) == 0


def test_pyramid_band_partition_and_crop():
    """Every log-f row belongs to exactly one band; band channels are zero
    outside their own rows (the 8192-band's smear cannot leak upward)."""
    fe = _pyramid()
    masks = torch.stack([fe.get_buffer(f"row_mask_{b}") for b in range(4)])  # (4, R)
    assert torch.all(masks.sum(dim=0) == 1.0)
    out = fe(torch.randn(1, SR))
    for b in range(4):
        off_rows = fe.get_buffer(f"row_mask_{b}") == 0
        assert out[0, 2 * b : 2 * b + 2, off_rows].abs().max() == 0.0


def _comb_60(dur_s: float = 2.0) -> torch.Tensor:
    t = torch.arange(int(SR * dur_s), dtype=torch.float64) / SR
    sig = torch.zeros_like(t)
    for k in range(1, int(4000 / 60.0) + 1):
        sig += (1.0 / k**0.5) * torch.sin(2 * math.pi * k * 60.0 * t + 0.7 * k)
    sig = (sig / sig.abs().max()).to(torch.float32)
    return sig + 0.005 * torch.randn_like(sig)


def test_cross_band_comb_alignment_on_log_axis():
    """G8a-specific: a 60 rev/s comb's k=2 tooth (120 Hz, 8192 band) and k=20
    tooth (1200 Hz, 2048 band) land on the correct rows of the SHARED log-f
    grid — the bands agree on where frequencies live."""
    torch.manual_seed(0)
    fe = _pyramid()
    out = fe(_comb_60().unsqueeze(0))
    f_log = fe.get_buffer("f_log")
    # (tooth Hz, band index, mag channel = 2*band)
    for tooth_hz, ch in [(120.0, 0), (1200.0, 4)]:
        r_exp = int(torch.argmin((f_log - tooth_hz).abs()))
        mag = out[0, ch, :, 5:-5].mean(dim=-1)
        lo = max(0, r_exp - 2)  # +-2 rows ~ +-3% in f: excludes adjacent teeth
        r_meas = lo + int(mag[lo : r_exp + 3].argmax())
        assert abs(r_meas - r_exp) <= 1, (tooth_hz, r_exp, r_meas)


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
    model = SimpleConvV2TransformerPyramid()
    block = model.encoder[0]
    assert isinstance(block, ResidualConvBlock2d)
    assert block.conv.in_channels == 8
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
