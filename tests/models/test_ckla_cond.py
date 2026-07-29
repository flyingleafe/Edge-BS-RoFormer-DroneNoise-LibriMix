"""Tests for the conditional RPS refiner (``models.ckla.SimpleConvV2CKLACond``):
forward/shape/grad contract (mirroring ``tests/models/test_ckla.py``'s style),
the bounded-residual guarantee, conditioning-length resampling, and the
registry key."""

from __future__ import annotations

import pytest
import tdseries as td
import torch
import torch.nn.functional as F

from losses.pit import RPSMSELoss
from models.ckla import SimpleConvV2CKLACond
from tasks.codecs import RPSPredictionCodec


def _small_model(**kw) -> SimpleConvV2CKLACond:
    torch.manual_seed(0)
    return SimpleConvV2CKLACond(
        n_fft=256, hop_length=64, num_rotors=4, p_init=1.0, readout="phase_only", **kw
    )


def test_forward_shape_and_grads():
    model = _small_model()
    audio = torch.randn(2, 8000)
    cond = torch.rand(2, 4, 126) * 80.0
    out = model(audio, cond)
    assert out.shape == (2, 4, 126)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    # The conditioning path must actually receive gradient.
    first = model.cond_mlp[0]
    assert isinstance(first, torch.nn.Linear)
    w = first.weight.grad
    assert w is not None and w.abs().sum() > 0


def test_output_is_bounded_residual_on_conditioning():
    model = _small_model(max_delta=3.0)
    model.eval()
    audio = torch.randn(1, 8000)
    cond = torch.rand(1, 4, 126) * 80.0
    with torch.no_grad():
        out = model(audio, cond)
    assert ((out - cond).abs() <= 3.0 + 1e-5).all()
    # Rotor identity is structural: shifting one conditioning row moves the
    # SAME output row by construction of the residual.
    cond2 = cond.clone()
    cond2[:, 2] += 10.0
    with torch.no_grad():
        out2 = model(audio, cond2)
    assert ((out2[:, 2] - out[:, 2]) - 10.0).abs().max() <= 2 * 3.0 + 1e-5


def test_cond_frame_count_resampled_to_trunk_grid():
    model = _small_model()
    model.eval()
    audio = torch.randn(1, 8000)  # trunk emits 126 frames at n_fft 256 / hop 64
    cond = torch.rand(1, 4, 251) * 80.0
    with torch.no_grad():
        out = model(audio, cond)
    assert out.shape == (1, 4, 126)
    c_res = F.interpolate(cond, size=126, mode="linear", align_corners=True)
    assert ((out - c_res).abs() <= model.max_delta + 1e-5).all()


def test_rejects_bad_cond_shape():
    model = _small_model()
    with pytest.raises(ValueError, match="cond must be"):
        model(torch.randn(1, 8000), torch.rand(1, 3, 126))


def test_amp_autocast_backward_through_codec_and_rpsmse_loss():
    """Mirror ``training/loop.py``'s AMP structure: ``codec.call_model`` runs
    under autocast, the loss is computed OUTSIDE on the half-precision model
    output vs the fp32 GT target. Regression test for the first-batch GPU
    crash ``Found dtype Float but expected Half`` at
    ``scaler.scale(loss).backward()``: ``RPSMSELoss`` fed mixed dtypes to
    ``F.mse_loss``, whose backward demands exactly matching dtypes — unlike
    the promoting elementwise ops ``PITMSELoss`` is built from (which is why
    the unconditional CKLA model trained fine under the same loop). CPU bf16
    autocast exercises the same mixed-dtype regime as CUDA fp16."""
    probe = torch.nn.Linear(4, 4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        if probe(torch.randn(2, 4)).dtype != torch.bfloat16:
            pytest.skip("CPU bf16 autocast inactive on this torch build")

    model = _small_model()
    frame_rate = (16000, 64)
    batch = td.Frame(
        {
            "mixture": td.uniform(torch.randn(2, 8000), (16000, 1), dims=("batch", "time")),
            "rps_cond": td.uniform(
                torch.rand(2, 4, 126) * 80.0, frame_rate, dims=("batch", "rotor", "time")
            ),
            "rps": td.uniform(
                torch.rand(2, 4, 126) * 80.0, frame_rate, dims=("batch", "rotor", "time")
            ),
        }
    )
    codec = RPSPredictionCodec(frame_rate=frame_rate, use_cond=True)
    inputs = codec.to_inputs(batch)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = codec.call_model(model, inputs)
    # The mixed-dtype regime under test: half-precision pred, fp32 target.
    assert outputs.dtype == torch.bfloat16
    pred_frame = codec.to_frame(outputs, batch)

    loss = RPSMSELoss()(pred_frame, batch)
    assert loss.dtype == torch.float32
    loss.backward()  # raised "Found dtype Float but expected BFloat16" pre-fix
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    first = model.cond_mlp[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.weight.grad is not None


def test_registry_builds_cond_variant():
    from models.ckla import ComplexKLALayer
    from models.registry import build_model

    m = build_model(
        "simple_conv_v2_ckla_phaseonly_cond",
        n_fft=256,
        hop_length=64,
        num_rotors=4,
        p_init=1.0,
    )
    assert isinstance(m, SimpleConvV2CKLACond)
    for blk in m.head.blocks:
        mixer = blk.mixer
        assert isinstance(mixer, ComplexKLALayer)
        assert mixer.readout == "phase_only"
    # Head input is widened by the conditioning embedding.
    assert m.head.in_proj.in_features == 128 + m.cond_embed_dim
