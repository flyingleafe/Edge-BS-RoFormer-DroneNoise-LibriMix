"""Tests for the HG-CKLA harmonic-gather refiner (``models.hg_ckla``).

Ground truth is synthetic: a comb of pure tones at a known rotation rate, for
which the measurement operator has an exact answer — the gather must land on
the teeth, and the innovation angles must return the rate error between the
conditioning and the comb. The remaining tests pin the model contract
(shape/finiteness/gradients, the bounded residual, the anneal cap) and the
registry + codec path the training loop uses.
"""

from __future__ import annotations

import math

import pytest
import tdseries as td
import torch
import torch.nn.functional as F

from losses.pit import RPSMSELoss
from models.hg_ckla import (
    HGCKLACell,
    HGCKLARefiner,
    harmonic_positions,
    innovation_phasors,
    physics_rate_error,
    soft_gather,
)
from tasks.codecs import RPSPredictionCodec

SR = 16000
N_FFT = 2048
HOP = 512


def _comb(f0: float, n_harmonics: int = 10, seconds: float = 2.0, amp: float = 1.0) -> torch.Tensor:
    """A stationary comb of pure tones at ``k * f0`` Hz, shape (1, N)."""
    t = torch.arange(int(seconds * SR)) / SR
    x = torch.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        x = x + amp * torch.cos(2 * math.pi * k * f0 * t + 0.3 * k)
    return x.unsqueeze(0)


def _model(**kw) -> HGCKLARefiner:
    torch.manual_seed(0)
    return HGCKLARefiner(n_fft=N_FFT, hop_length=HOP, sample_rate=SR, **kw)


def _cells(model: HGCKLARefiner) -> list[HGCKLACell]:
    return [c for c in model.cells if isinstance(c, HGCKLACell)]


def _gathered(model: HGCKLARefiner, audio: torch.Tensor, cond: torch.Tensor, n_harmonics: int):
    """(g, gi, valid, arg_u, log_mag, valid_t) for one (audio, cond) pair."""
    spec = model.stft(audio)
    x_re, x_im = spec.real.contiguous(), spec.imag.contiguous()
    pos = harmonic_positions(cond, n_harmonics, N_FFT, SR)
    prev_re = F.pad(x_re[..., :-1], (1, 0))
    prev_im = F.pad(x_im[..., :-1], (1, 0))
    g, gi, valid = soft_gather(x_re, x_im, pos, 1.5, 4.0, prev_re, prev_im)
    u_re, u_im, arg_u, log_mag, valid_t = innovation_phasors(
        g[0], gi[0], g[1], gi[1], cond, HOP, SR
    )
    return g, gi, valid, arg_u, log_mag, valid_t


# ─── the measurement operator ────────────────────────────────────────────────


def test_gather_lands_on_the_comb_teeth():
    """A gather at the true harmonics collects the comb; a gather between the
    teeth collects the floor. The teeth also come out FLAT — the window phase
    alignment is what makes that true (an unaligned real Gaussian cancels the
    main lobe at unlucky fractional offsets)."""
    model = _model(k_caps=(10,))
    f0 = 80.0
    audio = _comb(f0, n_harmonics=10)
    spec = model.stft(audio)
    x_re, x_im = spec.real.contiguous(), spec.imag.contiguous()
    n_frames = x_re.shape[-1]

    k = torch.arange(1, 11, dtype=torch.float32).view(1, 1, -1, 1)
    scale = N_FFT / SR
    pos_on = (k * f0 * scale).expand(1, 1, 10, n_frames).contiguous()
    pos_off = ((k + 0.5) * f0 * scale).expand(1, 1, 10, n_frames).contiguous()

    on_re, on_im, _ = soft_gather(x_re, x_im, pos_on)
    off_re, off_im, _ = soft_gather(x_re, x_im, pos_off)
    on = torch.sqrt(on_re**2 + on_im**2)[0, 0, :, 3:-3].mean(dim=-1)
    off = torch.sqrt(off_re**2 + off_im**2)[0, 0, :, 3:-3].mean(dim=-1)

    assert (on > 10.0 * off).all(), f"on-tooth {on} vs off-tooth {off}"
    # Flat across the teeth: max/min within 5 %.
    assert on.max() / on.min() < 1.05


def test_gather_positions_follow_the_conditioning():
    """Shifting the conditioning by +2 rev/s moves harmonic k by exactly
    2k/df bins, and the gathered energy follows the comb it now points at."""
    n_harm, shift = 10, 2.0
    cond = torch.full((1, 4, 7), 80.0)
    pos = harmonic_positions(cond, n_harm, N_FFT, SR)
    pos_shifted = harmonic_positions(cond + shift, n_harm, N_FFT, SR)
    k = torch.arange(1, n_harm + 1, dtype=torch.float32).view(1, 1, -1, 1)
    expected = shift * k * (N_FFT / SR)
    assert torch.allclose(pos_shifted - pos, expected.expand_as(pos), atol=1e-4)

    model = _model(k_caps=(10,))
    audio = _comb(82.0, n_harmonics=n_harm)
    n_frames = model.stft(audio).shape[-1]

    def energy(rate: float) -> torch.Tensor:
        c = torch.full((1, 1, n_frames), rate)
        g, gi, *_ = _gathered(model, audio, c, n_harm)
        return torch.sqrt(g[0] ** 2 + gi[0] ** 2)[0, 0, :, 3:-3].mean(dim=-1)

    at80, at81, at82 = energy(80.0), energy(81.0), energy(82.0)
    # The gathered energy peaks at the true rate, and the top harmonic — which
    # walks off its tooth k times faster than the fundamental — keeps less
    # than half of it 2 rev/s away. Harmonic 1 barely moves: the Gaussian
    # window IS a wide capture band, by design.
    assert float(at82.sum()) > float(at81.sum()) > float(at80.sum())
    assert float(at80[-1]) < 0.5 * float(at82[-1])


@pytest.mark.parametrize("df", [0.2, 0.4, 0.7, 1.0])
def test_innovation_physics_recovers_the_rate_error(df: float):
    """With the conditioning off the comb by ``df`` rev/s, the linear-physics
    path returns ``df`` — this is the measurement the whole cell is built
    around (K=10, clean comb, no noise)."""
    n_harm = 10
    f0 = 80.0
    model = _model(k_caps=(n_harm,))
    audio = _comb(f0, n_harmonics=n_harm)
    n_frames = model.stft(audio).shape[-1]
    cond = torch.full((1, 1, n_frames), f0 - df)

    _g, _gi, valid, arg_u, _log_mag, valid_t = _gathered(model, audio, cond, n_harm)
    k = torch.arange(1, n_harm + 1, dtype=torch.float32).view(1, 1, -1, 1)
    weights = k**2 * valid * valid_t * (arg_u.abs() <= 0.8 * math.pi)
    est = physics_rate_error(arg_u, weights, HOP, SR)

    recovered = float(est[0, 0, 3:-3].median())
    assert abs(recovered - df) / df < 0.10, f"df={df}, recovered={recovered}"


# ─── the model contract ──────────────────────────────────────────────────────


def test_forward_shape_finite_and_grads():
    model = _model()
    audio = torch.randn(2, 32000)
    cond = torch.rand(2, 4, 63) * 60.0 + 50.0
    out = model(audio, cond)
    assert out.shape == (2, 4, 63)
    assert torch.isfinite(out).all()

    out.pow(2).mean().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    # The two learned pieces of the measurement path must both be in the
    # gradient: the WP18 weights and the correction MLP.
    for cell in _cells(model):
        w_grad, mlp_grad = cell.w_param.grad, cell.in_proj.weight.grad
        assert w_grad is not None and w_grad.abs().sum() > 0
        assert mlp_grad is not None and mlp_grad.abs().sum() > 0


def test_output_is_bounded_residual_on_conditioning():
    model = _model(max_delta=5.0)
    model.eval()
    audio = torch.randn(1, 32000)
    cond = torch.rand(1, 4, 63) * 60.0 + 50.0
    with torch.no_grad():
        out = model(audio, cond)
    assert ((out - cond).abs() <= 5.0 + 1e-5).all()
    # Rotor identity is structural (residual on the conditioning row).
    cond2 = cond.clone()
    cond2[:, 2] += 10.0
    with torch.no_grad():
        out2 = model(audio, cond2)
    assert ((out2[:, 2] - out[:, 2]) - 10.0).abs().max() <= 2 * 5.0 + 1e-5


def test_cond_frame_count_resampled_and_bad_shape_rejected():
    model = _model()
    model.eval()
    audio = torch.randn(1, 32000)
    with torch.no_grad():
        out = model(audio, torch.rand(1, 4, 120) * 60.0 + 50.0)
    assert out.shape == (1, 4, 63)
    with pytest.raises(ValueError, match="cond must be"):
        model(audio, torch.rand(1, 3, 63))


def test_anneal_cap_ignores_high_harmonics():
    """Cell j reads harmonics 1..k_caps[j] and nothing above: the feature
    stack is exactly 3*k_cap+2 wide, and energy parked at harmonic 30 leaves
    a k_cap=10 model's output where it was."""
    model = _model(k_caps=(10,))
    model.eval()
    cell = _cells(model)[0]
    assert cell.k_cap == 10
    assert cell.in_proj.in_features == 3 * 10 + 2

    audio = _comb(80.0, n_harmonics=10)
    spec = model.stft(audio)
    cond = torch.full((1, 4, spec.shape[-1]), 79.5)
    feats = cell.measure(spec.real.contiguous(), spec.imag.contiguous(), cond)["feats"]
    assert feats.shape[2] == 3 * 10 + 2

    t = torch.arange(audio.shape[-1]) / SR
    extra = audio + 5.0 * torch.cos(2 * math.pi * 30 * 80.0 * t).unsqueeze(0)
    with torch.no_grad():
        base_out = model(audio, cond)
        extra_out = model(extra, cond)
    assert (base_out - extra_out).abs().max() < 1e-2
    # The same tone DOES reach a model whose cap covers harmonic 30.
    wide = _model(k_caps=(40,))
    wide.eval()
    with torch.no_grad():
        assert (wide(audio, cond) - wide(extra, cond)).abs().max() > 1e-2


def test_parameter_budget():
    """A refinement cell stack, not a trunk."""
    assert sum(p.numel() for p in _model().parameters()) < 500_000


# ─── registry + training seam ────────────────────────────────────────────────


def test_registry_and_codec_path():
    from models.registry import build_model

    model = build_model("hg_ckla_refiner", n_fft=N_FFT, hop_length=HOP, num_rotors=4)
    assert isinstance(model, HGCKLARefiner)

    frame_rate = (SR, HOP)
    batch = td.Frame(
        {
            "mixture": td.uniform(torch.randn(2, 32000), (SR, 1), dims=("batch", "time")),
            "rps_cond": td.uniform(
                torch.rand(2, 4, 63) * 60.0 + 50.0, frame_rate, dims=("batch", "rotor", "time")
            ),
            "rps": td.uniform(
                torch.rand(2, 4, 63) * 60.0 + 50.0, frame_rate, dims=("batch", "rotor", "time")
            ),
        }
    )
    codec = RPSPredictionCodec(frame_rate=frame_rate, use_cond=True)
    outputs = codec.call_model(model, codec.to_inputs(batch))
    loss = RPSMSELoss()(codec.to_frame(outputs, batch), batch)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
