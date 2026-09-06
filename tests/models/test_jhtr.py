"""Behavioral/numerical risks of the joint trajectory model, not learning claims.

Small widths/orders exercise the real DSP and all model paths on CPU. Passing
these checks says nothing about trained capture, oracle preservation or transfer.
"""

from __future__ import annotations

import math
from typing import TypedDict, Unpack, cast

import pytest
import torch
from torch import nn

from models.jhtr import (
    JHTR,
    _interloper_geometry,
    _LocalPatchEncoder,
    _modal_rates,
    _RefinementBlock,
)
from tracking.dsp import analytic_signal_tensor


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng():
        torch.manual_seed(37)
        yield


class _ModelOptions(TypedDict, total=False):
    n_fft: int
    hop_length: int
    num_rotors: int
    sample_rate: int
    n_blocks: int
    d_model: int
    n_heads: int
    k_max: int
    harmonic_chunk: int
    checkpoint_blocks: bool
    phase_products: bool
    reread: bool
    joint_slots: bool


def _model(**kwargs: Unpack[_ModelOptions]) -> JHTR:
    options: _ModelOptions = {
        "n_blocks": 2,
        "d_model": 16,
        "n_heads": 2,
        "k_max": 2,
        "harmonic_chunk": 1,
    }
    options.update(kwargs)
    return JHTR(**options)


def _tone(n: int = 16000) -> torch.Tensor:
    time = torch.arange(n) / 16000
    return (torch.cos(2 * math.pi * 160 * time) + 0.3 * torch.cos(2 * math.pi * 243 * time + 0.2))[
        None
    ]


def _conditioning(n: int = 16000) -> torch.Tensor:
    return torch.tensor([79.5, 80.5, 61.0, 100.0])[None, :, None].expand(1, 4, n // 512 + 1).clone()


def test_initializer_selects_a_mode_not_a_centroid_and_keeps_surrogate_gradients():
    candidates = torch.tensor([0.0, 60.0, 90.0])
    logits = torch.tensor([-100.0, math.log(0.49), math.log(0.51)], requires_grad=True)
    offsets = torch.zeros(1, requires_grad=True)
    rate = _modal_rates(logits, offsets, candidates)
    assert rate.item() == pytest.approx(90.0, abs=1e-5)
    rate.backward()
    assert logits.grad is not None
    assert offsets.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1] < 0 < logits.grad[2]
    assert offsets.grad.item() == pytest.approx(0.5)
    off = _modal_rates(torch.tensor([100.0, 0.0, 0.0]), torch.tensor([10.0]), candidates)
    assert off.item() == 0


@pytest.mark.parametrize("seconds,frames", [(1, 32), (4, 126), (8, 251)])
def test_silence_duplicate_and_off_seeds_keep_all_physical_frames(seconds, frames):
    model = _model(n_blocks=6, d_model=8, n_heads=2, k_max=1).eval()
    audio = torch.zeros(1, 16000 * seconds)
    cond = torch.tensor([0.0, 0.0, 80.0, 80.0])[None, :, None].expand(1, 4, frames)
    with torch.no_grad():
        out, diagnostics = model.forward_with_diagnostics(audio, cond)
    assert out.shape == (1, 4, frames)
    paths = diagnostics["trajectories"]
    assert paths.shape == (1, 7, 4, frames)
    torch.testing.assert_close(paths[:, 0], cond)
    torch.testing.assert_close(paths[:, -1], out)
    assert torch.isfinite(paths).all()
    assert ((paths >= 0) & (paths <= 150)).all()
    # Identical conditioning rows have identical computation, not positional
    # slot embeddings accidentally leaking out of the audio-only initializer.
    torch.testing.assert_close(paths[:, :, 0], paths[:, :, 1])
    torch.testing.assert_close(paths[:, :, 2], paths[:, :, 3])


@pytest.mark.parametrize("joint_slots", [True, False])
def test_conditional_rows_are_equivariant_through_every_block(joint_slots):
    model = _model(joint_slots=joint_slots).eval()
    audio, cond = _tone(), _conditioning()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        _, original = model.forward_with_diagnostics(audio, cond)
        _, permuted = model.forward_with_diagnostics(audio, cond[:, permutation])
    torch.testing.assert_close(
        permuted["trajectories"], original["trajectories"][:, :, permutation], atol=3e-5, rtol=1e-6
    )


def test_independent_slots_cannot_read_other_supplied_trajectories():
    model = _model(joint_slots=False).eval()
    audio, cond = _tone(), _conditioning()
    altered = cond.clone()
    altered[:, 1] = 31.5
    with torch.no_grad():
        out = model(audio, cond)
        other = model(audio, altered)
    torch.testing.assert_close(out[:, [0, 2, 3]], other[:, [0, 2, 3]], atol=1e-6, rtol=0)
    assert (out[:, 1] - other[:, 1]).abs().mean() > 40


def test_interlopers_include_high_foreign_orders_and_exclude_own_central_tooth():
    # A read of order 32 at 100 rev/s collides with FOREIGN ORDER 3200 of
    # the 1 rev/s candidate: an interloper list capped at reader k_max fails.
    rates = torch.tensor([100.0, 1.0, 0.0])[None, :, None]
    geometry = _interloper_geometry(rates, torch.tensor([32.0]), joint_slots=True)
    foreign = geometry[0, 0, 0, 0, 1, 0]
    torch.testing.assert_close(foreign[:2], torch.zeros(2))
    assert foreign[2].item() == pytest.approx(math.log1p(17))
    own = geometry[0, 0, 0, 0, 0, 2]
    torch.testing.assert_close(own[:2], torch.tensor([-100 / 128, 100 / 128]))
    assert own[2].item() == pytest.approx(math.log1p(2))
    assert geometry[0, 0, 0, 0, 2, :, 5].sum() == 0
    near_off = _interloper_geometry(
        torch.tensor([80.0, 0.50001])[None, :, None], torch.tensor([1.0]), joint_slots=True
    )
    assert torch.isfinite(near_off).all()
    # At order one there is no own lower tooth; do not report DC as one.
    assert near_off[0, 0, 0, 0, 0, 0, 3] == 0


def test_power_ablation_removes_phase_products_but_keeps_envelope_power():
    encoder = _LocalPatchEncoder(8)
    envelope = torch.ones(1, 1, 1, 3, 501, dtype=torch.complex64)
    phase = 0.04 * torch.arange(501)
    rotating = envelope * torch.polar(torch.ones_like(phase), phase)
    valid = torch.ones_like(envelope, dtype=torch.bool)
    reference = torch.ones(1)
    with torch.no_grad():
        power = encoder(envelope, valid, reference, False)
        phase_power = encoder(rotating, valid, reference, False)
        amplified = encoder(2 * envelope, valid, reference, False)
        full = encoder(envelope, valid, reference, True)
        phase_full = encoder(rotating, valid, reference, True)
    torch.testing.assert_close(power, phase_power, atol=2e-6, rtol=2e-6)
    assert (amplified - power).abs().max() > 1e-4
    assert (phase_full - full).abs().max() > 1e-4


def test_local_patch_has_exact_129_sample_support_and_frame_centres():
    encoder = _LocalPatchEncoder(4)
    with torch.no_grad():
        for layer in encoder.modules():
            if isinstance(layer, nn.Conv1d):
                layer.weight.fill_(0.03)
                assert layer.bias is not None
                layer.bias.zero_()
        base = torch.ones(1, 1, 1, 3, 501, dtype=torch.complex64)
        valid = torch.ones_like(base, dtype=torch.bool)
        reference = torch.ones(1)
        original = encoder(base, valid, reference, False)[..., 15, :]
        for offset in (-65, -64, 64, 65):
            changed = base.clone()
            changed[..., 16 * 15 + offset] = 2
            observed = encoder(changed, valid, reference, False)[..., 15, :]
            if abs(offset) == 65:
                torch.testing.assert_close(observed, original, atol=0, rtol=0)
            else:
                assert (observed - original).min() > 0


def test_patch_backward_retains_the_waveform_conditioned_carrier_path():
    model = _model(n_blocks=1)
    audio = _tone().requires_grad_()
    analytic = analytic_signal_tensor(audio, pad_samples=model.pad_samples)
    reference = (
        analytic[:, model.pad_samples : model.pad_samples + audio.shape[-1]].abs().square().mean(-1)
    )
    rate = torch.tensor(79.5, requires_grad=True)
    orders = torch.tensor([2.0])

    def observation(value):
        rates = value.expand(1, 1, 32)
        block = cast(_RefinementBlock, model.blocks[0])
        tokens = model._read_chunk(analytic, rates, reference, orders, 16000, block.patch)
        weight = torch.linspace(-1, 1, tokens.shape[-1])[None, None, None, None, :]
        return (tokens[..., 12:20, :] * weight).sum()

    value = observation(rate)
    rate_grad, audio_grad = torch.autograd.grad(value, (rate, audio))
    with torch.no_grad():
        finite_difference = (observation(rate + 0.01) - observation(rate - 0.01)) / 0.02
    assert torch.isfinite(audio_grad).all() and audio_grad.abs().sum() > 0
    assert torch.isfinite(rate_grad) and rate_grad.abs() > 1e-6
    torch.testing.assert_close(rate_grad, finite_difference, atol=2e-3, rtol=0.08)


@pytest.mark.parametrize("reread", [True, False])
def test_checkpointed_final_loss_trains_all_untied_readers_and_global_memory(reread):
    model = _model(reread=reread).train()
    audio = _tone().requires_grad_()
    cond = _conditioning().requires_grad_()
    out, diagnostics = model.forward_with_diagnostics(audio, cond)
    (out - cond.detach() - 0.7).square().mean().backward()
    assert diagnostics["trajectories"].requires_grad
    assert audio.grad is not None
    assert cond.grad is not None
    assert torch.isfinite(audio.grad).all() and audio.grad.abs().sum() > 0
    assert torch.isfinite(cond.grad).all()
    for module in model.blocks:
        block = cast(_RefinementBlock, module)
        gradient = cast(nn.Conv1d, block.patch.network[0]).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    gradient = cast(nn.Conv2d, model.coarse.order_encoder[0]).weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_audio_only_locator_receives_final_output_gradients():
    model = _model(n_blocks=1).train()
    out = model(_tone())
    # The locator is discrete forward, but its documented surrogate and bounded
    # sub-bin branch must receive the inherited final-output objective.
    (out - 50).square().mean().backward()
    for parameter in (model.slot_queries, model.initial_query.weight, model.initial_offset.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_updates_have_no_seed_tube_and_repeated_calls_reset_state():
    model = _model(n_blocks=6, k_max=1).eval()
    with torch.no_grad():
        for module in model.blocks:
            block = cast(_RefinementBlock, module)
            block.update.weight.zero_()
            assert block.update.bias is not None
            block.update.bias.fill_(2)
        cond = _conditioning()
        out, diagnostics = model.forward_with_diagnostics(_tone(), cond)
        repeated = model(_tone(), cond)
    torch.testing.assert_close(out, cond + 12, atol=1e-5, rtol=0)
    expected = cond[:, None] + 2 * torch.arange(7)[None, :, None, None]
    torch.testing.assert_close(diagnostics["trajectories"], expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(repeated, out, atol=0, rtol=0)


def test_cpu_neural_amp_keeps_finite_rates_and_backward():
    model = _model(n_blocks=1).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(_tone(), _conditioning())
        loss = (out - 70).square().mean()
    loss.backward()
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    block = cast(_RefinementBlock, model.blocks[0])
    gradient = cast(nn.Conv1d, block.patch.network[0]).weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
