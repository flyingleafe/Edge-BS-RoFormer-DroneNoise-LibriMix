"""Tests for tasks.noise_generation (batched geometry_to_rel_pos) and the
NoiseGenerationCodec fix (tasks.codecs) — the real bug REPLICATION.md § E1/E2/E3
documented: the codec used to pass mic_pos/rotor_pos/drone_id straight
through as kwargs, but PositionalHarmonicNoiseGen.forward wants a
precomputed rel_pos tensor and a conditioning code z. Also covers
models.registry.build_noise_gen_model's cond_dim>0 DroneCodebook wrapping."""

from __future__ import annotations

import numpy as np
import tdseries as td
import torch

from models.generative.codebook import DroneCodebook, geometry_to_rel_pos
from models.registry import build_noise_gen_model
from tasks.codecs import NoiseGenerationCodec

# ─── geometry_to_rel_pos: numpy (unbatched) + torch (batched) ────────────────


def test_geometry_to_rel_pos_numpy_unbatched():
    mic = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    rotor = np.array([[0.0, 0.0, 0.0]])  # (1, 3)
    rel = geometry_to_rel_pos(mic, rotor)
    assert rel.shape == (2, 1, 3)
    np.testing.assert_allclose(rel[0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rel[1, 0], [0.0, 1.0, 0.0])


def test_geometry_to_rel_pos_torch_batched():
    mic = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # (B=1, M=2, 3)
    rotor = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])  # (B=1, R=2, 3)
    rel = geometry_to_rel_pos(mic, rotor)
    assert rel.shape == (1, 2, 2, 3)
    torch.testing.assert_close(rel[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(rel[0, 1, 1], torch.tensor([-1.0, 0.0, 0.0]))


def test_geometry_to_rel_pos_rejects_mixed_batch_dims():
    import pytest

    mic = torch.zeros(1, 2, 3)  # batched
    rotor = torch.zeros(2, 3)  # unbatched
    with pytest.raises(ValueError):
        geometry_to_rel_pos(mic, rotor)


def test_geometry_to_rel_pos_torch_is_differentiable():
    mic = torch.zeros(1, 1, 3, requires_grad=True)
    rotor = torch.zeros(1, 1, 3)
    rel = geometry_to_rel_pos(mic, rotor)
    rel.sum().backward()
    assert mic.grad is not None


# ─── NoiseGenerationCodec: rel_pos fix + optional conditioning ────────────────


class _FakeGenerator(torch.nn.Module):
    """Records the exact args/kwargs it was called with, returns zeros."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def forward(self, rps, rel_pos, drone_names=None, **kwargs):
        self.calls.append((rps, rel_pos, drone_names, kwargs))
        b, m = rel_pos.shape[0], rel_pos.shape[1]
        return torch.zeros(b, m, rps.shape[-1])


def _batched_time_series(
    data: torch.Tensor, dims: tuple[str, ...], *, rate=(16000, 1)
) -> td.Series:
    idx = td.GridIndex.create(rate, int(data.shape[-1]), t_start=0)
    return td.Series(data, dims, {"time": idx})


def _noise_gen_batch(*, batch=2, mics=3, rotors=4, t=100) -> td.Frame:
    rps = torch.rand(batch, rotors, t)
    mic_pos = torch.rand(batch, mics, 3)
    rotor_pos = torch.rand(batch, rotors, 3)
    audio = torch.rand(batch, mics, t)
    return td.Frame(
        {
            "rps": _batched_time_series(rps, ("batch", "rotor", "time")),
            "mic_pos": td.wrap(mic_pos, dims=("batch", "mic", None)),
            "rotor_pos": td.wrap(rotor_pos, dims=("batch", "rotor", None)),
            "audio": _batched_time_series(audio, ("batch", "mic", "time")),
            "meta": td.Frame({"drone": ["dregon", "michaels"][:batch]}),
        }
    )


def test_noise_generation_codec_computes_rel_pos_not_raw_positions():
    batch = _noise_gen_batch(batch=2, mics=3, rotors=4)
    codec = NoiseGenerationCodec()
    inputs = codec.to_inputs(batch)

    assert "mic_pos" not in inputs and "rotor_pos" not in inputs  # not passed raw
    assert inputs["rel_pos"].shape == (2, 3, 4, 3)  # (B, M, R, 3)
    expected = geometry_to_rel_pos(
        torch.as_tensor(batch["mic_pos"].data), torch.as_tensor(batch["rotor_pos"].data)
    )
    torch.testing.assert_close(inputs["rel_pos"], expected)


def test_noise_generation_codec_unconditioned_call_model():
    batch = _noise_gen_batch()
    codec = NoiseGenerationCodec()
    model = _FakeGenerator()
    inputs = codec.to_inputs(batch)
    codec.call_model(model, inputs)

    rps, rel_pos, drone_names, kwargs = model.calls[0]
    assert drone_names is None
    assert kwargs == {}
    assert rel_pos.shape[-1] == 3


def test_noise_generation_codec_conditioned_resolves_drone_names_from_meta():
    batch = _noise_gen_batch(batch=2)
    codec = NoiseGenerationCodec(conditioned=True)
    model = _FakeGenerator()
    inputs = codec.to_inputs(batch)
    assert inputs["drone_names"] == ["dregon", "michaels"]

    codec.call_model(model, inputs)
    _rps, _rel_pos, drone_names, _kwargs = model.calls[0]
    assert drone_names == ["dregon", "michaels"]


def test_noise_generation_codec_conditioned_falls_back_to_default_drone_when_no_meta():
    batch = _noise_gen_batch(batch=2)
    batch = batch.with_entry("meta", td.Frame({}))  # no "drone" key
    codec = NoiseGenerationCodec(conditioned=True, default_drone="dregon")
    inputs = codec.to_inputs(batch)
    assert inputs["drone_names"] == ["dregon", "dregon"]


def test_noise_generation_codec_return_dict_exposes_harm_and_noise_amps():
    batch = _noise_gen_batch(batch=1, mics=2, rotors=4)
    codec = NoiseGenerationCodec(return_dict=True)
    outputs = {
        "audio": torch.rand(1, 2, 100),
        "harm_amps": torch.rand(1, 4, 1, 8, 10),
        "noise_amps": torch.rand(1, 4, 16, 10),
    }
    frame = codec.to_frame(outputs, batch)
    assert set(frame) == {"audio", "harm_amps", "noise_amps"}
    assert frame["harm_amps"].dims == ("batch", "rotor", None, None, None)
    assert frame["noise_amps"].dims == ("batch", "rotor", None, None)


def test_noise_generation_codec_to_frame_plain_tensor_output():
    batch = _noise_gen_batch(batch=1, mics=2, rotors=4)
    codec = NoiseGenerationCodec()
    outputs = torch.rand(1, 2, 100)
    frame = codec.to_frame(outputs, batch)
    assert set(frame) == {"audio"}
    assert frame["audio"].dims == ("batch", "mic", "time")


# ─── build_noise_gen_model: cond_dim>0 wraps a trainable DroneCodebook ────────


def test_build_noise_gen_model_unconditioned_returns_bare_generator():
    model = build_noise_gen_model("positional_harmonic_gen", n_harmonics=4, cond_dim=0)
    assert not hasattr(model, "codebook")


def test_build_noise_gen_model_conditioned_wraps_codebook_reachable_from_parameters():
    model = build_noise_gen_model(
        "positional_harmonic_gen",
        n_harmonics=4,
        cond_dim=8,
        drone_names=["dregon", "michaels"],
    )
    assert hasattr(model, "codebook")
    assert isinstance(model.codebook, DroneCodebook)

    # Codebook params must be reachable from model.parameters() -- this is
    # exactly what makes them trainable through the unified single-optimizer
    # training loop (training.loop.run_training builds
    # get_optimizer(model, ...) over model.parameters() only) and persisted
    # through the single-model checkpoint contract (torch.save(model.state_dict())).
    codebook_param_ids = {id(p) for p in model.codebook.parameters()}
    model_param_ids = {id(p) for p in model.parameters()}
    assert codebook_param_ids <= model_param_ids
    assert "codebook.codes.dregon" in model.state_dict()


def test_codebook_conditioned_noise_gen_forward_backprops_to_codebook():
    # Isolates _CodebookConditionedNoiseGen's own composition logic (name ->
    # z lookup -> generator call) from PositionalHarmonicNoiseGen's FFT-heavy
    # internals (backward through repeated rfft/irfft is flaky under this
    # sandbox's single-threaded MKL config at some sizes -- unrelated to what
    # this test actually verifies).
    from models.registry import _CodebookConditionedNoiseGen

    class _ZSensitiveGenerator(torch.nn.Module):
        def forward(self, rps, rel_pos, z):
            # output depends on z so a gradient reaches the codebook.
            return rps.unsqueeze(1) * z.view(*z.shape, 1, 1).sum(dim=1, keepdim=True)

    codebook = DroneCodebook(4, names=["dregon", "michaels"])
    model = _CodebookConditionedNoiseGen(_ZSensitiveGenerator(), codebook)

    rps = torch.rand(2, 4, 10)
    rel_pos = torch.rand(2, 3, 4, 3)
    out = model(rps, rel_pos, ["dregon", "michaels"])
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in codebook.parameters())


def test_build_noise_gen_model_conditioned_requires_drone_names():
    import pytest

    with pytest.raises(ValueError, match="drone_names"):
        build_noise_gen_model("positional_harmonic_gen", n_harmonics=4, cond_dim=8)
