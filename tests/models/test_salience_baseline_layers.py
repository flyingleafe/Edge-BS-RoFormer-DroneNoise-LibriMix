"""The two multi-pitch salience baselines on the lossless per-rotor readout.

`models.salience_crf` measured the framework's shared salience map as an
encode/decode pair on real training telemetry and found it loses 8.24 rev/s on
a PERFECT target, against 2.22e-16 for Gaussian per-rotor layers read by a CRF
plus a log-parabolic vertex fit. The three harmonic ports already moved onto
that pair; these tests lock the same move for `LateDeepSalience` and
`BasicPitchSalience`.

Two properties carry the whole change:

1. The layers reach the loss and the decoder in the shape both expect, and the
   INPUT does not move — the widening is confined to the output head.
2. ``n_maps == 1`` is the OLD MODEL. ``zoo.frame_model`` loads a checkpoint
   with the default ``strict=True``, thus the head parameter names and shapes
   must be exactly what they were, and the decode must still be the shared-map
   one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from losses.salience_layers import LayerPITSalienceBCELoss, layer_pit_bce
from models.harmonic_ports.layer_readout import split_maps
from models.salience_crf import gaussian_layer_target
from models.salience_rps import BasicPitchSalience, LateDeepSalience

# The output block of conf/model/{multif0,basic_pitch}_salience_l4.yaml.
OUT = dict(superres_out=True, out_fmin=0.0, out_fmax=150.0, out_bins=300)
# A small grid keeps the CRF's O(span * G * T) decode cheap; the port's own
# 0-150/300 grid is exercised by the shape test.
SMALL = dict(superres_out=True, out_fmin=0.0, out_fmax=50.0, out_bins=100)

BASELINES = [LateDeepSalience, BasicPitchSalience]
# One second at 16 kHz on both front ends' hop-256 time grid.
N_SAMPLES, N_FRAMES = 16000, 63

# The head parameter names and shapes a pre-`n_maps` checkpoint holds. The
# names must survive because `zoo.frame_model` loads with strict=True.
SINGLE_MAP_HEAD = {
    LateDeepSalience: {"cnn.squishy.1.weight": (1, 8, 1, 1)},
    BasicPitchSalience: {
        "net.contour_out.weight": (1, 8, 5, 5),
        "net.contour_out.bias": (1,),
    },
}
SUPERRES_HEAD = {
    "head.net.0.weight": (32, 1, 5, 1),
    "head.net.6.weight": (1, 32, 1, 1),
    "head.net.6.bias": (1,),
}


# The parameters the widening creates, per model — what a gradient must reach.
HEAD_PARAMS = {
    LateDeepSalience: ["cnn.squishy.1.weight", "head.net.6.weight"],
    BasicPitchSalience: ["net.contour_out.weight", "head.net.6.weight"],
}


def _build(cls, **kw):
    return cls(n_fft=2048, hop_length=512, num_rotors=4, **kw)


def _tracks(seed=0, n=2, r=4, t=24, step=0.4):
    """Trajectories whose per-frame slew stays well inside the transition band."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.standard_normal((n, 1, t)) * step, axis=-1) + 25.0
    off = rng.uniform(-3.0, 3.0, size=(n, r, 1))
    return torch.tensor(np.clip(base + off, 0.0, 49.0))


@pytest.mark.parametrize("cls", BASELINES)
def test_four_layers_ride_the_output_axis(cls):
    """The wire format stays 3-D; the layers are stacked along it."""
    m = _build(cls, n_maps=4, **OUT).eval()
    y = m(torch.randn(2, N_SAMPLES))
    assert y.shape == (2, 4 * 300, N_FRAMES)
    assert split_maps(y, 4).shape == (2, 4, 300, N_FRAMES)
    assert torch.equal(split_maps(y, 4).reshape(2, 1200, N_FRAMES), y)


@pytest.mark.parametrize("cls", BASELINES)
def test_one_map_keeps_the_old_head_and_the_old_decoder(cls):
    """`n_maps == 1` is the model that existed before this option.

    A pre-change reference cannot be constructed inside one test, so the
    checkpoint contract is asserted directly: the head parameters keep their
    names and their single-channel shapes, the output keeps its `(B, G, T)`
    shape, and `predict_rps` still falls through to the shared-map Hungarian
    decoder rather than the CRF.
    """
    m = _build(cls).eval()
    sd = m.state_dict()
    for name, shape in SINGLE_MAP_HEAD[cls].items():
        assert name in sd, f"{name} disappeared from the state dict"
        assert tuple(sd[name].shape) == shape

    m_sr = _build(cls, **OUT).eval()
    sd_sr = m_sr.state_dict()
    for name, shape in SUPERRES_HEAD.items():
        assert name in sd_sr, f"{name} disappeared from the state dict"
        assert tuple(sd_sr[name].shape) == shape

    assert m_sr(torch.randn(1, N_SAMPLES)).shape == (1, 300, N_FRAMES)
    # A checkpoint of the old model must still load, names and shapes intact.
    _build(cls, **OUT).load_state_dict(sd_sr, strict=True)
    # n_maps == 1 must fall through to SalienceRPSPredictor's Hungarian path.
    assert m_sr.predict_rps(torch.randn(1, N_SAMPLES)).shape == (1, 4, 32)


@pytest.mark.parametrize("cls", BASELINES)
def test_per_rotor_layers_need_a_linear_output_grid(cls):
    with pytest.raises(ValueError, match="LINEAR output grid"):
        _build(cls, n_maps=4)


@pytest.mark.parametrize("cls", BASELINES)
def test_the_round_trip_through_the_crf_is_exact(cls):
    """A PERFECTLY trained model's own target, through its own decode.

    Under `losses.LayerPITSalienceBCELoss` the optimum is
    ``sigmoid(z) == target``, so ``logit(target)`` is what a perfect model
    emits. ``LayerCRFReadout`` takes ``log sigmoid(z)`` of it, which is the log
    of a Gaussian, which a three-point parabolic fit inverts exactly.
    """
    m = _build(cls, n_maps=4, **SMALL).double().eval()
    rps = _tracks()
    tgt = gaussian_layer_target(rps, m.out_freqs, sigma_bins=1.0)
    ideal = torch.logit(tgt.clamp(1e-15, 1 - 1e-15))
    rec = m.decode_salience(ideal.reshape(ideal.shape[0], -1, ideal.shape[-1]))
    assert float((rec - rps).abs().max()) < 1e-6


@pytest.mark.parametrize("cls", BASELINES)
def test_predict_rps_recovers_a_four_rotor_comb_through_the_crf(cls):
    """The deployed path: audio-shaped call, layers set to the Gaussian target.

    `predict_rps` runs the trunk, so the trunk is bypassed here — the readout
    is what this test locks, exactly as `test_harmonic_port_layers` does for
    the ports. The decode returns the STFT grid, hence the interpolation.
    """
    m = _build(cls, n_maps=4, **SMALL).double().eval()
    n_grid = m.num_grid_frames(N_SAMPLES)
    rps = torch.full((1, 4, n_grid), 0.0, dtype=torch.float64)
    for i, speed in enumerate([18.0, 24.5, 31.25, 44.0]):
        rps[0, i] = speed
    tgt = gaussian_layer_target(rps, m.out_freqs, sigma_bins=1.0)
    ideal = torch.logit(tgt.clamp(1e-15, 1 - 1e-15)).reshape(1, -1, n_grid)

    m.forward = lambda audio, _y=ideal: _y  # type: ignore[method-assign]
    got = m.predict_rps(torch.randn(1, N_SAMPLES))
    assert got.shape == (1, 4, N_SAMPLES // 512 + 1)
    want = torch.tensor([18.0, 24.5, 31.25, 44.0], dtype=torch.float64)[:, None]
    assert float((got[0] - want).abs().max()) < 1.0


@pytest.mark.parametrize("cls", BASELINES)
def test_one_optimizer_step_decreases_the_layer_loss(cls):
    torch.manual_seed(0)
    m = _build(cls, n_maps=4, **SMALL).train()
    loss_fn = LayerPITSalienceBCELoss(out_fmin=0.0, out_fmax=50.0, out_bins=100)
    audio = torch.randn(2, N_SAMPLES)
    rps = _tracks(t=32).float()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def step_loss() -> torch.Tensor:
        layers, tgt = loss_fn.layers_and_target(m(audio), rps)
        return layer_pit_bce(layers, tgt)

    before = step_loss()
    opt.zero_grad()
    before.backward()
    opt.step()
    # Stay in train mode: the BatchNorm layers use batch statistics there, so
    # both readings are the same function of the same batch. An eval-mode
    # reading would score a DIFFERENT function, because one step barely moves
    # the running statistics off their initial values.
    with torch.no_grad():
        after = step_loss()
    assert float(after) < float(before)

    # The widened head itself must be on the graph. (The whole trunk is not
    # checked: `BasicPitchSalience` runs the contour branch only, thus the note
    # and onset branches carry no gradient — that predates this option.)
    for name in HEAD_PARAMS[cls]:
        grad = dict(m.named_parameters())[name].grad
        assert grad is not None and torch.any(grad != 0), f"no gradient reached {name}"


@pytest.mark.parametrize("cls", BASELINES)
def test_the_widening_touches_the_head_only(cls):
    """Everything but the two 1x1 convolutions keeps its shape."""
    one = _build(cls, **OUT).state_dict()
    four = _build(cls, n_maps=4, **OUT).state_dict()
    assert set(one) == set(four)
    widened = {k for k in one if tuple(one[k].shape) != tuple(four[k].shape)}
    expected = set(SINGLE_MAP_HEAD[cls]) | set(SUPERRES_HEAD)
    assert widened == expected, f"unexpected shape change: {widened ^ expected}"


def test_the_layer_readout_inverts_the_gaussian_target_on_the_output_grid():
    """The grid the two `_l4` configs declare, read by the log-parabolic fit."""
    from metrics.salience_layers import peak_readout
    from models.multif0.utils import linear_freq_grid

    grid = linear_freq_grid(0.0, 150.0, 300)
    rps = _tracks() * 2.0
    tgt = gaussian_layer_target(rps, grid, sigma_bins=1.0)
    ideal = torch.logit(tgt.clamp(1e-15, 1 - 1e-15))
    assert float((peak_readout(F.logsigmoid(ideal), grid) - rps).abs().max()) < 1e-9
