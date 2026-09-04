"""The learned partial-observation emission must CONTAIN the corner it replaces.

The zero-parameter corner (`head_mode="classical"`, eight microphones
power-averaged, a 15-bin floor) reads real DREGON cruise at 1.49 rev/s, better
than every trained model on that protocol. A learned emission is only worth
having if training can leave that corner, which means it must START there —
otherwise a run that helps cannot be told from a run that merely moved.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch

from data_processing.comb_bench import comb_clip
from models.comb_slots import _SOFTPLUS_ONE, PartialEmission, SlotCombNet

# The C1 corner, at a grid and harmonic count small enough for a unit test.
# `floor_hz=60` is part of the corner, not a convenience: it sets the 15-bin
# running median the eight-microphone result was measured with.
KW = dict(n_grid=180, k_max=16, floor_hz=60.0, use_checkpoint=False, n_iter=0, multichannel=True)


def eight_channel(seed: int = 7, spread: float = 11.0, n: int = 32000):
    """One static comb heard by eight microphones: ``(8, N)`` and its labels.

    Per-mic gains plus INDEPENDENT noise, because that is what the emission's
    channel axis is for: the residual is 90% per-mic incoherent on DREGON, so
    eight identical copies would test nothing.
    """
    a, rps, _ = comb_clip(seed=seed, spread=spread)
    mono = torch.tensor(np.asarray(a, dtype=np.float32)[:n])
    gains = torch.tensor([1.0, 0.8, 1.3, 0.6, 1.1, 0.9, 1.4, 0.7])[:, None]
    g = torch.Generator().manual_seed(seed)
    x = mono[None] * gains + 0.02 * torch.randn(8, mono.shape[0], generator=g)
    return x.float(), np.asarray(rps, dtype=np.float32)


def _net(**kw):
    return SlotCombNet(**{**KW, **kw}).eval()


def _emit(net) -> PartialEmission:
    assert isinstance(net.emit, PartialEmission)
    return net.emit


def test_partial_starts_at_the_classical_score():
    """At initialization the partial emission IS the classical mean over orders.

    The only departure is the weight the eight per-mic gates hold —
    ``8 sigmoid(-8) / (sigmoid(8) + 8 sigmoid(-8))`` = 2.7e-3 of the total —
    which is why the tolerance with `channels` is looser than without it.
    Measured on this clip: 2.6e-4 without the mics, 2.6e-4 with them; on a real
    8 s DREGON cruise clip, 1.3e-4 and 6.3e-4.
    """
    x, _ = eight_channel()
    base = _net()
    with torch.no_grad():
        s0, _ = base.forward(x)

    for parts, tol in (
        (("reliability", "empty_tooth", "floor_mix"), 1e-3),
        (("reliability", "channels", "empty_tooth", "floor_mix"), 2e-3),
    ):
        net = _net(emission="partial", parts=parts)
        with torch.no_grad():
            s1, _ = net.forward(x)
        assert float((s1 - s0).abs().max()) < tol, parts


def test_partial_decodes_the_same_path():
    """Same grid path, not merely a similar score."""
    x, _ = eight_channel()
    base = _net()
    net = _net(emission="partial")
    with torch.no_grad():
        d0 = base.decode(x, subgrid=False, octave=False, relocate=True)
        d1 = net.decode(x, subgrid=False, octave=False, relocate=True)
    assert torch.equal(d0, d1)


def test_batched_spectrum_matches_per_item():
    """``(B, C, N)`` must be B items of C microphones, not one item of B*C."""
    net = _net(emission="partial")
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 3, 8000, generator=g)
    pc, mean = net.spectrum(x, per_channel=True)
    assert pc.shape[:2] == (2, 3) and mean.shape[0] == 2
    for i in range(2):
        pci, mi = net.spectrum(x[i], per_channel=True)
        assert torch.allclose(pc[i : i + 1], pci, atol=1e-6)
        assert torch.allclose(mean[i : i + 1], mi, atol=1e-6)


def test_backward_reaches_every_new_parameter():
    """Every part of the emission must be on the CRF loss's path.

    `mlp.0` is checked with the output layer perturbed away from zero: the last
    layer is zero-initialized on purpose (that is what makes the corner exact),
    and a zero output layer sends exactly zero gradient to the layer below it,
    at initialization only.
    """
    x, rps = eight_channel(n=16000)
    net = _net(emission="partial", use_checkpoint=True)
    net.train()
    with torch.no_grad():
        cast(torch.nn.Linear, _emit(net).mlp[2]).weight.fill_(0.01)
    n_t = x.shape[-1] // net.hop_length + 1
    gt = torch.as_tensor(rps[:, :n_t])[None]
    net.loss(x[None], gt).backward()
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, name
        assert float(p.grad.abs().max()) > 0.0, name


def test_empty_tooth_charges_the_sub_harmonic():
    """The octave term must cost the half rate more than the truth.

    A comb at ``r/2`` covers every line of the comb at ``r`` and fills the gaps
    with whatever is there, so evidence at the predicted lines cannot reject it.
    `relu(tau - z)` charges the lines that land on the floor, which the odd
    orders of the half rate do and the truth's do not.
    """
    x, rps = eight_channel(seed=5, spread=0.0, n=32000)
    off = _net(emission="partial", parts=("empty_tooth",))
    on = _net(emission="partial", parts=("empty_tooth",))
    with torch.no_grad():
        _emit(on).lam_raw.fill_(_SOFTPLUS_ONE)  # softplus(.) == 1
        _emit(on).tau.fill_(math.log(2.0))
        _emit(off).tau.fill_(math.log(2.0))
        s_off, _ = off.forward(x)
        s_on, _ = on.forward(x)
    grid = off.grid.numpy()
    r = float(rps.mean())
    i_true = int(np.abs(grid - r).argmin())
    i_half = int(np.abs(grid - r / 2.0).argmin())
    drop = (s_off - s_on)[0, 0].numpy()  # slot 0, the un-notched one
    assert drop[i_half].mean() > drop[i_true].mean() > 0.0
