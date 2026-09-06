"""The v2 emission must CONTAIN the partial emission it extends.

`PartialEmission` is measured: every trained arm of the C1 campaign takes DREGON
cruise from 1.49 to 0.86-1.04 rev/s. The v2 groups
(`docs/slot-comb-v2-design.md` 3.4, 3.5 and 3.7) are only worth having if
training can leave that arm, which means they must START at it. The first test
locks that to 1e-6 on the score tensor; the others show that each new group
does what it was added for.

These tests build `SlotCombNet` objects, which hold a mask bank over the whole
band. They cap the thread count for that reason.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
import torch

from data_processing.comb_bench import comb_clip
from models.comb_slots import PARTIAL_PARTS, CombMaskBank, SlotCombNet
from models.comb_slots_emission_v2 import (
    PARTIAL_V2_PARTS,
    LearnedCombMaskBank,
    OffsetCombGather,
    PartialEmissionV2,
    warm_start,
)

torch.set_num_threads(4)

# The C1 corner, at a grid and harmonic count small enough for a unit test —
# the same keywords `test_comb_slots_partial.py` uses, so the two files measure
# the same object.
KW = dict(n_grid=180, k_max=16, floor_hz=60.0, use_checkpoint=False, n_iter=0, multichannel=True)


def eight_channel(seed: int = 7, spread: float = 11.0, n: int = 32000, centre: float = 75.0):
    """One static comb heard by eight microphones: ``(8, N)`` and its labels.

    A copy of the helper in `test_comb_slots_partial.py`. The two files must be
    readable and runnable on their own, and the helper is eight lines.
    """
    a, rps, _ = comb_clip(seed=seed, spread=spread, centre=centre)
    mono = torch.tensor(np.asarray(a, dtype=np.float32)[:n])
    gains = torch.tensor([1.0, 0.8, 1.3, 0.6, 1.1, 0.9, 1.4, 0.7])[:, None]
    g = torch.Generator().manual_seed(seed)
    x = mono[None] * gains + 0.02 * torch.randn(8, mono.shape[0], generator=g)
    return x.float(), np.asarray(rps, dtype=np.float32)


def _net(**kw) -> SlotCombNet:
    return SlotCombNet(**{**KW, **kw}).eval()


def _emit(net: SlotCombNet) -> PartialEmissionV2:
    assert isinstance(net.emit, PartialEmissionV2)
    return net.emit


def test_v2_starts_at_the_partial_emission():
    """With every v2 part on, the initialized score IS the partial emission's.

    Measured per group on this clip, as the maximum absolute difference over the
    ``(B, R, G, T)`` score tensor:

        no v2 part            0
        gap                   2.384e-7   softplus(-16) times the gap warp
        cross_order           0          the weight divides by softplus(0)
        read_width_learned    1.192e-7   one last-place unit of the smoothed read
        claim_width_learned   0          the bank is bit-identical
        all four              2.384e-7   the gap term, and nothing else

    The gap term is the only one that is not exact, and it is a CHOICE: the
    design initializes `mu` at -8, which costs 2.3e-4 here. See the module
    docstring on why -16 is the default and -2 is what a run that must learn the
    group should use.
    """
    x, _ = eight_channel()
    base = _net(emission="partial", parts=PARTIAL_PARTS)
    with torch.no_grad():
        s0, _ = base.forward(x)
    for parts in (PARTIAL_PARTS, PARTIAL_V2_PARTS):
        net = _net(emission="v2", parts=parts)
        with torch.no_grad():
            s1, _ = net.forward(x)
        assert float((s1 - s0).abs().max()) < 1e-6, parts


def test_learned_claim_bank_reproduces_the_fixed_one():
    """The learned bank is the fixed bank at its initial width.

    It is built from the MINIMUM distance to a harmonic instead of a maximum
    over Gaussians, which is the same function of the width and turns a chunked
    pass over 250 harmonics per forward into one exponential over ``(G, F)``.
    """
    net = _net(emission="v2", parts=PARTIAL_V2_PARTS)
    learned = cast(LearnedCombMaskBank, net.masks)
    fixed = CombMaskBank(net.grid, net.n_fft, net.sr, net.mask_k_max, net.gather.f_max, 1.5)
    assert float((fixed.bank - learned.bank()).abs().max()) < 1e-6


def test_backward_reaches_every_new_parameter():
    """One optimizer step, and every new group has a finite, non-zero gradient.

    The two zero-initialized output layers are perturbed first, exactly as
    `test_comb_slots_partial.py` perturbs `mlp[2]`: a zero output layer sends
    exactly zero gradient to the layer below it, at initialization only, and
    that is what makes the corner exact rather than a defect.

    The read width is raised to 0.7 bins by `warm_start`. At the default 0.15
    the read is a delta to 2.2e-10 and so is its derivative, so `s0` and `s1`
    receive a gradient near 1e-13 — real, but not one that trains. That is the
    honest property of the group and the reason `warm_start` exists.
    """
    x, rps = eight_channel(n=16000)
    net = _net(emission="v2", parts=PARTIAL_V2_PARTS, use_checkpoint=True)
    net.train()
    emit = warm_start(_emit(net), gap_mu=-2.0, read_sigma=0.7)
    with torch.no_grad():
        cast(torch.nn.Linear, emit.mlp[2]).weight.fill_(0.01)
        cast(torch.nn.Conv1d, emit.cross[4]).weight.fill_(0.01)
    n_t = x.shape[-1] // net.hop_length + 1
    gt = torch.as_tensor(rps[:, :n_t])[None]
    net.loss(x[None], gt).backward()
    seen = {"emit.mu", "emit.w_gap", "emit.alpha", "emit.s0", "emit.s1", "masks.width_raw"}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
        assert float(p.grad.abs().max()) > 0.0, name
        seen.discard(name)
    assert not seen, f"parameters missing from the model: {sorted(seen)}"
    opt = torch.optim.SGD(net.parameters(), lr=1e-4)
    opt.step()
    for name, p in net.named_parameters():
        assert bool(torch.isfinite(p).all()), name


def test_gap_charges_the_multiple():
    """The multiple discriminator must cost twice the rate more than the rate.

    A comb at `2 r` has every tooth of the truth, so no test on the teeth can
    reject it — the C1 campaign's remaining FLY124 error is two hover clips read
    at exactly twice the rate. Its GAPS are the odd harmonics of the truth and
    are full, which is what this term charges for.

    Measured on this clip (four rotors at 40 rev/s, so the multiple at 80 is on
    the grid), as the score at `r` minus the score at `2 r` per frame, averaged
    over frames. A positive margin means the truth wins:

        mu at the default -16   0.6627
        mu set to +2            0.9037

    The charge is worth 0.24 nats of margin here, against the 0.03 nats that
    separate the truth from its best decoy on real recordings.
    """
    x, rps = eight_channel(seed=5, spread=0.0, n=32000, centre=40.0)
    net = _net(emission="v2", parts=("gap",))
    grid = net.grid.numpy()
    r = float(rps.mean())
    i_true = int(np.abs(grid - r).argmin())
    i_two = int(np.abs(grid - 2.0 * r).argmin())
    with torch.no_grad():
        s_off, _ = net.forward(x)
        _emit(net).mu.fill_(2.0)
        s_on, _ = net.forward(x)

    # The rotors wander by 1.5 rev/s, which is four steps of this coarse test
    # grid, so each rate is read as the best candidate in a five-step window per
    # frame. A fixed index would measure the wander and not the charge.
    def margin(s):
        def peak(i):
            return s[max(0, i - 5) : i + 6].max(axis=0)

        return float((peak(i_true) - peak(i_two)).mean())

    m_off = margin(s_off[0, 0].numpy())
    m_on = margin(s_on[0, 0].numpy())
    assert m_on > m_off, f"margin {m_off:.4f} -> {m_on:.4f}"
    assert m_on > 0.0


def test_comb_conditioned_floor_at_alpha_zero_is_the_gap_power():
    """At ``alpha = 0`` the floor IS the power read between the teeth.

    The running median is corrupted below about 20 rev/s, where the teeth are
    five bins apart or closer and the neighbourhood the median measures is the
    comb. The gaps of the hypothesis are a floor that is correct for the
    hypothesis by construction.
    """
    x, _ = eight_channel(n=16000)
    net = _net(emission="v2", parts=("gap",))
    emit = _emit(net)
    with torch.no_grad():
        emit.alpha.zero_()
        pw = cast(torch.Tensor, net.spectrum(x))
        xf = emit.local_floor(pw.unsqueeze(1), net.floor_bins)
        rd = emit.reads(pw, xf[:, 0], net.gather)
        gap = cast(OffsetCombGather, emit.gap_gather)
        h_gap = gap(pw)
        ok = (cast(torch.Tensor, gap.valid)[None, :, :, None] > 0).expand_as(rd.floor)
    lhs = rd.floor.log()[ok]
    rhs = h_gap.clamp_min(1e-12).log()[ok]
    assert float((lhs - rhs).abs().max()) < 1e-4
    # And the median floor, which the mixture leaves at alpha = 1, is a
    # different quantity — otherwise the test would pass on a bug.
    assert float((rd.gf_med.log()[ok] - rhs).abs().max()) > 1.0


@pytest.mark.slow
def test_cross_order_memory_at_the_design_shape():
    """One forward and backward of the cross-order emission, at the design shape.

    Batch 4, one microphone, a 2 s crop, the grid from 10 rev/s at 900 points and
    40 orders — the shape section 3.5 costs its activations at. The design's own
    figure is 145 MB per crop per layer in float32, so 580 MB per layer at this
    batch, and the network holds five such tensors plus its 218 MB of inputs.

    Measured on CPU, without the per-channel checkpoint, so this is the whole
    graph: 4.65 GiB peak resident against 0.98 GiB before the step, that is
    3.67 GiB for one forward and backward. The number is the process's peak
    resident set, which also holds the interpreter, torch and the mask bank, so
    it is an UPPER bound on the tensors. A 40 GB accelerator holds it ten times
    over, and the checkpoint (`use_ckpt=True`) trades most of it back for one
    recomputation per channel.
    """
    import resource

    torch.manual_seed(0)
    net = SlotCombNet(
        r_lo=10.0,
        r_hi=100.0,
        n_grid=900,
        k_max=40,
        floor_hz=60.0,
        n_iter=0,
        use_checkpoint=False,
        emission="v2",
        parts=("cross_order",),
        n_mic=1,
    )
    net.train()
    x = torch.randn(4, 1, 32000) * 0.01  # four items of ONE microphone
    pwc, pw = cast(tuple[torch.Tensor, torch.Tensor], net.spectrum(x, per_channel=True))
    floor = _emit(net).local_floor(pw.unsqueeze(1), net.floor_bins)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    s = _emit(net)(pw.unsqueeze(1), floor, net.gather)
    s.sum().backward()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert s.shape == (4, 900, 63)
    print(f"peak {peak / 1024**2:.2f} GiB, {before / 1024**2:.2f} GiB before the step")
    assert peak / 1024**2 < 12.0, f"{peak / 1024**2:.1f} GiB peak, {before / 1024**2:.1f} before"


def test_v2_rejects_an_unknown_part():
    with pytest.raises(ValueError):
        _net(emission="v2", parts=("gap", "not_a_part"))


def test_read_kernel_is_a_delta_at_the_default_width():
    """The learned read starts AT the single interpolated bin.

    `sigma = 0.15` bins puts `exp(-0.5 / 0.15^2) = 2.2e-10` of the kernel on the
    neighbouring bin, so the read matches the plain gather to 1e-6 for any
    neighbour-to-centre power ratio below 4500. The guard against a zero width is
    the clamp in `sigma`, because the kernel exponent at zero is `0 / 0`.
    """
    emit = PartialEmissionV2(8, n_mic=1, parts=("read_width_learned",))
    ker = emit.read_kernel()
    mid = ker.shape[-1] // 2
    assert float(ker[:, 0, mid].min()) == pytest.approx(1.0, abs=1e-9)
    assert float(ker[:, 0, :mid].max()) < 1e-9
    with torch.no_grad():
        emit.s0.fill_(-30.0)  # softplus(-30) is 0 in float32
    assert bool(torch.isfinite(emit.read_kernel()).all())
    assert float(emit.sigma().min()) == pytest.approx(1e-3)
    with torch.no_grad():
        emit.s0.fill_(math.log(math.expm1(2.0)))
    wide = emit.read_kernel()
    assert wide.shape[-1] == 17  # 4 widths each side, at 2 bins
    assert float(wide.sum(dim=-1).min()) == pytest.approx(1.0, abs=1e-6)
