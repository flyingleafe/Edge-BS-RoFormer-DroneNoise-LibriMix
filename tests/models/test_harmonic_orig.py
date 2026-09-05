"""HarmoF0 and HPPNet as published — the controls for the harmonic ports.

What is locked here is the CONTROL property: these two arms are the paper
architectures, on the paper grid, reaching the `salience_rps` task through the
same seam every other salience row uses. So the file checks two different
things and both matter:

* the harmonic organs are BIT-IDENTICAL to the upstream source (`MRDConv`,
  `HarmonicDilatedConv`, `FreqGroupLSTM`, and HPPNet's whole `CNNTrunk` with
  its published ``[1, 4]`` frequency pool), against reference implementations
  quoted verbatim from the two repositories;
* the framework contract holds — the shape, a gradient that moves the loss, and
  a decode that returns the rotor speeds a perfect model would have encoded.

The verbatim references are copied from
``WX-Wei/HarmoF0@3b22236 harmof0/layers.py`` and
``WX-Wei/HPPNet@4dbe905 hppnet/{nets,lstm}.py``. Set ``HARMOF0_SRC`` /
``HPPNET_SRC`` to a checkout of either repository and the last two tests
also compare against the real files; without them those tests skip.
"""

# The two reference blocks below are quoted VERBATIM from the upstream
# repositories, list literals for `kernel_size`/`dilation` included. Rewriting
# them to satisfy the type checker would defeat the point of the comparison.
# pyright: reportArgumentType=false

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_ports.harmof0_orig import HarmoF0Orig, MRDConv, harmonic_dilation_list
from models.harmonic_ports.hppnet_orig import (
    HPPNET_DILATIONS,
    CNNTrunk,
    FreqGroupLSTM,
    HarmonicDilatedConv,
    HPPNetOrig,
)
from models.multif0.utils import salience_target_from_resampled_rps
from models.registry import build_model

ORIGS = [HarmoF0Orig, HPPNetOrig]
SR, N_BINS = 16000, 352


# ─── the framework contract ────────────────────────────────────────────────


@pytest.mark.parametrize("cls", ORIGS)
def test_forward_emits_the_352_bin_map_on_the_frame_grid(cls):
    m = cls().eval()
    audio = torch.randn(2, SR)
    y = m(audio)
    assert y.shape == (2, N_BINS, SR // 512 + 1)
    assert y.shape[-1] == m.num_grid_frames(SR)
    assert m.output_freqs().shape == (N_BINS,)


def test_the_two_arms_share_one_axis():
    """One `conf/loss/salience_bce_orig.yaml` has to serve both arms."""
    a, b = HarmoF0Orig().output_freqs(), HPPNetOrig().output_freqs()
    assert np.allclose(a, b, rtol=0, atol=1e-9)
    # ...and it is the grid the loss config builds from (27.5, 352, 48).
    assert np.allclose(a, 27.5 * 2.0 ** (np.arange(N_BINS) / 48.0))


@pytest.mark.parametrize("cls", ORIGS)
def test_one_optimizer_step_decreases_the_bce(cls):
    torch.manual_seed(0)
    m = cls().train()
    audio = torch.randn(2, SR)
    rps = torch.full((2, 4, SR // 512 + 1), 60.0) + torch.arange(4).view(1, 4, 1) * 10.0
    target = m.salience_target_from_frame_rps(rps, SR, blur_bins=2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    before = F.binary_cross_entropy_with_logits(m(audio), target)
    before.backward()
    opt.step()
    opt.zero_grad()
    after = F.binary_cross_entropy_with_logits(m(audio), target)
    assert float(after) < float(before)


@pytest.mark.parametrize("cls", ORIGS)
def test_predict_rps_recovers_four_steady_rotors_from_a_perfect_map(cls):
    """The readout round trip: a perfect model's own logits, back through `predict_rps`.

    A trained model's optimum under `losses.SalienceRPSBCELoss` is
    ``sigmoid(z) == target``, so ``logit(target)`` is what it emits. Feeding
    that through the SHIPPED decode (sigmoid -> Hungarian tracking -> resample
    to the frame grid) must return the four speeds. "Within one bin" on this
    log grid is 1.45% of the rate, i.e. 0.9-1.3 rev/s at 60-90.
    """
    m = cls().eval()
    audio = torch.randn(1, SR)
    n_grid = m.num_grid_frames(SR)
    speeds = torch.tensor([60.0, 70.0, 80.0, 90.0])
    rps = speeds.view(1, 4, 1).expand(1, 4, n_grid).contiguous()
    target = salience_target_from_resampled_rps(rps, m.output_freqs(), blur_bins=1)
    logits = torch.logit(target.clamp(1e-6, 1 - 1e-6))

    m.forward = lambda _audio, _y=logits: _y  # type: ignore[method-assign]
    out = m.predict_rps(audio)

    assert out.shape == (1, 4, SR // 512 + 1)
    got = np.sort(out[0, :, n_grid // 2].numpy())
    step = np.asarray(m.output_freqs())
    tol = float(np.max(np.diff(step)[(step[:-1] >= 55) & (step[:-1] <= 95)]))
    assert np.allclose(got, speeds.numpy(), atol=tol)


@pytest.mark.parametrize("cls", ORIGS)
def test_predict_rps_runs_on_real_audio(cls):
    m = cls().eval()
    assert m.predict_rps(torch.randn(2, SR)).shape == (2, 4, SR // 512 + 1)


@pytest.mark.parametrize(
    ("key", "cls"), [("harmof0_orig", HarmoF0Orig), ("hppnet_orig", HPPNetOrig)]
)
def test_the_registry_builds_them(key, cls):
    from models.registry import model_types

    assert isinstance(build_model(key), cls)
    assert key in model_types()


def test_the_published_pools_are_reachable():
    """HPPNet's two pools are deviations, not deletions — the flags restore them."""
    m = HPPNetOrig(freq_pool=4, time_pooling=True).eval()
    y = m(torch.randn(1, SR))
    assert y.shape == (1, 88, SR // 512 + 1)  # one bin per semitone, full frame rate
    assert m.output_freqs().shape == (88,)


def test_the_layer_readout_mixin_is_wired():
    """`n_maps > 1` stacks per-rotor layers on the output axis, as in the ports."""
    from models.harmonic_ports.layer_readout import split_maps

    m = HarmoF0Orig(n_maps=4).eval()
    y = m(torch.randn(1, SR))
    assert y.shape == (1, 4 * N_BINS, SR // 512 + 1)
    assert split_maps(y, 4).shape == (1, 4, N_BINS, SR // 512 + 1)


# ─── parity with the published blocks ──────────────────────────────────────
#
# Verbatim from WX-Wei/HarmoF0@3b22236 `harmof0/layers.py` and
# WX-Wei/HPPNet@4dbe905 `hppnet/{nets,lstm}.py`. Do not tidy these — their
# value is that they are the upstream text.


class _UpstreamMRDConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list):
        super().__init__()
        self.dilation_list = dilation_list
        self.conv_list = []
        for _i in range(len(dilation_list)):
            self.conv_list += [nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1])]
        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, specgram):
        dilation = self.dilation_list[0]
        y = self.conv_list[0](specgram)
        y = F.pad(y, pad=[0, dilation])
        y = y[:, :, :, dilation:]
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i]
            x = self.conv_list[i](specgram)
            x = x[:, :, :, dilation:]
            n_freq = x.size()[3]
            y[:, :, :, :n_freq] += x
        return y


class _UpstreamHarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 48])
        self.conv_2 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 76])
        self.conv_3 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 96])
        self.conv_4 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 111])
        self.conv_5 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 124])
        self.conv_6 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 135])
        self.conv_7 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 144])
        self.conv_8 = nn.Conv2d(c_in, c_out, [1, 3], padding="same", dilation=[1, 152])

    def forward(self, x):
        x = (
            self.conv_1(x)
            + self.conv_2(x)
            + self.conv_3(x)
            + self.conv_4(x)
            + self.conv_5(x)
            + self.conv_6(x)
            + self.conv_7(x)
            + self.conv_8(x)
        )
        return torch.relu(x)


def _hdc_state(mine: HarmonicDilatedConv) -> dict:
    """Our ``convs.<i>.*`` -> upstream's ``conv_<i+1>.*``."""
    return {
        f"conv_{i + 1}.{k}": v for i, c in enumerate(mine.convs) for k, v in c.state_dict().items()
    }


@torch.no_grad()
def test_mrdconv_matches_the_published_block():
    mine = MRDConv(4, 6, harmonic_dilation_list(12, 48))
    theirs = _UpstreamMRDConv(4, 6, harmonic_dilation_list(12, 48))
    theirs.load_state_dict(mine.state_dict())
    x = torch.randn(2, 4, 5, N_BINS)
    assert torch.equal(mine(x), theirs(x))


@torch.no_grad()
def test_harmonic_dilated_conv_matches_the_published_block():
    mine = HarmonicDilatedConv(4, 6)
    theirs = _UpstreamHarmonicDilatedConv(4, 6)
    theirs.load_state_dict(_hdc_state(mine))
    x = torch.randn(2, 4, 7, N_BINS)
    assert torch.equal(mine(x), theirs(x))


def test_the_harmonic_offsets_are_the_published_numbers():
    """``round(log2(k) * 48)``: HarmoF0 computes them, HPPNet hard-codes them."""
    assert harmonic_dilation_list(12, 48) == [0, 48, 76, 96, 111, 124, 135, 144, 152, 159, 166, 172]
    assert HPPNET_DILATIONS == (48, 76, 96, 111, 124, 135, 144, 152)
    assert harmonic_dilation_list(9, 48)[1:] == list(HPPNET_DILATIONS)
    branches = HarmonicDilatedConv(1, 1).convs
    assert all(c.dilation == (1, d) for c, d in zip(branches, HPPNET_DILATIONS, strict=True))


# ─── parity against a real checkout, when one is available ─────────────────


def _load_upstream(env_var: str, rel: str, name: str, deps: dict[str, str] | None = None):
    """Import one upstream module from ``$env_var/<rel>``, or skip the test.

    The two repositories' packages import h5py / torchvision / matplotlib at
    ``__init__`` time, so the module is loaded by path under a synthetic package
    and its intra-package dependencies are loaded first, by hand.
    """
    root = os.environ.get(env_var)
    if not root or not (Path(root) / rel).is_file():
        pytest.skip(f"set {env_var} to a checkout of the upstream repository")
    sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    pkg_name = f"_upstream_{name}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str((Path(root) / rel).parent)]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

    def _load(path: Path, mod_name: str):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    for dep_rel, dep_name in (deps or {}).items():
        _load(Path(root) / dep_rel, f"{pkg_name}.{dep_name}")
    return _load(Path(root) / rel, f"{pkg_name}.target")


@torch.no_grad()
def test_mrdconv_matches_the_upstream_source():
    up = _load_upstream("HARMOF0_SRC", "harmof0/layers.py", "harmof0")
    mine = MRDConv(4, 6, harmonic_dilation_list(12, 48))
    theirs = up.MRDConv(4, 6, harmonic_dilation_list(12, 48))
    theirs.load_state_dict(mine.state_dict())
    x = torch.randn(2, 4, 5, N_BINS)
    assert torch.equal(mine(x), theirs(x))


@torch.no_grad()
def test_the_hppnet_trunk_matches_the_upstream_source():
    """The WHOLE published trunk, pool included — not just the harmonic block."""
    up = _load_upstream(
        "HPPNET_SRC",
        "hppnet/nets.py",
        "hppnet",
        deps={"hppnet/constants.py": "constants", "hppnet/lstm.py": "lstm"},
    )
    mine_hdc, theirs_hdc = HarmonicDilatedConv(4, 6), up.HarmonicDilatedConv(4, 6)
    theirs_hdc.load_state_dict(_hdc_state(mine_hdc))
    x = torch.randn(2, 4, 7, N_BINS)
    assert torch.equal(mine_hdc(x), theirs_hdc(x))

    mine_lstm, theirs_lstm = FreqGroupLSTM(8, 1, 16, sigmoid=True), up.FreqGroupLSTM(8, 1, 16)
    theirs_lstm.load_state_dict(mine_lstm.state_dict())
    x = torch.randn(2, 8, 9, 20)
    assert torch.allclose(mine_lstm(x), theirs_lstm(x), atol=1e-6)

    mine_trunk = CNNTrunk(1, 16, 128, freq_pool=4).eval()
    theirs_trunk = up.CNNTrunk(1, 16, 128).eval()
    renamed = {
        k.replace(f"conv_3.convs.{i}.", f"conv_3.conv_{i + 1}.")
        if f"conv_3.convs.{i}." in k
        else k: v
        for k, v in mine_trunk.state_dict().items()
        for i in range(8)
        if f"conv_3.convs.{i}." in k or (i == 0 and "conv_3.convs." not in k)
    }
    theirs_trunk.load_state_dict(renamed, strict=True)
    x = torch.randn(1, 1, 12, N_BINS)
    assert torch.equal(mine_trunk(x), theirs_trunk(x))
