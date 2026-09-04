"""Tests for the three v2 fixes of the HG-CKLA refiner (``models.hg_ckla``).

Probe P4 measured the trained v1 refiner on the frozen real split and found
three defects: from a PERFECT initialization it drifts 0.41 rev/s away at
cruise, iterating it walks out (cruise 2.09 -> 2.12 -> 2.17 rev/s), and its
pull is a fixed ~40 % of any offset inside +-2 rev/s. The three opt-in flags
repair one defect each. These tests pin two things: that the flags OFF leave
v1 untouched, and that the flags ON keep the properties v1 already had (a
fixed point at a clean comb, recovery from a 1.5 rev/s offset) while adding
the ones it lacked (a smoother that reduces frame-to-frame scatter, a gain in
(0, 1) that comes from a variance, and gradients into the new parameters).

``tests/models/test_hg_ckla.py`` holds the v1 contract itself; it must keep
passing unchanged, and it does, because every flag defaults to ``False``.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from models.hg_ckla import HGCKLACell, HGCKLARefiner, measurement_variance

SR = 16000
N_FFT = 2048
HOP = 512
REPO = Path(__file__).resolve().parents[2]

#: The commit that introduced v1 (``HG-CKLA v1: state-conditioned
#: harmonic-gather refiner``). Pinned, not ``HEAD``: once v2 is committed,
#: ``HEAD`` is the v2 file and a parity test against it would compare the new
#: module with itself and pass for the wrong reason.
V1_SHA = "fb7743eeb5e075615889ac1f11e9109147cbb5ff"

V2_FLAGS = dict(state_features=True, kalman_gain=True, smoother=True)

#: Four rotor rates spread wide enough that harmonics 1..40 of one comb do
#: not sit on top of another comb's at every order.
RATES4 = [61.0, 89.0, 127.0, 173.0]


def _comb(rates: list[float], n_harmonics: int = 40, seconds: float = 2.0) -> torch.Tensor:
    """Sum of one pure-tone comb per rotor, shape (1, N).

    Same construction as ``tests/models/test_hg_ckla.py::_comb``, with one
    comb per rate so the twin gate and the four-rotor state are exercised.
    """
    t = torch.arange(int(seconds * SR)) / SR
    x = torch.zeros_like(t)
    for r, f0 in enumerate(rates):
        for k in range(1, n_harmonics + 1):
            x = x + torch.cos(2 * math.pi * k * f0 * t + 0.3 * k + 0.7 * r)
    return x.unsqueeze(0)


def _model(seed: int = 0, **kw) -> HGCKLARefiner:
    torch.manual_seed(seed)
    return HGCKLARefiner(n_fft=N_FFT, hop_length=HOP, sample_rate=SR, **kw)


def _v1_module() -> ModuleType:
    """``src/models/hg_ckla.py`` as it stood at :data:`V1_SHA`, importable."""
    try:
        source = subprocess.check_output(
            ["git", "show", f"{V1_SHA}:src/models/hg_ckla.py"], cwd=REPO, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"v1 source at {V1_SHA[:7]} is not reachable from this checkout: {exc}")
    path = REPO / ".cache" / "hg_ckla_v1_reference.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("hg_ckla_v1_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ─── 1. v1 parity with every flag off ────────────────────────────────────────


def test_flags_off_is_bit_identical_to_v1():
    """With no flag set, the refiner IS v1: same state-dict keys, same
    parameter values from the same seed, and the same output bit for bit.
    The v1 run (``hb_hgckla_ref``) has to stay reproducible."""
    v1 = _v1_module()
    kw: dict[str, Any] = dict(n_fft=N_FFT, hop_length=HOP, sample_rate=SR, k_caps=(10, 25))

    torch.manual_seed(0)
    old = v1.HGCKLARefiner(**kw)
    torch.manual_seed(0)
    new = HGCKLARefiner(**kw)

    old_state, new_state = old.state_dict(), new.state_dict()
    assert list(old_state) == list(new_state)
    for key, value in old_state.items():
        assert torch.equal(value, new_state[key]), f"init differs at {key}"

    old.eval()
    new.eval()
    torch.manual_seed(1)
    audio = torch.randn(2, 16000)
    cond = torch.rand(2, 4, 32) * 60.0 + 50.0
    with torch.no_grad():
        out_old = old(audio, cond)
        out_new = new(audio, cond)
    assert torch.equal(out_old, out_new)


# ─── 2. the fixed-point property at zero noise ───────────────────────────────


def test_true_conditioning_is_a_fixed_point():
    """Given the TRUE rates on a clean comb, the v2 model must stay put.

    This is the property P4/M3 says v1 does not have on real audio (0.41
    rev/s of drift at cruise, on 96.6 % of frames). At zero noise the
    untrained model must not move at all."""
    audio = _comb(RATES4)
    model = _model(**V2_FLAGS)
    model.eval()
    n_frames = model.stft(audio).shape[-1]
    cond = torch.tensor(RATES4).view(1, 4, 1).expand(1, 4, n_frames).contiguous()
    with torch.no_grad():
        out = model(audio, cond)
    drift = float((out - cond).abs().mean())
    assert drift < 0.05, f"untrained v2 drifts {drift:.4f} rev/s from the truth"


# ─── 3. recovery from an offset, as v1 does ──────────────────────────────────


def test_recovers_a_1p5_offset():
    """The v2 stack keeps v1's untrained pull: a 1.5 rev/s conditioning error
    on a clean comb is gone after the three cells.

    ONE rotor, because that is the scene the v1 figure was measured on
    (``conf/experiment/hb_hgckla_ref.md``: "on a clean synthetic comb it pulls
    a 1.5 rev/s conditioning error to 0.04 rev/s before any training"). Four
    rotors of forty harmonics put 160 lines inside 4 kHz, closer together
    than the twin band, so the gate masks three quarters of the reads and
    neither version measures the offset there."""
    audio = _comb([80.0])
    model = _model(num_rotors=1, **V2_FLAGS)
    v1_like = _model(num_rotors=1)  # every flag off == v1 (test 1)
    model.eval()
    v1_like.eval()
    n_frames = model.stft(audio).shape[-1]
    truth = torch.full((1, 1, n_frames), 80.0)
    with torch.no_grad():
        out = model(audio, truth - 1.5)
        ref = v1_like(audio, truth - 1.5)
    # Frame 0 has no predecessor, so no innovation exists there by
    # construction; score the frames that carry a measurement.
    err = float((out - truth)[..., 1:].abs().mean())
    err_v1 = float((ref - truth)[..., 1:].abs().mean())
    assert err < 0.1, f"residual after three cells: {err:.4f} rev/s"
    assert err <= err_v1, f"v2 {err:.4f} rev/s is worse than v1 {err_v1:.4f}"


# ─── 4. the smoother reduces frame-to-frame scatter ──────────────────────────


def test_smoother_reduces_scatter():
    """The RTS pass is the fix for the walk-out P4/M5 measured (2.09 -> 2.12
    -> 2.17 over three passes). Its signature is less scatter frame to
    frame, at identical weights, once the comb carries noise."""
    rates = RATES4
    torch.manual_seed(3)
    audio = _comb(rates) + 30.0 * torch.randn(1, int(2.0 * SR))

    smoothed = _model(state_features=True, kalman_gain=True, smoother=True)
    filtered = _model(state_features=True, kalman_gain=True, smoother=False)
    filtered.load_state_dict(smoothed.state_dict())
    smoothed.eval()
    filtered.eval()
    n_frames = smoothed.stft(audio).shape[-1]
    cond = torch.tensor(rates).view(1, 4, 1).expand(1, 4, n_frames).contiguous()
    with torch.no_grad():
        out_s = smoothed(audio, cond)
        out_f = filtered(audio, cond)

    def scatter(x: torch.Tensor) -> float:
        return float((x[..., 1:] - x[..., :-1]).abs().mean())

    assert scatter(out_s) < scatter(out_f), (
        f"smoothed scatter {scatter(out_s):.5f} >= filtered {scatter(out_f):.5f}"
    )


# ─── 5. the carried variance and its gradients ───────────────────────────────


def test_gain_variance_and_gradients():
    """``K`` is a Kalman gain (strictly inside 0 and 1), ``R_phys`` is a
    positive finite variance, and the loss reaches every new parameter."""
    rates = RATES4
    audio = _comb(rates)
    model = _model(**V2_FLAGS)
    n_frames = model.stft(audio).shape[-1]
    cond = torch.tensor(rates).view(1, 4, 1).expand(1, 4, n_frames).contiguous()

    spec = model.stft(audio)
    cell = model.cells[0]
    assert isinstance(cell, HGCKLACell)
    df, info = cell(spec.real.contiguous(), spec.imag.contiguous(), cond, None)

    gains, r_phys, r_t = info["K"], info["R_phys"], info["R"]
    assert gains.shape == df.shape == r_phys.shape
    assert bool((gains > 0.0).all()) and bool((gains < 1.0).all())
    assert bool((r_phys > 0.0).all()) and bool(torch.isfinite(r_phys).all())
    assert bool((r_t > 0.0).all()) and bool(torch.isfinite(r_t).all())

    out = model(audio, cond - 1.0)
    out.pow(2).mean().backward()
    for i, c in enumerate(model.cells):
        assert isinstance(c, HGCKLACell)
        for name, param in (
            ("q_raw", c.q_raw),
            ("p0_raw", c.p0_raw),
            ("head_r.weight", c.head_r.weight),
        ):
            assert param.grad is not None, f"cell {i}: no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"cell {i}: non-finite grad in {name}"
            assert float(param.grad.abs().sum()) > 0.0, f"cell {i}: zero grad for {name}"


# ─── the pieces the fixes are built from ─────────────────────────────────────


def test_state_features_widen_the_input_and_carry_between_cells():
    """Fix 1 adds exactly four scalars, and cell 0 sees them as zeros while a
    later cell sees the carry it was handed."""
    plain = _model(k_caps=(10,))
    rich = _model(k_caps=(10,), **V2_FLAGS)
    p_cell, r_cell = plain.cells[0], rich.cells[0]
    assert isinstance(p_cell, HGCKLACell) and isinstance(r_cell, HGCKLACell)
    assert p_cell.in_proj.in_features == 3 * 10 + 2
    assert r_cell.in_proj.in_features == 3 * 10 + 2 + 4

    audio = _comb([72.0, 80.0, 88.0, 96.0])
    spec = rich.stft(audio)
    x_re, x_im = spec.real.contiguous(), spec.imag.contiguous()
    cond = torch.full((1, 4, spec.shape[-1]), 80.0)
    first = r_cell.measure(x_re, x_im, cond, None)["feats"]
    # A constant state and no carry: three of the four extra scalars are zero.
    assert float(first[:, :, -3:].abs().max()) == 0.0

    carry = {
        "acc": torch.full_like(cond, 0.2),
        "df_prev": torch.full_like(cond, 0.5),
        "R_prev": torch.full_like(cond, 1e-2),
    }
    second = r_cell.measure(x_re, x_im, cond, carry)["feats"]
    assert float(second[:, :, -3].mean()) == pytest.approx(0.2)
    assert float(second[:, :, -2].mean()) == pytest.approx(0.5 / r_cell.max_step)
    assert float(second[:, :, -1].mean()) == pytest.approx(math.log10(1e-2) / 3.0)


def test_measurement_variance_shrinks_with_agreement():
    """Harmonics that agree give a small variance; a disagreeing one raises
    it; an empty mask returns the sentinel. This is the whole content of the
    ``R_phys`` measurement."""
    arg = torch.zeros(1, 1, 8, 3)
    weights = torch.ones(1, 1, 8, 3)
    df = torch.zeros(1, 1, 3)
    agree = measurement_variance(arg, weights, df, HOP, SR)
    assert float(agree.max()) == pytest.approx(1e-4)

    arg[0, 0, 4, :] = 1.0
    disagree = measurement_variance(arg, weights, df, HOP, SR)
    assert float(disagree.min()) > float(agree.max())

    empty = measurement_variance(arg, torch.zeros_like(weights), df, HOP, SR)
    assert float(empty.min()) == pytest.approx(1e3)


def test_smoother_without_kalman_gain_is_rejected():
    """The RTS pass smooths the carried variance; there is nothing to smooth
    without it."""
    with pytest.raises(ValueError, match="smoother=True needs kalman_gain=True"):
        _model(kalman_gain=False, smoother=True)
