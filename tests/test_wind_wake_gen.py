"""Tests for the wind-wake noise channel (``models.generative.wind_wake_gen``).

All CPU-cheap and dataset-free: they exercise the physics (module B wake gate,
module A grey-box dynamics), the learned transduction (module C), autograd, and
the incoherence/silence invariants the design relies on. The real-audio geometry
de-risk lives separately in ``scripts/wind_wake_validation.py``.

The ``models`` namespace package resolves to the *main* checkout via the editable
install, so prepend this worktree's ``src`` to pick up the new module here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.generative.wind_wake_gen import (  # noqa: E402
    QuadDynamics,
    WindWakeChannel,
    induced_velocity,
    ou_envelope,
    wake_flow_speed,
)


def _rig(r: int = 4, m: int = 8):
    """A small static hover rig: mic cloud below a square rotor plane."""
    torch.manual_seed(0)
    mic = torch.randn(m, 3) * 0.1
    mic[:, 2] -= 0.1  # push mics below the rotor plane (downstream of -z downwash)
    ang = torch.arange(r) * (2 * torch.pi / r)
    rotor = torch.stack([0.2 * torch.cos(ang), 0.2 * torch.sin(ang), torch.full((r,), 0.19)], -1)
    return mic, rotor


# ---------------------------------------------------------------------------
# Shapes / interface
# ---------------------------------------------------------------------------


def test_forward_shapes_static_and_dict():
    mic, rotor = _rig()
    ch = WindWakeChannel(sample_rate=16000, n_env=16)
    rps = torch.rand(1, 4, 2048) * 40 + 50
    w = ch(rps, mic, rotor)
    assert w.shape == (1, 8, 2048)
    d = ch(rps, mic, rotor, return_dict=True)
    assert d["wind"].shape == (1, 8, 2048)
    assert d["flow_speed"].shape == (1, 8, 16)
    assert d["filter_mags"].shape == (1, 8, 16, 65)


def test_batched_geometry_and_n_samples():
    mic, rotor = _rig()
    ch = WindWakeChannel(sample_rate=16000, n_env=16)
    rps = torch.rand(3, 4, 2048) * 40 + 50
    # per-clip (batched) geometry
    w = ch(rps, mic.expand(3, -1, -1), rotor.expand(3, -1, -1), n_samples=1000)
    assert w.shape == (3, 8, 1000)


def test_wake_flow_speed_per_rotor_shape():
    mic, rotor = _rig()
    rps = torch.full((1, 4, 16), 80.0)
    axis = torch.tensor([0.0, 0.0, -1.0]).reshape(1, 1, 3).expand(1, 4, 3)
    u, u_r = wake_flow_speed(
        mic.unsqueeze(0),
        rotor.unsqueeze(0),
        axis,
        0.127,
        rps,
        k=1.0,
        alpha=0.5,
        beta=0.3,
        gate_softness=0.1,
        return_per_rotor=True,
    )
    assert u.shape == (1, 8, 16)
    assert u_r.shape == (1, 8, 4, 16)


# ---------------------------------------------------------------------------
# Physics invariants — module B
# ---------------------------------------------------------------------------


def test_induced_velocity_monotone():
    r = torch.tensor(0.127)
    k = torch.tensor(1.0)
    rps = torch.tensor([10.0, 50.0, 90.0])
    v = induced_velocity(rps, r, k)
    assert torch.all(v[1:] > v[:-1])  # increases with rps
    assert induced_velocity(torch.tensor(50.0), torch.tensor(0.2), k) > v[1]  # and with R


def test_gate_geometry_in_column_vs_off():
    """A mic downstream on the axis gets flow; upstream / far-lateral do not."""
    mic = torch.tensor(
        [
            [0.0, 0.0, -0.3],  # 0: directly downstream (below rotor)
            [0.0, 0.0, 0.3],  # 1: upstream (above rotor)
            [0.5, 0.0, -0.3],
        ],  # 2: downstream but far off-axis
    ).unsqueeze(0)
    rotor = torch.zeros(1, 1, 3)
    axis = torch.tensor([0.0, 0.0, -1.0]).reshape(1, 1, 3)
    rps = torch.full((1, 1, 8), 80.0)
    u = wake_flow_speed(
        mic, rotor, axis, 0.127, rps, k=1.0, alpha=0.5, beta=0.3, gate_softness=0.1
    ).mean(-1)[0]  # [3]
    assert u[0] > 5 * u[1]  # downstream >> upstream
    assert u[0] > 50 * u[2]  # downstream >> far off-axis
    assert u[0] > 1.0  # meaningful flow speed (m/s)


def test_convection_skew_bends_wake():
    """Lateral relative wind bends the column toward a laterally-offset mic."""
    mic = torch.tensor([[0.15, 0.0, -0.3]]).unsqueeze(0)  # offset in +x, downstream
    rotor = torch.zeros(1, 1, 3)
    axis = torch.tensor([0.0, 0.0, -1.0]).reshape(1, 1, 3)
    rps = torch.full((1, 1, 8), 80.0)
    u_still = wake_flow_speed(
        mic, rotor, axis, 0.127, rps, k=1.0, alpha=0.5, beta=0.3, gate_softness=0.1
    ).mean()
    v_rel = torch.tensor([15.0, 0.0, 0.0]).reshape(1, 3)  # wind pushing wake +x
    u_wind = wake_flow_speed(
        mic, rotor, axis, 0.127, rps, k=1.0, alpha=0.5, beta=0.3, gate_softness=0.1, v_rel=v_rel
    ).mean()
    assert u_wind > 2 * u_still  # bent column now grazes the offset mic


def test_incoherent_superposition_never_negative():
    mic, rotor = _rig()
    rps = torch.rand(2, 4, 16) * 40 + 50
    axis = torch.tensor([0.0, 0.0, -1.0]).reshape(1, 1, 3).expand(2, 4, 3)
    u = wake_flow_speed(
        mic.expand(2, -1, -1),
        rotor.expand(2, -1, -1),
        axis,
        0.127,
        rps,
        k=1.0,
        alpha=0.5,
        beta=0.3,
        gate_softness=0.1,
    )
    assert torch.all(u >= 0)


# ---------------------------------------------------------------------------
# Transduction / synthesis — module C
# ---------------------------------------------------------------------------


def test_rps_zero_gives_silence():
    mic, rotor = _rig()
    ch = WindWakeChannel(sample_rate=16000, n_env=16)
    w = ch(torch.zeros(1, 4, 2048), mic, rotor, apply_gust=False)
    assert w.abs().max() < 1e-6


def test_wind_channel_incoherent_across_mics():
    """Two mics with similar flow exposure still get uncorrelated noise."""
    mic, rotor = _rig(m=8)
    ch = WindWakeChannel(sample_rate=16000, n_env=16)
    rps = torch.full((1, 4, 4096), 80.0)
    w = ch(rps, mic, rotor, apply_gust=False)[0]  # [8, T]
    # off-diagonal correlations should be small in magnitude
    c = torch.corrcoef(w)
    off = c[~torch.eye(8, dtype=torch.bool)]
    assert off.abs().mean() < 0.15


def test_level_increases_with_flow_speed():
    tr = WindWakeChannel(sample_rate=16000, n_env=8).transduction
    lo = tr.filter_mags(torch.full((1, 1, 8), 1.0))
    hi = tr.filter_mags(torch.full((1, 1, 8), 8.0))
    assert hi.sum() > lo.sum()  # more flow -> more wind energy
    # and the corner shifts up: hi has relatively more high-frequency content
    nyq_half = hi.shape[-1] // 2
    assert hi[..., nyq_half:].sum() / hi.sum() > lo[..., nyq_half:].sum() / lo.sum()


def test_ou_envelope_positive_and_meanish_one():
    torch.manual_seed(0)
    g = ou_envelope((4, 2000), torch.tensor(0.1), torch.tensor(0.3), dt=0.01)
    assert torch.all(g > 0)
    assert 0.7 < g.mean() < 1.3  # de-biased ~ 1


def test_reproducible_with_generator():
    mic, rotor = _rig()
    ch = WindWakeChannel(sample_rate=16000, n_env=16)
    rps = torch.rand(1, 4, 2048) * 40 + 50
    ch.eval()
    a = ch(rps, mic, rotor, generator=torch.Generator().manual_seed(7))
    b = ch(rps, mic, rotor, generator=torch.Generator().manual_seed(7))
    c = ch(rps, mic, rotor, generator=torch.Generator().manual_seed(8))
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


# ---------------------------------------------------------------------------
# Grey-box dynamics — module A
# ---------------------------------------------------------------------------


def test_quad_dynamics_hover_zero_airspeed():
    dyn = QuadDynamics(n_rotors=4, hover_rps=90.0)
    ang = torch.arange(4) * (torch.pi / 2)
    rotor = torch.stack([0.2 * torch.cos(ang), 0.2 * torch.sin(ang), torch.zeros(4)], -1)
    rps = torch.full((1, 4, 32), 90.0)  # exactly hover, balanced
    v = dyn(rps, rotor, dt=0.03)
    assert v.abs().max() < 1e-3  # balanced hover -> no drift


def test_quad_dynamics_imbalance_produces_motion():
    dyn = QuadDynamics(n_rotors=4, hover_rps=90.0)
    ang = torch.arange(4) * (torch.pi / 2)
    rotor = torch.stack([0.2 * torch.cos(ang), 0.2 * torch.sin(ang), torch.zeros(4)], -1)
    balanced = dyn(torch.full((1, 4, 32), 90.0), rotor, dt=0.03).norm(dim=-1).max()
    rps = torch.full((1, 4, 32), 90.0)
    rps[:, 0] += 8.0  # one rotor faster -> torque imbalance -> tilt -> translate
    imbalanced = dyn(rps, rotor, dt=0.03).norm(dim=-1).max()
    assert imbalanced > balanced + 0.05


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_dynamics", [False, True])
def test_gradients_flow(use_dynamics):
    mic, rotor = _rig()
    mic = mic.clone().requires_grad_(True)
    ch = WindWakeChannel(sample_rate=16000, n_env=16, use_dynamics=use_dynamics)
    rps = torch.rand(1, 4, 2048) * 40 + 50
    loss = ch(rps, mic, rotor).pow(2).mean()
    loss.backward()
    # physics + transduction parameters
    assert ch.raw_k.grad is not None and torch.isfinite(ch.raw_k.grad).all()
    assert ch.raw_alpha.grad is not None and ch.raw_alpha.grad != 0
    assert ch.transduction.raw_level.grad is not None and ch.transduction.raw_level.grad != 0
    # geometry (positions are optimisable — used later for calibration)
    assert mic.grad is not None and torch.isfinite(mic.grad).all() and mic.grad.abs().sum() > 0
    if use_dynamics:
        assert ch.dynamics is not None
        assert ch.dynamics.log_ct_over_m.grad is not None
