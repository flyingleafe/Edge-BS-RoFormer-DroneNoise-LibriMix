"""
Wind-wake noise channel for the rotor-noise generator.

The coherent generator (:class:`PositionalHarmonicNoiseGen`) maps rotor-speed
trajectories to a *propagating* acoustic field: a single emitter per rotor, then
``1/r`` spreading and a delay to every microphone. Real ego-noise carries a
second thing that no shared-emitter + propagation model can produce — **flow
noise** (pseudo-sound): the turbulent downwash of a rotor striking a diaphragm
directly. It does not propagate (no ``1/r``, no inter-mic delay) and it is
**spatially incoherent** (``γ²`` collapses at low frequency), so it must live in
its own additive channel outside the coherent path.

Design (see the branch design note ``wind_model_design.html``)
-------------------------------------------------------------
Physics decides *where* the air flows and *how fast*; a small learned head
decides only *what that flow does to a microphone*. Three modules in series,
then an additive mix at each mic:

- **A — RPS → airspeed** (:class:`QuadDynamics`, physics). The four rotor speeds
  are a quadrotor's control inputs, so a calibrated grey-box rigid-body model
  turns them into the body-frame relative wind ``V_rel(t)`` that bends the wake.
  Hover / a static rig ⇒ ``V_rel = 0`` and this module is skipped.
- **B — wake flow field** (:func:`wake_flow_speed`, physics, no learned params
  except three interpretable aero constants). Each rotor emits a downwash column
  along its thrust axis, bent downstream by ``V_rel``; a microphone's local flow
  speed is the induced velocity times a smooth in-column gate. Rotors superpose
  incoherently. Everything is closed-form and differentiable in the calibrated
  positions.
- **C — flow → microphone transduction** (:class:`WindTransduction`, learned).
  The only part fit from audio: how a flow of speed ``U`` becomes low-frequency
  pressure at a diaphragm — dynamic pressure ``q = ½ρU²`` sets the level, ``U/ℓ``
  sets a corner frequency, and a learned low-pass shapes the band. A slow
  Ornstein–Uhlenbeck envelope on ``U`` models wake meander (gust intermittency).
  The noise is realized **independently per microphone**, so the channel is
  incoherent by construction.

Because exposure is a geometric function, an array *in* the wake (DREGON) gets
gusts and an array *above/forward* of the disk (Michael's) gets near-silence,
with no per-array tuning — this is the design's central generalization claim and
the thing the CPU de-risk in ``scripts/wind_wake_validation.py`` checks.

Contract
--------
:class:`WindWakeChannel` mirrors :class:`PositionalHarmonicNoiseGen`'s inputs:
RPS ``[B, R, T]`` at audio rate plus geometry, output per-mic audio ``[B, M, T]``.
Nothing beyond RPS + geometry is needed at inference; the auxiliary telemetry
(IMU / DJI logs) only calibrates the physics during training.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dsp import frequency_filter
from .positional_harmonic_gen import PositionalHarmonicNoiseGen

Scalar = torch.Tensor | float

SPEED_OF_SOUND = 343.0
RHO_AIR = 1.204  # kg/m^3, sea-level air density (for dynamic pressure q = ½ρU²)
GRAVITY = 9.81  # m/s^2


def _pos(raw: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained parameter to a strictly-positive value (softplus)."""
    return F.softplus(raw) + 1e-6


def _as_batched(x: torch.Tensor, batch: int, name: str, last: int = 3) -> torch.Tensor:
    """Promote a per-clip geometry tensor ``[N, last]`` to ``[B, N, last]``.

    Accepts an already-batched ``[B, N, last]`` unchanged. Keeps the caller from
    having to broadcast static geometry (the common case: one rig for the whole
    batch).
    """
    if x.dim() == 2:
        if x.shape[-1] != last:
            raise ValueError(f"{name} last dim must be {last}, got {tuple(x.shape)}")
        return x.unsqueeze(0).expand(batch, -1, -1)
    if x.dim() == 3:
        if x.shape[0] not in (1, batch) or x.shape[-1] != last:
            raise ValueError(
                f"{name} must be [{batch}, N, {last}] (or [N, {last}]), got {tuple(x.shape)}"
            )
        return x.expand(batch, -1, -1) if x.shape[0] == 1 else x
    raise ValueError(f"{name} must be 2-D or 3-D, got {tuple(x.shape)}")


# ---------------------------------------------------------------------------
# Module B — wake flow field (physics)
# ---------------------------------------------------------------------------


def induced_velocity(rps: torch.Tensor, rotor_radius: torch.Tensor, k: Scalar) -> torch.Tensor:
    """Rotor induced (downwash) velocity, ``v_i = k · rps · R``.

    A tip-speed-proportional surrogate for momentum-theory hover induced velocity
    (``v_i ∝ √(T/ρA)`` with ``T ∝ (ΩR)²`` ⇒ ``v_i ∝ ΩR ∝ rps·R``). The absolute
    scale is folded into the learned dimensionless ``k`` (and later absorbed by
    the transduction gain), so only the *relative* dependence on rps and radius
    is physical here.

    Args:
        rps: ``[...]`` rotor speed in rev/s.
        rotor_radius: scalar or broadcastable rotor radius in metres.
        k: scalar induced-velocity coefficient.

    Returns:
        ``[...]`` induced velocity in m/s (same shape as ``rps``).
    """
    return k * rps * rotor_radius


@overload
def wake_flow_speed(
    mic_pos: torch.Tensor,
    rotor_pos: torch.Tensor,
    rotor_axis: torch.Tensor,
    rotor_radius: Scalar,
    rps: torch.Tensor,
    *,
    k: Scalar,
    alpha: Scalar,
    beta: Scalar,
    gate_softness: Scalar,
    v_rel: torch.Tensor | None = ...,
    c_free: Scalar | None = ...,
    return_per_rotor: Literal[False] = ...,
) -> torch.Tensor: ...
@overload
def wake_flow_speed(
    mic_pos: torch.Tensor,
    rotor_pos: torch.Tensor,
    rotor_axis: torch.Tensor,
    rotor_radius: Scalar,
    rps: torch.Tensor,
    *,
    k: Scalar,
    alpha: Scalar,
    beta: Scalar,
    gate_softness: Scalar,
    v_rel: torch.Tensor | None = ...,
    c_free: Scalar | None = ...,
    return_per_rotor: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...
def wake_flow_speed(
    mic_pos: torch.Tensor,
    rotor_pos: torch.Tensor,
    rotor_axis: torch.Tensor,
    rotor_radius: Scalar,
    rps: torch.Tensor,
    *,
    k: Scalar,
    alpha: Scalar,
    beta: Scalar,
    gate_softness: Scalar,
    v_rel: torch.Tensor | None = None,
    c_free: Scalar | None = None,
    return_per_rotor: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-microphone local flow speed ``U_m(t)`` from the bent-wake-column model.

    For rotor ``r`` at ``p_r`` with unit downwash axis ``a_r`` and induced
    velocity ``v_i``, the wake convects along ``ĉ_r = normalize(2 v_i a_r +
    V_rel)`` — the classic ``2 v_i`` far-wake speed plus the freestream, which
    tilts the column by the skew angle ``χ = atan(|V_⊥| / 2 v_i)``. For a
    microphone at ``q_m`` with displacement ``d = q_m − p_r``:

    - along-axis distance ``s = d · ĉ_r`` (downstream ⇒ ``s > 0``),
    - perpendicular distance ``ρ = ‖d − s ĉ_r‖``,
    - in-column gate ``g = σ(s / gate_softness) · exp(−ρ² / 2(αR)²) · (1 + β s₊)⁻¹``.

    The mic flow speed is ``U_{m,r} = v_i · g``; rotors superpose **incoherently**
    ``U_m = √(Σ_r U_{m,r}² + U_free²)``, where the optional direct-freestream term
    ``U_free = c_free · |V_rel|`` accounts for wind striking an exposed mic even
    outside any column. In hover (``V_rel = 0``) the freestream term vanishes and
    ``ĉ_r = a_r``, recovering a straight downwash column — the regime the DREGON
    single-motor de-risk validates.

    All arguments broadcast over a leading batch ``B`` and a trailing (frame /
    time) axis ``L``; positions are static per clip.

    Args:
        mic_pos: ``[B, M, 3]`` microphone positions (m).
        rotor_pos: ``[B, R, 3]`` rotor-hub positions (m).
        rotor_axis: ``[B, R, 3]`` unit downwash direction per rotor (points the
            way the air is pushed; for an upright rig ``[0, 0, −1]``).
        rotor_radius: ``[B, R]`` or scalar rotor radius (m).
        rps: ``[B, R, L]`` rotor speed in rev/s at the evaluation (frame) rate.
        k, alpha, beta, gate_softness: aero constants (positive scalars).
        v_rel: ``[B, L, 3]`` or ``[B, 3]`` world-frame relative wind (m/s), or
            ``None`` for hover / static (⇒ 0).
        c_free: optional scalar freestream-coupling coefficient (needs ``v_rel``).
        return_per_rotor: also return the per-rotor ``U_{m,r}`` tensor.

    Returns:
        ``U_m`` ``[B, M, L]``; or ``(U_m, U_per_rotor[B, M, R, L])`` if
        ``return_per_rotor``.
    """
    # Rotor -> mic displacement d[b, r, m] = mic_m − rotor_r  → [B, R, M, 3].
    disp = mic_pos.unsqueeze(1) - rotor_pos.unsqueeze(2)
    return _flow_speed_from_disp(
        disp,
        rotor_axis,
        rotor_radius,
        rps,
        k=k,
        alpha=alpha,
        beta=beta,
        gate_softness=gate_softness,
        v_rel=v_rel,
        c_free=c_free,
        return_per_rotor=return_per_rotor,
    )


def _flow_speed_from_disp(
    disp: torch.Tensor,
    rotor_axis: torch.Tensor,
    rotor_radius: Scalar,
    rps: torch.Tensor,
    *,
    k: Scalar,
    alpha: Scalar,
    beta: Scalar,
    gate_softness: Scalar,
    v_rel: torch.Tensor | None = None,
    c_free: Scalar | None = None,
    return_per_rotor: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Core wake-gate evaluation from a precomputed rotor→mic displacement.

    Shared by :func:`wake_flow_speed` (which builds ``disp`` from absolute
    positions) and :meth:`WindWakeChannel.flow_speed_rel` (which gets ``disp``
    from the generator's ``rel_pos``). ``disp`` is ``[B, R, M, 3]`` (rotor r →
    mic m); everything else matches :func:`wake_flow_speed`.
    """
    b, r, _ = rps.shape
    dtype, device = rps.dtype, rps.device

    if rotor_axis.dim() == 2:
        rotor_axis = rotor_axis.unsqueeze(0).expand(b, -1, -1)
    axis = F.normalize(rotor_axis, dim=-1)  # [B, R, 3]

    if not torch.is_tensor(rotor_radius):
        rotor_radius = torch.as_tensor(rotor_radius, dtype=dtype, device=device)
    radius = (
        rotor_radius.expand(b, r) if rotor_radius.dim() else rotor_radius.reshape(1, 1).expand(b, r)
    )

    v_i = induced_velocity(rps, radius.unsqueeze(-1), k)  # [B, R, L]

    # Convection direction ĉ_r(t). With v_rel it bends; add a [B,R,L,3] axis.
    conv = 2.0 * v_i.unsqueeze(-1) * axis.unsqueeze(2)  # [B, R, L, 3]
    if v_rel is not None:
        vr = v_rel.unsqueeze(1) if v_rel.dim() == 2 else v_rel  # [B,1|L,3] -> broadcast on R,L
        vr = vr.reshape(b, 1, -1, 3)  # [B,1,1|L,3]
        conv = conv + vr
    chat = F.normalize(conv, dim=-1)  # [B, R, L, 3]

    d = disp.unsqueeze(3)  # [B, R, M, 1, 3] to broadcast against chat's L
    chat_e = chat.unsqueeze(2)  # [B, R, 1, L, 3]
    s = (d * chat_e).sum(-1)  # along-axis distance [B, R, M, L]
    perp_vec = d - s.unsqueeze(-1) * chat_e  # [B, R, M, L, 3]
    perp = torch.linalg.vector_norm(perp_vec, dim=-1)  # [B, R, M, L]

    width = (alpha * radius).reshape(b, r, 1, 1).clamp_min(1e-4)  # [B, R, 1, 1]
    downstream = torch.sigmoid(s / gate_softness)  # smooth 𝟙[s>0]
    radial = torch.exp(-(perp**2) / (2.0 * width**2))
    decay = 1.0 / (1.0 + beta * F.relu(s))
    gate = downstream * radial * decay  # [B, R, M, L]

    u_per_rotor = v_i.unsqueeze(2) * gate  # [B, R, M, L]
    u_sq = (u_per_rotor**2).sum(1)  # incoherent sum over rotors -> [B, M, L]

    if c_free is not None and v_rel is not None:
        vmag = torch.linalg.vector_norm(
            v_rel if v_rel.dim() == 3 else v_rel.unsqueeze(1), dim=-1
        )  # [B, L] or [B,1]
        u_free = (c_free * vmag).reshape(b, 1, -1)  # [B,1,L|1]
        u_sq = u_sq + u_free**2

    u = torch.sqrt(u_sq.clamp_min(0.0) + 1e-12)  # [B, M, L]
    if return_per_rotor:
        return u, u_per_rotor.transpose(1, 2)  # [B, M, L], [B, M, R, L]
    return u


# ---------------------------------------------------------------------------
# Module A — RPS → airspeed (grey-box quadrotor dynamics)
# ---------------------------------------------------------------------------


class QuadDynamics(nn.Module):
    """Grey-box quadrotor forward dynamics: per-rotor RPS → world-frame velocity.

    A *reduced* rigid-body model — enough to turn the four control inputs into the
    relative wind that bends the wake, not a full flight simulator. Thrust and
    reaction torque are quadratic in rps (``T_i = c_T rps_i²``); the net vertical
    thrust drives vertical acceleration about a hover anchor, and the thrust
    *imbalance* across the (calibrated) rotor positions produces roll/pitch
    torques that tilt the body, redirecting thrust into horizontal acceleration.
    Double-integrating gives world velocity ``v(t)``; with still indoor air the
    body-frame relative wind is ``V_rel = −Rᵀ v`` (returned in the world frame for
    the wake model, small-angle so ``R ≈ I`` here).

    The physically-meaningful constants (``c_T/m``, inertia, drag, the hover rps)
    are parameters to be **calibrated against IMU / DJI logs at training time**;
    this module only defines the differentiable forward map and its hover anchor.
    It is *not* exercised by the static de-risk (``V_rel = 0`` there); it exists so
    the same channel is motion-aware for free-flight clips.

    Args:
        n_rotors: number of rotors (4).
        hover_rps: rps at which total thrust balances gravity (anchors ``c_T/m``).
        spin_dirs: ``+1/−1`` per rotor for reaction-torque yaw (unused for wake).
    """

    def __init__(
        self,
        n_rotors: int = 4,
        hover_rps: float = 90.0,
        spin_dirs: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.n_rotors = n_rotors
        self.hover_rps = float(hover_rps)
        # c_T/m fixed by the hover anchor: n_rotors · (c_T/m) · hover_rps² = g.
        ct_over_m = GRAVITY / (n_rotors * hover_rps**2)
        self.log_ct_over_m = nn.Parameter(torch.log(torch.tensor(ct_over_m)))
        # Inertia-normalised torque gain and linear drag — free, calibrated later.
        self.log_torque_gain = nn.Parameter(torch.log(torch.tensor(ct_over_m * 4.0)))
        self.log_drag = nn.Parameter(torch.log(torch.tensor(0.5)))
        if spin_dirs is None:
            spin_dirs = tuple((-1) ** i for i in range(n_rotors))
        self.register_buffer("spin_dirs", torch.tensor(spin_dirs, dtype=torch.float32))

    def forward(
        self,
        rps: torch.Tensor,
        rotor_pos: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Integrate world-frame velocity from a per-rotor rps trajectory.

        Args:
            rps: ``[B, R, L]`` rotor speed (rev/s) at frame rate.
            rotor_pos: ``[B, R, 3]`` or ``[R, 3]`` rotor-hub positions (m); the
                in-plane offset from the centroid is the moment arm.
            dt: seconds between frames (``1 / frame_rate``).

        Returns:
            ``V_rel`` ``[B, L, 3]`` world-frame relative wind (still-air ⇒ minus
            the ground velocity).
        """
        b, _, ll = rps.shape
        rotor_pos = _as_batched(rotor_pos, b, "rotor_pos")
        ct_over_m = torch.exp(self.log_ct_over_m)
        torque_gain = torch.exp(self.log_torque_gain)
        drag = torch.exp(self.log_drag)

        thrust = ct_over_m * rps**2  # [B, R, L] (per-mass)
        total_thrust = thrust.sum(1)  # [B, L]

        # Moment arms about the rotor centroid (in-plane x, y).
        arms = rotor_pos - rotor_pos.mean(1, keepdim=True)  # [B, R, 3]
        # Roll/pitch angular accel from thrust imbalance: τ = Σ arm × (thrust ẑ).
        # (thrust ẑ) × arm gives torque about x,y; use arm_x,y directly.
        tau_x = torque_gain * (arms[..., 1].unsqueeze(-1) * thrust).sum(1)  # [B, L]
        tau_y = -torque_gain * (arms[..., 0].unsqueeze(-1) * thrust).sum(1)  # [B, L]

        # Integrate tilt (small-angle): angular accel -> angular vel -> angle.
        roll = torch.cumsum(torch.cumsum(tau_x, dim=1) * dt, dim=1) * dt  # [B, L]
        pitch = torch.cumsum(torch.cumsum(tau_y, dim=1) * dt, dim=1) * dt  # [B, L]

        # Small-angle horizontal accel = g·tilt; vertical accel = thrust − g.
        vel = rps.new_zeros(b, 3)
        vs = []
        for t in range(ll):
            a = torch.stack(
                [
                    GRAVITY * torch.sin(pitch[:, t]),
                    -GRAVITY * torch.sin(roll[:, t]),
                    total_thrust[:, t] - GRAVITY,
                ],
                dim=-1,
            )  # [B, 3]
            vel = vel + (a - drag * vel) * dt
            vs.append(vel)
        v = torch.stack(vs, dim=1)  # [B, L, 3] — stack (not in-place) keeps autograd
        return -v  # still air: airspeed = -ground velocity


# ---------------------------------------------------------------------------
# Module C — flow -> microphone transduction (learned)
# ---------------------------------------------------------------------------


def ou_envelope(
    shape: tuple[int, ...],
    tau: torch.Tensor,
    sigma: torch.Tensor,
    dt: float,
    *,
    generator: torch.Generator | None = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Sample a slow, positive Ornstein–Uhlenbeck meander envelope (mean ≈ 1).

    Log-domain OU: ``x_{t+1} = x_t e^{−dt/τ} + √(1−e^{−2dt/τ}) σ ε``. Exponentiated
    and de-biased so ``E[g] ≈ 1``, giving a multiplicative gust envelope on the
    flow speed. Time is the **last** axis.

    Args:
        shape: output shape ``[..., L]``.
        tau: OU correlation time (s, positive).
        sigma: log-amplitude of the meander (positive).
        dt: seconds per frame.
        generator: optional RNG for reproducibility.

    Returns:
        Envelope tensor of ``shape``, strictly positive.
    """
    ll = shape[-1]
    a = torch.exp(-dt / tau)
    noise_scale = torch.sqrt((1.0 - a**2).clamp_min(1e-8)) * sigma
    eps = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    # Start FROM the stationary distribution N(0, sigma^2) rather than from 0.
    # Starting at 0 biases the opening frames toward g = exp(-sigma^2/2) < 1 and,
    # worse, makes the sampler's marginal disagree with the stationary marginal
    # that `WindTransduction.expected_mags` integrates over — so the likelihood
    # would be fitting a slightly different process than the synthesizer draws.
    xt = eps[..., 0] * sigma
    xs = [xt]
    for t in range(1, ll):
        xt = a * xt + noise_scale * eps[..., t]
        xs.append(xt)
    x = torch.stack(xs, dim=-1)  # [..., L] — stack (not in-place) keeps autograd
    return torch.exp(x - 0.5 * sigma**2)


class WindTransduction(nn.Module):
    """Learned map from per-mic flow speed ``U_m(t)`` to incoherent wind audio.

    A microphone/windscreen property (hence **shared across drones**): dynamic
    pressure ``q = ½ρU²`` sets the band level ``A·(q+ε)^γ``; the corner frequency
    ``f_c = f_floor + c_fc·U`` scales with flow speed; and a learned low-pass
    ``H(f) = (1 + (f/f_c)²)^(−p/2)`` shaped by a small residual MLP over
    ``log(f/f_c)`` sets the spectral roll-off. The band magnitude is realized as
    **independent** filtered white noise per microphone, so the channel is
    spatially incoherent by construction. A slow OU envelope on ``U`` adds gust
    intermittency.

    Args:
        sample_rate: audio rate (Hz).
        n_freqs: filter magnitude grid size (0..Nyquist).
        n_env: envelope / filter-frame rate (frames per clip).
        rho: air density for ``q`` (kg/m³).
        mlp_hidden: hidden width of the shape-residual MLP (0 disables it).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_freqs: int = 65,
        n_env: int = 64,
        rho: float = RHO_AIR,
        mlp_hidden: int = 16,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_freqs = n_freqs
        self.n_env = n_env
        self.rho = rho

        # Positive-constrained physical parameters (softplus of these raws).
        self.raw_level = nn.Parameter(torch.tensor(0.0))  # A ~ 0.69
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))  # γ ~ 0.69 (near 0.5–1)
        self.raw_order = nn.Parameter(torch.tensor(0.54))  # p ~ 1.0 (softplus(.54)≈1)
        self.raw_fc = nn.Parameter(torch.tensor(2.0))  # c_fc Hz per (m/s)
        self.raw_ffloor = nn.Parameter(torch.tensor(2.0))  # f_floor Hz
        self.raw_tau = nn.Parameter(torch.tensor(0.0))  # OU τ (s)
        self.raw_sigma = nn.Parameter(torch.tensor(-1.0))  # OU σ (small)

        freqs = torch.linspace(0.0, sample_rate / 2, n_freqs)
        self.register_buffer("freqs", freqs)

        self.shape_mlp: nn.Sequential | None = None
        if mlp_hidden > 0:
            out = nn.Linear(mlp_hidden, 1)
            # Start as a no-op so the parametric low-pass is the prior at init.
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            self.shape_mlp = nn.Sequential(nn.Linear(1, mlp_hidden), nn.GELU(), out)
        self.log_mlp_gate = nn.Parameter(torch.tensor(-2.0))  # small residual

    def filter_mags(self, u: torch.Tensor) -> torch.Tensor:
        """Per-mic, per-frame magnitude response from flow speed ``U``.

        Args:
            u: ``[B, M, n_env]`` flow speed (m/s).

        Returns:
            ``[B, M, n_env, n_freqs]`` magnitude response on a 0..Nyquist grid.
        """
        level_c = _pos(self.raw_level)
        gamma = _pos(self.raw_gamma)
        order = _pos(self.raw_order)
        c_fc = _pos(self.raw_fc)
        f_floor = _pos(self.raw_ffloor)

        q = 0.5 * self.rho * u**2  # [B, M, n_env]
        eps = 1e-8
        level = level_c * ((q + eps).pow(gamma) - eps**gamma)  # 0 at U=0, smooth
        f_c = (f_floor + c_fc * u).clamp_min(1e-3)  # [B, M, n_env]

        # Normalised frequency u_f = f / f_c, shape [B, M, n_env, n_freqs].
        freqs = torch.linspace(
            0.0, self.sample_rate / 2, self.n_freqs, device=u.device, dtype=u.dtype
        )
        f = freqs.reshape(1, 1, 1, -1)
        u_f = f / f_c.unsqueeze(-1)
        low_pass = (1.0 + u_f**2).pow(-order / 2.0)

        mlp = self.shape_mlp
        if mlp is not None:
            log_uf = torch.log(u_f.clamp_min(1e-4)).unsqueeze(-1)  # [...,1]
            residual = mlp(log_uf).squeeze(-1)  # [B,M,n_env,n_freqs]
            low_pass = low_pass * torch.exp(torch.sigmoid(self.log_mlp_gate) * residual)

        return level.unsqueeze(-1) * low_pass  # [B, M, n_env, n_freqs]

    def expected_power(
        self,
        u: torch.Tensor,
        *,
        apply_gust: bool = True,
        n_quad: int = 9,
    ) -> torch.Tensor:
        """RMS magnitude response, with the gust **marginalized out**.

        The gust is the reason the sampling path cannot be used as a training
        target: it is an unobservable latent, so no realization of it is the
        "right" one. What the likelihood needs is the response averaged over the
        gust distribution, ``sqrt(E_g[|H(U g)|^2])``.

        ``g = exp(x - s^2/2)`` with ``x ~ N(0, s^2)``, and ``H`` depends on ``g``
        nonlinearly (through both the band level ``q^gamma`` and the corner
        ``f_c``), so the expectation has no closed form. It is computed by
        Gauss--Hermite quadrature, which for a smooth integrand against a
        Gaussian is accurate to machine precision at a handful of nodes — far
        cheaper and lower-variance than Monte Carlo over gust draws.

        This is a *moment* match: the marginal is a scale mixture of Rayleighs
        rather than a Rayleigh, so matching ``E[power]`` is exact in the second
        moment and approximate in the tails. That is the right trade here, since
        the Whittle term the variance feeds is a second-moment statistic.

        **Validity range.** The band level goes as ``U^(2*gamma)``, so the gust
        enters the *power* as ``g^(4*gamma)`` and its expectation is the
        log-normal moment ``exp(p(p-1) s^2 / 2)`` with ``p = 4*gamma``. That
        grows quickly: at the default ``gamma ~ 0.69`` it is 1.3x at ``s = 0.31``
        (the initialization) but 14x at ``s = 1.0``, where the predicted power is
        carried by rare gusts. In that regime the moment match is a poor summary
        of a tail-dominated mixture *and* the quadrature needs many more nodes
        (measured: 9 nodes are exact at ``s = 0.31``, 0.3% off at ``s = 0.69``,
        17% off at ``s = 1.04``). ``tests/models/test_generative_spectral_stats``
        asserts convergence over the range the model actually occupies; if a
        training run drives the learned gust ``sigma`` far past ~0.7, this
        approximation — not the optimizer — is the thing to revisit.

        Args:
            u: ``[B, M, n_env]`` flow speed (m/s).
            apply_gust: marginalize over the OU gust (False = gust-free response).
            n_quad: Gauss--Hermite nodes. See the validity note above.

        Returns:
            ``[B, M, n_env, n_freqs]`` expected **power** response. Power rather
            than magnitude so no ``sqrt``/square round trip sits in the graph —
            see PositionalHarmonicNoiseGen.spectral_stats for why that matters.
        """
        if not apply_gust:
            return self.filter_mags(u).pow(2)
        sigma = _pos(self.raw_sigma)
        nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_quad))
        nodes_t = u.new_tensor(nodes)
        weights_t = u.new_tensor(weights)
        weights_t = weights_t / weights_t.sum()
        power = torch.zeros_like(self.filter_mags(u))
        for node, weight in zip(nodes_t, weights_t):
            # x = sigma * node  (HermiteE nodes are already unit-variance scaled).
            # The exponent is clamped because the outermost nodes sit at ~4 sigma:
            # if training drives sigma up, exp(4 sigma) overflows and poisons the
            # whole objective with NaN. Clamping caps the modelled gust at e^8
            # (~3000x), far beyond any physical gust, and keeps the failure mode
            # a bounded bias instead of a crash. See the validity note above.
            g = torch.exp((sigma * node - 0.5 * sigma**2).clamp(-8.0, 8.0))
            power = power + weight * self.filter_mags(u * g).pow(2)
        return power.clamp_min(0.0)

    def forward(
        self,
        u: torch.Tensor,
        n_samples: int,
        dt: float,
        *,
        noise: torch.Tensor | None = None,
        apply_gust: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Synthesize incoherent per-mic wind audio from flow speed.

        Args:
            u: ``[B, M, n_env]`` per-mic flow speed (m/s).
            n_samples: output length ``T``.
            dt: seconds per envelope frame (for the OU gust).
            noise: optional ``[B, M, T]`` white-noise excitation (else drawn); pass
                it to control/seed the realization (e.g. reproducible synthesis).
            apply_gust: multiply ``U`` by an OU meander envelope before transducing.
            generator: RNG for the gust envelope and the default noise draw.

        Returns:
            ``[B, M, T]`` wind audio, independent (incoherent) across mics.
        """
        b, m, _ = u.shape
        device, dtype = u.device, u.dtype
        if apply_gust:
            g = ou_envelope(
                (b, m, self.n_env),
                _pos(self.raw_tau),
                _pos(self.raw_sigma),
                dt,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            u = u * g

        mags = self.filter_mags(u)  # [B, M, n_env, n_freqs]
        mags = mags.reshape(b * m, self.n_env, self.n_freqs)
        if noise is None:
            noise = torch.randn(b, m, n_samples, device=device, dtype=dtype, generator=generator)
        noise = noise.reshape(b * m, n_samples)
        wind = frequency_filter(noise, mags)  # [B*M, T]
        return wind.reshape(b, m, n_samples)


# ---------------------------------------------------------------------------
# Top module — RPS + geometry -> incoherent per-mic wind
# ---------------------------------------------------------------------------


class WindWakeChannel(nn.Module):
    """RPS + geometry → additive, incoherent per-microphone wind-noise channel.

    Ties modules A/B/C together and exposes the same ``rps + geometry → [B, M, T]``
    contract as :class:`PositionalHarmonicNoiseGen`, so its output is simply
    **added** to the coherent generator's output at each microphone. The physics
    (wake geometry, quad dynamics) carries no per-array switches: which mics get
    gusts is decided entirely by geometry, so joint DREGON + Michael's training
    needs the same forward path for both.

    Args:
        sample_rate: audio rate (Hz).
        n_rotors: rotors per drone.
        rotor_radius: default rotor radius (m); override per call.
        n_freqs / n_env: transduction filter grid / frame rate.
        use_dynamics: attach :class:`QuadDynamics` (module A). If False, only the
            hover / static path is available (``v_rel`` must be given or 0).
        hover_rps: hover anchor for the dynamics.
        mlp_hidden: transduction shape-residual MLP width.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_rotors: int = 4,
        rotor_radius: float = 0.127,  # ~10-inch prop, DREGON-scale
        n_freqs: int = 65,
        n_env: int = 64,
        use_dynamics: bool = False,
        hover_rps: float = 90.0,
        mlp_hidden: int = 16,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_rotors = n_rotors
        self.rotor_radius = float(rotor_radius)
        self.n_env = n_env

        # Aero constants (positive via softplus of these raws).
        self.raw_k = nn.Parameter(torch.tensor(0.54))  # k ~ 1.0
        self.raw_alpha = nn.Parameter(torch.tensor(-0.43))  # α ~ 0.5 (wake width /R)
        self.raw_beta = nn.Parameter(torch.tensor(-1.0))  # β decay, small
        self.raw_gate = nn.Parameter(torch.tensor(-2.3))  # gate softness ~0.1 m
        self.raw_cfree = nn.Parameter(torch.tensor(-2.3))  # freestream coupling

        self.transduction = WindTransduction(
            sample_rate=sample_rate, n_freqs=n_freqs, n_env=n_env, mlp_hidden=mlp_hidden
        )
        self.dynamics = (
            QuadDynamics(n_rotors=n_rotors, hover_rps=hover_rps) if use_dynamics else None
        )

    def flow_speed(
        self,
        rps: torch.Tensor,
        mic_pos: torch.Tensor,
        rotor_pos: torch.Tensor,
        *,
        rotor_axis: torch.Tensor | None = None,
        rotor_radius: torch.Tensor | float | None = None,
        v_rel: torch.Tensor | None = None,
        return_per_rotor: bool = False,
    ):
        """Compute per-mic flow speed ``U_m`` at envelope rate (module B).

        ``rps`` is ``[B, R, T]`` at audio rate; it is average-pooled to ``n_env``
        frames. ``v_rel`` may be given directly, or computed from the dynamics
        module when attached. Returns ``U_m`` ``[B, M, n_env]``.
        """
        if rps.dim() != 3:
            raise ValueError(f"rps must be [B, R, T], got {tuple(rps.shape)}")
        b = rps.shape[0]
        mic_pos = _as_batched(mic_pos, b, "mic_pos")
        rotor_pos = _as_batched(rotor_pos, b, "rotor_pos")
        if rotor_axis is None:
            rotor_axis = (
                rps.new_tensor([0.0, 0.0, -1.0]).reshape(1, 1, 3).expand(b, self.n_rotors, 3)
            )
        rr = self.rotor_radius if rotor_radius is None else rotor_radius
        if not torch.is_tensor(rr):
            rr = rps.new_tensor(float(rr))

        rps_env = F.adaptive_avg_pool1d(rps, self.n_env)  # [B, R, n_env]

        if v_rel is None and self.dynamics is not None:
            # V_rel is slow, so integrate the dynamics at the coarse envelope rate
            # (n_env steps) rather than audio rate (T steps) — same trajectory,
            # ~T/n_env cheaper, and keeps the Python loop CPU-light.
            t_audio = rps.shape[-1]
            env_dt = (t_audio / self.sample_rate) / self.n_env
            v_rel = self.dynamics(rps_env, rotor_pos, env_dt)  # [B, n_env, 3]

        c_free = _pos(self.raw_cfree) if v_rel is not None else None
        return wake_flow_speed(
            mic_pos,
            rotor_pos,
            rotor_axis,
            rr,
            rps_env,
            k=_pos(self.raw_k),
            alpha=_pos(self.raw_alpha),
            beta=_pos(self.raw_beta),
            gate_softness=_pos(self.raw_gate),
            v_rel=v_rel,
            c_free=c_free,
            return_per_rotor=return_per_rotor,
        )

    def forward(
        self,
        rps: torch.Tensor,
        mic_pos: torch.Tensor,
        rotor_pos: torch.Tensor,
        *,
        rotor_axis: torch.Tensor | None = None,
        rotor_radius: torch.Tensor | float | None = None,
        v_rel: torch.Tensor | None = None,
        n_samples: int | None = None,
        noise: torch.Tensor | None = None,
        apply_gust: bool = True,
        generator: torch.Generator | None = None,
        return_dict: bool = False,
    ):
        """Render the incoherent per-mic wind channel.

        Args:
            rps: ``[B, R, T]`` rotor speed (rev/s) at audio rate.
            mic_pos: ``[B, M, 3]`` or ``[M, 3]`` microphone positions (m).
            rotor_pos: ``[B, R, 3]`` or ``[R, 3]`` rotor-hub positions (m).
            rotor_axis: optional ``[B, R, 3]`` / ``[R, 3]`` unit downwash dirs
                (default straight down ``[0, 0, −1]``).
            rotor_radius: optional rotor radius override (m).
            v_rel: optional world-frame relative wind ``[B, n_env, 3]`` / ``[B, 3]``;
                if None and the dynamics module is attached it is derived from RPS,
                else hover (0).
            n_samples: output length ``T`` (defaults to the rps length).
            noise: optional ``[B, M, T]`` white-noise excitation (seed control).
            apply_gust: apply the OU gust envelope.
            generator: RNG for the gust and default noise draw.
            return_dict: also return the intermediate ``flow_speed`` / ``filter_mags``.

        Returns:
            ``[B, M, T]`` wind audio, or a dict with ``{"wind", "flow_speed",
            "filter_mags"}``.
        """
        t = rps.shape[-1] if n_samples is None else n_samples
        dt = self.n_env and (t / self.sample_rate) / self.n_env  # seconds per env frame
        u = self.flow_speed(
            rps,
            mic_pos,
            rotor_pos,
            rotor_axis=rotor_axis,
            rotor_radius=rotor_radius,
            v_rel=v_rel,
        )  # [B, M, n_env]
        assert isinstance(u, torch.Tensor)  # return_per_rotor not passed
        wind = self.transduction(u, t, dt, noise=noise, apply_gust=apply_gust, generator=generator)
        if return_dict:
            return {
                "wind": wind,
                "flow_speed": u,
                "filter_mags": self.transduction.filter_mags(u),
            }
        return wind

    def flow_speed_rel(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        *,
        rotor_axis: torch.Tensor | None = None,
        rotor_radius: torch.Tensor | float | None = None,
        v_rel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-mic flow speed ``U_m`` from the generator's ``rel_pos`` (module B).

        ``rel_pos`` is the coherent generator's rotor→mic vector, either
        ``[B, M, R, 3]`` (M observers) or ``[B, R, 3]`` (single observer). It *is*
        the wake-gate displacement, so no absolute positions are needed. Hover by
        default (``v_rel=0``); pass ``v_rel`` for free-flight. Returns
        ``[B, M, n_env]``.
        """
        if rps.dim() != 3:
            raise ValueError(f"rps must be [B, R, T], got {tuple(rps.shape)}")
        b = rps.shape[0]
        if rel_pos.dim() == 3:  # [B, R, 3] single observer -> disp [B, R, 1, 3]
            disp = rel_pos.unsqueeze(2)
        elif rel_pos.dim() == 4:  # [B, M, R, 3] -> [B, R, M, 3]
            disp = rel_pos.transpose(1, 2)
        else:
            raise ValueError(
                f"rel_pos must be [B, M, R, 3] or [B, R, 3], got {tuple(rel_pos.shape)}"
            )

        if rotor_axis is None:
            rotor_axis = (
                rps.new_tensor([0.0, 0.0, -1.0]).reshape(1, 1, 3).expand(b, self.n_rotors, 3)
            )
        rr = self.rotor_radius if rotor_radius is None else rotor_radius
        if not torch.is_tensor(rr):
            rr = rps.new_tensor(float(rr))
        rps_env = F.adaptive_avg_pool1d(rps, self.n_env)  # [B, R, n_env]
        c_free = _pos(self.raw_cfree) if v_rel is not None else None
        u = _flow_speed_from_disp(
            disp,
            rotor_axis,
            rr,
            rps_env,
            k=_pos(self.raw_k),
            alpha=_pos(self.raw_alpha),
            beta=_pos(self.raw_beta),
            gate_softness=_pos(self.raw_gate),
            v_rel=v_rel,
            c_free=c_free,
        )
        assert isinstance(u, torch.Tensor)
        return u

    def forward_rel(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        *,
        v_rel: torch.Tensor | None = None,
        n_samples: int | None = None,
        noise: torch.Tensor | None = None,
        apply_gust: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Wind audio from the generator's ``rel_pos`` — matches the coherent
        generator's output shape (``[B, M, T]``, or ``[B, T]`` for a single
        observer) so it can be summed directly onto it.
        """
        single = rel_pos.dim() == 3
        t = rps.shape[-1] if n_samples is None else n_samples
        dt = (t / self.sample_rate) / self.n_env
        u = self.flow_speed_rel(rps, rel_pos, v_rel=v_rel)  # [B, M, n_env]
        wind = self.transduction(u, t, dt, noise=noise, apply_gust=apply_gust, generator=generator)
        return wind.squeeze(1) if single else wind

    def expected_power_rel(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        *,
        v_rel: torch.Tensor | None = None,
        apply_gust: bool = True,
        n_quad: int = 5,
    ) -> torch.Tensor:
        """Gust-marginalized expected **power** per microphone.

        The distributional counterpart of :meth:`forward_rel`: same flow-speed
        physics, but returns the *spectral envelope* of the channel instead of
        one realization of it. ``[B, M, n_env, n_freqs]``. See
        :meth:`WindTransduction.expected_mags`.
        """
        u = self.flow_speed_rel(rps, rel_pos, v_rel=v_rel)  # [B, M, n_env]
        return self.transduction.expected_power(u, apply_gust=apply_gust, n_quad=n_quad)


def _resample_envelope(mags: torch.Tensor, frames: int, freqs: int) -> torch.Tensor:
    """Resample a ``[B, M, n_frames, n_freqs]`` power envelope onto a new grid.

    Both axes are uniform over fixed physical spans (the clip duration and
    ``0..Nyquist``), so a plain bilinear resize is the correct resampling.
    """
    if mags.shape[-2] == frames and mags.shape[-1] == freqs:
        return mags
    b, m = mags.shape[0], mags.shape[1]
    out = F.interpolate(
        mags.reshape(b * m, 1, mags.shape[-2], mags.shape[-1]),
        size=(frames, freqs),
        mode="bilinear",
        align_corners=True,
    )
    return out.reshape(b, m, frames, freqs)


class PositionalHarmonicPlusWindGen(nn.Module):
    """Coherent position-aware generator **plus** the additive wind-wake channel.

    Composes :class:`PositionalHarmonicNoiseGen` (harmonic bank + broadband, the
    coherent propagated field) with a :class:`WindWakeChannel` (incoherent flow
    noise), summed at each microphone. Exposes the exact
    :class:`tasks.noise_generation` contract ``forward(rps, rel_pos, z=None) →
    [B, M, T]`` — a drop-in for the coherent generator — so the codebook
    conditioner, codec, and training loop are unchanged. Per-drone conditioning
    ``z`` drives only the coherent emitter; the wind transduction is drone-shared
    (a microphone property), so it takes no ``z``.

    Extra kwargs (``**coherent_kwargs``) flow to the coherent generator (matching
    :func:`models.registry.build_noise_gen_model`'s call), so this class slots
    into ``NOISE_GEN_MODEL_REGISTRY`` next to the bare generator.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_harmonics: int = 100,
        cond_dim: int = 0,
        n_rotors: int = 4,
        wind_n_env: int = 64,
        wind_n_freqs: int = 65,
        wind_use_dynamics: bool = False,
        wind_mlp_hidden: int = 16,
        **coherent_kwargs,
    ):
        super().__init__()
        self.coherent = PositionalHarmonicNoiseGen(
            sample_rate=sample_rate, n_harmonics=n_harmonics, cond_dim=cond_dim, **coherent_kwargs
        )
        self.wind = WindWakeChannel(
            sample_rate=sample_rate,
            n_rotors=n_rotors,
            n_env=wind_n_env,
            n_freqs=wind_n_freqs,
            use_dynamics=wind_use_dynamics,
            mlp_hidden=wind_mlp_hidden,
        )
        self._register_load_state_dict_pre_hook(self._remap_coherent_only_checkpoint)

    @staticmethod
    def _remap_coherent_only_checkpoint(
        state_dict, prefix, _local_metadata, _strict, _missing, _unexpected, _errors
    ) -> None:
        """Accept a *coherent-only* checkpoint by nesting its keys under ``coherent.``.

        Adding the wind channel moved the generator's weights from ``<prefix>*``
        to ``<prefix>coherent.*``, which would otherwise make every
        magnitude-trained checkpoint silently load nothing here (``strict=False``
        reports it, but a warm start that loads zero weights looks like a fresh
        model, not an error). Warm-starting the wind arms from the coherent
        baseline is exactly what makes the likelihood trainable at all — an
        untrained mean puts ``(r - a)^2 / sigma2`` at ~1e12 and the first step
        destroys the model — so this remap is load-bearing, not a convenience.

        Only applies when the checkpoint has no ``coherent.``-prefixed keys, so a
        native wind checkpoint is passed through untouched.
        """
        own = [k for k in state_dict if k.startswith(prefix)]
        if not own or any(k.startswith(f"{prefix}coherent.") for k in own):
            return
        for key in own:
            suffix = key[len(prefix) :]
            if suffix.startswith("wind."):
                continue
            state_dict[f"{prefix}coherent.{suffix}"] = state_dict.pop(key)

    def forward(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        """``[B, M, T]`` = coherent(rps, rel_pos, z) + wind(rps, rel_pos)."""
        wind = self.wind.forward_rel(rps, rel_pos)
        if return_dict:
            out = self.coherent(rps, rel_pos, z=z, return_dict=True, **kwargs)
            out["audio"] = out["audio"] + wind
            out["wind"] = wind
            return out
        coherent = self.coherent(rps, rel_pos, z=z, **kwargs)
        return coherent + wind

    def spectral_stats(
        self,
        rps: torch.Tensor,
        rel_pos: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        v_rel: torch.Tensor | None = None,
        n_quad: int = 5,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predict the observed field as a distribution (mean + variance).

        This is the training-time counterpart of :meth:`forward` and the reason
        the wind channel can be fitted at all. Wind is *stochastic and
        unobservable*: the recording contains one gust realization that no model
        can reproduce, so comparing a synthesized gust to it — which
        :class:`losses.MultiScaleSTFTLoss` does — measures mostly the difference
        between two independent draws. The gradient is then dominated by that
        difference rather than by any parameter error, and the fitted level is
        biased low (see :mod:`losses.spectral_likelihood`).

        Here nothing is sampled. The wind contributes only **power**, added to
        the coherent generator's own broadband branch — the two are independent,
        so their powers sum:

            noise_psd = coherent_broadband_psd + wind_psd

        Returns ``{"coherent": [B, M, T], "noise_psd": [B, M, n_env, F]}``.
        """
        stats = self.coherent.spectral_stats(rps, rel_pos, z=z, **kwargs)
        wind_psd = self.wind.expected_power_rel(rps, rel_pos, v_rel=v_rel, n_quad=n_quad)
        base = stats["noise_psd"]  # [B, M, t_n, F_g]
        # Both envelopes are uniform over the same spans (clip duration,
        # 0..Nyquist), so they can be resampled onto a common grid — but always
        # onto the FINER of the two. Downsampling either one would discard
        # resolution the model actually has, and (with align_corners sampling
        # rather than averaging) would quietly lose power in the process.
        frames = max(base.shape[-2], wind_psd.shape[-2])
        freqs = max(base.shape[-1], wind_psd.shape[-1])
        total = (
            _resample_envelope(base, frames, freqs) + _resample_envelope(wind_psd, frames, freqs)
        ).clamp_min(0.0)
        return {"coherent": stats["coherent"], "noise_psd": total}
