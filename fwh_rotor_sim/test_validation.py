"""Validation tests for the FWH rotor simulator."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from fwh_rotor_sim import Blade, FWHRotorSolver, Rotor


def test_stationary_dipole():
    """Compare against analytic stationary dipole solution.

    A stationary harmonic dipole p(t) = p0 sin(omega t) at origin
    produces pressure at observer x: p'(t) = (1/4πr) (r̂ · ṗ(t-r/c))
    """
    print("\n=== Test: Stationary Dipole ===")
    # Create a fake "rotor" with one blade at zero radius (stationary)
    blade = Blade(radius=1e-6, chord=0.01, twist_deg=0.0, hub_radius=0.0, n_radial=1)
    rotor = Rotor(blade=blade, num_blades=1)

    # Override: set blade force manually as a harmonic dipole
    # This is a hack to test the FWH kernel without full rotor aerodynamics
    solver = FWHRotorSolver(rotor, c0=343.0)

    # We'll test via a direct FWH call with prescribed dipole
    from fwh_rotor_sim.fwh import Farassat1ASolver

    fwh = Farassat1ASolver(c0=343.0)

    omega = 2 * torch.pi * 100  # 100 Hz
    p0 = torch.tensor([0.0, 0.0, 1.0])  # dipole along z

    t = torch.linspace(0, 0.1, 1000)
    x_obs = torch.tensor([1.0, 0.0, 0.0])
    r = torch.norm(x_obs)
    r_hat = x_obs / r

    def y_func(tau):
        if tau.dim() == 1:
            return torch.zeros(1, tau.shape[0], 3)
        return torch.zeros(tau.shape[0], tau.shape[1], 3)

    def v_func(tau):
        if tau.dim() == 1:
            return torch.zeros(1, tau.shape[0], 3)
        return torch.zeros(tau.shape[0], tau.shape[1], 3)

    def F_func(tau):
        # Dipole force: F = p0 * sin(omega * tau)
        if tau.dim() == 1:
            F = p0[None, None, :] * torch.sin(omega * tau)[None, :, None]
            return F.expand(1, tau.shape[0], 3)
        F = p0[None, None, :] * torch.sin(omega * tau)[..., None]
        return F.expand(tau.shape[0], tau.shape[1], 3)

    def Fdot_func(tau):
        if tau.dim() == 1:
            Fdot = p0[None, None, :] * omega * torch.cos(omega * tau)[None, :, None]
            return Fdot.expand(1, tau.shape[0], 3)
        Fdot = p0[None, None, :] * omega * torch.cos(omega * tau)[..., None]
        return Fdot.expand(tau.shape[0], tau.shape[1], 3)

    def Mdot_func(tau):
        if tau.dim() == 1:
            return torch.zeros(1, tau.shape[0], 3)
        return torch.zeros(tau.shape[0], tau.shape[1], 3)

    p_num = fwh.compute_pressure(
        t, x_obs, y_func, v_func, F_func, Fdot_func, Mdot_func=Mdot_func, include_term3=False
    )

    # Analytic: p'(t) = (1/4πr) (r̂ · ṗ(t-r/c))
    tau_analytic = t - r / 343.0
    pdot = omega * torch.cos(omega * tau_analytic)
    p_analytic = (1.0 / (4 * torch.pi * r)) * r_hat[2] * pdot  # only z component contributes

    error = torch.max(torch.abs(p_num - p_analytic))
    print(f"Max error vs analytic: {error:.6e} Pa")
    assert error < 1e-3, f"Stationary dipole error too large: {error}"
    print("PASSED")
    return p_num, p_analytic, t


def test_hovering_rotor():
    """Test a hovering rotor and check physical trends."""
    print("\n=== Test: Hovering Rotor ===")

    # Small drone rotor: 6 inch (0.152 m) radius, 2 blades
    blade = Blade(
        radius=0.152,
        chord=0.020,
        twist_deg=lambda r: 15.0 * (1.0 - (r - 0.02) / 0.132),
        hub_radius=0.02,
        n_radial=30,
    )
    rotor = Rotor(blade=blade, num_blades=2)
    solver = FWHRotorSolver(rotor, c0=343.0, rho0=1.225)

    # Observer 1m in x-direction, in plane of rotation
    x_obs = torch.tensor([1.0, 0.0, 0.0])

    # 5000 RPM = 523.6 rad/s
    Omega = 2 * torch.pi * 5000 / 60  # ~523.6 rad/s

    # Compute one rotation period
    T_rev = 2 * torch.pi / Omega
    t = torch.linspace(0, 2 * T_rev, 2000)

    p = solver.compute_pressure(x_obs, t, Omega)

    # Basic sanity checks
    assert torch.isfinite(p).all(), "Pressure has NaN/Inf"
    print(f"Pressure range: [{p.min():.6e}, {p.max():.6e}] Pa")

    # Check periodicity: signal should repeat every rotation period
    p_T = p[:1000]
    p_2T = p[1000:2000]
    periodicity_error = torch.max(torch.abs(p_T - p_2T))
    print(f"Periodicity error (T vs 2T): {periodicity_error:.6e}")

    # SPL check: fundamental frequency should be at BPF = B * Omega / (2π)
    bpf = rotor.B * Omega / (2 * torch.pi)
    print(f"Blade Passage Frequency: {bpf:.1f} Hz")

    # Compute spectrum
    dt = t[1] - t[0]
    p_windowed = p * torch.hann_window(p.shape[0])
    spectrum = torch.fft.rfft(p_windowed)
    freqs = torch.fft.rfftfreq(p.shape[0], dt)
    psd = torch.abs(spectrum) ** 2

    # Peak should be near BPF
    peak_idx = torch.argmax(psd[1:]) + 1  # skip DC
    peak_freq = freqs[peak_idx]
    print(f"Spectral peak at {peak_freq.item():.1f} Hz (expected ~{bpf:.1f} Hz)")

    # SPL (dB re 20 μPa)
    prms = torch.sqrt(torch.mean(p**2))
    spl = 20 * torch.log10(prms / 20e-6)
    print(f"SPL: {spl.item():.1f} dB")

    freq_error = abs(peak_freq.item() - bpf)
    assert freq_error < 10.0, f"Spectral peak far from BPF: {peak_freq.item():.1f} vs {bpf:.1f} Hz"
    print("PASSED")
    return t, p, freqs, psd


def test_variable_speed():
    """Test with time-varying rotor speed."""
    print("\n=== Test: Variable Speed ===")

    blade = Blade(
        radius=0.152,
        chord=0.020,
        twist_deg=15.0,
        hub_radius=0.02,
        n_radial=20,
    )
    rotor = Rotor(blade=blade, num_blades=2)
    solver = FWHRotorSolver(rotor, c0=343.0)

    x_obs = torch.tensor([1.0, 0.0, 0.0])
    t = torch.linspace(0, 0.2, 2000)

    # Sinusoidally varying speed: 4000-6000 RPM
    Omega_mean = 2 * torch.pi * 5000 / 60
    Omega_amp = 2 * torch.pi * 1000 / 60
    freq_mod = 5.0  # 5 Hz modulation

    def Omega_func(tau):
        return Omega_mean + Omega_amp * torch.sin(2 * torch.pi * freq_mod * tau)

    p = solver.compute_pressure(x_obs, t, Omega_func)

    assert torch.isfinite(p).all(), "Variable speed pressure has NaN/Inf"
    print(f"Variable speed pressure range: [{p.min():.6e}, {p.max():.6e}] Pa")
    print("PASSED")
    return t, p


def _test_multi_observer_core(dtype=torch.float32):
    """Core multi-observer test, parameterized by float dtype."""
    label = "float32" if dtype == torch.float32 else "float64"
    print(f"\n--- dtype={label} ---")

    blade = Blade(
        radius=0.152,
        chord=0.020,
        twist_deg=lambda r: 15.0 * (1.0 - (r - 0.02) / 0.132),
        hub_radius=0.02,
        n_radial=30,
    )
    rotor = Rotor(blade=blade, num_blades=2)
    solver = FWHRotorSolver(rotor, c0=343.0, rho0=1.225)

    # Two observers at different positions
    x_obs_multi = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # observer 0: in-plane
            [0.0, 0.0, -1.0],  # observer 1: on-axis below
        ],
        dtype=dtype,
    )

    Omega = torch.tensor(2 * torch.pi * 5000 / 60, dtype=dtype)
    T_rev = 2 * torch.pi / Omega
    t = torch.linspace(0, 2 * T_rev, 1000, dtype=dtype)

    # Vectorized call
    p_multi = solver.compute_pressure(x_obs_multi, t, Omega)
    assert p_multi.shape == (2, 1000), f"Expected shape (2, 1000), got {p_multi.shape}"
    assert torch.isfinite(p_multi).all(), f"Multi-observer pressure has NaN/Inf ({label})"

    # Verify against single-observer calls
    p0 = solver.compute_pressure(x_obs_multi[0], t, Omega)
    p1 = solver.compute_pressure(x_obs_multi[1], t, Omega)

    err0 = torch.max(torch.abs(p_multi[0] - p0)).item()
    err1 = torch.max(torch.abs(p_multi[1] - p1)).item()
    print(f"Max error vs single-observer: obs0={err0:.6e}, obs1={err1:.6e}")

    # Tolerances scale with machine epsilon
    eps = np.finfo(np.float32).eps if dtype == torch.float32 else np.finfo(np.float64).eps
    # ~10x epsilon is a safe tight tolerance for this reduction chain
    tol = 10 * eps
    assert err0 < tol, f"Observer 0 mismatch ({label}): {err0} >= {tol}"
    assert err1 < tol, f"Observer 1 mismatch ({label}): {err1} >= {tol}"

    # In-plane observer should have larger amplitude than on-axis
    rms0 = p_multi[0].std().item()
    rms1 = p_multi[1].std().item()
    print(f"RMS pressure: in-plane={rms0:.6e}, on-axis={rms1:.6e}")
    assert rms0 > rms1, "In-plane observer should be louder than on-axis"

    return t, p_multi


def test_multi_observer():
    """Test vectorized multi-observer computation at float32 and float64."""
    print("\n=== Test: Multi-Observer Vectorization ===")

    t, p_multi = _test_multi_observer_core(dtype=torch.float32)
    _, _ = _test_multi_observer_core(dtype=torch.float64)

    print("PASSED")
    return t, p_multi


if __name__ == "__main__":
    print("Running FWH Rotor Simulator Validation Tests")
    print("=" * 50)

    p_num, p_analytic, t = test_stationary_dipole()

    t_rot, p_rot, freqs, psd = test_hovering_rotor()

    t_var, p_var = test_variable_speed()

    t_multi, p_multi = test_multi_observer()

    print("\n" + "=" * 50)
    print("All validation tests passed!")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t[:500], p_num[:500], label="Numeric")
    ax.plot(t[:500], p_analytic[:500], "--", label="Analytic", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Stationary Dipole Validation")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_rot, p_rot)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Hovering Rotor Pressure Signal")
    ax.grid(True)

    ax = axes[1, 0]
    ax.semilogy(freqs[:100], psd[:100])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Rotor Noise Spectrum")
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t_var, p_var)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Variable Speed Rotor")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("/tmp/fwh_validation.png", dpi=150)
    print("\nPlot saved to /tmp/fwh_validation.png")
