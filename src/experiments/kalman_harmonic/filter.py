"""RPS-driven complex Kalman harmonic tracker (Phase 0 — no learning).

Tracks the complex amplitude of every rotor harmonic with a scalar
information-form Kalman filter per (rotor, harmonic) channel, then subtracts
the re-synthesized harmonic noise. Causal, streaming, O(R·H) per sample.

Model (per channel k = (rotor r, harmonic h), demodulated coordinates):

    c_t = ā · c_{t-1} + ε_t,   ε_t ~ CN(0, p_k)        ā = exp(−γ/sr)
    y_t = c_t + ν_t,           ν_t ~ CN(0, 1/q)

where y_t = 2 · v_t · exp(−i·φ_k(t)) is the mic sample demodulated by the
RPS-integrated phase φ_k(t) = 2π Σ_{s≤t} f_k(s)/sr (synchronous detection).
This is the change of variables z = c·e^{iφ} applied to a rotating-latent
model z_t = e^{(−γ + i·2π f)Δt} z_{t-1} + ..., cf. KLA (arXiv:2602.10743)
with complex transition: rotation drops out of the precision recursion
(|a|² = ā²) and lives only in the demodulation.

Interpretation of terms:
  * speech + broadband noise = the *measurement noise* of this model (1/q);
  * RPS jitter / accumulated phase error = the *process noise* p_k — this is
    the robustness mechanism. p_k grows like h² because a fundamental-freq
    error of σ_f Hz produces per-step phase error 2π·h·σ_f/sr at harmonic h;
  * the Kalman gain q/(λ⁻+q) is a per-harmonic adaptive tracking bandwidth
    (classical PLL/RLS equivalences apply).

The demodulated measurement also contains the image term c̄·e^{−2iφ} and
cross-terms from other harmonics; they oscillate at ≥ the fundamental rate
and are rejected by the tracker's low bandwidth (standard synchronous
detection argument) — at the price of the diagonal approximation: channels
with nearly coincident frequencies (4 rotors in hover!) compete for the same
energy. Phase 0 measures that cost against the joint lstsq solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrackerConfig:
    gamma: float = 10.0  # amplitude mean-reversion rate [1/s] (~1/coherence time)
    p_base: float = 1e-7  # process-noise floor per step (phase-drift slack)
    p_h2_scale: float = 1e-8  # h²-scaled component: p_k = p_base + p_h2_scale·h²
    meas_prec: float | None = None  # q; None → 1/var(wav) (speech+broadband floor)
    init_prec: float = 1e-4  # λ_0: nearly flat prior over amplitudes


def harmonic_phases(fund_freqs: torch.Tensor, n_harmonics: int, sr: int) -> torch.Tensor:
    """Integrated phase of each harmonic. fund_freqs [R, T] Hz → φ [R, H, T].

    float64 accumulation: at 16 kHz a float32 cumsum drifts audibly within
    seconds, which would masquerade as RPS error.
    """
    h = torch.arange(1, n_harmonics + 1, dtype=torch.float64, device=fund_freqs.device)
    freqs = h[None, :, None] * fund_freqs.to(torch.float64)[:, None, :]  # [R,H,T]
    return torch.cumsum(2.0 * torch.pi * freqs / sr, dim=-1)


@torch.no_grad()
def kalman_harmonic_track(
    wav: torch.Tensor,  # [T] real mixture
    fund_freqs: torch.Tensor,  # [R, T] fundamental (= n_blades·RPS) in Hz
    n_harmonics: int,
    sr: int = 16000,
    cfg: TrackerConfig | None = None,
) -> dict:
    """Run the tracker; returns noise estimate and enhanced signal.

    Returns dict with:
      noise_hat [T]   — strictly causal harmonic-noise estimate (prior mean:
                        the prediction for sample t uses samples < t only, so
                        the current speech sample cannot leak into it);
      enhanced  [T]   — wav − noise_hat;
      c_post [R,H,T]  — posterior complex amplitudes (analysis output);
      lam    [R,H,T]  — posterior precisions (per-harmonic confidence).
    """
    cfg = cfg or TrackerConfig()
    T = wav.shape[-1]
    R = fund_freqs.shape[0]
    H = n_harmonics
    dev = wav.device

    phi = harmonic_phases(fund_freqs, H, sr)  # [R,H,T] f64
    rot = torch.exp(-1j * phi)  # e^{−iφ}, c128
    demod = 2.0 * wav.to(torch.float64)[None, None, :] * rot  # y_t  [R,H,T]

    # Nyquist guard: freeze channels while their frequency is out of band.
    h_idx = torch.arange(1, H + 1, dtype=torch.float64, device=dev)
    freqs = h_idx[None, :, None] * fund_freqs.to(torch.float64)[:, None, :]
    alive = (freqs < sr / 2).reshape(R * H, T)  # [K,T] bool

    abar = torch.tensor(float(torch.exp(torch.tensor(-cfg.gamma / sr))), dtype=torch.float64)
    a2 = abar * abar
    q = cfg.meas_prec if cfg.meas_prec is not None else 1.0 / wav.to(torch.float64).var().item()
    p = (cfg.p_base + cfg.p_h2_scale * h_idx**2).repeat(R)  # [K]

    K = R * H
    y = demod.reshape(K, T)
    rot_f = rot.reshape(K, T)

    eta = torch.zeros(K, dtype=torch.complex128, device=dev)
    lam = torch.full((K,), cfg.init_prec, dtype=torch.float64, device=dev)

    noise_hat = torch.empty(T, dtype=torch.float64, device=dev)
    c_post = torch.empty(K, T, dtype=torch.complex128, device=dev)
    lam_out = torch.empty(K, T, dtype=torch.float64, device=dev)

    for t in range(T):
        live = alive[:, t]
        # ---- predict (information form; KLA eqs with |a|² = ā²) ----
        denom = a2 + p * lam
        lam_prior = lam / denom
        eta_prior = (abar / denom) * eta
        # strictly causal noise prediction: re-modulate the prior mean
        # (rot = e^{−iφ}, so e^{+iφ} = rot.conj())
        c_prior = eta_prior / lam_prior.clamp_min(1e-30)
        noise_hat[t] = ((c_prior * rot_f[:, t].conj()).real * live).sum()
        # ---- update ----
        lam_new = lam_prior + q
        eta_new = eta_prior + q * y[:, t]
        lam = torch.where(live, lam_new, lam_prior)
        eta = torch.where(live, eta_new, eta_prior)
        c_post[:, t] = eta / lam.clamp_min(1e-30)
        lam_out[:, t] = lam

    enhanced = wav.to(torch.float64) - noise_hat
    return {
        "noise_hat": noise_hat.to(wav.dtype),
        "enhanced": enhanced.to(wav.dtype),
        "c_post": c_post.reshape(R, H, T),
        "lam": lam_out.reshape(R, H, T),
    }
