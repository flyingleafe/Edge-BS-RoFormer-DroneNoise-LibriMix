"""Tests for the RPS-jitter Ornstein-Uhlenbeck injection in the harmonic emitter.

Covers:
(a) OU statistics  -- injected delta has std ~= sigma and correlation time ~= tau;
(b) train/eval convention -- random in train, deterministic in eval, overridable;
(c) spectral effect -- a high harmonic (k=30) is broadened far more than a low one
    (k=2), because the shared fundamental jitter scales the k-th harmonic by k;
(d) config path -- build_noise_gen_model plumbs the knob, and the three E6
    experiments Hydra-compose green.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from models.generative import HarmonicNoiseGenNew, PositionalHarmonicNoiseGen
from models.generative.dsp import harmonic_freq_series, oscillator_bank
from models.registry import build_noise_gen_model


def _emitter_of(model) -> HarmonicNoiseGenNew:
    gen = model.generator  # _CodebookConditionedNoiseGen -> PositionalHarmonicNoiseGen
    assert isinstance(gen, PositionalHarmonicNoiseGen)
    emitter = gen.emitter
    assert isinstance(emitter, HarmonicNoiseGenNew)
    return emitter


SR = 16000
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# (a) OU statistics
# ---------------------------------------------------------------------------


def _autocorr_time(x: np.ndarray, dt: float) -> float:
    """Correlation time from the 1/e crossing of the (row-averaged) autocorr."""
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    max_lag = min(n - 1, int(round(0.3 / dt)))  # look out to 0.3 s
    var = np.mean(np.sum(x * x, axis=-1))
    acf = np.array([np.mean(np.sum(x[:, : n - k] * x[:, k:], axis=-1)) for k in range(max_lag + 1)])
    acf = acf / var
    below = np.where(acf < np.exp(-1.0))[0]
    return float(below[0] * dt) if len(below) else max_lag * dt


def test_ou_statistics_std_and_correlation_time():
    torch.manual_seed(0)
    sigma, tau = 0.5, 0.05
    emitter = HarmonicNoiseGenNew(
        n_harmonics=8, sample_rate=SR, n_oscillators=1, rps_jitter_sigma=sigma, rps_jitter_tau=tau
    )
    t = 3 * SR  # 3 s
    f0s = torch.full((1, 64, t), 100.0)  # 64 independent rows for statistics
    delta = (emitter._apply_rps_jitter(f0s) - f0s).detach().numpy().reshape(64, t)

    # std ~= sigma within +/-20%
    assert delta.std() == pytest.approx(sigma, rel=0.2)
    # correlation time ~= tau within +/-50%
    tau_est = _autocorr_time(delta, dt=1.0 / SR)
    assert tau_est == pytest.approx(tau, rel=0.5)


def test_ou_jitter_off_when_sigma_zero():
    emitter = HarmonicNoiseGenNew(
        n_harmonics=8, sample_rate=SR, n_oscillators=1, use_diff_noise=False, rps_jitter_sigma=0.0
    )
    rps = torch.full((1, 1, SR), 100.0)
    emitter.train()
    # sigma=0 => no jitter even in train mode => forward is phase-only-random;
    # with zero phases pinned it must be exactly reproducible.
    ip = torch.zeros(1, 1, 8)
    a = emitter(rps, initial_phases=ip)
    b = emitter(rps, initial_phases=ip)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# (b) train/eval convention (mirrors the initial_phases convention)
# ---------------------------------------------------------------------------


def _jitter_emitter():
    torch.manual_seed(0)
    # use_diff_noise=False so the only randomness sources are phases + jitter;
    # pin phases to zero so we isolate the jitter contribution.
    return HarmonicNoiseGenNew(
        n_harmonics=16,
        sample_rate=SR,
        n_oscillators=1,
        use_diff_noise=False,
        rps_jitter_sigma=0.5,
        rps_jitter_tau=0.05,
    )


def test_eval_is_deterministic_train_is_random():
    emitter = _jitter_emitter()
    rps = torch.full((1, 1, SR), 100.0)
    ip = torch.zeros(1, 1, 16)  # pin phases so jitter is the only randomiser

    emitter.eval()  # jitter off at eval => identical
    assert torch.allclose(
        emitter(rps, initial_phases=ip), emitter(rps, initial_phases=ip), atol=1e-6
    )

    emitter.train()  # jitter on in train => different
    assert not torch.allclose(
        emitter(rps, initial_phases=ip), emitter(rps, initial_phases=ip), atol=1e-5
    )


def test_eval_forward_override_enables_jitter():
    emitter = _jitter_emitter().eval()
    rps = torch.full((1, 1, SR), 100.0)
    ip = torch.zeros(1, 1, 16)
    # explicit override turns jitter on even at eval => outputs differ
    a = emitter(rps, initial_phases=ip, rps_jitter=True)
    b = emitter(rps, initial_phases=ip, rps_jitter=True)
    assert not torch.allclose(a, b, atol=1e-5)
    # and forcing it off in train mode restores determinism
    emitter.train()
    c = emitter(rps, initial_phases=ip, rps_jitter=False)
    d = emitter(rps, initial_phases=ip, rps_jitter=False)
    assert torch.allclose(c, d, atol=1e-6)


# ---------------------------------------------------------------------------
# (c) spectral effect: k=30 broadens, k=2 barely
# ---------------------------------------------------------------------------


def _minus3db_width(wav: np.ndarray, f_lo: float, f_hi: float) -> float:
    """-3 dB (amplitude/sqrt2) spectral width (Hz) of the peak in [f_lo, f_hi]."""
    n = len(wav)
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(wav * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    peak = mag[band].max()
    above = (mag >= peak / np.sqrt(2.0)) & band
    df = freqs[1] - freqs[0]
    return float(above.sum() * df)


def test_high_harmonic_broadens_more_than_low():
    torch.manual_seed(0)
    sigma, tau = 0.5, 0.05
    emitter = HarmonicNoiseGenNew(
        n_harmonics=40, sample_rate=SR, n_oscillators=1, rps_jitter_sigma=sigma, rps_jitter_tau=tau
    )
    f0 = 100.0
    t = 4 * SR  # long clip => fine frequency resolution
    f0s = torch.full((1, 1, t), f0)
    amps = torch.ones(1, 1, 40, t)
    phases = torch.zeros(1, 1, 40)

    def render(jittered: bool) -> np.ndarray:
        base = emitter._apply_rps_jitter(f0s) if jittered else f0s
        freqs = harmonic_freq_series(base, 40)
        wav = oscillator_bank(freqs, amps, initial_phases=phases, sr=SR)  # [1, 1, T]
        return wav[0, 0].detach().numpy()

    clean = render(False)
    jit = render(True)

    for k, grow_min, grow_max, is_low in [(2, None, 1.3, True), (30, 2.0, None, False)]:
        fc = k * f0
        half = max(3.0 * k * sigma, 5.0)  # search window scaled by expected spread
        w_clean = _minus3db_width(clean, fc - half, fc + half)
        w_jit = _minus3db_width(jit, fc - half, fc + half)
        ratio = w_jit / w_clean
        if is_low:
            assert ratio < grow_max, f"k={k} width ratio {ratio:.2f} not < {grow_max}"
        else:
            assert ratio >= grow_min, f"k={k} width ratio {ratio:.2f} not >= {grow_min}"


# ---------------------------------------------------------------------------
# (d) config path + Hydra composition
# ---------------------------------------------------------------------------


def test_build_noise_gen_model_plumbs_jitter():
    model = build_noise_gen_model(
        "positional_harmonic_gen",
        cond_dim=16,
        drone_names=["dregon", "michaels"],
        rps_jitter_sigma=0.6,
        rps_jitter_tau=0.016,
    )
    emitter = _emitter_of(model)
    assert emitter.rps_jitter_sigma == pytest.approx(0.6)
    assert emitter.rps_jitter_tau == pytest.approx(0.016)


def test_build_noise_gen_model_plumbs_random_phases():
    model = build_noise_gen_model(
        "positional_harmonic_gen",
        cond_dim=16,
        drone_names=["dregon", "michaels"],
        use_random_phases=True,
    )
    assert _emitter_of(model).use_random_phases is True


@pytest.mark.parametrize(
    "experiment",
    ["e6_noisegen_baseline", "e6_noisegen_randphase", "e6_noisegen_jitter"],
)
def test_e6_experiments_compose(experiment: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from training.config import register_configs

    register_configs()
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        cfg = compose(
            config_name="config", overrides=[f"experiment={experiment}", "validate_only=true"]
        )
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
