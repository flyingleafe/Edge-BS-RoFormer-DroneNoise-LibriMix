"""Statistical correctness of the Rice/Whittle spectral likelihood.

These tests are the justification for replacing the magnitude loss on the
stochastic branches, so they assert the *estimator* properties rather than
merely exercising the code path:

- the new objective recovers the true power of a random component (unbiased),
  while the magnitude loss it replaces converges to the Rayleigh median and is
  low by a factor ``ln 2``;
- it reduces to magnitude matching when the component is deterministic;
- it is invariant to the realization's phase, which is what lets a generator
  with unknown rotor phases be trained at all;
- the analytic ``sigma2 = mags^2 * ||w||^2`` conversion agrees with a
  Monte-Carlo estimate of what the sampling path actually produces.
"""

from __future__ import annotations

import math

import pytest
import torch

from losses.spectral_likelihood import (
    SpectralLikelihood,
    rice_nll,
    split_coherence,
)
from models.generative.dsp import frequency_filter


def _fit_scalar(objective, grid):
    """Argmin of ``objective`` over a 1-D grid (a search, not a gradient step —
    so a failure indicts the objective and not the optimizer)."""
    values = torch.stack([objective(g) for g in grid])
    return grid[int(values.argmin())]


class TestEstimatorIsUnbiased:
    """The property the old loss lacks: the right answer for random signals."""

    @pytest.fixture(scope="class")
    def rayleigh(self):
        torch.manual_seed(0)
        true_power = 0.7
        # CN(0, true_power): torch.randn(cfloat) has unit total variance.
        x = torch.randn(200_000, dtype=torch.cfloat) * math.sqrt(true_power)
        return x.abs(), true_power

    def test_whittle_limit_recovers_true_power(self, rayleigh):
        r, true_power = rayleigh
        a = torch.zeros(())  # pure-noise bin
        grid = torch.linspace(0.3 * true_power, 2.0 * true_power, 601)
        fitted = _fit_scalar(lambda s2: rice_nll(r, a, s2).mean(), grid)
        assert fitted == pytest.approx(true_power, rel=0.02)

    def test_magnitude_loss_is_biased_low_by_ln2(self, rayleigh):
        """The defect being fixed, pinned numerically so it cannot regress
        silently: an L1 magnitude fit lands on the Rayleigh median, whose power
        is ``ln 2`` (-1.6 dB) of the truth."""
        r, true_power = rayleigh
        grid = torch.linspace(0.2, 1.5, 1301) * math.sqrt(true_power)
        fitted_amp = _fit_scalar(lambda a: (a - r).abs().mean(), grid)
        ratio = float(fitted_amp**2 / true_power)
        assert ratio == pytest.approx(math.log(2.0), rel=0.03)
        assert 10 * math.log10(ratio) == pytest.approx(-1.6, abs=0.1)


class TestLimits:
    def test_reduces_to_magnitude_matching_when_deterministic(self):
        """With negligible variance the minimizer is ``a = r`` — so the loss is
        not a regression anywhere the old one was already correct."""
        r = torch.tensor([1.0, 2.0, 0.5])
        grid = torch.linspace(0.1, 3.0, 581)
        for target in r:
            fitted = _fit_scalar(lambda a, t=target: rice_nll(t, a, torch.tensor(1e-4)), grid)
            assert fitted == pytest.approx(float(target), abs=0.02)

    def test_zero_amplitude_equals_whittle_closed_form(self):
        r = torch.rand(64) + 0.1
        s2 = torch.rand(64) + 0.1
        got = rice_nll(r, torch.zeros_like(r), s2)
        want = s2.log() + r.pow(2) / s2
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_is_finite_across_extreme_snr(self):
        r = torch.tensor([0.0, 1e-8, 1.0, 1e4])
        for a in (0.0, 1e-8, 1.0, 1e4):
            for s2 in (1e-8, 1.0, 1e4):
                out = rice_nll(r, torch.tensor(a), torch.tensor(s2))
                assert torch.isfinite(out).all(), (a, s2, out)


class TestCoherenceSplit:
    def test_conserves_total_power(self):
        a = torch.tensor([2.0, 3.0])
        s2 = torch.tensor([0.5, 0.25])
        for gamma in (0.0, 0.3, 0.75, 1.0):
            a_eff, s2_eff = split_coherence(a, s2, torch.tensor(gamma))
            torch.testing.assert_close(a_eff.pow(2) + s2_eff, a.pow(2) + s2)

    def test_endpoints(self):
        a, s2 = torch.tensor([2.0]), torch.tensor([0.5])
        a1, s1 = split_coherence(a, s2, torch.tensor(1.0))
        torch.testing.assert_close(a1, a)
        torch.testing.assert_close(s1, s2)
        a0, s0 = split_coherence(a, s2, torch.tensor(0.0))
        torch.testing.assert_close(a0, torch.zeros_like(a))
        torch.testing.assert_close(s0, s2 + a.pow(2))


class TestPhaseInvariance:
    """A generator does not know the recording's rotor phases, so the objective
    must not reward matching them.

    Frequency is bin-centred (``500 = 8 * 16000/256``) throughout so that
    ordinary spectral leakage — which genuinely is phase-dependent for an
    off-bin tone, and is a property of the STFT rather than of this loss — does
    not confound what is being tested.
    """

    @staticmethod
    def _setup(mag: float):
        torch.manual_seed(1)
        sr, n_fft, samples = 16000, 256, 16384
        core = SpectralLikelihood(n_ffts=(n_fft,))
        t = torch.arange(samples) / sr
        target = (torch.sin(2 * math.pi * 500 * t) + 0.1 * torch.randn(samples)).unsqueeze(0)
        mags = torch.full((1, 4, n_fft // 2 + 1), mag)
        preds = {p: torch.sin(2 * math.pi * 500 * t + p).unsqueeze(0) for p in (0.0, 0.7, math.pi)}
        return core, target, mags, preds

    def test_sign_flip_is_exactly_free(self):
        """``phase = pi`` negates the waveform. A magnitude-domain objective
        cannot see that; a waveform-domain one is dominated by it."""
        core, target, mags, preds = self._setup(0.05)
        assert float(core(target, preds[0.0], mags)) == pytest.approx(
            float(core(target, preds[math.pi], mags)), rel=1e-6
        )
        wave = [float(((target - preds[p]) ** 2).mean()) for p in (0.0, math.pi)]
        assert max(wave) / min(wave) > 10

    def test_phase_dependence_vanishes_at_a_realistic_noise_floor(self):
        """Residual phase sensitivity is leakage between neighbouring bins, so
        it shrinks as the noise floor rises to a realistic level, while a
        waveform loss stays phase-dominated no matter what."""
        spreads = {}
        for mag in (0.05, 0.5):
            core, target, mags, preds = self._setup(mag)
            losses = [float(core(target, preds[p], mags)) for p in preds]
            spreads[mag] = (max(losses) - min(losses)) / abs(losses[0])
        # ~22 dB coherent-to-noise, the regime rotor audio actually sits in.
        assert spreads[0.5] < 0.01
        # ...and the dependence is monotonically a high-SNR leakage artefact.
        assert spreads[0.5] < 0.05 * spreads[0.05]

        core, target, mags, preds = self._setup(0.5)
        wave = [float(((target - preds[p]) ** 2).mean()) for p in preds]
        assert (max(wave) - min(wave)) / abs(wave[0]) > 10


class TestNoisePowerConversion:
    def test_analytic_sigma2_matches_monte_carlo(self):
        """``sigma2 = mags^2 * ||w||^2`` must equal the power the *sampling* path
        actually produces, or the likelihood would be fitting a different model
        than the synthesizer realizes."""
        torch.manual_seed(0)
        n_fft, n_grid, batch, samples = 1024, 65, 32, 32768
        mags = torch.zeros(batch, n_grid)
        mags[:, 20:40] = 0.7
        audio = frequency_filter(torch.randn(batch, samples), mags)
        window = torch.hann_window(n_fft)
        spec = torch.stft(
            audio, n_fft=n_fft, hop_length=256, window=window, return_complex=True, center=True
        )
        measured = spec.abs().pow(2).mean(dim=(0, 2))
        # Interior of the passband only (band edges leak across bins).
        lo = int(22 / (n_grid - 1) * (n_fft // 2))
        hi = int(38 / (n_grid - 1) * (n_fft // 2))
        analytic = 0.7**2 * window.pow(2).sum()
        assert measured[lo:hi].mean() == pytest.approx(float(analytic), rel=0.05)


class TestEndToEndFit:
    def test_recovers_a_mixed_tone_plus_noise_spectrum(self):
        """The realistic case: a deterministic tone on top of shaped noise. The
        objective must recover the noise level from a single realization of the
        noise without the tone contaminating the estimate."""
        torch.manual_seed(0)
        sr, samples, n_grid = 16000, 32768, 65
        true_level = 0.3
        mags = torch.full((24, n_grid), true_level)
        noise = frequency_filter(torch.randn(24, samples), mags)
        t = torch.arange(samples) / sr
        tone = 2.0 * torch.sin(2 * math.pi * 1000 * t).unsqueeze(0).expand(24, -1)
        observed = tone + noise

        core = SpectralLikelihood(n_ffts=(1024,))
        grid = torch.linspace(0.4, 2.0, 161) * true_level

        def objective(level):
            m = torch.full((24, 4, n_grid), float(level))
            return core(observed, tone, m)

        assert float(_fit_scalar(objective, grid)) == pytest.approx(true_level, rel=0.12)

    def test_gradients_flow_to_both_branches(self):
        torch.manual_seed(0)
        core = SpectralLikelihood(n_ffts=(256,))
        target = torch.randn(2, 4096)
        coherent = torch.randn(2, 4096, requires_grad=True)
        mags = torch.full((2, 4, 33), 0.1, requires_grad=True)
        core(target, coherent, mags).backward()
        assert coherent.grad is not None and torch.isfinite(coherent.grad).all()
        assert mags.grad is not None and torch.isfinite(mags.grad).all()
        assert mags.grad.abs().sum() > 0
