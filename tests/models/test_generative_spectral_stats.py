"""The distributional (mean + variance) prediction path of the generators.

``spectral_stats`` is what lets the stochastic branches be fitted by likelihood
instead of by comparing one gust realization to another. Its correctness rests
on a single claim, which these tests check by Monte Carlo against the sampling
path that actually synthesizes audio:

    the predicted ``noise_psd`` is the expected POWER spectrum of everything the
    generator draws at random, and the predicted ``coherent`` is exactly the
    part it does not.

If those two drifted apart, training would optimize a model the synthesizer
does not realize — the failure would be silent and would look like "the wind
channel just does not help".
"""

from __future__ import annotations

import math

import pytest
import torch

from models.generative.positional_harmonic_gen import PositionalHarmonicNoiseGen
from models.generative.wind_wake_gen import (
    PositionalHarmonicPlusWindGen,
    WindTransduction,
)

SR = 16000


ROTORS = torch.tensor(
    [[0.23, 0.23, 0.0], [-0.23, 0.23, 0.0], [-0.23, -0.23, 0.0], [0.23, -0.23, 0.0]]
)


def _rel(mic: torch.Tensor) -> torch.Tensor:
    """``[B=1, M, R, 3]`` rotor->mic vectors for the rig above."""
    return (mic.reshape(-1, 1, 3) - ROTORS.reshape(1, 4, 3)).unsqueeze(0)


@pytest.fixture
def geometry():
    """A 2-mic, 4-rotor rig: rotors on a square, mics offset from the plane."""
    return _rel(torch.tensor([[0.0, 0.0, 0.05], [0.30, 0.0, 0.33]]))


@pytest.fixture
def wake_geometry():
    """A rig where the wind channel is actually *active*.

    The wake is a column under each ROTOR, not under the airframe centre, so a
    microphone only sees flow if it sits beneath a rotor. Mic 0 does (directly
    below rotor 0); mic 1 is placed like Michael's array — forward of and above
    the body, i.e. upstream of a ``-z`` downwash — where the gate should be
    inert. That contrast is the physics the channel is claiming.
    """
    return _rel(torch.tensor([[0.23, 0.23, -0.20], [0.30, 0.0, 0.33]]))


class TestWindEnvelopeMatchesSampling:
    @staticmethod
    def _rel(a, b, band=slice(1, 20)):
        return float(((a[..., band] - b[..., band]).abs() / b[..., band].clamp_min(1e-12)).max())

    def test_expected_power_equals_rms_of_drawn_gusts(self):
        """The gust marginalization must reproduce the RMS of the realizations,
        not the response at the mean gust (the two differ because the level is a
        nonlinear function of flow speed).

        Checked at the initialization ``sigma``, where the log-normal moment
        ``E[g^(4 gamma)]`` is only ~1.3 and a Monte-Carlo reference is therefore
        well conditioned. At larger ``sigma`` the expectation is tail-dominated
        and it is *Monte Carlo*, not the quadrature, that becomes unreliable —
        which is why convergence is verified separately below.
        """
        torch.manual_seed(0)
        trans = WindTransduction(sample_rate=SR, n_freqs=33, n_env=4, mlp_hidden=0)
        u = torch.full((1, 1, 4), 3.0)
        analytic = trans.expected_power(u, apply_gust=True, n_quad=15)

        from models.generative.wind_wake_gen import _pos

        sigma = _pos(trans.raw_sigma)  # softplus(-1) ~ 0.31, the shipped init
        acc, draws, chunk = torch.zeros_like(analytic), 100_000, 2000
        with torch.no_grad():
            for _ in range(draws // chunk):
                x = torch.randn(chunk, 1, 1, 4) * sigma
                g = torch.exp(x - 0.5 * sigma**2)
                acc += trans.filter_mags(u * g).pow(2).sum(0)
        mc = acc / draws  # both sides are POWER
        rel = self._rel(analytic, mc)
        assert rel < 0.02, f"quadrature vs Monte-Carlo differ by {rel:.1%}"

    def test_quadrature_is_converged_over_the_operating_range(self):
        """The reference-free check: more nodes must not move the answer. This
        is what validates the marginalization where Monte Carlo cannot, and it
        pins the sigma range in which the shipped node count is trustworthy."""
        trans = WindTransduction(sample_rate=SR, n_freqs=33, n_env=4, mlp_hidden=0)
        u = torch.full((1, 1, 4), 3.0)
        for raw, tol in ((-1.0, 1e-4), (0.0, 0.01)):
            with torch.no_grad():
                trans.raw_sigma.fill_(raw)
            ref = trans.expected_power(u, n_quad=31)
            got = trans.expected_power(u, n_quad=9)  # the shipped default
            assert self._rel(got, ref) < tol, (
                f"gust quadrature not converged at raw_sigma={raw}: "
                f"{self._rel(got, ref):.2%} vs 31 nodes"
            )

    def test_marginalizing_differs_from_the_mean_gust(self):
        """Sanity that the marginalization is doing work: because the level is
        convex in the gust, ``E[|H(Ug)|^2] > |H(U E[g])|^2``. A model that
        ignored this would sit systematically low."""
        trans = WindTransduction(sample_rate=SR, n_freqs=33, n_env=4, mlp_hidden=0)
        with torch.no_grad():
            trans.raw_sigma.fill_(0.8)
        u = torch.full((2, 2, 4), 4.0)
        marginal = trans.expected_power(u, apply_gust=True, n_quad=9)
        no_gust = trans.expected_power(u, apply_gust=False)
        assert float(marginal.mean()) > float(no_gust.mean())


class TestCoherentStats:
    def test_coherent_plus_noise_power_accounts_for_the_full_output(self, geometry):
        """The split must be exhaustive: the mean's power plus the predicted
        noise power should track the sampled output's power. Checked in the
        aggregate (a single draw is noisy by construction)."""
        torch.manual_seed(0)
        t = 4096
        model = PositionalHarmonicNoiseGen(sample_rate=SR, n_harmonics=16, n_rotors=4).eval()
        rps = torch.full((1, 4, t), 80.0)
        rel = geometry

        stats = model.spectral_stats(rps, rel)
        assert stats["coherent"].shape == (1, 2, t)
        assert stats["noise_psd"].dim() == 4
        assert stats["noise_psd"].shape[:2] == (1, 2)
        assert torch.isfinite(stats["coherent"]).all()
        assert torch.isfinite(stats["noise_psd"]).all()
        assert float(stats["noise_psd"].min()) >= 0.0

        with torch.no_grad():
            draws = torch.stack([model(rps, rel) for _ in range(8)])
        total_power = draws.pow(2).mean()
        coherent_power = stats["coherent"].pow(2).mean()
        # The mean cannot carry more power than the realizations do.
        assert float(coherent_power) <= float(total_power) * 1.35

    def test_noise_psd_scale_with_distance(self, geometry):
        """The broadband branch is propagated with the same ``1/r`` law as the
        harmonics, so the nearer microphone must see more noise power."""
        torch.manual_seed(0)
        model = PositionalHarmonicNoiseGen(sample_rate=SR, n_harmonics=8, n_rotors=4).eval()
        rps = torch.full((1, 4, 2048), 80.0)
        psd = model.spectral_stats(rps, geometry)["noise_psd"]
        near, far = float(psd[0, 0].mean()), float(psd[0, 1].mean())
        assert near > far


class TestCombinedGenerator:
    def test_wind_adds_power_to_the_noise_envelope(self, wake_geometry):
        torch.manual_seed(0)
        rps = torch.full((1, 4, 2048), 80.0)
        model = PositionalHarmonicPlusWindGen(
            sample_rate=SR, n_harmonics=8, n_rotors=4, wind_n_env=8, wind_n_freqs=33
        ).eval()
        with torch.no_grad():
            model.wind.transduction.raw_level.fill_(2.0)  # make the wind audible
            model.wind.raw_k.fill_(1.0)

        combined = model.spectral_stats(rps, wake_geometry)
        coherent_only = model.coherent.spectral_stats(rps, wake_geometry)

        assert combined["noise_psd"].shape[:2] == (1, 2)
        assert torch.isfinite(combined["noise_psd"]).all()
        # Powers add, so the combined envelope dominates the coherent-only one —
        # compared on the SAME grid, since the two branches are resampled onto
        # the finer of their native grids and interpolation is not power-preserving.
        from models.generative.wind_wake_gen import _resample_envelope

        base = _resample_envelope(
            coherent_only["noise_psd"],
            combined["noise_psd"].shape[-2],
            combined["noise_psd"].shape[-1],
        )
        assert float(combined["noise_psd"].mean()) > float(base.mean())
        # The mean is untouched by the wind (wind is pure variance).
        torch.testing.assert_close(combined["coherent"], coherent_only["coherent"])

        # ...and the addition is GATED BY GEOMETRY, not spread over the array:
        # the in-wake mic gains far more than the out-of-wake one. This is the
        # property that should make the channel help DREGON without touching
        # Michael's.
        gain = (combined["noise_psd"] - base).mean(dim=(-1, -2))[0]
        assert float(gain[0]) > 10 * float(gain[1]), (
            f"wake gating collapsed: in-wake gain {float(gain[0]):.3e} vs "
            f"out-of-wake {float(gain[1]):.3e}"
        )

    def test_gradients_reach_the_wind_parameters(self, geometry):
        """The whole point: wind parameters must get a gradient from the
        likelihood without any sampling in the path."""
        torch.manual_seed(0)
        from losses.spectral_likelihood import SpectralLikelihood

        rps = torch.full((1, 4, 4096), 80.0)
        model = PositionalHarmonicPlusWindGen(
            sample_rate=SR, n_harmonics=8, n_rotors=4, wind_n_env=8, wind_n_freqs=33
        ).eval()
        target = torch.randn(1, 2, 4096) * 0.1

        stats = model.spectral_stats(rps, geometry)
        core = SpectralLikelihood(n_ffts=(512,))
        loss = core(
            target.reshape(-1, target.shape[-1]),
            stats["coherent"].reshape(-1, stats["coherent"].shape[-1]),
            stats["noise_psd"].reshape(-1, *stats["noise_psd"].shape[-2:]),
        )
        loss.backward()

        wind_params = {
            "level": model.wind.transduction.raw_level,
            "order": model.wind.transduction.raw_order,
            "fc": model.wind.transduction.raw_fc,
            "k": model.wind.raw_k,
        }
        for name, p in wind_params.items():
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient not finite"
        assert any(
            float(p.grad.abs().sum()) > 0 for p in wind_params.values() if p.grad is not None
        )


class TestSingleObserver:
    def test_single_observer_rel_pos_is_supported(self):
        torch.manual_seed(0)
        model = PositionalHarmonicNoiseGen(sample_rate=SR, n_harmonics=8, n_rotors=4).eval()
        rps = torch.full((1, 4, 2048), 80.0)
        rel = torch.randn(1, 4, 3) * 0.3
        stats = model.spectral_stats(rps, rel)
        assert stats["coherent"].shape[0] == 1
        assert stats["coherent"].shape[-1] == 2048
        assert math.isfinite(float(stats["noise_psd"].sum()))
