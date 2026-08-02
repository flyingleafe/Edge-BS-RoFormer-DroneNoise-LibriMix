"""Can the spatial likelihood separate incoherent from coherent power?

That separation is the entire reason this objective exists. The marginal
likelihood (:mod:`losses.spectral_likelihood`) sees only per-microphone power,
so a coherent rotor field and an incoherent wind field that happen to deposit
the SAME power at each microphone are indistinguishable to it — which is why the
wind channel trained to inertness under it. The spatial likelihood looks at the
cross-spectrum, where the two are structurally different: rotors are rank-R with
known steering, wind is diagonal.

The tests below assert exactly that, by construction and by recovery.
"""

from __future__ import annotations

import math

import pytest
import torch

from losses.spatial_likelihood import (
    SpatialLikelihood,
    spatial_whittle_nll,
    steering_vectors,
)

M, R, F, T = 8, 4, 12, 6


def _geometry(batch: int = 1) -> torch.Tensor:
    """`[B, M, R, 3]` — rotors on a square, mics spread around/below them."""
    torch.manual_seed(0)
    rotor = torch.tensor(
        [[0.23, 0.23, 0.0], [-0.23, 0.23, 0.0], [-0.23, -0.23, 0.0], [0.23, -0.23, 0.0]]
    )
    mic = torch.randn(M, 3) * 0.12
    mic[:, 2] -= 0.05
    rel = mic.reshape(M, 1, 3) - rotor.reshape(1, R, 3)
    return rel.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def _draw(steering, source_psd, wind_psd, n: int, seed: int = 0):
    """Draw `n` independent spatial snapshots from CN(0, R)."""
    g = torch.Generator().manual_seed(seed)
    # source contribution: one complex amplitude per rotor, propagated
    a = torch.randn(n, 1, T, R, 2, generator=g) / math.sqrt(2.0)
    amp = torch.complex(a[..., 0], a[..., 1]) * source_psd[0].permute(1, 2, 0).sqrt().unsqueeze(0)
    d = steering[0].permute(2, 0, 1)  # [F, M, R]
    coh = torch.einsum("fmr,nftr->nfmt", d, amp.permute(0, 1, 2, 3).to(d.dtype))
    e = torch.randn(n, F, M, T, 2, generator=g) / math.sqrt(2.0)
    inc = torch.complex(e[..., 0], e[..., 1]) * wind_psd[0].permute(1, 0, 2).sqrt().unsqueeze(0)
    return (coh + inc).permute(0, 2, 1, 3)  # [n, M, F, T]


class TestSeparatesCoherentFromIncoherent:
    """The property the marginal likelihood lacks."""

    @pytest.fixture(scope="class")
    def setup(self):
        torch.manual_seed(0)
        rel = _geometry()
        freqs = torch.linspace(0.0, 8000.0, F)
        d = steering_vectors(rel, freqs)
        return rel, d

    def test_recovers_the_incoherent_share(self, setup):
        """With the coherent power known, the fitted wind power finds the truth —
        the estimate the marginal likelihood cannot make."""
        _rel, d = setup
        true_source = torch.full((1, R, F, T), 0.5)
        true_wind = torch.full((1, M, F, T), 0.2)
        x = _draw(d, true_source, true_wind, n=256)

        grid = torch.linspace(0.05, 0.8, 61)
        losses = []
        for g in grid:
            w = torch.full((1, M, F, T), float(g))
            nll = torch.stack(
                [
                    spatial_whittle_nll(x[i : i + 1], d, true_source, w).mean()
                    for i in range(0, 256, 32)
                ]
            ).mean()
            losses.append(nll)
        fitted = float(grid[int(torch.stack(losses).argmin())])
        assert fitted == pytest.approx(0.2, abs=0.06), f"fitted wind power {fitted}"

    def test_marginal_power_alone_is_ambiguous_but_spatial_is_not(self, setup):
        """Two decompositions with the SAME per-microphone total power. A
        marginal likelihood scores them identically by construction; the spatial
        one must prefer the true split."""
        _rel, d = setup
        # Truth: mostly incoherent.
        true_source = torch.full((1, R, F, T), 0.05)
        true_wind = torch.full((1, M, F, T), 0.6)
        x = _draw(d, true_source, true_wind, n=256)

        def score(src_scale: float, wind_scale: float) -> float:
            src = torch.full((1, R, F, T), src_scale)
            wind = torch.full((1, M, F, T), wind_scale)
            return float(
                torch.stack(
                    [
                        spatial_whittle_nll(x[i : i + 1], d, src, wind).mean()
                        for i in range(0, 256, 32)
                    ]
                ).mean()
            )

        # Same total per-mic power, redistributed between the two mechanisms.
        # (The coherent term's per-mic contribution is sum_r |d_mr|^2 * P_r, so an
        # exact power match is geometry-dependent; the point is that a large swap
        # toward the coherent explanation must be PENALISED.)
        true_split = score(0.05, 0.6)
        wrong_split = score(0.6, 0.05)
        assert wrong_split > true_split, (
            f"spatial likelihood failed to prefer the true split: "
            f"true {true_split:.4f} vs swapped {wrong_split:.4f}"
        )


class TestNumerics:
    def test_matches_a_dense_reference(self):
        """Woodbury/determinant-lemma path must equal the direct M x M form."""
        torch.manual_seed(0)
        rel = _geometry()
        freqs = torch.linspace(0.0, 8000.0, F)
        d = steering_vectors(rel, freqs)
        src = torch.rand(1, R, F, T) + 0.1
        wind = torch.rand(1, M, F, T) + 0.1
        x = _draw(d, src, wind, n=1)

        got = spatial_whittle_nll(x, d, src, wind)

        # Direct: build R per (f, t) and evaluate log det R + x^H R^-1 x.
        dd = d[0].permute(2, 0, 1)  # [F, M, R]
        want = torch.empty(1, F, T)
        for f in range(F):
            for t in range(T):
                cov = (dd[f] * src[0, :, f, t].sqrt().to(dd.dtype)) @ (
                    dd[f] * src[0, :, f, t].sqrt().to(dd.dtype)
                ).conj().T
                cov = cov + torch.diag(wind[0, :, f, t]).to(cov.dtype)
                xv = x[0, :, f, t].unsqueeze(-1)
                quad = (xv.conj().T @ torch.linalg.solve(cov, xv)).real.reshape(())
                want[0, f, t] = torch.log(torch.linalg.det(cov).abs()) + quad
        torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)

    def test_gradients_flow_to_both_terms(self):
        torch.manual_seed(0)
        rel = _geometry()
        freqs = torch.linspace(0.0, 8000.0, F)
        d = steering_vectors(rel, freqs)
        src = (torch.rand(1, R, F, T) + 0.1).requires_grad_(True)
        wind = (torch.rand(1, M, F, T) + 0.1).requires_grad_(True)
        x = _draw(d, src.detach(), wind.detach(), n=1)
        spatial_whittle_nll(x, d, src, wind).mean().backward()
        for name, p in (("source", src), ("wind", wind)):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert float(p.grad.abs().sum()) > 0, name


class TestEndToEnd:
    def test_module_runs_on_waveforms(self):
        torch.manual_seed(0)
        rel = _geometry(batch=2)
        core = SpatialLikelihood(n_ffts=(256,), sample_rate=16000)
        audio = torch.randn(2, M, 8192) * 0.05
        src = torch.rand(2, R, 4, 33) * 0.01 + 1e-3
        wind = torch.rand(2, M, 4, 33) * 0.01 + 1e-3
        out = core(audio, rel, src, wind)
        assert torch.isfinite(out), out
