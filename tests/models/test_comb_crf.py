"""The CRF must be the deployed decoder, not merely similar to it."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models import comb_crf
from tracking.comb_seed import _viterbi_ridge

SLEW, DT, STIFF = 12.0, 0.032, 40.0


def _setup(n_g=90, n_t=30, seed=0):
    torch.manual_seed(seed)
    grid = np.linspace(30.0, 100.0, n_g)
    scores = torch.randn(2, n_g, n_t, dtype=torch.float64)
    step_free = SLEW * DT / (grid[1] - grid[0])
    span, pen = comb_crf.band_penalty(step_free, STIFF, dtype=torch.float64)
    return grid, scores, span, pen


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_viterbi_is_the_classical_ridge(seed):
    """Index for index — the loss below trains on exactly what is deployed."""
    grid, scores, span, pen = _setup(seed=seed)
    got = comb_crf.viterbi(scores, span, pen)
    for b in range(scores.shape[0]):
        ref = _viterbi_ridge(scores[b].numpy().T, grid, SLEW, DT, STIFF)
        assert np.array_equal(got[b].numpy(), ref)


def test_log_partition_dominates_the_best_path():
    """`log Z` sums over every path, so it cannot fall below the best one."""
    _, scores, span, pen = _setup()
    path = comb_crf.viterbi(scores, span, pen)
    nll = comb_crf.crf_nll(scores, span, pen, path)
    assert bool((nll >= -1e-9).all())


def test_log_partition_collapses_to_viterbi_when_sharpened():
    """Scaled up, the soft recursion must approach the hard one it replaces."""
    _, scores, span, pen = _setup(n_g=40, n_t=12)
    for scale in (50.0, 500.0):
        s, p = scores * scale, pen * scale
        gap = (comb_crf.log_partition(s, span, p)
               - comb_crf.path_score(s, span, p, comb_crf.viterbi(s, span, p)))
        assert float(gap.max()) < 60.0 / scale * 40.0


def test_marginals_normalize_and_pass_gradients():
    _, scores, span, pen = _setup(n_g=40, n_t=12)
    scores = scores.requires_grad_(True)
    m = comb_crf.posterior_marginals(scores, span, pen)
    assert torch.allclose(m.sum(dim=1), torch.ones_like(m.sum(dim=1)), atol=1e-6)
    m.sum().backward()
    assert bool(torch.isfinite(scores.grad).all())
