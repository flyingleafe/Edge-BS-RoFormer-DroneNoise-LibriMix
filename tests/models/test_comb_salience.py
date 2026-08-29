"""The comb-salience family must CONTAIN the classical scan, not approximate it."""

import numpy as np
import pytest
import torch

from data_processing.comb_bench import comb_clip
from models.comb_salience import CombGather, CombScoreHead
from tracking.comb_seed import _periodogram, comb_score

SR = 16000


@pytest.mark.parametrize("seed", [7000, 7137, 7274])
def test_classical_corner_case_is_exact(seed):
    """`CombGather` + a classical head reproduces `comb_score` to round-off.

    This is the whole architectural claim: the learned family has the classical
    method as a point in its parameter space, so training can only depart from
    it if departing helps. If this test drifts, that claim is gone.

    The grid must be passed VERBATIM. `np.arange` and `torch.linspace` disagree
    by about 1e-12 rev/s over this range, which at the 40th harmonic is enough
    to move the interpolation and show up as a 1e-11 mismatch — the operator is
    exact, the discretization is what differs.
    """
    y, _, _ = comb_clip(seed, centre=75.0, spread=11.0)
    n = int(SR * 0.25)
    s0 = len(y) // 2
    pw, f, noise, _ = _periodogram(y[s0 : s0 + n], SR)
    n_fft = 2 * (len(f) - 1)
    grid = np.arange(30.0, 100.0, 0.02)

    ref = comb_score(pw, f, noise, grid, k_max=40, f_max=7500.0)
    gather = CombGather(k_max=40, sr=SR, n_fft=n_fft, f_max=7500.0, grid=grid)
    h = gather(torch.tensor(pw, dtype=torch.float64)[None, :, None])
    got = CombScoreHead(40, "classical")(h, torch.full_like(h, float(noise)), gather.count)

    assert np.abs(got[0, :, 0].numpy() - ref).max() < 1e-13


@pytest.mark.parametrize("mode", ["learned", "learned_cond"])
def test_learned_head_starts_at_the_classical_score(mode):
    """A freshly built learned head must equal the classical one exactly.

    Training departs from the classical method only by learning; at
    initialization the two heads are the same function, so any gain is
    attributable to training rather than to a different starting point.
    """
    torch.manual_seed(0)
    h = torch.rand(2, 40, 64, 5, dtype=torch.float64) * 1e4
    floor = torch.full_like(h, 3.0)
    count = torch.full((64,), 40.0, dtype=torch.float64)
    grid = torch.linspace(30.0, 100.0, 64, dtype=torch.float64)
    classical = CombScoreHead(40, "classical")(h, floor, count, grid)
    learned = CombScoreHead(40, mode).double()(h, floor, count, grid)
    assert torch.allclose(classical, learned, atol=1e-12)
