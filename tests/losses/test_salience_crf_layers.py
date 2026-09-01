"""The CRF layer loss must agree with the decoder it is named after."""

from __future__ import annotations

import numpy as np
import tdseries as td
import torch

from losses.salience_crf_layers import LayerPITCRFLoss, gold_indices, layer_pit_crf_nll
from models.multif0.utils import linear_freq_grid
from models.salience_crf import (
    band_for_rev_s,
    crf_decode_layers,
    gaussian_layer_target,
)

R, G, T = 4, 60, 40
FMIN, FMAX = 0.0, 150.0


def _setup(max_step=25.0):
    grid = linear_freq_grid(FMIN, FMAX, G)
    step = float(grid[1] - grid[0])
    span, pen = band_for_rev_s(max_step, step)
    return grid, span, pen


def _trajectory(seed=0):
    """Four smooth rotor tracks, one of them stopped for the whole clip."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = 70.0 + 8.0 * np.sin(2 * np.pi * t / T)
    rps = np.stack([base + rng.uniform(-3, 3) for _ in range(R - 1)] + [np.zeros(T)])
    return torch.tensor(rps, dtype=torch.float32).unsqueeze(0)  # (1, R, T)


def test_nll_is_non_negative_and_perfect_scores_lower_than_wrong():
    grid, span, pen = _setup()
    rps = _trajectory()
    gold = gold_indices(rps, grid)

    perfect = torch.log(gaussian_layer_target(rps, grid, sigma_bins=1.0).clamp_min(1e-30))
    wrong = torch.log(
        gaussian_layer_target(rps.flip(-1) + 20.0, grid, sigma_bins=1.0).clamp_min(1e-30)
    )

    good = layer_pit_crf_nll(perfect, gold, span, pen)
    bad = layer_pit_crf_nll(wrong, gold, span, pen)
    assert good >= -1e-4, good
    assert good < bad, (good, bad)


def test_gold_path_is_what_the_decoder_returns_on_a_perfect_layer():
    """The loss's gold indices and the deployed Viterbi path must coincide."""
    grid, span, pen = _setup()
    rps = _trajectory(seed=3)
    perfect = torch.log(gaussian_layer_target(rps, grid, sigma_bins=1.0).clamp_min(1e-30))
    decoded = crf_decode_layers(perfect, grid, span, pen, logits=True, subgrid=False)
    gold_rate = torch.tensor(np.asarray(grid), dtype=torch.float32)[gold_indices(rps, grid)]
    assert torch.equal(decoded, gold_rate)


def test_permutation_invariance():
    grid, span, pen = _setup()
    rps = _trajectory(seed=1)
    gold = gold_indices(rps, grid)
    scores = torch.log(gaussian_layer_target(rps, grid, sigma_bins=1.0).clamp_min(1e-30))
    a = layer_pit_crf_nll(scores, gold, span, pen)
    b = layer_pit_crf_nll(scores[:, [2, 0, 3, 1]], gold, span, pen)
    assert torch.allclose(a, b, atol=1e-5), (a, b)


def test_stopped_rotor_is_index_zero_not_dropped():
    grid, _, _ = _setup()
    rps = _trajectory()
    gold = gold_indices(rps, grid)
    assert int(gold[0, -1].max()) == 0


def test_gradient_reaches_the_logits():
    grid, span, pen = _setup(max_step=8.0)
    rps = _trajectory(seed=2)
    gold = gold_indices(rps, grid)
    z = torch.zeros(1, R, G, T, requires_grad=True)
    loss = layer_pit_crf_nll(torch.nn.functional.logsigmoid(z), gold, span, pen)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0


def test_frame_adapter_matches_the_functional_form():
    grid = linear_freq_grid(FMIN, FMAX, G)
    rps = _trajectory(seed=4)
    loss = LayerPITCRFLoss(out_fmin=FMIN, out_fmax=FMAX, out_bins=G, n_layers=R)
    logits = torch.randn(1, R * G, T)
    scores, gold = loss.scores_and_gold(logits, rps)
    assert scores.shape == (1, R, G, T)
    assert torch.equal(gold, gold_indices(rps, grid))


def test_half_precision_inputs_do_not_overflow_the_neg_sentinel():
    """AMP feeds float16 scores; -1e30 is not representable in half."""
    grid, span, pen = _setup(max_step=8.0)
    rps = _trajectory(seed=5)
    gold = gold_indices(rps, grid)
    scores = torch.log(gaussian_layer_target(rps, grid, sigma_bins=1.0).clamp_min(1e-30)).half()
    out = layer_pit_crf_nll(scores, gold, span, pen.half())
    assert torch.isfinite(out), out


def test_frame_adapter_runs_under_autocast():
    """The loss must be float32 inside, whatever autocast is doing outside."""
    rps = _trajectory(seed=6)
    loss = LayerPITCRFLoss(out_fmin=FMIN, out_fmax=FMAX, out_bins=G, n_layers=R, max_step_rev_s=8.0)
    pred = td.Frame(
        {
            "salience": td.uniform(
                torch.randn(1, R * G, T).numpy(), 100, dims=("batch", "freq", "time"), t_start=0.0
            )
        }
    )
    tgt = td.Frame(
        {"rps": td.uniform(rps.numpy(), 100, dims=("batch", "rotor", "time"), t_start=0.0)}
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        out = loss(pred, tgt)
    assert out.dtype == torch.float32 and torch.isfinite(out), (out.dtype, out)
