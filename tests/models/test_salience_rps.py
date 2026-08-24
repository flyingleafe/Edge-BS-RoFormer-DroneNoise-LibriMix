"""Regression test for the ``salience_target_from_frame_rps`` refactor
(models.salience_rps.SalienceRPSPredictor): its core (post-resample,
nearest-bin + blur) math was extracted into
``models.multif0.utils.salience_target_from_resampled_rps`` so
``losses.salience.SalienceRPSBCELoss`` could reuse it without duplicating
the quantization logic (see REPLICATION.md § C7/C8). This test proves the
model method still produces exactly the same result as calling the shared
function directly with the model's own grid/resample — i.e. the refactor
didn't change behavior.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.multif0.utils import (
    linear_freq_grid,
    salience_target_from_resampled_rps,
    salience_to_rps_segmented,
)
from models.salience_rps import BasicPitchSalience


def test_salience_target_from_frame_rps_matches_shared_function_directly():
    torch.manual_seed(0)
    model = BasicPitchSalience(n_fft=512, hop_length=256, num_rotors=4, sr=16000, n_harmonics=2)
    model.eval()

    n_samples = 4000
    rps = torch.rand(2, 4, 20) * 60.0 + 30.0  # (B, R, T_stft), batched

    target = model.salience_target_from_frame_rps(rps, n_samples, blur_bins=1)

    n_grid = model.num_grid_frames(n_samples)
    rps_grid = F.interpolate(rps.float(), size=n_grid, mode="linear", align_corners=False)
    expected = salience_target_from_resampled_rps(rps_grid, model.output_freqs(), blur_bins=1)

    assert target.shape == (2, model.n_bins, n_grid)
    assert torch.equal(target, expected)


def test_salience_target_from_frame_rps_unbatched_roundtrips_batch_dim():
    torch.manual_seed(1)
    model = BasicPitchSalience(n_fft=512, hop_length=256, num_rotors=4, sr=16000, n_harmonics=2)
    n_samples = 4000
    rps_unbatched = torch.rand(4, 20) * 60.0 + 30.0  # (R, T_stft), no batch dim

    target = model.salience_target_from_frame_rps(rps_unbatched, n_samples, blur_bins=0)
    assert target.dim() == 2  # (n_bins, T_grid) -- squeezed back to unbatched

    target_batched = model.salience_target_from_frame_rps(
        rps_unbatched.unsqueeze(0), n_samples, blur_bins=0
    )
    assert torch.equal(target, target_batched.squeeze(0))


# ─── Zero-RPS decode convention ──────────────────────────────────────────────
# A frame with no salience peak above the tracker threshold must decode to
# 0 rev/s for every rotor. Silence == zero rotor speed is the project-wide
# convention (docs/experiments/honest-base-frontends.md); the tracker used to
# carry the last speed forward across a dark frame, which turned a stopped
# rotor into a phantom hover.


def _linear_grid_map(freqs: np.ndarray, n_frames: int) -> torch.Tensor:
    """All-dark salience map on ``freqs``, shape ``(n_bins, n_frames)``."""
    return torch.zeros(len(freqs), n_frames)


def test_all_dark_salience_decodes_to_zero_rps():
    freqs = linear_freq_grid(20.0, 130.0, 220)
    salience = _linear_grid_map(freqs, 24)

    rps, merge = salience_to_rps_segmented(salience, num_rotors=4, freqs=freqs, threshold=0.3)

    assert rps.shape == (4, 24)
    assert torch.all(rps == 0.0)
    assert not merge.any()


def test_four_peaks_decode_to_their_frequencies():
    freqs = linear_freq_grid(20.0, 130.0, 220)
    targets = [60.0, 70.0, 80.0, 90.0]
    salience = _linear_grid_map(freqs, 12)
    for f in targets:
        salience[int(np.abs(freqs - f).argmin()), :] = 1.0

    rps, _merge = salience_to_rps_segmented(salience, num_rotors=4, freqs=freqs, threshold=0.3)

    spacing = float(np.diff(freqs)[0])
    decoded = np.sort(rps[:, -1].numpy())
    assert np.allclose(decoded, targets, atol=spacing)
    # Every frame carries the same four speeds.
    assert np.allclose(np.sort(rps.numpy(), axis=0), np.array(targets)[:, None], atol=spacing)


def test_map_going_dark_midway_decodes_speeds_then_zeros():
    freqs = linear_freq_grid(20.0, 130.0, 220)
    targets = [60.0, 70.0, 80.0, 90.0]
    n_frames, dark_from = 20, 10
    salience = _linear_grid_map(freqs, n_frames)
    for f in targets:
        salience[int(np.abs(freqs - f).argmin()), :dark_from] = 1.0

    rps, _merge = salience_to_rps_segmented(salience, num_rotors=4, freqs=freqs, threshold=0.3)

    spacing = float(np.diff(freqs)[0])
    lit = np.sort(rps[:, :dark_from].numpy(), axis=0)
    assert np.allclose(lit, np.array(targets)[:, None], atol=spacing)
    # No hold-over: every dark frame is exactly zero for every rotor.
    assert torch.all(rps[:, dark_from:] == 0.0)
