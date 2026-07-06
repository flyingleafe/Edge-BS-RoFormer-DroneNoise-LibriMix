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

import torch
import torch.nn.functional as F

from models.multif0.utils import salience_target_from_resampled_rps
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
