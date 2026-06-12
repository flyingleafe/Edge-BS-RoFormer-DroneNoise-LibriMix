"""Tests for salience-map RPS baselines (LateDeepSalience, BasicPitchSalience).

Covers: round-trip quantization on both CQT grids, target smoothing, model
forward/predict_rps shapes, and a one-step BCE backward pass.
"""

import torch
import torch.nn.functional as F

from models.multif0.utils import rps_to_salience, salience_to_rps_segmented
from models.salience_rps import BasicPitchSalience, LateDeepSalience

# Grid descriptors: (fmin, n_bins, bins_per_octave). n_octaves/over_sample are
# ignored once n_bins/bins_per_octave are given, so we only carry the essentials.
HCQT_GRID = (32.7, 360, 60)
BP_GRID = (27.5, 264, 36)


def _to_salience(rps, n_grid, grid, *, blur_bins=0):
    fmin, n_bins, bpo = grid
    return rps_to_salience(
        rps,
        n_grid,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bpo,
        hcqt_sr=16000,
        hcqt_hop=256,
        rps_sr=1000.0,
        blur_bins=blur_bins,
    )


def _track(sal, grid):
    fmin, n_bins, bpo = grid
    return salience_to_rps_segmented(
        sal, num_rotors=4, fmin=fmin, n_bins=n_bins, bins_per_octave=bpo, threshold=0.0
    )


def _ramp_rps(n_rotors=4, t=2000):
    base = torch.tensor([40.0, 55.0, 70.0, 85.0])[:n_rotors].view(n_rotors, 1)
    return base + torch.linspace(0.0, 3.0, t).view(1, t)


def _sorted_rmse(rec, gt):
    rec_s, _ = rec.sort(0)
    gt_s, _ = gt.sort(0)
    return ((rec_s - gt_s) ** 2).mean().sqrt().item()


def test_roundtrip_quantization_floor():
    """GT RPS -> binary salience -> Hungarian tracking recovers RPS within ~1 bin."""
    rps = _ramp_rps(t=2000)  # 2.0 s at rps_sr=1000
    # Match the salience time grid to the RPS span: 2.0 s * 16000/256 = 125 frames.
    n_grid = 125
    for grid in (HCQT_GRID, BP_GRID):
        n_bins = grid[1]
        sal = _to_salience(rps, n_grid, grid)
        assert sal.shape == (n_bins, n_grid)
        # binary target: ~4 active bins per frame
        assert abs((sal > 0.5).float().sum(0).mean().item() - 4.0) < 0.5
        rec, merge = _track(sal, grid)
        rps_g = F.interpolate(
            rps.unsqueeze(0), size=n_grid, mode="linear", align_corners=False
        ).squeeze(0)
        rmse = _sorted_rmse(rec, rps_g)
        # Well-separated slow ramps: tracking error stays near the bin-quantization floor.
        assert rmse < 3.0, f"grid n_bins={n_bins} round-trip RMSE {rmse:.2f} Hz too high"


def test_target_blur():
    """blur_bins=0 stays binary; blur_bins>0 produces a soft [0,1] target."""
    rps = _ramp_rps()
    binary = _to_salience(rps, 200, HCQT_GRID, blur_bins=0)
    soft = _to_salience(rps, 200, HCQT_GRID, blur_bins=2)
    vals = binary.unique().tolist()
    assert set(vals).issubset({0.0, 1.0}), f"binary target not binary: {vals}"
    assert soft.max().item() <= 1.0 and soft.min().item() >= 0.0
    # blur spreads each rotor over 2*blur+1 bins → more nonzeros
    assert (soft > 0).float().sum() > (binary > 0).float().sum()


def test_model_shapes_and_backward():
    """forward -> (B, n_bins, T); predict_rps -> (B, 4, T_stft); BCE backprops."""
    audio = torch.randn(2, 16000 * 2)
    rps = _ramp_rps(t=2000)
    for builder, n_bins in ((LateDeepSalience, 360), (BasicPitchSalience, 264)):
        model = builder(n_fft=2048, hop_length=512)
        assert model.outputs_salience is True
        assert model.n_bins == n_bins

        logits = model(audio)
        assert logits.shape[0] == 2 and logits.shape[1] == n_bins
        assert torch.isfinite(logits).all()

        # Target on the model's grid, aligned to the model's actual frame count.
        target = model.salience_target(rps, audio.shape[-1], blur_bins=2)
        if target.shape[-1] != logits.shape[-1]:
            target = F.interpolate(
                target.unsqueeze(0), size=logits.shape[-1], mode="linear", align_corners=False
            ).squeeze(0)
        target = target.unsqueeze(0).expand(2, -1, -1)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads, "no gradients produced"
        assert all(torch.isfinite(g).all() for g in grads)

        with torch.no_grad():
            rps_pred = model.predict_rps(audio, threshold=0.0)
        t_stft = audio.shape[-1] // 512 + 1
        assert rps_pred.shape == (2, 4, t_stft)
        assert torch.isfinite(rps_pred).all()


if __name__ == "__main__":
    test_roundtrip_quantization_floor()
    test_target_blur()
    test_model_shapes_and_backward()
    print("all salience_rps tests passed")
