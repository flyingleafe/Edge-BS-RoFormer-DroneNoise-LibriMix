"""Numerical gates for the tensor trajectory reader, using real transforms.

No renderer, learned model, or fitted calibration enters these tests. Phase
products remove a channel's constant phase, not independent mixture phases.
The model owns shared-Q normalization, floor descriptors and lag/energy masks;
the reader's validity describes geometry only.
"""

from __future__ import annotations

import math

import pytest
import torch

from tracking.dsp import analytic_signal_tensor, demodulate_trajectories

SR = 16000
HOP = 512
ENV = 500
LAG = 16


def _tone(n: int, frequency: float = 800.0, phase: float = 0.0) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64) / SR
    return torch.cos(2 * math.pi * frequency * t + phase).float()[None]


def _read(audio, rates, orders=(10.0,), *, pad=8000, bands=(8.0, 32.0, 128.0), chunk=4):
    analytic = analytic_signal_tensor(audio, pad_samples=pad)
    z, valid = demodulate_trajectories(
        analytic,
        rates,
        torch.tensor(orders, device=audio.device),
        n_samples=audio.shape[-1],
        pad_samples=pad,
        half_bandwidths=bands,
        harmonic_chunk=chunk,
    )
    return analytic, z, valid


def _products(z, valid, lag=LAG):
    return z[..., lag:] * z[..., :-lag].conj(), valid[..., lag:] & valid[..., :-lag]


def _central(mask, duration, *, lag=LAG):
    times = torch.arange(lag, mask.shape[-1] + lag, device=mask.device) / ENV
    return mask & (times >= 0.5) & (times <= duration - 0.5)


def _inferred_error(z, valid, duration, order=10.0):
    product, mask = _products(z, valid)
    mask = _central(mask, duration)
    return torch.angle(product[mask].mean()) / (2 * math.pi * order * LAG / ENV)


@pytest.mark.parametrize("n", [1023, 1024])
def test_analytic_transform_keeps_real_part_and_removes_negative_image(n):
    audio = _tone(n, frequency=812.5)
    analytic = analytic_signal_tensor(audio, pad_samples=37)
    expected = torch.nn.functional.pad(audio, (37, 37))
    assert analytic.dtype == torch.complex64
    # Two complex64 FFTs incur absolute roundoff even where the padded signal is zero.
    torch.testing.assert_close(analytic.real, expected, atol=1e-6, rtol=2e-6)
    spectrum = torch.fft.fft(analytic)
    negative = spectrum[..., spectrum.shape[-1] // 2 + 1 :]
    assert negative.abs().max() < 2e-6 * spectrum.abs().max()


@pytest.mark.parametrize("candidate,expected", [(79.5, 0.5), (80.0, 0.0), (80.5, -0.5)])
def test_tone_product_sign_and_sub_bin_precision(candidate, expected):
    audio = _tone(2 * SR)
    rates = torch.full((1, 1, audio.shape[-1] // HOP + 1), candidate)
    _, z, valid = _read(audio, rates)
    assert z.dtype == torch.complex64
    for width in range(3):
        error = _inferred_error(z[..., width, :], valid[..., width, :], 2)
        assert abs(float(error) - expected) < 0.02
    # 32 ms at order ten and +0.5 rev/s must rotate about +1.005 rad.
    q, mask = _products(z[..., 1, :], valid[..., 1, :])
    angle = torch.angle(q[_central(mask, 2)].mean())
    assert abs(float(angle) - 2 * math.pi * 10 * expected * 0.032) < 0.04


@pytest.mark.parametrize("hold_early", [False, True])
def test_exact_linear_chirp_is_stationary_on_physical_timestamps(hold_early):
    n = 2 * SR
    count = 32 if hold_early else n // HOP + 1
    frame_times = torch.arange(count, dtype=torch.float64) * HOP / SR
    rates = (65 + 12 * frame_times).float()[None, None]
    t = torch.arange(n, dtype=torch.float64) / SR
    stop = float(frame_times[-1])
    running = t.clamp_max(stop)
    # Analytic integral of the linear ramp followed by endpoint holding.
    cycles = 65 * running + 6 * running.square() + (t - running) * (65 + 12 * stop)
    audio = torch.cos(2 * math.pi * 10 * cycles).float()[None]
    _, z, valid = _read(audio, rates, bands=(32.0,))
    q, mask = _products(z, valid)
    errors = torch.angle(q[_central(mask, 2)]) / (2 * math.pi * 10 * LAG / ENV)
    assert errors.abs().mean() < 0.02
    assert torch.quantile(errors.abs(), 0.95) < 0.02


@pytest.mark.parametrize("smooth_edges", [False, True])
def test_constant_phase_gauge_preserves_shared_reference_features(smooth_edges):
    n = 4 * SR
    t = torch.arange(n, dtype=torch.float64) / SR
    # Check the rectangular crop itself as well as a smoothly gated tone.
    taper = torch.sin(math.pi * t / 4).square().float()[None] if smooth_edges else 1.0
    audio = torch.cat([_tone(n, phase=p) * taper for p in (0.0, 0.73)])
    rates = torch.full((2, 1, n // HOP + 1), 79.5)
    analytic, z, valid = _read(audio, rates)
    reference = analytic[:, 8000 : 8000 + n].abs().square().mean(-1)
    power = z.abs().square() / reference[:, None, None, None, None]
    q, mask = _products(z, valid)
    q = q / reference[:, None, None, None, None]
    times = torch.arange(z.shape[-1]) / ENV
    keep = valid[0] & (times >= 1) & (times <= 3)
    torch.testing.assert_close(power[0][keep], power[1][keep], atol=1e-4, rtol=0)
    keep_q = _central(mask[0], 4) & (times[LAG:] >= 1) & (times[LAG:] <= 3)
    torch.testing.assert_close(q[0][keep_q], q[1][keep_q], atol=1e-4, rtol=0)


def test_overlapping_sources_do_not_have_independent_phase_invariance():
    n = 2 * SR
    audio = torch.cat([_tone(n) + 0.8 * _tone(n, 804.0, p) for p in (0.0, math.pi / 2)])
    rates = torch.full((2, 1, n // HOP + 1), 80.0)
    analytic, z, valid = _read(audio, rates, bands=(32.0,))
    reference = analytic[:, 8000 : 8000 + n].abs().square().mean(-1)
    power = z.abs().square() / reference[:, None, None, None, None]
    q, mask = _products(z, valid)
    q = q / reference[:, None, None, None, None]
    keep = valid[0] & valid[1]
    assert (power[0][keep] - power[1][keep]).abs().mean() > 0.2
    keep_q = _central(mask[0] & mask[1], 2)
    assert (q[0][keep_q] - q[1][keep_q]).abs().mean() > 0.2


def test_colored_floor_can_have_high_short_lag_correlation_without_a_rotor():
    generator = torch.Generator().manual_seed(709)
    white = torch.randn(1, 4 * SR, generator=generator)
    frequencies = torch.fft.rfftfreq(white.shape[-1], d=1 / SR)
    color = (1 + (frequencies / 1000).square()).rsqrt()
    audio = torch.fft.irfft(torch.fft.rfft(white) * color, n=white.shape[-1])
    rates = torch.full((1, 1, audio.shape[-1] // HOP + 1), 80.0)
    analytic, z, valid = _read(audio, rates, bands=(8.0,))
    reference = analytic[:, 8000:-8000].abs().square().mean()
    assert torch.isfinite(z).all()
    assert torch.isfinite(z.abs().square() / reference).all()
    q, mask = _products(z, valid, lag=1)
    keep = _central(mask, 4, lag=1)
    correlation = (
        q[keep].mean().abs()
        / (z[..., 1:][keep].abs().square().mean() * z[..., :-1][keep].abs().square().mean()).sqrt()
    )
    assert correlation > 0.98  # Created by the filter, not a rotor-confidence label.


def test_silence_has_zero_reads_and_finite_waveform_and_rate_gradients():
    audio = torch.zeros(1, SR, requires_grad=True)
    rates = torch.tensor([0.0, 80.0]).reshape(1, 2, 1).requires_grad_()
    analytic, z, valid = _read(audio, rates)
    assert analytic.abs().square().mean() == 0
    assert torch.equal(z, torch.zeros_like(z))
    assert not valid[:, 0].any()  # OFF geometry invalid even without energy masking.
    assert valid[:, 1, :, :, 250].all()  # Geometry is explicitly NOT energy validity.
    loss = z.abs().square().sum()
    d_audio, d_rates = torch.autograd.grad(loss, (audio, rates))
    assert torch.equal(d_audio, torch.zeros_like(audio))
    assert torch.equal(d_rates, torch.zeros_like(rates))


def test_masks_exclude_off_dc_nyquist_and_crop_boundaries():
    n = SR
    rates = torch.tensor([0.0, 0.5, 1.0, 80.0, 799.5]).reshape(1, 5, 1)
    _, z, valid = _read(_tone(n), rates, orders=(0.0, 1.0, 10.0), bands=(8.0, 32.0))
    assert valid.dtype == torch.bool and valid.shape == z.shape
    assert not valid[..., :125].any()
    assert not valid[..., 376:].any()
    assert not valid[:, :2].any()
    assert not valid[:, :, 0].any()
    assert not valid[:, 2, 1].any()  # 1 Hz crosses DC in both bands.
    assert valid[0, 2, 2, 0, 125:376].all()  # 10 +/- 8 Hz is in range.
    assert not valid[0, 2, 2, 1].any()  # 10 +/- 32 Hz crosses DC.
    assert valid[0, 3, 2, :, 125:376].all()
    assert not valid[0, 4, 2].any()  # 7995 +/- 8 Hz crosses Nyquist.


@pytest.mark.parametrize("seconds,frames", [(1, 32), (4, 126), (8, 251)])
def test_envelope_and_prediction_grids_keep_all_boundary_frames(seconds, frames):
    n = seconds * SR
    rates = torch.full((1, 1, n // HOP + 1), 80.0)
    _, z, valid = _read(_tone(n), rates, bands=(32.0,))
    assert z.shape == (1, 1, 1, 1, n // 32 + 1)
    assert z[..., ::16].shape[-1] == frames
    assert not valid[..., 0].any() and not valid[..., -1].any()
    frame_times = torch.arange(frames) * HOP / SR
    env_times = torch.arange(z.shape[-1])[::16] / ENV
    torch.testing.assert_close(frame_times, env_times, atol=0, rtol=0)


def test_non_aligned_padding_preserves_exact_envelope_timestamps():
    # Give the filter a periodic demodulated tone; no crop edges obscure the
    # clock test. Neither the pad nor the original length aligns to stride32.
    n, pad = 16030, 8001
    length = n + 2 * pad  # 32032, exactly divisible by the envelope stride.
    residual = 10 * SR / length
    samples = torch.arange(length, dtype=torch.float64) - pad
    phase = 2 * math.pi * (800 + residual) * samples / SR
    analytic = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)[None]
    z, valid = demodulate_trajectories(
        analytic,
        torch.full((1, 1, 1), 80.0),
        torch.tensor([10.0]),
        n_samples=n,
        pad_samples=pad,
        half_bandwidths=(32.0,),
    )
    time = torch.arange(n // 32 + 1, dtype=torch.float64) / ENV
    expected = torch.polar(torch.ones_like(time), 2 * math.pi * residual * time).to(torch.complex64)
    torch.testing.assert_close(z[0, 0, 0, 0], expected, atol=2e-6, rtol=2e-6)
    assert valid.shape[-1] == n // 32 + 1


def test_doubling_zero_padding_preserves_interior_inferred_rate():
    n = 4 * SR
    rates = torch.full((1, 1, n // HOP + 1), 79.5)
    audio = _tone(n)
    _, a, va = _read(audio, rates, pad=SR // 2)
    _, b, vb = _read(audio, rates, pad=SR)
    assert torch.equal(va, vb)
    for width in range(3):
        ea = _inferred_error(a[..., width, :], va[..., width, :], 4)
        eb = _inferred_error(b[..., width, :], vb[..., width, :], 4)
        assert abs(float(ea) - 0.5) < 0.02
        assert abs(float(eb) - 0.5) < 0.02
        assert abs(float(ea - eb)) < 0.02


def test_harmonic_chunking_preserves_values_and_trajectory_gradients():
    n = SR
    audio = _tone(n) + 0.4 * _tone(n, 1200)
    analytic = analytic_signal_tensor(audio)
    rates = torch.full((1, 1, n // HOP + 1), 79.7, requires_grad=True)
    orders = torch.tensor([10.0, 15.0, 10.5])
    results, gradients = [], []
    for chunk in (1, 3):
        z, valid = demodulate_trajectories(
            analytic,
            rates,
            orders,
            n_samples=n,
            harmonic_chunk=chunk,
        )
        q, mask = _products(z, valid)
        gradients.append(torch.autograd.grad(q.imag[mask].mean(), rates)[0])
        results.append(z)
    torch.testing.assert_close(results[0], results[1], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
def test_complex64_autograd_agrees_with_centered_finite_difference(device):
    n = 2 * SR
    audio = _tone(n).to(device)
    gain = torch.tensor(1.0, device=device, requires_grad=True)
    offset = torch.tensor(0.3, device=device, requires_grad=True)

    def objective(delta, amplitude):
        rates = torch.full((1, 1, n // HOP + 1), 80.0, device=device) + delta
        _, z, valid = _read(audio * amplitude, rates, bands=(32.0,))
        q, mask = _products(z, valid)
        return q.imag[_central(mask, 2)].mean()

    value = objective(offset, gain)
    d_rate, d_gain = torch.autograd.grad(value, (offset, gain))
    epsilon = 0.002
    with torch.no_grad():
        finite_rate = (objective(offset + epsilon, gain) - objective(offset - epsilon, gain)) / (
            2 * epsilon
        )
        finite_gain = (objective(offset, gain + epsilon) - objective(offset, gain - epsilon)) / (
            2 * epsilon
        )
    # For a positive candidate error the residual rotates negatively; this
    # objective remains on a locally decreasing, unwrapped branch.
    assert d_rate < -1 and finite_rate < -1
    torch.testing.assert_close(d_rate, finite_rate, atol=2e-3, rtol=3e-3)
    torch.testing.assert_close(d_gain, finite_gain, atol=2e-3, rtol=3e-3)
    torch.testing.assert_close(d_gain, 2 * value, atol=2e-5, rtol=2e-5)
