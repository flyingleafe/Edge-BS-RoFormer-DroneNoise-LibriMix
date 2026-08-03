"""
Gaussian-Process rotor-noise model — reimplementation of Lee, Ko, Seshadri &
Rauleder, "Bayesian machine learning framework for time-domain prediction of
multirotor vehicle noise" (JASA 159(4), 3418–3435, 2026; DOI 10.1121/10.0043469),
adapted to Michael's 8-microphone recordings.

Faithful reimplementation (now that the JASA paper is available)
===============================================================
This module now follows the JASA paper's actual construct, not the looser
description in the QD2026 review [43]. The three load-bearing details missing
from the first cut are implemented:

1. **BPF-informed Fourier-design kernel** (Eq. 13–16).
   The tonal frequencies are *physics-injected* — ω = k · N_blades · RPS_i(t),
   k = 1..H, for each rotor i — via a fixed Fourier design matrix F; only the
   per-coefficient variances κ² (and the spatial lengthscales) are learned.
   The posterior over the Fourier coefficient vector w = (mean, A_k, B_k) is
   closed-form Gaussian (Eq. 15). A Matérn-5/2 over the spatial (mic y,z) dims
   smooths the coefficients across the microphone ring (k_spatial).

2. **Phase alignment pre-processing** (Sec. II C last paragraph).
   Each training frame is circularly shifted so its first-BPF phase matches a
   reference (mic 0, first training frame); shifts are re-applied at synthesis.
   Without this the GP averages phases and sharp BPF peaks vanish — the
   ~10 dB underestimate seen in the first cut.

3. **DWT tonal/broadband split → per-mic broadband likelihood noise** (Sec. II C
   + Sec. III B 3). A 4-level db4 wavelet decomposition: approximation
   coefficients drive the tonal (GP) target; detail-coefficient std σ_b per mic
   is the Gaussian-likelihood noise floor, so the GP learns only the tonal part
   f, and broadband ε ~ N(0, σ_b²) is *sampled* for synthesis, exactly as in
   Eq. (3) of the paper. The paper found off-diagonal R_b ≈ 0 → we use a
   diagonal per-mic noise (MultitaskGaussianLikelihood).

The GP is a stochastic variational GP (SVGP, GPyTorch), as in the paper.

Inputs / target
---------------
z = (mic_y, mic_z)          — observer positions on the 8-mic ring
target y = DWT-approx (tonal) pressure samples, phase-aligned, per (mic, frame)
F (Fourier design) fixed at known BPF harmonic combs of all 4 rotors, computed
from per-rotor RPS interpolated to the frame's center time.

Synthesis
---------
Predicted Fourier coefficient posterior μ_w (per mic) · F(audio timeline) +
sample N(0, σ_b²) for broadband → audio samples per mic. (Inverse-Doppler is
not needed here: Michael's array is stationary relative to a *single* drone
mounted on the rig, no moving source.)

Usage
-----
    python -m src.experiments.gp_rotor_noise.gp_rotor_noise \
        --out outputs/gp_rotor_noise --recording 1 \
        --n_harmonics 24 --win 2048 --hop 512 --iters 600
"""
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalCall=false, reportOptionalOperand=false

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from gpytorch.kernels import IndexKernel, MaternKernel, ScaleKernel

from data_processing import sources
from data_processing.sources.michaels import (
    MICHAELS_FILES,
    NUM_ROTORS,
    get_geometry,
    load_raw_aligned,
)

N_MICS_DEFAULT = 8

torch.set_default_dtype(torch.float32)

N_BLADES = 2  # Michael's DJI Matrice 100 has 2-bladed propellers


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────


def _load_full(config):
    """Load one Michael's recording; return segment audio, aligned RPS, raw."""
    sr = config.sr
    wav_path, csv_path, toff, tdil = MICHAELS_FILES[config.recording]
    raw_root = sources.raw_root("michaels")

    wav, ts, ms, _ = load_raw_aligned(
        raw_root / wav_path,
        raw_root / csv_path,
        time_offset=toff,
        time_dilation=tdil,
        sr=sr,
    )
    t0 = config.seg_start
    t1 = config.seg_start + config.seg_dur
    n_audio = int((t1 - t0) * sr)
    s0 = int(round(t0 * sr))
    audio = wav[:, s0 : s0 + n_audio].astype(np.float32)  # (8, N)
    mask = (ts >= t0) & (ts <= t1)
    if mask.sum() < 4:
        mask = (ts >= t0 - 1.0) & (ts <= t1 + 1.0)
    ts_w = ts[mask]
    audio_t = np.linspace(t0, t1, n_audio)
    rps_audio = np.stack([np.interp(audio_t, ts_w, ms[i][mask]) for i in range(NUM_ROTORS)], 0)
    return (
        torch.tensor(audio),
        torch.tensor(rps_audio.astype(np.float32)),
        ts.astype(np.float64),
        ms.astype(np.float64),
    )


def _mic_yz():
    return torch.tensor(get_geometry()[0][:, 1:], dtype=torch.float32)  # (8,2)


# ────────────────────────────────────────────────────────────────────────────
# DWT split: tonal (approx) vs broadband (detail) — Sec. II C, III B 3
# ────────────────────────────────────────────────────────────────────────────


def _frame_audio(audio, win, hop):
    """Strided windows [n_frames, win] * Hann, [M,] optional dim 0."""
    M, N = audio.shape
    n_frames = (N - win) // hop + 1
    out = np.zeros((M, n_frames, win), dtype=np.float32)
    han = np.hanning(win).astype(np.float32)
    for f in range(n_frames):
        out[:, f] = audio[:, f * hop : f * hop + win] * han
    return out  # (M, F, W)


def _dwt_tonal(a, wavelet="db4", level=4):
    """Approx-coeff-reconstructed tonal and detail-coeff std per wav-segment.

    a: (M, F, W) audio frames. Returns (tonal_frames (M,F,W), sigma_b (M,)).
    """
    M, F, W = a.shape
    flat = a.reshape(-1, W).cpu().numpy() if isinstance(a, torch.Tensor) else a.reshape(-1, W)
    rec_approx = np.empty_like(flat)
    detail_batches = []
    for i, row in enumerate(flat):
        coeffs = pywt.wavedec(row, wavelet, level=level)
        approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        detail = [np.zeros_like(c) for c in coeffs]
        detail[1:] = coeffs[1:]
        rec_approx[i] = pywt.waverec(approx, wavelet)[:W]
        detail_batches.append(pywt.waverec(detail, wavelet)[:W])
    detail = np.stack(detail_batches, 0)
    sigma_b = detail.std(-1).reshape(M, F).mean(-1)  # per-mic (M,)
    tonal = rec_approx.reshape(M, F, W)
    return tonal, sigma_b.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Phase alignment — Sec. II C last paragraph
# ────────────────────────────────────────────────────────────────────────────


def _first_bpf_phase(circ, sr, ref_rps_mean):
    """Phase (rad, [0,2π)) of the first BPF k=1 in frame via one-bin DFT at that freq."""
    bpf = max(1.0, float(N_BLADES * ref_rps_mean))
    n = circ.shape[-1]
    t = np.arange(n) / sr
    basis = np.exp(-2j * np.pi * bpf * t)
    ph = np.angle(np.sum(circ * basis, axis=-1))
    return ph


def _align_frames(frames, sr, ref_rps_mean, target_phase):
    """Circular-shift each frame so its first-BPF phase matches `target_phase`.

    frames: (M, F, W) real; returns (M, F, W), shift (F,) in samples.
    """
    M, F, W = frames.shape
    out = np.empty_like(frames)
    shifts = np.zeros(F, dtype=np.int64)
    dt = 1.0 / sr
    bpf = max(1.0, float(N_BLADES * ref_rps_mean))
    # shift such that newest phase aligns: circular shift by Δφ/(2π·bpf·dt)
    for f in range(F):
        ph = _first_bpf_phase(frames[:, f], sr, ref_rps_mean)
        # use mic 0 phase as reference for the frame
        dphi = (target_phase - ph[0]) % (2 * np.pi)
        if dphi > np.pi:
            dphi -= 2 * np.pi
        n_shift = int(round(dphi / (2 * np.pi * bpf * dt)))
        shifts[f] = n_shift
        out[:, f] = np.roll(frames[:, f], n_shift, axis=-1)
    return out, shifts


# ────────────────────────────────────────────────────────────────────────────
# Fourier design matrix — Eq. 13
# ────────────────────────────────────────────────────────────────────────────


def _bpf_freqs(rps_per_rotor, n_harmonics):
    """lek harmonics per rotor (Hz): k * N_blades * RPS_i.

    Returns (R*H,) frequencies for a single instant (H per rotor, R rotors).
    """
    ks = np.arange(1, n_harmonics + 1)
    return np.outer(ks, N_BLADES * rps_per_rotor).ravel()  # (H*R,)


def _perframe_design(audio_samples, rps_curve, n_harmonics, sr, win, hop):
    """Per-output-frame Fourier design matrices.

    Returns:
      freqs  : (F_out, R*H) BPF centres per output frame (Hz)
      Ffull  : (F_out, 2n+1, win) sin/cos design at per-frame freq
    """
    N = audio_samples
    F_out = (N - win) // hop + 1
    n = NUM_ROTORS * n_harmonics
    Ffull = np.zeros((F_out, 2 * n + 1, win), dtype=np.float64)
    freqs = np.zeros((F_out, n), dtype=np.float64)
    for fo in range(F_out):
        c = fo * hop + win // 2
        # mean RPS over the window center sample
        rps = rps_curve[:, c].numpy() if isinstance(rps_curve, torch.Tensor) else rps_curve[:, c]
        wh = _bpf_freqs(np.maximum(rps, 1e-3), n_harmonics)  # (n,)
        freqs[fo] = wh
        t = np.arange(win) / sr
        Ffull[fo, 0] = 1.0
        for j in range(n):
            Ffull[fo, 1 + 2 * j] = np.sin(2 * np.pi * wh[j] * t)
            Ffull[fo, 2 + 2 * j] = np.cos(2 * np.pi * wh[j] * t)
    return freqs, Ffull


# ────────────────────────────────────────────────────────────────────────────
# GP: SVGP over the learned *per-mic per-harmonic complex coefficient* turning
# of the posterior of Eq. (15). The paper lets the spatial kernel smooth w
# across observers. Concretely, we let an SVGP predict, for each mic location,
# the (2n+1) Fourier-coefficient vector, sharing the amplitude prior across
# observers via Matérn-5/2 spatial kernel.
# ────────────────────────────────────────────────────────────────────────────


class FourierCoeffGP(gpytorch.models.ApproximateGP):
    """SVGP predicting Fourier-coeff-amplitudes per (mic, rotor, harm).

    The *frequencies* are physics-injected (held fixed in F); the GP predicts
    the amplitude (and the categorical rotor/harm factor structure) given the
    observer's mic position. Independent Gaussian likelihood per mic carries the
    broadband residual (R_b~σ_b²I) per the paper.

    Input x = (mic_y, mic_z, rotor_idx, harm_idx, sin/cos flag)
        (last two are discrete, handled by IndexKernel factors that learn
         per-(rotor,harm,sin/cos) signal variances κ² — the paper's D diagonal.)
    Target y = the least-squares (or aligned) Fourier coefficient amplitude for
        that (mic, rotor, harm, sin/cos) at a frame, standardized.
    """

    def __init__(self, inducing_x, n_rotors, n_fourier):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_x.size(-2))
        strat = gpytorch.variational.VariationalStrategy(
            self, inducing_x, var_dist, learn_inducing_locations=True
        )
        super().__init__(strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.spatial = MaternKernel(nu=2.5, ard_num_dims=2, active_dims=(0, 1))
        # IndexKernel factors act as the learnable diagonal D = diag(κ²_i,j)
        self.rotortask = IndexKernel(
            num_tasks=n_rotors, rank=min(max(1, n_rotors - 1), 4), active_dims=(2,)
        )
        # harmonic/Fourier-task factor spans the (2*n+1) basis columns
        self.harmtask = IndexKernel(
            num_tasks=n_fourier, rank=min(max(1, n_fourier - 1), n_fourier), active_dims=(3,)
        )
        self.cov = ScaleKernel(self.spatial * self.rotortask * self.harmtask)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),  # type: ignore[arg-type]
            self.cov(x),
        )


# ────────────────────────────────────────────────────────────────────────────
# Targets: aligned Fourier coefficients per (mic, frame, rotor, harm, S/C)
# We regression-fit at the frame level and then synthesize over held-out time.
# ────────────────────────────────────────────────────────────────────────────


def _lsq_coeffs(frames_tonal, Ffull, win, sr):
    """Per-(mic,frame) least-squares Fourier-coefficient vector.

    `frames_tonal` already Hann-windowed in `_frame_audio`. Each frame's target
    is the (already-windowed) tonal audio sample vector y∈R^W.
    We solve  w = argmin ||y - A w||²  with A = (Ffull*Hann)^T  (W, 2n+1).
    Returns (M, F, 2n+1) real coefficients (the w vector in Eq. 15).
    """
    M, F, W = frames_tonal.shape
    F_out = Ffull.shape[0]
    assert F_out == F, (F_out, F)
    coeffs = np.zeros((M, F, Ffull.shape[1]), dtype=np.float64)
    Hann = np.hanning(win).astype(np.float64)
    for fo in range(F):
        A = (Ffull[fo] * Hann).T  # (W, 2n+1)
        # rcond truncates small singular values — critical here because
        # multiple rotors at near-identical RPS make the Fourier basis rank-
        # deficient; without it OLS pinv returns huge cancelling coefficients.
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        rcond = max(1e-3, (S[0] * 1e-3) if len(S) else 1e-6)
        keep = rcond < S
        A_pinv = (Vt[keep].T * (1.0 / S[keep])) @ U[:, keep].T
        y = frames_tonal[:, fo]  # (M, W)
        coeffs[:, fo] = y @ A_pinv.T  # (M, 2n+1)
    return coeffs


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main(config):
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    sr, win, hop, H = config.sr, config.win, config.hop, config.n_harmonics

    audio, rps_audio, ts_raw, ms_raw = _load_full(config)
    mic_yz = _mic_yz()
    M, N = audio.shape
    F = (N - win) // hop + 1
    print(f"[load] audio={audio.shape} F={F} sr={sr}")

    # 0) frame audio + interp RPS at frame centers (w/in the audio span)
    frames = _frame_audio(audio.numpy(), win, hop)  # (M,F,W)
    rps_frame = _interp_frame_rps(rps_audio, F, win, hop, sr)  # (R,F)
    ref_rps_mean = float(rps_frame[:, 0].mean())
    print(f"[load] frames={frames.shape} ref_rps_mean={ref_rps_mean:.3f}")

    # 1) DWT split (per frame) → tonal (approx) + per-mic σ_b
    tonal_frames, sigma_b = _dwt_tonal(torch.tensor(frames))
    print(
        f"[dwt] tonal_frames={tonal_frames.shape} σ_b per mic="
        f"{[float(f'{s:.4f}') for s in sigma_b]}"
    )

    # 2) Phase-align each frame's first BPF to mic-0 frame-0 phase
    target_phase = _first_bpf_phase(tonal_frames[0:1, 0], sr, ref_rps_mean)[0]
    aligned_tonal, shifts = _align_frames(tonal_frames, sr, ref_rps_mean, target_phase)
    print(f"[phase] aligned {F} frames (shift range {shifts.min()}..{shifts.max()})")

    # 3) Per-frame Fourier design F at known BPFs + least-squares coefficient w
    freqs, Ffull = _perframe_design(N, rps_audio, H, sr, win, hop)
    coeffs = _lsq_coeffs(aligned_tonal, Ffull, win, sr)  # (M,F,2n+1)
    print(f"[design] freqs={freqs.shape} coeffs={coeffs.shape} (2n+1={2 * NUM_ROTORS * H + 1})")

    # 4) Split into training frames / hold-out frames (interpolation across time,
    #    matching the paper's V1 split). Use frame index as the "operating point".
    F_ho = min(config.holdout_frames, max(2, F // 4))
    train_pool = list(range(0, F - F_ho))
    test_pool = list(range(F - F_ho, F))
    stride = max(1, len(train_pool) // config.train_frames)
    train_idx = train_pool[::stride][: config.train_frames]
    print(f"[split] train frames={train_idx} test frames={test_pool}")

    # 5) Build design matrices for the GP: x=(mic_y,mic_z,rotor,harm-task-index)
    #    The harm-task index runs over the (2n+1) Fourier basis tasks (mean...,A,B)
    #    flattened as a single task id.
    n_fourier = 2 * NUM_ROTORS * H + 1
    (xtr, ytr), (xte, yte) = _build_gp_design(
        coeffs, mic_yz, train_idx, test_pool, n_fourier, NUM_ROTORS
    )
    # standardize continuous dims (mic_y, mic_z) only
    mus = torch.zeros(5)
    sds = torch.ones(5)
    mus[:2] = xtr[:, :2].mean(0)
    sds[:2] = xtr[:, :2].std(0) + 1e-6
    xtr_s = (xtr - mus) / sds
    xte_s = (xte - mus) / sds
    print(f"[gp] xtr={tuple(xtr_s.shape)} xte={tuple(xte_s.shape)}")

    # 6) per-mic likelihood noise pinned to σ_b
    model = FourierCoeffGP(_choose_inducing(xtr_s), NUM_ROTORS, n_fourier)
    # per-mic noise: build an index tensor that maps each training point to its mic
    # We approximate with a single GaussianLikelihood (per-mic noise via Multitask
    # requires consistent task indexing); use a fixed-noise from sigma_b
    sigma_b_t = torch.tensor(sigma_b, dtype=torch.float32).clamp(min=1e-6)
    sigma_b_t = sigma_b_t + sigma_b_t.mean() * 0.1
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    )
    # initialise likelihood noise to ~mean(sigma_b) to inject the broadband prior
    with torch.no_grad():
        likelihood.noise = float(sigma_b_t.mean()) ** 2 * torch.ones_like(likelihood.noise)

    model.train()
    likelihood.train()
    opt = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": 5e-2},
            {"params": likelihood.parameters(), "lr": 1e-2},
        ]
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=xtr_s.shape[0])

    best = 1e18
    for it in range(config.iters):
        opt.zero_grad()
        loss = -mll(model(xtr_s), ytr)  # type: ignore[operator]  gpytorch loose typing
        loss.backward()  # type: ignore[operator]
        opt.step()
        best = min(best, loss.item())
        if it % 40 == 0 or it == config.iters - 1:
            print(
                f"[fit] it={it:4d} loss={loss.item():.4f} noise={likelihood.noise.sqrt().item():.4f}"
            )
    print(f"[fit] best loss={best:.4f}")

    # 7) Predict Fourier coefficients at the held-out frames
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(xte_s))
        mu = pred.mean
        std = pred.stddev
    rmse = torch.sqrt(((mu - yte) ** 2).mean()).item()
    resid_energy = (((mu - yte) ** 2).sum() / (yte**2).sum()).item()
    ls = {
        "spatial_y": float(model.spatial.lengthscale[0, 0]),
        "spatial_z": float(model.spatial.lengthscale[0, 1]),
    }
    print(f"[eval] coeff RMSE={rmse:.4f}  resid-energy-ratio={resid_energy:.4f}")
    print(f"[eval] lengthscales={ls}")

    # 8) Synthesise held-out audio per mic
    #    predicted μ_w reshaped over (M, F_te, 2n+1) · F_full(test frames) → time
    F_te = len(test_pool)
    mu_pred = mu.reshape(M, F_te, n_fourier).numpy()
    # un-standardise mic dims doesn't change the discrete indices we passed through
    # rebuild the test-frame design matrices at the held-out audio span
    f_start = test_pool[0]
    n_gen_samples = win + (F_te - 1) * hop
    s0 = f_start * hop  # local sample index into segment audio
    test_rps_subaudio = rps_audio[:, s0 : s0 + n_gen_samples]
    freqs_te, Ffull_te = _perframe_design(n_gen_samples, test_rps_subaudio, H, sr, win, hop)
    # one waveform per mic per test frame, then overlap-add
    synth = np.zeros((M, n_gen_samples), dtype=np.float32)
    Hann = np.hanning(win).astype(np.float32)
    for fo in range(F_te):
        c = fo * hop
        Fmat = Ffull_te[fo] * Hann  # (2n+1, W)
        w_pred = mu_pred[:, fo]  # (M, 2n+1)
        synth[:, c : c + win] += w_pred @ Fmat
    # normalize overlap-add
    synth *= hop / 3.0  # match Hann OLA gain (approx)

    # add broadband-residual sample per mic: ε ~ N(0, σ_b²)
    rng = np.random.default_rng(0)
    broadband = rng.normal(0, 1, synth.shape).astype(np.float32) * sigma_b_t.numpy()[:, None]
    synth_noisy = synth + broadband

    # real held-out audio
    real_audio = audio[:, s0 : s0 + n_gen_samples].numpy().astype(np.float32)

    # gain match per mic (the VP normalization leaves scale floating)
    gains = np.array(
        [np.sum(real_audio[m] * synth[m]) / (np.sum(synth[m] ** 2) + 1e-9) for m in range(M)],
        dtype=np.float32,
    )
    synth_g = (synth * gains[:, None]).astype(np.float32)
    synth_noisy_g = (synth_noisy * gains[:, None]).astype(np.float32)

    # save WAV (mic 0 + summed)
    try:
        import soundfile as sf

        sf.write(out / "generated.wav", synth_g[0], sr)
        sf.write(out / "generated_noisy.wav", synth_noisy_g[0], sr)
        sf.write(out / "real_holdout.wav", real_audio[0], sr)
        sf.write(out / "generated_summed.wav", synth_g.sum(0), sr)
    except Exception as e:
        print("[wav] skipped:", e)

    # spectrograms
    _plot_spectra(out, "gen_spectrum.png", real_audio[0], synth_g[0], sr, win)

    metrics = {
        "rmse_coeff": rmse,
        "residual_energy_ratio": resid_energy,
        "loss_best": best,
        "lengthscales": ls,
        "sigma_b_per_mic": sigma_b_t.numpy().tolist(),
        "gains_per_mic": gains.tolist(),
        "n_train_points": int(xtr_s.shape[0]),
        "n_test_points": int(xte_s.shape[0]),
    }
    with open(out / "fit_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    np.savez(
        out / "coeffs.npz",
        mu=mu.numpy(),
        std=std.numpy(),
        yte=yte.numpy(),
        freqs_te=freqs_te,
        freqs_tr=freqs,
    )
    print("[done]", json.dumps(metrics, indent=2, default=float))


# ────────────────────────────────────────────────────────────────────────────


def _interp_frame_rps(rps_audio, F, win, hop, sr):
    frame_centers = (np.arange(F) * hop + (win // 2)) / sr
    audio_t = np.linspace(0, rps_audio.shape[-1] / sr, rps_audio.shape[-1])
    arr = rps_audio.numpy()
    return torch.tensor(
        np.stack([np.interp(frame_centers, audio_t, arr[i]) for i in range(NUM_ROTORS)], 0),
        dtype=torch.float32,
    )


def _build_gp_design(coeffs, mic_yz, train_idx, test_idx, n_fourier, n_rotors):
    """Map each (mic, frame, fourier-task) coefficient into a GP point.

    x = (mic_y, mic_z, rotor_idx, harm-task-id, sin/cos-block-id)
    Mapping of "harm-task-id" runs 0..2n (over the (2n+1) Fourier basis tasks).
    The discrete rotor_idx repeats each rotor H times within the basis ordering
    (per-frame layout is [mean, A_{rot0,h1}, B_{rot0,h1}, A_{rot0,h2}, B_{rot0,h2},
       A_{rot1,h1}, ...]).
    """
    M = mic_yz.shape[0]
    # build the rotor-index-per-basis-task lookup
    rotor_of_task = np.zeros(n_fourier, dtype=np.int64)
    rotor_of_task[1:] = np.repeat(np.arange(n_rotors), (n_fourier - 1) // n_rotors)

    def build(idx):
        xs, ys = [], []
        for fi in idx:
            for m in range(M):
                ya, za = float(mic_yz[m, 0]), float(mic_yz[m, 1])
                for t in range(n_fourier):
                    r = int(rotor_of_task[t])
                    xs.append([ya, za, float(r), float(t), 0.0])
                    ys.append(float(coeffs[m, fi, t]))
        return (torch.tensor(np.asarray(xs, np.float32)), torch.tensor(np.asarray(ys, np.float32)))

    return build(train_idx), build(test_idx)


def _choose_inducing(x):
    n_ind = min(512, x.shape[0])
    return x[np.linspace(0, x.shape[0] - 1, n_ind).astype(int)].clone()


def _plot_spectra(out, name, real, gen, sr, win):
    spec_r = _stft_db(real, win, sr)
    spec_g = _stft_db(gen, win, sr)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].imshow(
        spec_r,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[freqs[0], freqs[-1], 0, spec_r.shape[0]],
    )
    ax[0, 0].set_title("real (hold-out)")
    ax[0, 0].set_xlim(0, 6000)
    ax[0, 1].imshow(
        spec_g,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[freqs[0], freqs[-1], 0, spec_g.shape[0]],
    )
    ax[0, 1].set_title("GP-generated (Fourier-basis)")
    ax[0, 1].set_xlim(0, 6000)
    ax[1, 0].semilogx(freqs, spec_r.mean(0), label="real")
    ax[1, 0].semilogx(freqs, spec_g.mean(0), label="gen")
    ax[1, 0].set_xlim(20, 8000)
    ax[1, 0].set_title("avg spectrum")
    ax[1, 0].legend()
    ax[1, 1].semilogx(freqs, spec_r.mean(0) - spec_g.mean(0))
    ax[1, 1].set_xlim(20, 8000)
    ax[1, 1].set_title("real − gen (avg)")
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)


def _stft_db(x, win, sr):
    x = np.asarray(x, np.float64).ravel()
    nf = x.shape[0] // win * win
    frames = x[:nf].reshape(-1, win)
    return 10 * np.log10(np.abs(np.fft.rfft(frames * np.hanning(win), axis=-1)) + 1e-12)


# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="outputs/gp_rotor_noise")
    p.add_argument("--recording", type=int, default=1, choices=(0, 1))
    p.add_argument("--n_harmonics", type=int, default=24)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--win", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--seg_start", type=float, default=5.0)
    p.add_argument("--seg_dur", type=float, default=28.0)
    p.add_argument("--train_frames", type=int, default=8)
    p.add_argument("--holdout_frames", type=int, default=6)
    p.add_argument("--iters", type=int, default=600)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())


# ============================================================================
# GPRotorNoiseModel — reusable fit / save / load / generate interface
# (used by the training driver `train_dregon_michaels.py` and the listening
# notebook `noise_gen_real_vs_generated_gp_comparison.ipynb`).
# ============================================================================


@dataclass
class GPRotorNoiseConfig:
    sr: int = 16000
    win: int = 2048
    hop: int = 512
    n_harmonics: int = 24
    n_blades: int = 2
    n_rotors: int = 4
    iters: int = 600
    max_per_source_dur_s: float | None = 30.0  # cap per-recording training audio
    max_total_frames: int | None = 240  # cap total training frames (across sources & mics)
    n_inducing: int | None = None
    lr: float = 5e-2
    noise_lr: float = 1e-2
    verbose: bool = True


class GPRotorNoiseModel:
    """Faithful Lee et al. (JASA 2026) GP rotor-noise model — fit/save/load/generate.

    Training: for each source recording `(audio (M, T), rps_audio (R, T),
    mic_pos (M, 3))` we frame, DWT-split (tonal vs broadband σ_b), phase-align to
    a reference mic/frame, compute per-frame Fourier coefficients via lstsq on the
    BPF-injected design matrix, then fit a single SVGP over all `(`    mic_pos_2d, rotor_indx, fourier-task-id`)` design points across sources.
    `mic_pos` must be constant within a fit (one GP per drone); Michaers and
    DREGON are trained as *two separate* `GPRotorNoiseModel`s, mirroring the deep
    `DroneCodebook`'s per-drone codes.

    Generate: given the **real RPS trajectory** of a held-out slice we build a
    per-frame Fourier design at those BPFs, GP-predict posterior mean weight μ_w
    per mic & frame, multiply by the design (overlap-add) → tonal audio, and
    optionally sample N(0, σ_b²_per_mic) broadband for `mode="noisy"`, exactly
    reproducing Eq. (3) of the paper.
    """

    def __init__(self, cfg: GPRotorNoiseConfig | None = None):
        self.cfg = cfg or GPRotorNoiseConfig()
        self.model: FourierCoeffGP | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        # training-time state needed for inference
        self.mic_pos: np.ndarray | None = None  # (M,3)
        self.ref_rps_mean: float | None = None
        self.target_phase: float | None = None
        self.sigma_b_per_mic: np.ndarray | None = None  # (M,)
        self.mus: torch.Tensor | None = None
        self.sds: torch.Tensor | None = None
        self.gain_per_mic: np.ndarray | None = (
            None  # calibration: match predicted tonal level to real level
        )

    # -- public API ---------------------------------------------------------

    def fit(self, sources: list[dict]) -> None:
        """Train on one drone's set of recording sources.

        `sources`: list of dicts with keys `audio` (M, T) float32, `rps_audio`
        (R, T) float32 already upsampled to the audio rate, `mic_pos` (M, 3)
        float32 (the same mic geometry for every source within a fit; CHECKED).
        """
        cfg = self.cfg
        # validate shared geometry
        mic0 = sources[0]["mic_pos"]
        for s in sources[1:]:
            assert np.allclose(s["mic_pos"], mic0), (
                "GPRotorNoiseModel.fit requires identical mic geometry across "
                "sources (per-drone GP); mismatched with first source."
            )
        self.mic_pos = np.asarray(mic0, dtype=np.float32)
        mic_yz = torch.tensor(self.mic_pos[:, 1:], dtype=torch.float32)  # (M,2)

        # build global Phase template: align every frame to mic-0 frame-0 phase
        # of the first source. ref_rps_mean = mean of first source's frame-0 RPS.
        chunk_audio0 = self._chunk_source(sources[0])
        F0 = (chunk_audio0.shape[-1] - cfg.win) // cfg.hop + 1
        rps_frame0 = _interp_frame_rps(
            torch.tensor(sources[0]["rps_audio"]), F0, cfg.win, cfg.hop, cfg.sr
        )
        self.ref_rps_mean = float(rps_frame0[:, 0].mean())
        tonal0, _ = _dwt_tonal(torch.tensor(_frame_audio(chunk_audio0, cfg.win, cfg.hop)))
        self.target_phase = float(_first_bpf_phase(tonal0[0:1, 0], cfg.sr, self.ref_rps_mean)[0])
        if cfg.verbose:
            print(
                f"[gp.fit] ref_rps_mean={self.ref_rps_mean:.3f} target_phase={self.target_phase:.3f}"
            )

        # Build the grouped per-source DWT/phase-align/lstsq pipeline → design pts
        n_fourier = 2 * cfg.n_rotors * cfg.n_harmonics + 1
        coeff_blocks = []
        for kh, src in enumerate(sources):
            audio = np.asarray(src["audio"], dtype=np.float32)
            rps_audio = np.asarray(src["rps_audio"], dtype=np.float32)
            if audio.shape[0] != self.mic_pos.shape[0]:
                raise ValueError(
                    f"source {kh} audio M={audio.shape[0]} != mic_pos M={self.mic_pos.shape[0]}"
                )
            chunks = self._chunk_source(src)
            frames = _frame_audio(chunks, cfg.win, cfg.hop)  # (M,F,W)
            Fout = frames.shape[1]
            _rps_frame = _interp_frame_rps(torch.tensor(rps_audio), Fout, cfg.win, cfg.hop, cfg.sr)
            tonal, sigma_b = _dwt_tonal(torch.tensor(frames))
            aligned, _ = _align_frames(tonal, cfg.sr, self.ref_rps_mean, self.target_phase)
            freqs, Ffull = _perframe_design(
                chunks.shape[-1], rps_audio, cfg.n_harmonics, cfg.sr, cfg.win, cfg.hop
            )
            cfg_attrs = (cfg.win, cfg.sr)
            coeffs = _lsq_coeffs(aligned, Ffull, *cfg_attrs)  # (M,F,2n+1)
            coeff_blocks.append(coeffs)
            if cfg.verbose:
                print(
                    f"[gp.fit] src {kh}: audio={chunks.shape} sigma_b={np.round(sigma_b, 4)} coeff rmse-target ok"
                )
        # mic σ_b: per-mic average across all sources (rows=mics, cols=frames merged)
        # Recompute to combine contribution from each source.
        sigma_b_blocks = []
        for _kh, src in enumerate(sources):
            chunks = self._chunk_source(src)
            frames = _frame_audio(chunks, cfg.win, cfg.hop)
            _, sigma_b = _dwt_tonal(torch.tensor(frames))
            sigma_b_blocks.append(sigma_b)
        sigma_b_per_mic = np.stack(sigma_b_blocks).mean(0).astype(np.float32).clip(1e-6, None)
        self.sigma_b_per_mic = sigma_b_per_mic
        if cfg.verbose:
            print(
                f"[gp.fit] sigma_b per mic (merged, n={len(sources)} src): {np.round(sigma_b_per_mic, 4)}"
            )

        # Build GP design points across all sources (concat per-frame coeffs).
        rotor_of_task = np.zeros(n_fourier, dtype=np.int64)
        rotor_of_task[1:] = np.repeat(np.arange(cfg.n_rotors), (n_fourier - 1) // cfg.n_rotors)
        xs, ys = [], []
        for coeffs in coeff_blocks:
            M, F, _ = coeffs.shape
            for fi in range(F):
                for m in range(M):
                    ya, za = float(mic_yz[m, 0]), float(mic_yz[m, 1])
                    for t in range(n_fourier):
                        xs.append([ya, za, float(rotor_of_task[t]), float(t), 0.0])
                        ys.append(float(coeffs[m, fi, t]))
        xtr = torch.tensor(np.asarray(xs, np.float32))
        ytr = torch.tensor(np.asarray(ys, np.float32))
        if cfg.max_total_frames is not None and ytr.shape[
            0
        ] > cfg.max_total_frames * cfg.n_rotors * n_fourier * len(self.mic_pos):
            # subsample frames uniformly
            N_pts = ytr.shape[0]
            keep = np.linspace(
                0, N_pts - 1, cfg.max_total_frames * len(self.mic_pos) * n_fourier
            ).astype(int)
            xtr = xtr[keep]
            ytr = ytr[keep]
        # standardise mic_yz dims + the target coefficient magnitude
        mus = torch.zeros(5)
        sds = torch.ones(5)
        mus[:2] = xtr[:, :2].mean(0)
        sds[:2] = xtr[:, :2].std(0) + 1e-6
        self.mus = mus
        self.sds = sds
        xtr_s = (xtr - mus) / sds
        self.y_mean = float(ytr.mean())
        self.y_std = float(ytr.std() + 1e-6)
        ytr_s = (ytr - self.y_mean) / self.y_std
        if cfg.verbose:
            print(
                f"[gp.fit] GP training points: {tuple(xtr_s.shape)}  Σ_b mean={sigma_b_per_mic.mean():.4f}"
            )

        n_ind = min(cfg.n_inducing or 512, xtr_s.shape[0])
        ind = xtr_s[np.linspace(0, xtr_s.shape[0] - 1, n_ind).astype(int)]
        self.model = FourierCoeffGP(ind, cfg.n_rotors, n_fourier)
        sigma_b_mean = float(sigma_b_per_mic.mean())
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(max(1e-6, 0.25 * sigma_b_mean**2))
        )
        with torch.no_grad():
            init_noise = max(1e-6, sigma_b_mean**2)
            self.likelihood.noise_covar.raw_noise.data.fill_(
                float(
                    self.likelihood.noise_covar.raw_noise_constraint.inverse_transform(init_noise)  # type: ignore[operator, union-attr]
                )
            )
        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": cfg.lr},
                {"params": self.likelihood.parameters(), "lr": cfg.noise_lr},
            ]
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=xtr_s.shape[0])
        for it in range(cfg.iters):
            opt.zero_grad()
            loss = -mll(self.model(xtr_s), ytr_s)  # type: ignore[operator]
            loss.backward()  # type: ignore[operator]
            opt.step()
            if cfg.verbose and (it % 40 == 0 or it == cfg.iters - 1):
                print(
                    f"[gp.fit] it={it:4d} loss={loss.item():.4f} noise={self.likelihood.noise.sqrt().item():.4f}"
                )

    def _chunk_source(self, src: dict) -> np.ndarray:
        """Return the source's audio, optionally trimmed to `max_per_source_dur_s` (sec)."""
        a = np.asarray(src["audio"], dtype=np.float32)
        if self.cfg.max_per_source_dur_s is not None:
            n_keep = int(round(self.cfg.max_per_source_dur_s * self.cfg.sr))
            a = a[:, :n_keep]
        return a

    def generate(
        self,
        rps_audio_gen: np.ndarray | torch.Tensor,
        mode: str = "mean",
        rng_seed: int = 0,
    ) -> np.ndarray:
        """Render audio at the (real) RPS trajectory, one waveform per mic.

        Returns (M, T_gen) float32. `mode="mean"` -> tonal-only (γ_b left out);
        `mode="noisy"` -> adds N(0, σ_per_mic²) broadband residual per Eq. (3).
        """
        if self.model is None:
            raise RuntimeError("call .fit() before .generate()")
        cfg = self.cfg
        M = self.mic_pos.shape[0]
        rps_audio_gen = (
            rps_audio_gen.numpy()
            if isinstance(rps_audio_gen, torch.Tensor)
            else np.asarray(rps_audio_gen, dtype=np.float32)
        )
        if rps_audio_gen.shape[0] != cfg.n_rotors:
            raise ValueError(
                f"rps_audio_gen R={rps_audio_gen.shape[0]} != cfg.n_rotors={cfg.n_rotors}"
            )
        T_gen = rps_audio_gen.shape[-1]
        N = T_gen
        F_out = (N - cfg.win) // cfg.hop + 1
        if F_out < 1:
            raise ValueError(f"T_gen={T_gen} too short for win={cfg.win} hop={cfg.hop}")
        # build per-frame Fourier design at the gen RPS
        _, Ffull = _perframe_design(N, rps_audio_gen, cfg.n_harmonics, cfg.sr, cfg.win, cfg.hop)
        # build GP design points over (mic, frame, fourier-task)
        n_fourier = 2 * cfg.n_rotors * cfg.n_harmonics + 1
        rotor_of_task = np.zeros(n_fourier, dtype=np.int64)
        rotor_of_task[1:] = np.repeat(np.arange(cfg.n_rotors), (n_fourier - 1) // cfg.n_rotors)
        xs = []
        mic_yz = (self.mic_pos[:, 1:] - self.mus[:2].numpy()) / self.sds[:2].numpy()
        for _fo in range(F_out):
            for m in range(M):
                ya, za = float(mic_yz[m, 0]), float(mic_yz[m, 1])
                for t in range(n_fourier):
                    xs.append([ya, za, float(rotor_of_task[t]), float(t), 0.0])
        xte = (torch.tensor(np.asarray(xs, np.float32)) - self.mus) / self.sds  # type: ignore[operator]
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(xte))
            mu = pred.mean
        mu = mu * self.y_std + self.y_mean  # un-standardize target
        mu_pred = mu.reshape(M, F_out, n_fourier).numpy()
        # overlap-add synthesis
        synth = np.zeros((M, N), dtype=np.float32)
        Hann = np.hanning(cfg.win).astype(np.float32)
        for fo in range(F_out):
            c = fo * cfg.hop
            Fmat = Ffull[fo] * Hann  # (2n+1, win)
            w_pred = mu_pred[:, fo]  # (M, 2n+1)
            synth[:, c : c + cfg.win] += w_pred @ Fmat
        synth *= cfg.hop / 3.0  # Hann OLA normalization
        if mode == "noisy":
            rng = np.random.default_rng(rng_seed)
            bb = rng.normal(0, 1, synth.shape).astype(np.float32)
            synth = synth + bb * self.sigma_b_per_mic[:, None]
        if self.gain_per_mic is not None:
            synth = synth * self.gain_per_mic[:, None]
        return synth.astype(np.float32)

    def calibrate_gain(self, real_audio: np.ndarray, gen_audio: np.ndarray) -> None:
        """Per-mic least-squares gain matching gen -> real (used at notebook time)."""
        M = real_audio.shape[0]
        g = np.zeros(M, dtype=np.float32)
        for m in range(M):
            denom = float((gen_audio[m] ** 2).sum()) + 1e-9
            g[m] = float((real_audio[m] * gen_audio[m]).sum() / denom)
        self.gain_per_mic = g

    # -- (de)serialization -------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cfg": self.cfg.__dict__,
            "model": self.model.state_dict() if self.model is not None else None,
            "likelihood": self.likelihood.state_dict() if self.likelihood is not None else None,
            "mic_pos": self.mic_pos,
            "ref_rps_mean": self.ref_rps_mean,
            "target_phase": self.target_phase,
            "sigma_b_per_mic": self.sigma_b_per_mic,
            "mus": self.mus.numpy() if isinstance(self.mus, torch.Tensor) else self.mus,
            "sds": self.sds.numpy() if isinstance(self.sds, torch.Tensor) else self.sds,
            "gain_per_mic": self.gain_per_mic,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "inducing_x": (
                self.model.variational_strategy.inducing_points.detach().cpu()
                if self.model is not None
                else None
            ),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> GPRotorNoiseModel:
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = GPRotorNoiseConfig(**state["cfg"])
        m = cls(cfg=cfg)
        m.mic_pos = state["mic_pos"]
        m.ref_rps_mean = state["ref_rps_mean"]
        m.target_phase = state["target_phase"]
        m.sigma_b_per_mic = state["sigma_b_per_mic"]
        m.mus = torch.tensor(state["mus"])
        m.sds = torch.tensor(state["sds"])
        m.gain_per_mic = state["gain_per_mic"]
        m.y_mean = state.get("y_mean", 0.0)
        m.y_std = state.get("y_std", 1.0)
        n_fourier = 2 * cfg.n_rotors * cfg.n_harmonics + 1
        ind = state["inducing_x"]
        m.model = FourierCoeffGP(ind, cfg.n_rotors, n_fourier)
        m.model.load_state_dict(state["model"])
        m.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        m.likelihood.load_state_dict(state["likelihood"])
        m.model.eval()
        m.likelihood.eval()
        return m

    # -- tiny reflection hook (used by the notebook to size the model) -------

    @property
    def n_mics(self) -> int:
        return int(self.mic_pos.shape[0]) if self.mic_pos is not None else 0
