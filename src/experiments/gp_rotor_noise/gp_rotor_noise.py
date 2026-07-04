"""
Gaussian-Process rotor-noise model — reimplementation of Ko & Kim,
"GP-Based Time-Domain Modeling for Multi-Rotor Noise Prediction"
(Quiet Drones 2026, paper 43), adapted to Michael's recordings.

Faithful-construct, tractable-granularity adaptation
=====================================================
Paper [43] is a *review*; its full kernel detail lives in two lineage papers we
don't have (Lee et al. JASA 2026; M. Kim & Ko INTER-NOISE 2026), and a *literal*
16-kHz sample-rate time-domain GP is O(N^3) per segment — intractable without
the SKI / KeOps machinery they don't publish. We instead implement the paper's
*construct* at a granularity the project already has machinery for: a GP over
the per-frame **variable-phasor (VP) harmonic coefficients** of the recording,
synthesised back to a waveform via `inverse_harmonic_VP_transform`.

The kernel faithfully mirrors Eq. 1 of [43]:

    k(z,z') = k_spatial((y,z),(y',z')) ⊙ k_tonal((RPS,t),(RPS',t'))

* `k_spatial` = Matérn-3/2 over the 8-microphone ring positions (y,z) [m].
* `k_tonal`  = RBF with ARD over instantaneous **per-rotor RPS** (rev/s) —
  the paper's V_inf / throttle "operating condition", here refined to per-rotor
  speed (the actual driver of that rotor's harmonic comb).
* Discrete `rotor_idx` and `harm_idx` factors via learnable `IndexKernel`s give
  the "band-based tonal kernel" variant of [43] (each harmonic represented over
  a learned similarity, not a delta line).
* **Broadband residual is modelled as the GP likelihood noise**, not a separate
  structured covariance term — exactly as [43] states ("broadband variability
  was not modeled using an additional structured covariance term ... incorporated
  through the likelihood model as residual uncertainty").

Target  y = log|V[mic, rotor, harm, frame]|    (complex VP coefficient magnitude)
Inputs z = (mic_y, mic_z, rps_i, rotor_idx, harm_idx)

Synthesis: predicted posterior mean `exp(mu)` × empirical (circular-mean) complex
phase template → `inverse_harmonic_VP_transform` → audio. A second, noisier
realisation adds `std`-scaled broadband residual samples to `log|V|` before
exp, concretely realising the "broadband = likelihood residual" decomposition.

Outputs (under --out): gen_spectrum.png, generated.wav, generated_noisy.wav,
real_holdout.wav, fit_metrics.json, coeffs.npz.

Usage
-----
    python -m src.experiments.gp_rotor_noise.gp_rotor_noise \
        --out outputs/gp_rotor_noise --recording 1 \
        --n_harmonics 12 --train_frames 6 --holdout_frames 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.kernels import IndexKernel, MaternKernel, RBFKernel, ScaleKernel

from data_processing.michaels import (
    MICHAELS_FILES,
    NUM_ROTORS,
    _load_michaels_data_raw,
    get_geometry,
)
from models.generative.harmonic_transform import (
    harmonic_lstsq_VP_transform,
    inverse_harmonic_VP_transform,
)

torch.set_default_dtype(torch.float32)


# ────────────────────────────────────────────────────────────────────────────


def _load_full(config):
    sr = config.sr
    wav_path, csv_path, toff, tdil = MICHAELS_FILES[config.recording]
    from data_processing.michaels import _DATA_ROOT as MROOT

    wav, ts, ms, _ = _load_michaels_data_raw(
        Path(MROOT) / wav_path,
        Path(MROOT) / csv_path,
        time_offset=toff,
        time_dilation=tdil,
        sr=sr,
    )
    return audio_segment(wav, ts, ms, sr, config)


def audio_segment(wav, ts_raw, ms_raw, sr, config):
    t0 = config.seg_start
    t1 = config.seg_start + config.seg_dur
    n_audio = int((t1 - t0) * sr)
    s0 = int(round(t0 * sr))
    s1 = s0 + n_audio
    audio = wav[:, s0:s1].astype(np.float32)  # (8, N)
    # interpolate per-rotor RPS onto audio timeline (ts_raw anchored at wav start, sec)
    mask = (ts_raw >= t0) & (ts_raw <= t1)
    if mask.sum() < 4:
        mask = (ts_raw >= t0 - 1.0) & (ts_raw <= t1 + 1.0)
    ts_w = ts_raw[mask]
    audio_t = np.linspace(t0, t1, n_audio)
    rps_audio = np.stack([np.interp(audio_t, ts_w, ms_raw[i][mask]) for i in range(NUM_ROTORS)], 0)
    return (
        torch.tensor(audio),
        torch.tensor(rps_audio.astype(np.float32)),  # (4, N)
        ts_raw.astype(np.float64),
        ms_raw.astype(np.float64),
    )


def _mic_yz():
    return torch.tensor(get_geometry()[0][:, 1:], dtype=torch.float32)  # (8,2)


def _extract_V(audio, rps_audio, H, win, hop, sr):
    Vs = [
        harmonic_lstsq_VP_transform(
            rps_audio, audio[m], n_harmonics=H, window_len=win, hop_len=hop, sr=sr
        )
        for m in range(audio.shape[0])
    ]
    return torch.stack(Vs, 0)  # (8, 4, H, F) complex


def _frame_rps(rps_audio, F, win, hop, sr):
    frame_centers = (np.arange(F) * hop + (win // 2)) / sr
    audio_t = np.linspace(0, rps_audio.shape[-1] / sr, rps_audio.shape[-1])
    arr = rps_audio.numpy()
    return torch.tensor(
        np.stack([np.interp(frame_centers, audio_t, arr[i]) for i in range(NUM_ROTORS)], 0),
        dtype=torch.float32,
    )  # (4,F)


# ── GP model ───────────────────────────────────────────────────────────────


class RotNoiseGP(gpytorch.models.ApproximateGP):
    """Variational GP over (mic_y, mic_z, rps_i, rotor_idx, harm_idx).

    k = scale * [ spatial·tonal · rotor_factor · harm_factor ]   (⊙ ⊙ ⊙)
    where spatial = Matérn-3/2(y,z), tonal = RBF(rps_i), and the two
    IndexKernel factors play the role of [43]'s "band-based tonal kernel" over
    discrete rotor / harmonic indices.
    """

    def __init__(self, inducing_x, n_rotors, n_harm):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_x.size(-2))
        strat = gpytorch.variational.VariationalStrategy(
            self, inducing_x, var_dist, learn_inducing_locations=True
        )
        super().__init__(strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.spatial = MaternKernel(nu=1.5, ard_num_dims=2, active_dims=(0, 1))
        self.tonal = RBFKernel(ard_num_dims=1, active_dims=(2,))
        self.rotor = IndexKernel(
            num_tasks=n_rotors, rank=min(max(1, n_rotors - 1), 4), active_dims=(3,)
        )
        self.harm = IndexKernel(
            num_tasks=n_harm, rank=min(max(1, n_harm - 1), n_harm), active_dims=(4,)
        )
        self.cov = ScaleKernel(self.spatial * self.tonal * self.rotor * self.harm)

    def forward(self, x):
        # ConstantMean([N, D]) -> [N]; ProductKernel over all 5 input dims.
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),  # type: ignore[arg-type]  gpytorch loose typing
            self.cov(x),
        )


# ────────────────────────────────────────────────────────────────────────────


def _design_xy(V_mag, rps_frame, mic_yz, frames):
    M, R, H, F = V_mag.shape
    feat, target = [], []
    for fi in frames:
        for m in range(M):
            ya, za = float(mic_yz[m, 0]), float(mic_yz[m, 1])
            for r in range(R):
                rr = float(rps_frame[r, fi])
                for h in range(H):
                    feat.append([ya, za, rr, float(r), float(h)])
                    target.append(float(V_mag[m, r, h, fi]))
    x = torch.tensor(feat, dtype=torch.float32)
    y = torch.log(torch.tensor(target, dtype=torch.float32) + 1e-8)
    return x, y


def main(config):
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    sr, win, hop, H = config.sr, config.win, config.hop, config.n_harmonics

    audio, rps_audio, ts_raw, ms_raw = _load_full(config)
    mic_yz = _mic_yz()
    F = (audio.shape[-1] - win) // hop + 1
    print(f"[load] audio={tuple(audio.shape)} rps_audio={tuple(rps_audio.shape)} F={F}")

    V = _extract_V(audio, rps_audio, H, win, hop, sr)
    rps_frame = _frame_rps(rps_audio, F, win, hop, sr)
    V_mag = V.abs() + 1e-8
    print(f"[vp] V={tuple(V.shape)} rps_frame={[float(v) for v in rps_frame.mean(1)]}")

    F_ho = min(config.holdout_frames, max(2, F // 4))
    train_pool = list(range(0, F - F_ho))
    test_pool = list(range(F - F_ho, F))
    stride = max(1, len(train_pool) // config.train_frames)
    train_idx = train_pool[::stride][: config.train_frames]
    print(f"[split] train frames={train_idx}  test frames={test_pool}")

    xtr, ytr = _design_xy(V_mag, rps_frame, mic_yz, train_idx)
    xte, yte = _design_xy(V_mag, rps_frame, mic_yz, test_pool)
    print(f"[design] xtr={tuple(xtr.shape)} xte={tuple(xte.shape)}")

    # standardise continuous dims (mic_y, mic_z, rps) — leave rotor/harm idx raw
    mus = torch.zeros(5)
    sds = torch.ones(5)
    mus[:3] = xtr[:, :3].mean(0)
    sds[:3] = xtr[:, :3].std(0) + 1e-6
    xtr = (xtr - mus) / sds
    xte = (xte - mus) / sds

    n_ind = min(256, xtr.shape[0])
    ind = xtr[np.linspace(0, xtr.shape[0] - 1, n_ind).astype(int)]
    model = RotNoiseGP(ind, NUM_ROTORS, H)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
    )
    model.train()
    likelihood.train()
    opt = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": 5e-2},
            {"params": likelihood.parameters(), "lr": 5e-2},
        ]
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=xtr.shape[0])

    best = 1e9
    for it in range(config.iters):
        opt.zero_grad()
        loss = -mll(model(xtr), ytr)  # type: ignore[operator]  gpytorch loose typing
        loss.backward()
        opt.step()
        best = min(best, loss.item())
        if it % 40 == 0 or it == config.iters - 1:
            print(
                f"[fit] it={it:4d} loss={loss.item():.4f} noise={likelihood.noise.sqrt().item():.4f}"
            )
    print(f"[fit] best loss={best:.4f}")

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(xte))
        mu = pred.mean
        std = pred.stddev

    rmse = torch.sqrt(((torch.exp(mu) - torch.exp(yte)) ** 2).mean()).item()
    resid_energy = (
        ((torch.exp(mu) - torch.exp(yte)) ** 2).sum() / (torch.exp(yte) ** 2).sum()
    ).item()
    ls = {
        "spatial_y": float(model.spatial.lengthscale[0, 0]),
        "spatial_z": float(model.spatial.lengthscale[0, 1]),
        "tonal_rps": float(model.tonal.lengthscale[0, 0]),
    }
    print(f"[eval] log-mag RMSE={rmse:.4f}  residual-energy ratio={resid_energy:.4f}")
    print(f"[eval] lengthscales={ls}")

    M = mic_yz.shape[0]
    F_te = len(test_pool)
    mu_pred = mu.reshape(M, NUM_ROTORS, H, F_te)

    # phase template = circular-mean complex phasor per (mic, rotor, harm) over train frames
    Vtr = V[:, :, :, train_idx]
    Vtr_u = Vtr / (Vtr.abs() + 1e-9)
    phase_tmpl = Vtr_u.mean(-1)  # (M,R,H)
    phase_unit = phase_tmpl / (phase_tmpl.abs() + 1e-9)  # unit phasor

    # align phase (M,R,H) over the F_te axis: phase_unit.unsqueeze(-1) -> (M,R,H,1)
    V_mean = torch.exp(mu_pred) * phase_unit.unsqueeze(-1)  # (M,R,H,F_te)
    # broadband-residual realisation: additative log-mag perturbation ~ N(0, std^2)
    resid = torch.randn_like(mu_pred) * std.reshape(*mu_pred.shape)
    V_noisy = torch.exp(mu_pred + resid) * phase_unit.unsqueeze(-1)  # (M,R,H,F_te)

    # build rps curve for the held-out audio span, interpolated to (4, n_gen_samples)
    # inverse_VP needs n_gen_samples >= win + (F_te-1)*hop so unfold works.
    f_start = train_pool[-1] + 1  # first held-out frame index (start of hold-out)
    n_gen_samples = win + (F_te - 1) * hop
    gen_t0 = config.seg_start + (f_start * hop) / sr
    gen_t1 = gen_t0 + n_gen_samples / sr
    mask = (ts_raw >= gen_t0) & (ts_raw <= gen_t1)
    if mask.sum() < 2:
        mask = (ts_raw >= gen_t0 - 2) & (ts_raw <= gen_t1 + 2)
    interp_t = np.linspace(gen_t0, gen_t1, n_gen_samples)
    rps_gen = torch.tensor(
        np.stack(
            [np.interp(interp_t, ts_raw[mask], ms_raw[i][mask]) for i in range(NUM_ROTORS)], 0
        ),
        dtype=torch.float32,
    )

    # inverse VP -> waveforms (sum rotors -> mono)
    # Squeeze the trailing 1 dim from V_mean[m] to match FrequencyIndex rank
    def inv(Vr):
        return inverse_harmonic_VP_transform(
            rps_gen, Vr.squeeze(-1).contiguous(), n_harmonics=H, window_len=win, hop_len=hop, sr=sr
        ).sum(0)

    gen_mean = inv(V_mean[0]).float()
    gen_noisey = inv(V_noisy[0]).float()

    # real held-out audio for mic 0 (segment-local: audio starts at seg_start;
    # frame f spans local samples [f*hop, f*hop + win + (F_te-1)*hop))
    s_gen0 = f_start * hop
    real_m0 = audio[0, s_gen0 : s_gen0 + n_gen_samples].numpy().astype(np.float32)

    # let the generated amplitude match the real level (a free gain — the GP
    # predicts log|V| whose absolute scale floats under the VP normalization)
    g = float((real_m0 * gen_mean.numpy()).sum() / ((gen_mean.numpy() ** 2).sum() + 1e-12))
    gen_mean = (gen_mean * g).clamp(-1, 1)
    gen_noisey = (gen_noisey * g).clamp(-1, 1)

    try:
        import soundfile as sf

        sf.write(out / "generated.wav", gen_mean.numpy().astype(np.float32), sr)
        sf.write(out / "generated_noisy.wav", gen_noisey.numpy().astype(np.float32), sr)
        sf.write(out / "real_holdout.wav", real_m0, sr)
    except Exception as e:
        print("[wav] skipped:", e)

    # spectrograms
    def stft_amp(x):
        x = np.asarray(x, np.float64).ravel()
        nf = x.shape[0] // win * win
        frames = x[:nf].reshape(-1, win)
        return np.abs(np.fft.rfft(frames * np.hanning(win), axis=-1)) + 1e-12

    spec_r = 10 * np.log10(stft_amp(real_m0))
    spec_g = 10 * np.log10(stft_amp(gen_mean.numpy()))
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
    ax[0, 1].set_title("GP-generated")
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
    fig.savefig(out / "gen_spectrum.png", dpi=110)
    plt.close(fig)

    metrics = {
        "rmse_logmag": rmse,
        "residual_energy_ratio": resid_energy,
        "loss_best": best,
        "lengthscales": ls,
        "gain_match": g,
        "n_train_points": int(xtr.shape[0]),
        "n_test_points": int(xte.shape[0]),
    }
    with open(out / "fit_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    np.savez(out / "coeffs.npz", mu=mu.numpy(), std=std.numpy(), yte=yte.numpy())
    print("[done]", json.dumps(metrics, indent=2))


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
    p.add_argument("--iters", type=int, default=400)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
