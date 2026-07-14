"""Static analytic comb (E8) vs. the full neural generator, same RPS
trajectory: one spectrogram each, side by side. Answers round-4 critique
item 3 ("explain what is analytic static comb... show spectrograms").
"""

import sys
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_processing.dregon import load_dregon_timeframes
from data_processing.frames import get_meta
from data_processing.online_mixing import interpolate_rps_to_stft_grid
from data_processing.rotor_spectral_model import StaticCombNoisePool
from models.generative.positional_harmonic_gen import propagate as _propagate
from models.registry import build_noise_gen_model
from tasks.noise_generation import geometry_to_rel_pos

SR = 16000
DEVICE = "cpu"
DRONES = ["dregon", "michaels"]
CKPT = PROJECT_ROOT / "omnirun-outputs/r2-artifacts/e6_noisegen_jitter_latreg_perdrone_best.ckpt"
DRONE = "dregon"


def stft_mag(x, n_fft=1024, hop=256):
    return (
        torch.stft(
            torch.from_numpy(np.ascontiguousarray(x)).float(),
            n_fft=n_fft,
            hop_length=hop,
            window=torch.hann_window(n_fft),
            return_complex=True,
        )
        .abs()
        .numpy()
    )


def main():
    # ── same real DREGON clip + RPS trajectory as prepare_jitter_decompose.py ──
    _dr = load_dregon_timeframes(
        PROJECT_ROOT / "data", splits=["in_flight_noise"], target_sr=SR, download=False
    )
    rec = {f"dregon:{get_meta(tf, 'recording_id')}": tf for tf in _dr}
    rid = next((r for r in rec if r.startswith("dregon:free-flight")), next(iter(rec)))
    tf = rec[rid]
    t0 = tf["audio"].t_start
    dur_s = 4.0

    def _pick(dur_s, stride_s=2.0):
        avail = tf["audio"].t_end - t0
        best_s, best_m, s = 0.0, -1.0, 0.0
        while s + dur_s <= avail:
            m = float(
                interpolate_rps_to_stft_grid(
                    tf.time[t0 + s : t0 + s + dur_s], n_frames=16, hop_length=int(dur_s * SR / 16)
                ).mean()
            )
            if m > best_m:
                best_s, best_m = s, m
            s += stride_s
        return best_s

    s = _pick(dur_s)
    sl = tf.time[t0 + s : t0 + s + dur_s]
    n = int(round(dur_s * SR))
    rps = interpolate_rps_to_stft_grid(sl, n_frames=n, hop_length=1)  # [R, n]
    rel_all = geometry_to_rel_pos(sl["mic_pos"].data, sl["rotor_pos"].data)
    mic = 0

    # ── neural generator (E6, jitter+latreg, DREGON code), same RPS ──
    sd = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    has_pd = any(k.startswith("log_jitter_sigma") for k in sd)
    model = (
        build_noise_gen_model(
            "positional_harmonic_gen",
            sample_rate=SR,
            n_harmonics=100,
            cond_dim=16,
            drone_names=DRONES,
            rps_jitter_sigma=0.6,
            rps_jitter_tau=0.016,
            z_noise_std=0.1,
            film_spectral_norm=True,
            learn_rps_jitter_sigma=has_pd,
        )
        .to(DEVICE)
        .eval()
    )
    model.load_state_dict(sd, strict=True)
    codebook: Any = cast(Any, model).codebook
    log_jitter_sigma: Any = cast(Any, model).log_jitter_sigma
    z = codebook.codes[DRONE].detach().cpu().numpy()
    sigma_on = F.softplus(log_jitter_sigma[DRONE]).item() if has_pd else 0.6
    g = cast(Any, model).generator

    rps_t = torch.from_numpy(rps)[None].float().to(DEVICE)
    rel_t = torch.from_numpy(rel_all[mic : mic + 1])[None].float().to(DEVICE)
    b, R, T = rps_t.shape
    zt = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)
    folded = rps_t.reshape(b * R, 1, T)
    z_folded = zt[None].repeat_interleave(R, dim=0)
    sig_folded = torch.full((b * R,), sigma_on, device=DEVICE)
    torch.manual_seed(0)
    with torch.no_grad():
        out = g.emitter(
            folded, z=z_folded, rps_jitter=True, rps_jitter_sigma=sig_folded, return_dict=True
        )
        harm = out["harm_noise"].sum(-2).reshape(b, R, T)
        broad = out["diff_noise"].reshape(b, R, T)

        def obs(src, rsrc):
            return (
                _propagate(
                    src,
                    rsrc,
                    sample_rate=SR,
                    c=g.speed_of_sound,
                    ref_distance=g.ref_distance,
                    eps=g.eps,
                )[0, 0]
                .cpu()
                .numpy()
            )

        gen_mix = sum(
            obs(harm[:, r : r + 1] + broad[:, r : r + 1], rel_t[:, :, r : r + 1]) for r in range(R)
        )

    # ── static comb (E8): the pool samples its own RPS window internally (no
    # hook to inject an external trajectory), so this is NOT frame-identical to
    # the real DREGON clip driving the generator panel above -- it is seeded to
    # land in the same cruise regime (~80 rev/s) for a fair visual comparison.
    # Honesty over convenience: the caption below says so explicitly.
    pool = StaticCombNoisePool(sample_rate=SR, duration_s=dur_s, seed=0)
    tf_comb = pool.sample_timeframe(np.random.default_rng(3), duration_s=dur_s)
    comb_audio = np.asarray(tf_comb["audio"].data)
    comb_mix = comb_audio[mic, :n]
    if comb_mix.shape[-1] < n:
        comb_mix = np.pad(comb_mix, (0, n - comb_mix.shape[-1]))
    same_traj = False

    HOP = 256
    mag_gen = stft_mag(gen_mix, hop=HOP)
    mag_comb = stft_mag(comb_mix, hop=HOP)
    # Each panel normalised to its OWN peak (the two signals are at very
    # different absolute scales -- unit-RMS-ish analytic comb vs. the neural
    # generator's own learned level -- so a shared vmin/vmax would just make
    # one of the two panels look uniformly dim for no acoustic reason).

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, mag, title in (
        (axs[0], mag_comb, "Analytic static comb (E8)"),
        (axs[1], mag_gen, "Neural generator (E6, jitter+latreg)"),
    ):
        db = 20 * np.log10(mag + 1e-6)
        ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            vmin=db.max() - 80,
            vmax=db.max(),
            cmap="magma",
            extent=(0, dur_s, 0, SR / 2),
        )
        ax.set_ylim(0, 3000)
        ax.set_xlabel("time (s)")
        ax.set_title(title, fontsize=11)
    axs[0].set_ylabel("Hz")
    note = (
        "same RPS trajectory (real DREGON clip)"
        if same_traj
        else "matched RPS regime, NOT frame-identical"
    )
    fig.suptitle(f"Same 4 rotors, {note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outp = HERE / "assets" / "static_comb_vs_generator.png"
    fig.savefig(outp, dpi=140)
    print("saved", outp, "same_traj=", same_traj)


if __name__ == "__main__":
    main()
