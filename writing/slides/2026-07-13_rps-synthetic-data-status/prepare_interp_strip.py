"""Generate the drone-embedding interpolation strips (E4-5 of the narrative).

Loads the trained per-drone generator checkpoint (E6 jitter_latreg_perdrone)
and renders the generator's output at several interpolated embedding codes
alpha in [0,1] between the DREGON and Michael's codebook entries, on a fixed
real RPS trajectory. Mirrors notebooks/drone_embedding_explorer.ipynb's core
render() path, without the interactive widgets.
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
from data_processing.michaels import load_michaels_timeframes
from data_processing.online_mixing import _extract_audio_array, interpolate_rps_to_stft_grid
from models.registry import build_noise_gen_model
from tasks.noise_generation import geometry_to_rel_pos

SR = 16000
DEVICE = "cpu"
DRONES = ["dregon", "michaels"]
CKPT = PROJECT_ROOT / "omnirun-outputs/r2-artifacts/e6_noisegen_jitter_latreg_perdrone_best.ckpt"


def main():
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
    Z = {d: codebook.codes[d].detach().cpu().numpy() for d in DRONES}
    SIG = {d: (F.softplus(log_jitter_sigma[d]).item() if has_pd else 0.6) for d in DRONES}
    print("sigma:", SIG)

    _dr = load_dregon_timeframes(
        PROJECT_ROOT / "data", splits=["in_flight_noise"], target_sr=SR, download=False
    )
    _mi = load_michaels_timeframes(data_root=PROJECT_ROOT / "data", sr=SR)
    rec = {f"dregon:{get_meta(tf, 'recording_id')}": tf for tf in _dr}
    rec.update({f"michaels:FLY{get_meta(tf, 'recording_id')}": tf for tf in _mi})

    def _pick(rec_id, dur_s, stride_s=2.0):
        tf = rec[rec_id]
        t0 = tf["audio"].t_start
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

    def _load_window(rec_id, dur_s=4.0):
        tf = rec[rec_id]
        t0 = tf["audio"].t_start
        s = _pick(rec_id, dur_s)
        sl = tf.time[t0 + s : t0 + s + dur_s]
        n = int(round(dur_s * SR))
        tgt = _extract_audio_array(sl, target_len=n)
        rps = interpolate_rps_to_stft_grid(sl, n_frames=n, hop_length=1)
        rel = geometry_to_rel_pos(sl["mic_pos"].data, sl["rotor_pos"].data)
        return dict(rec_id=rec_id, real_all=tgt, rps=rps, rel_all=rel, n_mics=tgt.shape[0])

    _drid = next(
        (r for r in rec if r.startswith("dregon:free-flight")),
        next(r for r in rec if r.startswith("dregon:")),
    )
    _miid = next(
        (r for r in rec if r.startswith("michaels:FLY125")),
        next(r for r in rec if r.startswith("michaels:")),
    )
    win = {"dregon": _load_window(_drid), "michaels": _load_window(_miid)}

    zd, zm = Z["dregon"], Z["michaels"]

    def z_at(alpha):
        return zd * (1 - alpha) + zm * alpha

    def sigma_at(alpha):
        return max(0.0, (1 - alpha) * SIG["dregon"] + alpha * SIG["michaels"])

    def render(z, sigma, path, mic=0, seed=0):
        z = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)
        w: Any = win[path]
        torch.manual_seed(seed)
        mic = min(mic, w["n_mics"] - 1)
        with torch.no_grad():
            return (
                cast(Any, model)
                .generator(
                    torch.from_numpy(w["rps"])[None].float().to(DEVICE),
                    torch.from_numpy(w["rel_all"][mic : mic + 1])[None].float().to(DEVICE),
                    z=z[None],
                    rps_jitter=True,
                    rps_jitter_sigma=torch.tensor([max(0.0, float(sigma))], device=DEVICE),
                )[0, 0]
                .cpu()
                .numpy()
            )

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

    def make_strip(path, alphas, fname, title, drone_start, drone_end):
        """Strip with a REAL panel prepended (drone_start, matching alpha=0)
        and a REAL panel appended (drone_end, matching alpha=1), so every
        generated-texture strip is bracketed by the two real drones it's
        interpolating between. Laid out as a 2-row grid (not one long row) so
        each spectrogram panel is large enough to read on a slide."""
        n_panels = len(alphas) + 2
        n_cols = (n_panels + 1) // 2  # 2 rows, ceil(n_panels/2) cols
        fig, axgrid = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 6.6), sharey=True, sharex=True)
        axs = axgrid.reshape(-1)
        for ax in axs[n_panels:]:
            ax.axis("off")

        # Precompute all panel signals first so the whole strip (real anchors +
        # generated interpolants) shares ONE dB normalization — otherwise each
        # panel's independent vmax hides real brightness differences between
        # real and generated audio.
        real_start_audio = win[drone_start]["real_all"][0]
        gen_audios = [render(z_at(a), sigma_at(a), path, mic=0) for a in alphas]
        real_end_audio = win[drone_end]["real_all"][0]
        all_S = [
            20 * np.log10(stft_mag(x) + 1e-6)
            for x in [real_start_audio, *gen_audios, real_end_audio]
        ]
        shared_vmax = max(S.max() for S in all_S)
        shared_vmin = shared_vmax - 80

        def _panel(ax, S, label, real=False):
            ax.imshow(
                S,
                origin="lower",
                aspect="auto",
                vmin=shared_vmin,
                vmax=shared_vmax,
                cmap="magma",
                extent=[0, 4.0, 0, SR / 2],
            )
            ax.set_title(label, fontsize=13)
            ax.set_xlabel("s")
            if real:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#c0392b")
                    spine.set_linewidth(2.5)

        _panel(axs[0], all_S[0], f"REAL ({drone_start})", real=True)
        for ax, a, S in zip(axs[1:-1], alphas, all_S[1:-1]):
            _panel(ax, S, f"alpha={a:.2f}")
        _panel(axs[n_panels - 1], all_S[-1], f"REAL ({drone_end})", real=True)

        for row in axgrid:
            row[0].set_ylabel("Hz")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        outp = HERE / "assets" / fname
        fig.savefig(outp, dpi=130)
        print("saved", outp)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    make_strip(
        "dregon",
        alphas,
        "interp_strip_dregon_to_michaels.png",
        "Generator embedding interpolation: DREGON -> Michael's (on DREGON RPS trajectory), real audio bracketing each end",
        drone_start="dregon",
        drone_end="michaels",
    )
    make_strip(
        "michaels",
        list(reversed(alphas)),
        "interp_strip_michaels_to_dregon.png",
        "Generator embedding interpolation: Michael's -> DREGON (on Michael's RPS trajectory), real audio bracketing each end",
        drone_start="michaels",
        drone_end="dregon",
    )

    # --- per-drone real | generated pair for the generator-improvements slides ---
    def make_real_vs_gen(drone, fname):
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
        real_audio = win[drone]["real_all"][0]
        gen_audio = render(z_at(0.0 if drone == "dregon" else 1.0), SIG[drone], drone, mic=0)
        # Shared dB normalization across the pair: otherwise each panel's own
        # vmax hides a real brightness/energy difference between real and
        # generated audio (see critique round 2, item 2).
        S_real = 20 * np.log10(stft_mag(real_audio) + 1e-6)
        S_gen = 20 * np.log10(stft_mag(gen_audio) + 1e-6)
        shared_vmax = max(S_real.max(), S_gen.max())
        shared_vmin = shared_vmax - 80
        for ax, S, label in [
            (axs[0], S_real, f"REAL ({drone})"),
            (axs[1], S_gen, f"generated ({drone})"),
        ]:
            ax.imshow(
                S,
                origin="lower",
                aspect="auto",
                vmin=shared_vmin,
                vmax=shared_vmax,
                cmap="magma",
                extent=[0, 4.0, 0, SR / 2],
            )
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("s")
        axs[0].set_ylabel("Hz")
        fig.suptitle(f"Real vs. generated: {drone}", fontsize=12)
        plt.tight_layout()
        outp = HERE / "assets" / fname
        fig.savefig(outp, dpi=130)
        print("saved", outp)

    make_real_vs_gen("dregon", "real_vs_gen_dregon.png")
    make_real_vs_gen("michaels", "real_vs_gen_michaels.png")


if __name__ == "__main__":
    main()
