"""Jitter-off vs jitter-on, per-rotor harmonic/broadband decomposition.

Mirrors notebooks/drone_embedding_explorer.ipynb's "View 3" (render_components
/ Decompose), without the interactive widgets: renders the SAME conditioning
(DREGON code, DREGON RPS trajectory) through the E6 per-drone generator with
rps_jitter_sigma forced to 0 (jitter off) and to its learned value (jitter on),
splits each into harmonic-bank vs broadband-residual per rotor, and overlays
the two conditions' time-averaged spectra per rotor so the linewidth
difference (and the harmonic-peaks-under-broadband-floor gap) is visible.
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
from data_processing.online_mixing import _extract_audio_array, interpolate_rps_to_stft_grid
from models.generative.positional_harmonic_gen import propagate as _propagate
from models.registry import build_noise_gen_model
from tasks.noise_generation import geometry_to_rel_pos

SR = 16000
DEVICE = "cpu"
DRONES = ["dregon", "michaels"]
CKPT = PROJECT_ROOT / "omnirun-outputs/r2-artifacts/e6_noisegen_jitter_latreg_perdrone_best.ckpt"
DRONE = "dregon"
FMAX_ZOOM = 3000.0  # mid-range zoom for the line panels, per the user's ask


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


def spec_from_mag(S):
    return 20 * np.log10(S.mean(axis=1) + 1e-6)


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
    z = codebook.codes[DRONE].detach().cpu().numpy()
    sigma_on = F.softplus(log_jitter_sigma[DRONE]).item() if has_pd else 0.6
    print(f"sigma_on ({DRONE}):", sigma_on)

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
    real_all = _extract_audio_array(sl, target_len=n)
    rps = interpolate_rps_to_stft_grid(sl, n_frames=n, hop_length=1)
    rel_all = geometry_to_rel_pos(sl["mic_pos"].data, sl["rotor_pos"].data)
    mic = 0

    g = cast(Any, model).generator

    def render_components(sigma, seed=0):
        rps_t = torch.from_numpy(rps)[None].float().to(DEVICE)  # [1, R, T]
        rel_t = torch.from_numpy(rel_all[mic : mic + 1])[None].float().to(DEVICE)  # [1, 1, R, 3]
        b, R, T = rps_t.shape
        zt = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)
        folded = rps_t.reshape(b * R, 1, T)
        z_folded = zt[None].repeat_interleave(R, dim=0)
        sig_folded = torch.full((b * R,), max(0.0, float(sigma)), device=DEVICE)
        torch.manual_seed(seed)
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

            harm_r = [obs(harm[:, r : r + 1], rel_t[:, :, r : r + 1]) for r in range(R)]
            broad_r = [obs(broad[:, r : r + 1], rel_t[:, :, r : r + 1]) for r in range(R)]
        return harm_r, broad_r, R

    harm_off, broad_off, R = render_components(0.0)
    harm_on, broad_on, _ = render_components(sigma_on)

    HOP = 256
    total_off = [harm_off[r] + broad_off[r] for r in range(R)]
    total_on = [harm_on[r] + broad_on[r] for r in range(R)]
    mag_real = stft_mag(real_all[mic], hop=HOP)
    mag_off = [stft_mag(x, hop=HOP) for x in total_off]
    mag_on = [stft_mag(x, hop=HOP) for x in total_on]
    mag_broad_on = [stft_mag(x, hop=HOP) for x in broad_on]
    vmax = max(20 * np.log10(X.max() + 1e-6) for X in mag_off + mag_on)
    vmin = vmax - 80
    FREQS = np.fft.rfftfreq(1024, d=1 / SR)

    # High-resolution (Welch) spectra of the harmonic-only component, off vs on,
    # for a zoomed single-peak linewidth comparison (the coarse STFT above can't
    # resolve OU-jitter broadening, which is only a few Hz).
    def welch_db(x, nperseg=4096):
        from scipy.signal import welch

        f, p = welch(x, fs=SR, nperseg=nperseg)
        return f, 10 * np.log10(p + 1e-14)

    zoom_bands = []
    for r in range(R):
        f_on, p_on_db = welch_db(harm_on[r])
        band = np.logical_and(f_on > 300, f_on < 2500)
        pk = f_on[band][np.argmax(p_on_db[band])]
        zoom_bands.append((pk - 120, pk + 120))

    fig = plt.figure(figsize=(3.4 * R + 2.0, 10.6))
    gs = fig.add_gridspec(4, R, height_ratios=[1.5, 1.5, 1.2, 1.3], hspace=0.6, wspace=0.28)

    def _spec_panel(row, r, mag, label):
        ax = fig.add_subplot(gs[row, r])
        ax.imshow(
            20 * np.log10(mag + 1e-6),
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            extent=(0, dur_s, 0, SR / 2),
        )
        ax.set_ylim(0, FMAX_ZOOM)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelbottom=False)
        if r == 0:
            ax.set_ylabel("Hz")
        return ax

    for r in range(R):
        # jitter OFF vs ON look near-identical at STFT resolution (that's the
        # point of row 3's zoom below); show ON only here, to save vertical space.
        _spec_panel(0, r, mag_on[r], f"rotor {r} — total (jitter ON)")

    real_line = spec_from_mag(mag_real)
    row2_lines = []
    for r in range(R):
        line_off = spec_from_mag(mag_off[r])
        line_on = spec_from_mag(mag_on[r])
        line_broad = spec_from_mag(mag_broad_on[r])
        row2_lines.append((line_off, line_on, line_broad))
    # Shared y-range across all four rotor panels (row 2), restricted to the
    # displayed x-range (0..FMAX_ZOOM), so panels are directly comparable.
    band_mask = FREQS <= FMAX_ZOOM
    row2_vals = [real_line[band_mask]]
    for line_off, line_on, line_broad in row2_lines:
        row2_vals += [line_off[band_mask], line_on[band_mask], line_broad[band_mask]]
    row2_ymin = min(v.min() for v in row2_vals) - 3
    row2_ymax = max(v.max() for v in row2_vals) + 3

    last_axl = None
    for r in range(R):
        axl = fig.add_subplot(gs[1, r])
        line_off, line_on, line_broad = row2_lines[r]
        axl.plot(FREQS, real_line, lw=0.7, c="0.6", alpha=0.7, label="real (mic)")
        axl.plot(FREQS, line_off, lw=1.1, c="tab:blue", label="jitter OFF")
        axl.plot(FREQS, line_on, lw=1.1, c="tab:red", label="jitter ON")
        axl.plot(FREQS, line_broad, lw=1.0, c="tab:orange", ls="--", label="broadband floor (on)")
        axl.set_xlim(0, FMAX_ZOOM)
        axl.set_ylim(row2_ymin, row2_ymax)
        axl.grid(alpha=0.25)
        axl.set_xlabel("Hz")
        if r == 0:
            axl.set_ylabel("dB")
        last_axl = axl
    # One shared, readable legend outside the rightmost row-2 panel (avoids the
    # in-plot corner collision and the previous tiny 6.5pt font).
    assert last_axl is not None
    last_axl.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    for r in range(R):
        lo, hi = zoom_bands[r]
        f_off, p_off_db = welch_db(harm_off[r])
        f_on, p_on_db = welch_db(harm_on[r])
        axz = fig.add_subplot(gs[2, r])
        axz.plot(f_off, p_off_db, lw=1.3, c="tab:blue", label="jitter OFF")
        axz.plot(f_on, p_on_db, lw=1.3, c="tab:red", label="jitter ON")
        axz.set_xlim(lo, hi)
        band = np.logical_and(f_on > lo, f_on < hi)
        if band.sum() > 0:
            ymax = p_on_db[band].max()
            axz.set_ylim(ymax - 60, ymax + 5)
        axz.grid(alpha=0.25)
        axz.set_xlabel("Hz (zoom on one harmonic)")
        axz.set_title(f"rotor {r}: linewidth", fontsize=9)
        if r == 0:
            axz.set_ylabel("dB")
            axz.legend(fontsize=6.5, loc="upper right")

    # Row 4: WHERE the loss improvement materializes. Per-(freq,time) |Δ log-mag|
    # vs real, jitter OFF minus jitter ON, on the SAME comb-masked STFT grid the
    # slide-2 table's numbers come from (not the time-averaged line in row 2).
    # Positive (red) = jitter ON has smaller error there = where jitter wins.
    real_db = 20 * np.log10(mag_real + 1e-6)
    diff_maps = []
    for r in range(R):
        off_db = 20 * np.log10(mag_off[r] + 1e-6)
        on_db = 20 * np.log10(mag_on[r] + 1e-6)
        d = np.abs(off_db - real_db) - np.abs(on_db - real_db)
        # Smooth across time (3-frame moving average) to average out the OU
        # jitter's own frame-to-frame randomness and reveal the systematic
        # (frequency-band) structure of where the fix helps.
        kernel = np.ones(3) / 3
        d = np.apply_along_axis(lambda v, k=kernel: np.convolve(v, k, mode="same"), 1, d)
        diff_maps.append(d)
    band_vals = np.concatenate([np.abs(d[FREQS <= FMAX_ZOOM]).ravel() for d in diff_maps])
    diff_vmax = float(np.percentile(band_vals, 97))
    for r in range(R):
        axd = fig.add_subplot(gs[3, r])
        im = axd.imshow(
            diff_maps[r],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-diff_vmax,
            vmax=diff_vmax,
            extent=(0, dur_s, 0, SR / 2),
        )
        axd.set_ylim(0, FMAX_ZOOM)
        axd.set_xlabel("time (s)", fontsize=8)
        axd.set_title(f"rotor {r}: |Δ|off − |Δ|on", fontsize=9)
        if r == 0:
            axd.set_ylabel("Hz")
        if r == R - 1:
            cbar = fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("dB (red = jitter wins)", fontsize=7)
            cbar.ax.tick_params(labelsize=6.5)

    fig.suptitle(
        "Jitter off vs. on, per rotor (DREGON code + trajectory, mid-range zoom to "
        f"{FMAX_ZOOM / 1000:.0f} kHz)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 0.83, 0.95))
    outp = HERE / "assets" / "jitter_decompose.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print("saved", outp)


if __name__ == "__main__":
    main()
