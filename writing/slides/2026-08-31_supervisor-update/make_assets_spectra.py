#!/usr/bin/env python
"""Two spectrogram figures for the 2026-08-31 supervisor deck.

Both figures put REAL drone noise next to SYNTHETIC noise rendered over the
*same* rotor-speed trajectory, so a difference between panels is a difference
in the noise model and nothing else.

``assets/families_row.pdf``
    real | neural generator | static analytic comb (1 row, 3 columns).

``assets/stoch_samples.pdf``
    real | three stochastic-comb draws with different seeds (1 row, 4 columns).
    The comb spacing is identical in all three draws (one trajectory); the
    timbre, the floor and the line breathing are not.

The trajectory is the four-rotor RPS label of one clip of the frozen split
``dload:DREGON-LM-V4-michaels-valid-full``. The default clip is 20 —
``free-flight_nosource_room1``, a DREGON cruise window with NO speech and no
played source in it, so the "real" panel is drone noise alone. (Clip 8 is a
stop/start transition but carries played white noise; clip 36 is all-stopped.)

Every panel is drawn by the project's own renderer
(``plots.timeframe.renderers.make_log_spectrogram_series`` + the
``"audio_spectrogram"`` renderer), like
``writing/papers/2026-08_wrapup/make_figures.py``. That renderer draws a linear
0-4 kHz frequency axis with dB magnitudes — a harmonic comb is a set of equally
spaced lines on it, which is what the slide is about.

Levels: every panel is scaled to a common RMS before its spectrogram, and one
color scale is shared across the panels of a figure. Absolute level is
therefore NOT comparable between panels — only spectral structure is.

Usage::

    PYTHONPATH=src python writing/slides/2026-08-31_supervisor-update/make_assets_spectra.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

DECK = Path(__file__).resolve().parent
REPO_ROOT = DECK.parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tdseries as td  # noqa: E402

from plots.timeframe.registry import TrackContext, get_renderer  # noqa: E402
from plots.timeframe.renderers import make_log_spectrogram_series  # noqa: E402

# --- configuration ---------------------------------------------------------

DATA_DIR = "dload:DREGON-LM-V4-michaels-valid-full"
SR = 16_000
N_FFT = 2048
HOP = 512
FMAX_HZ = 4000.0
CLIP = 20  # DREGON free-flight_nosource_room1, cruise, no played source
CHANNEL = 0
DRONE = "dregon"  # the codebook entry / geometry matching clip 20's rig
PANEL_RMS = 0.1  # every panel normalised here before its spectrogram

#: Draw seeds, chosen by eye from a seed sweep (see the module docstring): the
#: comb profile of `COMB_SEED` carries lines through ~3 kHz like the real clip,
#: and the three stochastic draws differ in where the energy sits and in how
#: high the floor is while sharing one trajectory.
COMB_SEED = 42
STOCH_SEEDS = (1, 4, 11)

#: The generator of record: the M3 per-rotor conditioned generator at the epoch
#: every downstream curriculum run used (`conf/online_mix/m3cur_s1_dload.yaml`,
#: `m3abl_*`, `hb_m3mixed_dload.yaml`), selected comb-aware rather than by the
#: mrstft monitor.
GEN_CKPT = "r2://ml-data/artifacts/gen_m3_refined_all_perrotor/checkpoints/ep30_mrstft_6.7391.ckpt"
GEN_N_HARMONICS = 100  # must match the checkpoint's training value

STYLE = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "pdf.compression": 9,
}


# --- sources ---------------------------------------------------------------


def load_real(clip: int, channel: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(audio (T,), tracks (R, T))`` — one real clip and its RPS label
    resampled from the hop grid onto the audio grid."""
    from data_processing.frame_datasets import DregonLMFrameDataset

    ds = DregonLMFrameDataset(
        data_dir=DATA_DIR, n_fft=N_FFT, hop_length=HOP, sample_rate=SR, channel=channel
    )
    frame = ds[clip]
    audio = np.asarray(frame["mixture"].data, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio[0]
    rps_hop = np.asarray(frame["rps"].data, dtype=np.float64)  # (R, F)
    t_hop = np.linspace(0.0, len(audio) / SR, rps_hop.shape[1])
    t_aud = np.arange(len(audio)) / SR
    tracks = np.stack([np.interp(t_aud, t_hop, r) for r in rps_hop])
    print(
        f"clip {clip} ch{channel}: {len(audio) / SR:.1f} s, "
        f"rps {tracks.min():.1f}-{tracks.max():.1f} rev/s, "
        f"per-rotor means {np.round(tracks.mean(1), 1).tolist()}"
    )
    return audio, tracks


def render_neural(tracks: np.ndarray, *, seed: int = 0, device: str = "cpu") -> np.ndarray:
    """The trained neural generator, conditioned on ``tracks``.

    Same call the online-mix producer makes
    (``data_processing.generated_noise._producer_loop``): the emitter, this
    drone's codebook entry plus the per-rotor deltas, the rig geometry, random
    harmonic phases and the learned OU line jitter — only the excitation is
    ours instead of a synthetic trajectory. Rendered at microphone 0 only.
    """
    import torch

    from data_processing.generated_noise import _load_generator, load_geometry
    from models.generative.codebook import geometry_to_rel_pos

    params = {
        "checkpoint": GEN_CKPT,
        "sample_rate": SR,
        "n_harmonics": GEN_N_HARMONICS,
        "no_diff_noise": False,
        "model_name": "positional_harmonic_gen",
    }
    gb = _load_generator(params, device)
    mic_pos, rotor_pos = load_geometry(DRONE, "dload:DREGON")
    rel = torch.from_numpy(geometry_to_rel_pos(mic_pos[:1], rotor_pos)).float().to(device)
    z = gb.z_map[DRONE]
    if gb.rotor_deltas is not None:
        z = z.unsqueeze(0) + gb.rotor_deltas  # (R, d)
    rng = np.random.default_rng(seed)
    rps_t = torch.from_numpy(tracks[None]).float().to(device)
    n_rotors = tracks.shape[0]
    phases = (
        torch.from_numpy(rng.uniform(0.0, 2.0 * np.pi, size=(1, n_rotors, GEN_N_HARMONICS)))
        .float()
        .to(device)
    )
    kwargs: dict[str, Any] = {"initial_phases": phases}
    sigma = gb.sigma_map.get(DRONE)
    if sigma is not None:
        kwargs["rps_jitter"] = True
        kwargs["rps_jitter_sigma"] = torch.full((1,), float(sigma), device=device)
    with torch.no_grad():
        audio = gb.model(rps_t, rel.unsqueeze(0), z.unsqueeze(0), **kwargs).cpu().numpy()
    return np.asarray(audio[0, 0], dtype=np.float64)


def render_static_comb(tracks: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """The static analytic comb over ``tracks``.

    ``StaticCombNoisePool.render`` with the trajectory supplied instead of
    drawn: one profile per clip (``sample_profile``), a comb at ``k*rps(t)``
    with time-constant amplitudes (``_comb_waveform``), a pink-ish broadband
    floor (``_floor_waveform``), and the ``(rps/ref)^2.5`` level law. The four
    rotors SHARE one profile, as rotors of one airframe do — the convention
    `data_processing.comb_bench` settled on; the pool's per-rotor draw gives a
    loudness spread no real quadrotor has.
    """
    from data_processing.rotor_spectral_model import (
        ProfileRanges,
        _comb_waveform,
        _floor_waveform,
        sample_profile,
    )

    rng = np.random.default_rng(seed)
    n_rotors, n_t = tracks.shape
    ref = max(float(np.median(tracks)), 20.0)
    prof = sample_profile(
        rng,
        ProfileRanges(),
        n_harmonics=100,
        ref_rps=ref,
        sample_rate=SR,
        min_harm_above_floor=0.30,
    )
    a_k = np.asarray(prof.a_k, dtype=np.float64)
    out = np.zeros(n_t)
    for r in range(n_rotors):
        comb = _comb_waveform(tracks[r], a_k, SR, rng)
        floor = _floor_waveform(n_t, prof.floor_tilt, prof.floor_level, SR, rng)
        out += (comb + floor) * (np.maximum(tracks[r], 0.0) / 80.0) ** 2.5
    return out


def render_stochastic(tracks: np.ndarray, *, seed: int) -> np.ndarray:
    """One draw of the stochastic-comb family over ``tracks``.

    ``stochastic_rotor_noise.sample_params`` draws the clip's timbre, linewidths
    and floor; ``synthesize`` realises it by filtering white noise, so every
    line is a Lorentzian whose power breathes and every bin flickers. The four
    rotors share one airframe timbre (``rotor_similarity``), as in
    ``data_processing.comb_bench_stochastic``.
    """
    from data_processing.stochastic_rotor_noise import StochasticRanges, sample_params, synthesize

    rng = np.random.default_rng(seed)
    ranges = StochasticRanges(rotor_similarity=(0.9, 0.95))
    params = sample_params(rng, ranges, n_rotors=tracks.shape[0], n_harmonics=100, sample_rate=SR)
    audio, _ = synthesize(
        params, tracks, rng=rng, n_mics=1, line_mode="stochastic", normalize_rms=PANEL_RMS
    )
    return np.asarray(audio, dtype=np.float64).reshape(-1)[: tracks.shape[1]]


# --- drawing ---------------------------------------------------------------


def spec_db(audio: np.ndarray) -> Any:
    """Level-normalised audio -> the project's 0-4 kHz dB spectrogram track."""
    x = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x**2))) or 1.0
    series = td.uniform((x / rms * PANEL_RMS).astype(np.float32), SR, dims=("time",), t_start=0.0)
    return make_log_spectrogram_series(series, n_fft=N_FFT, hop_length=HOP, fmax=FMAX_HZ)


def draw(ax: Any, track: Any, title: str, clim: tuple[float, float], t_max: float) -> Any:
    ctx = TrackContext(
        ax=ax, name="spectrogram", t_start=0.0, t_end=t_max, style={"_hints": track.hints}
    )
    get_renderer("audio_spectrogram")(track.series, ctx)
    mesh = ax.collections[-1]
    mesh.set_clim(*clim)
    mesh.set_rasterized(True)  # a vector spectrogram mesh is megabytes
    ax.set_title(title, pad=4)
    ax.set_xlim(0.0, t_max)
    ax.set_ylim(0.0, FMAX_HZ)
    ax.set_yticks([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
    ax.set_yticklabels(["0", "1", "2", "3", "4"])
    ax.set_ylabel("")
    ax.set_xlabel("Time (s)")
    return mesh


def figure_row(
    panels: list[tuple[str, np.ndarray]], *, width: float, height: float, t_max: float
) -> Any:
    """One row of spectrograms on a shared color scale, with one colorbar."""
    tracks = [(label, spec_db(a)) for label, a in panels]
    pooled = np.concatenate(
        [np.asarray(t.series.data, dtype=np.float64).ravel() for _, t in tracks]
    )
    # Pooled percentiles, not per-panel: one scale for the whole row. The low
    # end sits at the 30th percentile, which was the setting that kept the
    # comb's high orders visible without washing out the real panel's floor.
    clim = (float(np.percentile(pooled, 30.0)), float(np.percentile(pooled, 99.8)))

    fig, axes = plt.subplots(1, len(tracks), figsize=(width, height), sharey=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, (label, track) in zip(axes, tracks, strict=True):
        mesh = draw(ax, track, label, clim, t_max)
    axes[0].set_ylabel("Frequency (kHz)")
    fig.tight_layout(rect=(0, 0, 0.935, 1))
    cax = fig.add_axes((0.948, 0.20, 0.011, 0.62))
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("dB", labelpad=2)
    cb.outline.set_visible(False)
    return fig


def save(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {out_dir / stem}.{{pdf,png}}")


# --- driver ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=int, default=CLIP)
    ap.add_argument("--channel", type=int, default=CHANNEL)
    ap.add_argument("--out-dir", default=str(DECK / "assets"))
    ap.add_argument("--seconds", type=float, default=0.0, help="crop to this many seconds (0=all)")
    ap.add_argument("--seed", type=int, default=7, help="neural-generator phase seed")
    ap.add_argument("--comb-seed", type=int, default=COMB_SEED)
    ap.add_argument(
        "--stoch-seeds",
        default=",".join(str(s) for s in STOCH_SEEDS),
        help="comma-separated seeds, one per stochastic panel",
    )
    args = ap.parse_args(argv)

    plt.rcParams.update(STYLE)
    out_dir = Path(args.out_dir)

    real, tracks = load_real(args.clip, args.channel)
    if args.seconds > 0:
        n = int(args.seconds * SR)
        real, tracks = real[:n], tracks[:, :n]
    t_max = len(real) / SR

    print("rendering the neural generator ...")
    neural = render_neural(tracks, seed=args.seed)
    print("rendering the static comb ...")
    comb = render_static_comb(tracks, seed=args.comb_seed)

    print("figure 1 (families) ...")
    fig = figure_row(
        [("real", real), ("neural generator", neural), ("static comb", comb)],
        width=9.6,
        height=2.75,
        t_max=t_max,
    )
    save(fig, out_dir, "families_row")

    print("rendering three stochastic-comb draws ...")
    stoch_seeds = [int(t) for t in str(args.stoch_seeds).split(",") if t.strip()]
    stoch = [render_stochastic(tracks, seed=s) for s in stoch_seeds]

    print("figure 2 (stochastic family) ...")
    fig = figure_row(
        [("real", real)] + [(f"stochastic sample {i + 1}", s) for i, s in enumerate(stoch)],
        width=12.6,
        height=2.75,
        t_max=t_max,
    )
    save(fig, out_dir, "stoch_samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
