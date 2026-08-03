"""Helpers for the 4-way noise comparison notebook (real / CONA / deep-gen / GP).

Row conventions (all rows resampled/rendered at ``SR`` = 16 kHz, body-frame
geometry, one or two reference microphones):

- **REAL** — a near-constant-RPS cruise window sliced from a full recording
  ``td.Frame`` (DREGON free-flight / Michael's), same machinery as
  ``notebooks/noise_gen_real_vs_generated.ipynb``.
- **CONA** — the constant-rps auralized case from the dload ``drone-egonoise``
  sweep nearest to the window's mean RPS; mics picked as the shell mic closest
  in body-frame position to each real reference mic (44.1 kHz -> 16 kHz).
- **DEEP** — the corrected-geometry ``gen_v1_corrected`` generator rendered on
  the window's true telemetry RPS trajectory at the real mic positions.
- **GP** — ``EgonoiseGPModel`` FM-trajectory synthesis (tonal + broadband)
  driven by the rotor-mean telemetry RPS, queried at the real mic body
  positions (44.1 kHz -> 16 kHz).

Metrics follow the E6 comb-mask convention (``docs/experiments/
noise-gen-linewidth.md``): comb-masked mean |Δ log-mag| per k-band along
telemetry harmonic tracks, plus msSTFT, harmonic-masked broadband floor/tilt,
numeric harmonic linewidth (FWHM), and inter-mic magnitude-squared coherence.
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalSubscript=false

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import coherence as _scipy_coherence
from scipy.signal import resample_poly

SR = 16_000
CONA_FS = 44_100
COMB_BANDS: dict[str, range] = {
    "k<10": range(1, 10),
    "10-25": range(10, 25),
    "25-40": range(25, 40),
}


# ════════════════════════════════════════════════════════════════════════════
# REAL rows: recording loading + cruise-window selection + slicing
# ════════════════════════════════════════════════════════════════════════════
def load_dregon_recording(project_root: Path, recording_id: str = "free-flight_nosource_room1"):
    """Load ONE DREGON in-flight recording as a ``td.Frame`` (memory-friendly:
    avoids materialising the whole split like ``load_dregon_timeframes``)."""
    from data_processing.sources.dregon import discover_recordings, get_geometry, load_timeframe

    dregon_dir = Path(project_root) / "data" / "DREGON"
    geometry = get_geometry(dregon_dir)
    samples = [s for s in discover_recordings(dregon_dir) if s["recording_id"] == recording_id]
    if not samples:
        raise FileNotFoundError(f"recording {recording_id!r} not found under {dregon_dir}")
    return load_timeframe(samples[0], geometry=geometry, target_sr=SR)


def slice_window(tf: Any, start_s: float, dur_s: float) -> dict[str, Any]:
    """Slice ``[start_s, start_s + dur_s]`` (relative to recording start).

    Returns ``target (M, T)``, audio-rate ``rps (R, T)``, ``rel (M, R, 3)``,
    and the body-frame ``mic_pos (M, 3)`` / ``rotor_pos (R, 3)``.
    Same slicing/interp calls as ``noise_gen_real_vs_generated.ipynb``.
    """
    from data_processing.online_mixing import _extract_audio_array, interpolate_rps_to_stft_grid
    from tasks.noise_generation import geometry_to_rel_pos

    t0 = tf["audio"].t_start
    avail = tf["audio"].t_end - t0
    if start_s < 0 or start_s + dur_s > avail:
        raise ValueError(f"window [{start_s}, {start_s + dur_s}] outside [0, {avail:.1f}] s")
    sl = tf.time[t0 + start_s : t0 + start_s + dur_s]
    n = int(round(dur_s * SR))
    target = _extract_audio_array(sl, target_len=n)  # (M, T)
    rps = interpolate_rps_to_stft_grid(sl, n_frames=n, hop_length=1)  # (R, T)
    mic_pos = np.asarray(sl["mic_pos"].data, dtype=np.float64)
    rotor_pos = np.asarray(sl["rotor_pos"].data, dtype=np.float64)
    rel = geometry_to_rel_pos(mic_pos, rotor_pos)[: target.shape[0]]
    return {
        "start_s": start_s,
        "dur_s": dur_s,
        "target": target,
        "rps": rps,
        "rel": rel,
        "mic_pos": mic_pos,
        "rotor_pos": rotor_pos,
    }


def find_cruise_window(
    tf: Any,
    dur_s: float = 2.0,
    rps_range: tuple[float, float] = (40.0, 85.0),
    step_s: float = 0.5,
    margin_s: float = 2.0,
) -> tuple[float, float, float]:
    """Scan the recording for the flattest cruise window.

    Minimises the max per-rotor RPS std over candidate windows whose rotor-mean
    RPS lies inside ``rps_range`` (the CONA sweep support). Uses the telemetry
    series directly (cheap). Returns ``(start_s, rps_mean, rps_maxstd)``.
    """
    from data_processing.mixing import resolve_motor_tracks
    from data_processing.sources.dregon import clean_command_spikes

    _, rps_key, needs_clean = resolve_motor_tracks(tf)
    rps_series = tf[rps_key]
    if needs_clean:
        rps_series = rps_series.map_data(clean_command_spikes)
    t0 = float(tf["audio"].t_start)
    t_end = float(tf["audio"].t_end)
    grid_hz = 50.0
    ts = np.arange(t0, t_end, 1.0 / grid_hz)
    ts = ts[
        (ts >= max(t0, float(rps_series.t_start))) & (ts <= min(t_end, float(rps_series.t_end)))
    ]
    vals = np.asarray(rps_series.interpolate(ts), dtype=np.float64)  # (R, N)
    best: tuple[float, float, float] | None = None
    start = margin_s
    while start + dur_s <= (t_end - t0) - margin_s:
        m = (ts >= t0 + start) & (ts <= t0 + start + dur_s)
        if m.sum() >= 8:
            w = vals[:, m]
            mean, maxstd = float(w.mean()), float(w.std(axis=1).max())
            if rps_range[0] <= mean <= rps_range[1] and (best is None or maxstd < best[2]):
                best = (start, mean, maxstd)
        start += step_s
    if best is None:
        raise RuntimeError("no cruise window found in the requested RPS range")
    return best


# ════════════════════════════════════════════════════════════════════════════
# DEEP row: corrected-geometry generator
# ════════════════════════════════════════════════════════════════════════════
def load_deep_generator(ckpt_path: Path, build_kwargs: dict[str, Any] | None = None):
    """Rebuild the ``gen_v1_corrected`` composite (generator + codebook +
    learnable per-drone jitter sigma) and load its ``best.ckpt`` state-dict.

    ``build_kwargs`` overrides default to the exact
    ``conf/model/positional_harmonic_gen_cond_jitter_latreg_perdrone.yaml``
    params (the v1 experiment config).
    """
    from models.registry import build_noise_gen_model

    kwargs: dict[str, Any] = dict(
        model_name="positional_harmonic_gen",
        sample_rate=SR,
        n_harmonics=100,
        use_diff_noise=True,
        cond_dim=16,
        drone_names=["dregon", "michaels"],
        rps_jitter_sigma=0.6,
        rps_jitter_tau=0.016,
        learn_rps_jitter_sigma=True,
        z_noise_std=0.1,
        film_spectral_norm=True,
    )
    if build_kwargs:
        kwargs.update(build_kwargs)
    model = build_noise_gen_model(**kwargs)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval()


def render_deep(
    model: Any,
    rps: np.ndarray,
    rel: np.ndarray,
    drone: str,
    *,
    rps_jitter: bool = True,
) -> np.ndarray:
    """Render the generator on a real RPS trajectory -> ``(M, T)`` float32.

    ``rps_jitter=True`` exposes the (eval-off-by-default) learned OU linewidth
    — the E6 rendering convention for jitter arms.
    """
    with torch.no_grad():
        out = model(
            torch.from_numpy(np.ascontiguousarray(rps)).float()[None],
            torch.from_numpy(np.ascontiguousarray(rel)).float()[None],
            [drone],
            rps_jitter=True if rps_jitter else None,
        )
    return out[0].cpu().numpy().astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# CONA row: dload ``drone-egonoise`` constant-rps cases
# ════════════════════════════════════════════════════════════════════════════
def cona_inventory(
    dataset: str = "drone-egonoise", version: str | None = None
) -> dict[str, dict[str, Any]]:
    """``key -> {drone, rps, seed, bpf_hz, n_blades}`` for every case (meta only)."""
    import dload

    from data_processing.streams import open_repository

    ds = open_repository().dataset(dataset, version)
    out: dict[str, dict[str, Any]] = {}
    for key, fields in ds.samples():
        meta = dload.codecs.json_from(fields["meta"])
        out[key] = {
            "drone": str(meta["drone"]),
            "rps": float(meta["rps"]),
            "seed": int(meta["seed"]),
            "bpf_hz": float(meta["bpf_hz"]),
            "n_blades": int(meta["n_blades"]),
        }
    return out


def nearest_cona_key(
    inventory: dict[str, dict[str, Any]], drone: str, rps_mean: float, seed: int = 0
) -> str:
    """The constant-rps case of ``drone`` (at ``seed``) nearest ``rps_mean``."""
    cands = {k: m for k, m in inventory.items() if m["drone"] == drone and m["seed"] == seed}
    if not cands:
        raise KeyError(f"no CONA cases for drone={drone} seed={seed}")
    return min(cands, key=lambda k: abs(cands[k]["rps"] - rps_mean))


def fetch_cona_case(
    key: str,
    cache_dir: Path,
    dataset: str = "drone-egonoise",
    version: str | None = None,
) -> dict[str, np.ndarray]:
    """Fetch one case's npz (disk-cached) -> dict of arrays
    (``audio``/``tonal``/``broadband`` (64, N) @ 44.1 kHz, ``mics_body`` (64, 3))."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{key}.npz"
    if not cached.exists():
        from data_processing.streams import open_repository

        ds = open_repository().dataset(dataset, version)
        for k, fields in ds.samples():
            if k == key:
                cached.write_bytes(bytes(fields["arrays"]))
                break
        else:
            raise KeyError(f"case {key!r} not found in {dataset}")
    arr = np.load(io.BytesIO(cached.read_bytes()))
    return {name: np.asarray(arr[name]) for name in arr.files}


def nearest_mic(mics_body: np.ndarray, ref_pos: np.ndarray) -> tuple[int, float]:
    """Index of the shell mic closest (Euclidean, body frame) to ``ref_pos``."""
    d = np.linalg.norm(np.asarray(mics_body, np.float64) - np.asarray(ref_pos, np.float64), axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def resample_to_sr(x: np.ndarray, fs_in: int = CONA_FS, fs_out: int = SR) -> np.ndarray:
    """Polyphase resample along the last axis (44 100 -> 16 000 uses 160/441)."""
    from math import gcd

    g = gcd(fs_out, fs_in)
    return resample_poly(np.asarray(x, np.float64), fs_out // g, fs_in // g, axis=-1).astype(
        np.float32
    )


# ════════════════════════════════════════════════════════════════════════════
# GP row: FM-trajectory synthesis from the trained EgonoiseGPModel
# ════════════════════════════════════════════════════════════════════════════
def load_gp(ckpt_path: Path):
    from experiments.gp_rotor_noise.train_egonoise_gp import EgonoiseGPModel

    return EgonoiseGPModel.load(ckpt_path)


def render_gp(
    gp: Any,
    mics_body: np.ndarray,
    rps_traj_sr: np.ndarray,
    *,
    broadband: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """FM synthesis at body-frame mic positions from a rotor-mean RPS trajectory.

    ``rps_traj_sr``: rotor-mean trajectory sampled at ``SR``; internally
    upsampled to the GP's native 44.1 kHz, synthesized (comb FM + broadband),
    and resampled back -> ``(M, T)`` at ``SR``.
    """
    n16 = rps_traj_sr.shape[-1]
    n44 = int(round(n16 * CONA_FS / SR))
    t16 = np.arange(n16) / SR
    t44 = np.arange(n44) / CONA_FS
    traj44 = np.interp(t44, t16, np.asarray(rps_traj_sr, np.float64))
    out44 = gp.synthesize(
        np.asarray(mics_body, np.float64),
        traj44,
        broadband="on" if broadband else "none",
        seed=seed,
    )
    out16 = resample_to_sr(out44)
    return out16[:, :n16]


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════
def logspec(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """dB log-magnitude STFT of a 1-D signal (E6 comb-metric convention)."""
    X = torch.stft(
        torch.from_numpy(np.ascontiguousarray(x)).float(),
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    return 20 * np.log10(np.abs(X.numpy()) + 1e-6)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(np.asarray(x, np.float64)))) + 1e-12))


def align_rms(x: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale ``x`` (single scalar) to the RMS of ``ref``.

    Returns ``(scaled, level_offset_db)`` where the offset is the row's native
    level minus the real reference level (positive = row was louder).
    """
    off = rms_db(x) - rms_db(ref)
    return (np.asarray(x, np.float64) * 10 ** (-off / 20)).astype(np.float32), off


def comb_error(
    real_1d: np.ndarray,
    row_1d: np.ndarray,
    rps: np.ndarray,
    n_fft: int = 1024,
    hop: int = 256,
) -> dict[str, float]:
    """Comb-masked mean |Δ log-mag| (dB) along telemetry tracks k·r_rotor(t),
    per k-band — byte-compatible with the E6 notebook convention."""
    St, Sp = logspec(real_1d, n_fft, hop), logspec(row_1d, n_fft, hop)
    n_bins, n_t = St.shape
    rf = rps[:, ::hop][:, :n_t]
    if rf.shape[1] < n_t:
        rf = np.pad(rf, ((0, 0), (0, n_t - rf.shape[1])), mode="edge")
    res: dict[str, float] = {}
    for band, ks in COMB_BANDS.items():
        errs = []
        for k in ks:
            for ridx in range(rf.shape[0]):
                bins = np.round(k * rf[ridx] * n_fft / SR).astype(int)
                valid = (bins > 0) & (bins < n_bins)
                if valid.sum():
                    tt = np.arange(n_t)[valid]
                    errs.append(np.abs(St[bins[valid], tt] - Sp[bins[valid], tt]))
        res[band] = float(np.concatenate(errs).mean()) if errs else float("nan")
    return res


_MSSTFT_LOSS = None


def msstft_distance(real_1d: np.ndarray, row_1d: np.ndarray) -> float:
    """Multi-scale STFT distance, same construction as the generator training
    loss (sizes [2048,1024,512,256,128], log_weight=1, L1)."""
    global _MSSTFT_LOSS
    if _MSSTFT_LOSS is None:
        from models.registry import build_noise_gen_loss

        _MSSTFT_LOSS = build_noise_gen_loss(
            stft_sizes=[2048, 1024, 512, 256, 128], log_weight=1.0, loss_type="L1"
        )
    with torch.no_grad():
        return float(
            _MSSTFT_LOSS(
                torch.from_numpy(np.ascontiguousarray(row_1d)).float()[None],
                torch.from_numpy(np.ascontiguousarray(real_1d)).float()[None],
            ).item()
        )


def floor_and_tilt(
    x_1d: np.ndarray,
    n_fft: int = 8192,
    f_lo: float = 300.0,
    f_hi: float = 4000.0,
    n_sub: int = 12,
    pctl: float = 10.0,
) -> tuple[float, float]:
    """Broadband floor level + spectral tilt via a lower-envelope estimate.

    With four distinct rotor speeds the harmonic comb is too dense at high k
    for explicit track masking (the tracks tile the whole band), so the
    inter-harmonic floor is estimated as the ``pctl``-th percentile of the
    Welch PSD (dB) inside ``n_sub`` log-spaced sub-bands of [f_lo, f_hi] —
    harmonic peaks occupy well under 90% of bins in every sub-band, so the low
    percentile lands between the lines. Returns ``(floor_db_at_1khz,
    tilt_db_per_octave)`` from a straight-line fit of envelope dB vs
    log2(f/1 kHz)."""
    from scipy.signal import welch

    nperseg = min(n_fft, len(x_1d))
    freqs, P = welch(
        np.asarray(x_1d, np.float64),
        fs=SR,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    Pdb = 10 * np.log10(P + 1e-20)
    edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n_sub + 1)
    fx, fy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        if sel.sum() >= 4:
            fx.append(np.log2(np.sqrt(lo * hi) / 1000.0))
            fy.append(float(np.percentile(Pdb[sel], pctl)))
    if len(fx) < 4:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.asarray(fx), np.asarray(fy), 1)
    return float(intercept), float(slope)


def linewidth_fwhm(
    x_1d: np.ndarray,
    f0_mean: float,
    ks: tuple[int, ...] = (10, 15, 20, 25),
    nperseg: int = 8192,
) -> dict[int, float]:
    """Numeric FWHM (Hz) of the strongest spectral peak near harmonic k·f0.

    Welch power spectrum (Hann, 50% overlap, 4x zero-padding) — segment
    averaging suppresses the speckle of jitter-broadened lines that defeats a
    single long FFT. Floor = 25th percentile of the ±0.35·f0 search band; FWHM
    walked out from the peak. Resolution floor ≈ 1.44·SR/nperseg Hz (~2.8 Hz
    at the default): clean tones (CONA/GP) pin to it — report "≤ resolution".
    """
    from scipy.signal import welch

    nperseg = min(nperseg, len(x_1d))
    freqs, P = welch(
        np.asarray(x_1d, np.float64),
        fs=SR,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nperseg * 4,
    )
    hz_per_bin = float(freqs[1] - freqs[0])
    out: dict[int, float] = {}
    for k in ks:
        fc = k * f0_mean
        band = (freqs >= fc - 0.45 * f0_mean) & (freqs <= fc + 0.45 * f0_mean)
        if band.sum() < 8:
            out[k] = float("nan")
            continue
        Pb = P[np.where(band)[0]]
        floor = float(np.percentile(Pb, 25))
        pk = int(np.argmax(Pb))
        half = floor + (Pb[pk] - floor) / 2.0
        lo = pk
        while lo > 0 and Pb[lo] > half:
            lo -= 1
        hi = pk
        while hi < len(Pb) - 1 and Pb[hi] > half:
            hi += 1
        if lo == 0 or hi == len(Pb) - 1:  # peak not resolved inside the band
            out[k] = float("nan")
            continue
        out[k] = float((hi - lo) * hz_per_bin)
    return out


def band_coherence(
    x_1d: np.ndarray,
    y_1d: np.ndarray,
    bands: tuple[tuple[float, float], ...] = ((0, 500), (500, 1500), (1500, 4000)),
    nperseg: int = 2048,
) -> dict[str, float]:
    """Mean magnitude-squared coherence between two mics, per frequency band."""
    f, C = _scipy_coherence(x_1d, y_1d, fs=SR, nperseg=nperseg, noverlap=nperseg // 2)
    out: dict[str, float] = {}
    for lo, hi in bands:
        sel = (f >= lo) & (f < hi)
        out[f"{int(lo)}-{int(hi)}Hz"] = float(C[sel].mean()) if sel.sum() else float("nan")
    return out
