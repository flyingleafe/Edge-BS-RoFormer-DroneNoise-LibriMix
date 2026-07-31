"""Evaluate noise-generator variants by valid MSSTFT on the corrected-geometry
swapped DREGON+Michael's split, and render real-vs-generated spectrograms.

Reuses the *exact* training-time validation computation (codec + MultiScaleSTFT
loss + AuraMRSTFTMetric over the valid loader) so the numbers are directly
comparable to the ``val/loss`` / ``mrstft`` logged during training. Each variant
is composed from its own Hydra experiment config, with ``data`` overridden to the
corrected ``frames:`` stream so every model is scored on the same real audio at
the corrected mic geometry.

Variants (checkpoints on R2 ``r2://ml-data/artifacts/<exp>/checkpoints/best.ckpt``):
  old_wronggeom : e6_noisegen_jitter_latreg_perdrone  (wrong-geometry baseline)
  v1_corrected  : gen_v1_corrected   (corrected geom, E6-perdrone arch)
  v2_perrotor   : gen_v2_perrotor    (+ per-rotor sub-embeddings)
  v3_wind       : gen_v3_wind         (+ wind-wake channel)

Run (from the worktree root, with R2 creds in .env)::

    python scripts/eval_noise_gen_variants.py --variants old_wronggeom v1_corrected \
        --val-samples 128 --out /tmp/.../gen_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from training.config import (  # noqa: E402
    build_dataset,
    build_metrics,
    build_task_and_codec,
    instantiate_model,
    register_configs,
)

register_configs()  # put RootConfig/base_config into Hydra's ConfigStore
from training.loop import (  # noqa: E402
    _forward,
    _iter_samples,
    _make_loader,
    _to_device,
    _warm_start,
)

VARIANTS: dict[str, tuple[str, str]] = {
    "old_wronggeom": (
        "e6_noisegen_jitter_latreg_perdrone",
        "r2://ml-data/artifacts/e6_noisegen_jitter_latreg_perdrone/checkpoints/best.ckpt",
    ),
    "v1_corrected": (
        "gen_v1_corrected",
        "r2://ml-data/artifacts/gen_v1_corrected/checkpoints/best.ckpt",
    ),
    "v2_perrotor": (
        "gen_v2_perrotor",
        "r2://ml-data/artifacts/gen_v2_perrotor/checkpoints/best.ckpt",
    ),
    "v3_wind": (
        "gen_v3_wind",
        "r2://ml-data/artifacts/gen_v3_wind/checkpoints/best.ckpt",
    ),
    # Retrains on the RECALIBRATED Michael's labels (michaels-frames
    # fdef818432e9). Same archs as v1/v2 — only the labels differ. NOTE these
    # score against a valid set that also moved, so compare them to v1/v2 as
    # "each model against the labels it was trained with", per drone.
    "v1_recal": (
        "gen_v1_recal",
        "r2://ml-data/artifacts/gen_v1_recal/checkpoints/best.ckpt",
    ),
    "v2_recal": (
        "gen_v2_recal",
        "r2://ml-data/artifacts/gen_v2_recal/checkpoints/best.ckpt",
    ),
}


def _compose(exp: str, ckpt: str, val_samples: int):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_ROOT / "conf"), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"experiment={exp}",
                "data=noise_rps_dregon_michaels_swapped_stream",
                f"data.valid.params.val_samples={val_samples}",
                f"checkpoint={ckpt}",
                "logging.enabled=false",
                "artifacts.enabled=false",
            ],
        )


def _rps_mean(frame) -> float:
    """Mean rotor speed (rev/s) of a sample — used to select the free-flight
    regime (>=~45) and exclude the idle/takeoff segments that dominate the
    val_at_start swapped split."""
    try:
        return float(np.asarray(frame["rps"].data).mean())
    except Exception:  # noqa: BLE001
        return float("nan")


def eval_variant(
    exp: str,
    ckpt: str,
    *,
    device: torch.device,
    val_samples: int,
    batch_size: int,
    min_flight_rps: float,
    illustration_frames: dict | None = None,
) -> tuple[dict[str, tuple[float, int]], dict]:
    """Score one variant on the *free-flight* subset (mean RPS >= ``min_flight_rps``)
    of the corrected-geometry swapped valid set, and — if ``illustration_frames``
    are given — generate audio for those clips too.

    Returns ``({group: (mrstft, n)}, {drone: (real_1d, gen_1d)})``, where
    ``group`` is ``"all"`` plus one key per drone (``dregon``/``michaels``,
    resolved from the clip geometry by :func:`_drone_of`). The per-drone split
    is what makes a Michael's-only claim testable — the pooled number mixes two
    rigs whose label quality differs. We report only the ``mrstft`` metric (the
    training monitor); the raw MSSTFT *loss* is a different STFT implementation
    and is intentionally not reported alongside it."""
    from data_processing.collate import frame_collate

    cfg = _compose(exp, ckpt, val_samples)
    _task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model).to(device)
    _warm_start(model, str(cfg.checkpoint), device)
    metric_suite = build_metrics(cfg.metrics)
    valid_loader = _make_loader(
        build_dataset(cfg.data.valid), batch_size=batch_size, num_workers=0, shuffle=False
    )

    model.eval()
    pairs: list = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = _to_device(batch, device)
            pred = _forward(codec, model, batch, device=device, amp=False)
            pred_cpu = pred.map_data(lambda t: t.detach().cpu())
            batch_cpu = batch.map_data(lambda t: t.detach().cpu())
            for pi, ti in zip(_iter_samples(pred_cpu), _iter_samples(batch_cpu)):
                if _rps_mean(ti) >= min_flight_rps:  # free-flight only
                    pairs.append((pi, ti))

    def _score(subset: list) -> tuple[float, int]:
        if not subset:
            return float("nan"), 0
        agg = metric_suite.evaluate(subset).aggregate("mean")
        return float(agg.get("mrstft", float("nan"))), len(subset)

    scores: dict[str, tuple[float, int]] = {"all": _score(pairs)}
    by_drone: dict[str, list] = {}
    for pi, ti in pairs:
        by_drone.setdefault(_drone_of(ti), []).append((pi, ti))
    for drone in sorted(by_drone):
        scores[drone] = _score(by_drone[drone])

    illust_gen: dict = {}
    if illustration_frames:
        with torch.no_grad():
            for drone, f in illustration_frames.items():
                batch = _to_device(frame_collate([f]), device)
                pred = _forward(codec, model, batch, device=device, amp=False)
                gen = pred.map_data(lambda t: t.detach().cpu())["audio"].data
                illust_gen[drone] = (_mic0(f["audio"].data), _mic0(gen))
    return scores, illust_gen


def _drone_of(frame) -> str:
    """DREGON vs Michael's from geometry (frame meta is empty on the stream): the
    Michael's mic ring sits at z≈0.33 m, DREGON's mics at z≈0."""
    try:
        if "mic_pos" in frame:
            z = float(np.asarray(frame["mic_pos"].data)[:, 2].mean())
            return "michaels" if z > 0.15 else "dregon"
    except Exception:  # noqa: BLE001
        pass
    return "dregon"


SPEC_FMAX = 4000.0  # repo convention (make_log_spectrogram_series default) — the
# band where the rotor harmonic stack lives; above it is essentially broadband.


def _spectrogram(
    audio: np.ndarray, sr: int = 16000, n_fft: int = 2048, hop: int = 512, fmax: float = SPEC_FMAX
):
    """dB log-STFT with real Hz / seconds axes, matching the repo's spectrogram
    convention (``src/plots/timeframe`` ``make_spectrogram_series``: n_fft=2048,
    hop=512, ``20*log10``, ``fmax=4000``). Cropping to ``fmax`` spreads the
    ~80 Hz-spaced rotor harmonics over the axis so they resolve as distinct
    horizontal lines. Returns ``(S_dB[freq,time], freqs_Hz, times_s)``."""
    if audio.ndim > 1:
        audio = audio[0]  # first microphone
    x = torch.as_tensor(np.ascontiguousarray(audio), dtype=torch.float32)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
        center=True,
    )
    s = 20.0 * np.log10(np.abs(X.numpy()) + 1e-8)  # [freq, time]
    nyq = sr / 2.0
    max_bin = max(1, min(int(round(fmax / nyq * s.shape[0])), s.shape[0]))
    s = s[:max_bin]
    freqs = np.linspace(0.0, fmax, s.shape[0])
    times = np.arange(s.shape[1]) * hop / sr
    return s, freqs, times


def _mic0(a) -> np.ndarray:
    """Reduce audio to the first microphone's 1-D waveform (handles [B,M,T]/[M,T]/[T])."""
    a = np.asarray(a)
    while a.ndim > 1:
        a = a[0]
    return a


def find_illustration_frames(seconds: int, sr: int, min_flight_rps: float, take: int = 96):
    """Find one clean *mid-flight* clip per drone for illustration. The swapped
    ``val_at_start`` split holds out only takeoff, so sustained-cruise clips live
    in the TRAIN region — we build that pool at ``seconds``-long chunks, scan the
    first ``take``, and keep the highest mean-RPS clip per drone. These clips are
    *illustrative* (the model saw this region in training); the held-out metric is
    the separate free-flight score. Returns ``({drone: Frame}, {drone: rps})``."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=gen_v1_corrected",
                "data=noise_rps_dregon_michaels_swapped_stream",
                f"data.train.params.chunk_size={seconds * sr}",
                "data.train.params.train_samples=512",
                "logging.enabled=false",
                "artifacts.enabled=false",
            ],
        )
    ds = build_dataset(cfg.data.train)
    best: dict[str, tuple[float, object]] = {}
    for i in range(take):
        try:
            f = ds[i]  # type: ignore[index]
        except (IndexError, KeyError, StopIteration):
            break
        d = _drone_of(f)
        r = _rps_mean(f)
        if r >= min_flight_rps and (d not in best or r > best[d][0]):
            best[d] = (r, f)
    frames = {d: fr for d, (_, fr) in best.items()}
    rps = {d: r for d, (r, _) in best.items()}
    return frames, rps


def render_spectrograms(illust: dict, rps_by_drone: dict, sr: int, seconds: int, out_path: Path):
    """Grid: rows = REAL + each variant, cols = one mid-flight DREGON + Michael's
    clip. dB log-STFT to ``SPEC_FMAX`` (repo convention, harmonics resolve),
    real Hz/seconds axes. ``illust`` = ``{variant: {drone: (real_1d, gen_1d)}}``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(illust)
    drones = ["dregon", "michaels"]
    real_by_drone = {d: rg[0] for d, rg in illust[names[0]].items()}
    nrows = len(names) + 1

    def _draw(ax, audio, is_bottom):
        s, fr, tm = _spectrogram(audio, sr)
        vmax = float(s.max())
        ax.pcolormesh(tm, fr, s, cmap="magma", vmin=vmax - 70.0, vmax=vmax, shading="auto")
        ax.set_ylim(0, SPEC_FMAX)
        if is_bottom:
            ax.set_xlabel("Time (s)")

    fig, axes = plt.subplots(nrows, 2, figsize=(11, 1.95 * nrows), squeeze=False)
    for c, d in enumerate(drones):
        real = real_by_drone.get(d)
        ax = axes[0][c]
        if real is not None:
            _draw(ax, real, False)
        ax.set_title(f"REAL — {d}  (RPS≈{rps_by_drone.get(d, 0):.0f})")
        if c == 0:
            ax.set_ylabel("real\nFreq (Hz)", fontsize=8)
        for r_i, name in enumerate(names, start=1):
            axc = axes[r_i][c]
            gen = illust[name].get(d, (None, None))[1]
            if gen is not None:
                _draw(axc, gen, r_i == nrows - 1)
            if c == 0:
                axc.set_ylabel(f"{name}\nFreq (Hz)", fontsize=8)
    fig.suptitle(
        f"Generated vs real drone noise — {seconds}s mid-flight clip, dB log-STFT "
        f"(n_fft=2048, 0–{SPEC_FMAX:.0f} Hz)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=120)
    print(f"\nspectrogram grid -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--val-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--min-flight-rps",
        type=float,
        default=45.0,
        help="score only free-flight samples (mean RPS >= this); "
        "excludes the idle/takeoff segments the val_at_start split holds out",
    )
    ap.add_argument("--out", type=Path, default=_ROOT / "gen_eval")
    ap.add_argument(
        "--illustrate-seconds",
        type=int,
        default=8,
        help="length (s) of the mid-flight illustration clip",
    )
    ap.add_argument("--no-spectrograms", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    illust_frames: dict = {}
    rps_by_drone: dict = {}
    if not args.no_spectrograms:
        print(f"finding {args.illustrate_seconds}s mid-flight illustration clips ...")
        illust_frames, rps_by_drone = find_illustration_frames(
            args.illustrate_seconds, 16000, args.min_flight_rps
        )
        print(f"  clips: {[(d, f'{r:.0f} rps') for d, r in rps_by_drone.items()]}")

    rows: list[tuple[str, dict[str, tuple[float, int]]]] = []
    illust_by_variant: dict[str, dict] = {}
    for name in args.variants:
        exp, ckpt = VARIANTS[name]
        print(f"=== {name}  ({exp}) ===")
        try:
            scores, illust_gen = eval_variant(
                exp,
                ckpt,
                device=device,
                val_samples=args.val_samples,
                batch_size=args.batch_size,
                min_flight_rps=args.min_flight_rps,
                illustration_frames=illust_frames or None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        rows.append((name, scores))
        illust_by_variant[name] = illust_gen
        for group, (mr, n) in scores.items():
            print(
                f"  {group:<9} mrstft = {mr:.3f}   (free-flight n={n}, RPS>={args.min_flight_rps:g})"
            )

    groups = ["all"] + sorted({g for _, s in rows for g in s if g != "all"})
    header = "".join(f"{g + ' ↑':>12}{'n':>7}" for g in groups)
    width = 16 + len(groups) * 19
    print("\n" + "=" * width)
    print(f"{'variant':<16}{header}")
    print("-" * width)
    for name, scores in rows:
        cells = "".join(
            f"{scores.get(g, (float('nan'), 0))[0]:>12.3f}{scores.get(g, (0, 0))[1]:>7d}"
            for g in groups
        )
        print(f"{name:<16}{cells}")
    csv = args.out / "msstft_comparison.csv"
    csv.write_text(
        "variant,group,mrstft,n_flight\n"
        + "".join(
            f"{name},{g},{scores[g][0]:.6f},{scores[g][1]}\n"
            for name, scores in rows
            for g in groups
            if g in scores
        )
    )
    print(f"\ntable -> {csv}")

    if not args.no_spectrograms and any(illust_by_variant.values()):
        render_spectrograms(
            illust_by_variant,
            rps_by_drone,
            16000,
            args.illustrate_seconds,
            args.out / "spectrograms.png",
        )


if __name__ == "__main__":
    main()
