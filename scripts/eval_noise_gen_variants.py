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
) -> tuple[float, int, list]:
    """Score one variant on the *free-flight* subset (mean RPS >= ``min_flight_rps``)
    of the corrected-geometry swapped valid set. Returns ``(mrstft, n_flight, pairs)``
    where ``pairs`` are the flight-only ``(pred, target)`` sample frames. We report
    only the ``mrstft`` metric (the training monitor); the raw MSSTFT *loss* is a
    different STFT implementation and is intentionally not reported alongside it."""
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
    metrics = metric_suite.evaluate(pairs).aggregate("mean") if pairs else {}
    mrstft = float(metrics.get("mrstft", float("nan")))
    return mrstft, len(pairs), pairs


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


def _logspec(audio: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio[0]
    x = torch.as_tensor(audio, dtype=torch.float32)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=n_fft // 4, window=torch.hann_window(n_fft), return_complex=True
    )
    return torch.log1p(spec.abs()).numpy()


def render_spectrograms(results: dict[str, list], sr: int, out_path: Path) -> None:
    """Grid: rows = variants (+ a 'real' row), cols = one dregon + one michaels
    sample; log-STFT magnitude. Uses the first sample of each drone per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    drones = ["dregon", "michaels"]
    # pick a fixed real target per drone (from the first variant's pairs)
    ref_pairs = results[names[0]]
    real_by_drone = {}
    for _pred, tgt in ref_pairs:
        d = _drone_of(tgt)
        if d not in real_by_drone and "audio" in tgt:
            real_by_drone[d] = np.asarray(tgt["audio"].data)
    nrows = len(names) + 1
    fig, axes = plt.subplots(nrows, 2, figsize=(9, 2.4 * nrows), squeeze=False)
    for c, d in enumerate(drones):
        real = real_by_drone.get(d)
        ax = axes[0][c]
        if real is not None:
            ax.imshow(_logspec(real), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"REAL — {d}")
        ax.set_ylabel("real" if c == 0 else "")
        for r, name in enumerate(names, start=1):
            gen = None
            for pred, tgt in results[name]:
                if _drone_of(tgt) == d and "audio" in pred:
                    gen = np.asarray(pred["audio"].data)
                    break
            axc = axes[r][c]
            if gen is not None:
                axc.imshow(_logspec(gen), origin="lower", aspect="auto", cmap="magma")
            axc.set_ylabel(name if c == 0 else "")
            axc.set_xticks([])
            axc.set_yticks([])
    fig.suptitle("Generated vs real noise — log-STFT magnitude (corrected geometry valid)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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
    ap.add_argument("--no-spectrograms", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[tuple[str, float, int]] = []
    pairs_by_variant: dict[str, list] = {}
    for name in args.variants:
        exp, ckpt = VARIANTS[name]
        print(f"=== {name}  ({exp}) ===")
        try:
            mrstft, n_flight, pairs = eval_variant(
                exp,
                ckpt,
                device=device,
                val_samples=args.val_samples,
                batch_size=args.batch_size,
                min_flight_rps=args.min_flight_rps,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        rows.append((name, mrstft, n_flight))
        pairs_by_variant[name] = pairs
        print(f"  mrstft = {mrstft:.3f}   (free-flight n={n_flight}, RPS>={args.min_flight_rps:g})")

    print("\n" + "=" * 52)
    print(f"{'variant':<16} {'mrstft ↑':>10} {'n_flight':>10}")
    print("-" * 52)
    for name, mr, n in rows:
        print(f"{name:<16} {mr:>10.3f} {n:>10d}")
    csv = args.out / "msstft_comparison.csv"
    csv.write_text(
        "variant,mrstft,n_flight\n" + "".join(f"{n},{mr:.6f},{nf}\n" for n, mr, nf in rows)
    )
    print(f"\ntable -> {csv}")

    if not args.no_spectrograms and pairs_by_variant:
        render_spectrograms(pairs_by_variant, sr=16000, out_path=args.out / "spectrograms.png")


if __name__ == "__main__":
    main()
