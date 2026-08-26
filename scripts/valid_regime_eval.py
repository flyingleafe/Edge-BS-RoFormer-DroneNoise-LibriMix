"""Score checkpoints on the frozen validation split, split by flight regime.

One row per checkpoint: the frame-weighted PIT mean squared error over the whole
split (the number training reports as ``val/mse``), and the PIT mean absolute
error inside each of the three regimes.

The regimes are defined on the TARGET speeds of each frame, so they do not
depend on the model:

  zero    max target < 1 rev/s              (every rotor stopped)
  low     1 <= max target, mean < 45        (warm-up, take-off, landing)
  flight  mean target >= 45                 (mid-flight)

Matching is per-frame Hungarian, pooled over channels and clips, which is the
convention every rotor-speed number in this project uses.

    python scripts/valid_regime_eval.py --exp stoch_s1_scv2 m3abl_comb_scv2_s1
    python scripts/valid_regime_eval.py --exp stoch_s1_scv2 --ckpt last --out r.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VALID = "dload:DREGON-LM-V4-michaels-valid-full"
REGIMES = ("zero", "low", "flight")
RIGS = ("dregon", "michaels")


def clip_rigs() -> list[str]:
    """The rig each clip of the split came from, in dataset order.

    The frozen split is 22 clips from three DREGON room1 recordings and 15 from
    Michael's FLY124 — two different airframes, arrays and rooms. Averaging over
    both hides a model that reads one rig and not the other, which is the whole
    question a synthetic-only model is asked.
    """
    import json

    from data_processing.streams import ensure_local

    root = Path(ensure_local(VALID.removeprefix("dload:")))
    rows = json.loads((root / "metadata.json").read_text())
    if isinstance(rows, dict):
        rows = next(iter(rows.values()))
    out = []
    for row in rows:
        rid = str(row.get("recording_id", ""))
        out.append("michaels" if "michael" in rid.lower() or rid.upper().startswith("FLY") else "dregon")
    return out


def frame_regimes(target: np.ndarray) -> np.ndarray:
    """``(F,)`` regime label per frame, from the ``(R, F)`` target speeds."""
    labels = np.full(target.shape[1], "low", dtype=object)
    labels[target.max(axis=0) < 1.0] = "zero"
    labels[target.mean(axis=0) >= 45.0] = "flight"
    return labels


def pit_abs_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``(R, F)`` absolute error after per-frame Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    n_rotors, n_frames = target.shape
    out = np.empty((n_rotors, n_frames), dtype=np.float64)
    for i in range(n_frames):
        cost = np.abs(pred[:, None, i] - target[None, :, i])
        rows, cols = linear_sum_assignment(cost)
        out[:, i] = np.abs(pred[rows, i] - target[cols, i])
    return out


def score(
    experiment: str,
    ckpt: str,
    channels: int,
    limit: int | None,
    rescale_rms: float | None = None,
) -> dict:
    """Score one checkpoint.

    ``rescale_rms`` scales every clip to that root-mean-square level before the
    model sees it. A synthetic pool hands its chunks over at a fixed level while
    a real recording arrives at whatever level it was recorded at, so a
    synthetic-only model can be reading its evaluation data far from where it
    learned; this measures how much of its error that accounts for, without
    retraining anything.
    """
    import tdseries as td

    import zoo
    from data_processing.frame_datasets import DregonLMFrameDataset
    from data_processing.frames import audio_series

    model = zoo.load(experiment, ckpt=ckpt, device="cpu")
    rigs = clip_rigs()
    errors: dict[str, list[np.ndarray]] = {r: [] for r in REGIMES}
    by_rig: dict[str, dict[str, list[np.ndarray]]] = {
        rig: {r: [] for r in REGIMES} for rig in RIGS
    }
    squared: list[np.ndarray] = []
    for channel in range(channels):
        dataset = DregonLMFrameDataset(
            data_dir=VALID, n_fft=2048, hop_length=512, sample_rate=16000, channel=channel
        )
        n = len(dataset) if limit is None else min(limit, len(dataset))
        for i in range(n):
            frame = dataset[i]
            target = np.asarray(frame["rps"].data, dtype=np.float64)
            if rescale_rms is not None:
                mixture = np.asarray(frame["mixture"].data, dtype=np.float32).reshape(1, -1)
                rms = float(np.sqrt(np.mean(np.square(mixture)))) or 1.0
                frame = td.Frame(
                    {"mixture": audio_series(mixture / rms * float(rescale_rms), 16000)}
                )
            pred = np.asarray(model(frame)["rps_pred"].data, dtype=np.float64)
            width = min(pred.shape[1], target.shape[1])
            err = pit_abs_error(pred[:, :width], target[:, :width])
            labels = frame_regimes(target[:, :width])
            squared.append((err**2).mean(axis=0))
            rig = rigs[i] if i < len(rigs) else "dregon"
            for regime in REGIMES:
                mask = labels == regime
                if mask.any():
                    errors[regime].append(err[:, mask].ravel())
                    by_rig[rig][regime].append(err[:, mask].ravel())
    all_squared = np.concatenate(squared)
    row = {
        "experiment": experiment,
        "ckpt": ckpt,
        "rescale_rms": rescale_rms,
        "aggregate_mse": float(all_squared.mean()),
        "all_mae": float(np.concatenate([np.concatenate(v) for v in errors.values() if v]).mean()),
        "n_frames": int(all_squared.size),
    }
    for regime in REGIMES:
        vals = np.concatenate(errors[regime]) if errors[regime] else np.array([np.nan])
        row[f"{regime}_mae"] = float(np.mean(vals))
        row[f"{regime}_frames"] = int(vals.size)
    for rig in RIGS:
        pooled = [v for vals in by_rig[rig].values() for v in vals]
        row[f"{rig}_all_mae"] = float(np.concatenate(pooled).mean()) if pooled else float("nan")
        for regime in REGIMES:
            vals = by_rig[rig][regime]
            row[f"{rig}_{regime}_mae"] = (
                float(np.concatenate(vals).mean()) if vals else float("nan")
            )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="+", required=True, help="experiment names in the zoo")
    parser.add_argument("--ckpt", default="best")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="clips per channel (debug)")
    parser.add_argument(
        "--rescale-rms",
        type=float,
        default=None,
        action="append",
        help="scale every clip to this RMS before scoring; repeatable",
    )
    parser.add_argument("--out", default=None, help="write the rows as JSON here")
    args = parser.parse_args()

    rows = []
    levels = args.rescale_rms or [None]
    for experiment in args.exp:
        for level in levels:
            try:
                row = score(experiment, args.ckpt, args.channels, args.limit, level)
            except Exception as exc:  # noqa: BLE001 — one bad checkpoint must not stop the sweep
                print(f"{experiment}: FAILED ({exc!r})", flush=True)
                continue
            rows.append(row)
            tag = "native" if level is None else f"rms={level:g}"
            print(
                f"{row['experiment']:26s} {tag:10s} aggregate {row['aggregate_mse']:8.2f}  "
                f"all-MAE {row['all_mae']:6.2f}  "
                f"zero {row['zero_mae']:6.2f}  low {row['low_mae']:6.2f}  "
                f"flight {row['flight_mae']:6.2f}",
                flush=True,
            )
            for rig in RIGS:
                print(
                    f"{'':26s} {rig:10s} all-MAE {row[f'{rig}_all_mae']:6.2f}  "
                    f"zero {row[f'{rig}_zero_mae']:6.2f}  "
                    f"low {row[f'{rig}_low_mae']:6.2f}  "
                    f"flight {row[f'{rig}_flight_mae']:6.2f}",
                    flush=True,
                )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
