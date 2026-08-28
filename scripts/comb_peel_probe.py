"""Is FOUR-rotor recovery achievable on the synthetic families at all?

Every trained model in this campaign emits a fixed fan: its predicted rotor
spread sits near 10 rev/s whatever the true spread does (see
docs/experiments/synthetic-solvability-limits.md). Under PIT-MSE that is exactly
the optimal answer for a model with no per-rotor information, so the fan alone
does not say whether the information is absent or merely unused.

This measures the ceiling with no network involved. Two model-free estimators
run on the same frames:

  peel  Find the best f0 by harmonic sum, suppress its comb, repeat four times.
        If the four rotors are separately identifiable from the spectrum, this
        recovers them.
  fan   Find ONE f0, then answer with four speeds evenly spaced by a fixed
        width centred on it — the same degenerate strategy the networks learned,
        built by hand.

peel far better than fan means the information is there and the networks are
leaving it on the table, so the lever is the curriculum or the loss. peel close
to fan means four-rotor recovery is near-unidentifiable in this family, and that
is a limit of the data rather than of any model.

    python scripts/comb_peel_probe.py --policy conf/online_mix/ladder_r4_dload.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_processing.frame_datasets import OnlineMixFrameDataset  # noqa: E402

SR, NFFT, HOP = 16000, 2048, 512
GRID = np.arange(30.0, 121.0, 0.25)
FREQS = np.fft.rfftfreq(NFFT, 1.0 / SR)
KS = np.arange(1, 61)
FAN_WIDTH = 10.0          # what the networks converged on
CRUISE_MIN = 45.0


def harmonic_score(db: np.ndarray, f0: float) -> float:
    """Comb energy at f0 above its own local background, in dB."""
    k = KS * f0
    k = k[k < FREQS[-1] - 10.0]
    if k.size < 8:
        return -np.inf
    i = np.searchsorted(FREQS, k)
    lo = np.clip(i - 4, 0, len(db) - 1)
    hi = np.clip(i + 4, 0, len(db) - 1)
    return float(db[i].mean() - 0.5 * (db[lo].mean() + db[hi].mean()))


def best_f0(db: np.ndarray) -> float:
    scores = np.array([harmonic_score(db, f) for f in GRID])
    return float(GRID[int(np.argmax(scores))])


def suppress(db: np.ndarray, f0: float) -> np.ndarray:
    """Replace the comb's bins with the local background, so the next pass
    cannot lock onto the same rotor again."""
    out = db.copy()
    for k in KS * f0:
        if k >= FREQS[-1] - 10.0:
            break
        i = int(np.searchsorted(FREQS, k))
        a, b = max(i - 2, 0), min(i + 3, len(out))
        ref = np.concatenate([out[max(i - 8, 0):a], out[b:min(i + 9, len(out))]])
        if ref.size:
            out[a:b] = np.median(ref)
    return out


def pit_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return min(float(np.mean(np.abs(np.asarray(p) - truth)))
               for p in itertools.permutations(pred))


def run(policy: str, n_clips: int, snr_db: float, frames_per_clip: int) -> dict:
    cfg = OmegaConf.load(policy)
    cfg.duration_s = 8.0
    for st in cfg.policy.stages:
        for k in ("augmentations", "noise_augmentations", "noise_time_warp"):
            if k in st:
                del st[k]
        st.snr_db = snr_db
    ds = OnlineMixFrameDataset.from_config(cfg, flatten_channels=True)

    peel_e, fan_e, spreads = [], [], []
    seen = 0
    for frame in ds:
        if seen >= n_clips:
            break
        rps = np.asarray(frame["rps"].data, dtype=np.float64)
        if float(np.mean(rps)) < CRUISE_MIN:
            continue
        audio = np.asarray(frame["mixture"].data, dtype=np.float64)
        x = audio[0] if audio.ndim > 1 else audio
        n_fr = 1 + (len(x) - NFFT) // HOP
        win = np.hanning(NFFT)
        step = max(n_fr // frames_per_clip, 1)
        for t in range(0, n_fr, step):
            spec = np.abs(np.fft.rfft(x[t * HOP:t * HOP + NFFT] * win))
            db = 20.0 * np.log10(np.maximum(spec, 1e-12))
            col = min(int(t * rps.shape[1] / max(n_fr, 1)), rps.shape[1] - 1)
            truth = np.sort(rps[:, col])

            work, picks = db.copy(), []
            for _ in range(4):
                f = best_f0(work)
                picks.append(f)
                work = suppress(work, f)
            peel_e.append(pit_mae(np.sort(np.array(picks)), truth))

            c = best_f0(db)
            fan = c + (np.arange(4) - 1.5) * (FAN_WIDTH / 3.0)
            fan_e.append(pit_mae(np.sort(fan), truth))
            spreads.append(float(truth.max() - truth.min()))
        seen += 1

    return {
        "policy": policy, "n_clips": seen, "n_frames": len(peel_e),
        "snr_db": snr_db,
        "peel_mae": float(np.median(peel_e)) if peel_e else float("nan"),
        "fan_mae": float(np.median(fan_e)) if fan_e else float("nan"),
        "peel_mean": float(np.mean(peel_e)) if peel_e else float("nan"),
        "fan_mean": float(np.mean(fan_e)) if fan_e else float("nan"),
        "true_spread": float(np.median(spreads)) if spreads else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--snr-db", type=float, default=-30.0)
    ap.add_argument("--frames-per-clip", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    print(f"{'policy':<44s} {'peel':>7s} {'fan':>7s} {'spread':>7s} {'frames':>7s}")
    for p in args.policy:
        r = run(p, args.n, args.snr_db, args.frames_per_clip)
        rows.append(r)
        print(f"{Path(p).stem:<44s} {r['peel_mae']:7.2f} {r['fan_mae']:7.2f} "
              f"{r['true_spread']:7.2f} {r['n_frames']:7d}")
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
