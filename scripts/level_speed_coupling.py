"""How much does loudness tell you about rotor speed — in real flight, and in
each synthetic stream?

The campaign's 14 synthetic arms trade ramp accuracy against cruise accuracy at
Spearman -0.58, and the level sweep says why the two cells are different tasks:
cruise survives a 4x change of level almost untouched (arm H reads 2.05-2.16
rev/s on DREGON cruise at every level), while EVERY model's zero and ramp cells
explode when the clip is rescaled (the target's zero cell goes 2.87 -> 61.25).
Reading a stopped or slow rotor needs level; reading a cruising one needs
spacing and must ignore level.

A real flight gives both at once, because one recorder gain covers the whole
flight and the airframe gets louder as the rotors speed up. A synthetic stream
gives both only if its per-window random gain is NARROW compared with the
speed-driven level change. This script measures exactly that, per source:

    spearman(window RMS, window mean RPS)     how much level tells you
    dB span of level attributable to speed    against
    dB span of the random per-window gain     what buries it

Run remotely (uni-cpu). One row per source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SR = 16000
WIN = 1.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def windows(audio: np.ndarray, rps: np.ndarray, rps_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (rms, mean rps) over non-overlapping WIN-second windows."""
    n = int(WIN * SR)
    out_l, out_r = [], []
    for s in range(0, audio.shape[-1] - n + 1, n):
        seg = audio[..., s : s + n]
        rms = float(np.sqrt(np.mean(np.square(seg))))
        if rms <= 0:
            continue
        t0, t1 = s / SR, (s + n) / SR
        m = (rps_t >= t0) & (rps_t < t1)
        if not m.any():
            continue
        out_l.append(20 * np.log10(rms))
        out_r.append(float(np.mean(rps[:, m])))
    return np.asarray(out_l), np.asarray(out_r)


def real_rows(limit: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    from data_processing.streams import DloadFrameDataset

    rows = []
    for name in ("DREGON-frames", "michaels-frames"):
        ds = DloadFrameDataset(name)
        kept = 0
        for i, fr in enumerate(ds):
            if kept >= limit:
                break
            if "rps" not in fr or "audio" not in fr:
                continue  # motor runs and clean sources carry no rotor track
            kept += 1
            audio = np.asarray(fr["audio"].data, dtype=np.float64)
            if audio.ndim > 1:
                audio = audio[0]
            rps = np.asarray(fr["rps"].data, dtype=np.float64)
            stamps = np.asarray(fr["rps"].tindex.abs_stamps, dtype=np.float64)
            sr_in = float(fr["audio"].tindex.rate)
            if abs(sr_in - SR) > 1:
                step = max(int(round(sr_in / SR)), 1)
                audio = audio[::step]
            rows.append((f"real:{name}:{i}", *windows(audio, rps, stamps - stamps[0])))
    return rows


def stream_rows(policy: str, n_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    import yaml

    from data_processing.online_mixing import build_noise_stream

    cfg = yaml.safe_load(Path(policy).read_text())
    # The synthetic engines only — a real or silence source would mix two
    # different level laws into one number and the measure would mean nothing.
    src = [s for s in cfg["sources"]["noise"] if s.get("kind") in ("stochastic", "static_comb")]
    pipeline, _ = build_noise_stream(
        src, sample_rate=SR, window_s=WIN, seed=int(cfg.get("base_seed", 0))
    )
    lv, sp = [], []
    for i, chunk in enumerate(pipeline):
        if i >= n_chunks:
            break
        audio = np.asarray(chunk["audio"].data, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio[0]
        rps = np.asarray(chunk["rps"].data, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms <= 0:
            continue
        lv.append(20 * np.log10(rms))
        sp.append(float(np.mean(rps)))
    return np.asarray(lv), np.asarray(sp)


def report(label: str, lv: np.ndarray, sp: np.ndarray) -> dict:
    keep = np.isfinite(lv) & np.isfinite(sp)
    lv, sp = lv[keep], sp[keep]
    row = {
        "source": label,
        "n": int(lv.size),
        "spearman": spearman(lv, sp),
        "level_span_db": float(np.percentile(lv, 95) - np.percentile(lv, 5)) if lv.size else float("nan"),
    }
    # Level span attributable to speed: the fitted slope times the speed span.
    if lv.size > 3 and sp.std() > 0:
        slope = float(np.polyfit(sp, lv, 1)[0])
        row["db_per_revs"] = slope
        row["speed_driven_db"] = float(slope * (np.percentile(sp, 95) - np.percentile(sp, 5)))
        row["residual_db"] = float(np.std(lv - np.polyval(np.polyfit(sp, lv, 1), sp)))
    print(
        f"{label:34s} n={row['n']:5d}  spearman(level,rps) {row['spearman']:+.3f}  "
        f"level span {row['level_span_db']:6.1f} dB  "
        f"speed-driven {row.get('speed_driven_db', float('nan')):6.1f} dB  "
        f"residual {row.get('residual_db', float('nan')):5.1f} dB",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", nargs="*", default=[])
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--real-limit", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    lv_all, sp_all = [], []
    for label, lv, sp in real_rows(args.real_limit):
        if lv.size:
            lv_all.append(lv)
            sp_all.append(sp)
    if lv_all:
        rows.append(report("REAL (all recordings pooled)", np.concatenate(lv_all), np.concatenate(sp_all)))
        for label, lv, sp in real_rows(args.real_limit):
            if lv.size > 3:
                rows.append(report(label, lv, sp))
    for policy in args.policy:
        try:
            lv, sp = stream_rows(policy, args.chunks)
            rows.append(report(Path(policy).stem, lv, sp))
        except Exception as exc:  # noqa: BLE001
            print(f"{Path(policy).stem}: FAILED ({exc!r})", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
