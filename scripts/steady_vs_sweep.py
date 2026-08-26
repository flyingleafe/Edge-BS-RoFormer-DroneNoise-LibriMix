"""Do the arms fail on STEADY low-speed frames, or on sweeping ones?

The ramp cell holds the whole remaining in-domain gap, and the two rigs ramp in
opposite ways: DREGON's ramp frames sweep (|d rps/dt| median 24.72 rev/s per
second) while Michael's mostly hold still (median 0.23) — the warm-up idle.
Michael's carries 1071 of the split's 1253 ramp frames.

Every arm trained on phase ranges that shortened the warm-up idle about fourfold
against the ranges calibrated to these recordings, so a sustained low-speed comb
is nearly absent from every synthetic stream. Arm ID restores it.

This tests that diagnosis WITHOUT waiting for arm ID to train. Split the real
low-speed frames by whether the true track is holding or sweeping, and score the
existing models on each half. If the synthetic arms are much worse than the
target on HELD frames specifically — and comparatively closer on swept ones —
the missing idle is the explanation. If they fail equally on both, it is not,
and arm ID will not help.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from valid_regime_eval import VALID, clip_rigs, pit_abs_error  # noqa: E402

FR = 16000 / 512.0  # STFT frames per second
HELD, SWEPT = 1.0, 5.0  # rev/s per second


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tdseries as td
    import torch

    import zoo
    from data_processing.frame_datasets import DregonLMFrameDataset
    from data_processing.frames import audio_series

    rigs = clip_rigs()
    ds = DregonLMFrameDataset(
        data_dir=VALID, n_fft=2048, hop_length=512, sample_rate=16000, flatten_channels=False
    )

    rows = []
    head = f"{'model':26s} {'rig':9s} {'held':>8s} {'swept':>8s} {'n_held':>7s} {'n_swept':>8s}"
    print(head)
    print("-" * len(head))
    for name in args.exp:
        try:
            model = zoo.load(name, ckpt="best", device="cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED ({exc!r})", flush=True)
            continue
        acc = {r: {"held": [], "swept": []} for r in ("dregon", "michaels")}
        for i in range(len(ds)):
            frame = ds[i]
            target = np.asarray(frame["rps"].data, dtype=np.float64)
            rig = rigs[i] if i < len(rigs) else "dregon"
            mean = target.mean(axis=0)
            low = (target.max(axis=0) >= 1.0) & (mean < 45.0)
            if not low.any():
                continue
            rate = np.abs(np.gradient(mean)) * FR
            audio = np.asarray(frame["mixture"].data, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[None]
            for ch in range(min(args.channels, audio.shape[0])):
                one = td.Frame({"mixture": audio_series(audio[ch][None], 16000)})
                with torch.no_grad():
                    pred = np.asarray(model(one)["rps_pred"].data, dtype=np.float64)
                w = min(pred.shape[1], target.shape[1])
                err = pit_abs_error(pred[:, :w], target[:, :w])
                lw, rw = low[:w], rate[:w]
                for key, sel in (("held", lw & (rw < HELD)), ("swept", lw & (rw >= SWEPT))):
                    if sel.any():
                        acc[rig][key].append(err[:, sel].ravel())
        row = {"experiment": name}
        for rig in ("dregon", "michaels"):
            h = np.concatenate(acc[rig]["held"]) if acc[rig]["held"] else np.array([np.nan])
            s = np.concatenate(acc[rig]["swept"]) if acc[rig]["swept"] else np.array([np.nan])
            row[rig] = {"held": float(h.mean()), "swept": float(s.mean()),
                        "n_held": int(h.size), "n_swept": int(s.size)}
            print(f"{name if rig == 'dregon' else '':26s} {rig:9s} "
                  f"{h.mean():8.2f} {s.mean():8.2f} {h.size:7d} {s.size:8d}", flush=True)
        rows.append(row)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
