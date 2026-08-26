"""Average the synthetic-only arms instead of choosing between them.

Every router built so far SELECTS one specialist per frame, and all four judge
designs plateaued at 5.47 against a 3.72 oracle ceiling, because the regime
boundary is not identifiable from these models to the accuracy the oracle
assumes. Averaging is a different mechanism: it needs no regime decision at all,
and an ensemble usually beats its own best member.

The one thing that must be right is rotor order. Each model emits R rotor tracks
in an arbitrary order — the whole campaign is scored under permutation-invariant
matching for exactly that reason — so averaging raw outputs would mix rotor 1 of
one model with rotor 3 of another and destroy the estimate. Every model is
therefore Hungarian-aligned to a reference model on the time-mean of its tracks
before it is combined.

Three combiners are reported:

    mean     the ensemble mean, all members equal
    median   the ensemble median, robust to one member failing badly
    oracle   the true regime picks the member that owns that cell (the ceiling
             the routers were chasing), for reference

Scored on the same frozen split, same per-frame PIT matching, same rig split.
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

from valid_regime_eval import (  # noqa: E402
    REGIMES,
    RIGS,
    VALID,
    clip_rigs,
    frame_regimes,
    pit_abs_error,
)

DEFAULT_MEMBERS = [
    "stoch_s1h_scv2",
    "stoch_s1s_both",
    "m3abl_comb_unigru128_s1",
    "stoch_s1v_ground",
    "stoch_s1x_scv2",
]
ROUTE = {
    "zero": "m3abl_comb_unigru128_s1",
    "low": "stoch_s1s_both",
    "flight": "stoch_s1h_scv2",
}


def align_to(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Permute `other`'s rotor axis onto `ref`'s, by time-mean cost.

    Without this the mean mixes one model's rotor 1 with another's rotor 3. The
    match is made once per clip on the time-mean of each track rather than per
    frame, so the permutation cannot flicker within a clip.
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.abs(ref.mean(axis=1)[:, None] - other.mean(axis=1)[None, :])
    _, col = linear_sum_assignment(cost)
    return other[col]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", nargs="*", default=DEFAULT_MEMBERS)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    import tdseries as td
    import zoo
    from data_processing.frame_datasets import DregonLMFrameDataset
    from data_processing.frames import audio_series

    models = {n: zoo.load(n, ckpt="best", device="cpu") for n in args.members}
    rigs = clip_rigs()
    dataset = DregonLMFrameDataset(
        data_dir=VALID, n_fft=2048, hop_length=512, sample_rate=16000,
        flatten_channels=False,
    )
    n_clips = len(dataset) if args.limit is None else min(len(dataset), args.limit)

    keys = ["mean", "median", "oracle", *[f"single:{n}" for n in args.members]]
    err = {k: {r: {g: [] for g in REGIMES} for r in RIGS} for k in keys}

    for i in range(n_clips):
        frame = dataset[i]
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        rig = rigs[i] if i < len(rigs) else "dregon"
        audio = np.asarray(frame["mixture"].data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None]
        for ch in range(min(args.channels, audio.shape[0])):
            one = td.Frame({"mixture": audio_series(audio[ch][None], 16000)})
            preds = {}
            for n, m in models.items():
                with torch.no_grad():
                    preds[n] = np.asarray(m(one)["rps_pred"].data, dtype=np.float64)
            width = min(min(p.shape[1] for p in preds.values()), target.shape[1])
            tgt = target[:, :width]
            preds = {n: p[:, :width] for n, p in preds.items()}

            ref_name = args.members[0]
            ref = preds[ref_name]
            stack = np.stack(
                [preds[n] if n == ref_name else align_to(ref, preds[n]) for n in args.members]
            )
            combined = {"mean": stack.mean(axis=0), "median": np.median(stack, axis=0)}

            errs = {n: pit_abs_error(p, tgt) for n, p in preds.items()}
            for k, v in combined.items():
                errs[k] = pit_abs_error(v, tgt)

            labels = frame_regimes(tgt)
            for regime in REGIMES:
                m = labels == regime
                if not m.any():
                    continue
                for n in args.members:
                    err[f"single:{n}"][rig][regime].append(errs[n][:, m].ravel())
                for k in ("mean", "median"):
                    err[k][rig][regime].append(errs[k][:, m].ravel())
                err["oracle"][rig][regime].append(errs[ROUTE[regime]][:, m].ravel())

    rows = []
    head = f"{'system':34s} {'rig':9s} {'all':>7s} {'zero':>7s} {'low':>7s} {'flight':>7s}"
    print(head)
    print("-" * len(head))
    for key in keys:
        row: dict = {"system": key}
        for rig in [*RIGS, "both"]:
            cells = {}
            for regime in REGIMES:
                vals = (
                    [v for r in RIGS for v in err[key][r][regime]]
                    if rig == "both"
                    else err[key][rig][regime]
                )
                cells[regime] = float(np.concatenate(vals).mean()) if vals else float("nan")
            pooled = [
                v for r in ([*RIGS] if rig == "both" else [rig])
                for regime in REGIMES for v in err[key][r][regime]
            ]
            allm = float(np.concatenate(pooled).mean()) if pooled else float("nan")
            row[rig] = {"all": allm, **cells}
            print(
                f"{key if rig == RIGS[0] else '':34s} {rig:9s} {allm:7.2f} "
                f"{cells['zero']:7.2f} {cells['low']:7.2f} {cells['flight']:7.2f}"
            )
        rows.append(row)
        print()
    print("target r4hb_scv2                   both         2.67    2.87    3.48    2.49")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows, "members": args.members}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
