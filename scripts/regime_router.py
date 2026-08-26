"""Route each frame to the synthetic-only arm that is best in its regime.

No single synthetic-only arm holds more than one cell of the rig-by-regime grid,
and across all 14 arms the ramp and cruise cells are anti-correlated at Spearman
-0.58: an arm that learns to read slow rotors gives up cruise, and one with
cruise precision cannot read a ramp. The target sits inside that whole frontier,
so the trade-off belongs to our synthetic streams rather than to the task.

Until an arm breaks the frontier, a router over the arms that already exist is
the honest synthetic-only system. Every specialist here was trained on synthetic
audio alone, so the ensemble is still real-audio-free. Weighting the campaign's
per-cell bests by their frame counts gives 3.73 rev/s against the target's 2.67,
a ratio of 1.39 where the best SINGLE arm is 3.02 — worth measuring for real.

The router has to decide the regime from the audio, which is the part that can
fail. Two numbers are reported for exactly that reason:

    oracle   regime taken from the ground-truth rotor track — the ceiling
    routed   regime inferred from the specialists' own predictions — the system

The gap between them is what regime confusion costs. A `single` row per
specialist is printed too, so the router is never credited with a gain that one
model already had.
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

#: regime -> the arm that holds that cell on the frozen split.
DEFAULT_ROUTE = {
    "zero": "m3abl_comb_unigru128_s1",
    "low": "stoch_s1s_both",
    "flight": "stoch_s1h_scv2",
}


def infer_regimes(preds: dict[str, np.ndarray]) -> np.ndarray:
    """Per-frame regime from the specialists' own predictions.

    The median across specialists, not any one of them: the zero specialist
    reads its own comb-less floor as about 39 rev/s, so letting it vote alone
    would put ramp frames in the cruise bin. The thresholds are the evaluation's
    own (max rotor < 1 is zero, mean < 45 is low).
    """
    stack = np.stack(list(preds.values()))  # (model, rotor, frame)
    med = np.median(stack, axis=0)
    labels = np.full(med.shape[1], "low", dtype=object)
    labels[med.max(axis=0) < 1.0] = "zero"
    labels[med.mean(axis=0) >= 45.0] = "flight"
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", nargs="*", default=None,
                    help="regime=experiment pairs; default is the campaign's per-cell bests")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    route = dict(DEFAULT_ROUTE)
    for pair in args.route or []:
        k, _, v = pair.partition("=")
        route[k] = v

    import torch

    import zoo
    from data_processing.frame_datasets import DregonLMFrameDataset

    names = sorted(set(route.values()))
    models = {n: zoo.load(n, ckpt="best", device="cpu") for n in names}
    rigs = clip_rigs()

    dataset = DregonLMFrameDataset(
        data_dir=VALID, n_fft=2048, hop_length=512, sample_rate=16000,
        flatten_channels=False,
    )
    n_clips = len(dataset) if args.limit is None else min(len(dataset), args.limit)

    # err[key][rig][regime] -> list of per-frame absolute errors
    keys = ["oracle", "routed", *[f"single:{n}" for n in names]]
    err = {k: {r: {g: [] for g in REGIMES} for r in RIGS} for k in keys}
    confusion = np.zeros((len(REGIMES), len(REGIMES)), dtype=np.int64)

    for i in range(n_clips):
        frame = dataset[i]
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        rig = rigs[i] if i < len(rigs) else "dregon"
        audio = np.asarray(frame["mixture"].data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None]
        for ch in range(min(args.channels, audio.shape[0])):
            import tdseries as td

            from data_processing.frames import audio_series

            one = td.Frame({"mixture": audio_series(audio[ch][None], 16000)})
            preds = {}
            for n, m in models.items():
                with torch.no_grad():
                    preds[n] = np.asarray(m(one)["rps_pred"].data, dtype=np.float64)
            width = min(min(p.shape[1] for p in preds.values()), target.shape[1])
            tgt = target[:, :width]
            preds = {n: p[:, :width] for n, p in preds.items()}

            true_lab = frame_regimes(tgt)
            got_lab = infer_regimes(preds)
            for a, ra in enumerate(REGIMES):
                for b, rb in enumerate(REGIMES):
                    confusion[a, b] += int(((true_lab == ra) & (got_lab == rb)).sum())

            errs = {n: pit_abs_error(p, tgt) for n, p in preds.items()}
            for n in names:
                for regime in REGIMES:
                    m = true_lab == regime
                    if m.any():
                        err[f"single:{n}"][rig][regime].append(errs[n][:, m].ravel())
            # oracle: the true regime picks the specialist.
            # routed: the inferred regime picks it; scored in the TRUE regime's
            # cell, so a misroute shows up where it actually happened.
            for regime in REGIMES:
                m = true_lab == regime
                if not m.any():
                    continue
                err["oracle"][rig][regime].append(errs[route[regime]][:, m].ravel())
                picked = np.empty((tgt.shape[0], int(m.sum())))
                sub = got_lab[m]
                for g in REGIMES:
                    sel = sub == g
                    if sel.any():
                        picked[:, sel] = errs[route[g]][:, m][:, sel]
                err["routed"][rig][regime].append(picked.ravel())

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
                v
                for r in ([*RIGS] if rig == "both" else [rig])
                for regime in REGIMES
                for v in err[key][r][regime]
            ]
            allm = float(np.concatenate(pooled).mean()) if pooled else float("nan")
            row[rig] = {"all": allm, **cells}
            print(
                f"{key if rig == RIGS[0] else '':34s} {rig:9s} {allm:7.2f} "
                f"{cells['zero']:7.2f} {cells['low']:7.2f} {cells['flight']:7.2f}"
            )
        rows.append(row)
        print()

    print("regime confusion (rows = true, cols = inferred):")
    print(f"{'':9s}" + "".join(f"{g:>9s}" for g in REGIMES))
    for a, ra in enumerate(REGIMES):
        tot = max(confusion[a].sum(), 1)
        print(f"{ra:9s}" + "".join(f"{100 * confusion[a, b] / tot:8.1f}%" for b in range(len(REGIMES))))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"rows": rows, "confusion": confusion.tolist(), "route": route}, indent=1)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
