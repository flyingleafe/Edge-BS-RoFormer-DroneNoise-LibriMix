"""Score checkpoints on HELD-OUT SYNTHETIC data from their own training policy.

The companion to ``scripts/valid_regime_eval.py``, which scores the same
checkpoints on the real frozen split. Every arm in the stochastic-comb campaign
validated on the real split ONLY, so the campaign never measured how well its
models do on the distribution they were actually trained on. Without that
number a poor real score is ambiguous: the model may be failing to learn the
synthetic task at all, or learning it perfectly and failing to transfer. Those
two call for opposite fixes.

The metric, the regime boundaries and the per-frame Hungarian matching are
imported from ``valid_regime_eval`` so the two numbers are directly comparable.

HELD OUT MEANS a different ``base_seed`` on the same policy, so the noise
realizations and rotor trajectories are fresh draws the model never saw. It
does NOT mean a different distribution — that is the point. This measures
in-distribution generalization, and the gap to the real-split score is the
sim-to-real gap.

    python scripts/synth_regime_eval.py --exp stoch_s1id_scv2 --n 240
    python scripts/synth_regime_eval.py --exp a b --policy conf/online_mix/x.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from valid_regime_eval import REGIMES, frame_regimes, pit_abs_error  # noqa: E402

#: The policy each experiment trained on. The eval must use the arm's OWN
#: stream — scoring one arm on another's synthetic data measures transfer
#: between synthetic families, which is a different question.
DEFAULT_POLICY = {
    "stoch_s1id_scv2": "conf/online_mix/stoch_s1id_dload.yaml",
    "stoch_s1id_tr": "conf/online_mix/stoch_s1id_dload.yaml",
    "stoch_s1id_trmed": "conf/online_mix/stoch_s1id_dload.yaml",
    "stoch_s1id_trbig": "conf/online_mix/stoch_s1id_dload.yaml",
    "stoch_s1idv_scv2": "conf/online_mix/stoch_s1idv_dload.yaml",
    "stoch_s1id_fromcomb": "conf/online_mix/stoch_s1id_dload.yaml",
    "stoch_s1s_both": "conf/online_mix/stoch_s1s_dload.yaml",
    "stoch_s1h_scv2": "conf/online_mix/stoch_s1h_dload.yaml",
    # The comb-only arms, scored on the ANALYTIC STATIC COMB they trained on.
    # That distribution is far simpler than the stochastic family — one fixed
    # amplitude profile per clip, comb spacing the only cue — so it is the
    # clean test of whether these models can fit an easy harmonic task at all.
    # If they cannot, the limit is the model, not the data.
    "m3abl_comb_scv2_s1": "conf/online_mix/m3abl_comb_s1_dload.yaml",
    "m3abl_comb_unigru128_s1": "conf/online_mix/m3abl_comb_s1_dload.yaml",
    "m3abl_comb_transformer_s1": "conf/online_mix/m3abl_comb_s1_dload.yaml",
    # The SALIENCE rows on synthetic curricula (conf/experiment/sal150_*.yaml,
    # conf/experiment/salstd_*.yaml). Each is scored on the family it trained
    # on, which is the whole point of those runs.
    "sal150_comb": "conf/online_mix/m3abl_comb_s1_dload.yaml",
    "salstd_comb": "conf/online_mix/m3abl_comb_s1_dload.yaml",
    "sal150_stoch": "conf/online_mix/stoch_s1_dload.yaml",
    "salstd_stoch": "conf/online_mix/stoch_s1_dload.yaml",
    # The real-trained rows have no stochastic policy of their own. They are
    # scored on ARM ID's stream via --policy, as the control that says whether a
    # bad synthetic score means the model is weak or the stream is hard.
}


def build_stream(policy: str, base_seed: int, duration_s: float | None, augment: bool):
    """The arm's own training stream, with two things optionally controlled.

    A naive comparison against the real-split score confounds THREE differences
    at once, and each moves the number a lot:

    1. Chunk length. The policies train on 1 s chunks; the frozen split is 8 s
       clips. A recurrent trunk reading 32 frames is not the same model as one
       reading ~250, so ``duration_s`` must match before the two numbers can be
       subtracted.
    2. Augmentation. The stream fires random gain, spectral recolor, reverb and
       time-warp at probability 1.0. Those are a training device, not a
       property of the synthetic distribution, and the real split has none of
       them. (They are label-CONSISTENT — ``freq_scale`` and the time warp
       rescale the RPS labels with the audio — so leaving them on is not
       cheating, it is just a harder task than the one being compared to.)
    3. The noise family itself, which is the only difference anyone wants to
       measure.

    Passing ``augment=False`` and the real split's ``duration_s`` isolates (3).
    """
    from omegaconf import OmegaConf

    from data_processing.frame_datasets import OnlineMixFrameDataset

    cfg = OmegaConf.load(policy)
    # A fresh base_seed makes every noise realization and rotor trajectory a
    # draw the training run never saw, while leaving the distribution alone.
    cfg.base_seed = int(base_seed)
    if duration_s is not None:
        cfg.duration_s = float(duration_s)
    if not augment:
        for stage in cfg.policy.stages:
            for key in ("augmentations", "noise_augmentations", "noise_time_warp"):
                if key in stage:
                    del stage[key]
    return OnlineMixFrameDataset.from_config(cfg, flatten_channels=True)


def _salience_rps(inner, frame, device: str, threshold: float) -> np.ndarray:
    """``(R, T_stft)`` predicted speeds from a SALIENCE model.

    A salience model's codec emits ``salience``, not ``rps_pred`` — the map is
    turned into speeds by the model's own ``predict_rps`` (sigmoid, then
    segmented Hungarian tracking, then the resample back onto the STFT grid the
    frame's ``rps`` entry lives on). Going through the model rather than the
    codec is what the salience rows of docs/experiments/unified-baseline-eval.md
    do, and it keeps the decode identical to the one that produced them.

    The frames are mono here (``flatten_channels: true``), thus one row.
    """
    import torch

    wav = torch.as_tensor(np.asarray(frame["mixture"].data), dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    pred = inner.predict_rps(wav.to(device), threshold=threshold)
    return np.asarray(pred[0].detach().cpu(), dtype=np.float64)


def score(
    experiment: str,
    policy: str,
    n: int,
    base_seed: int,
    ckpt: str,
    duration_s: float | None = None,
    augment: bool = True,
    device: str = "cpu",
    threshold: float = 0.3,
) -> dict:
    import zoo

    model = zoo.load(experiment, ckpt=ckpt, device=device)
    inner = getattr(model, "model", None)
    salience = bool(getattr(inner, "outputs_salience", False))
    stream = build_stream(policy, base_seed, duration_s, augment)

    errors: dict[str, list[np.ndarray]] = {r: [] for r in REGIMES}
    squared: list[np.ndarray] = []
    seen = 0
    for frame in stream:
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        if salience:
            pred = _salience_rps(inner, frame, device, threshold)
        else:
            pred = np.asarray(model(frame)["rps_pred"].data, dtype=np.float64)
        width = min(pred.shape[1], target.shape[1])
        err = pit_abs_error(pred[:, :width], target[:, :width])
        labels = frame_regimes(target[:, :width])
        squared.append((err**2).mean(axis=0))
        for regime in REGIMES:
            mask = labels == regime
            if mask.any():
                errors[regime].append(err[:, mask].ravel())
        seen += 1
        if seen >= n:
            break

    all_squared = np.concatenate(squared)
    row = {
        "experiment": experiment,
        "ckpt": ckpt,
        "domain": "synthetic",
        "policy": policy,
        "base_seed": base_seed,
        "duration_s": duration_s,
        "augment": augment,
        "salience": salience,
        "n_samples": seen,
        "aggregate_mse": float(all_squared.mean()),
        "all_mae": float(
            np.concatenate([np.concatenate(v) for v in errors.values() if v]).mean()
        ),
        "n_frames": int(all_squared.size),
    }
    for regime in REGIMES:
        vals = np.concatenate(errors[regime]) if errors[regime] else np.array([np.nan])
        row[f"{regime}_mae"] = float(np.mean(vals))
        row[f"{regime}_frames"] = int(vals.size)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", required=True, help="experiment names in the zoo")
    ap.add_argument("--policy", default=None, help="override the policy YAML for every --exp")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--n", type=int, default=240, help="samples drawn from the stream")
    ap.add_argument(
        "--base-seed",
        type=int,
        default=99000,
        help="policy base_seed; must differ from the training policy's to be held out",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="override the policy's duration_s; use 8.0 to match the real frozen split",
    )
    ap.add_argument(
        "--no-augment",
        action="store_true",
        help="strip the training augmentation blocks, which the real split does not have",
    )
    ap.add_argument("--device", default="cpu", help="torch device for the model")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="salience models only: peak threshold passed to predict_rps",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for exp in args.exp:
        policy = args.policy or DEFAULT_POLICY.get(exp)
        if policy is None:
            print(f"{exp}: SKIPPED (no policy known; pass --policy)", flush=True)
            continue
        try:
            row = score(
                exp,
                policy,
                args.n,
                args.base_seed,
                args.ckpt,
                duration_s=args.duration,
                augment=not args.no_augment,
                device=args.device,
                threshold=args.threshold,
            )
        except Exception as exc:  # a missing checkpoint must not kill the batch
            print(f"{exp}: FAILED ({exc!r})", flush=True)
            continue
        rows.append(row)
        print(
            f"{exp:22s} SYNTH d={row['duration_s'] or 'policy'} "
            f"aug={'on' if row['augment'] else 'off'}  aggregate {row['aggregate_mse']:8.2f}  "
            f"all-MAE {row['all_mae']:6.2f}  zero {row['zero_mae']:6.2f}  "
            f"low {row['low_mae']:6.2f}  flight {row['flight_mae']:6.2f}  "
            f"[{row['n_frames']} frames]",
            flush=True,
        )
    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
