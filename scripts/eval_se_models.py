#!/usr/bin/env python3
"""Score an F1 SE-baseline checkpoint on a valid set, per (category, SNR).

Companion to ``scripts/eval_se_anchors.py`` (identical CSV shape) so the model
rows and the noisy/Wiener anchors assemble into one table. Uses the same
model+codec machinery as ``eval.py`` (which stays THE standard eval entry
point) but groups by category as well as input_snr and writes a tidy per-model
CSV under ``results/f1_eval/`` — what ``eval.py`` cannot do in one pass (it
overwrites one eval/ dir per experiment and groups by SNR only). For SGMSE+ the
codec calls ``model(x)`` with no target → the reverse-SDE sampler runs → enhanced.

    python scripts/eval_se_models.py --experiment f1_dcunet_a --valid SE-valid-drone
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data_processing.frame_datasets import SEValidFrameDataset
from data_processing.frames import get_meta
from metrics.separation import pesq, sdr, si_sdr, stoi
from training.config import build_task_and_codec, instantiate_model

SR = 16000
# arch (from f1_<arch>_<pass>) -> conf/model config name
ARCH_MODEL = {
    "edge_bs_rof": "a1_edge_bs_rof_fa",
    "dcunet": "a1_baseline_dcunet",
    "tfgridnet": "f1_tfgridnet",
    "mpsenet": "f1_mpsenet",
    "sgmse": "f1_sgmse",
}


def _r2_client():
    """Boto3 client for the ml-data bucket, or None if no creds. Lets a cluster
    eval job self-fetch a ckpt uploaded by training and push its result CSV back
    (omnirun pull is unreliable). Creds ship via .env with omnirun jobs."""
    acct = os.environ.get("R2_ACCOUNT_ID")
    if not acct and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        acct = os.environ.get("R2_ACCOUNT_ID")
    if not acct:
        return None
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


def _maybe_fetch_ckpt(exp: str, ckpt: str) -> str:
    """If ckpt is missing locally, download artifacts/<exp>/checkpoints/<name>
    from R2 (its basename). No-op when the file already exists."""
    if Path(ckpt).exists():
        return ckpt
    client = _r2_client()
    if client is None:
        return ckpt
    key = f"artifacts/{exp}/checkpoints/{Path(ckpt).name}"
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    client.download_file("ml-data", key, ckpt)
    print(f"[eval] fetched {key} from R2 -> {ckpt}", flush=True)
    return ckpt


def _arch_of(experiment: str) -> str:
    body = experiment[len("f1_") :] if experiment.startswith("f1_") else experiment
    body = body.rsplit("_", 1)[0]  # strip trailing _a/_b
    if body not in ARCH_MODEL:
        raise ValueError(f"cannot map experiment {experiment!r} -> arch (got {body!r})")
    return body


def _metrics(ref: np.ndarray, est: np.ndarray) -> dict[str, float]:
    ref = np.asarray(ref, np.float32).reshape(-1)
    est = np.asarray(est, np.float32).reshape(-1)
    n = min(ref.shape[0], est.shape[0])
    ref, est = ref[:n], est[:n]
    out = {
        "si_sdr": float(si_sdr(ref[None, :], est[None, :])),
        "sdr": float(np.asarray(sdr(ref[None, None, :], est[None, None, :])).reshape(-1)[0]),
    }
    try:
        out["pesq"] = float(pesq(ref, est, SR))
    except Exception:
        out["pesq"] = float("nan")
    try:
        out["estoi"] = float(stoi(ref, est, SR, extended=True))
    except Exception:
        out["estoi"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--valid", required=True, help="SE-valid-drone | SE-valid-harmonic")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--by-category", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap n clips (0=all; for quick checks)")
    ap.add_argument("--batch", type=int, default=8, help="forward batch size")
    ap.add_argument(
        "--per-snr",
        type=int,
        default=0,
        help="balanced subsample: cap clips per (category, SNR) group (0=all). Means are "
        "stable at ~25; use to keep CPU eval of heavy models tractable.",
    )
    ap.add_argument(
        "--r2-upload",
        action="store_true",
        help="upload the result CSV to R2 artifacts/<exp>/eval/<basename> after writing "
        "(so a cluster GPU eval job's output survives the broken omnirun pull).",
    )
    args = ap.parse_args()

    model_cfg = OmegaConf.load(f"conf/model/{ARCH_MODEL[_arch_of(args.experiment)]}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # flash-attention SDPA has no CPU backend; fall back to standard attention
        # for CPU eval (the trained weights are attention-backend-agnostic).
        with contextlib.suppress(Exception):
            model_cfg.params.config.model.flash_attn = False
    _task, codec = build_task_and_codec(model_cfg)
    model = instantiate_model(model_cfg).to(device)
    ckpt = _maybe_fetch_ckpt(
        args.experiment, args.checkpoint or f"results/{args.experiment}/best.ckpt"
    )
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    ds = SEValidFrameDataset(args.valid, sample_rate=SR)
    n_iter = len(ds) if not args.limit else min(args.limit, len(ds))
    # Batch the (slow on CPU) model forward; valid clips are all the same length.
    mixes = [
        torch.as_tensor(np.asarray(ds[i]["mixture"].data), dtype=torch.float32)
        for i in range(n_iter)
    ]
    tgts = [np.asarray(ds[i]["target"].data, np.float32).reshape(-1) for i in range(n_iter)]
    keys = [
        (
            str(get_meta(ds[i], "category", "all")) if args.by_category else "all",
            float(get_meta(ds[i], "input_snr")),
        )
        for i in range(n_iter)
    ]
    if args.per_snr:  # balanced subsample: keep the first per_snr clips of each group
        seen: dict[tuple, int] = defaultdict(int)
        keep = []
        for i, k in enumerate(keys):
            if seen[k] < args.per_snr:
                seen[k] += 1
                keep.append(i)
        mixes = [mixes[i] for i in keep]
        tgts = [tgts[i] for i in keep]
        keys = [keys[i] for i in keep]
        n_iter = len(keep)
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for start in range(0, n_iter, args.batch):
            batch = torch.stack(mixes[start : start + args.batch]).to(device)  # (b, T)
            out = codec.call_model(model, {"mixture": batch})
            out = np.asarray(out.detach().cpu())
            out = out.reshape(out.shape[0], -1)  # (b, T)
            for j in range(out.shape[0]):
                for m, v in _metrics(tgts[start + j], out[j]).items():
                    acc[keys[start + j]][m].append(v)

    metrics = ["si_sdr", "sdr", "pesq", "estoi"]
    out_path = Path(args.out or f"results/f1_eval/{args.experiment}__{args.valid}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "input_snr", "method", "n", *metrics])
        for cat, snr in sorted(acc):
            vals = acc[(cat, snr)]
            n = len(vals["si_sdr"])
            row = [cat, snr, args.experiment, n]
            row += [f"{float(np.nanmean(vals[m])):.4f}" if vals[m] else "nan" for m in metrics]
            w.writerow(row)
    print(f"wrote {out_path} ({len(acc)} groups)")
    # Echo the CSV to stdout so the result is recoverable from `omnirun logs`
    # even without R2 (and readable at a glance).
    print("----- CSV -----\n" + out_path.read_text() + "----- END CSV -----", flush=True)
    if args.r2_upload:
        client = _r2_client()
        if client is not None:
            key = f"artifacts/{args.experiment}/eval/{out_path.name}"
            client.upload_file(str(out_path), "ml-data", key)
            print(f"[eval] uploaded result -> R2 {key}", flush=True)


if __name__ == "__main__":
    main()
