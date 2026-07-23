#!/usr/bin/env python3
"""Bespoke score-matching training loop for the SGMSE+ F1 baseline.

The framework's ``train.py`` loop is discriminative (same forward train/eval,
Series outputs, cheap validation) and cannot host score-based diffusion, so
SGMSE+ trains here instead: the model's ``forward(mix, target)`` returns the
sigma^2-weighted DSM scalar loss (one cheap score-net eval), we backprop that,
and EMA-track weights. Periodic validation runs the (expensive) PC sampler on a
small SE-valid-drone subset for a rough SI-SDR signal; checkpoints are saved as
a plain ``model.state_dict()`` to ``results/<exp>/{best,last}.ckpt`` — the exact
format ``eval.py`` loads, so FINAL scoring stays on the single eval entry point
(``python eval.py experiment=f1_sgmse_a metrics=separation_full``; the SE codec
calls ``model(x)`` with no target → the model samples → enhanced).

    python scripts/train_sgmse.py --experiment f1_sgmse_a --steps 200000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_processing.collate import frame_collate
from data_processing.frame_datasets import OnlineMixFrameDataset, SEValidFrameDataset
from losses._common import get_tensor
from metrics.separation import si_sdr
from training.config import instantiate_model
from utils.paths import get_data_root

PASS = {"a": ("se_drone_only", "SE-valid-drone"), "b": ("se_all_harmonic", "SE-valid-drone")}


def _stream(policy_name: str, local_speech: bool):
    cfg = cast(
        dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.load(f"conf/online_mix/{policy_name}.yaml"), resolve=False
        ),
    )
    if local_speech:
        root = get_data_root() / "data" / "librispeech" / "LibriSpeech" / "train-clean-100"
        cfg["sources"]["speech"] = [
            {
                "kind": "audio_files",
                "root": str(root),
                "glob": "**/*.flac",
                "exclude": cfg["sources"]["speech"][0]["exclude"],
                "cache": {"mode": "none"},
            }
        ]
    return OnlineMixFrameDataset.from_config(cfg)


def _r2_client():
    """Boto3 client for the checkpoint bucket (ml-data), or None if no creds.

    Creds ship with every omnirun job via .env; fall back to parsing .env so a
    bare run still works. Periodic upload is what lets a cancelled job's progress
    survive — ``omnirun pull`` is unreliable and best/last.ckpt only live in the
    job's ``results/`` otherwise.
    """
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


def _r2_upload(client, exp: str, local: Path, name: str) -> None:
    if client is None:
        return
    try:
        client.upload_file(str(local), "ml-data", f"artifacts/{exp}/checkpoints/{name}")
        print(f"[sgmse] uploaded {name} -> R2", flush=True)
    except Exception as e:  # noqa: BLE001 - upload is best-effort
        print(f"[sgmse] R2 upload failed ({name}): {e}", flush=True)


def _r2_resume(client, exp: str, run_dir: Path, model, device) -> bool:
    if client is None:
        return False
    key = f"artifacts/{exp}/checkpoints/last.ckpt"
    try:
        client.head_object(Bucket="ml-data", Key=key)
    except Exception:  # noqa: BLE001 - no prior ckpt is the normal cold-start case
        return False
    dst = run_dir / "last.ckpt"
    client.download_file("ml-data", key, str(dst))
    model.load_state_dict(torch.load(dst, map_location=device, weights_only=True))
    print("[sgmse] resumed from R2 last.ckpt", flush=True)
    return True


def _sisdr(model, vds, n, device) -> float:
    model.eval()
    idxs = list(range(0, len(vds), max(1, len(vds) // n)))[:n]
    scores = []
    with torch.no_grad():
        for i in idxs:
            fr = vds[i]
            mix = torch.as_tensor(np.asarray(fr["mixture"].data), dtype=torch.float32)
            tgt = np.asarray(fr["target"].data, dtype=np.float32)
            enh = model(mix[None, None, :].to(device)).squeeze().cpu().numpy()
            scores.append(float(si_sdr(tgt[None, :], enh[None, : tgt.shape[0]])))
    model.train()
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="f1_sgmse_a | f1_sgmse_b")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--val-samples", type=int, default=12)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--local-speech", action="store_true")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="wall-clock cap (0=off)")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="cold-start even if an R2 last.ckpt exists (default: resume so chunked "
        "wall-clock-limited jobs accumulate weights across restarts).",
    )
    args = ap.parse_args()

    p = args.experiment.rsplit("_", 1)[-1]
    policy, valid_name = PASS[p]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate_model(OmegaConf.load("conf/model/f1_sgmse.yaml")).to(device)
    n_params = sum(pp.numel() for pp in [model] for pp in pp.parameters())
    print(f"[sgmse] {args.experiment}: {n_params / 1e6:.1f}M params on {device}", flush=True)

    loader = DataLoader(
        _stream(policy, args.local_speech),
        batch_size=args.batch,
        num_workers=args.num_workers,
        collate_fn=frame_collate,
        drop_last=True,
    )
    vds = SEValidFrameDataset(valid_name, sample_rate=16000)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path(args.results_root) / args.experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    r2 = _r2_client()
    if not args.no_resume:
        _r2_resume(r2, args.experiment, run_dir, model, device)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="se-baselines", name=args.experiment, config=vars(args))

    best = -1e9
    t0 = time.time()
    model.train()
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.map_data(lambda t: t.to(device))
        mix = get_tensor(batch, "mixture")
        tgt = get_tensor(batch, "target")
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
            tgt = tgt.unsqueeze(1)
        loss = model(mix, target=tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % 50 == 0:
            print(
                f"[sgmse] step {step} loss {float(loss):.4f} ({step / (time.time() - t0):.2f} it/s)",
                flush=True,
            )
            if run:
                run.log({"train/loss": float(loss)}, step=step)
        if step % args.val_every == 0 or step == args.steps:
            sisdr = _sisdr(model, vds, args.val_samples, device)
            print(f"[sgmse] step {step} val/si_sdr {sisdr:.3f}", flush=True)
            if run:
                run.log({"val/si_sdr": sisdr}, step=step)
            torch.save(model.state_dict(), run_dir / "last.ckpt")
            _r2_upload(r2, args.experiment, run_dir / "last.ckpt", "last.ckpt")
            if sisdr > best:
                best = sisdr
                torch.save(model.state_dict(), run_dir / "best.ckpt")
                _r2_upload(r2, args.experiment, run_dir / "best.ckpt", "best.ckpt")
                print(f"[sgmse] new best si_sdr {best:.3f} -> best.ckpt", flush=True)
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[sgmse] wall-clock cap hit at step {step}", flush=True)
            torch.save(model.state_dict(), run_dir / "last.ckpt")
            _r2_upload(r2, args.experiment, run_dir / "last.ckpt", "last.ckpt")
            break
    print("[sgmse] done", flush=True)
    torch.save(model.state_dict(), run_dir / "last.ckpt")
    _r2_upload(r2, args.experiment, run_dir / "last.ckpt", "last.ckpt")


if __name__ == "__main__":
    main()
