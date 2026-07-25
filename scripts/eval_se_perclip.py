#!/usr/bin/env python3
"""Per-CLIP SE scoring on a valid set — the tidy-data backbone for analysis.

Unlike ``scripts/eval_se_models.py`` (which writes per-(category, SNR) *group
means*), this writes ONE ROW PER CLIP with all its metadata (``category``,
``recording_id``, ``input_snr``) alongside the four metrics. That lets a
notebook aggregate *any* subset on the fly — pick a set of categories, a set of
models, per-SNR curves — instead of being locked to whatever grouping the eval
chose. One method per run (a model experiment name, or the ``noisy`` / ``wiener``
anchor), one CSV per (method, valid) so re-evaluating one model is cheap.

    python scripts/eval_se_perclip.py --method f1_dcunet_a --valid SE-valid-harmonic
    python scripts/eval_se_perclip.py --method noisy      --valid SE-valid-drone
    python scripts/eval_se_perclip.py --method wiener     --valid SE-valid-harmonic

Model methods self-fetch ``best.ckpt`` from R2 when absent locally (so a cluster
GPU job is self-sufficient) and can ``--r2-upload`` the result CSV back.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data_processing.frame_datasets import SEValidFrameDataset
from data_processing.frames import get_meta
from metrics.separation import pesq, sdr, si_sdr, stoi

SR = 16000
ANCHORS = ("noisy", "wiener")
# arch (from f1_<arch>_<pass>) -> conf/model config name
ARCH_MODEL = {
    "edge_bs_rof": "a1_edge_bs_rof_fa",
    "dcunet": "a1_baseline_dcunet",
    "tfgridnet": "f1_tfgridnet",
    "mpsenet": "f1_mpsenet",
    "sgmse": "f1_sgmse",
}
METRICS = ["si_sdr", "sdr", "pesq", "pesq_nb", "estoi", "gain_db", "corr"]


def _r2_client():
    """Boto3 client for the ml-data bucket, or None if no creds (mirrors
    eval_se_models). Lets a cluster job self-fetch a ckpt and push results back."""
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
    if Path(ckpt).exists():
        return ckpt
    client = _r2_client()
    if client is None:
        return ckpt
    key = f"artifacts/{exp}/checkpoints/{Path(ckpt).name}"
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    client.download_file("ml-data", key, ckpt)
    print(f"[perclip] fetched {key} from R2 -> {ckpt}", flush=True)
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
    # `gain_db` and `corr` separate the two ways a model can score badly, which
    # (sdr, si_sdr) alone cannot: the pair is consistent with BOTH a near-null
    # estimate and an over-loud one (the quadratic in ||est|| has two roots).
    # A model collapsed toward silence shows gain_db << 0 with sdr pinned at
    # ~0 dB across every input SNR (since ||ref - est||^2 -> ||ref||^2);
    # a model that is merely distorted shows gain_db ~ 0 and low corr.
    ref_e = float(np.sum(ref**2))
    est_e = float(np.sum(est**2))
    eps = 1e-12
    out = {
        "si_sdr": float(si_sdr(ref[None, :], est[None, :])),
        "sdr": float(np.asarray(sdr(ref[None, None, :], est[None, None, :])).reshape(-1)[0]),
        "gain_db": 10.0 * float(np.log10((est_e + eps) / (ref_e + eps))),
        "corr": float(abs(np.dot(ref, est)) / (np.sqrt(ref_e * est_e) + eps)),
    }
    try:
        out["pesq"] = float(pesq(ref, est, SR))
    except Exception:
        out["pesq"] = float("nan")
    try:
        out["estoi"] = float(stoi(ref, est, SR, extended=True))
    except Exception:
        out["estoi"] = float("nan")
    # `pesq` above is WIDEBAND (metrics.separation.pesq picks "wb" at >=16 kHz).
    # The 2023 survey ran everything at 8 kHz, so its PESQ is NARROWBAND, and
    # PESQ-NB scores systematically higher than PESQ-WB on identical audio —
    # enough to account for a whole point. Reporting both makes the comparison
    # against the paper a like-for-like one instead of an apples-to-oranges gap.
    try:
        import librosa

        out["pesq_nb"] = float(
            pesq(
                librosa.resample(ref, orig_sr=SR, target_sr=8000),
                librosa.resample(est, orig_sr=SR, target_sr=8000),
                8000,  # already at 8 kHz -> the wrapper keeps it there and picks "nb"
            )
        )
    except Exception:
        out["pesq_nb"] = float("nan")
    return out


def _estimates_model(
    method: str,
    ds,
    idxs,
    batch_size: int,
    device,
    model_cfg_path: str | None = None,
    ckpt_path: str | None = None,
) -> list[np.ndarray]:
    """Run the model forward over the selected clips, return per-clip estimates.

    ``model_cfg_path``/``ckpt_path`` override the ARCH_MODEL lookup and the
    default checkpoint path — needed for custom experiment names (e.g.
    ``f1_dcunet_a_lossA``) whose arch can't be parsed from the name."""
    cfg_name = model_cfg_path or f"conf/model/{ARCH_MODEL[_arch_of(method)]}.yaml"
    model_cfg = OmegaConf.load(cfg_name)
    if device.type == "cpu":
        with contextlib.suppress(Exception):
            model_cfg.params.config.model.flash_attn = False
    from training.config import build_task_and_codec, instantiate_model

    _, codec = build_task_and_codec(model_cfg)
    model = instantiate_model(model_cfg).to(device)
    ckpt = _maybe_fetch_ckpt(method, ckpt_path or f"results/{method}/best.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    mixes = [torch.as_tensor(np.asarray(ds[i]["mixture"].data), dtype=torch.float32) for i in idxs]
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(mixes), batch_size):
            batch = torch.stack(mixes[start : start + batch_size]).to(device)  # (b, T)
            est = codec.call_model(model, {"mixture": batch})
            est = np.asarray(est.detach().cpu()).reshape(len(batch), -1)
            out.extend(est[j] for j in range(est.shape[0]))
    return out


def _estimates_anchor(method: str, ds, idxs) -> list[np.ndarray]:
    from scipy.signal import wiener

    out: list[np.ndarray] = []
    for i in idxs:
        mix = np.asarray(ds[i]["mixture"].data, np.float32).reshape(-1)
        if method == "noisy":
            out.append(mix)
        else:  # wiener
            w = wiener(mix).astype(np.float32)
            out.append(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="f1_<arch>_<a|b> | noisy | wiener")
    ap.add_argument("--valid", required=True, help="SE-valid-drone | SE-valid-harmonic")
    ap.add_argument("--limit", type=int, default=0, help="cap n clips (0=all; quick checks)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--r2-upload", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=leave default)")
    ap.add_argument(
        "--model-cfg", default=None, help="override conf/model/*.yaml (custom exp names)"
    )
    ap.add_argument("--checkpoint", default=None, help="override the checkpoint path")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SEValidFrameDataset(args.valid, sample_rate=SR)
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    idxs = list(range(n))

    if args.method in ANCHORS:
        ests = _estimates_anchor(args.method, ds, idxs)
    else:
        ests = _estimates_model(
            args.method, ds, idxs, args.batch, device, args.model_cfg, args.checkpoint
        )

    out_path = Path(args.out or f"results/f1_perclip/{args.method}__{args.valid}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "valid", "clip_id", "category", "input_snr", *METRICS])
        for i, est in zip(idxs, ests):
            tgt = np.asarray(ds[i]["target"].data, np.float32).reshape(-1)
            m = _metrics(tgt, est)
            w.writerow(
                [
                    args.method,
                    args.valid,
                    str(get_meta(ds[i], "id", i)),
                    str(get_meta(ds[i], "category", "all")),
                    float(get_meta(ds[i], "input_snr")),
                    *[f"{m[k]:.4f}" for k in METRICS],
                ]
            )
    print(f"wrote {out_path} ({n} clips)", flush=True)
    if args.r2_upload:
        client = _r2_client()
        if client is not None:
            key = f"artifacts/{args.method}/perclip/{out_path.name}"
            client.upload_file(str(out_path), "ml-data", key)
            print(f"[perclip] uploaded -> R2 {key}", flush=True)


if __name__ == "__main__":
    main()
