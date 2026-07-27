#!/usr/bin/env python3
"""Phase-1 diagnostic: is the si_sdr_mrstft loss's MRSTFT term swamping SI-SDR
and driving DCUNet to over-attenuate? Runs the CONVERGED f1_dcunet_a checkpoint
on SE-valid-drone, per SNR, and reports for each SNR group:

  * the two loss TERM values (si_sdr vs mrstft, as CompositeLoss.last_breakdown),
  * each term's GRADIENT NORM w.r.t. the model output (who actually drives the
    update),
  * the enhanced/target ENERGY RATIO (over-attenuation shows as < 1),
  * SI-SDR (sanity vs the eval CSV).

No training, no writes. Pure measurement.

    python scripts/diagnose_dcunet_loss.py --experiment f1_dcunet_a --valid SE-valid-drone
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from data_processing.collate import frame_collate
from data_processing.frame_datasets import SEValidFrameDataset
from data_processing.frames import get_meta
from losses._common import get_tensor
from metrics.separation import si_sdr
from training.config import build_losses, build_task_and_codec, instantiate_model

SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="f1_dcunet_a")
    ap.add_argument("--valid", default="SE-valid-drone")
    ap.add_argument("--model-cfg", default="conf/model/a1_baseline_dcunet.yaml")
    ap.add_argument("--loss-cfg", default="conf/loss/si_sdr_mrstft.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--per-snr", type=int, default=16, help="clips per SNR group")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = OmegaConf.load(args.model_cfg)
    _, codec = build_task_and_codec(model_cfg)
    model = instantiate_model(model_cfg).to(device)
    ckpt = args.checkpoint or f"results/{args.experiment}/best.ckpt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    composite = build_losses(OmegaConf.load(args.loss_cfg)).to(device)

    ds = SEValidFrameDataset(args.valid, sample_rate=SR)
    by_snr: dict[float, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        snr = float(get_meta(ds[i], "input_snr"))
        if len(by_snr[snr]) < args.per_snr:
            by_snr[snr].append(i)

    print(f"# {args.experiment} on {args.valid} — loss={args.loss_cfg}")
    print(
        f"{'SNR':>5} | {'si_sdr_term':>11} {'mrstft_term':>11} {'mrstft/sisdr':>12} | "
        f"{'|grad|_sisdr':>12} {'|grad|_mrstft':>13} {'grad_ratio':>10} | "
        f"{'E_enh/E_tgt':>11} | {'SI-SDR':>7}"
    )
    for snr in sorted(by_snr):
        batch = frame_collate([ds[i] for i in by_snr[snr]]).map_data(lambda t: t.to(device))
        inputs = codec.to_inputs(batch)
        enhanced = codec.call_model(model, inputs)  # differentiable
        if not torch.is_tensor(enhanced):
            enhanced = get_tensor(enhanced, "enhanced")
        enhanced = enhanced.reshape(len(by_snr[snr]), -1)
        enhanced.requires_grad_(True)
        pred_frame = codec.to_frame(enhanced, batch)

        # per-term values + gradient norms w.r.t. the model output
        term_vals, term_grads = {}, {}
        for name in composite._order:  # noqa: SLF001 - diagnostic introspection
            fn = (
                composite._loss_modules[name]  # noqa: SLF001
                if name in composite._loss_modules  # noqa: SLF001
                else composite._plain[name]  # noqa: SLF001
            )
            v = fn(pred_frame, batch) * composite._weights[name]  # noqa: SLF001
            term_vals[name] = float(v.detach())
            g = torch.autograd.grad(v, enhanced, retain_graph=True)[0]
            term_grads[name] = float(g.norm())

        with torch.no_grad():
            est = enhanced.detach()
            tgt = get_tensor(batch, "target").reshape(len(by_snr[snr]), -1)
            n = min(est.shape[-1], tgt.shape[-1])
            e_ratio = float(
                (est[..., :n].pow(2).mean() / tgt[..., :n].pow(2).mean().clamp_min(1e-12)).sqrt()
            )
            sisdr_val = float(
                np.mean(
                    [
                        si_sdr(tgt[j, :n].cpu().numpy()[None], est[j, :n].cpu().numpy()[None])
                        for j in range(est.shape[0])
                    ]
                )
            )

        si_v = term_vals.get("si_sdr", float("nan"))
        mr_v = term_vals.get("mrstft", float("nan"))
        gsi = term_grads.get("si_sdr", float("nan"))
        gmr = term_grads.get("mrstft", float("nan"))
        print(
            f"{snr:5.0f} | {si_v:11.3f} {mr_v:11.3f} {mr_v / max(abs(si_v), 1e-9):12.1f} | "
            f"{gsi:12.4f} {gmr:13.4f} {gmr / max(gsi, 1e-9):10.1f} | "
            f"{e_ratio:11.3f} | {sisdr_val:7.2f}"
        )


if __name__ == "__main__":
    main()
