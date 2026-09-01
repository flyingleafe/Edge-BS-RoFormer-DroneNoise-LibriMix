#!/usr/bin/env python
"""Peak GPU memory and step time per (model, loss, batch size) — the batch-size decision.

The salv2 grid fixes ONE batch size across three trunks and two objectives, so
the binding constraint has to be measured rather than guessed. The CRF loss is
the reason: its forward algorithm keeps a ``(B, 2*span+1, G)`` tensor per time
step for the backward pass, and ``span`` is set by the transition band, so its
memory is linear in ``T * span * G`` and is expected to dominate every trunk.

Usage:
    python scripts/probe_batch_memory.py --seconds 4 --batches 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from losses.salience_crf_layers import layer_pit_crf_nll  # noqa: E402
from losses.salience_layers import layer_pit_bce  # noqa: E402
from models.multif0.utils import linear_freq_grid  # noqa: E402
from models.salience_crf import band_for_rev_s, gaussian_layer_target  # noqa: E402
from training.config import instantiate_model  # noqa: E402

SR, HOP, R, G = 16000, 512, 4, 300
GRID = linear_freq_grid(0.0, 150.0, G)


def _rps(b: int, t: int, device) -> torch.Tensor:
    base = 70.0 + 8.0 * torch.sin(torch.linspace(0, 6.28, t, device=device))
    return (base[None, None] + torch.linspace(-4, 4, R, device=device)[None, :, None]).repeat(
        b, 1, 1
    )


def objective(kind: str, out: torch.Tensor, rps: torch.Tensor) -> torch.Tensor:
    if kind == "pit_mse":  # the regressor: output IS rev/s
        return ((out - rps) ** 2).mean()
    layers = out.reshape(out.shape[0], R, G, out.shape[-1])
    tgt_rps = torch.nn.functional.interpolate(
        rps, size=out.shape[-1], mode="linear", align_corners=False
    )
    if kind == "bce":
        tgt = gaussian_layer_target(tgt_rps, GRID, sigma_bins=1.0)
        return layer_pit_bce(layers, tgt, focus=120.0)
    step = float(GRID[1] - GRID[0])
    span, pen = band_for_rev_s(float(kind.split(":")[1]), step)
    gold = torch.round((tgt_rps - float(GRID[0])) / step).clamp(0, G - 1).long()
    return layer_pit_crf_nll(torch.nn.functional.logsigmoid(layers), gold, span, pen)


def probe(model_cfg: str, kind: str, b: int, secs: float, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = instantiate_model(OmegaConf.load(model_cfg)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n = int(SR * secs)
    audio = torch.randn(b, n, device=device)
    rps = _rps(b, n // HOP + 1, device)
    t0 = 0.0
    for it in range(3):
        opt.zero_grad(set_to_none=True)
        loss = objective(kind, model(audio), rps)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        if it == 0:
            t0 = time.time()
    dt = (time.time() - t0) / 2.0
    peak = torch.cuda.max_memory_allocated() / 2**30
    params = sum(p.numel() for p in model.parameters()) / 1e6
    del model, opt, audio, rps
    return {"peak_gib": round(peak, 2), "step_s": round(dt, 3), "params_m": round(params, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--out", default="results/probe_batch_memory.json")
    a = ap.parse_args()

    cells = [
        ("conf/model/harmof0_rps_l4.yaml", "harmof0", ["bce", "crf:25", "crf:8"]),
        ("conf/model/hppnet_rps_l4.yaml", "hppnet", ["bce", "crf:25", "crf:8"]),
        ("conf/model/simple_conv_v2.yaml", "scv2", ["pit_mse"]),
    ]
    dev = "cuda"
    print(
        torch.cuda.get_device_name(0),
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
        flush=True,
    )
    rows = []
    for cfg, name, kinds in cells:
        if not Path(cfg).exists():
            print(f"SKIP {name}: {cfg} missing", flush=True)
            continue
        for kind in kinds:
            for b in a.batches:
                try:
                    r = probe(cfg, kind, b, a.seconds, dev)
                    r.update(model=name, loss=kind, batch=b)
                    rows.append(r)
                    print(
                        f"{name:8s} {kind:8s} b={b:3d}  {r['peak_gib']:6.2f} GiB  "
                        f"{r['step_s']:6.3f} s  ({r['params_m']} M)",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"{name:8s} {kind:8s} b={b:3d}  OOM", flush=True)
                    rows.append({"model": name, "loss": kind, "batch": b, "oom": True})
                    torch.cuda.empty_cache()
                    break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
