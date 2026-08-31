"""Verification probe for `models.harmonic_ports.hft_rps.HFTRPS`.

Four checks, in the order a port must pass them: shapes, gradient flow, an
overfit on a handful of analytic comb clips, and peak GPU memory + step time at
the training configuration. Run it on a GPU to get (4); (1)-(3) run anywhere.

    python scripts/hft_rps_probe.py --device cuda --steps 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from losses.salience import auto_pos_weight, salience_bce_loss  # noqa: E402
from models.harmonic_ports.hft_rps import HFTRPS  # noqa: E402
from models.multif0.utils import salience_target_from_resampled_rps  # noqa: E402


def comb_clip(rates: np.ndarray, n: int, sr: int = 16000, k_max: int = 40,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """One analytic static comb per rotor, plus white noise."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(n) / sr
    x = 0.05 * rng.standard_normal(n)
    for r in rates:
        for k in range(1, k_max + 1):
            f = k * r
            if f >= 7500.0:
                break
            x += (1.0 / k) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def make_batch(n_clips: int, dur_s: float, sr: int, seed: int):
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    audio, rps = [], []
    for _ in range(n_clips):
        base = rng.uniform(45.0, 95.0)
        rates = base + rng.uniform(-6.0, 6.0, size=4)
        audio.append(comb_clip(rates, n, sr, rng=rng))
        rps.append(np.repeat(rates[:, None], 64, axis=1))
    return torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(rps)).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="rows in the memory probe")
    ap.add_argument("--dur", type=float, default=1.0)
    ap.add_argument("--valid-dur", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="{}", help="JSON kwargs for HFTRPS")
    args = ap.parse_args()

    dev = torch.device(args.device)
    kw = json.loads(args.model)
    torch.manual_seed(0)
    model = HFTRPS(**kw).to(dev)
    report: dict = {"device": args.device, "model_kwargs": kw}
    report["params"] = int(sum(p.numel() for p in model.parameters()))

    # (1) SHAPES
    n = int(args.dur * 16000)
    x = torch.randn(2, n, device=dev)
    with torch.no_grad():
        y = model(x, return_attention=True)
    report["shape"] = {
        "logits": list(y.shape),
        "expected_T": model.num_grid_frames(n),
        "attention": list(model.last_attention.shape),
        "grid_step_revs": float(model.output_freqs()[1] - model.output_freqs()[0]),
        "finite": bool(torch.isfinite(y).all()),
    }
    model.last_attention = None

    # (2) GRADIENT FLOW
    tgt = torch.zeros_like(y)
    tgt[:, 120, :] = 1.0
    loss = salience_bce_loss(model(x), tgt)
    loss.backward()
    report["grad"] = {
        "no_grad": [k for k, p in model.named_parameters() if p.grad is None],
        "nonfinite": [
            k for k, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ],
        "zero": [
            k for k, p in model.named_parameters()
            if p.grad is not None and float(p.grad.abs().sum()) == 0.0
        ],
    }
    model.zero_grad(set_to_none=True)

    # (3) OVERFIT a handful of analytic comb clips
    audio, rps = make_batch(args.clips, args.dur, 16000, seed=7)
    audio, rps = audio.to(dev), rps.to(dev)
    freqs = model.output_freqs()
    pw = auto_pos_weight(len(freqs), 4, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    curve = []
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        logits = model(audio)
        rps_g = torch.nn.functional.interpolate(rps, size=logits.shape[-1], mode="linear",
                                                align_corners=False)
        target = salience_target_from_resampled_rps(rps_g, freqs, blur_bins=1)
        loss = salience_bce_loss(logits, target, pos_weight=pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            curve.append((step, float(loss)))
    model.eval()
    with torch.no_grad():
        pred = model.predict_rps(audio, chunk_size=0)
    # per-frame permutation-invariant absolute error, the campaign's metric
    from itertools import permutations
    err = []
    p_np, t_np = pred.cpu().numpy(), rps.cpu().numpy()
    width = min(p_np.shape[-1], t_np.shape[-1])
    for b in range(p_np.shape[0]):
        best = None
        for perm in permutations(range(4)):
            e = np.abs(p_np[b, list(perm), :width] - t_np[b, :, :width]).mean()
            best = e if best is None else min(best, e)
        err.append(best)
    report["overfit"] = {
        "curve": curve,
        "loss_first": curve[0][1],
        "loss_last": curve[-1][1],
        "pit_mae_revs": float(np.mean(err)),
        "seconds": time.time() - t0,
        "steps": args.steps,
    }

    # (4) MEMORY / STEP TIME at the training and validation shapes
    if dev.type == "cuda":
        mem = {}
        for tag, rows, dur in (("train", args.batch, args.dur),
                               ("valid", 8, args.valid_dur)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            xb = torch.randn(rows, int(dur * 16000), device=dev)
            if tag == "train":
                model.train()
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                salience_bce_loss(out, torch.zeros_like(out)).backward()
                opt.step()
                torch.cuda.synchronize()
                t1 = time.time()
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    out = model(xb)
                    salience_bce_loss(out, torch.zeros_like(out)).backward()
                    opt.step()
                torch.cuda.synchronize()
                mem[f"{tag}_step_s"] = (time.time() - t1) / 3
            else:
                model.eval()
                with torch.no_grad():
                    torch.cuda.synchronize()
                    t1 = time.time()
                    model(xb)
                    torch.cuda.synchronize()
                    mem[f"{tag}_step_s"] = time.time() - t1
            mem[f"{tag}_rows"] = rows
            mem[f"{tag}_dur_s"] = dur
            mem[f"{tag}_peak_GB"] = torch.cuda.max_memory_allocated() / 2**30
        mem["device_name"] = torch.cuda.get_device_name(0)
        mem["device_total_GB"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        report["memory"] = mem

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
