"""Benchmark the position-aware harmonic noise generator's forward/backward.

No data needed — RPS and geometry are synthetic — so it runs anywhere (laptop,
Slurm/Colab/Kaggle GPU). Times a batch-32, 1 s forward and forward+backward,
reporting the median over ``--iters`` steps.

Usage::

    .venv/bin/python scripts/bench_noise_gen_forward.py --device cpu --iters 5
    .venv/bin/python scripts/bench_noise_gen_forward.py --device cuda --iters 30

Baseline (pre-optimization, from the task prompt): a batch-32 CPU/T4/P100 step
took ~2.9-4.5 s. On this machine's CPU the optimized forward is ~1.9-2.0 s (the
col2im/exp/scan hotspots removed); GPU should land in the tens-of-ms range.
"""

from __future__ import annotations

import argparse
import time

import torch

from models.registry import build_noise_gen_model

SR = 16000


def make_inputs(batch: int, rotors: int, mics: int, seconds: float, device: torch.device):
    """Synthetic per-rotor RPS + a plausible quad geometry (no dataset needed)."""
    t = int(round(seconds * SR))
    torch.manual_seed(0)
    rps = (torch.rand(batch, rotors, t) * 40 + 60).to(device)  # 60-100 Hz
    rotor = torch.tensor([[0.2, 0.2, 0.0], [-0.2, 0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]])[
        :rotors
    ]
    mic = torch.randn(mics, 3) * 0.03
    rel = (mic[None, :, None, :] - rotor[None, None, :, :]).expand(batch, mics, rotors, 3)
    rel = rel.contiguous().to(device)
    names = (["dregon", "michaels"] * ((batch + 1) // 2))[:batch]
    initial_phases = (torch.rand(batch, rotors, 100) * 2 * torch.pi).to(device)
    return rps, rel, names, initial_phases


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def median_time(fn, iters: int, device: torch.device) -> float:
    # warmup
    for _ in range(2):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rotors", type=int, default=4)
    ap.add_argument("--mics", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--n-harmonics", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    model = build_noise_gen_model(
        "positional_harmonic_gen",
        sample_rate=SR,
        n_harmonics=args.n_harmonics,
        cond_dim=16,
        drone_names=["dregon", "michaels"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rps, rel, names, initial_phases = make_inputs(
        args.batch, args.rotors, args.mics, args.seconds, device
    )

    def forward():
        return model(rps, rel, names, initial_phases=initial_phases)

    model.eval()
    with torch.no_grad():
        fwd = median_time(forward, args.iters, device)

    model.train()  # train mode for the backward pass (RPS jitter is off: sigma=0)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        out = model(rps, rel, names, initial_phases=initial_phases)
        out.pow(2).mean().backward()

    fwd_bwd = median_time(forward_backward, args.iters, device)

    print(f"device={device.type}  batch={args.batch}  rotors={args.rotors}  mics={args.mics}")
    print(f"seconds={args.seconds}  n_harmonics={args.n_harmonics}  params={n_params:,}")
    print(f"iters={args.iters} (median)")
    print("-" * 56)
    print(f"{'stage':<20}{'ms/step':>12}{'baseline (prompt)':>24}")
    print(f"{'forward':<20}{fwd * 1e3:>12.1f}{'2900-4500 (CPU)':>24}")
    print(f"{'forward+backward':<20}{fwd_bwd * 1e3:>12.1f}{'-':>24}")


if __name__ == "__main__":
    main()
