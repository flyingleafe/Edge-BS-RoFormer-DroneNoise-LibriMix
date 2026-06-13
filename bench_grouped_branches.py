"""Benchmark LateDeep's two separate mag/phase branches vs the fused grouped
(groups=2) stack. Forward AND forward+backward (training is what matters).
Device-aware: CUDA if available, else CPU.

    uv run python bench_grouped_branches.py            # B=128, H=4, T=63 (~1s @16k)
    uv run python bench_grouped_branches.py --batch 256 --time 125
"""

import argparse
import time

import torch

from models.multif0.model import LateDeep


def timeit(fn, device, n_iters, warmup):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)  # effective batch (16 x 8 ch)
    ap.add_argument("--harmonics", type=int, default=4)  # n_harmonics at fmin=27.5
    ap.add_argument("--freq", type=int, default=360)
    ap.add_argument("--time", type=int, default=63)  # ~1s @16k, hop 256
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = args.harmonics
    torch.manual_seed(0)

    unfused = LateDeep(n_harmonics=H, fused_branches=False).to(device)
    fused = LateDeep(n_harmonics=H, fused_branches=True).to(device)
    # Load the SAME weights into the fused model (exercises the remap pre-hook).
    fused.load_state_dict(unfused.state_dict())

    mag = torch.randn(args.batch, H, args.freq, args.time, device=device)
    phase = torch.randn(args.batch, H, args.freq, args.time, device=device)

    print(f"device={device}  batch={args.batch}  H={H}  freq={args.freq}  time={args.time}  "
          f"iters={args.iters}")

    # ── correctness (remap + equivalence) ──
    unfused.eval()
    fused.eval()
    with torch.no_grad():
        a = unfused(mag, phase)
        b = fused(mag, phase)
    print(f"  correctness: max|Δ|={ (a-b).abs().max().item():.2e}  shape={tuple(b.shape)}\n")

    # ── forward (eval) ──
    with torch.no_grad():
        t_uf = timeit(lambda: unfused(mag, phase), device, args.iters, args.warmup)
        t_f = timeit(lambda: fused(mag, phase), device, args.iters, args.warmup)
    print(f"  forward (eval)      unfused {t_uf:8.2f} ms | fused {t_f:8.2f} ms | "
          f"{t_uf/t_f:.2f}x")

    # ── forward + backward (train) ──
    unfused.train()
    fused.train()

    def step(model):
        model.zero_grad(set_to_none=True)
        out = model(mag, phase)
        out.sum().backward()

    t_uf_b = timeit(lambda: step(unfused), device, args.iters, args.warmup)
    t_f_b = timeit(lambda: step(fused), device, args.iters, args.warmup)
    print(f"  fwd+bwd (train)     unfused {t_uf_b:8.2f} ms | fused {t_f_b:8.2f} ms | "
          f"{t_uf_b/t_f_b:.2f}x")


if __name__ == "__main__":
    main()
