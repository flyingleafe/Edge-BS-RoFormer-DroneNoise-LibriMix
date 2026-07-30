"""Benchmark: sequential vs parallel (associative-scan) vs fused-Triton CKLA scan.

Times forward+backward of ``models.ckla.complex_kla_scan`` (Python loop over
T) against ``complex_kla_scan_parallel`` (log-depth associative scan) and,
on CUDA, ``models.ckla_triton.complex_kla_scan_triton`` (single fused
scan+readout kernel per direction, chunked-recompute backward) at the
training-relevant size B=32, T=250, N=16, D=128 (8 s context at hop 512),
on CPU and, when available, CUDA. The Triton arm is CUDA-only (the CPU
interpreter is a debugging tool, orders of magnitude off) and skipped
cleanly elsewhere.

Kernel-count reasoning
----------------------
The sequential loop launches ~15 tiny elementwise kernels per step (phi,
kappa, den, the complex eta update, lam update, clamp, readout mul+sum),
each touching only (B, N, D) ≈ 65k elements — far below the size where a GPU
kernel's launch latency (~5–10 µs) amortises. At T=250 that is ~3750
launches per layer forward (and a similar dependent chain in backward), so
the GPU idles between kernels and utilisation collapses.

The parallel version runs two pairwise (Blelloch-style) inclusive scans —
a 2x2 Moebius-matrix scan for lambda and a complex first-order linear scan
for eta. Each scan does O(T) combine *work* but only ceil(log2 T) = 8
dependent levels, and every level is a handful of batched ops over the whole
remaining sequence (level l touches T/2^l steps x B x N x D elements in one
kernel). Total launches drop to a few hundred, each large enough to fill the
device; the arithmetic cost is ~2x the sequential recurrence, which is
irrelevant on GPU (launch-bound) and a modest constant on CPU where the
vectorised full-sequence ops beat 250 Python-loop iterations anyway.

Run:  python scripts/bench_ckla_scan.py [--repeats 5]
"""

from __future__ import annotations

import argparse
import time

import torch

from models.ckla import complex_kla_scan, complex_kla_scan_parallel

B, T, N, D = 32, 250, 16, 128


def make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device="cpu").manual_seed(0)

    def r(*shape, lo=None, hi=None):
        x = torch.rand(*shape, generator=g) if lo is not None else torch.randn(*shape, generator=g)
        if lo is not None:
            x = x * (hi - lo) + lo
        return x.to(device)

    abar_mag = torch.exp(-r(N, D, lo=0.05, hi=3.0) * r(N, D, lo=0.005, hi=0.5))
    pbar = r(N, D, lo=1e-3, hi=0.5)
    omega = r(B, T, N, lo=-3.14, hi=3.14)
    cos_w, sin_w = torch.cos(omega), torch.sin(omega)
    k, q = r(B, T, N), r(B, T, N)
    v = r(B, T, D)
    lam_v = r(B, T, D, lo=0.01, hi=2.0)
    return abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q


def bench(fn, inputs, device: torch.device, repeats: int) -> float:
    """Median wall time (s) of forward + backward w.r.t. all inputs."""
    leaves = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    times = []
    for _ in range(repeats + 1):  # first iteration = warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_re, y_im = fn(*leaves)
        (y_re.square().mean() + y_im.square().mean()).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        for leaf in leaves:
            leaf.grad = None
    times = sorted(times[1:])
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(f"complex-KLA scan fwd+bwd, B={B} T={T} N={N} D={D}, fp32")
    for device in devices:
        inputs = make_inputs(device)
        t_seq = bench(complex_kla_scan, inputs, device, args.repeats)
        t_par = bench(complex_kla_scan_parallel, inputs, device, args.repeats)
        print(
            f"  {device.type:4s}: sequential {t_seq * 1e3:8.1f} ms | "
            f"parallel {t_par * 1e3:8.1f} ms | speedup {t_seq / t_par:5.2f}x"
        )
        if device.type != "cuda":
            continue
        from models.ckla_triton import HAS_TRITON, complex_kla_scan_triton

        if not HAS_TRITON:
            print("  cuda: triton not installed — skipping the triton arm")
            continue
        t_tri = bench(complex_kla_scan_triton, inputs, device, args.repeats)
        print(
            f"  {device.type:4s}: triton     {t_tri * 1e3:8.1f} ms | "
            f"speedup vs sequential {t_seq / t_tri:5.2f}x, vs parallel {t_par / t_tri:5.2f}x"
        )


if __name__ == "__main__":
    main()
