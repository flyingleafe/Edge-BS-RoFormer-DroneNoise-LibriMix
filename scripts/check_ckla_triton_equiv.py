"""CUDA equivalence check: fused Triton CKLA scan vs the sequential reference.

Runs forward + backward for both paths on random inputs at the production
shape and asserts elementwise closeness. Must run in a process that has NOT
imported triton's CPU interpreter (``TRITON_INTERPRET``): the interpreter
monkey-patches triton globals at import and breaks compiled-mode codegen in
the same process — which is exactly why the pytest CUDA test invokes this
script via subprocess instead of calling the op inline.

Exit codes: 0 = pass, 1 = mismatch/error, 3 = no CUDA (treat as skip).
"""

import os
import sys

import numpy as np
import torch


def main() -> int:
    if os.environ.get("TRITON_INTERPRET"):
        print("refusing to run with TRITON_INTERPRET set (would not test compiled mode)")
        return 1
    if not torch.cuda.is_available():
        print("no CUDA device — skipping")
        return 3

    from models.ckla import complex_kla_scan
    from models.ckla_triton import HAS_TRITON, complex_kla_scan_triton

    if not HAS_TRITON:
        print("triton not installed — skipping")
        return 3

    rng = np.random.default_rng(4242)
    B, T, N, D = 4, 250, 16, 128
    abar = torch.as_tensor(rng.uniform(0.3, 0.999, (N, D)), dtype=torch.float32)
    pbar = torch.as_tensor(rng.uniform(1e-4, 1.0, (N, D)), dtype=torch.float32)
    w = rng.standard_normal((B, T, N))
    cos_w = torch.as_tensor(np.cos(w), dtype=torch.float32)
    sin_w = torch.as_tensor(np.sin(w), dtype=torch.float32)
    k = torch.as_tensor(rng.standard_normal((B, T, N)), dtype=torch.float32)
    v = torch.as_tensor(rng.standard_normal((B, T, D)), dtype=torch.float32)
    lam_v = torch.as_tensor(rng.uniform(0.05, 2.0, (B, T, D)), dtype=torch.float32)
    q = torch.as_tensor(rng.standard_normal((B, T, N)), dtype=torch.float32)
    base = (abar, cos_w, sin_w, pbar, k, v, lam_v, q)
    w_re = torch.as_tensor(rng.standard_normal((B, T, D)), dtype=torch.float32, device="cuda")
    w_im = torch.as_tensor(rng.standard_normal((B, T, D)), dtype=torch.float32, device="cuda")

    def run(fn):
        tens = tuple(t.clone().cuda().requires_grad_(True) for t in base)
        y_re, y_im = fn(*tens)
        ((y_re * w_re).sum() + (y_im * w_im).sum()).backward()
        grads = []
        for t in tens:
            assert t.grad is not None
            grads.append(t.grad.cpu().numpy())
        return y_re.detach().cpu().numpy(), y_im.detach().cpu().numpy(), grads

    names = ("abar_mag", "cos_w", "sin_w", "pbar", "k", "v", "lam_v", "q")
    yr_s, yi_s, g_seq = run(complex_kla_scan)
    yr_t, yi_t, g_tri = run(complex_kla_scan_triton)
    ok = True
    for label, a, b, rtol, atol in [
        ("y_re", yr_s, yr_t, 1e-4, 1e-5),
        ("y_im", yi_s, yi_t, 1e-4, 1e-5),
        # T=250 fp32 accumulation: grads at a slightly looser atol
        # (summation-order rounding over 250 steps).
        *[(f"grad:{n}", gs, gt, 1e-3, 1e-4) for n, gs, gt in zip(names, g_seq, g_tri)],
    ]:
        err = np.max(np.abs(a - b))
        rel = err / (np.max(np.abs(a)) + 1e-30)
        line_ok = np.allclose(b, a, rtol=rtol, atol=atol)
        ok &= bool(line_ok)
        print(f"{label:14s} max_abs_err={err:.3e} rel={rel:.3e} {'OK' if line_ok else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
