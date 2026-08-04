#!/usr/bin/env python3
"""Micro-benchmark CLI for the project's hot computational kernels.

One shared harness (warmup + CUDA-synchronized median wall time) over four
targets, replacing the per-kernel bench scripts:

* ``--target ckla_scan`` (was ``bench_ckla_scan.py``): forward+backward of
  the sequential ``complex_kla_scan`` vs the log-depth associative
  ``complex_kla_scan_parallel`` vs (CUDA only) the fused Triton kernel.
  ``--shape B,T,N,D`` (default ``32,250,16,128`` — 8 s context at hop 512).
* ``--target cqt`` (was ``bench_cqt_gpu.py``): multif0 HCQT (3 separate
  CQTs) vs the stacked HCQT (1 CQT + freq-shift), forward, plus a
  magnitude/phase agreement check. ``--shape BATCH,SAMPLES``
  (default ``128,16000``).
* ``--target grouped_branches`` (was ``bench_grouped_branches.py``):
  LateDeep separate mag/phase branches vs the fused ``groups=2`` stack,
  forward and forward+backward, with the weight-remap equivalence check.
  ``--shape B,H,F,T`` (default ``128,4,360,63``).
* ``--target noise_gen`` (was ``bench_noise_gen_forward.py``): the
  position-aware harmonic noise generator, forward and forward+backward on
  synthetic RPS + geometry. ``--shape B,ROTORS,MICS,SAMPLES``
  (default ``32,4,8,16000``).

``vk_bench_opt_job.sh`` is gone too; the VK regression bench it drove stays
as ``scripts/vk_bench.py`` (fixtures + scoring live there), e.g.::

    python scripts/vk_bench.py --out-suffix _opt
    python scripts/vk_bench.py --cases free-flight_nosource_room1 --solver splu

Examples::

    python scripts/bench.py --target ckla_scan --device cuda --iters 5
    python scripts/bench.py --target cqt --shape 64,16000
    python scripts/bench.py --target noise_gen --device cpu --iters 5
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import torch

SR = 16000


def timeit(fn: Callable[[], object], device: torch.device, iters: int, warmup: int) -> float:
    """Median wall time (ms) of ``fn()``, CUDA-synchronized around each call."""

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def _report(rows: list[tuple[str, float]]) -> None:
    """Print (label, ms) rows with speedups relative to the first row."""
    base = rows[0][1]
    for label, ms in rows:
        rel = f"  ({base / ms:5.2f}x vs {rows[0][0]})" if ms != base else ""
        print(f"  {label:<28}{ms:10.2f} ms{rel}")


def _shape(arg: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not arg:
        return default
    dims = tuple(int(x) for x in arg.split(","))
    if len(dims) != len(default):
        raise SystemExit(f"--shape needs {len(default)} comma-separated ints, got {arg!r}")
    return dims


# ── targets ─────────────────────────────────────────────────────────────────
def bench_ckla_scan(device: torch.device, iters: int, warmup: int, shape: str | None) -> None:
    from models.ckla import complex_kla_scan, complex_kla_scan_parallel

    b, t, n, d = _shape(shape, (32, 250, 16, 128))
    g = torch.Generator(device="cpu").manual_seed(0)

    def r(*s: int, lo: float | None = None, hi: float | None = None) -> torch.Tensor:
        x = torch.rand(*s, generator=g) if lo is not None else torch.randn(*s, generator=g)
        if lo is not None and hi is not None:
            x = x * (hi - lo) + lo
        return x.to(device)

    abar_mag = torch.exp(-r(n, d, lo=0.05, hi=3.0) * r(n, d, lo=0.005, hi=0.5))
    pbar = r(n, d, lo=1e-3, hi=0.5)
    omega = r(b, t, n, lo=-3.14, hi=3.14)
    inputs = (
        abar_mag,
        torch.cos(omega),
        torch.sin(omega),
        pbar,
        r(b, t, n),
        r(b, t, d),
        r(b, t, d, lo=0.01, hi=2.0),
        r(b, t, n),
    )

    def step(fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]) -> Callable[[], None]:
        leaves = tuple(x.detach().clone().requires_grad_(True) for x in inputs)

        def run() -> None:
            y_re, y_im = fn(*leaves)
            (y_re.square().mean() + y_im.square().mean()).backward()
            for leaf in leaves:
                leaf.grad = None

        return run

    print(f"complex-KLA scan fwd+bwd, B={b} T={t} N={n} D={d}, fp32, {device.type}")
    rows = [
        ("sequential", timeit(step(complex_kla_scan), device, iters, warmup)),
        ("parallel (assoc scan)", timeit(step(complex_kla_scan_parallel), device, iters, warmup)),
    ]
    if device.type == "cuda":
        from models.ckla_triton import HAS_TRITON, complex_kla_scan_triton

        if HAS_TRITON:
            rows.append(
                ("triton (fused)", timeit(step(complex_kla_scan_triton), device, iters, warmup))
            )
        else:
            print("  triton not installed — skipping the triton arm")
    _report(rows)


def bench_cqt(device: torch.device, iters: int, warmup: int, shape: str | None) -> None:
    import numpy as np

    from models.multif0.nnaudio_cqt import HCQT_nnAudio, HCQTStacked_nnAudio

    batch, samples = _shape(shape, (128, SR))
    torch.manual_seed(0)
    audio = torch.randn(batch, samples, device=device)
    kw: dict[str, Any] = {
        "sr": SR,
        "fmin": 32.7,
        "n_octaves": 6,
        "over_sample": 5,
        "harmonics": [1, 2, 3],
        "hop_length": 256,
        "log_mag": True,
    }
    sep = HCQT_nnAudio(**kw).to(device).eval()
    stk = HCQTStacked_nnAudio(**kw).to(device).eval()
    print(f"HCQT mag+dphase, batch={batch} samples={samples}, {device.type}")
    with torch.no_grad():
        rows = [
            ("separate (3 CQTs)", timeit(lambda: sep(audio), device, iters, warmup)),
            ("stacked (1 CQT + shift)", timeit(lambda: stk(audio), device, iters, warmup)),
        ]
        m_sep, dp_sep = sep(audio[:4])
        m_stk, dp_stk = stk(audio[:4])
    _report(rows)
    mc = np.corrcoef(m_sep.flatten().cpu().numpy(), m_stk.flatten().cpu().numpy())[0, 1]
    dd = (dp_sep - dp_stk).cpu().numpy()
    dd = np.abs((dd + np.pi) % (2 * np.pi) - np.pi)
    print(
        f"  agreement: shape {tuple(m_sep.shape)} | mag corr={mc:.4f} | "
        f"dphase mean|Δ|={dd.mean():.4f} rad max={dd.max():.4f} rad"
    )


def bench_grouped_branches(
    device: torch.device, iters: int, warmup: int, shape: str | None
) -> None:
    from models.multif0.model import LateDeep

    b, h, freq, t = _shape(shape, (128, 4, 360, 63))
    torch.manual_seed(0)
    unfused = LateDeep(n_harmonics=h, fused_branches=False).to(device)
    fused = LateDeep(n_harmonics=h, fused_branches=True).to(device)
    fused.load_state_dict(unfused.state_dict())  # exercises the remap pre-hook
    mag = torch.randn(b, h, freq, t, device=device)
    phase = torch.randn(b, h, freq, t, device=device)

    unfused.eval()
    fused.eval()
    with torch.no_grad():
        a, bb = unfused(mag, phase), fused(mag, phase)
    print(f"LateDeep branches, B={b} H={h} F={freq} T={t}, {device.type}")
    print(f"  correctness: max|Δ|={(a - bb).abs().max().item():.2e} shape={tuple(bb.shape)}")
    with torch.no_grad():
        _report(
            [
                ("unfused forward", timeit(lambda: unfused(mag, phase), device, iters, warmup)),
                ("fused forward", timeit(lambda: fused(mag, phase), device, iters, warmup)),
            ]
        )

    unfused.train()
    fused.train()

    def step(model: torch.nn.Module) -> None:
        model.zero_grad(set_to_none=True)
        model(mag, phase).sum().backward()

    _report(
        [
            ("unfused fwd+bwd", timeit(lambda: step(unfused), device, iters, warmup)),
            ("fused fwd+bwd", timeit(lambda: step(fused), device, iters, warmup)),
        ]
    )


def bench_noise_gen(device: torch.device, iters: int, warmup: int, shape: str | None) -> None:
    from models.registry import build_noise_gen_model

    b, rotors, mics, samples = _shape(shape, (32, 4, 8, SR))
    torch.manual_seed(0)
    model = build_noise_gen_model(
        "positional_harmonic_gen",
        sample_rate=SR,
        n_harmonics=100,
        cond_dim=16,
        drone_names=["dregon", "michaels"],
    ).to(device)
    rps = (torch.rand(b, rotors, samples) * 40 + 60).to(device)  # 60-100 Hz
    rotor = torch.tensor([[0.2, 0.2, 0.0], [-0.2, 0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]])[
        :rotors
    ]
    mic = torch.randn(mics, 3) * 0.03
    rel = (mic[None, :, None, :] - rotor[None, None, :, :]).expand(b, mics, rotors, 3)
    rel = rel.contiguous().to(device)
    names = (["dregon", "michaels"] * ((b + 1) // 2))[:b]
    phases = (torch.rand(b, rotors, 100) * 2 * torch.pi).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"positional_harmonic_gen, B={b} rotors={rotors} mics={mics} "
        f"samples={samples} params={n_params:,}, {device.type}"
    )

    model.eval()
    with torch.no_grad():
        fwd = timeit(lambda: model(rps, rel, names, initial_phases=phases), device, iters, warmup)

    model.train()  # train mode for the backward pass (RPS jitter off: sigma=0)

    def step() -> None:
        model.zero_grad(set_to_none=True)
        out = model(rps, rel, names, initial_phases=phases)
        out.pow(2).mean().backward()

    _report([("forward", fwd), ("forward+backward", timeit(step, device, iters, warmup))])


TARGETS: dict[str, Callable[[torch.device, int, int, str | None], None]] = {
    "ckla_scan": bench_ckla_scan,
    "cqt": bench_cqt,
    "grouped_branches": bench_grouped_branches,
    "noise_gen": bench_noise_gen,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--shape", default=None, help="comma-separated dims (target-specific)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    TARGETS[args.target](device, args.iters, args.warmup, args.shape)


if __name__ == "__main__":
    main()
