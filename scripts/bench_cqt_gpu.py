"""Benchmark multif0 HCQT (3 separate CQTs) vs stacked HCQT (1 CQT + freq-shift),
both producing magnitude AND unwrapped phase-differential. Device-aware: runs on
CUDA if available (the case that matters), else CPU.

Run on the GPU node:
    uv run python bench_cqt_gpu.py
Optionally:  --batch 128 --dur 1.0 --harmonics 1,2,3 --iters 30
"""

import argparse
import time

import numpy as np
import torch

from models.multif0.nnaudio_cqt import HCQT_nnAudio, HCQTStacked_nnAudio


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
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=128)  # effective batch (16 x 8 ch)
    ap.add_argument("--dur", type=float, default=1.0)
    ap.add_argument("--harmonics", type=str, default="1,2,3")
    ap.add_argument("--over_sample", type=int, default=5)
    ap.add_argument("--n_octaves", type=int, default=6)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harmonics = [int(h) for h in args.harmonics.split(",")]
    N = int(args.sr * args.dur)
    torch.manual_seed(0)
    audio = torch.randn(args.batch, N, device=device)

    kw = dict(
        sr=args.sr,
        fmin=32.7,
        n_octaves=args.n_octaves,
        over_sample=args.over_sample,
        harmonics=harmonics,
        hop_length=args.hop,
        log_mag=True,
    )
    sep = HCQT_nnAudio(**kw).to(device).eval()
    stk = HCQTStacked_nnAudio(**kw).to(device).eval()

    print(
        f"device={device}  batch={args.batch}  dur={args.dur}s  sr={args.sr}  "
        f"harmonics={harmonics}  iters={args.iters}"
    )
    print(f"  separate: {len(harmonics)} CQTs of {kw['n_octaves'] * 12 * args.over_sample} bins")
    print(f"  stacked : 1 CQT of {stk.cqt.n_bins} bins + {len(harmonics)} freq-shifts\n")

    with torch.no_grad():
        t_sep = timeit(lambda: sep(audio), device, args.iters, args.warmup)
        t_stk = timeit(lambda: stk(audio), device, args.iters, args.warmup)

        # correctness: same shape + magnitude/phase agreement on overlapping bins
        m_sep, dp_sep = sep(audio[:4])
        m_stk, dp_stk = stk(audio[:4])

    print(f"  HCQT_nnAudio   (3 CQTs + phase): {t_sep:8.2f} ms")
    print(f"  HCQTStacked    (1 CQT  + phase): {t_stk:8.2f} ms   ({t_sep / t_stk:.2f}x faster)\n")

    assert m_sep.shape == m_stk.shape, (m_sep.shape, m_stk.shape)
    mc = np.corrcoef(m_sep.flatten().cpu().numpy(), m_stk.flatten().cpu().numpy())[0, 1]
    # phase-diff circular agreement
    dd = (dp_sep - dp_stk).cpu().numpy()
    dd = np.abs((dd + np.pi) % (2 * np.pi) - np.pi)
    print(
        f"  output shape {tuple(m_sep.shape)} match | mag corr={mc:.4f} | "
        f"dphase mean|Δ|={dd.mean():.4f} rad  max={dd.max():.4f} rad"
    )


if __name__ == "__main__":
    main()
