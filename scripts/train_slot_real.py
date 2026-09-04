#!/usr/bin/env python
"""Train the slot-comb PARTIAL emission on real 8-mic windows (candidate C1).

WHAT IS BEING TRAINED, AND ON TOP OF WHAT. `scripts/train_comb_slots.py` trains
the rate-conditioned head on the synthetic comb families. This trains the
learned partial-observation emission (`models.comb_slots.PartialEmission`) on
the corner that already works on REAL audio: `head_mode="classical"`,
`n_iter=0`, eight microphones power-averaged, a 15-bin (60 Hz) running-median
floor and the decode `subgrid=True, octave=False, relocate=True`, which reads
DREGON cruise at 1.49 rev/s with zero trained parameters (probe P1c). Every part
of the emission starts at zero effect, so step 0 IS that corner.

THE LOSS IS THE CRF NLL OF THE GOLD TRAJECTORY, unchanged: `log Z -
score(gold)` over the same chain the deployed Viterbi maximizes, so lowering it
cannot mean anything but making the deployed path more likely.

SELECTION IS NOT THE FROZEN SPLIT. The 37-clip split is the test set and FLY124
is an unseen rig; three campaigns have been inverted by monitors that are not
the task metric. Selection is the mean PIT MAE of the deployed decoder over a
FIXED set of real windows drawn from the training policy with a different base
seed (`experiments.slot_real.select_set`). The frozen split is scored at every
validation FOR THE RECORD.

THE DECODE CONFIGURATION IS FIXED at validation: `subgrid=True, octave=False,
relocate=True`. The decoder's own octave move costs FLY124 heavily (22.4 ->
31.1) and buys nothing at eight microphones, so it is off, and the empty-tooth
term in the emission is the octave lever this run is actually testing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def build_model(args, device: str):
    from models.comb_slots import SlotCombNet

    parts = tuple(p for p in args.parts.split(",") if p and p != "none")
    net = SlotCombNet(
        sr=16000,
        n_fft=4096,
        hop_length=512,
        r_lo=30.0,
        r_hi=100.0,
        n_grid=700,
        k_max=args.k_max,
        f_max=7500.0,
        head_mode="classical",
        floor_hz=args.floor_hz,
        n_rot=4,
        n_iter=0,
        slew=12.0,
        stiff=40.0,
        multichannel=True,
        use_checkpoint=True,
        emission="partial" if parts else "classical",
        parts=parts,
    ).to(device)
    return net, parts


def trainable(net) -> list[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def state_dict_trainable(net) -> dict:
    return {k: v.detach().cpu() for k, v in net.named_parameters() if v.requires_grad}


def batch(streams, order: list[int], step: int, size: int, device: str):
    """One batch from the stream this step belongs to: ``(B, C, N)`` and ``(B, R, T)``."""
    st = streams[order[step % len(order)]]
    auds, gts = zip(*[next(st) for _ in range(size)])
    n = min(a.shape[-1] for a in auds)
    t = min(g.shape[-1] for g in gts)
    au = torch.as_tensor(np.stack([a[..., :n] for a in auds]), device=device)
    gt = torch.as_tensor(np.stack([g[..., :t] for g in gts]), device=device)
    return au, gt


def validate(net, sel, clips, device, args, step: int, frozen: bool = True) -> dict:
    from experiments import slot_real as sr

    t0 = time.time()
    row: dict = {"step": step}
    row["select"] = sr.score_windows(net, sel, device=device)
    print(
        f"    selection PIT MAE {row['select']:.3f} rev/s "
        f"({len(sel)} windows, {time.time() - t0:.0f} s)",
        flush=True,
    )
    if clips and frozen:
        row["frozen"] = sr.score_real(net, clips, device=device, name=f"step {step}")
        row["frozen"].pop("rows", None)
    if args.val_parts:
        for name in ("comb", "stoch"):
            row[name] = sr.score_part(net, name, device=device, n=16)
            print(
                f"    {name}: mean {row[name]['mean']:.3f} median "
                f"{row[name]['median']:.3f} 1x {row[name]['frac_one']:.3f} "
                f"1/2 {row[name]['frac_half']:.3f}",
                flush=True,
            )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="c1")
    ap.add_argument(
        "--parts",
        default=",".join(("reliability", "channels", "empty_tooth", "floor_mix")),
        help="comma list; 'none' is the zero-parameter classical corner (eval only)",
    )
    ap.add_argument("--data", default="both", choices=("real", "partial", "both"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-s", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-clips", type=int, default=0, help="0 = the whole frozen split")
    ap.add_argument(
        "--frozen-every",
        type=int,
        default=1,
        help="score the frozen split every Nth validation. The split is the "
        "RECORD, not the selection metric, and it costs ~47 s per clip "
        "with the partial emission, so a long run can afford it less often",
    )
    ap.add_argument("--val-parts", action="store_true", help="also score comb/stoch (n=16)")
    ap.add_argument("--select-n", type=int, default=48)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--floor-hz", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="wall-clock budget; save and print CHAIN: continue before it",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    from experiments import slot_real as sr

    t_start = time.time()
    out = Path(args.out or f"results/slot_real/{args.name}")
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    net, parts = build_model(args, dev)
    params = trainable(net)
    n_par = sum(p.numel() for p in params)
    print(
        f"device={dev} parts={parts or '(classical corner)'} data={args.data} trainable={n_par}",
        flush=True,
    )

    opt = torch.optim.Adam(params, lr=args.lr) if params else None
    step0, best, hist = 0, float("inf"), []
    state_path = out / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        net.load_state_dict(st["params"], strict=False)
        if opt is not None and st.get("opt"):
            opt.load_state_dict(st["opt"])
        step0, best, hist = st["step"], st["best"], st["history"]
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed from {state_path} at step {step0} (best {best:.4f})", flush=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    sel = sr.select_set(n=args.select_n, crop_s=args.crop_s)
    clips = sr.real_clips()
    if args.val_clips:
        # Spread, not the first n: clips 0 and 1 are a ground clip and a ramp,
        # so a smoke that takes the head of the list never sees a cruise number.
        clips = clips[:: max(1, len(clips) // args.val_clips)][: args.val_clips]
    print(f"selection {len(sel)} windows, frozen split {len(clips)} clips", flush=True)

    streams, order = [], []
    if args.data in ("real", "both"):
        order.append(len(streams))
        streams.append(sr.windows(sr.POLICY_REAL, crop_s=args.crop_s, seed=args.seed, epoch=step0))
    if args.data in ("partial", "both"):
        order.append(len(streams))
        streams.append(
            sr.windows(sr.POLICY_PARTIAL, crop_s=args.crop_s, seed=args.seed + 1, epoch=step0)
        )

    # ── Step 0 ───────────────────────────────────────────────────────────────
    if not hist:
        row = validate(net, sel, clips, dev, args, step0)
        row["loss"] = None
        hist.append(row)
        best = row["select"]
        torch.save(state_dict_trainable(net), out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=1))

    # ── Train ────────────────────────────────────────────────────────────────
    budget = args.max_minutes * 60.0 if args.max_minutes > 0 else float("inf")
    net.train()
    t0, done = time.time(), step0
    for step in range(step0 + 1, args.steps + 1):
        if opt is None:
            break
        au, gt = batch(streams, order, step, args.batch, dev)
        loss = net.loss(au, gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        done = step
        if step % 25 == 0 or step == step0 + 1:
            acc = " ".join(f"{s.kept}/{s.seen}" for s in streams)
            print(
                f"step {step:5d}  loss {loss.item():10.2f}  "
                f"{(time.time() - t0) / max(1, step - step0):.2f}s/step  "
                f"crop accept {acc}",
                flush=True,
            )
        if step % args.val_every == 0:
            n_val = step // args.val_every
            row = validate(
                net, sel, clips, dev, args, step, frozen=(n_val % max(1, args.frozen_every) == 0)
            )
            row["loss"] = float(loss.item())
            row["accept"] = [s.accept for s in streams]
            hist.append(row)
            star = ""
            if row["select"] < best:
                best = row["select"]
                torch.save(state_dict_trainable(net), out / "best.pt")
                star = "  *"
            print(f"step {step:5d} select={row['select']:.4f}{star}", flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            net.train()
        if time.time() - t_start > budget and step < args.steps:
            torch.save(
                {
                    "params": state_dict_trainable(net),
                    "opt": opt.state_dict(),
                    "step": step,
                    "best": best,
                    "history": hist,
                    "torch_rng": torch.get_rng_state(),
                },
                state_path,
            )
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            print(f"CHAIN: continue  (step {step}/{args.steps}, best {best:.4f})", flush=True)
            return 0

    # ── Report at the best checkpoint ────────────────────────────────────────
    if (out / "best.pt").exists():
        net.load_state_dict(torch.load(out / "best.pt", map_location=dev), strict=False)
    final = sr.score_real(net, clips, device=dev, name=f"{args.name} best")
    final.pop("rows", None)
    report = {
        "name": args.name,
        "parts": list(parts),
        "data": args.data,
        "steps": done,
        "trainable": n_par,
        "select_best": best,
        "frozen": final,
        "history": hist,
    }
    if args.val_parts:
        report["parts_eval"] = {
            n: sr.score_part(net, n, device=dev, n=16) for n in ("comb", "stoch")
        }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    if opt is not None:
        torch.save(
            {
                "params": state_dict_trainable(net),
                "opt": opt.state_dict(),
                "step": done,
                "best": best,
                "history": hist,
                "torch_rng": torch.get_rng_state(),
            },
            state_path,
        )
    print(f"CHAIN: done  (best selection {best:.4f}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
