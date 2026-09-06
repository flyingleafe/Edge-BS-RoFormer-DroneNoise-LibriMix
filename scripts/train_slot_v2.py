#!/usr/bin/env python
"""Train the slot-comb CRF v2 — `scripts/train_slot_real.py` with the v2 groups.

WHAT IS NEW, AND WHY EACH GROUP HAS A FLAG. `docs/slot-comb-v2-design.md` maps
every regime the C1 arms lose to ONE parameter group, and each group is a family
that contains the current setting at initialization. The ablation the design
asks for is "one part at a time from the corner", so every group is a flag here
and the corner is every flag off. With no v2 flag this script IS
`scripts/train_slot_real.py --mono`: the same model, the same policy, the same
selection set size, the same chain markers.

    --off-state              § 3.1  one OFF state per chain; zero frames
    --grid-lo 10 --n-grid 900 § 3.2  the grid below 30 rev/s
    --learned-transition     § 3.3  the hinge becomes a learned penalty
    --emission v2 --v2-parts § 3.4/3.5/3.7  gap gather, cross-order net, widths
    --rate-prior             § 3.6  the pairwise prior across slots

THE LOSS NOW COVERS EVERY FRAME. C1 rejected any crop in which a rotor left the
30-100 rev/s grid, so ground, warm-up and the low half of every ramp never
reached the loss — and those frames are 60.9 rev/s of the best arm's error.
With `--off-state` the sampler keeps every crop and hands the model the FULL
trajectory, zeros included; the model decides what a zero frame is (the gold
state is OFF below 0.5 rev/s) and masks the frames between 0.5 rev/s and the
grid floor inside `SlotCombNet.loss`. Without `--off-state` the C1 acceptance
window is used, because a model with no OFF state cannot express a zero and
would learn to put it at the grid edge.

THE DATA MODES ARE THE CURRICULUM. `--data comb`, `stoch`, `real` and the two
mixtures `comb+real` / `stoch+real` name the online-mix policies of
`experiments.slot_v2.POLICIES`. Every mode is read MONO — one microphone per
crop, drawn uniformly — which is the protocol every neural baseline is read on.
`--init <best.pt>` warm-starts one stage from the previous one, so the
curriculum comb -> stoch -> real is three chained runs and not a new script.

SELECTION IS PER MODE AND IS NEVER THE FROZEN SPLIT. 48 windows drawn from the
arm's OWN training policy with a different base seed. The 37-clip split is the
test set and FLY124 is an unseen rig; it is scored at every validation FOR THE
RECORD only.

THE CONSTRUCTOR NAMES OF THE OTHER TWO GROUPS ARE ASSUMED. `off_state` and
`learned_transition` are written by the chain branch and `emission="v2"` /
`v2_parts` by the emission branch. This script checks the signature of
`SlotCombNet` before it builds anything and says exactly which keyword is
missing, and `--model-kwarg NAME=VALUE` passes any keyword through without an
edit, so a renamed group costs one command-line argument and not a merge.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: our flag -> the `SlotCombNet` keyword we assume the other two branches used.
#: Checked against the real signature at build time; `--model-kwarg` overrides.
V2_KEYWORDS = {
    "off_state": "off_state",  # branch slotcomb-v2-chain, § 3.1
    "learned_transition": "learned_transition",  # branch slotcomb-v2-chain, § 3.3
    "rate_prior": "rate_prior",  # this branch, § 3.6
    "v2_parts": "v2_parts",  # branch slotcomb-v2-emission, § 3.4/3.5/3.7
}
#: the parts of the v2 emission, in the order of the design's sections
V2_PARTS = ("gap", "cross_order", "read_width", "claim_width")
#: the C1 emission parts. `channels` is dropped in mono, where it is meaningless
C1_PARTS = ("reliability", "channels", "empty_tooth", "floor_mix")


# ─── The model ───────────────────────────────────────────────────────────────


def parse_kv(items: list[str], what: str) -> dict[str, Any]:
    """``["a=1", "b=x"]`` -> ``{"a": 1, "b": "x"}``, values read as YAML scalars."""
    out: dict[str, Any] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--{what}: expected NAME=VALUE, got {item!r}")
        out[name.strip()] = yaml.safe_load(value)
    return out


def model_kwargs(args) -> dict[str, Any]:
    """The full `SlotCombNet` constructor call, as a JSON-safe dict.

    This dict IS `config.json`'s ``model`` block, so what trains and what a dump
    or a probe rebuilds cannot drift.
    """
    parts = tuple(p for p in args.parts.split(",") if p and p != "none")
    if not args.multi_mic and "channels" in parts:
        # A mono input has ONE channel and it is already the power mean, so the
        # per-mic candidates the `channels` part adds do not exist.
        parts = tuple(p for p in parts if p != "channels")
        print("mono: dropped the 'channels' part (a mono input has no extra mics)", flush=True)
    kw: dict[str, Any] = {
        "sr": 16000,
        "n_fft": 4096,
        "hop_length": 512,
        "r_lo": float(args.grid_lo),
        "r_hi": float(args.grid_hi),
        "n_grid": int(args.n_grid),
        "k_max": int(args.k_max),
        "f_max": 7500.0,
        "head_mode": "classical",
        "floor_hz": float(args.floor_hz),
        "n_rot": 4,
        "n_iter": 0,
        "slew": 12.0,
        "stiff": 40.0,
        "multichannel": True,
        "use_checkpoint": True,
        "emission": args.emission if parts or args.emission == "v2" else "classical",
        "parts": list(parts),
    }
    if args.mask_k_max:
        kw["mask_k_max"] = int(args.mask_k_max)
    if args.off_state:
        kw[V2_KEYWORDS["off_state"]] = True
    if args.learned_transition:
        kw[V2_KEYWORDS["learned_transition"]] = True
    if args.rate_prior:
        kw[V2_KEYWORDS["rate_prior"]] = True
    if args.emission == "v2":
        kw[V2_KEYWORDS["v2_parts"]] = [p for p in args.v2_parts.split(",") if p and p != "none"]
    kw.update(parse_kv(args.model_kwarg, "model-kwarg"))
    return kw


def check_signature(kw: dict[str, Any]) -> None:
    """Fail with a readable message when a v2 keyword is not in the model yet.

    The three v2 branches land separately. A keyword the merged model does not
    have would otherwise surface as a bare ``TypeError`` deep in a chained job.
    """
    from models.comb_slots import SlotCombNet

    have = set(inspect.signature(SlotCombNet.__init__).parameters)
    missing = sorted(k for k in kw if k not in have)
    if missing:
        raise SystemExit(
            f"SlotCombNet has no keyword {missing}. The group has not landed yet, or it "
            f"was named differently; pass the real name with --model-kwarg NAME=VALUE. "
            f"Known keywords: {sorted(have - {'self'})}"
        )


def build_model(kw: dict[str, Any], device: str):
    from experiments import slot_v2 as sv

    check_signature(kw)
    return sv.build_from_config({"model": kw}, device=device)


def trainable(net) -> list[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def state_dict_trainable(net) -> dict:
    return {k: v.detach().cpu() for k, v in net.named_parameters() if v.requires_grad}


def warm_start(net, path: str, device: str) -> None:
    """Load a previous arm's trainable head, and say what did not match.

    The curriculum (comb -> stoch -> real) changes the DATA and keeps the model,
    but a stage that also changes a group has a different head, so the load is
    ``strict=False`` and both directions are printed: a missing key is a
    parameter this arm must learn from its initialization, an unexpected key is
    one the previous arm had and this one does not.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "best.pt"
    state = torch.load(p, map_location=device)
    report = net.load_state_dict(state, strict=False)
    own = {k for k, v in net.named_parameters() if v.requires_grad}
    missing = sorted(own - set(state))
    print(
        f"--init {p}: loaded {len(set(state) & own)} of {len(own)} trainable tensors; "
        f"missing {missing or 'none'}; unexpected {sorted(report.unexpected_keys) or 'none'}",
        flush=True,
    )


# ─── Data ────────────────────────────────────────────────────────────────────


def acceptance(args, kw: dict[str, Any]) -> tuple[float, float]:
    """The rotor-rate window a training crop must stay inside.

    Full range with an OFF state, because that is the whole point of § 3.1 and
    § 3.2: the zero and below-grid frames must reach the loss. Without one, the
    C1 window, one grid step inside each edge.
    """
    from experiments.slot_v2 import FULL_RANGE

    if args.full_range or (args.off_state and not args.grid_range):
        if not args.off_state:
            # An honest warning, not a refusal: a full-range smoke is a useful
            # check of the sampler. But without an OFF state `SlotCombNet.loss`
            # has no gold cell for a stopped rotor and puts it at the grid edge,
            # so a real arm trained this way learns the wrong thing.
            print(
                "WARNING: --full-range without --off-state. A zero-rate frame has no gold "
                "state, so the loss will put it at the grid edge. Use --off-state.",
                flush=True,
            )
        return FULL_RANGE
    return (float(kw["r_lo"]) + 1.0, float(kw["r_hi"]) - 1.0)


def batch(streams, order: list[int], step: int, size: int, device: str):
    """One batch from the stream this step belongs to: ``(B, C, N)`` and ``(B, R, T)``."""
    st = streams[order[step % len(order)]]
    auds, gts = zip(*[next(st) for _ in range(size)], strict=True)
    n = min(a.shape[-1] for a in auds)
    t = min(g.shape[-1] for g in gts)
    au = torch.as_tensor(np.stack([a[..., :n] for a in auds]), device=device)
    gt = torch.as_tensor(np.stack([g[..., :t] for g in gts]), device=device)
    return au, gt


# ─── Validation ──────────────────────────────────────────────────────────────


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
        scorer = sr.score_real if args.multi_mic else sr.score_real_mono
        row["frozen"] = scorer(net, clips, device=device, name=f"step {step}")
        row["frozen"].pop("rows", None)
    if args.val_parts:
        for name in ("comb", "stoch"):
            row[name] = sr.score_part(net, name, device=device, n=args.val_part_n)
            print(
                f"    {name}: mean {row[name]['mean']:.3f} median "
                f"{row[name]['median']:.3f} 1x {row[name]['frac_one']:.3f} "
                f"1/2 {row[name]['frac_half']:.3f}",
                flush=True,
            )
    return row


# ─── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    from experiments.slot_v2 import DATA_MODES

    g = ap.add_argument_group("the v2 parameter groups (every one off by default)")
    g.add_argument("--off-state", action="store_true", help="§ 3.1 one OFF state per chain")
    g.add_argument("--learned-transition", action="store_true", help="§ 3.3 learned penalty")
    g.add_argument("--rate-prior", action="store_true", help="§ 3.6 pairwise prior across slots")
    g.add_argument(
        "--emission",
        default="partial",
        choices=("classical", "partial", "v2"),
        help="partial = the C1 emission (default); v2 adds the groups of --v2-parts",
    )
    g.add_argument("--v2-parts", default=",".join(V2_PARTS), help="comma list, or 'none'")
    g.add_argument("--parts", default=",".join(C1_PARTS), help="C1 emission parts, or 'none'")
    g.add_argument("--grid-lo", type=float, default=30.0, help="§ 3.2 the grid floor, rev/s")
    g.add_argument("--grid-hi", type=float, default=100.0)
    g.add_argument("--n-grid", type=int, default=700, help="§ 3.2 use 900 with --grid-lo 10")
    g.add_argument("--k-max", type=int, default=40)
    g.add_argument("--floor-hz", type=float, default=60.0)
    g.add_argument("--mask-k-max", type=int, default=0, help="0 = f_max / grid_lo, as the model")
    g.add_argument(
        "--model-kwarg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="pass any SlotCombNet keyword through, for a group named differently",
    )

    d = ap.add_argument_group("data")
    d.add_argument("--data", default="real", choices=DATA_MODES)
    d.add_argument(
        "--multi-mic",
        action="store_true",
        help="read all eight microphones. The default is MONO, the protocol "
        "every neural baseline is read on",
    )
    d.add_argument(
        "--full-range",
        action="store_true",
        help="keep every crop, whatever the labels do (implied by --off-state)",
    )
    d.add_argument(
        "--grid-range",
        action="store_true",
        help="force the C1 acceptance window even with --off-state",
    )
    d.add_argument("--crop-s", type=float, default=2.0)

    t = ap.add_argument_group("training")
    t.add_argument("--name", default="v2")
    t.add_argument("--out", default="")
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init", default="", help="warm start from a previous arm's best.pt")
    t.add_argument(
        "--init-param",
        action="append",
        default=[],
        metavar="PARAM=VALUE",
        help="overwrite one trainable parameter at build time (a scalar fills the "
        "tensor); repeatable; ignored on resume",
    )
    t.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="wall-clock budget; save and print CHAIN: continue before it",
    )
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument(
        "--threads",
        type=int,
        default=4,
        help="torch CPU threads. 4 by default: a CPU smoke must leave the box usable",
    )

    v = ap.add_argument_group("validation and selection")
    v.add_argument("--val-every", type=int, default=100)
    v.add_argument("--select-n", type=int, default=48)
    v.add_argument("--val-clips", type=int, default=0, help="0 = the whole frozen split")
    v.add_argument(
        "--frozen-every",
        type=int,
        default=1,
        help="score the frozen split every Nth validation; 0 = never",
    )
    v.add_argument("--val-parts", action="store_true", help="also score comb/stoch")
    v.add_argument("--val-part-n", type=int, default=16)
    v.add_argument(
        "--smoke",
        action="store_true",
        help="2 steps, 1 s crops, a tiny grid: the CPU check that every seam runs",
    )
    return ap


def apply_smoke(args) -> None:
    """The smallest run that still touches every seam, on a CPU in a minute.

    The frozen split is OFF here (``--frozen-every 0``). It is 37 clips of 8 s
    read one microphone at a time, so even one clip is 24 decodes over a run,
    and it is `experiments.slot_real` code that every C1 arm already exercises.
    What the smoke has to prove is THIS script's plumbing: the model builds, the
    config is written, the selection set draws, the stream yields, the loss
    steps, the best head is saved and the chain markers print. Pass
    ``--frozen-every 1 --val-clips 1`` to add one clip back.
    """
    args.steps, args.batch, args.crop_s = 2, 1, 1.0
    args.n_grid, args.k_max, args.mask_k_max = 60, 8, 40
    args.select_n, args.val_every = 2, 2
    args.val_clips, args.val_part_n = 1, 2
    args.frozen_every = 0
    args.max_minutes = args.max_minutes or 0.0


# ─── Main ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    if args.threads:
        torch.set_num_threads(args.threads)

    from experiments import slot_real as sr
    from experiments import slot_v2 as sv

    t_start = time.time()
    out = Path(args.out or f"results/slot_v2/{args.name}")
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    kw = model_kwargs(args)
    accept = acceptance(args, kw)
    net = build_model(kw, dev)
    named = dict(net.named_parameters())
    for name, value in parse_kv(args.init_param, "init-param").items():
        if name not in named:
            raise SystemExit(f"--init-param: no trainable parameter {name!r}; have {sorted(named)}")
        with torch.no_grad():
            named[name].fill_(float(value))
        print(f"init override {name} = {float(value)}", flush=True)
    if args.init:
        warm_start(net, args.init, dev)
    params = trainable(net)
    n_par = sum(p.numel() for p in params)
    print(
        f"device={dev} data={args.data} emission={kw['emission']} parts={kw['parts']} "
        f"grid={kw['r_lo']:g}-{kw['r_hi']:g} x {kw['n_grid']} accept={accept} "
        f"trainable={n_par}",
        flush=True,
    )

    config = {
        "script": "scripts/train_slot_v2.py",
        "name": args.name,
        "model": kw,
        "decode": dict(sv.DECODE_KW),
        "train": {
            "data": args.data,
            "mono": not args.multi_mic,
            "accept": list(accept),
            "steps": args.steps,
            "batch": args.batch,
            "crop_s": args.crop_s,
            "lr": args.lr,
            "seed": args.seed,
            "select_n": args.select_n,
            "init": args.init,
            "smoke": bool(args.smoke),
        },
    }
    sv.save_config(out, config)

    opt = torch.optim.Adam(params, lr=args.lr) if params else None
    step0, best, hist = 0, float("inf"), []
    state_path = out / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        net.load_state_dict(st["params"], strict=False)
        if opt is not None and st.get("opt"):
            opt.load_state_dict(st["opt"])
        step0, best, hist = st["step"], st["best"], st["history"]
        torch.set_rng_state(st["torch_rng"].cpu())  # a CUDA map_location makes it a cuda tensor
        print(f"resumed from {state_path} at step {step0} (best {best:.4f})", flush=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    sel = sv.select_set(
        args.data,
        n=args.select_n,
        crop_s=args.crop_s,
        accept=accept,
        mono=not args.multi_mic,
    )
    clips = sr.real_clips() if args.frozen_every else []
    if args.val_clips and clips:
        # Spread, not the first n: clips 0 and 1 are a ground clip and a ramp,
        # so a smoke that takes the head of the list never sees a cruise number.
        clips = clips[:: max(1, len(clips) // args.val_clips)][: args.val_clips]
    print(
        f"selection {len(sel)} windows from '{args.data}' "
        f"({'8 mic' if args.multi_mic else 'mono'}), frozen split {len(clips)} clips",
        flush=True,
    )
    streams = sv.streams_for_mode(
        args.data,
        crop_s=args.crop_s,
        seed=args.seed,
        epoch=step0,
        accept=accept,
        mono=not args.multi_mic,
    )
    order = list(range(len(streams)))

    # ── Step 0 ───────────────────────────────────────────────────────────────
    if not hist:
        row = validate(net, sel, clips, dev, args, step0, frozen=bool(args.frozen_every))
        row["loss"] = None
        hist.append(row)
        best = row["select"]
        torch.save(state_dict_trainable(net), out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=1))

    # ── Train ────────────────────────────────────────────────────────────────
    budget = args.max_minutes * 60.0 if args.max_minutes > 0 else float("inf")
    net.train()
    t0, done = time.time(), step0
    n_skipped = n_restored = 0
    for step in range(step0 + 1, args.steps + 1):
        if opt is None:
            break
        au, gt = batch(streams, order, step, args.batch, dev)
        loss = net.loss(au, gt)
        opt.zero_grad(set_to_none=True)
        # NaN GUARD, kept from `train_slot_real.py`. Two C1 arms went non-finite
        # near step 400 and every later validation read NaN, so the run lost
        # 1100 steps. A non-finite loss skips the step; a non-finite parameter
        # after the step restores the last best head and resets the optimizer.
        if not torch.isfinite(loss):
            n_skipped += 1
            print(
                f"step {step:5d}  loss non-finite -> step skipped ({n_skipped} so far)", flush=True
            )
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            n_restored += 1
            print(
                f"step {step:5d}  non-finite parameter -> restored best.pt ({n_restored} so far)",
                flush=True,
            )
            if (out / "best.pt").exists():
                net.load_state_dict(torch.load(out / "best.pt", map_location=dev), strict=False)
            else:
                for p in params:
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
            opt.state.clear()
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
            frozen = bool(args.frozen_every) and n_val % max(1, args.frozen_every) == 0
            row = validate(net, sel, clips, dev, args, step, frozen=frozen)
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
            save_state(out, net, opt, step, best, hist)
            print(f"CHAIN: continue  (step {step}/{args.steps}, best {best:.4f})", flush=True)
            return 0

    # ── Report at the best checkpoint ────────────────────────────────────────
    if (out / "best.pt").exists():
        net.load_state_dict(torch.load(out / "best.pt", map_location=dev), strict=False)
    report: dict[str, Any] = {
        **config,
        "steps": done,
        "trainable": n_par,
        "select_best": best,
        "skipped_steps": n_skipped,
        "restored_steps": n_restored,
        "history": hist,
    }
    if clips:
        scorer = sr.score_real if args.multi_mic else sr.score_real_mono
        final = scorer(net, clips, device=dev, name=f"{args.name} best")
        final.pop("rows", None)
        report["frozen"] = final
    if args.val_parts:
        report["parts_eval"] = {
            n: sr.score_part(net, n, device=dev, n=args.val_part_n) for n in ("comb", "stoch")
        }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    if opt is not None:
        save_state(out, net, opt, done, best, hist)
    print(f"CHAIN: done  (best selection {best:.4f}) -> {out}", flush=True)
    return 0


def save_state(out: Path, net, opt, step: int, best: float, hist: list) -> None:
    """The resume file a chained segment restarts from."""
    torch.save(
        {
            "params": state_dict_trainable(net),
            "opt": opt.state_dict(),
            "step": step,
            "best": best,
            "history": hist,
            "torch_rng": torch.get_rng_state(),
        },
        out / "state.pt",
    )
    (out / "history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
