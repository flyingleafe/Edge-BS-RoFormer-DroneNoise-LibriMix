#!/usr/bin/env python
"""Train the slot-comb CRF v2 — `scripts/train_slot_real.py` with the v2 groups.

WHAT IS NEW, AND WHY EACH GROUP HAS A FLAG. `docs/slot-comb-v2-design.md` maps
every regime the C1 arms lose to ONE parameter group, and each group is a family
that contains the current setting at initialization. The ablation the design
asks for is "one part at a time from the corner", so every group is a flag here
and the corner is every flag off. With no v2 flag this script IS
`scripts/train_slot_real.py --mono`: the same model, the same policy, the same
selection set size, the same chain markers.

    --off-state                     § 3.1  one OFF state per chain; zero frames
    --grid-lo 10 --n-grid 900       § 3.2  the grid below 30 rev/s
    --learned-transition            § 3.3  the hinge becomes a learned penalty
    --emission v2 --v2-parts ...    § 3.4/3.5/3.7  gap, cross-order net, widths
    --rate-prior                    § 3.6  the pairwise prior across slots

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

TWO SETTINGS A V2 ARM MUST NOT FORGET, both defaulted here.

* `--warm-start` (ON whenever the emission is v2). At the exact corner the gap
  charge and the read width have vanishing gradients — about 1e-7 and 1e-13 —
  so an arm that must LEARN them starts at `gap_mu = -2.0` and
  `read_sigma = 0.7`, which is the decision campaign arm A7 made for the octave
  charge. `--no-warm-start` keeps the exact corner, for a parity test.
* `--mask-below-grid` (ON whenever the grid floor is under 30 rev/s). A frame
  between `zero_rps` and the grid floor has no gold cell, so it leaves the gold
  path instead of being charged at the grid edge. Evaluation still counts it as
  an error.

`--model-kwarg NAME=VALUE` passes any `SlotCombNet` keyword through, and the
signature is checked before anything is built, so a group that is renamed
costs one command-line argument and not an edit.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Short aliases for the v2 emission parts, so `--v2-parts read_width` names
#: `read_width_learned`. The model's own spelling is always accepted.
V2_ALIASES = {"read_width": "read_width_learned", "claim_width": "claim_width_learned"}
#: The two values the warm start moves off the corner (see the module docstring).
WARM_OFF_THETA0 = 0.0
WARM_OFF_THETA1 = 1.0
WARM_OFF_C = 1.0
WARM_GAP_MU, WARM_READ_SIGMA = -2.0, 0.7
#: The grid floor under which the below-grid mask is turned on by default.
GRID_FLOOR = 30.0


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


def emission_parts(args) -> list[str]:
    """The one ``parts`` tuple `SlotCombNet` takes, over both vocabularies.

    `PartialEmissionV2` reads ONE list holding the four C1 parts and the four v2
    parts, so `--parts` and `--v2-parts` are two selectors over one vocabulary
    and not two arguments. The v2 half is added only when the emission is v2.
    """
    parts = [p for p in args.parts.split(",") if p and p != "none"]
    if args.emission == "v2":
        parts += [V2_ALIASES.get(p, p) for p in args.v2_parts.split(",") if p and p != "none"]
    if not args.multi_mic and "channels" in parts:
        # A mono input has ONE channel and it is already the power mean, so the
        # per-mic candidates the `channels` part adds do not exist.
        parts = [p for p in parts if p != "channels"]
        print("mono: dropped the 'channels' part (a mono input has no extra mics)", flush=True)
    return parts


def model_kwargs(args) -> dict[str, Any]:
    """The full `SlotCombNet` constructor call, as a JSON-safe dict.

    This dict IS `config.json`'s ``model`` block, so what trains and what a dump
    or a probe rebuilds cannot drift.
    """
    parts = emission_parts(args)
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
        "parts": parts,
        "zero_rps": float(args.zero_rps),
    }
    if args.mask_k_max:
        kw["mask_k_max"] = int(args.mask_k_max)
    if args.off_state:
        kw["off_state"] = True
    if args.learned_transition:
        kw["learned_transition"] = True
        kw["trans_slew"] = float(args.trans_slew)
    if args.rate_prior:
        kw["rate_prior"] = True
    if mask_below_grid(args):
        kw["mask_below_grid"] = True
    kw.update(parse_kv(args.model_kwarg, "model-kwarg"))
    return kw


def mask_below_grid(args) -> bool:
    """Whether the frames between ``zero_rps`` and the grid floor leave the loss.

    ON by default whenever the grid reaches under 30 rev/s, because that is the
    only reason to extend it: those frames have no gold cell, so charging them
    at the grid edge would teach the edge. ``--no-mask-below-grid`` forces it
    off, and it is meaningless at the C1 grid, where the sampler rejects such a
    crop anyway.
    """
    if args.no_mask_below_grid:
        return False
    return bool(args.mask_below_grid or float(args.grid_lo) < GRID_FLOOR)


# ── W&B ───────────────────────────────────────────────────────────────────────
# The C1 trainer logged to files only. A v2 arm logs to W&B like train.py:
# one run per arm (`slotv2_<name>`), resumed across chain segments through
# the run id stored next to best.pt, with the file history backfilled on the
# first init so a segment that starts logging late shows the whole curve.
# Every call is guarded: a W&B failure never stops a training segment.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "flyingleafe")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "harmonic-noise-suppression")


class _WandB:
    def __init__(self, args, out: Path, kw: dict[str, Any], hist: list[dict]):
        self.run = None
        if getattr(args, "no_wandb", False):
            return
        try:
            import wandb

            id_path = out / "wandb_run_id.txt"
            fresh = not id_path.exists()
            run_id = uuid.uuid4().hex[:8] if fresh else id_path.read_text().strip()
            self.run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=f"slotv2_{args.name}",
                id=run_id,
                resume="allow",
                group="slot_v2",
                tags=["slot_v2", args.data],
                config={"model": kw, "train": vars(args)},
                dir=str(out),
            )
            if fresh:
                id_path.write_text(run_id)
                for i, row in enumerate(hist):
                    self.validation(row.get("step", i * args.val_every), row)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb: disabled ({exc})", flush=True)
            self.run = None

    def step(self, step: int, loss: float, s_per_step: float) -> None:
        if self.run is None:
            return
        try:
            import wandb

            wandb.log({"train/loss": loss, "train/s_per_step": s_per_step}, step=step)
        except Exception:  # noqa: BLE001
            pass

    def validation(self, step: int, row: dict, net=None) -> None:
        if self.run is None:
            return
        try:
            import wandb

            data: dict[str, Any] = {"val/select": row.get("select")}
            if net is not None:
                data.update(param_summary(net))
            for part in ("comb", "stoch"):
                if isinstance(row.get(part), dict):
                    data[f"val/{part}_mean"] = row[part].get("mean")
            fr = row.get("frozen")
            if isinstance(fr, dict):
                for k, v in fr.items():
                    if isinstance(v, (int, float)):
                        data[f"frozen/{k}"] = v
            if row.get("loss") is not None:
                data["train/loss_at_val"] = row["loss"]
            wandb.log({k: v for k, v in data.items() if v is not None}, step=step)
        except Exception:  # noqa: BLE001
            pass

    def finish(self, best: float, done: bool) -> None:
        if self.run is None:
            return
        try:
            import wandb

            self.run.summary["select_best"] = best
            self.run.summary["chain_done"] = done
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass


def param_summary(net) -> dict[str, float]:
    """A few scalars per parameter group, so a run's curve says WHICH group moved."""
    out: dict[str, float] = {}
    for name in ("theta0", "theta1", "c1", "c2"):
        q = getattr(net, name, None)
        if isinstance(q, torch.Tensor):
            out[f"param/{name}"] = float(q.detach())
    tr = getattr(net, "trans", None)
    if tr is not None and hasattr(tr, "d"):
        out["param/trans_d_sum"] = float(torch.nn.functional.softplus(tr.d.detach()).sum())
    em = getattr(net, "emit", None)
    if em is not None:
        for name in ("mu", "s0", "s1"):
            q = getattr(em, name, None)
            if isinstance(q, torch.Tensor) and q.numel() == 1:
                out[f"param/emit_{name}"] = float(q.detach())
        for name in ("alpha", "a"):
            q = getattr(em, name, None)
            if isinstance(q, torch.Tensor):
                out[f"param/emit_{name}_mean"] = float(q.detach().float().mean())
        mb = getattr(em, "masks", None)
        if mb is not None and hasattr(mb, "width_raw"):
            out["param/claim_width_raw"] = float(mb.width_raw.detach())
    pr = getattr(net, "rate_prior", None)
    if pr is not None and hasattr(pr, "v"):
        out["param/prior_v_abs_sum"] = float(pr.v.detach().abs().sum())
    return out


def use_warm_start(args) -> bool:
    """Whether the two gradient-starved emission knobs start off the corner."""
    if args.no_warm_start:
        return False
    return bool(args.warm_start or args.emission == "v2")


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


def build_model(kw: dict[str, Any], device: str, warm: bool = False):
    """`SlotCombNet` from the constructor keywords, warm-started if asked.

    `SlotCombNet.__init__` does not forward keywords to its emission, so the
    warm start is a call on the built object — see the module docstring for why
    a v2 arm needs one.
    """
    from experiments import slot_v2 as sv

    check_signature(kw)
    net = sv.build_from_config({"model": kw}, device=device)
    if warm and kw.get("off_state"):
        # THE OFF STATE MUST START REACHABLE TO TRAIN. At the corner
        # `theta0 = -1e4` prices a gold OFF frame at 1e4 nats, so a crop with
        # stopped rotors returns a loss near 1e6 and dominates every gradient,
        # while Adam moves `theta0` by about the learning rate per step (a
        # 1e7-step climb). Measured 2026-09-06 on the static-comb stage:
        # selection 17.4 -> 11.2 -> 18.8 with losses of 1e6 at steps 275 and
        # 375. The warm start puts the OFF unary at minus the contrast
        # statistic (`theta0 = 0`, `theta1 = 1`): below the best ON score on
        # a rotor frame (contrast 1-2) and near it on a no-rotor frame
        # (contrast 0.24-0.30), with one nat to enter or leave.
        with torch.no_grad():
            net.theta0.fill_(WARM_OFF_THETA0)
            net.theta1.fill_(WARM_OFF_THETA1)
            net.c1.fill_(WARM_OFF_C)
            net.c2.fill_(WARM_OFF_C)
        print(
            f"warm start: OFF state theta0={WARM_OFF_THETA0} theta1={WARM_OFF_THETA1} "
            f"c1=c2={WARM_OFF_C}",
            flush=True,
        )
    if warm and kw.get("emission") == "v2":
        from models.comb_slots_emission_v2 import PartialEmissionV2
        from models.comb_slots_emission_v2 import warm_start as warm_start_emission

        assert isinstance(net.emit, PartialEmissionV2)
        warm_start_emission(net.emit, gap_mu=WARM_GAP_MU, read_sigma=WARM_READ_SIGMA)
        print(
            f"warm start: gap_mu={WARM_GAP_MU} read_sigma={WARM_READ_SIGMA} "
            "(the corner's gradients on these two are ~1e-7 and ~1e-13)",
            flush=True,
        )
    return net


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
    § 3.2: the zero and below-grid frames must reach the loss, and
    `SlotCombNet.loss` is what decides what each of them means. Without an OFF
    state, the C1 window, one grid step inside each edge — a model that cannot
    express a zero would otherwise learn to put it at the grid edge.
    """
    from experiments.slot_v2 import FULL_RANGE

    if args.grid_range or not (args.full_range or kw.get("off_state")):
        return (float(kw["r_lo"]) + 1.0, float(kw["r_hi"]) - 1.0)
    # Two honest warnings, not refusals: a full-range smoke is a useful check of
    # the sampler, and an arm may want one of the two masks alone.
    if not kw.get("off_state"):
        print(
            "WARNING: --full-range without --off-state. A zero-rate frame has no gold "
            "state, so the loss will put it at the grid edge. Use --off-state.",
            flush=True,
        )
    elif not kw.get("mask_below_grid"):
        print(
            f"WARNING: --off-state without the below-grid mask. A frame between "
            f"{kw.get('zero_rps', 0.5)} and {kw['r_lo']:g} rev/s is charged at the grid "
            "edge. Use --mask-below-grid, or lower --grid-lo.",
            flush=True,
        )
    return FULL_RANGE


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
    from models.comb_slots import PARTIAL_PARTS
    from models.comb_slots_emission_v2 import V2_PARTS

    g.add_argument("--off-state", action="store_true", help="§ 3.1 one OFF state per chain")
    g.add_argument("--learned-transition", action="store_true", help="§ 3.3 learned penalty")
    g.add_argument(
        "--trans-slew",
        type=float,
        default=30.0,
        help="§ 3.3 the slew, in rev/s^2, that sets the learned band's WIDTH. The "
        "hinge the parameters start at still comes from --slew",
    )
    g.add_argument("--rate-prior", action="store_true", help="§ 3.6 pairwise prior across slots")
    g.add_argument(
        "--emission",
        default="partial",
        choices=("classical", "partial", "v2"),
        help="partial = the C1 emission (default); v2 adds the parts of --v2-parts",
    )
    g.add_argument(
        "--v2-parts",
        default=",".join(V2_PARTS),
        help=f"comma list of {list(V2_PARTS)}, or 'none'; read only with --emission v2. "
        "'read_width' and 'claim_width' are accepted as short names",
    )
    g.add_argument(
        "--parts",
        default=",".join(PARTIAL_PARTS),
        help=f"C1 emission parts {list(PARTIAL_PARTS)}, or 'none'",
    )
    g.add_argument("--grid-lo", type=float, default=30.0, help="§ 3.2 the grid floor, rev/s")
    g.add_argument("--grid-hi", type=float, default=100.0)
    g.add_argument("--n-grid", type=int, default=700, help="§ 3.2 use 900 with --grid-lo 10")
    g.add_argument(
        "--zero-rps",
        type=float,
        default=0.5,
        help="a true rate at or under this is a stopped rotor: the gold state is OFF",
    )
    g.add_argument(
        "--mask-below-grid",
        action="store_true",
        help="§ 3.2 drop the frames between --zero-rps and the grid floor from the "
        "gold path. ON by default when --grid-lo is under 30",
    )
    g.add_argument("--no-mask-below-grid", action="store_true", help="force the mask off")
    g.add_argument(
        "--warm-start",
        action="store_true",
        help="start the gap charge and the read width off the corner. ON by "
        "default with --emission v2, where their corner gradients are ~1e-7 and ~1e-13",
    )
    g.add_argument("--no-warm-start", action="store_true", help="keep the exact corner")
    g.add_argument("--k-max", type=int, default=40)
    g.add_argument("--floor-hz", type=float, default=60.0)
    g.add_argument(
        "--mask-k-max",
        type=int,
        default=0,
        help="0 = the model's own f_max / grid_lo. At --grid-lo 10 that is 750 "
        "harmonics and a 121 s bank build, so cap it for a short run",
    )
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
    t.add_argument("--no-wandb", action="store_true", help="file logging only")
    t.add_argument(
        "--min-in-grid",
        type=float,
        default=0.5,
        help="keep a crop only if this fraction of its rotor-frames is inside the grid "
        "(the balance rule against silence-heavy pools; 0 = keep every crop)",
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
    warm = use_warm_start(args)
    net = build_model(kw, dev, warm=warm)
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
        f"off_state={kw.get('off_state', False)} "
        f"mask_below_grid={kw.get('mask_below_grid', False)} "
        f"learned_transition={kw.get('learned_transition', False)} "
        f"rate_prior={kw.get('rate_prior', False)} trainable={n_par}",
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
            "warm_start": warm,
            "smoke": bool(args.smoke),
        },
    }
    sv.save_config(out, config)

    # ONE learning rate. A 10x group for the chain scalars (tried 2026-09-06)
    # let theta0 run away within 250 steps on the real pool.
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
        min_in_grid=args.min_in_grid,
        grid=(float(kw["r_lo"]), float(kw["r_hi"])),
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

    wb = _WandB(args, out, kw, hist)

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
            wb.step(step, float(loss.item()), (time.time() - t0) / max(1, step - step0))
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
            row["step"] = step
            wb.validation(step, row, net)
            net.train()
        if time.time() - t_start > budget and step < args.steps:
            save_state(out, net, opt, step, best, hist)
            print(f"CHAIN: continue  (step {step}/{args.steps}, best {best:.4f})", flush=True)
            wb.finish(best, done=False)
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
    wb.finish(best, done=True)
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
