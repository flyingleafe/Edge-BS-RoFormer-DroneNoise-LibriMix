#!/usr/bin/env python3
"""Inspect a trained checkpoint without a dataset or a training run.

Checkpoint reference forms (``--ckpt``):

* a local path (``results/<exp>/best.ckpt``),
* an ``r2://<bucket>/<key>`` URI (downloaded once into the checkpoint cache),
* ``zoo:<experiment>[/<file>]`` — the artifact-store convention
  ``r2://ml-data/artifacts/<experiment>/checkpoints/<file or best.ckpt>``;
  the only form ``--report variance_share`` accepts, since that report must
  BUILD the model (via ``zoo.load``) and run a forward pass.

Reports:

* ``params`` (was ``probe_wind_params.py``): every state-dict tensor
  matching ``--filter``; near-scalars print raw + softplus values (the
  ``raw_*`` convention), larger tensors print shape and |mean|.
* ``variance_share`` (was ``probe_wind_share.py``): for a generative noise
  model with per-channel ``spectral_stats``, the share of the predicted PSD
  each channel carries (a channel at ~0% or ~100% makes any A/B against a
  control degenerate — see GOALS.md's wind-comparison gate). Synthetic
  quad geometry, RPS pinned at 80 rev/s.
* ``spectra``: per-matrix singular-value summary (top σ, σ-decay) of every
  ≥2-D tensor matching ``--filter`` — a quick collapsed-vs-spread look at
  what a layer learned.

Examples::

    python scripts/probe_ckpt.py --ckpt zoo:gen_w4_lik_wind_mm --report params --filter .wind.
    python scripts/probe_ckpt.py --ckpt zoo:gen_h1_hybrid_wind --report variance_share
    python scripts/probe_ckpt.py --ckpt results/f1_dcunet_a/best.ckpt --report spectra
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SR = 16000


def parse_ref(ref: str) -> tuple[str | None, str]:
    """``--ckpt`` ref → ``(experiment or None, path-or-uri)``."""
    if ref.startswith("zoo:"):
        exp, _, filename = ref[len("zoo:") :].partition("/")
        if not exp:
            raise ValueError(f"empty experiment in {ref!r}")
        return exp, f"r2://ml-data/artifacts/{exp}/checkpoints/{filename or 'best.ckpt'}"
    return None, ref


def load_state(ref: str) -> dict[str, torch.Tensor]:
    from utils.checkpoints import resolve_checkpoint_uri

    _, uri = parse_ref(ref)
    state: Any = torch.load(resolve_checkpoint_uri(uri), map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


# ── reports ─────────────────────────────────────────────────────────────────
def report_params(state: dict[str, torch.Tensor], name_filter: str) -> None:
    total = sum(v.numel() for v in state.values())
    matched = {k: v for k, v in state.items() if name_filter in k}
    print(f"{len(matched)}/{len(state)} tensors match {name_filter!r}; total params {total:,}")
    for name, value in sorted(matched.items()):
        if value.numel() <= 4:
            raw = [float(x) for x in value.reshape(-1)]
            softplus = [float(F.softplus(torch.tensor(x))) for x in raw]
            print(
                f"  {name:56s} raw={', '.join(f'{x:+9.4f}' for x in raw)}  "
                f"softplus={', '.join(f'{x:10.5f}' for x in softplus)}"
            )
        else:
            print(
                f"  {name:56s} {str(tuple(value.shape)):16s} "
                f"|mean|={float(value.abs().mean()):.6f} std={float(value.std()):.6f}"
            )


def report_spectra(state: dict[str, torch.Tensor], name_filter: str) -> None:
    print(f"{'tensor':56s} {'shape':>16s} {'top σ':>10s} {'σ2/σ1':>7s} {'σ-tail':>7s}")
    for name, value in sorted(state.items()):
        if name_filter not in name or value.ndim < 2:
            continue
        mat = value.reshape(value.shape[0], -1).float()
        s = torch.linalg.svdvals(mat)
        top = float(s[0])
        ratio = float(s[1] / s[0]) if s.numel() > 1 and top > 0 else float("nan")
        tail = float(s[s.numel() // 2 :].sum() / s.sum()) if top > 0 else float("nan")
        print(f"{name:56s} {str(tuple(value.shape)):>16s} {top:10.4f} {ratio:7.3f} {tail:7.3f}")


def report_variance_share(ref: str) -> None:
    """Per-channel share of the predicted noise PSD (generative models)."""
    import zoo

    experiment, _ = parse_ref(ref)
    if experiment is None:
        raise SystemExit(
            "--report variance_share needs a zoo:<experiment> ref (it builds the model)"
        )
    _, _, filename = ref[len("zoo:") :].partition("/")
    fm = zoo.load(experiment, ckpt=(Path(filename).stem if filename else "best"))
    model: Any = fm.model
    if not hasattr(model, "spectral_stats"):
        raise SystemExit(
            f"{type(model).__name__} has no spectral_stats — not a generative noise model"
        )

    generator: Any = getattr(model, "generator", model)
    wind: Any = getattr(generator, "wind", None)
    if wind is not None:
        scalars = {
            n[len("raw_") :]: float(F.softplus(p.detach()).reshape(-1)[0])
            for n, p in wind.named_parameters()
            if n.startswith("raw_") and p.numel() == 1
        }
        if scalars:
            print("learned wind scalars: " + "  ".join(f"{k}={v:.4f}" for k, v in scalars.items()))

    # Synthetic quad geometry: 4 rotors at ±20 cm, an 8-mic body-centred cloud.
    torch.manual_seed(0)
    rotor = torch.tensor([[0.2, 0.2, 0.0], [-0.2, 0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]])
    mic = torch.randn(8, 3) * 0.03
    rel = (mic[None, :, None, :] - rotor[None, None, :, :]).contiguous()
    rps = torch.full((1, 4, SR), 80.0)
    drone_names = list(getattr(model, "drone_names", None) or ["dregon"])
    for drone in drone_names:
        with torch.no_grad():
            total = float(model.spectral_stats(rps, rel, [drone])["noise_psd"].mean())
            wind_p = float(wind.expected_power_rel(rps, rel).mean()) if wind is not None else 0.0
        line = f"{drone:9s} total psd {total:12.4e}"
        if wind is not None:
            line += f" | wind {wind_p:12.4e} | wind share {wind_p / max(total, 1e-30):.3e}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--ckpt", required=True, help="r2://uri | local path | zoo:<experiment>[/<file>]"
    )
    ap.add_argument("--report", choices=("params", "variance_share", "spectra"), default="params")
    ap.add_argument("--filter", default="", help="substring filter on tensor names")
    args = ap.parse_args()

    print(f"checkpoint: {args.ckpt}")
    if args.report == "variance_share":
        report_variance_share(args.ckpt)
        return
    state = load_state(args.ckpt)
    if args.report == "params":
        report_params(state, args.filter)
    else:
        report_spectra(state, args.filter)


if __name__ == "__main__":
    main()
