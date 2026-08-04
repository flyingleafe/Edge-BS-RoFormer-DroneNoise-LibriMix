#!/usr/bin/env python3
"""Generic SE scoring CLI — one method, one valid set, one tidy CSV.

One tool replaces the old per-purpose eval scripts. The mapping of their
primary invocations:

* ``eval_se_perclip.py --method f1_dcunet_a --valid SE-valid-drone`` →
  ``se_eval.py --method f1_dcunet_a --valid SE-valid-drone``
  (per-clip rows are the default granularity).
* ``eval_se_models.py --experiment f1_dcunet_a --valid SE-valid-drone
  --by-category`` →
  ``se_eval.py --method f1_dcunet_a --valid SE-valid-drone
  --granularity group --group-by category,input_snr``.
* ``eval_se_anchors.py --dataset SE-valid-drone`` → two runs,
  ``se_eval.py --method noisy --valid SE-valid-drone --granularity group``
  and the same with ``--method wiener``.
* ``diagnose_dcunet_loss.py --experiment f1_dcunet_a --valid SE-valid-drone``
  → ``se_eval.py --method f1_dcunet_a --valid SE-valid-drone
  --loss-breakdown --grad-norms --loss-cfg conf/loss/si_sdr_mrstft.yaml``.
* ``run_se_perclip_eval.sh`` / ``run_f2_perclip_eval.sh`` → a shell loop over
  ``se_eval.py --method <m> --valid <v> --r2-upload`` (the method × valid
  matrices live in the experiment docs, not in checked-in drivers).

Model methods load through ``zoo.load`` (Hydra experiment compose + local
``results/<exp>/best.ckpt`` else the R2 artifact store), so any experiment
with a ``conf/experiment/<name>.yaml`` works with no per-arch table. The
``--valid`` spec is ``NAME[@VERSION][#CATEGORY]``.

The metric set is the union the old scripts computed: ``si_sdr``/``sdr``
plus ``gain_db``/``corr`` (a near-null estimate and an over-loud one are
indistinguishable from (sdr, si_sdr) alone — the quadratic in ``||est||``
has two roots), ``pesq`` (wideband at 16 kHz), ``pesq_nb`` (resampled to
8 kHz — the 2023 survey's narrowband scale) and ``estoi``.

Importable pieces: :func:`compute_metrics`, :func:`parse_valid`,
:func:`load_valid`, :func:`fetch_checkpoint`, :func:`model_estimates`,
:func:`anchor_estimates`, :func:`group_rows`.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SR = 16000
ANCHORS = ("noisy", "wiener")
METRICS = ["si_sdr", "sdr", "pesq", "pesq_nb", "estoi", "gain_db", "corr"]
CLIP_FIELDS = ["method", "valid", "clip_id", "category", "input_snr"]


# ── metrics (the ONE implementation) ────────────────────────────────────────
def compute_metrics(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> dict[str, float]:
    """All seven SE metrics for one (reference, estimate) pair, with guards.

    PESQ / eSTOI failures (too-short or silent clips) come back as NaN
    instead of raising, and both signals are flattened + truncated to the
    common length first.
    """
    from metrics.separation import pesq, sdr, si_sdr, stoi

    ref = np.asarray(ref, np.float32).reshape(-1)
    est = np.asarray(est, np.float32).reshape(-1)
    n = min(ref.shape[0], est.shape[0])
    ref, est = ref[:n], est[:n]
    ref_e = float(np.sum(ref**2))
    est_e = float(np.sum(est**2))
    eps = 1e-12
    out = {
        "si_sdr": float(si_sdr(ref[None, :], est[None, :])),
        "sdr": float(np.asarray(sdr(ref[None, None, :], est[None, None, :])).reshape(-1)[0]),
        "gain_db": 10.0 * float(np.log10((est_e + eps) / (ref_e + eps))),
        "corr": float(abs(np.dot(ref, est)) / (np.sqrt(ref_e * est_e) + eps)),
    }
    try:
        out["pesq"] = float(pesq(ref, est, sr))
    except Exception:
        out["pesq"] = float("nan")
    try:
        out["estoi"] = float(stoi(ref, est, sr, extended=True))
    except Exception:
        out["estoi"] = float("nan")
    # PESQ-NB at 8 kHz: the like-for-like column against 8 kHz literature
    # (PESQ-NB scores systematically higher than PESQ-WB on identical audio).
    try:
        import librosa

        out["pesq_nb"] = float(
            pesq(
                librosa.resample(ref, orig_sr=sr, target_sr=8000),
                librosa.resample(est, orig_sr=sr, target_sr=8000),
                8000,
            )
        )
    except Exception:
        out["pesq_nb"] = float("nan")
    return out


# ── valid-set resolution ────────────────────────────────────────────────────
def parse_valid(spec: str) -> tuple[str, str | None, str | None]:
    """``NAME[@VERSION][#CATEGORY]`` → ``(name, version, category)``."""
    spec = str(spec).strip()
    if not spec:
        raise ValueError("empty --valid spec")
    category = None
    if "#" in spec:
        spec, _, category = spec.partition("#")
        category = category or None
    version = None
    if "@" in spec:
        spec, _, version = spec.partition("@")
        version = version or None
    if not spec:
        raise ValueError("--valid spec has no dataset name")
    return spec, version, category


def load_valid(spec: str, sample_rate: int = SR) -> Any:
    """Instantiate the ``SEValidFrameDataset`` a ``--valid`` spec names."""
    from data_processing.frame_datasets import SEValidFrameDataset

    name, version, category = parse_valid(spec)
    return SEValidFrameDataset(name, version=version, sample_rate=sample_rate, category=category)


# ── estimates ───────────────────────────────────────────────────────────────
def fetch_checkpoint(experiment: str, dest: str | Path) -> str:
    """Return ``dest`` if it exists, else fetch the artifact-store copy.

    Resolves ``r2://ml-data/artifacts/<experiment>/checkpoints/<basename>``
    into the local checkpoint cache (kept for notebook callers that manage
    the model config themselves; the CLI path goes through ``zoo.load``).
    """
    from utils.checkpoints import resolve_checkpoint_uri

    dest = Path(dest)
    if dest.exists():
        return str(dest)
    return resolve_checkpoint_uri(f"r2://ml-data/artifacts/{experiment}/checkpoints/{dest.name}")


def _mixture(frame: Any) -> np.ndarray:
    return np.asarray(frame["mixture"].data, np.float32).reshape(-1)


def _disable_flash_attention(model: torch.nn.Module) -> None:
    """Best-effort CPU fallback: flash-attention kernels are CUDA-only."""
    for module in model.modules():
        if hasattr(module, "flash"):
            with contextlib.suppress(Exception):
                setattr(module, "flash", False)  # noqa: B010 - Module.__setattr__ is typed Tensor|Module


def anchor_estimates(method: str, ds: Any, idxs: list[int]) -> list[np.ndarray]:
    """``noisy`` (identity) and ``wiener`` (scipy classical filter) anchors."""
    from scipy.signal import wiener

    out: list[np.ndarray] = []
    for i in idxs:
        mix = _mixture(ds[i])
        if method == "noisy":
            out.append(mix)
        else:
            w = wiener(mix).astype(np.float32)
            out.append(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0))
    return out


def model_estimates(
    method: str,
    ds: Any,
    idxs: list[int],
    batch_size: int,
    device: torch.device,
    ckpt: str = "best",
) -> list[np.ndarray]:
    """Run experiment ``method``'s checkpoint over the selected clips."""
    import zoo

    fm = zoo.load(method, ckpt=ckpt, device=device)
    if device.type == "cpu":
        _disable_flash_attention(fm.model)
    mixes = [torch.as_tensor(_mixture(ds[i])) for i in idxs]
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(mixes), batch_size):
            batch = torch.stack(mixes[start : start + batch_size]).to(device)  # (b, T)
            est = fm.codec.call_model(fm.model, {"mixture": batch})
            est = np.asarray(est.detach().cpu()).reshape(len(batch), -1)
            out.extend(est[j] for j in range(est.shape[0]))
    return out


# ── row assembly ────────────────────────────────────────────────────────────
def clip_rows(
    method: str, valid: str, ds: Any, idxs: list[int], ests: list[np.ndarray]
) -> list[dict[str, Any]]:
    from data_processing.frames import get_meta

    rows = []
    for i, est in zip(idxs, ests):
        fr = ds[i]
        tgt = np.asarray(fr["target"].data, np.float32).reshape(-1)
        rows.append(
            {
                "method": method,
                "valid": valid,
                "clip_id": str(get_meta(fr, "id", i)),
                "category": str(get_meta(fr, "category", "all")),
                "input_snr": float(get_meta(fr, "input_snr")),
                **compute_metrics(tgt, est),
            }
        )
    return rows


def group_rows(
    rows: list[dict[str, Any]],
    group_by: list[str],
    metrics: list[str] = METRICS,
) -> list[dict[str, Any]]:
    """Aggregate per-clip rows to per-group nan-means, plus an ``n`` count."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_by)].append(r)
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        members = groups[key]
        agg: dict[str, Any] = dict(zip(group_by, key))
        agg["method"] = members[0].get("method")
        agg["valid"] = members[0].get("valid")
        agg["n"] = len(members)
        for m in metrics:
            vals = [r[m] for r in members if m in r]
            agg[m] = float(np.nanmean(vals)) if vals else float("nan")
        out.append(agg)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: f"{v:.4f}" if isinstance(v, float) and k in METRICS else v
                    for k, v in r.items()
                }
            )


# ── loss diagnostics (the old diagnose_dcunet_loss) ─────────────────────────
def loss_breakdown(
    method: str,
    valid: str,
    ds: Any,
    device: torch.device,
    loss_cfg: str,
    per_group: int,
    grad_norms: bool,
    ckpt: str = "best",
) -> None:
    """Per-SNR loss-term values (and gradient norms w.r.t. the model output).

    Answers "which term drives the update": prints each CompositeLoss term's
    weighted value, its gradient norm at the model output, the
    enhanced/target energy ratio (over-attenuation shows as < 1) and SI-SDR.
    """
    from omegaconf import OmegaConf

    import zoo
    from data_processing.collate import frame_collate
    from data_processing.frames import get_meta
    from losses._common import get_tensor
    from training.config import build_losses

    fm = zoo.load(method, ckpt=ckpt, device=device)
    if device.type == "cpu":
        _disable_flash_attention(fm.model)
    composite = build_losses(OmegaConf.load(loss_cfg)).to(device)

    by_snr: dict[float, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        snr = float(get_meta(ds[i], "input_snr"))
        if len(by_snr[snr]) < per_group:
            by_snr[snr].append(i)

    names = list(composite._order)  # noqa: SLF001 - diagnostic introspection
    head = f"{'SNR':>5} | " + " ".join(f"{n + '_term':>12}" for n in names)
    if grad_norms:
        head += " | " + " ".join(f"{'|g|_' + n:>12}" for n in names)
    head += f" | {'E_enh/E_tgt':>11} | {'SI-SDR':>7}"
    print(f"# {method} on {valid} — loss={loss_cfg}")
    print(head)
    for snr in sorted(by_snr):
        batch = frame_collate([ds[i] for i in by_snr[snr]]).map_data(lambda t: t.to(device))
        inputs = fm.codec.to_inputs(batch)
        enhanced = fm.codec.call_model(fm.model, inputs)  # differentiable
        if not torch.is_tensor(enhanced):
            enhanced = get_tensor(enhanced, "enhanced")
        enhanced = enhanced.reshape(len(by_snr[snr]), -1)
        enhanced.requires_grad_(True)
        pred_frame = fm.codec.to_frame(enhanced, batch)

        term_vals, term_grads = {}, {}
        for name in names:
            fn = (
                composite._loss_modules[name]  # noqa: SLF001
                if name in composite._loss_modules  # noqa: SLF001
                else composite._plain[name]  # noqa: SLF001
            )
            v = fn(pred_frame, batch) * composite._weights[name]  # noqa: SLF001
            term_vals[name] = float(v.detach())
            if grad_norms:
                g = torch.autograd.grad(v, enhanced, retain_graph=True)[0]
                term_grads[name] = float(g.norm())

        with torch.no_grad():
            est = enhanced.detach()
            tgt = get_tensor(batch, "target").reshape(len(by_snr[snr]), -1)
            n = min(est.shape[-1], tgt.shape[-1])
            e_ratio = float(
                (est[..., :n].pow(2).mean() / tgt[..., :n].pow(2).mean().clamp_min(1e-12)).sqrt()
            )
            sisdr = float(
                np.mean(
                    [
                        compute_metrics(tgt[j, :n].cpu().numpy(), est[j, :n].cpu().numpy())[
                            "si_sdr"
                        ]
                        for j in range(est.shape[0])
                    ]
                )
            )
        line = f"{snr:5.0f} | " + " ".join(f"{term_vals[n]:12.4f}" for n in names)
        if grad_norms:
            line += " | " + " ".join(f"{term_grads[n]:12.4f}" for n in names)
        line += f" | {e_ratio:11.3f} | {sisdr:7.2f}"
        print(line)


# ── CLI ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--method", required=True, help="noisy | wiener | <experiment name>")
    ap.add_argument("--valid", required=True, help="SE valid set: NAME[@VERSION][#CATEGORY]")
    ap.add_argument("--granularity", choices=("clip", "group"), default="clip")
    ap.add_argument(
        "--group-by",
        default="category,input_snr",
        help="comma-separated meta fields for --granularity group",
    )
    ap.add_argument("--batch", type=int, default=8, help="model forward batch size")
    ap.add_argument("--limit", type=int, default=0, help="cap n clips (0=all; quick checks)")
    ap.add_argument(
        "--per-group",
        type=int,
        default=0,
        help="balanced subsample: cap clips per (category, input_snr) group (0=all)",
    )
    ap.add_argument("--ckpt", default="best", help="checkpoint: name | path | r2:// URI")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument(
        "--r2-upload",
        action="store_true",
        help="upload the CSV to artifacts/<method>/<granularity>/<basename> via ArtifactStore",
    )
    ap.add_argument(
        "--loss-breakdown",
        action="store_true",
        help="print per-SNR loss-term values instead of writing a CSV (experiment methods only)",
    )
    ap.add_argument(
        "--grad-norms", action="store_true", help="with --loss-breakdown: per-term grad norms"
    )
    ap.add_argument(
        "--loss-cfg", default="conf/loss/si_sdr_mrstft.yaml", help="loss YAML for --loss-breakdown"
    )
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = load_valid(args.valid)

    if args.loss_breakdown or args.grad_norms:
        if args.method in ANCHORS:
            raise SystemExit("--loss-breakdown needs an experiment method (a trained model)")
        loss_breakdown(
            args.method,
            args.valid,
            ds,
            device,
            args.loss_cfg,
            args.per_group or 16,
            grad_norms=args.grad_norms,
            ckpt=args.ckpt,
        )
        return

    from data_processing.frames import get_meta

    idxs = list(range(len(ds) if not args.limit else min(args.limit, len(ds))))
    if args.per_group:
        seen: dict[tuple, int] = defaultdict(int)
        kept = []
        for i in idxs:
            key = (str(get_meta(ds[i], "category", "all")), float(get_meta(ds[i], "input_snr")))
            if seen[key] < args.per_group:
                seen[key] += 1
                kept.append(i)
        idxs = kept

    if args.method in ANCHORS:
        ests = anchor_estimates(args.method, ds, idxs)
    else:
        ests = model_estimates(args.method, ds, idxs, args.batch, device, ckpt=args.ckpt)

    rows = clip_rows(args.method, args.valid, ds, idxs, ests)
    if args.granularity == "group":
        rows = group_rows(rows, [k.strip() for k in args.group_by.split(",") if k.strip()])

    out_path = Path(
        args.out
        or f"results/se_eval/{args.method}__{args.valid.replace('#', '_')}__{args.granularity}.csv"
    )
    _write_csv(out_path, rows)
    print(f"wrote {out_path} ({len(rows)} rows)", flush=True)
    if args.granularity == "group":
        # Echo grouped CSVs so results are recoverable from `omnirun logs`.
        print("----- CSV -----\n" + out_path.read_text() + "----- END CSV -----", flush=True)
    if args.r2_upload:
        from training.artifacts import ArtifactStore

        store = ArtifactStore(experiment_name=args.method)
        uri = store.upload_file(out_path, f"{args.granularity}/{out_path.name}")
        if uri:
            print(f"uploaded -> {uri}", flush=True)


if __name__ == "__main__":
    main()
