"""Real 8-microphone windows and the frozen split, for the slot-comb emission (C1).

WHAT THIS IS FOR. `models.comb_slots.SlotCombNet` with `head_mode="classical"`,
`n_iter=0`, eight microphones power-averaged and a 15-bin (60 Hz) running-median
floor reads the frozen real clips at DREGON cruise 1.49 rev/s with ZERO trained
parameters — better than every trained model on that protocol (probe P1c,
`docs/rps-tracking-architecture-candidates.md` § 5). Candidate C1 trains a
learned emission on top of that exact corner. This module is the data and
scoring seam it needs:

* :func:`real_clips` — the frozen 37-clip split, eight microphones each, the
  same regrouping the probe used (`rps_bench.part("real")` holds one FRAME per
  microphone, so a clip is eight consecutive frames).
* :func:`score_real` / :func:`table` — the probe's reporting format, kept so a
  trained emission is read against 1.49 / 21 / 36 / 76 (DREGON cruise / FLY124
  cruise / ramp / ground) line for line.
* :func:`score_real_mono` — the SAME clips and the same aggregate, one
  microphone at a time. Eight microphones power-averaged is a lever no neural
  baseline has, so the eight-microphone table is not a comparison with them;
  the mono table is, frame for frame with `rps_bench.part("real")`.
* :func:`windows` — training crops from an online-mix policy, eight channels
  wide and label-aligned.
* :func:`select_set` — a FIXED set of real windows, drawn from the same policy
  with a different base seed, for checkpoint selection.

WHY SELECTION IS NOT THE FROZEN SPLIT. FLY124 is an unseen rig and the split is
the test set. The campaign has already been burned three times by monitors that
are not the task metric (wall W6), so selection runs the DEPLOYED decoder on its
own held-out windows, and the frozen split is scored at every validation for the
record only.

WHY A CROP IS REJECTED. The CRF loss needs a gold grid cell in every frame, and
the grid is 30-100 rev/s. A crop in which any rotor leaves ``[r_lo+1, r_hi-1]``
has no gold path, so it is drawn again. Ground, warm-up and the low half of a
ramp are therefore OUT OF SCOPE for this loss — a real limitation of training
through this decoder, not an oversight: a rotor at 3 rev/s puts its harmonics
3 Hz apart, below one 3.9 Hz analysis bin.
"""

from __future__ import annotations

import pickle
import time
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "POLICY_PARTIAL",
    "POLICY_REAL",
    "WindowStream",
    "real_clips",
    "score_part",
    "score_real",
    "score_real_mono",
    "select_set",
    "table",
    "windows",
]

POLICY_REAL = "conf/online_mix/slot_real_dload.yaml"
POLICY_PARTIAL = "conf/online_mix/slot_partial_dload.yaml"
CACHE = Path(".cache/slot_real")
SR, HOP = 16000, 512
_PERMS = list(permutations(range(4)))


# ─── The frozen split ─────────────────────────────────────────────────────────


def clip_phase(gt: np.ndarray) -> str:
    """``ground`` / ``ramp`` / ``cruise`` from the labels alone.

    The rule the probes used: every rotor stopped is ground, a mean per-rotor
    excursion over 15 rev/s is a ramp, everything else is cruise.
    """
    if np.all(gt.mean(axis=1) < 1.0):
        return "ground"
    if float((gt.max(axis=1) - gt.min(axis=1)).mean()) > 15.0:
        return "ramp"
    return "cruise"


def _rig_by_sample() -> dict[str, str]:
    """Sample id -> rig, read from the validation dataset's own metadata.

    A TRAP the probes already paid for: the FRAME's ``meta.recording_id`` is the
    SAMPLE id (``sample_00007``), not the recording. The recording each sample
    was cut from lives in the materialized dataset's ``metadata.json``, and
    without it every clip reads as one rig — which silently empties the DREGON
    column of the table this module exists to print.
    """
    import json

    from omegaconf import OmegaConf

    from data_processing.streams import resolve_source
    from experiments.rps_bench import PARTS

    cfg = OmegaConf.load(PARTS["real"][0])
    root = Path(resolve_source(str(cfg.valid.params.data_dir)))
    meta = json.loads((root / "metadata.json").read_text())
    rows = meta.get("valid") or next(iter(meta.values()))
    return {str(r["id"]): str(r.get("recording_id", "")) for r in rows}


def real_clips(*, cache: bool = True) -> list[dict[str, Any]]:
    """The 37 frozen real clips, eight microphones each.

    ``rps_bench.part("real")`` holds ONE FRAME PER MICROPHONE — frame ``i`` is
    clip ``i // 8``, microphone ``i % 8`` — so the clips are eight consecutive
    frames regrouped. Cached as one array because the regrouping costs an R2
    pull and 150 MB of decode.
    """
    disk = CACHE / "real_clips.npz"
    if cache and disk.exists():
        z = np.load(disk, allow_pickle=True)
        audio, rps, meta = z["audio"], z["rps"], list(z["meta"])
        return [
            {"audio": audio[i], "rps": rps[i].astype(np.float64), **meta[i]}
            for i in range(len(meta))
        ]

    from data_processing.frames import meta_dict
    from experiments import rps_bench as rb
    from metrics._common import get_array

    frames = rb.part("real")
    by_sample = _rig_by_sample()
    n_clip = len(frames) // 8
    n_mic = 8
    first = np.asarray(frames[0]["mixture"].data, dtype=np.float32).ravel()
    audio = np.zeros((n_clip, n_mic, first.size), dtype=np.float32)
    rps = np.zeros((n_clip, 4, frames[0]["rps"].data.shape[-1]), dtype=np.float32)
    meta: list[dict[str, Any]] = []
    for c in range(n_clip):
        for m in range(n_mic):
            y = np.asarray(frames[c * n_mic + m]["mixture"].data, dtype=np.float32).ravel()
            audio[c, m, : y.size] = y[: audio.shape[2]]
        gt = np.asarray(get_array(frames[c * n_mic], "rps"), dtype=np.float64)
        rps[c] = gt
        md = meta_dict(frames[c * n_mic])
        sample = str(md.get("recording_id", md.get("sample_id", "")))
        rec = by_sample.get(sample, sample)
        meta.append(
            {
                "clip": c,
                "phase": clip_phase(gt),
                # The DREGON recordings are the only ones whose id starts this
                # way; everything else in this split is the DJI M100 (FLY124).
                "rig": "DREGON" if rec.startswith("free-flight") else "FLY124",
                "recording_id": rec,
                "sample_id": sample,
            }
        )
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(disk, audio=audio, rps=rps, meta=np.array(meta, dtype=object))
    return [
        {"audio": audio[i], "rps": rps[i].astype(np.float64), **meta[i]} for i in range(len(meta))
    ]


# ─── Scoring ──────────────────────────────────────────────────────────────────


def _align(pred: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray]:
    """PIT MAE and the label rows permuted onto the prediction's rows."""
    from experiments.rps_bench import resample_like_metric

    g = resample_like_metric(np.asarray(gt, dtype=np.float64), pred.shape[-1])
    cost = np.abs(pred[:, None] - g[None, :]).mean(-1)
    best, bp = None, _PERMS[0]
    for p in _PERMS:
        tot = sum(cost[k, p[k]] for k in range(4))
        if best is None or tot < best:
            best, bp = tot, p
    return float(best or 0.0) / 4.0, np.stack([g[bp[k]] for k in range(4)])


def _ratios(rows: list[dict], rig: str) -> dict[str, float]:
    """Prediction / truth over the cruise rotor-frames of one rig.

    The three fractions are the octave read: ``one`` is the truth, ``half`` is
    the sub-harmonic the short-comb failure lands on, ``two`` the multiple.
    """
    rr = [
        np.asarray(r["pred"])[np.asarray(r["gt"]) > 1.0]
        / np.asarray(r["gt"])[np.asarray(r["gt"]) > 1.0]
        for r in rows
        if r["phase"] == "cruise" and r["rig"] == rig
    ]
    if not rr:
        return {
            "one": float("nan"),
            "half": float("nan"),
            "two": float("nan"),
            "median": float("nan"),
            "n": 0,
        }
    v = np.concatenate(rr)
    return {
        "one": float((np.abs(v - 1.0) <= 0.1).mean()),
        "half": float((np.abs(v - 0.5) <= 0.05).mean()),
        "two": float((np.abs(v - 2.0) <= 0.2).mean()),
        "median": float(np.median(v)),
        "n": int(v.size),
    }


def table(
    rows: list[dict], name: str = "model", floor_bins: int = 0, quiet: bool = False
) -> dict[str, Any]:
    """The probe's aggregate, printed and returned.

    Same columns as `p1c.py::table`, so a trained emission reads against the
    zero-parameter corner without a second format to reconcile.
    """
    out: dict[str, Any] = {"floor_bins": int(floor_bins)}
    for ph in ("ground", "ramp", "cruise"):
        v = [r["mae"] for r in rows if r["phase"] == ph]
        out[ph] = {
            "n": len(v),
            "mean": float(np.mean(v)) if v else float("nan"),
            "median": float(np.median(v)) if v else float("nan"),
        }
    for rig in ("DREGON", "FLY124"):
        v = [r["mae"] for r in rows if r["rig"] == rig and r["phase"] == "cruise"]
        out[rig + "_cruise"] = {
            "n": len(v),
            "mean": float(np.mean(v)) if v else float("nan"),
            "median": float(np.median(v)) if v else float("nan"),
        }
        out[rig + "_ratio"] = _ratios(rows, rig)
    allv = [r["mae"] for r in rows]
    out["all"] = {"mean": float(np.mean(allv)), "median": float(np.median(allv))}
    out["wall_mean"] = float(np.mean([r.get("wall", 0.0) for r in rows]))
    out["per_clip"] = {int(r["clip"]): round(float(r["mae"]), 3) for r in rows}
    if quiet:
        return out
    print(
        f"{'configuration':30s} {'bins':>5s} {'ground':>8s} {'ramp':>8s} {'cruise':>8s} "
        f"{'DRGcru':>8s} {'FLYcru':>8s} {'mean':>8s} {'med':>8s} {'s/clip':>7s}"
    )
    print(
        f"{name:30s} {out['floor_bins']:5d} {out['ground']['mean']:8.2f} "
        f"{out['ramp']['mean']:8.2f} {out['cruise']['mean']:8.2f} "
        f"{out['DREGON_cruise']['mean']:8.2f} {out['FLY124_cruise']['mean']:8.2f} "
        f"{out['all']['mean']:8.2f} {out['all']['median']:8.2f} {out['wall_mean']:7.1f}"
    )
    print(
        f"{'configuration':30s} {'DRG 1x':>7s} {'DRG 1/2':>8s} {'DRG med':>8s} "
        f"{'FLY 1x':>7s} {'FLY 1/2':>8s} {'FLY 2x':>7s} {'FLY med':>8s}"
    )
    d, f = out["DREGON_ratio"], out["FLY124_ratio"]
    print(
        f"{name:30s} {d['one']:7.3f} {d['half']:8.3f} {d['median']:8.3f} "
        f"{f['one']:7.3f} {f['half']:8.3f} {f['two']:7.3f} {f['median']:8.3f}"
    )
    return out


@torch.no_grad()
def score_real(
    net,
    clips: list[dict[str, Any]],
    device: str = "cpu",
    name: str = "model",
    quiet: bool = False,
    **decode_kw,
) -> dict[str, Any]:
    """Decode every frozen clip on eight microphones and aggregate.

    ``decode_kw`` defaults to the P1c optimum: sub-grid on, the decoder's own
    octave move OFF (it costs FLY124 22.4 -> 31.1 and buys nothing at eight
    microphones), relocation on.
    """
    from experiments.rps_bench import pit_mae

    kw = {"subgrid": True, "octave": False, "relocate": True, **decode_kw}
    was_training = net.training
    net.eval()
    rows = []
    for clip in clips:
        t0 = time.time()
        au = torch.as_tensor(clip["audio"], device=device)  # (8, N) -> one item
        pred = net.decode(au, **kw)[0].cpu().numpy().astype(np.float64)
        mae = pit_mae(pred, clip["rps"])
        _, gt = _align(pred, clip["rps"])
        rows.append(
            {
                "clip": clip["clip"],
                "phase": clip["phase"],
                "rig": clip["rig"],
                "mae": float(mae),
                "wall": time.time() - t0,
                "pred": pred,
                "gt": gt,
            }
        )
    net.train(was_training)
    out = table(rows, name=name, floor_bins=int(net.floor_bins), quiet=quiet)
    out["rows"] = [{k: v for k, v in r.items() if k not in ("pred", "gt")} for r in rows]
    return out


@torch.no_grad()
def score_real_mono(
    net,
    clips: list[dict[str, Any]],
    device: str = "cpu",
    name: str = "model",
    quiet: bool = False,
    **decode_kw,
) -> dict[str, Any]:
    """Decode every frozen clip ONE MICROPHONE AT A TIME and aggregate per frame.

    WHY THIS EXISTS. Every neural model this decoder is read against sees ONE
    microphone and is scored per mono frame — `rps_bench.part("real")` is 296
    mono frames, eight per clip. The eight-microphone power average of
    :func:`score_real` is a lever those models do not have, so its table is not
    a comparison with them. This function keeps the clips, the labels and the
    aggregate of :func:`score_real` and changes only the input: 8 decodes per
    clip, 296 rows, directly comparable with the per-frame neural numbers.

    ``decode_kw`` defaults to the P1c decode, as in :func:`score_real`.
    """
    from experiments.rps_bench import pit_mae

    kw = {"subgrid": True, "octave": False, "relocate": True, **decode_kw}
    was_training = net.training
    net.eval()
    rows = []
    for clip in clips:
        for m in range(int(clip["audio"].shape[0])):
            t0 = time.time()
            # `(1, N)`: one microphone, and the channel axis kept, so `spectrum`
            # reads it as one item of one channel and not as one item of N.
            au = torch.as_tensor(clip["audio"][m : m + 1], device=device)
            pred = net.decode(au, **kw)[0].cpu().numpy().astype(np.float64)
            mae = pit_mae(pred, clip["rps"])
            _, gt = _align(pred, clip["rps"])
            rows.append(
                {
                    "clip": clip["clip"],
                    "mic": m,
                    "phase": clip["phase"],
                    "rig": clip["rig"],
                    "mae": float(mae),
                    "wall": time.time() - t0,
                    "pred": pred,
                    "gt": gt,
                }
            )
    net.train(was_training)
    out = table(rows, name=name, floor_bins=int(net.floor_bins), quiet=quiet)
    # `table` keys `per_clip` by the clip, and there are eight rows per clip
    # here, so it would keep the last microphone alone. The clip's number is
    # the mean over its microphones.
    per_clip: dict[int, list[float]] = {}
    for r in rows:
        per_clip.setdefault(int(r["clip"]), []).append(r["mae"])
    out["per_clip"] = {c: round(float(np.mean(v)), 3) for c, v in per_clip.items()}
    # The predictions stay in the rows: the dump writes them, and the labels do
    # not (they are `real_clips()` and the caller already holds them).
    out["rows"] = [{k: v for k, v in r.items() if k != "gt"} for r in rows]
    return out


@torch.no_grad()
def score_part(
    net, name: str, device: str = "cpu", n: int | None = None, **decode_kw
) -> dict[str, Any]:
    """The mono synthetic parts (``comb`` / ``stoch``) of the benchmark.

    A frame there is ONE microphone of an 8 s clip, so the audio is mono and the
    eight-channel lever is absent by construction. The half-rate fraction is the
    number that matters: it is what the trained CNN port fails on (probe P2b).
    """
    from experiments import rps_bench as rb

    kw = {"subgrid": True, "octave": False, "relocate": True, **decode_kw}
    was_training = net.training
    net.eval()
    maes, one, half = [], [], []
    for f in rb.part(name, n=n):
        au = torch.as_tensor(
            np.asarray(f["mixture"].data, dtype=np.float32).ravel()[None], device=device
        )
        pred = net.decode(au, **kw)[0].cpu().numpy().astype(np.float64)
        gt = np.asarray(f["rps"].data, dtype=np.float64)
        maes.append(rb.pit_mae(pred, gt))
        _, g = _align(pred, gt)
        m = g > 1.0
        if m.any():
            r = pred[m] / g[m]
            one.append(np.abs(r - 1.0) <= 0.1)
            half.append(np.abs(r - 0.5) <= 0.05)
    net.train(was_training)
    return {
        "n": len(maes),
        "mean": float(np.mean(maes)) if maes else float("nan"),
        "median": float(np.median(maes)) if maes else float("nan"),
        "frac_one": float(np.concatenate(one).mean()) if one else float("nan"),
        "frac_half": float(np.concatenate(half).mean()) if half else float("nan"),
    }


# ─── Training windows ─────────────────────────────────────────────────────────


class WindowStream:
    """An infinite stream of ``(audio (C, n), rps (4, T))`` crops from a policy.

    A crop is KEPT only if every rotor stays inside ``[r_lo, r_hi]`` on every
    frame of it — see the module docstring for why. `seen` and `kept` carry the
    acceptance fraction, which is worth logging: the honest real pool spends
    most of its chunks on ground, silence and the low half of a ramp.

    ``mono=True`` keeps ONE microphone of each accepted crop, drawn uniformly
    at random per crop, and still yields ``(1, n)`` — the channel axis stays so
    that a batch is ``(B, 1, N)`` and `SlotCombNet.spectrum` reads it as B items
    of one channel. This is the input the neural baselines get, so a model
    trained on it is comparable with them.
    """

    def __init__(
        self,
        path: str = POLICY_REAL,
        crop_s: float = 2.0,
        seed: int = 0,
        r_lo: float = 31.0,
        r_hi: float = 99.0,
        base_seed: int | None = None,
        epoch: int = 0,
        hop: int = HOP,
        sr: int = SR,
        mono: bool = False,
    ):
        from omegaconf import OmegaConf

        from data_processing.frame_datasets import OnlineMixFrameDataset

        cfg = OmegaConf.load(path)
        if base_seed is not None:
            cfg.base_seed = int(base_seed)  # a different draw of the same policy
        self.ds = OnlineMixFrameDataset.from_config(cfg, flatten_channels=False)
        # A chained run restarts the stream from its own beginning. `set_epoch`
        # is what moves it on, so a resumed link does not retrain on the chunks
        # the previous link already saw.
        if epoch:
            self.ds.set_epoch(int(epoch))
        self.hop, self.sr = int(hop), int(sr)
        self.n_crop = int(round(float(crop_s) * self.sr))
        self.t_crop = self.n_crop // self.hop + 1
        self.r_lo, self.r_hi = float(r_lo), float(r_hi)
        self.mono = bool(mono)
        self.rng = np.random.default_rng(int(seed))
        self.seen = self.kept = 0
        self._it = iter(self.ds)

    @property
    def accept(self) -> float:
        return self.kept / max(1, self.seen)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        from experiments.rps_bench import resample_like_metric

        while True:
            frame = next(self._it)
            aud = np.asarray(frame["mixture"].data, dtype=np.float32)
            if aud.ndim == 1:
                aud = aud[None]
            rps = np.asarray(frame["rps"].data, dtype=np.float64)
            n_t = aud.shape[1] // self.hop + 1
            if rps.shape[-1] != n_t:
                rps = resample_like_metric(rps, n_t)
            self.seen += 1
            if aud.shape[1] < self.n_crop:
                continue
            f0 = int(self.rng.integers(0, (aud.shape[1] - self.n_crop) // self.hop + 1))
            a = aud[:, f0 * self.hop : f0 * self.hop + self.n_crop]
            g = rps[:, f0 : f0 + self.t_crop]
            if g.shape[-1] < self.t_crop:
                continue
            if not bool(((g >= self.r_lo) & (g <= self.r_hi)).all()):
                continue
            self.kept += 1
            if self.mono:
                m = int(self.rng.integers(0, a.shape[0]))
                a = a[m : m + 1]
            return np.ascontiguousarray(a), g.astype(np.float32)


def windows(
    path: str = POLICY_REAL,
    crop_s: float = 2.0,
    seed: int = 0,
    r_lo: float = 31.0,
    r_hi: float = 99.0,
    **kw,
) -> WindowStream:
    """An infinite iterator of accepted training crops. See :class:`WindowStream`."""
    return WindowStream(path, crop_s=crop_s, seed=seed, r_lo=r_lo, r_hi=r_hi, **kw)


def select_set(
    n: int = 48,
    seed: int = 20260904,
    crop_s: float = 2.0,
    path: str = POLICY_REAL,
    base_seed: int = 777_000,
    cache: bool = True,
    mono: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A FIXED set of real windows for checkpoint selection.

    Drawn once from the training policy with a DIFFERENT ``base_seed``, so it is
    the same distribution as training and none of the same chunks, and cached so
    every run and every restart selects on identical audio.

    The mono set is cached under its own name. The two sets are different audio
    at the same settings, and one silently read as the other would compare a
    one-microphone model against an eight-microphone selection number.
    """
    stem = "select_mono" if mono else "select"
    disk = CACHE / f"{stem}_n{n}_c{crop_s:g}_b{base_seed}.pkl"
    if cache and disk.exists():
        return pickle.loads(disk.read_bytes())
    st = windows(path, crop_s=crop_s, seed=seed, base_seed=base_seed, mono=mono)
    out = [next(st) for _ in range(int(n))]
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        disk.write_bytes(pickle.dumps(out))
    return out


@torch.no_grad()
def score_windows(
    net, wins: list[tuple[np.ndarray, np.ndarray]], device: str = "cpu", **decode_kw
) -> float:
    """Mean PIT MAE of the deployed decoder over a fixed window set.

    The window shape is whatever the stream yields — ``(8, n)`` or the mono
    ``(1, n)``. `SlotCombNet.spectrum` reads both as ONE item, so no branch is
    needed here.
    """
    from experiments.rps_bench import pit_mae

    kw = {"subgrid": True, "octave": False, "relocate": True, **decode_kw}
    was_training = net.training
    net.eval()
    errs = []
    for a, g in wins:
        pred = net.decode(torch.as_tensor(a, device=device), **kw)[0].cpu().numpy()
        errs.append(pit_mae(pred.astype(np.float64), g.astype(np.float64)))
    net.train(was_training)
    return float(np.mean(errs))
