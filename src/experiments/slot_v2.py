"""The v2 seam: data modes, selection sets, the saved config, and a Frame adapter.

WHAT THIS MODULE IS. `docs/slot-comb-v2-design.md` adds seven parameter groups
to `models.comb_slots.SlotCombNet`. The groups are model code; everything
AROUND them is here, so the trainer (`scripts/train_slot_v2.py`), the dump
(`scripts/slot_dump.py`) and the cue probe (`scripts/rps_cue_probe.py`) build
and read one object and not three.

Four things live here.

* THE DATA MODES. C1 trained on real windows only (arm A6b), and the design
  makes the curriculum comb -> stoch -> real explicit. A mode names one or two
  online-mix policies; every mode is read MONO, one microphone per crop drawn
  uniformly, because that is the protocol every neural baseline is read on.
* THE FULL-RANGE SAMPLER. C1 threw away every crop in which a rotor left the
  30-100 rev/s grid, so ground, warm-up and the low half of each ramp were out
  of the loss — which is 60.9 rev/s of the arm's error on the zero frames. With
  the OFF state (§ 3.1) and the grid from 10 rev/s (§ 3.2) those frames are
  expressible, so the sampler must stop dropping them. `WindowStream` already
  takes the acceptance window as an argument, so full range is that window
  opened to everything, and no change to `experiments.slot_real`.
* THE SAVED CONFIG. The trainer writes the constructor keywords next to
  `best.pt` as `config.json`. Every reader rebuilds the same model from it, so
  a dump or a probe cannot silently read a checkpoint with the wrong grid, the
  wrong emission or the wrong groups.
* THE FRAME ADAPTER. `zoo.load` returns a `FrameModel` — a `td.Frame` in, a
  `td.Frame` out. `SlotCombNet` speaks tensors, so :class:`SlotFrameModel` puts
  it behind the same interface and the probes need one code path.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "DATA_MODES",
    "DECODE_KW",
    "POLICIES",
    "SlotFrameModel",
    "build_from_config",
    "load_config",
    "load_arm",
    "mode_policies",
    "policy_path",
    "save_config",
    "select_set",
    "streams_for_mode",
    "windows",
]

#: data mode name -> the online-mix policy that renders it
POLICIES: dict[str, str] = {
    # The honest real pool of both rigs, with ground, warm-up and landing in,
    # plus the zero-labeled silence arm. `scripts/train_slot_real.py` § A6b.
    "real": "conf/online_mix/slot_real_dload.yaml",
    # The analytic static comb, and the stochastic family, at the salience/CRF
    # grid. Both carry `full_flight` (ground -> cruise -> ground) and a silence
    # arm, so both supply the zero and below-grid frames v2 must learn.
    "comb": "conf/online_mix/salv2_comb.yaml",
    "stoch": "conf/online_mix/salv2_stoch.yaml",
}

#: every training mode: one policy, or a synthetic one alternating with real
DATA_MODES = ("comb", "stoch", "real", "comb+real", "stoch+real")

#: the P1c decode, which is the deployed one. The decoder's own octave move is
#: off: at eight microphones it cost the untrained corner 22.4 -> 31.1 on
#: FLY124, and § 3.4 puts the multiple discriminator in the unary instead.
DECODE_KW: dict[str, Any] = {"subgrid": True, "octave": False, "relocate": True}

CACHE = Path(".cache/slot_v2")
#: the acceptance window that keeps every frame, whatever the labels do
FULL_RANGE = (0.0, float("inf"))


def policy_path(name: str) -> str:
    """A mode name -> its policy YAML. A path passes through unchanged.

    The path form is not a second interface for the trainer, whose ``--data``
    takes a mode name. It is what lets a test drive this module from a tiny
    local policy, and what lets a one-off arm try a policy that has no name
    yet, without an entry in :data:`POLICIES` that would then have to be
    removed again.
    """
    return POLICIES.get(name, name)


def mode_policies(mode: str) -> list[str]:
    """``"comb+real"`` -> ``["comb", "real"]``. Each item is a mode or a policy path."""
    names = [p for p in str(mode).split("+") if p]
    bad = [p for p in names if p not in POLICIES and not Path(policy_path(p)).exists()]
    if bad or not names:
        raise ValueError(f"unknown data mode {mode!r}; the modes are {list(DATA_MODES)}")
    return names


# ─── Training windows ─────────────────────────────────────────────────────────


def windows(
    policy: str,
    *,
    crop_s: float = 2.0,
    seed: int = 0,
    epoch: int = 0,
    accept: tuple[float, float] = FULL_RANGE,
    base_seed: int | None = None,
    mono: bool = True,
):
    """One infinite stream of mono crops from a named policy.

    ``accept`` is the rotor-rate window a crop must stay inside. The default
    keeps every crop, which is what the OFF state and the extended grid need;
    pass the grid edges to reproduce the C1 sampler.
    """
    from experiments.slot_real import WindowStream

    return WindowStream(
        policy_path(policy),
        crop_s=crop_s,
        seed=seed,
        r_lo=float(accept[0]),
        r_hi=float(accept[1]),
        base_seed=base_seed,
        epoch=epoch,
        mono=bool(mono),
    )


def streams_for_mode(
    mode: str,
    *,
    crop_s: float = 2.0,
    seed: int = 0,
    epoch: int = 0,
    accept: tuple[float, float] = FULL_RANGE,
    mono: bool = True,
) -> list[Any]:
    """One stream per policy of the mode, each with its own seed.

    The trainer takes one batch from each in turn, so a two-policy mode is a
    50/50 mixture at the step level and not at the sample level. That is the
    granularity `scripts/train_slot_real.py` used for real + partial.
    """
    return [
        windows(p, crop_s=crop_s, seed=seed + i, epoch=epoch, accept=accept, mono=mono)
        for i, p in enumerate(mode_policies(mode))
    ]


def select_set(
    mode: str,
    n: int = 48,
    *,
    seed: int = 20260906,
    crop_s: float = 2.0,
    base_seed: int = 777_000,
    accept: tuple[float, float] = FULL_RANGE,
    mono: bool = True,
    cache: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A FIXED set of windows for checkpoint selection, from the TRAINING policy.

    WHY NOT THE FROZEN SPLIT. The 37-clip split is the test set and FLY124 is an
    unseen rig. Three campaigns were inverted by a monitor that is not the task
    metric, so selection runs the deployed decoder on its own held-out windows
    drawn from the same policy with a DIFFERENT base seed.

    WHY PER MODE. A comb arm selected on real windows would be selected on data
    it never trains on, and the frozen split would leak in through the back
    door. A two-policy mode splits the count evenly between its policies.

    The cache name carries the mode and the acceptance window, because two sets
    that differ in either are different audio and one read as the other would
    compare two arms on two benchmarks.
    """
    names = mode_policies(mode)
    tag = "full" if accept == FULL_RANGE else f"{accept[0]:g}-{accept[1]:g}"
    safe = "_".join(Path(m).stem for m in mode.split("+"))
    stem = f"select_{safe}_n{n}_c{crop_s:g}_b{base_seed}_{tag}"
    stem += "" if mono else "_multi"
    disk = CACHE / f"{stem}.pkl"
    if cache and disk.exists():
        return pickle.loads(disk.read_bytes())
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i, p in enumerate(names):
        want = n // len(names) + (1 if i < n % len(names) else 0)
        st = windows(
            p,
            crop_s=crop_s,
            seed=seed + i,
            accept=accept,
            base_seed=base_seed + 1000 * i,
            mono=mono,
        )
        out.extend(next(st) for _ in range(want))
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        disk.write_bytes(pickle.dumps(out))
    return out


# ─── The saved config ─────────────────────────────────────────────────────────


def save_config(out: Path, config: dict[str, Any]) -> Path:
    """Write ``config.json`` next to ``best.pt``. Returns the path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    path.write_text(json.dumps(config, indent=1, default=str))
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a ``config.json``, given the file itself or the directory holding it."""
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    if not p.exists():
        raise SystemExit(f"no config.json at {p}; the trainer writes one next to best.pt")
    cfg = json.loads(p.read_text())
    if "model" not in cfg:
        raise SystemExit(f"{p} has no 'model' block of constructor keywords")
    return cfg


def build_from_config(config: dict[str, Any], device: str = "cpu"):
    """`SlotCombNet` from a saved config's ``model`` block."""
    from models.comb_slots import SlotCombNet

    kw = dict(config["model"])
    for key in ("parts", "v2_parts", "floor_widths"):
        if isinstance(kw.get(key), list):
            kw[key] = tuple(kw[key])
    return SlotCombNet(**kw).to(device)


def load_arm(path: str | Path, device: str = "cpu", ckpt: str = "best.pt"):
    """A trained arm: the model of ``config.json`` with ``best.pt`` loaded into it.

    ``path`` is the directory the trainer wrote, or the checkpoint itself (the
    config is then looked up beside it). The checkpoint holds the TRAINABLE
    parameters only, so it is loaded with ``strict=False`` and the unexpected
    keys are reported — a key the model does not have means the config and the
    checkpoint disagree, which is the failure this pair exists to prevent.
    """
    p = Path(path)
    d = p if p.is_dir() else p.parent
    weights = (d / ckpt) if p.is_dir() else p
    net = build_from_config(load_config(d), device=device)
    if weights.exists():
        report = net.load_state_dict(torch.load(weights, map_location=device), strict=False)
        if report.unexpected_keys:
            print(f"{weights}: unexpected keys {report.unexpected_keys}", flush=True)
    else:
        print(f"{weights} is absent: the untrained corner of {d / 'config.json'}", flush=True)
    return net.eval()


# ─── The Frame adapter ────────────────────────────────────────────────────────


class SlotFrameModel:
    """`SlotCombNet` behind the `zoo.FrameModel` interface: Frame in, Frame out.

    The output Frame carries ``rps_pred`` on the model's own frame grid, which
    is what `experiments.rps_bench.Readout` and `scripts/rps_cue_probe.speeds`
    read from a regressor. Only ONE microphone is read, whatever the input
    carries, because every probe and every table this feeds is the mono
    protocol.
    """

    def __init__(self, net, device: str = "cpu", decode_kw: dict[str, Any] | None = None):
        self.model = net.to(device).eval()
        self.device = device
        self.decode_kw = {**DECODE_KW, **(decode_kw or {})}

    def __call__(self, frame):
        import tdseries as td

        from data_processing.frames import rps_series
        from metrics._common import get_array

        key = "mixture" if "mixture" in frame else "audio"
        x = np.asarray(get_array(frame, key), dtype=np.float32)
        # `(1, N)`: one microphone with the channel axis kept, so `spectrum`
        # reads one item of one channel and not one item of N samples.
        mono = x.reshape(-1) if x.ndim == 1 else x.reshape(x.shape[0], -1)[0]
        au = torch.as_tensor(mono[None], device=self.device)
        with torch.no_grad():
            pred = self.model.decode(au, **self.decode_kw)[0].cpu().numpy()
        return td.Frame(
            {
                "rps_pred": rps_series(
                    pred.astype(np.float32),
                    sample_rate=int(self.model.sr),
                    hop_length=int(self.model.hop_length),
                )
            }
        )
