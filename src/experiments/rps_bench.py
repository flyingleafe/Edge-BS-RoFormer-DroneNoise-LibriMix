"""The RPS benchmark parts and a browser over model outputs on them.

Three parts, the same for every model -- the sets ``scripts/rps_dump.py``
dumps and ``docs/experiments/rps-error-profile.md`` reads::

    comb    conf/data/salv2_comb_nomix.yaml  valid  (32 flights x 8 mics, 4 silent)
    stoch   conf/data/salv2_stoch_nomix.yaml valid  (32 flights, 8 silent)
    real    conf/data/m3cur_s2.yaml          valid  (dload:DREGON-LM-V4-michaels-valid-full, 37 clips)

plus the speech twins ``comb_speech`` / ``stoch_speech`` (same flights, a
LibriSpeech talker at -30 to 0 dB). A frame is one microphone of an 8 s clip:
frame ``i`` of a synthetic part is flight ``i // 8``, mic ``i % 8``.

Notebook use (``notebooks/rps_tracking.ipynb`` section 6)::

    from experiments import rps_bench as rb
    rb.experiments()                                   # what the dump holds
    rb.worst("real", "hppnet_r4_l4")                   # the frames worth a look
    f = rb.compare(["hppnet_r4_l4", "r4hb_scv2"], "real", 71)
    rb.show(f, fmax=2000)                              # spectrogram + GT + one row per model
    dwym({p: rb.overlay("salv2_hppnet_stoch_nomix", p, 8) for p in ("comb", "stoch", "real")})

Predictions come from the dump (``results/rps_dump/<part>/<exp>.npz``) when it
holds the experiment, else from the checkpoint through ``zoo.load`` on the
CPU (``source="live"`` forces that). A part is built once per process and
pickled under ``.cache/rps_bench/``: a synthetic part is a minute of
synthesis, the real part an R2 pull.

This module sits in ``experiments`` because it imports across the whole
stack (zoo, plots, metrics, training, data_processing); nothing in ``src``
imports it, scripts and notebooks do.
"""

from __future__ import annotations

import pickle
from itertools import permutations
from pathlib import Path
from typing import Any, cast

import matplotlib.figure
import numpy as np
import pandas as pd
import tdseries as td
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from data_processing.frames import meta_dict, rps_series
from losses.pit import align_rps_to_gt
from metrics import RPSMetric
from metrics._common import get_array
from metrics.salience_layers import LayerPeakRPSMetric, peak_readout
from models.multif0.utils import linear_freq_grid
from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.renderers import make_spectrogram_series
from training.config import build_dataset
from utils.audio import first_channel

__all__ = [
    "PARTS",
    "Readout",
    "build_set",
    "compare",
    "experiments",
    "overlay",
    "parse_sets",
    "part",
    "pit_mae",
    "resample_like_metric",
    "show",
    "worst",
]

#: part name -> (data yaml whose ``valid`` block defines it, param overrides)
PARTS: dict[str, tuple[str, dict[str, Any]]] = {
    "comb": ("conf/data/salv2_comb_nomix.yaml", {}),
    "stoch": ("conf/data/salv2_stoch_nomix.yaml", {}),
    "real": ("conf/data/m3cur_s2.yaml", {}),
    "real_nospeech": ("conf/data/m3cur_s2_nospeech.yaml", {}),
    "comb_speech": ("conf/data/salv2_comb_nomix.yaml", {"speech": True}),
    "stoch_speech": ("conf/data/salv2_stoch_nomix.yaml", {"speech": True}),
}
RATE = (16000, 512)  # the label / prediction frame grid
DUMP_ROOT = Path("results/rps_dump")
PROFILE = Path("results/rps_profile/frames.csv")
CACHE_DIR = Path(".cache/rps_bench")
_PERMS = list(permutations(range(4)))


# ─── Parts ────────────────────────────────────────────────────────────────────


def parse_sets(spec: str) -> list[tuple[str, str, dict[str, Any]]]:
    """``name=path[:key=value...]`` items, comma separated -> (name, path, overrides).

    A bare name that is a key of :data:`PARTS` expands to that part.
    """
    out = []
    for item in filter(None, spec.split(",")):
        if "=" not in item and item.strip() in PARTS:
            path, kv = PARTS[item.strip()]
            out.append((item.strip(), path, dict(kv)))
            continue
        name, _, rest = item.partition("=")
        path, *overrides = rest.split(":")
        kv = {}
        for ov in overrides:
            k, _, v = ov.partition("=")
            kv[k] = yaml.safe_load(v)  # `true` -> True, `8` -> 8, text stays text
        out.append((name.strip(), path.strip(), kv))
    if not out:
        raise SystemExit("--sets is empty")
    return out


def build_set(path: str, overrides: dict[str, Any]) -> Any:
    """The ``valid`` block of a Hydra data yaml, built as training builds it."""
    cfg = OmegaConf.load(path)
    spec: dict[str, Any] = dict(cast(dict, OmegaConf.to_container(cfg.valid, resolve=True)))
    params: dict[str, Any] = dict(spec.get("params") or {})
    params.update(overrides)
    spec["params"] = params
    return build_dataset(spec)


_parts: dict[str, list[td.Frame]] = {}


def part(name: str, *, n: int | None = None, cache: bool = True) -> list[td.Frame]:
    """The frames of one benchmark part, built once and pickled.

    ``n`` overrides the part's frame count (a smoke); a sized part is never
    cached on disk.
    """
    key = name if n is None else f"{name}[{n}]"
    if key in _parts:
        return _parts[key]
    path, overrides = PARTS[name]
    disk = CACHE_DIR / f"{name}.pkl"
    if n is None and cache and disk.exists():
        frames = pickle.loads(disk.read_bytes())
    else:
        params = dict(overrides)
        if n is not None:
            params["n"] = int(n)
        ds = build_set(path, params)
        frames = [ds[i] for i in range(len(ds) if n is None else min(len(ds), n))]
        if n is None and cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            disk.write_bytes(pickle.dumps(frames))
    _parts[key] = frames
    return frames


# ─── Readout ──────────────────────────────────────────────────────────────────


def resample_like_metric(gt: np.ndarray, n_t: int) -> np.ndarray:
    """``(R, Tg)`` -> ``(R, n_t)``, torch's linear ``align_corners=False``."""
    tg = gt.shape[-1]
    if tg == n_t:
        return gt
    pos = np.clip((np.arange(n_t) + 0.5) * tg / n_t - 0.5, 0, tg - 1)
    return np.stack([np.interp(pos, np.arange(tg), row) for row in gt])


def pit_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """MAE-optimal assignment over all 4! permutations, on the prediction's grid."""
    gt = resample_like_metric(np.asarray(gt, dtype=np.float64), pred.shape[-1])
    cost = np.abs(pred[:, None] - gt[None, :]).mean(-1)
    return min(sum(cost[k, p[k]] for k in range(4)) for p in _PERMS) / 4


class Readout:
    """Turn a model's output Frame into ``(R, T)`` rev/s plus the monitored metric.

    A regressor's ``rps_pred`` is taken as is. A salience port's layers are
    read by the peak + parabola readout its ``rps_mae`` metric uses
    (``metrics.salience_layers.peak_readout``), not by the CRF.
    """

    def __init__(
        self,
        fmin: float = 0.0,
        fmax: float = 150.0,
        bins: int = 300,
        n_layers: int = 4,
        rate: tuple[int, int] = RATE,
    ):
        self.grid = np.asarray(linear_freq_grid(fmin, fmax, bins), dtype=np.float64)
        self.n_layers = n_layers
        self.reg_metric = RPSMetric("mae_frame", rate=rate)
        self.sal_metric = LayerPeakRPSMetric(
            out_fmin=fmin, out_fmax=fmax, out_bins=bins, n_layers=n_layers, rate=rate
        )

    def __call__(self, pred: td.Frame, frame: td.Frame) -> tuple[np.ndarray, float]:
        if "rps_pred" in pred:
            arr = np.asarray(get_array(pred, "rps_pred"), dtype=np.float32)
            return arr, float(self.reg_metric(pred, frame))
        if "salience" in pred:
            metric = float(self.sal_metric(pred, frame))
            logits = torch.as_tensor(get_array(pred, "salience")).unsqueeze(0)  # (1, R*G, T)
            _, fg, n_t = logits.shape
            if fg != self.n_layers * len(self.grid):
                raise ValueError(
                    f"model emits {fg} bins; expected "
                    f"{self.n_layers} layers x {len(self.grid)} bins"
                )
            layers = logits.reshape(1, self.n_layers, len(self.grid), n_t).double()
            speeds = peak_readout(F.logsigmoid(layers), self.grid)[0].numpy()
            return speeds.astype(np.float32), metric
        raise KeyError(f"no rps_pred or salience in {list(pred.keys())}")


# ─── Predictions: dump first, checkpoint second ───────────────────────────────

_dumps: dict[tuple[str, str], Any] = {}
_models: dict[str, Any] = {}
_readout: Readout | None = None


def experiments(part_name: str | None = None, *, dump_root: Path = DUMP_ROOT) -> list[str]:
    """The experiments the dump holds (for one part, or for every part)."""
    parts = [part_name] if part_name else [p.name for p in dump_root.iterdir() if p.is_dir()]
    names: set[str] | None = None
    for p in parts:
        here = {f.stem for f in (dump_root / p).glob("*.npz") if not f.stem.startswith("_")}
        names = here if names is None else names & here
    return sorted(names or [])


def _dump_pred(exp: str, part_name: str, i: int, dump_root: Path) -> np.ndarray | None:
    key = (part_name, exp)
    if key not in _dumps:
        f = dump_root / part_name / f"{exp}.npz"
        _dumps[key] = np.load(f) if f.exists() else None
    z = _dumps[key]
    if z is None:
        return None
    return z["pred"][i, :, : z["n_t"][i]].astype(np.float64)


def _live_pred(exp: str, frame: td.Frame, device: str) -> np.ndarray:
    global _readout
    import zoo  # heavy (Hydra + torch); only when a checkpoint is really needed

    if exp not in _models:
        _models[exp] = zoo.load(exp, device=device)
    if _readout is None:
        _readout = Readout()
    return _readout(_models[exp](frame), frame)[0].astype(np.float64)


def overlay(
    exp: str,
    part_name: str,
    i: int,
    *,
    source: str = "auto",
    device: str = "cpu",
    dump_root: Path = DUMP_ROOT,
) -> td.Frame:
    """One model on one frame: ``audio`` + ``rps`` (label) + ``rps_pred``, ready for ``dwym``.

    ``rps_pred`` is PIT-aligned to the label and ``meta`` carries the
    experiment, the part, the frame index, the flight and mic, and the PIT MAE.
    ``source`` is ``"auto"`` (the dump when it holds ``exp``, else the
    checkpoint), ``"dump"`` or ``"live"``.
    """
    frame = part(part_name)[i]
    pred = None
    if source in ("auto", "dump"):
        pred = _dump_pred(exp, part_name, i, dump_root)
        if pred is None and source == "dump":
            raise FileNotFoundError(f"{dump_root / part_name / exp}.npz")
    if pred is None:
        pred = _live_pred(exp, frame, device)
    gt = np.asarray(get_array(frame, "rps"), dtype=np.float64)
    pred = align_rps_to_gt(pred, gt)
    meta = meta_dict(frame)
    meta.update(
        experiment=exp,
        part=part_name,
        index=i,
        flight=str(meta.get("recording_id", meta.get("sample_id", i // 8))),
        mae=pit_mae(pred, gt),
    )
    return td.Frame(
        {
            "audio": first_channel(frame["mixture"]),
            "rps": frame["rps"],
            "rps_pred": rps_series(
                pred.astype(np.float32), sample_rate=RATE[0], hop_length=RATE[1]
            ),
            "meta": td.Frame(meta),
        }
    )


def compare(exps: list[str], part_name: str, i: int, **kw: Any) -> td.Frame:
    """Several models on ONE frame: ``audio`` + ``rps`` + one aligned entry per experiment.

    Draw it with :func:`show`; ``meta.mae`` maps each experiment to its PIT MAE.
    """
    entries: dict[str, Any] = {}
    maes: dict[str, float] = {}
    for exp in exps:
        one = overlay(exp, part_name, i, **kw)
        entries.setdefault("audio", one["audio"])
        entries.setdefault("rps", one["rps"])
        entries[exp] = one["rps_pred"]
        maes[exp] = float(one["meta"]["mae"])
        base_meta = meta_dict(one)
    entries["meta"] = td.Frame({**base_meta, "experiment": list(exps), "mae": maes})
    return td.Frame(entries)


def show(
    frame: td.Frame, *, fmax: float | None = 2000.0, row_height: float = 2.2
) -> matplotlib.figure.Figure:
    """Spectrogram, label, then one PIT-aligned row per model, on one time axis."""
    preds = [
        k
        for k, v in frame.items()
        if k not in ("audio", "rps", "meta") and isinstance(v, td.Series)
    ]
    maes = dict(frame["meta"]["mae"]) if "meta" in frame and "mae" in frame["meta"] else {}
    spec = make_spectrogram_series(first_channel(frame["audio"]), fmax=fmax)
    tracks: list[Any] = [
        PlotTrack(
            series=spec.series, renderer=spec.renderer, hints={**spec.hints, "title": "spectrogram"}
        ),
        PlotTrack(series=frame["rps"], hints={"title": "rps (label)"}),
    ]
    for k in preds:
        title = f"{k}   PIT MAE {maes[k]:.2f} rev/s" if k in maes else k
        tracks.append(PlotTrack(series=frame[k], hints={"title": title}))
    meta = meta_dict(frame)
    fig = plot_timeframe(frame, tracks=tracks, figsize=(16, row_height * len(tracks) + 1.5))
    fig.suptitle(f"{meta['part']}  flight {meta['flight']}  mic {meta.get('channel', '?')}", y=1.02)
    return fig


def worst(part_name: str, exp: str, k: int = 8, *, profile: Path = PROFILE) -> pd.DataFrame:
    """The ``k`` worst frames of ``exp`` on ``part_name``, from the error profile.

    Needs ``scripts/rps_error_profile.py`` to have run over the dump. Silence
    clips are excluded (a stopped-rotor clip is a trivial 0 for most models).
    """
    fr = pd.read_csv(profile, low_memory=False)
    sel = cast(
        pd.DataFrame, fr[(fr["set"] == part_name) & (fr["exp"] == exp) & (fr["n_stopped"] < 4)]
    )
    cols = ["frame", "flight", "channel", "mae", "cls", "gt_mean", "gt_range", "n_stopped"]
    top = sel.sort_values(by="mae", ascending=False).head(k)
    return cast(pd.DataFrame, top[cols]).reset_index(drop=True)
