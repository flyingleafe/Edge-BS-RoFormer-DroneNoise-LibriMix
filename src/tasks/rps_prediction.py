"""RPS-prediction task module — §A of the task-separated architecture.

Provides the Python API for RPS-prediction evaluation and the thin CLI shim
that ``evaluate-rps`` calls.

See .pi/plans/rps-eval-plot-refactor-plan.md for the full architecture.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

import numpy as np
import torch
import torch.nn as nn

from utils.data import (
    EventSeries,
    TimeFrame,
    UniformSeries,
)
from utils.paths import get_results_path

# ── Constants ─────────────────────────────────────────────────────────────

SR_AUDIO: float = 16000.0
N_FFT: int = 2048
HOP: int = 512
FRAME_SR: float = SR_AUDIO / HOP  # ≈ 31.25 Hz
N_ROTORS: int = 4
DEVICE: str = "cpu"  # evaluation default


# ── Predictor protocol ────────────────────────────────────────────────────


@runtime_checkable
class RPSPredictor(Protocol):
    """Structural interface for RPS prediction.

    ``predict(audio, sr) -> (n_rotors, n_frames)`` — no inheritance required.
    """

    def predict(self, audio: np.ndarray, /, sr: float = SR_AUDIO) -> np.ndarray:
        """Predict rotor speeds for each STFT frame.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 waveform.
        sr : float
            Sample rate in Hz.

        Returns
        -------
        np.ndarray
            Shape ``(n_rotors, n_frames)``, float32, at frame rate ``sr/HOP``.
        """
        ...


# ── Predictor factory ─────────────────────────────────────────────────────


def load_predictor(spec: Any) -> RPSPredictor:
    """Return an ``RPSPredictor`` from a spec string or object.

    ``spec`` can be:
    * An existing ``RPSPredictor`` → returned as-is.
    * A string ``"Type@/path/to/ckpt.pt"`` → learned model loaded via
      ``tasks.checkpoints.load_model``, wrapped in an inference adapter.
    * A string ``"cepstral" | "hps" | "pyin" | "matched_filter" | "nmf"`` →
      classical predictor from ``classical_rps_predictors``.

    The factory is **idempotent**: passing an already-loaded predictor
    returns it unchanged.
    """
    if isinstance(spec, RPSPredictor):
        return spec

    if isinstance(spec, nn.Module):
        return _ModelPredictor(spec)

    if not isinstance(spec, str):
        raise TypeError(f"load_predictor expects str or RPSPredictor, got {type(spec).__name__}")

    s = spec.strip()

    # Classical baselines.
    # Canonical short-name → module attribute mapping
    _CLASSICAL_ATTR = {
        "pyin": "pyin_single_f0",
        "cepstral": "cepstral_tracker",
        "hps": "hps_tracker",
        "matched_filter": "matched_filter_tracker",
        "nmf": "nmf_tracker",
        "pyin_single_f0": "pyin_single_f0",
        "cepstral_tracker": "cepstral_tracker",
        "hps_tracker": "hps_tracker",
        "matched_filter_tracker": "matched_filter_tracker",
        "nmf_tracker": "nmf_tracker",
    }
    if s in _CLASSICAL_ATTR:
        import importlib

        mod = importlib.import_module("classical_rps_predictors")
        fn = getattr(mod, _CLASSICAL_ATTR[s])
        if hasattr(fn, "predict"):
            return fn
        return _ClassicalPredictor(fn, s)

    # Learned model: Type@ckpt.
    if "@" in s and not s.startswith("@"):
        from tasks.checkpoints import load_model

        model = load_model(s, device=DEVICE)
        return _ModelPredictor(model)

    raise ValueError(
        f"Unknown predictor spec {s!r}.  Expected 'Type@ckpt.pt' or one of {sorted(_CLASSICAL_ATTR)}."
    )


# ── Internal predictor wrappers ───────────────────────────────────────────


class _ModelPredictor:
    """Wrap a learned ``nn.Module`` to satisfy ``RPSPredictor``."""

    def __init__(self, model: nn.Module, device: str = DEVICE):
        self._model = model
        self._device = device

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: float = SR_AUDIO) -> np.ndarray:
        t = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self._device)
        if t.dim() == 1:
            # Mono (T,) → (1, T) → model → (1, R, F) → (R, F)
            out = self._model(t.unsqueeze(0)).squeeze(0)
        else:
            # Multichannel (C, T) → model treats C as batch → (C, R, F)
            out = self._model(t)
        return out.cpu().numpy()


class _ClassicalPredictor:
    """Wrap a classical predictor function to satisfy ``RPSPredictor``."""

    def __init__(self, fn, name: str):
        self._fn = fn
        self._name = name

    def predict(self, audio: np.ndarray, sr: float = SR_AUDIO) -> np.ndarray:
        return self._fn(audio, sr=sr)


_CLASSICAL_NAMES = {
    "cepstral",
    "hps",
    "pyin",
    "matched_filter",
    "nmf",
    "pyin_single_f0",
    "cepstral_tracker",
    "hps_tracker",
    "matched_filter_tracker",
    "nmf_tracker",
}


# ── Input-set loader ──────────────────────────────────────────────────────


def load_input_set(path: str | Path) -> Iterator[TimeFrame]:
    """Load a DREGON-LM-style dataset as ``Iterable[TimeFrame]``.

    Expects each sample in a subdirectory ``sample_XXXXX/`` containing:
    * ``mixture.wav``   — 16 kHz mono audio
    * ``rps.npy``       — raw motor RPS array (n_rotors, M) at native motor rate

    A sibling ``metadata.json`` (mapping sample id → ``{input_snr, ...}``)
    is read for tags.

    Yields
    ------
    TimeFrame
        Each frame has tracks ``{"audio": UniformSeries, "rps": EventSeries}``
        and tags ``{"id": ..., "input_snr": ...}``.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    # Load metadata for tags.
    metadata: dict[str, dict] = {}
    meta_path = root.parent / "dregon_lm_metadata.json"
    if not meta_path.is_file():
        meta_path = root / "metadata.json"
    if not meta_path.is_file():
        meta_path = get_results_path("rps_predictor_comparison/dregon_lm_metadata.json")
    if meta_path.is_file():
        with open(meta_path) as f:
            raw_meta = json.load(f)
        # Handle nested formats: {"train": [...], "valid": [...]} or flat list.
        if isinstance(raw_meta, dict):
            # Try the dataset split key matching the directory name.
            split_name = root.name  # e.g. "valid" or "train"
            if split_name in raw_meta and isinstance(raw_meta[split_name], list):
                metadata = {e["id"]: e for e in raw_meta[split_name]}
            else:
                # Check if direct sample dict.
                first_val = next(iter(raw_meta.values()))
                if isinstance(first_val, dict) and "input_snr" in first_val:
                    metadata = raw_meta
        elif isinstance(raw_meta, list):
            metadata = {e.get("id", e.get("sample", "")): e for e in raw_meta}

    import torchaudio

    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
            continue

        sid = sample_dir.name
        wav_path = sample_dir / "mixture.wav"
        rps_path = sample_dir / "rps.npy"

        if not wav_path.is_file() or not rps_path.is_file():
            continue

        # Load audio.
        waveform, file_sr = torchaudio.load(str(wav_path))
        if file_sr != SR_AUDIO:
            raise ValueError(f"Expected {SR_AUDIO} Hz audio, got {file_sr} in {wav_path}")
        if waveform.shape[0] == 1:
            audio = waveform.squeeze(0).numpy().astype(np.float32)  # (T,) mono
        else:
            audio = waveform.numpy().astype(np.float32)  # (C, T) multichannel

        # Load motor RPS.
        rps_raw = np.load(str(rps_path)).astype(np.float64)  # (R, M)
        if rps_raw.ndim != 2 or rps_raw.shape[0] != N_ROTORS:
            raise ValueError(f"Expected RPS shape (4, M), got {rps_raw.shape} in {rps_path}")

        # Build RPS as EventSeries: timestamps at motor rate, co-extensive
        # with audio.  The legacy motor_sample_rate is per-sample and varies.
        # Derive it from the audio duration — the RPS array covers the same
        # time span.
        audio_dur_s = len(audio) / file_sr  # type: ignore[operator]
        M = rps_raw.shape[1]
        motor_sr = M / audio_dur_s if audio_dur_s > 0 else 1000.0
        motor_times = np.arange(M) / motor_sr  # float seconds

        rps_es = EventSeries.from_events(
            timestamps=motor_times,
            values=rps_raw,  # (R, M)
            t_start=0.0,
            t_end=audio_dur_s,
        )

        audio_us = UniformSeries.from_samples(audio, sr=file_sr, t_start=0.0)

        # Build tags.
        tags: dict[str, Any] = {"id": sid}
        meta_entry = metadata.get(sid) or _find_meta_entry(metadata, sid)
        if meta_entry:
            if "input_snr" in meta_entry:
                tags["input_snr"] = float(meta_entry["input_snr"])
            if "recording_id" in meta_entry:
                tags["recording_id"] = meta_entry["recording_id"]

        yield TimeFrame.from_tracks(
            {"audio": audio_us, "rps": rps_es},
            tags=tags,
        )


def _find_meta_entry(metadata: dict, sid: str) -> dict | None:
    """Try to find a metadata entry by exact or prefix match."""
    if sid in metadata:
        return metadata[sid]
    # Try stripping trailing segments (e.g. sample_00001_00 -> sample_00001).
    for k, v in metadata.items():
        if k in sid or sid in k:
            return v
    return None


# ── Evaluation ────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Per-sample metrics + aggregate + helpers."""

    per_sample: list[dict] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    model_spec: str = ""
    input_set_label: str = ""

    def per_snr(self) -> list[dict]:
        """Return per-SNR-bin stratified metrics (list of dicts)."""
        return _stratify_per_snr(self.per_sample)

    def to_json(self, path: str | Path) -> None:
        """Save per-sample + aggregate to a JSON file."""
        out = {
            "model_spec": self.model_spec,
            "input_set": self.input_set_label,
            "aggregate": self.aggregate,
            "per_sample": self.per_sample,
            "per_snr": self.per_snr(),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def to_wandb(self, run, *, figures: dict | None = None) -> None:
        """Log to a wandb run (dedicated eval run — §D).

        * ``aggregate`` → ``run.summary``
        * per-sample + per-SNR → ``wandb.Table``
        * full JSON → ``wandb.Artifact(type="eval")``
        * optional figures → ``wandb.Image``
        """
        try:
            import wandb
        except ImportError:
            print("wandb not installed; skipping wandb logging.")
            return

        run.summary.update(self.aggregate)

        # Per-sample table.
        keys = list(self.per_sample[0].keys()) if self.per_sample else []
        if keys:
            table = wandb.Table(columns=keys)
            for row in self.per_sample:
                table.add_data(*[row[k] for k in keys])
            run.log({"per_sample": table})

        # Per-SNR table.
        per_snr_rows = self.per_snr()
        if per_snr_rows:
            snr_keys = list(per_snr_rows[0].keys())
            snr_table = wandb.Table(columns=snr_keys)
            for row in per_snr_rows:
                snr_table.add_data(*[row[k] for k in snr_keys])
            run.log({"per_snr": snr_table})

        # Figures.
        if figures:
            for name, fig in figures.items():
                run.log({name: wandb.Image(fig)})

        # Artifact.
        art = wandb.Artifact(
            name=f"eval-{self.input_set_label or 'results'}",
            type="eval",
        )
        with art.new_file("metrics.json") as f:
            json.dump(
                {
                    "aggregate": self.aggregate,
                    "per_sample": self.per_sample,
                    "per_snr": per_snr_rows,
                },
                f,
                indent=2,
            )
        run.log_artifact(art)


# ── GT alignment strategies ──────────────────────────────────────────────


def _audio_len(audio: np.ndarray) -> int:
    """Number of time samples regardless of whether audio is (T,) or (C, T)."""
    return int(audio.shape[-1])


def _align_stft_timestamps(
    audio: np.ndarray,
    rps_es: EventSeries,
    *,
    sr: float = SR_AUDIO,
) -> np.ndarray:
    """Align GT RPS onto the exact STFT frame grid via timestamp canon (A).

    Returns ``(n_rotors, n_frames)`` numpy array.  Works for mono (T,) and
    multichannel (C, T) audio — only the time length matters.
    """
    n_frames = _audio_len(audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr
    return rps_es.interpolate(frame_times)  # (R, F)


def _align_shape_stretch(
    audio: np.ndarray,
    rps_es: EventSeries,
    *,
    sr: float = SR_AUDIO,
) -> np.ndarray:
    """Align GT RPS by endpoint-to-endpoint shape-stretch (legacy method B).

    Reproduces the legacy ``F.interpolate(size=n_frames, mode='linear',
    align_corners=False)`` behavior exactly.  Only correct when motor and
    audio spans match (DREGON-LM); misaligns on free-flight etc.
    """
    import torch.nn.functional as F

    if rps_es.values is None:
        raise ValueError("RPS EventSeries has no values — cannot shape-stretch")
    raw_rps = np.asarray(rps_es.values, dtype=np.float64)  # (R, M) — time-last
    n_frames = _audio_len(audio) // HOP + 1
    # Torch F.interpolate: (B, C, L) -> (B, C, N)
    t = torch.from_numpy(raw_rps).unsqueeze(0)  # (1, R, M)
    result = F.interpolate(t.float(), size=n_frames, mode="linear", align_corners=False).squeeze(
        0
    )  # (R, n_frames)
    return result.numpy()


# ── Primary entry: evaluate ───────────────────────────────────────────────


def evaluate(
    predictor: RPSPredictor | str,
    samples: Iterable[TimeFrame],
    *,
    model_spec: str = "",
    input_set_label: str = "",
    alignment: str = "stft_timestamps",
    verbose: bool = True,
) -> EvalResult:
    """Run inference + compute metrics on an input set.

    Parameters
    ----------
    predictor : RPSPredictor | str
        A predictor object or a spec string (passed to ``load_predictor``).
    samples : Iterable[TimeFrame]
        Each ``TimeFrame`` must have tracks ``"audio"`` and ``"rps"`` and
        tags ``{"id": ...}``.
    model_spec : str
        Human-readable label for the predictor (for logging).
    input_set_label : str
        Label for the input set (for logging).
    verbose : bool
        Print progress.

    Returns
    -------
    EvalResult
    """
    if isinstance(predictor, str) or not isinstance(predictor, RPSPredictor):
        predictor = load_predictor(predictor)

    if not model_spec:
        model_spec = str(predictor)

    # Resolve alignment strategy.
    if alignment == "stft_timestamps":
        _align_gt = _align_stft_timestamps
    elif alignment == "shape_stretch":
        _align_gt = _align_shape_stretch
    else:
        raise ValueError(
            f"Unknown alignment {alignment!r}; expected 'stft_timestamps' or 'shape_stretch'"
        )

    per_sample: list[dict] = []
    all_mse: list[float] = []
    all_mae_frame: list[float] = []
    all_mae_clip: list[float] = []
    all_r2: list[float] = []

    t0 = time.time()
    n = 0
    for frame in samples:
        if "audio" not in frame or "rps" not in frame:
            raise KeyError("TimeFrame missing required tracks 'audio' and 'rps'")

        audio_us = frame["audio"]
        rps_es = frame["rps"]

        if not isinstance(audio_us, UniformSeries):
            raise TypeError(f"audio track must be UniformSeries, got {type(audio_us).__name__}")
        if not isinstance(rps_es, EventSeries):
            raise TypeError(f"rps track must be EventSeries, got {type(rps_es).__name__}")

        audio = audio_us.samples  # (T,) or (C, T)
        sr = audio_us.sr
        sid = frame.tags.get("id", f"sample_{n:05d}")
        snr_tag = frame.tags.get("input_snr", None)
        per_ch_snr = frame.tags.get("input_snr_per_channel", None)

        # Predict: (R, F) for mono, (C, R, F) for multichannel.
        pred = predictor.predict(audio, sr=sr)

        # Align GT RPS onto the predicted frame grid (shared across channels).
        gt = _align_gt(audio, rps_es, sr=sr)  # (R, F_gt)

        # Normalise to the same number of frames.
        F = min(pred.shape[-1], gt.shape[-1])
        pred = pred[..., :F]
        gt = gt[..., :F]

        # Expand mono pred to (1, R, F) so the channel loop is uniform.
        if pred.ndim == 2:
            pred = pred[np.newaxis]  # (1, R, F)
        C = pred.shape[0]

        for ch in range(C):
            p_ch = pred[ch]  # (R, F)

            mse = float(np.mean((p_ch - gt) ** 2))
            mae_frame = float(np.mean(np.abs(p_ch - gt)))
            mae_clip = float(np.mean(np.abs(p_ch.mean(axis=-1) - gt.mean(axis=-1))))
            ss_res = float(np.sum((p_ch - gt) ** 2))
            ss_tot = float(np.sum((gt - gt.mean()) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-6 else None

            row: dict = {
                "sample": sid,
                "channel": ch,
                "mse": mse,
                "mae_frame": mae_frame,
                "mae_clip": mae_clip,
                "ss_tot": ss_tot,
                "r2": r2,
            }
            if snr_tag is not None:
                row["input_snr"] = snr_tag
            if per_ch_snr is not None and ch < len(per_ch_snr):
                row["input_snr_channel"] = per_ch_snr[ch]  # pyright: ignore[reportIndexIssue]

            per_sample.append(row)
            all_mse.append(mse)
            all_mae_frame.append(mae_frame)
            all_mae_clip.append(mae_clip)
            if r2 is not None:
                all_r2.append(r2)

        n += 1
        if verbose and n % 100 == 0:
            print(f"  {n} samples ({n * C} rows)  running MAE/clip={np.mean(all_mae_clip):.3f}")

    elapsed = time.time() - t0
    if verbose and n > 0:
        print(f"  Done {n} samples in {elapsed:.1f}s")

    agg = {
        "n_samples": n,
        "n_rows": len(per_sample),  # n_samples * n_channels
        "n_r2_valid": len(all_r2),
        "mse": float(np.mean(all_mse)) if all_mse else 0.0,
        "rmse": float(np.sqrt(np.mean(all_mse))) if all_mse else 0.0,
        "mae_frame": float(np.mean(all_mae_frame)) if all_mae_frame else 0.0,
        "mae_clip": float(np.mean(all_mae_clip)) if all_mae_clip else 0.0,
        "r2_mean": float(np.mean(all_r2)) if all_r2 else 0.0,
        "r2_median": float(np.median(all_r2)) if all_r2 else 0.0,
        "r2_std": float(np.std(all_r2)) if all_r2 else 0.0,
        "elapsed_s": round(elapsed, 1),
    }

    return EvalResult(
        per_sample=per_sample,
        aggregate=agg,
        model_spec=model_spec,
        input_set_label=input_set_label,
    )


# ── Per-SNR stratification ────────────────────────────────────────────────

_SNR_BINS = [
    (-30, -25),
    (-25, -20),
    (-20, -15),
    (-15, -10),
    (-10, -5),
    (-5, 0),
]


def _stratify_per_snr(per_sample: list[dict]) -> list[dict]:
    """Bucket per-sample metrics by ``input_snr`` tag."""
    bins: dict[tuple[float, float], list[dict]] = {b: [] for b in _SNR_BINS}
    for row in per_sample:
        snr = row.get("input_snr")
        if snr is None:
            continue
        snr = float(snr)
        for (lo, hi), bucket in bins.items():
            if lo <= snr < hi:
                bucket.append(row)
                break

    result: list[dict] = []
    for (lo, hi), bucket in bins.items():
        if not bucket:
            continue
        stats = _bin_stats(bucket)
        stats["snr_range"] = f"[{lo}, {hi})"
        result.append(stats)

    # Also overall.
    if per_sample:
        overall = _bin_stats(per_sample)
        overall["snr_range"] = "Overall"
        result.append(overall)

    return result


def _bin_stats(rows: list[dict]) -> dict:
    """Mean/std of scalar metrics over a set of per-sample rows."""
    ks = ["mse", "mae_frame", "mae_clip", "r2"]
    stats: dict[str, float] = {"n": len(rows)}
    for k in ks:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_std"] = float(np.std(vals))
    return stats
