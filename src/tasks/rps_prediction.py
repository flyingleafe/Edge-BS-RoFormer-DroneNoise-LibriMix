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
from fractions import Fraction
from pathlib import Path
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

import numpy as np
import tdseries as td
import torch
import torch.nn as nn

from data_processing.frames import get_meta, meta_dict, with_meta

# ── Constants ─────────────────────────────────────────────────────────────

SR_AUDIO: float = 16000.0
N_FFT: int = 2048
HOP: int = 512
FRAME_SR: Fraction = Fraction(16000, HOP)  # exact rate; ≈ 31.25 Hz — never a rounded float
N_ROTORS: int = 4
DEVICE: str = "cpu"  # evaluation default


# ── Permutation alignment ─────────────────────────────────────────────────

# Rotor counts are physically small (quadrotor = 4). An absurd R means the
# caller passed a transposed (F, R) array, and the (R, R, F) pairwise cost
# would be materialized over R = thousands of frames — fail fast instead of
# allocating it and taking the machine down. Same guard style as
# ``src/losses/pit.py::_MAX_PIT_SOURCES``.
_MAX_PIT_ROTORS = 8


def align_rps_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Reorder a prediction's rotor rows to best-match the ground truth.

    RPS predictors are trained with a permutation-invariant objective, so the
    rotor *order* of a prediction is arbitrary: predicted row ``i`` does not
    correspond to ground-truth rotor ``i``. Evaluation accounts for this by
    searching all rotor assignments (see :func:`evaluate`'s ``pit`` branch).
    **Any plot that overlays a prediction on the ground truth must apply the
    same matching**, otherwise rotors appear swapped even when the prediction is
    good.

    Returns ``pred`` with its rotor rows permuted so that ``pred[k]``
    corresponds to ``gt[k]`` under the assignment minimising total MSE. The
    optimal linear assignment is identical to the brute-force 4!-permutation PIT
    search used by evaluation, so plots and metrics agree.

    ``pred`` and ``gt`` are ``(R, F)`` with the rotor axis FIRST; they may
    differ in frame count (``gt`` is linearly resampled onto the prediction's
    grid for the cost computation). If the shapes are incompatible (not 2-D,
    mismatched/​<2 rotor counts) ``pred`` is returned unchanged. A rotor count
    above ``_MAX_PIT_ROTORS`` raises ``ValueError`` — it means the input is
    transposed ``(F, R)`` and matching would blow up over frames.
    """
    from scipy.optimize import linear_sum_assignment

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[0] != gt.shape[0] or pred.shape[0] < 2:
        return pred
    if pred.shape[0] > _MAX_PIT_ROTORS:
        raise ValueError(
            f"align_rps_to_gt got R={pred.shape[0]} rotors (max {_MAX_PIT_ROTORS}). "
            "Check the array layout — expected (R, F) with the rotor axis first; "
            "a transposed (F, R) input would match over frames instead of rotors."
        )

    R, F = pred.shape
    if gt.shape[1] != F:  # put GT on the prediction's frame grid
        xp = np.linspace(0.0, 1.0, gt.shape[1])
        xq = np.linspace(0.0, 1.0, F)
        gt = np.vstack([np.interp(xq, xp, gt[r]) for r in range(R)])

    # cost[i, j] = MSE(pred rotor i, gt rotor j); Hungarian == optimal PIT match.
    cost = np.mean((pred[:, None, :] - gt[None, :, :]) ** 2, axis=-1)  # (R, R)
    row, col = linear_sum_assignment(cost)
    aligned = np.empty_like(pred)
    aligned[col] = pred[row]
    return aligned


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
        # Mono (T,) → (1, T) → model → (1, R, F)
        # Multichannel (C, T) → model treats C as batch → (C, R, F)
        out = self._model(t.unsqueeze(0)).squeeze(0) if t.dim() == 1 else self._model(t)
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


def load_input_set(path: str | Path) -> Iterator[td.Frame]:
    """Load a DREGON-LM-style dataset as ``Iterable[td.Frame]``.

    Expects each sample in a subdirectory ``sample_XXXXX/`` containing:
    * ``mixture.wav``   — 16 kHz mono audio
    * ``rps.npy``       — raw motor RPS array (n_rotors, M) at native motor rate

    A sibling ``metadata.json`` (mapping sample id → ``{input_snr, ...}``)
    is read for tags.

    Yields
    ------
    td.Frame
        Each frame has entries ``{"audio": Series, "rps": Series, "meta": Frame}``
        (``meta`` carries ``id``, ``input_snr``, ...).
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
        meta_path = root.parent / "metadata.json"
    if not meta_path.is_file():
        meta_path = Path("results/rps_predictor_comparison/dregon_lm_metadata.json")
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
            audio_dims = ("time",)
        else:
            audio = waveform.numpy().astype(np.float32)  # (C, T) multichannel
            audio_dims = ("mic", "time")

        # Load motor RPS.
        rps_raw = np.load(str(rps_path)).astype(np.float64)  # (R, M)
        if rps_raw.ndim != 2 or rps_raw.shape[0] != N_ROTORS:
            raise ValueError(f"Expected RPS shape (4, M), got {rps_raw.shape} in {rps_path}")

        # Build RPS as a StampIndex Series: timestamps at motor rate,
        # co-extensive with audio. The legacy motor_sample_rate is per-sample
        # and varies. Derive it from the audio duration — the RPS array
        # covers the same time span. For multichannel audio is (C, T), so use
        # last dimension.
        audio_dur_s = audio.shape[-1] / file_sr
        M = rps_raw.shape[1]
        motor_sr = M / audio_dur_s if audio_dur_s > 0 else 1000.0
        motor_times = np.arange(M) / motor_sr  # float seconds

        rps_series = td.events(
            motor_times,
            rps_raw,  # (R, M)
            dims=("rotor", "time"),
            t_start=0.0,
            t_end=audio_dur_s,
        )

        audio_series = td.uniform(audio, file_sr, dims=audio_dims, t_start=0.0)

        # Build meta — propagate ALL metadata fields so downstream analysis
        # (per-recording, per-source-type, etc.) works without needing to
        # re-read metadata.json.
        tags: dict[str, Any] = {"id": sid}
        meta_entry = metadata.get(sid) or _find_meta_entry(metadata, sid)
        if meta_entry:
            for k, v in meta_entry.items():
                if k == "id":
                    continue
                # Cast known numeric tags.
                if k in ("input_snr", "motor_sample_rate", "start_time", "duration", "n_channels"):
                    tags[k] = float(v) if not isinstance(v, (float, int)) else v
                elif k in ("rps_shape",):
                    tags[k] = v  # list
                else:
                    tags[k] = v

        frame = td.Frame({"audio": audio_series, "rps": rps_series})
        yield with_meta(frame, **tags)


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
    rps_series: td.Series,
    *,
    sr: float = SR_AUDIO,
) -> np.ndarray:
    """Align GT RPS onto the exact STFT frame grid via timestamp canon (A).

    Returns ``(n_rotors, n_frames)`` numpy array.  Works for mono (T,) and
    multichannel (C, T) audio — only the time length matters.
    """
    n_frames = _audio_len(audio) // HOP + 1
    frame_times = np.arange(n_frames) * HOP / sr
    return np.asarray(rps_series.interpolate(frame_times))  # (R, F)


def _align_shape_stretch(
    audio: np.ndarray,
    rps_series: td.Series,
    *,
    sr: float = SR_AUDIO,
) -> np.ndarray:
    """Align GT RPS by endpoint-to-endpoint shape-stretch (legacy method B).

    Reproduces the legacy ``F.interpolate(size=n_frames, mode='linear',
    align_corners=False)`` behavior exactly.  Only correct when motor and
    audio spans match (DREGON-LM); misaligns on free-flight etc.
    """
    import torch.nn.functional as F

    if rps_series.data is None:
        raise ValueError("RPS series has no values — cannot shape-stretch")
    raw_rps = np.asarray(rps_series.data, dtype=np.float64)  # (R, M) — time-last
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
    samples: Iterable[td.Frame],
    *,
    model_spec: str = "",
    input_set_label: str = "",
    alignment: str = "stft_timestamps",
    pit: bool = False,
    verbose: bool = True,
) -> EvalResult:
    """Run inference + compute metrics on an input set.

    Parameters
    ----------
    predictor : RPSPredictor | str
        A predictor object or a spec string (passed to ``load_predictor``).
    samples : Iterable[td.Frame]
        Each ``Frame`` must have entries ``"audio"`` and ``"rps"`` and a
        ``"meta"`` entry with ``id``.
    model_spec : str
        Human-readable label for the predictor (for logging).
    input_set_label : str
        Label for the input set (for logging).
    pit : bool
        If True, evaluate with permutation-invariant alignment: for each
        channel, try all 4! = 24 rotor permutations of the GT and pick the
        one that minimises MSE.  This reveals how much error is motor-swapping
        vs genuine misprediction.
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
            raise KeyError("Frame missing required entries 'audio' and 'rps'")

        audio_series = frame["audio"]
        rps_series = frame["rps"]

        if not isinstance(audio_series, td.Series) or not isinstance(
            audio_series.tindex, td.GridIndex
        ):
            raise TypeError(
                "'audio' entry must be a uniform (GridIndex) Series, got "
                f"{type(audio_series).__name__}"
            )
        if not isinstance(rps_series, td.Series) or not isinstance(
            rps_series.tindex, td.StampIndex
        ):
            raise TypeError(
                f"'rps' entry must be a StampIndex Series, got {type(rps_series).__name__}"
            )

        audio = audio_series.data  # (T,) or (C, T)
        sr = audio_series.tindex.sr
        sid = get_meta(frame, "id", f"sample_{n:05d}")
        per_ch_snr = get_meta(frame, "input_snr_per_channel", None)

        # Predict: (R, F) for mono, (C, R, F) for multichannel.
        pred = predictor.predict(audio, sr=sr)

        # Align GT RPS onto the predicted frame grid (shared across channels).
        gt = _align_gt(audio, rps_series, sr=sr)  # (R, F_gt)

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

            if pit:
                # Use the project's canonical PIT implementation.
                from losses.pit import _permutations_tensor, pit_mse_loss

                p_t = torch.from_numpy(np.asarray(p_ch, dtype=np.float32)).unsqueeze(0)  # (1, 4, F)
                g_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0)  # (1, 4, F)
                perms = _permutations_tensor(p_t.size(1))
                result = pit_mse_loss(p_t, g_t, perms=perms, return_indices=True)
                assert isinstance(result, tuple)
                _, best_idx = result
                best_perm = perms[best_idx[0]].tolist()  # e.g. [0, 2, 1, 3]
                gt_ch = gt[best_perm]  # (R, F)
            else:
                best_perm = None
                gt_ch = gt

            mse = float(np.mean((p_ch - gt_ch) ** 2))
            mae_frame = float(np.mean(np.abs(p_ch - gt_ch)))
            mae_clip = float(np.mean(np.abs(p_ch.mean(axis=-1) - gt_ch.mean(axis=-1))))
            ss_res = float(np.sum((p_ch - gt_ch) ** 2))
            ss_tot = float(np.sum((gt_ch - gt_ch.mean()) ** 2))
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
            if pit and best_perm is not None:
                row["pit_perm"] = list(best_perm)
            # Propagate ALL frame meta (recording_id, source_type, etc.) to
            # every per-sample row so aggregations don't need metadata.json.
            for tag_key, tag_val in meta_dict(frame).items():
                if tag_key != "id":
                    row[tag_key] = tag_val
            if isinstance(per_ch_snr, (list, tuple)) and ch < len(per_ch_snr):
                row["input_snr_channel"] = per_ch_snr[ch]

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
