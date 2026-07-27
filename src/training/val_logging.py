"""Validation-time audio/figure sample logging + R2 upload.

Frame-native port of the old ``train.py::collect_audio_triples_by_snr`` /
``log_audio_triples_to_wandb`` (``git show d94ce9f:train.py``): pick a
handful of validation samples, SNR-stratified, and log their audio (and, for
tasks without an audio output, a prediction-vs-ground-truth figure) to wandb
every validation pass. The old version scanned a DN-LM-style metadata folder
on disk for SNR buckets; this version works directly off the per-sample
``(pred, target)`` Frame pairs the validation loop already computes
(:mod:`data_processing.collate` ``slice_sample`` output) via
``data_processing.frames.get_meta(target, "input_snr")``.

Kept out of :mod:`training.loop` behind the narrow :func:`log_validation_samples`
entry point (task/pairs/epoch/log_fn/artifact_store in, nothing out) so a
future multi-model training scheme (docs/refactor-unified-framework.md §
"Future expansions") can reuse sample selection + logging without depending
on the single-model loop.

Decoupled from the global ``wandb`` singleton on purpose: this module never
calls ``wandb.init``/``wandb.log``/``wandb.run`` itself. It only constructs
``wandb.Audio``/``wandb.Image`` *value objects* (pure functions of already-
computed arrays/figures — no network, no global state) and hands the
resulting ``{key: value}`` payload to a caller-supplied ``log_fn``. The
training loop supplies ``log_fn`` closing over its own module-level
``wandb`` name (see ``training.loop``'s ``wandb`` monkeypatch pattern in
``tests/training/test_loop.py``); tests here supply a plain
``list.append``-backed fake instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import tdseries as td

import wandb
from data_processing.frames import get_meta
from tasks.task import Task
from training.artifacts import ValSample

__all__ = ["select_val_sample_indices", "log_validation_samples", "ArtifactSink"]

logger = logging.getLogger(__name__)


@runtime_checkable
class ArtifactSink(Protocol):
    """Structural interface this module needs from an artifact store.

    Deliberately narrower than :class:`training.artifacts.ArtifactStore`
    (which satisfies it) so tests can pass a plain recording fake instead of
    a real store — same rationale as the ``Codec``/``Metric`` protocols
    elsewhere in this codebase.
    """

    def upload_val_samples(self, epoch: int, samples: Sequence[ValSample]) -> str | None: ...


# ─── Sample selection ──────────────────────────────────────────────────────


def select_val_sample_indices(targets: Sequence[td.Frame], num_samples: int) -> list[int]:
    """Pick up to ``num_samples`` indices into ``targets``.

    Stratified across low/mid/high buckets of a per-target scalar:
    mean ground-truth RPS when targets carry an ``"rps"`` entry (so the
    logged overlays always span idle AND in-flight windows — the old
    first-N fallback pinned every logged sample to the zero-RPS ground
    segments that open the unshuffled valid-full split), else ``input_snr``
    meta, else the first ``num_samples`` indices. Deterministic given the
    same ``targets`` ordering — the validation loader is not shuffled, so
    the same samples are picked epoch over epoch (stable evolution in
    wandb, as the old ``seed=0`` selection intended).
    """
    n = len(targets)
    if num_samples <= 0 or n == 0:
        return []
    num_samples = min(num_samples, n)

    def _mean_rps(t: td.Frame) -> float | None:
        if "rps" not in t:
            return None
        data = t["rps"].data
        if data is None:
            return None
        arr = np.asarray(data, dtype=np.float32)
        return float(arr.mean()) if arr.size else None

    values: list[float] = []
    rps_vals = [_mean_rps(t) for t in targets]
    if all(v is not None for v in rps_vals):
        values = [float(v) for v in rps_vals if v is not None]
    else:
        for t in targets:
            snr = get_meta(t, "input_snr", None)
            if snr is None:
                return list(range(num_samples))
            values.append(float(snr))

    order = sorted(range(n), key=lambda i: values[i])
    n_buckets = min(3, num_samples)
    buckets = [[int(i) for i in b] for b in np.array_split(np.array(order), n_buckets)]

    base, extra = divmod(num_samples, n_buckets)
    picked: list[int] = []
    for b, bucket in enumerate(buckets):
        take = base + (1 if b < extra else 0)
        if not bucket or take <= 0:
            continue
        step = max(1, len(bucket) // take)
        picked.extend(bucket[::step][:take])

    seen: set[int] = set()
    result: list[int] = []
    for i in picked:
        if i not in seen:
            seen.add(i)
            result.append(i)
    if len(result) < num_samples:
        for i in order:
            if i not in seen:
                seen.add(i)
                result.append(i)
            if len(result) >= num_samples:
                break
    return result[:num_samples]


# ─── Array helpers ──────────────────────────────────────────────────────────


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _audio_to_mono(series: td.Series) -> np.ndarray:
    """Mono waveform from an ``"time"``-having ``Series``: passed through if
    already 1-D, else averaged over every non-time axis (mirrors the old
    ``np.mean(mix, axis=0)`` channel-collapse for multichannel audio)."""
    arr = _to_numpy(series.data).astype(np.float32)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=tuple(range(arr.ndim - 1))).astype(np.float32)


def _sample_rate(series: td.Series) -> int:
    tindex = series.tindex
    if not isinstance(tindex, td.GridIndex):
        raise TypeError(f"expected a GridIndex time axis, got {type(tindex).__name__}")
    return int(round(float(tindex.sr)))


def _caption(sample_id: str, input_snr: float | None) -> str:
    if input_snr is None:
        return sample_id
    return f"{sample_id} {input_snr:+.1f} dB"


# ─── Per-task sample builders ───────────────────────────────────────────────


def _fill_audio_triple(
    vs: ValSample, payload: dict[str, Any], pred: td.Frame, target: td.Frame, caption: str
) -> bool:
    """speech_enhancement-style: mixture/target/output audio triple."""
    if "mixture" not in target or "target" not in target or "enhanced" not in pred:
        return False
    mixture_series = target["mixture"]
    sr = _sample_rate(mixture_series)
    mixture = _audio_to_mono(mixture_series)
    clean = _audio_to_mono(target["target"])
    output = _audio_to_mono(pred["enhanced"])

    vs.audio["mixture"] = (mixture, sr)
    vs.audio["target"] = (clean, sr)
    vs.audio["output"] = (output, sr)
    payload[f"samples/{vs.sample_id}/mixture"] = wandb.Audio(
        mixture, sample_rate=sr, caption=f"mixture {caption}"
    )
    payload[f"samples/{vs.sample_id}/target"] = wandb.Audio(
        clean, sample_rate=sr, caption=f"target {caption}"
    )
    payload[f"samples/{vs.sample_id}/output"] = wandb.Audio(
        output, sample_rate=sr, caption=f"output {caption}"
    )
    return True


def _fill_rps_overlay(
    vs: ValSample, payload: dict[str, Any], pred: td.Frame, target: td.Frame, caption: str
) -> bool:
    """rps_prediction-style: mixture audio + a pred-vs-GT RPS overlay figure
    (no audio output for this task)."""
    if "mixture" not in target:
        return False
    mixture_series = target["mixture"]
    sr = _sample_rate(mixture_series)
    mono = _audio_to_mono(mixture_series)
    vs.audio["mixture"] = (mono, sr)
    payload[f"samples/{vs.sample_id}/mixture"] = wandb.Audio(
        mono, sample_rate=sr, caption=f"mixture {caption}"
    )

    if "rps" in target and "rps_pred" in pred:
        try:
            import matplotlib.pyplot as plt

            from plots.rps_prediction.full_sequence import plot_full_sequence

            rps_gt = _to_numpy(target["rps"].data)
            rps_pred = _to_numpy(pred["rps_pred"].data)
            fig = plot_full_sequence(
                audio=mono, rps_gt=rps_gt, rps_pred=rps_pred, sr=float(sr), title=caption
            )
            image = wandb.Image(fig, caption=f"RPS pred vs GT — {caption}")
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            vs.figures["rps_overlay"] = buf.getvalue()
            payload[f"samples/{vs.sample_id}/rps_overlay"] = image
        except Exception:
            logger.warning(
                "val_logging: failed to build RPS overlay figure for sample %s",
                vs.sample_id,
                exc_info=True,
            )
    return True


def _fill_noise_gen_pair(
    vs: ValSample,
    payload: dict[str, Any],
    pred: td.Frame,
    target: td.Frame,
    caption: str,
    epoch: int,
) -> bool:
    """noise_generation-style: the REAL recorded drone noise (``target["audio"]``)
    alongside the model's GENERATED noise (``pred["audio"]``).

    Fixes the confusing fallback where noise_generation samples were routed
    through :func:`_fill_mixture_only` and logged the real recording under a
    ``"mixture"`` key (so validation "samples" looked like the untouched input
    audio). Captions carry the drone name (``meta.drone``) and epoch so the
    real/generated pair is unambiguous in wandb."""
    if "audio" not in target or "audio" not in pred:
        return False
    real_series = target["audio"]
    sr = _sample_rate(real_series)
    real = _audio_to_mono(real_series)
    generated = _audio_to_mono(pred["audio"])

    drone = get_meta(target, "drone", None)
    tag = f"{caption} — ep{epoch}"
    if drone is not None:
        tag = f"{drone} {tag}"

    vs.audio["real"] = (real, sr)
    vs.audio["generated"] = (generated, sr)
    payload[f"samples/{vs.sample_id}/real"] = wandb.Audio(
        real, sample_rate=sr, caption=f"real {tag}"
    )
    payload[f"samples/{vs.sample_id}/generated"] = wandb.Audio(
        generated, sample_rate=sr, caption=f"generated {tag}"
    )
    return True


def _fill_mixture_only(
    vs: ValSample, payload: dict[str, Any], target: td.Frame, caption: str
) -> bool:
    """Fallback for tasks without a dedicated builder: mixture/input audio only."""
    for key in ("mixture", "audio"):
        if key in target:
            series = target[key]
            sr = _sample_rate(series)
            mono = _audio_to_mono(series)
            vs.audio["mixture"] = (mono, sr)
            payload[f"samples/{vs.sample_id}/mixture"] = wandb.Audio(
                mono, sample_rate=sr, caption=f"mixture {caption}"
            )
            return True
    return False


# ─── Entry point ─────────────────────────────────────────────────────────────


def log_validation_samples(
    *,
    task: Task,
    pairs: Sequence[tuple[td.Frame, td.Frame]],
    epoch: int,
    num_samples: int,
    log_fn: Callable[[dict[str, Any]], None],
    metric_suite: Any | None = None,
    artifact_store: ArtifactSink | None = None,
) -> None:
    """Select + log up to ``num_samples`` validation samples for ``epoch``.

    ``pairs`` are the per-sample (no ``"batch"`` dim) ``(pred, target)``
    Frame pairs the validation loop already produces for
    :class:`~metrics.suite.MetricSuite`. Builds an (mixture, target, output)
    audio triple for audio-producing tasks (``speech_enhancement``), or
    mixture audio + a PIT-aligned RPS overlay figure for ``rps_prediction``
    (``plots.rps_prediction.full_sequence.plot_full_sequence``); any other
    task falls back to mixture-audio-only. Logs the resulting
    ``{wandb-key: wandb.Audio | wandb.Image}`` payload via ``log_fn`` and, if
    ``artifact_store`` is given, uploads the same samples to R2
    (``ArtifactStore.upload_val_samples``).

    No-op if ``num_samples <= 0`` or ``pairs`` is empty.
    """
    if num_samples <= 0 or not pairs:
        return

    targets = [target for _, target in pairs]
    indices = select_val_sample_indices(targets, num_samples)
    if not indices:
        return

    val_samples: list[ValSample] = []
    payload: dict[str, Any] = {}
    for idx in indices:
        pred, target = pairs[idx]
        raw_id = get_meta(target, "id", None)
        sample_id = str(raw_id) if raw_id is not None else f"val_ep{epoch}_{idx:03d}"
        raw_snr = get_meta(target, "input_snr", None)
        input_snr = float(raw_snr) if raw_snr is not None else None
        caption = _caption(sample_id, input_snr)

        vs = ValSample(sample_id=sample_id, input_snr=input_snr)
        if metric_suite is not None:
            try:
                vs.metrics = metric_suite.evaluate_one(pred, target)
            except Exception:
                logger.warning(
                    "val_logging: metric_suite.evaluate_one failed for sample %s",
                    sample_id,
                    exc_info=True,
                )

        built = False
        if task.name == "rps_prediction":
            built = _fill_rps_overlay(vs, payload, pred, target, caption)
        elif task.name == "speech_enhancement":
            built = _fill_audio_triple(vs, payload, pred, target, caption)
        elif task.name == "noise_generation":
            built = _fill_noise_gen_pair(vs, payload, pred, target, caption, epoch)
        if not built:
            built = _fill_mixture_only(vs, payload, target, caption)
        if built:
            val_samples.append(vs)

    if payload:
        log_fn(payload)
    if artifact_store is not None and val_samples:
        artifact_store.upload_val_samples(epoch, val_samples)
