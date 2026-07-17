"""Tests for ``training.val_logging`` — sample selection, triple/overlay
building, and the ``log_fn``/``artifact_store`` hand-off — all against
synthetic ``td.Frame`` batches, with wandb kept out of the loop entirely
(this module never touches the global ``wandb`` singleton; see its
docstring) except for constructing ``wandb.Audio``/``wandb.Image`` value
objects, which is side-effect-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tdseries as td
import wandb

from tasks.spec import FrameSpec
from tasks.task import Task
from training.val_logging import log_validation_samples, select_val_sample_indices

RPS_TASK = Task(name="rps_prediction", input_spec=FrameSpec({}), output_spec=FrameSpec({}))
SE_TASK = Task(name="speech_enhancement", input_spec=FrameSpec({}), output_spec=FrameSpec({}))
NOISE_GEN_TASK = Task(name="noise_generation", input_spec=FrameSpec({}), output_spec=FrameSpec({}))
OTHER_TASK = Task(name="salience_rps", input_spec=FrameSpec({}), output_spec=FrameSpec({}))


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list[Any]]] = []

    def upload_val_samples(self, epoch: int, samples: Any) -> str:
        self.calls.append((epoch, list(samples)))
        return "r2://fake/manifest.json"


class _FakeMetricSuite:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_one(self, pred: td.Frame, target: td.Frame) -> dict[str, float]:
        self.calls += 1
        return {"mse": 0.1}


def _mixture_series(duration_s: float = 0.5, sample_rate: int = 16000) -> td.Series:
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(n).astype(np.float32) * 0.05
    return td.uniform(audio, sample_rate, dims=("time",), t_start=0.0)


def _rps_series(
    # n_frames must exceed plots.rps_prediction.full_sequence's smoothing
    # window (~1s / frame_dur = sr/hop = 16000/512 ≈ 31 frames) — a shorter
    # RPS trace hits a numpy `convolve(..., mode="same")` quirk (output
    # length becomes the *kernel* length, not the signal length, when the
    # kernel is longer than the signal) and the plot call fails.
    num_rotors: int = 4,
    n_frames: int = 64,
    sample_rate: int = 16000,
    hop_length: int = 512,
) -> td.Series:
    rng = np.random.default_rng(1)
    values = rng.uniform(20.0, 100.0, size=(num_rotors, n_frames)).astype(np.float32)
    idx = td.GridIndex.create((sample_rate, hop_length), n_frames, t_start=0)
    return td.Series(values, ("rotor", "time"), {"time": idx})


def _rps_pair(
    sample_id: str, input_snr: float | None, *, with_meta_id: bool = True
) -> tuple[td.Frame, td.Frame]:
    mixture = _mixture_series()
    gt = _rps_series()
    pred = _rps_series()  # different rng seed reuse is fine — random pred
    meta: dict[str, Any] = {}
    if with_meta_id:
        meta["id"] = sample_id
    if input_snr is not None:
        meta["input_snr"] = input_snr
    target = td.Frame({"mixture": mixture, "rps": gt, "meta": td.Frame(meta)})
    pred_frame = td.Frame({"rps_pred": pred})
    return pred_frame, target


def _noise_gen_audio(seed: int, duration_s: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32) * 0.05


def _noise_gen_pair(
    sample_id: str, drone: str, *, sample_rate: int = 16000
) -> tuple[td.Frame, td.Frame, np.ndarray, np.ndarray]:
    """A noise_generation (pred, target) pair with DIFFERENT real/generated
    audio (1-D so ``_audio_to_mono`` is an exact passthrough), plus the two
    raw arrays for equality assertions."""
    real = _noise_gen_audio(0)
    generated = _noise_gen_audio(1)
    assert not np.array_equal(real, generated)
    real_series = td.uniform(real, sample_rate, dims=("time",), t_start=0.0)
    gen_series = td.uniform(generated, sample_rate, dims=("time",), t_start=0.0)
    target = td.Frame({"audio": real_series, "meta": td.Frame({"id": sample_id, "drone": drone})})
    pred_frame = td.Frame({"audio": gen_series})
    return pred_frame, target, real, generated


def _se_pair(sample_id: str, input_snr: float | None) -> tuple[td.Frame, td.Frame]:
    mixture = _mixture_series()
    clean = _mixture_series()
    output = _mixture_series()
    meta: dict[str, Any] = {"id": sample_id}
    if input_snr is not None:
        meta["input_snr"] = input_snr
    target = td.Frame({"mixture": mixture, "target": clean, "meta": td.Frame(meta)})
    pred_frame = td.Frame({"enhanced": output})
    return pred_frame, target


# ─── select_val_sample_indices ──────────────────────────────────────────────


def test_select_indices_snr_stratified_spreads_across_range():
    targets = [
        td.Frame({"meta": td.Frame({"input_snr": float(snr)})})
        for snr in range(-30, 0, 3)  # 10 targets, evenly spaced SNRs
    ]
    indices = select_val_sample_indices(targets, 3)

    assert len(indices) == 3
    assert len(set(indices)) == 3
    snrs = [float(targets[i]["meta"]["input_snr"]) for i in indices]
    # A spread selection covers low and high ends, not just one cluster.
    assert min(snrs) < -15.0
    assert max(snrs) > -15.0


def test_select_indices_falls_back_to_first_n_without_snr():
    targets = [td.Frame({"meta": td.Frame({})}) for _ in range(5)]
    indices = select_val_sample_indices(targets, 3)
    assert indices == [0, 1, 2]


def test_select_indices_falls_back_when_any_snr_missing():
    targets = [
        td.Frame({"meta": td.Frame({"input_snr": -10.0})}),
        td.Frame({"meta": td.Frame({})}),  # missing on this one
        td.Frame({"meta": td.Frame({"input_snr": -5.0})}),
    ]
    indices = select_val_sample_indices(targets, 2)
    assert indices == [0, 1]


def test_select_indices_caps_at_available_count():
    targets = [td.Frame({"meta": td.Frame({"input_snr": float(-i)})}) for i in range(3)]
    indices = select_val_sample_indices(targets, 10)
    assert set(indices) == {0, 1, 2}


def test_select_indices_empty_or_zero():
    assert select_val_sample_indices([], 3) == []
    targets = [td.Frame({"meta": td.Frame({"input_snr": -1.0})})]
    assert select_val_sample_indices(targets, 0) == []


# ─── log_validation_samples: rps_prediction ─────────────────────────────────


def test_log_validation_samples_rps_prediction_builds_mixture_and_overlay():
    pairs = [_rps_pair(f"sample_{i:03d}", input_snr=-30.0 + 5 * i) for i in range(4)]
    logged: list[dict[str, Any]] = []
    store = _FakeStore()
    metric_suite = _FakeMetricSuite()

    log_validation_samples(
        task=RPS_TASK,
        pairs=pairs,
        epoch=2,
        num_samples=2,
        log_fn=logged.append,
        metric_suite=metric_suite,
        artifact_store=store,
    )

    assert len(logged) == 1
    payload = logged[0]
    mixture_keys = [k for k in payload if k.endswith("/mixture")]
    assert len(mixture_keys) == 2
    assert all(isinstance(payload[k], wandb.Audio) for k in mixture_keys)
    overlay_keys = [k for k in payload if k.endswith("/rps_overlay")]
    assert len(overlay_keys) == 2
    assert all(isinstance(payload[k], wandb.Image) for k in overlay_keys)

    assert metric_suite.calls == 2
    assert len(store.calls) == 1
    epoch, uploaded = store.calls[0]
    assert epoch == 2
    assert len(uploaded) == 2
    for vs in uploaded:
        assert "mixture" in vs.audio
        assert "rps_overlay" in vs.figures
        assert vs.metrics == {"mse": 0.1}
        assert vs.input_snr is not None


# ─── log_validation_samples: speech_enhancement ─────────────────────────────


def test_log_validation_samples_speech_enhancement_builds_triple():
    pairs = [_se_pair(f"sample_{i:03d}", input_snr=-20.0 + i) for i in range(3)]
    logged: list[dict[str, Any]] = []
    store = _FakeStore()

    log_validation_samples(
        task=SE_TASK,
        pairs=pairs,
        epoch=0,
        num_samples=2,
        log_fn=logged.append,
        artifact_store=store,
    )

    assert len(logged) == 1
    payload = logged[0]
    for role in ("mixture", "target", "output"):
        keys = [k for k in payload if k.endswith(f"/{role}")]
        assert len(keys) == 2
        assert all(isinstance(payload[k], wandb.Audio) for k in keys)

    epoch, uploaded = store.calls[0]
    for vs in uploaded:
        assert set(vs.audio) == {"mixture", "target", "output"}
        assert vs.metrics == {}  # no metric_suite passed


# ─── log_validation_samples: noise_generation ───────────────────────────────


def test_log_validation_samples_noise_generation_logs_real_and_generated():
    pred, target, real, generated = _noise_gen_pair("sample_000", drone="dregon")
    logged: list[dict[str, Any]] = []
    store = _FakeStore()

    log_validation_samples(
        task=NOISE_GEN_TASK,
        pairs=[(pred, target)],
        epoch=4,
        num_samples=1,
        log_fn=logged.append,
        artifact_store=store,
    )

    assert len(logged) == 1
    payload = logged[0]
    real_keys = [k for k in payload if k.endswith("/real")]
    gen_keys = [k for k in payload if k.endswith("/generated")]
    assert len(real_keys) == 1
    assert len(gen_keys) == 1
    assert isinstance(payload[real_keys[0]], wandb.Audio)
    assert isinstance(payload[gen_keys[0]], wandb.Audio)
    # No leftover "mixture" key from the old fallback path.
    assert not any(k.endswith("/mixture") for k in payload)

    # The archived ValSample carries both arrays, each equal to its source and
    # distinct from the other (the original bug logged the real audio only).
    _epoch, uploaded = store.calls[0]
    (vs,) = uploaded
    assert set(vs.audio) == {"real", "generated"}
    got_real, _sr_r = vs.audio["real"]
    got_gen, _sr_g = vs.audio["generated"]
    assert np.array_equal(got_real, real)
    assert np.array_equal(got_gen, generated)
    assert not np.array_equal(got_real, got_gen)


# ─── Generic behavior ────────────────────────────────────────────────────────


def test_log_validation_samples_noop_when_num_samples_zero():
    pairs = [_rps_pair("s0", input_snr=-5.0)]
    logged: list[dict[str, Any]] = []
    store = _FakeStore()

    log_validation_samples(
        task=RPS_TASK,
        pairs=pairs,
        epoch=0,
        num_samples=0,
        log_fn=logged.append,
        artifact_store=store,
    )

    assert logged == []
    assert store.calls == []


def test_log_validation_samples_noop_when_no_pairs():
    logged: list[dict[str, Any]] = []
    log_validation_samples(task=RPS_TASK, pairs=[], epoch=0, num_samples=5, log_fn=logged.append)
    assert logged == []


def test_log_validation_samples_without_artifact_store_still_logs_to_wandb():
    pairs = [_rps_pair("s0", input_snr=-5.0)]
    logged: list[dict[str, Any]] = []

    log_validation_samples(task=RPS_TASK, pairs=pairs, epoch=0, num_samples=1, log_fn=logged.append)

    assert len(logged) == 1


def test_log_validation_samples_falls_back_to_mixture_only_for_unhandled_task():
    pairs = [_rps_pair("s0", input_snr=-5.0)]
    logged: list[dict[str, Any]] = []
    store = _FakeStore()

    log_validation_samples(
        task=OTHER_TASK,
        pairs=pairs,
        epoch=0,
        num_samples=1,
        log_fn=logged.append,
        artifact_store=store,
    )

    assert len(logged) == 1
    payload = logged[0]
    assert any(k.endswith("/mixture") for k in payload)
    assert not any(k.endswith("/rps_overlay") for k in payload)
    _epoch, uploaded = store.calls[0]
    assert set(uploaded[0].audio) == {"mixture"}


def test_log_validation_samples_sample_id_falls_back_when_meta_id_missing():
    pairs = [_rps_pair("ignored", input_snr=-5.0, with_meta_id=False)]
    logged: list[dict[str, Any]] = []

    log_validation_samples(task=RPS_TASK, pairs=pairs, epoch=3, num_samples=1, log_fn=logged.append)

    assert len(logged) == 1
    keys = list(logged[0].keys())
    # media samples use the "samples/" prefix (distinct from the "val/" scalar
    # charts they used to mask in wandb)
    assert any(k.startswith("samples/val_ep3_000/") for k in keys)
