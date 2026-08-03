"""Tests for the speech-enhancement online-mix task and the ``audio_pool``
noise source (telemetry-free dload audio datasets).

All streams are compiled pipelines over synthetic local-repo datasets (see
``conftest.py``); the pure mixing helpers (``_apply_one_augmentation_pair``,
``_scale_source_to_snr``) are unit-tested directly.
"""

from __future__ import annotations

import io
import itertools

import numpy as np
import pytest
import soundfile as sf
import tdseries as td

import data_processing.streams as streams
from data_processing.frames import audio_series, make_recording_frame
from data_processing.online_mixing import (
    _apply_one_augmentation_pair,
    _scale_source_to_snr,
    build_online_mix_pipeline,
)

from .conftest import SPEECH_DATASET, SR, speech_dataset  # noqa: F401


def _se_cfg(noise, **over):
    cfg = {
        "sample_rate": SR,
        "duration_s": 1.0,
        "base_seed": 123,
        "task": "speech_enhancement",
        "sources": {
            "noise": noise,
            "speech": [{"dataset": SPEECH_DATASET, "subpath": "LibriSpeech/train-clean-100"}],
        },
        "policy": {"snr_db": -10.0},
    }
    cfg.update(over)
    return cfg


def _take(cfg, n, start=0):
    pipe = build_online_mix_pipeline({**cfg, "start_sample_id": start})
    return list(itertools.islice(iter(pipe), n))


def _snr_db_of(frame) -> float:
    mix = np.asarray(frame["mixture"].data, dtype=np.float64)
    tgt = np.asarray(frame["target"].data, dtype=np.float64)
    noise = mix - tgt
    return float(10.0 * np.log10(np.mean(tgt**2) / np.mean(noise**2)))


# ─── SE-target mode ─────────────────────────────────────────────────────────────


def test_se_mode_yields_mixture_and_clean_target_shapes(speech_dataset):  # noqa: F811
    frames = _take(_se_cfg([{"kind": "audio_pool", "dataset": SPEECH_DATASET}]), 2)
    f = frames[0]
    assert set(f.keys()) == {"mixture", "target", "meta"}
    assert f["mixture"].dims == ("time",)
    assert np.asarray(f["mixture"].data).shape == (SR,)
    assert np.asarray(f["target"].data).shape == (SR,)


def test_se_mode_returned_pair_snr_matches_drawn_snr(speech_dataset):  # noqa: F811
    for snr in (-30.0, -12.5, 0.0):
        frames = _take(
            _se_cfg([{"kind": "audio_pool", "dataset": SPEECH_DATASET}], policy={"snr_db": snr}),
            3,
        )
        for f in frames:
            assert abs(_snr_db_of(f) - snr) < 1e-2, (snr, _snr_db_of(f))


def test_se_mode_multichannel_noise_reduced_to_mono(patched_repo, speech_dataset):  # noqa: F811
    # Multichannel tdframe noise -> the SE stream picks one mic (mono contract).
    rng = np.random.default_rng(0)
    arr = (0.1 * rng.standard_normal((4, 2 * SR))).astype(np.float32)
    frame = make_recording_frame(
        {"audio": td.uniform(arr, SR, dims=("mic", "time"), t_start=0.0)},
        meta={"recording_id": "multi"},
    )
    patched_repo.commit(
        "SE-MULTI", [("multi", streams.frame_to_sample(frame))], meta={"layout": "tdframe-v1"}
    )
    frames = _take(_se_cfg([{"kind": "audio_pool", "dataset": "SE-MULTI"}]), 2)
    assert all(f["mixture"].dims == ("time",) for f in frames)


def test_se_mode_requires_speech_pool(patched_repo, speech_dataset):  # noqa: F811
    cfg = _se_cfg([{"kind": "audio_pool", "dataset": SPEECH_DATASET}])
    del cfg["sources"]["speech"]
    with pytest.raises(ValueError, match="requires a sources.speech pool"):
        build_online_mix_pipeline(cfg)


def test_se_mode_is_deterministic(speech_dataset):  # noqa: F811
    cfg = _se_cfg(
        [{"kind": "audio_pool", "dataset": SPEECH_DATASET}],
        policy={"snr_db": {"uniform": {"low": -30.0, "high": 0.0}}},
    )
    a = _take(cfg, 4)
    b = _take(cfg, 4)
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(
            np.asarray(fa["mixture"].data), np.asarray(fb["mixture"].data)
        )
        np.testing.assert_array_equal(np.asarray(fa["target"].data), np.asarray(fb["target"].data))


def test_se_mode_silent_noise_never_zeroes_the_sample(patched_repo, speech_dataset):  # noqa: F811
    """A silent noise chunk would collapse the clean target to all-zeros via
    the source-to-noise scaling; the silence filter keeps such chunks out of
    the stream entirely (the all-zero-sample bug)."""
    silent = np.zeros((1, 2 * SR), np.float32)
    rng = np.random.default_rng(0)
    real = (0.1 * rng.standard_normal((1, 2 * SR))).astype(np.float32)
    samples = []
    for key, arr in (("silent", silent), ("real", real)):
        frame = make_recording_frame(
            {"audio": td.uniform(arr[0], SR, dims=("time",), t_start=0.0)},
            meta={"recording_id": key},
        )
        samples.append((key, streams.frame_to_sample(frame)))
    patched_repo.commit("SE-SILENT", samples, meta={"layout": "tdframe-v1"})

    frames = _take(_se_cfg([{"kind": "audio_pool", "dataset": "SE-SILENT"}]), 6)
    assert frames, "stream yielded nothing"
    for f in frames:
        assert float(np.mean(np.asarray(f["target"].data) ** 2)) > 0.0
        assert float(np.mean(np.asarray(f["mixture"].data) ** 2)) > 0.0


# ─── augmentation consistency ───────────────────────────────────────────────────


def test_augmentation_pair_applies_identical_transform():
    rng = np.random.default_rng(1)
    mixture = rng.standard_normal((1, 100)).astype(np.float32)
    target = 0.3 * mixture  # target is a scaled component of the mixture
    spec = {"probability": 1.0, "choices": [{"random_gain": {"min_db": -6, "max_db": 6}}]}
    mix2, tgt2 = _apply_one_augmentation_pair(mixture.copy(), target.copy(), spec, rng)
    # The gain factor is identical on both, so the ratio target/mixture is preserved.
    np.testing.assert_allclose(target / mixture, tgt2 / mix2, atol=1e-5)


def test_augmentation_pair_polarity_flips_both():
    mixture = np.array([[1.0, -2.0, 3.0]], np.float32)
    target = np.array([[0.5, -1.0, 1.5]], np.float32)
    spec = {"probability": 1.0, "choices": ["random_polarity"]}
    mix2, tgt2 = _apply_one_augmentation_pair(
        mixture.copy(), target.copy(), spec, np.random.default_rng(0)
    )
    np.testing.assert_allclose(mix2, -mixture)
    np.testing.assert_allclose(tgt2, -target)


def test_augmentation_preserves_se_pair_snr(speech_dataset):  # noqa: F811
    cfg = _se_cfg(
        [{"kind": "audio_pool", "dataset": SPEECH_DATASET}],
        policy={
            "snr_db": -8.0,
            "augmentations": {
                "probability": 1.0,
                "choices": [{"random_gain": {"min_db": -6, "max_db": 6}}, "random_polarity"],
            },
        },
    )
    for f in _take(cfg, 3):
        assert abs(_snr_db_of(f) - (-8.0)) < 1e-2


# ─── _scale_source_to_snr ───────────────────────────────────────────────────────


def test_scale_source_to_snr_global_and_per_channel():
    rng = np.random.default_rng(4)
    noise = rng.standard_normal((2, SR)).astype(np.float32)
    source = rng.standard_normal((2, SR)).astype(np.float32)
    scaled = _scale_source_to_snr(source, noise, -6.0)
    got = 10.0 * np.log10(np.mean(scaled**2) / np.mean(noise**2))
    assert abs(got - (-6.0)) < 1e-3

    scaled_pc = _scale_source_to_snr(source, noise, -6.0, per_channel=True)
    for c in range(2):
        got_c = 10.0 * np.log10(np.mean(scaled_pc[c] ** 2) / np.mean(noise[c] ** 2))
        assert abs(got_c - (-6.0)) < 1e-3


# ─── audio_pool source ──────────────────────────────────────────────────────────


def _publish_tdframe_audio(repo, name, recs):
    """recs: list[(key, (C,T) float32 array, sr)] -> tdframe-v1 dataset."""
    samples = []
    for key, arr, sr in recs:
        frame = make_recording_frame({"audio": audio_series(arr, sr)}, meta={"recording_id": key})
        samples.append((key, streams.frame_to_sample(frame)))
    repo.commit(name, samples, meta={"layout": streams.TDFRAME_LAYOUT})


def _publish_raw_audio(repo, name, recs):
    """recs: list[(key, (T,C) float32 array, sr)] -> raw wav-per-sample dataset."""
    samples = []
    for key, arr, sr in recs:
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV", subtype="FLOAT")
        samples.append((key, {"wav": buf.getvalue()}))
    repo.commit(name, samples, meta={})


def _chunk_fingerprints(cfg, n=12):
    """Distinct per-chunk max-abs values over the first n chunks (each fixture
    recording is a distinct constant amplitude, so this identifies recordings)."""
    frames = _take(cfg, n)
    return {round(float(np.abs(np.asarray(f["mixture"].data)).max()), 2) for f in frames}


def test_audio_pool_tdframe_multichannel_resample_and_loop(patched_repo, speech_dataset):  # noqa: F811
    # 2-channel 44.1 kHz recording, 0.5 s long -> resampled to 16 k and looped
    # to fill a 1 s chunk (then mixed mono for SE).
    sr = 44100
    arr = np.stack(
        [np.sin(np.linspace(0, 20, sr // 2)), np.cos(np.linspace(0, 20, sr // 2))]
    ).astype(np.float32)
    _publish_tdframe_audio(patched_repo, "AP-TDF", [("recA", arr, sr)])
    frames = _take(_se_cfg([{"kind": "audio_pool", "dataset": "AP-TDF"}]), 3)
    assert all(np.asarray(f["mixture"].data).shape == (SR,) for f in frames)


def test_audio_pool_raw_layout(patched_repo, speech_dataset):  # noqa: F811
    arr = np.random.default_rng(0).standard_normal((SR * 2, 1)).astype(np.float32)  # (T, C=1)
    _publish_raw_audio(patched_repo, "AP-RAW", [("recA", arr, SR)])
    frames = _take(_se_cfg([{"kind": "audio_pool", "dataset": "AP-RAW"}]), 2)
    assert all(np.asarray(f["mixture"].data).shape == (SR,) for f in frames)


def test_audio_pool_skips_non_audio_samples(patched_repo, speech_dataset):  # noqa: F811
    # Datasets like new-drone-noises interleave csv flight-log samples with wav;
    # the pool must skip the non-audio ones, not crash.
    arr = np.random.default_rng(0).standard_normal((SR, 1)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, SR, format="WAV", subtype="FLOAT")
    samples = [("wavA", {"wav": buf.getvalue()}), ("logB", {"csv": b"t,rps\n0,30\n"})]
    patched_repo.commit("AP-MIXED", samples, meta={})
    frames = _take(_se_cfg([{"kind": "audio_pool", "dataset": "AP-MIXED"}]), 4)
    assert len(frames) == 4


def test_audio_pool_include_and_exclude_keys(patched_repo, speech_dataset):  # noqa: F811
    # Keys match exactly OR as a substring; exclude wins on overlap.
    recs = [
        ("S1_seq1", 0.01 * np.ones((1, SR), np.float32), SR),
        ("S1_seq2", 0.02 * np.ones((1, SR), np.float32), SR),
        ("S2_speech", 0.03 * np.ones((1, SR), np.float32), SR),
    ]
    _publish_tdframe_audio(patched_repo, "AP-EXC", recs)

    # exclude by substring: "seq" drops both seq* recs -> only S2_speech mixes in
    cfg = _se_cfg(
        [{"kind": "audio_pool", "dataset": "AP-EXC", "exclude_keys": ["seq"], "channel": 0}],
        policy={"snr_db": -40.0},  # speech far below noise: noise amplitude dominates the max
    )
    seen = _chunk_fingerprints(cfg)
    assert seen == {round(0.03, 2)}

    cfg = _se_cfg(
        [{"kind": "audio_pool", "dataset": "AP-EXC", "include_keys": ["S1_seq"], "channel": 0}],
        policy={"snr_db": -40.0},
    )
    assert _chunk_fingerprints(cfg) == {round(0.01, 2), round(0.02, 2)}

    cfg = _se_cfg(
        [
            {
                "kind": "audio_pool",
                "dataset": "AP-EXC",
                "include_keys": ["S1_seq1", "S1_seq2"],
                "exclude_keys": ["S1_seq2"],
                "channel": 0,
            }
        ],
        policy={"snr_db": -40.0},
    )
    assert _chunk_fingerprints(cfg) == {round(0.01, 2)}


def test_audio_pool_holdout_shard_split(patched_repo, speech_dataset):  # noqa: F811
    # Multi-shard dataset: the last `valid_shards` whole shards are the valid
    # partition; train gets the complement. Four 1-sample shards.
    repo = patched_repo
    samples = []
    for i in range(4):
        arr = (0.01 * (i + 1)) * np.ones((1, SR), np.float32)
        frame = make_recording_frame(
            {"audio": audio_series(arr, SR)}, meta={"recording_id": f"r{i}"}
        )
        samples.append((f"r{i}", streams.frame_to_sample(frame)))
    repo.commit("AP-HOLD", samples, meta={"layout": streams.TDFRAME_LAYOUT}, target_shard_size=1)

    train_cfg = _se_cfg(
        [
            {
                "kind": "audio_pool",
                "dataset": "AP-HOLD",
                "channel": 0,
                "holdout": {"split": "train", "valid_shards": 1},
            }
        ],
        policy={"snr_db": -40.0},
    )
    valid_cfg = _se_cfg(
        [
            {
                "kind": "audio_pool",
                "dataset": "AP-HOLD",
                "channel": 0,
                "holdout": {"split": "valid", "valid_shards": 1},
            }
        ],
        policy={"snr_db": -40.0},
    )
    train_seen, valid_seen = _chunk_fingerprints(train_cfg), _chunk_fingerprints(valid_cfg)
    assert train_seen and valid_seen
    assert train_seen.isdisjoint(valid_seen)
    assert valid_seen == {round(0.04, 2)}  # the last shard's recording
