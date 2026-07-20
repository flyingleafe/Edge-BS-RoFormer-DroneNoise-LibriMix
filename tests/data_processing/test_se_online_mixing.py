"""Speech-enhancement online mixing (F1 baselines).

Two features under test, both added for the blind SE-baseline program
(``docs/se-baselines-plan.md``):

- **SE-target mode** of :class:`OnlineMixIterableDataset` — with
  ``task="speech_enhancement"`` the stream yields ``(mixture, clean_target)``
  instead of ``(audio, rps_target)``; the clean target is the gain-scaled speech
  exactly as mixed (SNR of the returned pair == the drawn SNR), post-mix
  augmentation is applied identically to mixture and target, and RPS is skipped.
- **``kind: audio_pool``** (:class:`DloadAudioPool`) — a telemetry-free,
  lazily-streamed dload-backed noise pool (random recording, random channel,
  resample, loop/pad), exercised against a hermetic ``LocalRemote`` bucket for
  both ``tdframe-v1`` and raw-audio publishing conventions.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import tdseries as td
from dload.cache import ShardCache
from dload.remote import LocalRemote
from dload.repo import Repository

import data_processing.streams as streams
from data_processing.frame_datasets import OnlineMixFrameDataset
from data_processing.frames import make_recording_frame
from data_processing.online_mixing import (
    DloadAudioPool,
    OnlineMixIterableDataset,
    _apply_one_augmentation_pair,
    _scale_source_to_snr,
)

SR = 16000


# ─── Stub pools (hermetic, no dload) for the SE-mode mixing tests ───────────────


class _StubNoisePool:
    """Fixed-content noise pool: returns the same audio Frame each draw."""

    def __init__(self, audio: np.ndarray, sample_rate: int = SR):
        self._audio = np.ascontiguousarray(audio, dtype=np.float32)  # (C, T)
        self._sr = sample_rate

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        n = int(round(duration_s * self._sr))
        a = self._audio[:, :n]
        if a.shape[0] == 1:
            series = td.uniform(a[0], self._sr, dims=("time",), t_start=0.0)
        else:
            series = td.uniform(a, self._sr, dims=("mic", "time"), t_start=0.0)
        return td.Frame({"audio": series})


class _StubSourcePool:
    """Fixed-content speech pool."""

    def __init__(self, audio: np.ndarray, sample_rate: int = SR):
        self._audio = np.ascontiguousarray(audio, dtype=np.float32)  # (T,)
        self._sr = sample_rate

    def sample_array(self, rng, *, channels, mode="independent"):
        return np.tile(self._audio[None, :], (channels, 1)).astype(np.float32)


def _se_dataset(policy, *, noise=None, speech=None, duration_s=1.0):
    rng = np.random.default_rng(0)
    noise = noise if noise is not None else 0.1 * rng.standard_normal((1, SR)).astype(np.float32)
    speech = speech if speech is not None else rng.standard_normal(SR).astype(np.float32)
    return OnlineMixIterableDataset(
        _StubNoisePool(noise),
        _StubSourcePool(speech),  # type: ignore[arg-type]
        policy=policy,
        base_seed=123,
        duration_s=duration_s,
        sample_rate=SR,
        task="speech_enhancement",
    )


# ─── SE-target mode ─────────────────────────────────────────────────────────────


def test_se_mode_yields_mixture_and_clean_target_shapes():
    ds = _se_dataset({"snr_db": -10.0})
    mix, tgt = ds.generate_sample(0)
    assert mix.shape == (1, SR)
    assert tgt.shape == (1, SR)
    assert mix.dtype == tgt.dtype


def test_se_mode_returned_pair_snr_matches_drawn_snr():
    # Fixed scalar SNR, no augmentation: the clean target sits at exactly the
    # requested SNR relative to the noise component (mixture - target).
    for snr in (-30.0, -12.5, 0.0):
        ds = _se_dataset({"snr_db": snr})
        mix, tgt = ds.generate_sample(3)
        noise = (mix - tgt).numpy()
        got = 10.0 * np.log10(np.mean(tgt.numpy() ** 2) / np.mean(noise**2))
        assert abs(got - snr) < 1e-2, (snr, got)


def test_se_mode_multichannel_noise_reduced_to_mono():
    # Real DREGON/Michael's noise is 8-channel; the SE stream must pick one mic
    # so mixture/target are mono (1, T) — the codec's mono SE contract.
    rng = np.random.default_rng(0)
    noise8 = 0.1 * rng.standard_normal((8, SR)).astype(np.float32)
    ds = _se_dataset({"snr_db": -10.0}, noise=noise8)
    mix, tgt = ds.generate_sample(0)
    assert mix.shape == (1, SR)
    assert tgt.shape == (1, SR)


def test_se_mode_requires_speech_pool():
    with pytest.raises(ValueError, match="requires a sources.speech pool"):
        OnlineMixIterableDataset(
            _StubNoisePool(np.zeros((1, SR), np.float32)),
            None,
            task="speech_enhancement",
            sample_rate=SR,
        )


def test_se_mode_is_deterministic():
    ds = _se_dataset({"snr_db": {"uniform": {"low": -30.0, "high": 0.0}}})
    m1, t1 = ds.generate_sample(9)
    m2, t2 = ds.generate_sample(9)
    assert np.allclose(m1.numpy(), m2.numpy())
    assert np.allclose(t1.numpy(), t2.numpy())


def test_rps_mode_unchanged_default_task():
    # Default task is still rps_prediction and still yields a (audio, rps) pair.
    rng = np.random.default_rng(0)
    noise = 0.1 * rng.standard_normal((1, SR)).astype(np.float32)

    class _NoisePoolWithRps(_StubNoisePool):
        def sample_timeframe(self, rng, duration_s):
            fr = super().sample_timeframe(rng, duration_s)
            n_frames = self._audio.shape[-1] // 512 + 1
            rps = td.Series(
                np.full((4, n_frames), 30.0, np.float32),
                ("rotor", "time"),
                {"time": td.GridIndex.create((SR, 512), n_frames, t_start=0)},
            )
            return td.Frame({"audio": fr["audio"], "rps": rps})

    ds = OnlineMixIterableDataset(
        _NoisePoolWithRps(noise), None, sample_rate=SR, task="rps_prediction"
    )
    audio, rps = ds.generate_sample(0)
    assert audio.shape == (1, SR)
    assert rps.shape[0] == 4


# ─── augmentation consistency ───────────────────────────────────────────────────


def test_augmentation_pair_applies_identical_transform():
    rng = np.random.default_rng(1)
    mixture = rng.standard_normal((1, 100)).astype(np.float32)
    target = 0.3 * mixture  # target is a scaled component of the mixture
    spec = {"probability": 1.0, "choices": [{"random_gain": {"min_db": -6, "max_db": 6}}]}
    mix2, tgt2 = _apply_one_augmentation_pair(mixture.copy(), target.copy(), spec, rng)
    # The gain factor is identical on both, so the ratio target/mixture is preserved.
    ratio_before = target / mixture
    ratio_after = tgt2 / mix2
    assert np.allclose(ratio_before, ratio_after, atol=1e-5)


def test_augmentation_pair_polarity_flips_both():
    mixture = np.array([[1.0, -2.0, 3.0]], np.float32)
    target = np.array([[0.5, -1.0, 1.5]], np.float32)
    spec = {"probability": 1.0, "choices": ["random_polarity"]}
    mix2, tgt2 = _apply_one_augmentation_pair(
        mixture.copy(), target.copy(), spec, np.random.default_rng(0)
    )
    assert np.allclose(mix2, -mixture)
    assert np.allclose(tgt2, -target)


def test_augmentation_preserves_se_pair_snr():
    # With augmentation on, the SNR of the (mixture, target) pair is unchanged,
    # because the same scalar hits both signals.
    ds = _se_dataset(
        {
            "snr_db": -8.0,
            "augmentations": {
                "probability": 1.0,
                "choices": [{"random_gain": {"min_db": -6, "max_db": 6}}, "random_polarity"],
            },
        }
    )
    mix, tgt = ds.generate_sample(2)
    noise = (mix - tgt).numpy()
    got = 10.0 * np.log10(np.mean(tgt.numpy() ** 2) / np.mean(noise**2))
    assert abs(got - (-8.0)) < 1e-2


# ─── OnlineMixFrameDataset SE packing ───────────────────────────────────────────


def test_frame_dataset_se_packs_mixture_target():
    cfg_ds = _se_dataset({"snr_db": -10.0})
    fds = OnlineMixFrameDataset(cfg_ds)
    frame = next(iter(fds))
    assert set(frame.keys()) == {"mixture", "target", "meta"}
    assert frame["mixture"].dims == ("time",)
    assert frame["target"].dims == ("time",)
    assert np.asarray(frame["mixture"].data).shape == (SR,)


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


# ─── DloadAudioPool (hermetic LocalRemote) ──────────────────────────────────────


@pytest.fixture
def patched_repo(tmp_path, monkeypatch) -> Repository:
    repo = Repository(
        LocalRemote(tmp_path / "remote"),
        ShardCache(tmp_path / "cache", None),
        lock_path=tmp_path / "dload.lock",
    )
    monkeypatch.setattr(streams, "_repository", repo)
    return repo


def _publish_tdframe_audio(repo, name, recs):
    """recs: list[(key, (C,T) float32 array, sr)] -> tdframe-v1 dataset."""
    samples = []
    for key, arr, sr in recs:
        dims = ("time",) if arr.shape[0] == 1 else ("mic", "time")
        data = arr[0] if arr.shape[0] == 1 else arr
        frame = make_recording_frame(
            {"audio": td.uniform(data, sr, dims=dims, t_start=0.0)},
            meta={"recording_id": key},
        )
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


def test_audio_pool_tdframe_multichannel_resample_and_loop(patched_repo):
    # 2-channel 44.1 kHz recording, 0.5 s long -> must resample to 16 k and loop
    # to fill a 1 s chunk.
    sr = 44100
    arr = np.stack(
        [np.sin(np.linspace(0, 20, sr // 2)), np.cos(np.linspace(0, 20, sr // 2))]
    ).astype(np.float32)
    _publish_tdframe_audio(patched_repo, "AP-TDF", [("recA", arr, sr)])
    pool = DloadAudioPool("AP-TDF", sample_rate=SR)
    assert pool._layout == "tdframe-v1"
    frame = pool.sample_timeframe(np.random.default_rng(0), 1.0)
    audio = frame["audio"]
    assert audio.dims == ("time",)
    assert np.asarray(audio.data).shape == (SR,)  # looped/padded to 1 s @ 16 k
    assert int(audio.tindex.sr) == SR


def test_audio_pool_raw_layout(patched_repo):
    arr = np.random.default_rng(0).standard_normal((SR * 2, 1)).astype(np.float32)  # (T, C=1)
    _publish_raw_audio(patched_repo, "AP-RAW", [("recA", arr, SR)])
    pool = DloadAudioPool("AP-RAW", sample_rate=SR)
    assert pool._layout is None
    frame = pool.sample_timeframe(np.random.default_rng(0), 1.0)
    assert np.asarray(frame["audio"].data).shape == (SR,)


def test_audio_pool_skips_non_audio_samples(patched_repo):
    # Datasets like new-drone-noises interleave csv flight-log samples with wav;
    # the pool must skip the non-audio ones, not crash.
    arr = np.random.default_rng(0).standard_normal((SR, 1)).astype(np.float32)
    import io as _io

    import soundfile as _sf

    buf = _io.BytesIO()
    _sf.write(buf, arr, SR, format="WAV", subtype="FLOAT")
    samples = [("wavA", {"wav": buf.getvalue()}), ("logB", {"csv": b"t,rps\n0,30\n"})]
    patched_repo.commit("AP-MIXED", samples, meta={})
    pool = DloadAudioPool("AP-MIXED", sample_rate=SR)
    for _ in range(20):  # would raise on the csv sample without the skip
        frame = pool.sample_timeframe(np.random.default_rng(0), 0.5)
        assert np.asarray(frame["audio"].data).shape == (SR // 2,)


def test_audio_pool_shard_weighting_covers_all_records(patched_repo):
    recs = [(f"r{i}", (0.01 * (i + 1)) * np.ones((1, SR), np.float32), SR) for i in range(4)]
    _publish_tdframe_audio(patched_repo, "AP-MULTI", recs)
    pool = DloadAudioPool("AP-MULTI", sample_rate=SR)
    assert pool.num_samples == 4
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        frame = pool.sample_timeframe(rng, 0.25)
        # each record is a distinct constant amplitude -> identify by max abs
        seen.add(round(float(np.abs(frame["audio"].data).max()), 4))
    assert len(seen) == 4  # all four recordings are reachable


def _fingerprints(pool, n=300):
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(n):
        frame = pool.sample_timeframe(rng, 0.25)
        seen.add(round(float(np.abs(frame["audio"].data).max()), 4))
    return seen


def test_audio_pool_holdout_train_valid_are_disjoint(patched_repo):
    recs = [(f"r{i}", (0.01 * (i + 1)) * np.ones((1, SR), np.float32), SR) for i in range(4)]
    _publish_tdframe_audio(patched_repo, "AP-HOLD", recs)
    train = DloadAudioPool("AP-HOLD", sample_rate=SR, holdout={"fraction": 0.5, "split": "train"})
    valid = DloadAudioPool("AP-HOLD", sample_rate=SR, holdout={"fraction": 0.5, "split": "valid"})
    train_seen, valid_seen = _fingerprints(train), _fingerprints(valid)
    assert train_seen and valid_seen
    assert train_seen.isdisjoint(valid_seen)  # no recording reused across the split
    assert train_seen | valid_seen == _fingerprints(DloadAudioPool("AP-HOLD", sample_rate=SR))


def test_source_pool_exclude_speakers(tmp_path):
    import soundfile as _sf

    from data_processing.online_mixing import AudioFileSourcePool

    for spk in ("103", "1034", "200"):
        d = tmp_path / spk / "chap"
        d.mkdir(parents=True)
        _sf.write(d / f"{spk}-0.flac", np.zeros(SR, np.float32), SR)
    pool = AudioFileSourcePool.from_config(
        {"root": str(tmp_path), "glob": "**/*.flac", "exclude": ["200"]},
        duration_s=1.0,
        sample_rate=SR,
    )
    assert all("/200/" not in p.as_posix() for p in pool.files)
    assert any("/103/" in p.as_posix() for p in pool.files)
