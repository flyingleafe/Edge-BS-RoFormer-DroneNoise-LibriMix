"""Tests for data_processing.derivations (dload derived-dataset declarations).

The fast checks (specs integrity, fingerprint stability, wav encoding, meta
shape) are torch-free — importing ``derivations`` must not pull torch, so
offline fingerprinting/adoption works on any box. The final round-trip test
imports ``streams`` (→ torch) lazily inside the test body, so
``-k "not roundtrip"`` stays torch-free.
"""

from __future__ import annotations

import io
import itertools
import subprocess
import sys
from pathlib import Path

import dload
import numpy as np
import pytest
import soundfile as sf

import data_processing.derivations as der

from .conftest import speech_dataset  # noqa: F401

SR = 16000


def test_import_is_torch_free():
    # Check a *fresh* interpreter: pytest shares sys.modules across the session,
    # so a sibling test file that imports torch would otherwise false-fail this.
    code = "import sys, data_processing.derivations as _; sys.exit('torch' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"importing derivations pulled torch:\n{r.stderr}"


def test_wav_bytes_roundtrip_mono_and_multichannel():
    rng = np.random.default_rng(0)
    for shape in [(SR,), (SR, 8)]:
        arr = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        raw, sr = sf.read(io.BytesIO(der.wav_bytes(arr, SR)), dtype="float32")
        assert sr == SR
        # Default WAV subtype (matches the disk-writing CLIs) is PCM_16, so
        # allow one 16-bit quantization step rather than exact equality.
        np.testing.assert_allclose(raw.reshape(arr.shape), arr, atol=2e-4)


def test_dataset_meta_shape():
    for name in der.SPECS:
        meta = der.dataset_meta(name)
        assert meta["layout"] == der.spec_layout(name)
        if meta["layout"] == "sample-dir-v1":
            assert meta["fields"] == der.SPECS[name]["fields"]
            assert meta["meta_sample"] == {
                "key": "_meta",
                "fields": {"metadata": "metadata.json"},
            }
            assert "mixture" in meta["fields"]


def test_specs_integrity():
    for name, entry in der.SPECS.items():
        assert entry["generator"] in der._GENERATORS, name
        assert isinstance(entry["adopt_only"], bool), name
        gen = entry["gen"]
        assert "recipe_version" in gen, f"{name} missing gen['recipe_version']"
        kind = entry["generator"]
        if kind in ("dregon_lm", "dn_lm"):
            assert entry["fields"], name
            for key in ("seed", "num_samples", "split", "parents", "params"):
                assert key in gen, f"{name} missing gen[{key!r}]"
            assert gen["split"] in ("train", "valid"), name
        elif kind == "se_valid":
            for key in ("seed", "categories", "category_noise", "per_snr", "snr_grid"):
                assert key in gen, f"{name} missing gen[{key!r}]"
        elif kind in ("source_frames", "raw_subset"):
            assert "source" in gen, f"{name} missing gen['source']"
        elif kind == "frame_subset":
            assert "parent" in gen and "include_keys" in gen, name
        elif kind == "beatvk_valid":
            assert "parents" in gen and "recordings" in gen, name
        # Parents are pinned dload URIs, never local data/ paths.
        for key in ("parents",):
            for uri in (gen.get(key) or {}).values():
                assert uri.startswith("dload:") and "@" in uri, f"{name}: {uri}"
        for key in ("parent",):
            if gen.get(key):
                assert gen[key].startswith("dload:") and "@" in gen[key], name
        for spec in gen.get("noise_sources") or []:
            assert "@" in spec["dataset"], f"{name}: unpinned noise source {spec}"


def test_fingerprint_stable_and_unique():
    fps = {name: der.fingerprint(name) for name in der.SPECS}
    # deterministic
    assert all(der.fingerprint(n) == fps[n] for n in der.SPECS)
    # every dataset gets a distinct derivation identity
    assert len(set(fps.values())) == len(fps)


def test_fingerprint_tracks_recipe_version(monkeypatch):
    """Bumping recipe_version must change the fingerprint (stale-snapshot guard)."""
    name = "DN-LM-train"
    before = der.fingerprint(name)
    bumped = dict(der.SPECS[name])
    bumped["gen"] = {**bumped["gen"], "recipe_version": 999}
    monkeypatch.setitem(der.SPECS, name, bumped)
    assert der.fingerprint(name) != before


def _lock_datasets() -> dict[str, str]:
    """``dload.lock``'s pinned ``{dataset: version}``, repo-root relative."""
    import tomllib

    lock_path = Path(der.__file__).resolve().parents[2] / "dload.lock"
    with lock_path.open("rb") as fh:
        return tomllib.load(fh)["datasets"]


def test_parents_match_dload_lock():
    """Pin-drift guard: PARENTS must carry the current dload.lock versions.
    Update deliberately — a change mints new derivation identities."""
    lock = _lock_datasets()
    for key, uri in der.PARENTS.items():
        name, _, sha = uri.removeprefix("dload:").partition("@")
        assert lock.get(name) == sha, (
            f"PARENTS[{key!r}] stale: {sha[:12]} != lock {lock.get(name, '')[:12]}"
        )


def test_lock_coverage():
    """Every dload.lock dataset is covered: a derivation spec, a raw source
    registry entry (or its frames dataset), or a declared historical pin."""
    from data_processing import sources

    lock = _lock_datasets()
    frames_names = {s.frames_name for s in sources.REGISTRY.values()}
    for name in lock:
        covered = (
            name in der.SPECS
            or name in sources.REGISTRY
            or name in frames_names
            or name in der.HISTORICAL_PINS
        )
        assert covered, f"{name}: no spec, source entry, or historical-pin declaration"


def test_build_pipeline_is_fingerprintable():
    # No ValueError from dload's fingerprint = the partial(module-fn, json-spec)
    # bridge is legal (no lambdas/locals/non-JSON params).
    for name in der.SPECS:
        assert isinstance(der.build_pipeline(name).fingerprint(), str)


def test_sample_convention_roundtrips_through_streams(tmp_path, monkeypatch):
    """The (key, {field: bytes}) convention derivations emits must decode and
    reconstruct through the real consumer (streams), using derivations'
    encoders + dataset_meta — the plumbing proof for a materialized dataset."""
    from dload.cache import ShardCache
    from dload.remote import LocalRemote
    from dload.repo import Repository

    import data_processing.streams as streams

    repo = Repository(
        LocalRemote(tmp_path / "remote"),
        ShardCache(tmp_path / "cache", None),
        lock_path=tmp_path / "dload.lock",
    )
    monkeypatch.setattr(streams, "_repository", repo)

    rng = np.random.default_rng(1)
    n, channels, hop = 3, 8, 512
    samples = []
    for i in range(n):
        mix = (rng.standard_normal((SR, channels)) * 0.1).astype(np.float32)
        rps = rng.uniform(20.0, 100.0, size=(4, 29)).astype(np.float32)
        samples.append(
            (
                f"sample_{i:05d}",
                {
                    "mixture": der.wav_bytes(mix, SR),
                    "vocals": der.wav_bytes(mix, SR),
                    "noise": der.wav_bytes(mix, SR),
                    "rps": dload.codecs.npy_bytes(rps),
                    "meta": dload.codecs.json_bytes(
                        {"id": f"sample_{i:05d}", "input_snr": -5.0 - i}
                    ),
                },
            )
        )
    samples.append(
        ("_meta", {"metadata": dload.codecs.json_bytes({"train": [{"id": "sample_00000"}]})})
    )

    name = "DREGON-LM-V4-train"
    manifest = repo.commit(name, samples, meta=der.dataset_meta(name))
    dataset = dload.Dataset(repo, manifest)

    # (a) streaming decode → training Frame
    from functools import partial

    frames = list(
        streams.to_frames(
            dataset.samples(), partial(streams.decode_dregon_lm, sample_rate=SR, hop_length=hop)
        )
    )
    assert len(frames) == n  # _meta dropped
    assert frames[0]["mixture"].dims == ("mic", "time")
    assert frames[0]["mixture"].shape == (channels, SR)
    assert frames[0]["rps"].dims == ("rotor", "time")

    # (b) ensure_local reconstructs the sample-dir tree + split metadata.json
    root = streams.ensure_local(name)
    assert (root / "sample_00000" / "mixture.wav").exists()
    assert (root / "sample_00000" / "rps.npy").exists()
    assert (root / "sample_00000" / "meta.json").exists()
    assert (root / "metadata.json").exists()  # from the _meta bookkeeping sample


def test_se_valid_category_grid_and_snr(patched_repo, speech_dataset):  # noqa: F811
    """``iter_se_valid_category`` yields per_snr clips per grid point, each at
    its grid SNR, with the clean target being the mixture's speech component.

    Guards the SE-valid rebuild path (the replacement for the deleted
    ``scripts/build_se_valid.py``) — see ``data_processing.derivations``.
    """
    from .conftest import SPEECH_DATASET

    snr_grid = [-20.0, -5.0]
    clips = list(
        der.iter_se_valid_category(
            "probe",
            [{"kind": "audio_pool", "dataset": SPEECH_DATASET}],
            per_snr=2,
            snr_grid=snr_grid,
            duration_s=0.5,
            sample_rate=SR,
            seed=7,
            heldout_speakers=["19"],
            librispeech=SPEECH_DATASET,
        )
    )
    assert len(clips) == 2 * len(snr_grid)
    assert [k for k, _ in clips] == [
        "probe_snr-20_0000",
        "probe_snr-20_0001",
        "probe_snr-05_0002",
        "probe_snr-05_0003",
    ]
    for _, frame in clips:
        mixture = np.asarray(frame["mixture"].data, np.float32).reshape(-1)
        target = np.asarray(frame["target"].data, np.float32).reshape(-1)
        snr = float(frame["meta"]["input_snr"])
        assert snr in snr_grid
        noise = mixture - target
        assert 10.0 * np.log10(np.mean(target**2) / np.mean(noise**2)) == pytest.approx(
            snr, abs=1e-2
        )


def test_se_valid_speaker_holdout_is_respected(patched_repo, speech_dataset):  # noqa: F811
    """The valid speech pool draws ONLY the held-out speakers — the guarantee
    that no speaker is shared with a training policy's ``exclude`` list."""
    from data_processing.online_mixing import build_speech_stream

    from .conftest import SPEECH_DATASET

    pipe = build_speech_stream(
        {"dataset": SPEECH_DATASET, "subpath": "LibriSpeech/train-clean-100", "include": ["19"]},
        sample_rate=SR,
        window_s=0.5,
        lanes=1,
        seed=3,
    )
    assert len(list(itertools.islice(pipe, 6))) == 6  # cycles the 4 speaker-19 files


def test_se_valid_frame_crops_the_off_by_one_noise_window():
    """A noise window one sample longer than the speech lane (inclusive
    time-slice bounds) must be cropped, not crash the broadcast — the defect
    that aborted the first SE-valid v2 re-derivation."""
    import tdseries as td

    from data_processing.frames import audio_series

    target_len = SR // 2
    noise_tf = td.Frame({"audio": audio_series(np.ones((1, target_len + 1), np.float32), SR)})
    speech = [np.full(target_len, 0.1, np.float32)]
    frame = der._se_valid_frame("probe", SR, [-10.0], 1, 0, noise_tf, speech, 0.0)
    mixture = np.asarray(frame["mixture"].data)
    target = np.asarray(frame["target"].data)
    assert mixture.shape[-1] == target_len
    assert target.shape[-1] == target_len
    noise = mixture - target
    assert 10.0 * np.log10(np.mean(target**2) / np.mean(noise**2)) == pytest.approx(-10.0, abs=1e-2)
