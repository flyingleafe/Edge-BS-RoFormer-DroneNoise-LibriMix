"""Tests for data_processing.external_datasets.

Registry integrity is torch-free. The build round-trip writes a synthetic
MIMII-like tree, runs the real builder, and serializes each frame through the
actual ``tdframe-v1`` codec (``streams.frame_to_sample`` → ``sample_to_frame``)
— the plumbing proof that a published sample decodes back to the same Frame.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

import data_processing.external_datasets as ext
from data_processing import streams

SR = 16000


def test_registry_integrity():
    for name, spec in ext.EXTERNAL_SPECS.items():
        assert spec.name == name
        assert callable(spec.builder)
        assert spec.download.kind in {"zenodo", "mendeley", "hf", "gdrive"}
        for key in ("source_url", "license", "description"):
            assert spec.provenance.get(key), f"{name} missing provenance[{key!r}]"


def test_dataset_meta_marks_layout():
    for name in ext.list_names():
        meta = ext.dataset_meta(name)
        assert meta[streams.LAYOUT_META_KEY] == "tdframe-v1"
        assert meta["license"]
        assert "source_url" in meta


def test_safe_key_never_leading_underscore():
    assert ext._safe_key("_meta") != "_meta"
    assert ext._safe_key("a/b/c.wav").startswith("a__b__c")
    assert ext._safe_key("///") == "sample"


def _write_mimii_tree(root, snr, machine, unit, condition, stem, channels=8, n=1600):
    d = root / f"{snr}_dB_{machine}" / machine / unit / condition
    d.mkdir(parents=True, exist_ok=True)
    audio = (np.random.default_rng(0).standard_normal((n, channels)) * 0.1).astype(np.float32)
    sf.write(str(d / f"{stem}.wav"), audio, SR)


def test_build_mimii_and_roundtrip(tmp_path):
    _write_mimii_tree(tmp_path, -6, "fan", "id_00", "normal", "00000000")
    _write_mimii_tree(tmp_path, -6, "fan", "id_00", "abnormal", "00000001")

    frames = list(ext.build_mimii(tmp_path))
    assert len(frames) == 2
    keys = {k for k, _ in frames}
    assert len(keys) == 2  # unique keys

    key, frame = frames[0]
    assert frame["audio"].dims == ("mic", "time")
    assert frame["audio"].shape == (8, 1600)
    assert frame["mic_pos"].shape == (8, 3)
    assert frame["source_pos"].shape == (1, 3)
    meta = frame["meta"]
    assert meta["dataset"] == "MIMII"
    assert meta["system"]["machine_type"] == "fan"
    assert meta["system"]["unit_id"] == "id_00"
    assert meta["operating"]["snr_db"] == -6
    assert meta["label"]["normal_vs_anomaly"] in ("normal", "abnormal")

    # Round-trip through the real codec.
    fields = streams.frame_to_sample(frame)
    frame2 = streams.sample_to_frame(fields)
    np.testing.assert_array_equal(np.asarray(frame["audio"].data), np.asarray(frame2["audio"].data))
    assert frame2["meta"]["system"]["machine_type"] == "fan"
    assert frame2["meta"]["operating"]["snr_db"] == -6
    np.testing.assert_array_equal(
        np.asarray(frame["mic_pos"].data), np.asarray(frame2["mic_pos"].data)
    )


def test_decode_drone_detection_row():
    """HF parquet row (audio bytes + int label) → mono Frame with class."""
    import io

    buf = io.BytesIO()
    audio = (np.random.default_rng(0).standard_normal((800, 1)) * 0.1).astype(np.float32)
    sf.write(buf, audio, SR, format="WAV")
    for label, expect in [(1, "drone"), (0, "no_drone")]:
        row = {"audio": {"bytes": buf.getvalue(), "path": "clip.wav"}, "label": label}
        key, frame = ext._decode_drone_detection_row(3, row)
        assert frame["meta"]["label"]["class"] == expect
        assert frame["meta"]["label"]["raw_label"] == label
        assert frame["audio"].dims == ("time",)  # mono
        assert key.startswith("000003_clip")


def test_decode_droneaudioset_row_multichannel():
    """HF parquet row (decoded array + file_path) → multichannel Frame with
    subset/throttle/mic-distance parsed from the path."""
    arr = (np.random.default_rng(1).standard_normal((4, 2000)) * 0.1).tolist()  # (C, T)
    row = {
        "audio": {"array": arr, "sampling_rate": 48000, "path": "x.wav"},
        "file_path": "drone-only/drone2-only/mic-dist-50cm/throttle-low/mic0-x.wav",
        "data_type": "drone",
    }
    key, frame = ext._decode_droneaudioset_row(7, row)
    assert frame["audio"].dims == ("mic", "time")
    assert frame["audio"].shape == (4, 2000)
    assert frame["meta"]["observation"]["mic_to_source_m"] == 0.5
    assert frame["meta"]["operating"]["throttle"] == "low"
    assert frame["meta"]["label"]["subset"] == "drone-only"
    assert frame["meta"]["system"]["drone_token"] == "drone2"


def test_hf_datasets_marked_streaming():
    assert ext.get("drone-detection-samples").streaming
    assert ext.get("DroneAudioSet").streaming
    assert not ext.get("MIMII").streaming
