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


def test_build_drone_detection_reads_labels(tmp_path):
    import csv

    (tmp_path / "audio").mkdir()
    for i, _label in enumerate([1, 0]):
        audio = (np.random.default_rng(i).standard_normal((800, 1)) * 0.1).astype(np.float32)
        sf.write(str(tmp_path / "audio" / f"clip_{i}.wav"), audio, SR)
    with open(tmp_path / "metadata.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "label"])
        w.writerow(["audio/clip_0.wav", 1])
        w.writerow(["audio/clip_1.wav", 0])

    frames = dict(ext.build_drone_detection(tmp_path))
    assert len(frames) == 2
    classes = {f["meta"]["label"]["class"] for f in frames.values()}
    assert classes == {"drone", "no_drone"}
    for f in frames.values():
        assert f["audio"].dims == ("time",)  # mono
