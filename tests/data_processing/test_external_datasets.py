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


def test_no_datasets_are_streaming():
    # HF parquet datasets snapshot-download then read local (faster/reliable than
    # per-row fsspec range reads over throttled HF).
    assert all(not ext.get(n).streaming for n in ext.list_names())


def test_build_drone_detection_from_local_parquet(tmp_path):
    """The parquet builder reads snapshotted local shards (audio bytes + label)."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    audio = (np.random.default_rng(0).standard_normal((800, 1)) * 0.1).astype(np.float32)
    sf.write(buf, audio, SR, format="WAV")
    rows = {
        "audio": [
            {"bytes": buf.getvalue(), "path": "a.wav"},
            {"bytes": buf.getvalue(), "path": "b.wav"},
        ],
        "label": [1, 0],
    }
    (tmp_path / "data").mkdir()
    pq.write_table(pa.table(rows), str(tmp_path / "data" / "train-00000.parquet"))

    frames = dict(ext.build_drone_detection(tmp_path))
    assert len(frames) == 2
    assert {f["meta"]["label"]["class"] for f in frames.values()} == {"drone", "no_drone"}


def test_build_droneaudioset_arrow_path(tmp_path):
    """DroneAudioSet parquet (list<list<double>> audio) read via arrow buffers,
    not to_pylist — the fix for the huge-array hang. Multichannel + path meta."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    arr = np.random.default_rng(2).standard_normal((2, 500)) * 0.1  # (C, T)
    audio = {"array": arr.tolist(), "sampling_rate": 48000, "path": "x.wav"}
    table = pa.table(
        {
            "audio": [audio, audio],
            "file_path": [
                "drone-only/mic-dist-25cm/throttle-high/x.wav",
                "source-only/y.wav",
            ],
            "data_type": ["drone", "source"],
        }
    )
    (tmp_path / "drone-only").mkdir()
    pq.write_table(table, str(tmp_path / "drone-only" / "train_001.parquet"))

    frames = list(ext.build_droneaudioset(tmp_path))
    assert len(frames) == 2
    _, frame = frames[0]
    assert frame["audio"].dims == ("mic", "time")
    assert frame["audio"].shape == (2, 500)
    assert int(frame["audio"].tindex.sr) == 48000
    assert frame["meta"]["observation"]["mic_to_source_m"] == 0.25
    assert frame["meta"]["operating"]["throttle"] == "high"
    assert frame["meta"]["label"]["subset"] == "drone-only"


def test_build_droneaudioset_samples_major(tmp_path):
    """Real DroneAudioSet is samples-major (outer list = time). The reshape
    path must not iterate the huge outer dim and must still yield (C, T)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tc = np.random.default_rng(3).standard_normal((500, 2)) * 0.1  # (T, C) samples-major
    audio = {"array": tc.tolist(), "sampling_rate": 16000, "path": "z.wav"}
    table = pa.table({"audio": [audio], "file_path": ["drone-only/z.wav"], "data_type": ["drone"]})
    (tmp_path / "drone-only").mkdir()
    pq.write_table(table, str(tmp_path / "drone-only" / "s.parquet"))

    frames = list(ext.build_droneaudioset(tmp_path))
    assert len(frames) == 1
    assert frames[0][1]["audio"].shape == (2, 500)  # transposed to (C, T)


def test_build_hustmotor_parses_header_and_channels(tmp_path):
    """HUST .txt: text header then tab-separated time,X,Y,Z,Sound → acoustic
    (Sound) as audio + 3-axis vibration track; sr from the time increment."""
    n, sr = 500, 25600
    t = np.arange(n) / sr
    cols = np.stack([t, t * 0 + 1, t * 0 + 2, t * 0 + 3, np.sin(2 * np.pi * 50 * t)], axis=1)
    header = "Title:\tBF_10HZ\nDAQ Settings:\nChannels:\nLegend\tX\tY\tZ\tSound\n"
    header += "Time (seconds) and Data Channels\n"
    lines = header + "\n".join("\t".join(f"{v:.8f}" for v in row) for row in cols)
    (tmp_path / "Raw data").mkdir()
    (tmp_path / "Raw data" / "BF_10HZ.txt").write_text(lines)

    frames = dict(ext.build_hustmotor(tmp_path))
    assert len(frames) == 1
    frame = next(iter(frames.values()))
    assert frame["audio"].dims == ("time",)
    assert frame["audio"].shape == (n,)  # the Sound channel
    assert frame["vibration"].shape == (3, n)  # X, Y, Z
    assert frame["meta"]["label"]["health"] == "bearing_fault"
    # the "Sound" column is the 50 Hz sine, not a constant vibration axis
    assert float(np.std(np.asarray(frame["audio"].data))) > 0.1


def test_build_kaist_reads_signal_struct(tmp_path):
    """KAIST .mat: Signal struct with y_values.values + x_values.increment."""
    from scipy.io import savemat

    arr = np.sin(2 * np.pi * 100 * np.arange(2000) / 51200).astype(np.float64)
    aco = tmp_path / "acoustic"
    aco.mkdir()
    savemat(
        str(aco / "0Nm_BPFI_03.mat"),
        {"Signal": {"x_values": {"increment": 1.0 / 51200}, "y_values": {"values": arr}}},
    )
    frames = dict(ext.build_kaist_acoustic(tmp_path))
    assert len(frames) == 1
    frame = next(iter(frames.values()))
    assert frame["audio"].shape == (2000,)
    assert int(frame["audio"].tindex.sr) == 51200
    assert frame["meta"]["label"]["fault"] == "BPFI"
    assert frame["meta"]["label"]["severity"] == "03"
