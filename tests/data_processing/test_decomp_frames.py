"""Unit tests for the ``decomp-frames-v1`` derivation and its loader.

No network: the dense-envelope transform and the chunk loader are exercised on
synthetic in-memory records. The derivation's carrier helpers are pinned against
``scripts/vk_decompose.py`` itself in ``tests/scripts/test_vk_decompose.py``,
where the script is importable.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from data_processing.derivations import dense_envelopes
from data_processing.frame_datasets import DecompFrameDataset

SR = 16000
STRIDE = DecompFrameDataset.ENV_STRIDE


# ---------------------------------------------------------------------------
# sparse tracks -> the dense (rotor, k) grid every recording shares


def test_dense_envelopes_places_tracks_and_masks_the_rest():
    n_mic, n_env, k_max = 3, 7, 10
    rotor = np.array([0, 0, 1, 1])
    k = np.array([1, 4, 2, 12])  # k=12 is above k_max and must be dropped
    amp = np.stack([np.full((4, n_env), float(m + 1)) for m in range(n_mic)])  # (mic, track, time)
    valid = np.ones((4, n_env), dtype=bool)
    valid[1, 3:] = False

    dense, mask = dense_envelopes(amp, valid, rotor, k, k_max)
    assert dense.shape == (n_mic, 2, k_max, n_env)
    assert mask.shape == (2, k_max, n_env)
    # Each track sits at [mic, rotor, k-1] with its microphone's value.
    np.testing.assert_allclose(dense[:, 0, 0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dense[:, 1, 1, 0], [1.0, 2.0, 3.0])
    # Everything that is not a solved track is zero and invalid.
    assert dense[:, 0, 2].max() == 0.0
    assert not mask[0, 2].any()
    assert not mask[1].any(axis=-1)[2:].any()  # the k=12 track is simply gone
    # The per-track validity survives.
    assert mask[0, 3, 0] and not mask[0, 3, 3]


# ---------------------------------------------------------------------------
# the loader


def _record(rid: str, drone: str, n_s: float = 8.0, rps: float = 60.0) -> dict:
    n_t = int(n_s * SR)
    n_env = n_t // STRIDE
    return {
        "recording_id": rid,
        "drone": drone,
        "sample_rate": SR,
        "rps": np.full((4, n_t), rps, dtype=np.float32),
        "residual": np.zeros((8, n_t), dtype=np.float32),
        "amp": np.arange(n_env, dtype=np.float32)[None, None, None, :]
        .repeat(8, 0)
        .repeat(4, 1)
        .repeat(80, 2),
        "amp_valid": np.ones((4, 80, n_env), dtype=bool),
        "mic_pos": np.zeros((8, 3), dtype=np.float32),
        "rotor_pos": np.ones((4, 3), dtype=np.float32),
        "span": (0, n_t),
    }


def test_chunks_are_fixed_shape_and_grid_aligned():
    ds = DecompFrameDataset([_record("a", "dregon")], chunk_size=16000, n_samples=8)
    shapes = set()
    for i in range(len(ds)):
        f = ds[i]
        shapes.add(
            (
                np.asarray(f["rps"].data).shape,
                np.asarray(f["amp"].data).shape,
                np.asarray(f["residual"].data).shape,
            )
        )
        # The envelope chunk is exactly the audio chunk's span: the loader cuts
        # by index off a stride-aligned start, so a batch always stacks.
        start = int(f["meta"]["start_sample"])
        assert start % STRIDE == 0
        amp = np.asarray(f["amp"].data)
        assert amp[0, 0, 0, 0] == start // STRIDE
        assert isinstance(f["amp"].tindex, td.GridIndex)
    assert shapes == {((4, 16000), (8, 4, 80, 100), (8, 16000))}


def test_idle_chunks_are_rejected():
    """A record below the flight gate is not drawn while a flying one exists."""
    idle = _record("idle", "dregon", rps=5.0)
    flying = _record("fly", "michaels", rps=70.0)
    ds = DecompFrameDataset([idle, flying], chunk_size=16000, n_samples=16, min_motor_rps=30.0)
    drawn = {str(ds[i]["meta"]["recording_id"]) for i in range(len(ds))}
    assert drawn == {"fly"}


def test_train_and_valid_spans_do_not_overlap():
    """The held-out block is in the MIDDLE, train is the two pieces around it."""
    rec = _record("a", "dregon")
    n_t = rec["rps"].shape[-1]
    n_val = int(round(0.1 * n_t)) // STRIDE * STRIDE
    v0 = (n_t - n_val) // 2 // STRIDE * STRIDE
    tr = DecompFrameDataset(
        [{**rec, "span": (0, v0)}, {**rec, "span": (v0 + n_val, n_t)}],
        chunk_size=1600,
        n_samples=32,
    )
    va = DecompFrameDataset(
        [{**rec, "span": (v0, v0 + n_val)}], chunk_size=1600, n_samples=32, split="valid"
    )
    va_starts = {int(va[i]["meta"]["start_sample"]) for i in range(len(va))}
    assert min(va_starts) >= v0 and max(va_starts) + 1600 <= v0 + n_val
    for i in range(len(tr)):
        start = int(tr[i]["meta"]["start_sample"])
        assert start + 1600 <= v0 or start >= v0 + n_val


# ---------------------------------------------------------------------------
# the combined (multi-rig) pool — the v3 decompositions are published per rig


def test_dataset_names_normalize_to_name_version_pairs():
    f = DecompFrameDataset._dataset_versions
    assert f("one", None) == [("one", None)]
    assert f(["a", "b"], "v9") == [("a", "v9"), ("b", "v9")]
    assert f(["a", "b"], ["v1", None]) == [("a", "v1"), ("b", None)]


def test_a_version_list_must_match_the_dataset_list():
    with pytest.raises(ValueError, match="one per dataset"):
        DecompFrameDataset._dataset_versions(["a", "b"], ["v1"])
    with pytest.raises(ValueError, match="at least one dataset"):
        DecompFrameDataset._dataset_versions([], None)


def _published(rid: str, drone: str, n_s: float = 8.0, rps: float = 70.0) -> td.Frame:
    """One published recording, in the shape ``_load_records`` decodes."""
    rec = _record(rid, drone, n_s=n_s, rps=rps)
    n_t = rec["rps"].shape[-1]
    n_env = n_t // STRIDE
    grid = td.GridIndex.create((SR, STRIDE), n_env, t_start=0.0)
    return td.Frame(
        {
            "rps": td.uniform(rec["rps"], SR, dims=("rotor", "time"), t_start=0.0),
            "residual": td.uniform(rec["residual"], SR, dims=("mic", "time"), t_start=0.0),
            "amp": td.Series(rec["amp"], ("mic", "rotor", "k", "time"), {"time": grid}),
            "amp_valid": td.Series(rec["amp_valid"], ("rotor", "k", "time"), {"time": grid}),
            "mic_pos": td.wrap(rec["mic_pos"], dims=("mic", None)),
            "rotor_pos": td.wrap(rec["rotor_pos"], dims=("rotor", None)),
            "meta": td.Frame(
                {"recording_id": rid, "drone": drone, "sample_rate": SR},
            ),
        }
    )


def test_two_per_rig_datasets_concatenate_into_one_pool(monkeypatch):
    """A combined arm names both v3 datasets; each record keeps its own rig id."""
    from data_processing import streams

    published = {
        "decomp-frames-v3-dregon": [_published("free-flight_nosource_room1", "dregon")],
        "decomp-frames-v3-michaels": [
            _published("FLY124", "michaels"),
            _published("FLY125", "michaels"),
        ],
    }
    monkeypatch.setattr(streams, "iter_published_frames", lambda name, ver=None: published[name])

    ds = DecompFrameDataset.build_train(
        dataset=list(published),
        chunk_size=16000,
        train_samples=24,
        min_motor_rps=30.0,
    )
    assert len(ds.records) == 6  # 3 recordings x 2 train spans around the middle block
    assert {r["drone"] for r in ds.records} == {"dregon", "michaels"}
    drawn = {(str(ds[i]["meta"]["drone"]), str(ds[i]["meta"]["recording_id"])) for i in range(24)}
    assert {d for d, _ in drawn} == {"dregon", "michaels"}
