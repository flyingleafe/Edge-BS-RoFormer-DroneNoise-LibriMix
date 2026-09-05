"""The noise-only twin of the frozen real validation split, read end to end.

``DREGON-LM-V4-michaels-valid-full-nospeech`` is the 23 source-free clips of
``DREGON-LM-V4-michaels-valid-full``, copied byte for byte (see
``derivations.generate_dregon_lm_subset``). This test pulls both pinned
datasets from R2 through the ``conf/data`` valid block that training builds and
checks that a twin clip is the same window, with the same audio and the same
rps, as the parent clip it was cut from.

Marked ``slow`` + ``network`` (both deselected by default, see ``addopts`` in
``pyproject.toml``): run with ``pytest -m "slow and network"`` and real R2
credentials in ``.env``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from dotenv import load_dotenv

load_dotenv()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.network,
    pytest.mark.skipif(
        not os.environ.get("R2_ACCOUNT_ID"),
        reason="R2_ACCOUNT_ID not set; skipping live dload pull",
    ),
]

PARENT = "DREGON-LM-V4-michaels-valid-full"
TWIN = f"{PARENT}-nospeech"
N_CLIPS = 23
N_CHANNELS = 8
N_PROBE = 2  # clips compared in full: one per source recording


def _valid_dataset(path: str, **overrides):
    """The ``valid`` block of a Hydra data yaml, built as training builds it."""
    from training.config import build_dataset

    spec = yaml.safe_load(Path(path).read_text())["valid"]
    spec["params"] = {**spec["params"], **overrides}
    return build_dataset(spec)


def _rows(name: str) -> list[dict]:
    from data_processing.streams import ensure_local

    root = ensure_local(name)
    return json.loads((root / "metadata.json").read_text())["valid"]


def test_twin_clips_match_the_parent_clips_they_were_cut_from():
    twin = _valid_dataset("conf/data/m3cur_s2_nospeech.yaml", flatten_channels=False)
    parent = _valid_dataset("conf/data/m3cur_s2.yaml", flatten_channels=False)
    assert len(twin) == N_CLIPS
    assert len(parent) == 37

    twin_rows, parent_rows = _rows(TWIN), _rows(PARENT)
    assert {r["source_type"] for r in twin_rows} == {"nosource"}
    by_id = {r["id"]: i for i, r in enumerate(parent_rows)}

    # One clip per source recording: the first DREGON one and the first FLY124 one.
    probes = []
    for recording in sorted({r["recording_id"] for r in twin_rows}):
        probes.append(next(i for i, r in enumerate(twin_rows) if r["recording_id"] == recording))
    assert len(probes) == N_PROBE

    for i in probes:
        row = twin_rows[i]
        t, p = twin[i], parent[by_id[row["source_id"]]]
        assert t["mixture"].data.shape == (N_CHANNELS, 8 * 16000)
        assert t["rps"].data.shape[0] == 4
        assert t["mixture"].data.shape == p["mixture"].data.shape
        assert t["rps"].data.shape == p["rps"].data.shape
        np.testing.assert_array_equal(t["mixture"].data, p["mixture"].data)
        np.testing.assert_array_equal(t["rps"].data, p["rps"].data)


def test_flattened_length_is_one_frame_per_clip_and_mic():
    twin = _valid_dataset("conf/data/m3cur_s2_nospeech.yaml")
    assert len(twin) == N_CLIPS * N_CHANNELS
    frame = twin[0]
    assert frame["mixture"].data.shape == (8 * 16000,)
    assert frame["rps"].data.shape[0] == 4
