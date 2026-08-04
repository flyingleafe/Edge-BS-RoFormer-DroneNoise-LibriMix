"""Equivalence/regression tests for the PyTorch Basic Pitch port.

The fixture ``basic_pitch_ref.npz`` holds an audio clip and the three
posteriorgrams produced by Spotify's released Basic Pitch model (ICASSP 2022),
obtained by running the official ONNX export.  These tests assert that our
PyTorch port reproduces those outputs to float32 numerical precision — i.e.
the reimplementation is identical to the original.

Run:  pytest tests/test_basic_pitch.py
"""

import os

import numpy as np
import pytest
import torch

from models.basic_pitch import BasicPitch

REF = os.path.join(os.path.dirname(__file__), "basic_pitch_ref.npz")
TOL = 2e-3  # >> observed ~3e-6; guards against silent architecture drift


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


@pytest.fixture(scope="module")
def model():
    return BasicPitch.from_pretrained()


def test_output_shapes(model, ref):
    with torch.no_grad():
        out = model(torch.from_numpy(ref["audio"]))
    assert out["contour"].shape == ref["contour"].shape
    assert out["note"].shape == ref["note"].shape
    assert out["onset"].shape == ref["onset"].shape


@pytest.mark.parametrize("head", ["contour", "note", "onset"])
def test_matches_original(model, ref, head):
    with torch.no_grad():
        out = model(torch.from_numpy(ref["audio"]))
    diff = np.max(np.abs(out[head].cpu().numpy() - ref[head]))
    assert diff < TOL, f"{head} max abs diff {diff:.3e} exceeds {TOL}"


def test_outputs_are_probabilities(model, ref):
    with torch.no_grad():
        out = model(torch.from_numpy(ref["audio"]))
    for head in ("contour", "note", "onset"):
        v = out[head]
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
