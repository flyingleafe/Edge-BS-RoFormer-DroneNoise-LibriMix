"""``zoo.FrameModel`` round-trip on the tiny synthetic RPS fixtures — CPU-fast,
no network, no Hydra compose (``zoo.load`` is exercised only down to its
building blocks, which are covered elsewhere: compose in scripts, checkpoint
resolution in ``utils.checkpoints``)."""

from __future__ import annotations

import tdseries as td
import torch

from data_processing.collate import frame_collate
from tasks.codecs import build_codec
from tasks.task import TASK_FACTORIES
from tests.training._fixtures import TinyRPSModel, make_tiny_frame
from zoo import FrameModel

HOP = 512
SR = 16000
PARAMS = {"frame_rate": (SR, HOP)}


def make_frame_model() -> FrameModel:
    task = TASK_FACTORIES["rps_prediction"](**PARAMS)
    codec = build_codec("rps_prediction", **PARAMS)
    model = TinyRPSModel(hop_length=HOP, num_rotors=4).eval()
    return FrameModel(model, codec, task, experiment="tiny_exp")


def test_unbatched_frame_round_trip():
    fm = make_frame_model()
    frame = make_tiny_frame(recording_id="t0", input_snr=-10.0, duration_s=0.5)
    n_frames = int(0.5 * SR) // HOP + 1

    out = fm(frame)

    assert isinstance(out, td.Frame)
    pred = out["rps_pred"]
    assert pred.dims == ("rotor", "time")
    assert tuple(pred.data.shape) == (4, n_frames)
    assert isinstance(pred.data, torch.Tensor)
    assert not pred.data.requires_grad


def test_batched_frame_passes_through_batched():
    fm = make_frame_model()
    frames = [
        make_tiny_frame(recording_id=f"t{i}", input_snr=-10.0, duration_s=0.5) for i in range(2)
    ]
    batch = frame_collate(frames)
    n_frames = int(0.5 * SR) // HOP + 1

    out = fm(batch)

    pred = out["rps_pred"]
    assert pred.dims == ("batch", "rotor", "time")
    assert tuple(pred.data.shape) == (2, 4, n_frames)


def test_attrs_and_repr():
    fm = make_frame_model()
    assert fm.task_name == "rps_prediction"
    assert fm.experiment == "tiny_exp"
    assert isinstance(fm.model, TinyRPSModel)
    text = repr(fm)
    assert "rps_prediction" in text
    assert "TinyRPSModel" in text
    assert "tiny_exp" in text
