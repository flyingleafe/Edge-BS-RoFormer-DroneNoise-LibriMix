"""A probe must preserve conditioning on the waveform's physical time span."""

import numpy as np
import pytest
import rps_cue_probe as cue
import tdseries as td
import torch

from data_processing.frames import audio_series
from models.jhtr import JHTR
from training.config import build_task_and_codec
from zoo.frame_model import FrameModel


@pytest.mark.parametrize("probe", ["freq", "cutoff"])
def test_conditional_probe_is_invariant_to_absolute_time_shift(probe: str) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = JHTR(n_blocks=1, d_model=16, n_heads=2, k_max=4, harmonic_chunk=2)
        # Exercise a non-identity model without needing an external checkpoint.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.001 * torch.randn_like(parameter))
    task, codec = build_task_and_codec(
        {
            "task": "rps_prediction",
            "task_params": {
                "n_channels": None,
                "sr": (16000, 1),
                "frame_rate": (16000, 512),
                "use_cond": True,
            },
        }
    )
    fm = FrameModel(model.eval(), codec, task, experiment="cue-probe", device="cpu")
    t = np.arange(16000) / 16000
    rates = np.tile(60 + 2 * np.arange(32) * 0.032, (4, 1)).astype(np.float32)
    frame = td.Frame(
        {
            "mixture": audio_series(np.sin(2 * np.pi * 120 * t)[None].astype(np.float32), 16000),
            "rps": td.uniform(np.full_like(rates, 60), (16000, 512), dims=("rotor", "time")),
            "rps_cond": td.uniform(rates, (16000, 512), dims=("rotor", "time")),
            "meta": td.Frame({"sample_id": 0}),
        }
    )

    def measure(source: td.Frame) -> np.ndarray:
        if probe == "freq":
            result = cue.freq_probe(fm, [source], [0], 1, list(cue.ALPHAS))
            return np.asarray(result["mean_speed"])
        result = cue.cutoff_probe(fm, [source], [0], list(cue.K_CUTS), "fir")
        return np.asarray([[row["mae"], row["median_ratio"]] for row in result["rows"]])

    np.testing.assert_allclose(
        measure(frame.shift(17.25)), measure(frame), atol=2e-5, rtol=0, equal_nan=False
    )
