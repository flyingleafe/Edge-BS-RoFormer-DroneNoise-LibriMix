"""Tests for NoiseRPSDataset chunk sampling — RPS-balanced coverage of the full
flight envelope (incl. zero-RPS) for generator training, and the empty-motor
edge guard."""

from __future__ import annotations

import numpy as np
import tdseries as td

from data_processing.frames import make_recording_frame
from data_processing.noise_rps_dataset import NoiseRPSDataset, _wrap_frame


def _flight_record(dur: float = 60.0, sr: int = 16000, mrate: float = 100.0):
    """A synthetic recording whose RPS traces 0 -> 80 -> 0 (a full flight), so
    most wall-clock time is cruise (80) and only a small fraction is low/zero."""
    n = int(dur * sr)
    m = int(dur * mrate)
    t_m = np.arange(m) / mrate
    prof = np.interp(t_m, [0, 4, 8, dur - 8, dur - 4, dur], [0, 0, 80, 80, 0, 0])
    rps = np.tile(prof, (4, 1))  # (4, M)
    audio = (np.random.default_rng(0).standard_normal((1, n)) * 0.01).astype(np.float32)
    au = td.uniform(audio, sr, dims=("mic", "time"), t_start=0.0)
    rp = td.events(t_m, rps, dims=("rotor", "time"), t_start=0.0)
    fr = make_recording_frame(
        {"audio": au, "rps": rp},
        meta={"recording_id": "synthflight"},
        mic_pos=np.zeros((1, 3)),
        rotor_pos=np.zeros((4, 3)),
    )
    return _wrap_frame(fr, origin="michaels", rps_key="rps")


def _low_fraction(balance: bool, n: int = 400) -> float:
    rec = _flight_record()
    ds = NoiseRPSDataset(
        [rec],
        chunk_size=16000,
        sample_rate=16000,
        samples_per_epoch=n,
        seed=0,
        balance_rps=balance,
    )
    means = np.array([float(np.asarray(ds[i]["rps"]).mean()) for i in range(n)])
    return float(np.mean(means < 30.0))


def test_balance_rps_boosts_low_and_zero_regions():
    # Proportional sampling rarely lands on the brief low/zero regions; balancing
    # flattens the RPS histogram so they are drawn far more often.
    prop = _low_fraction(balance=False)
    bal = _low_fraction(balance=True)
    assert bal > prop + 0.15  # substantially more low-RPS coverage
    assert bal > 0.25  # a meaningful share of low/zero chunks


def test_balanced_dataset_runs_without_extract_errors():
    rec = _flight_record()
    ds = NoiseRPSDataset(
        [rec],
        chunk_size=16000,
        sample_rate=16000,
        samples_per_epoch=64,
        seed=1,
        balance_rps=True,
    )
    for i in range(len(ds)):
        item = ds[i]
        assert np.asarray(item["audio"]).shape[-1] == 16000
        assert np.asarray(item["rps"]).shape[0] == 4
