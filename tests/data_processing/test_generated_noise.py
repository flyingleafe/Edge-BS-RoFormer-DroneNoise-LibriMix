"""Tests for the generated-noise source pool (option C: spawn producer + ring).

These spin up the real background producer process on CPU with a tiny random-init
checkpoint, so they exercise the shared-memory buffer, the seqlock read path, and
the TimeFrame wrapping end-to-end (just not GPU or model quality).
"""

from __future__ import annotations

import time
from typing import cast

import numpy as np
import pytest
import torch

from data_processing.online_mixing import MixedNoisePool, build_noise_pool
from utils.data import EventSeries, TimeFrame, UniformSeries


def _tiny_bundle(tmp_path, *, drone: str = "michaels", n_harm: int = 8, cond_dim: int = 8) -> str:
    from models.generative import PositionalHarmonicNoiseGen
    from tasks.noise_generation import DroneCodebook

    model = PositionalHarmonicNoiseGen(
        sample_rate=16000, n_harmonics=n_harm, use_diff_noise=True, cond_dim=cond_dim
    )
    cb = DroneCodebook(cond_dim, names=[drone])
    path = str(tmp_path / "tiny_gen.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "codebook": cb.state_dict(),
            "cond_dim": cond_dim,
            "drone_names": [drone],
        },
        path,
    )
    return path


def _make_pool(
    tmp_path,
    *,
    n_slots: int = 8,
    gen_batch: int = 2,
    warmup: int = 2,
    refresh: bool = True,
):
    from data_processing.generated_noise import GeneratedNoisePool

    ck = _tiny_bundle(tmp_path)
    return GeneratedNoisePool(
        ck,
        drone="michaels",
        device="cpu",
        n_harmonics=8,
        duration_s=0.25,
        sample_rate=16000,
        n_slots=n_slots,
        gen_batch=gen_batch,
        warmup=warmup,
        refresh=refresh,
        warmup_timeout_s=30.0,
        seed=1,
    )


def _rps_values(tf: TimeFrame) -> np.ndarray:
    return cast(np.ndarray, cast(EventSeries, tf["rps"]).values)


def test_generated_pool_yields_wellformed_timeframe(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        tf = pool.sample_timeframe(np.random.default_rng(0), 0.25)
        assert isinstance(tf, TimeFrame)
        assert set(tf) == {"audio", "rps"}
        audio = cast(UniformSeries, tf["audio"])
        assert audio.samples.shape == (8, int(round(0.25 * 16000)))
        assert int(round(audio.sr)) == 16000
        assert _rps_values(tf).shape[0] == 4  # 4 rotors, time-last
        assert tf.global_data["mic_positions"].shape == (8, 3)
        assert tf.global_data["rotor_positions"].shape == (4, 3)
        assert tf.tags["recording_id"] == "generated_michaels"
    finally:
        pool.close()


def test_generated_pool_rps_label_varies_across_draws(tmp_path):
    # RPS comes straight from the synthetic generator (independent of the random
    # init weights), so distinct slots must carry distinct labels.
    pool = _make_pool(tmp_path, n_slots=16, gen_batch=4, warmup=6)
    try:
        rng = np.random.default_rng(0)
        time.sleep(0.3)  # let a few batches land so multiple slots are filled
        labels = [_rps_values(pool.sample_timeframe(rng, 0.25)) for _ in range(16)]
        distinct = {lab.tobytes() for lab in labels}
        assert len(distinct) >= 2
        stacked = np.stack(labels)
        assert stacked.min() >= 25.0 and stacked.max() <= 130.0
    finally:
        pool.close()


def test_deterministic_bank_fills_once_and_stops(tmp_path):
    # refresh=False => producer fills every slot once, then exits (a reproducible
    # fixed generated bank rather than a live stream).
    pool = _make_pool(tmp_path, refresh=False, n_slots=6, gen_batch=2, warmup=2)
    try:
        deadline = time.time() + 20.0
        while pool._proc.is_alive() and time.time() < deadline:
            time.sleep(0.1)
        assert not pool._proc.is_alive()
        assert int(pool.shared["filled"][0].item()) == 6
        assert int(pool.shared["ready"].sum().item()) == 6
    finally:
        pool.close()


def test_unknown_drone_rejected(tmp_path):
    from data_processing.generated_noise import GeneratedNoisePool

    ck = _tiny_bundle(tmp_path, drone="michaels")
    with pytest.raises(RuntimeError):
        # producer dies (drone not in codebook) -> buffer never warms up
        pool = GeneratedNoisePool(
            ck,
            drone="dregon",
            device="cpu",
            n_harmonics=8,
            duration_s=0.25,
            dregon_dir="data/DREGON",
            n_slots=4,
            gen_batch=2,
            warmup=2,
            warmup_timeout_s=8.0,
            seed=1,
        )
        try:
            pool.sample_timeframe(np.random.default_rng(0), 0.25)
        finally:
            pool.close()


# --- MixedNoisePool / dispatch (no GPU, no spawn) --------------------------------


class _DummyPool:
    def __init__(self, tag, records=None):
        self.tag = tag
        self.records = records or []

    def sample_timeframe(self, rng, duration_s):
        return self.tag


def test_mixed_noise_pool_selects_by_weight_and_aggregates_records():
    a = _DummyPool("a", records=[{"x": 1}])
    b = _DummyPool("b", records=[{"y": 2}])
    pool = MixedNoisePool([a, b], [0.0, 1.0])
    assert all(pool.sample_timeframe(np.random.default_rng(i), 1.0) == "b" for i in range(5))
    assert pool.records == [{"x": 1}, {"y": 2}]


def test_build_noise_pool_dispatches_mixed(monkeypatch):
    import data_processing.generated_noise as gn
    import data_processing.online_mixing as om

    monkeypatch.setattr(
        om.TimeFrameNoisePool,
        "from_config",
        classmethod(lambda cls, cfg, **kw: _DummyPool("real")),
    )
    monkeypatch.setattr(
        gn.GeneratedNoisePool,
        "from_config",
        classmethod(lambda cls, cfg, **kw: _DummyPool("gen")),
    )
    real_only = build_noise_pool([{"kind": "dregon"}], duration_s=1.0, sample_rate=16000)
    assert isinstance(real_only, _DummyPool) and real_only.tag == "real"

    mixed = build_noise_pool(
        [{"kind": "dregon"}, {"kind": "generated", "weight": 3.0}],
        duration_s=1.0,
        sample_rate=16000,
    )
    assert isinstance(mixed, MixedNoisePool)
    assert len(mixed.pools) == 2
    # real default weight 1.0 vs generated 3.0 -> 0.25 / 0.75
    np.testing.assert_allclose(mixed.weights, [0.25, 0.75])
