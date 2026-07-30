"""Tests for the generated-noise source pool (option C: spawn producer + ring).

These spin up the real background producer process on CPU with a tiny random-init
checkpoint, so they exercise the shared-memory buffer, the seqlock read path, and
the Frame wrapping end-to-end (just not GPU or model quality).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import tdseries as td
import torch

from data_processing.frames import get_meta
from data_processing.online_mixing import build_noise_stream


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


def _tiny_flat_conditioned(
    tmp_path, *, drones=("dregon", "michaels"), n_harm: int = 8, cond_dim: int = 8
) -> str:
    """A flat `_CodebookConditionedNoiseGen` state_dict (the modern checkpoint
    format) with per-drone learnable σ + spectral-norm FiLM — exercises the
    `_load_generator` flat/registry-rebuild branch the interp mode relies on."""
    from models.registry import build_noise_gen_model

    composite = build_noise_gen_model(
        "positional_harmonic_gen",
        sample_rate=16000,
        n_harmonics=n_harm,
        use_diff_noise=True,
        cond_dim=cond_dim,
        drone_names=list(drones),
        rps_jitter_sigma=0.6,
        rps_jitter_tau=0.016,
        learn_rps_jitter_sigma=True,
        z_noise_std=0.1,
        film_spectral_norm=True,
    )
    path = str(tmp_path / "tiny_flat_gen.pt")
    torch.save(composite.state_dict(), path)
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


def _rps_values(tf: td.Frame) -> np.ndarray:
    return np.asarray(tf["rps"].data)


def test_generated_pool_yields_wellformed_timeframe(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        tf = pool.sample_timeframe(np.random.default_rng(0), 0.25)
        assert isinstance(tf, td.Frame)
        assert set(tf) == {"audio", "rps", "mic_pos", "rotor_pos", "meta"}
        audio = tf["audio"]
        assert audio.data.shape == (8, int(round(0.25 * 16000)))
        assert int(round(audio.tindex.sr)) == 16000
        assert _rps_values(tf).shape[0] == 4  # 4 rotors, time-last
        assert tf["mic_pos"].data.shape == (8, 3)
        assert tf["rotor_pos"].data.shape == (4, 3)
        assert get_meta(tf, "recording_id") == "generated_michaels"
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


@pytest.mark.skipif(
    not Path("data/DREGON").is_dir(),
    reason="needs local data/DREGON geometry (producer loads it before codebook lookup)",
)
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


@pytest.mark.skipif(
    not Path("data/DREGON").is_dir(),
    reason="interp mode loads both DREGON + Michael's geometry (needs data/DREGON)",
)
def test_generated_pool_interp_mode_yields_wellformed_timeframe(tmp_path):
    """Vicinal embedding + geometry sampling along the DREGON↔Michael's segment:
    the producer must load the flat conditioned checkpoint, interpolate z/rotor/σ,
    sample a mic rig, and still emit a well-formed (audio, exact-RPS) Frame."""
    from data_processing.generated_noise import GeneratedNoisePool

    ck = _tiny_flat_conditioned(tmp_path)
    pool = GeneratedNoisePool(
        ck,
        drone="dregon",  # nominal; interp overrides per-chunk
        device="cpu",
        n_harmonics=8,
        duration_s=0.25,
        dregon_dir="data/DREGON",
        n_slots=8,
        gen_batch=2,
        warmup=2,
        warmup_timeout_s=60.0,
        seed=1,
        interp={
            "endpoints": ["dregon", "michaels"],
            "alpha": {"low": 0.0, "high": 1.0},
            "embedding_noise": 0.15,
            "rotor_interp": True,
            "jitter_sigma": "interp",
            "mic_sampling": {
                "rigs": ["dregon", "michaels"],
                "prob": [0.5, 0.5],
                "jitter_std": 0.02,
            },
        },
    )
    try:
        assert pool.interp is not None
        tf = pool.sample_timeframe(np.random.default_rng(0), 0.25)
        assert isinstance(tf, td.Frame)
        audio = tf["audio"]
        assert audio.data.shape == (8, int(round(0.25 * 16000)))
        assert np.isfinite(np.asarray(audio.data)).all()
        rps = _rps_values(tf)
        assert rps.shape[0] == 4  # 4 rotors, time-last
        assert rps.min() >= 25.0 and rps.max() <= 130.0
    finally:
        pool.close()


# --- engine dispatch (no GPU, no spawn) --------------------------------


class _DummyPool:
    def __init__(self, tag, records=None):
        self.tag = tag
        self.records = records or []

    def sample_timeframe(self, rng, duration_s):
        return self.tag


def test_build_noise_stream_dispatches_engines(monkeypatch):
    # A generated/static engine spec becomes a random_stream.map(render)
    # sub-stream; an all-engine list is a single-stream pipeline.
    import data_processing.online_mixing as om

    monkeypatch.setattr(
        om, "_build_engine", lambda cfg, **kw: _DummyPool(str(cfg.get("kind")))
    )
    stream, ceiling = build_noise_stream(
        [{"kind": "generated", "weight": 3.0}], sample_rate=16000, window_s=1.0, seed=0
    )
    import itertools

    frames = list(itertools.islice(iter(stream), 3))
    assert all(f == "generated" for f in frames)
    assert ceiling == 8
