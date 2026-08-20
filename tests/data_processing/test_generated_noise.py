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
    from models.generative.codebook import DroneCodebook

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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


# --- per-rotor sub-embeddings + the vicinal knobs (no spawn, no data) ------
#
# These drive the producer's batch sampler directly (`make_sampler`), which is
# the same callable `_producer_loop` uses — so the conditioning/geometry the
# tests inspect is exactly what the emitter would be called with, minus the
# process boundary and the render.

_PERROTOR_DELTAS = torch.tensor(
    [
        [0.10, -0.20, 0.05, 0.30, -0.10, 0.00, 0.15, -0.05],
        [-0.30, 0.40, -0.15, 0.10, 0.25, -0.20, 0.05, 0.10],
        [0.50, 0.10, -0.40, -0.20, 0.30, 0.15, -0.25, 0.05],
        [-0.05, -0.35, 0.25, 0.45, -0.30, 0.20, 0.10, -0.15],
    ]
)


def _tiny_flat_perrotor(tmp_path, *, drones=("dregon", "michaels"), n_harm: int = 8) -> str:
    """A flat conditioned state_dict WITH `rotor_deltas` — the gen_m3 layout."""
    from models.registry import build_noise_gen_model

    cond_dim = int(_PERROTOR_DELTAS.shape[-1])
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
        per_rotor_deltas=True,
        n_rotors=4,
    )
    sd = composite.state_dict()
    sd["rotor_deltas"] = _PERROTOR_DELTAS.clone()  # trained deltas (they init at zero)
    path = str(tmp_path / "tiny_perrotor_gen.pt")
    torch.save(sd, path)
    return path


def _load_params(ckpt: str, **over) -> dict:
    params = {
        "checkpoint": ckpt,
        "sample_rate": 16000,
        "n_harmonics": 8,
        "no_diff_noise": False,
        "chunk_s": 0.25,
        "aggressiveness": 1.0,
        "rps_kind": "synthetic_intermittent",
        "flight_fs": 200.0,
        "gen_batch": 2,
        "drone": "dregon",
        "dregon_dir": "data/DREGON",
        "interp": None,
    }
    params.update(over)
    return params


def _interp_cfg(**over) -> dict:
    cfg = {
        "endpoints": ["dregon", "michaels"],
        "alpha": {"low": 0.3, "high": 0.3},  # pinned: isolates the knob under test
        "embedding_noise": 0.0,
        "rotor_interp": True,
        "jitter_sigma": "interp",
        "mic_sampling": {"rigs": ["dregon"], "prob": [1.0], "jitter_std": 0.0},
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def fake_geometry(monkeypatch):
    """Synthetic 8-mic / 4-rotor rigs, so the sampler tests need no DREGON tree."""
    import data_processing.generated_noise as gn

    ang = np.arange(8) * (2 * np.pi / 8)
    mics = {
        "dregon": np.stack([0.1 * np.cos(ang), 0.1 * np.sin(ang), np.zeros(8)], axis=-1),
        "michaels": np.stack([0.2 * np.cos(ang), 0.2 * np.sin(ang), np.full(8, 0.05)], axis=-1),
    }
    rotors = {
        "dregon": np.array(
            [[0.3, 0.3, 0.0], [-0.3, 0.3, 0.0], [-0.3, -0.3, 0.0], [0.3, -0.3, 0.0]]
        ),
        "michaels": np.array(
            [[0.4, 0.4, 0.1], [-0.4, 0.4, 0.1], [-0.4, -0.4, 0.1], [0.4, -0.4, 0.1]]
        ),
    }
    monkeypatch.setattr(gn, "load_geometry", lambda drone, _dir="": (mics[drone], rotors[drone]))
    return mics, rotors


def test_load_generator_rebuilds_per_rotor_checkpoint(tmp_path):
    """The gen_m3 checkpoint carries `rotor_deltas` next to the generator; the
    composite must be rebuilt WITH per_rotor_deltas or the strict load rejects
    the key (missing=0 / unexpected=0 is the contract)."""
    from data_processing.generated_noise import _load_generator
    from models.registry import build_noise_gen_model

    ck = _tiny_flat_perrotor(tmp_path)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    assert "rotor_deltas" in sd

    gb = _load_generator(_load_params(ck), "cpu")
    assert gb.rotor_deltas is not None
    torch.testing.assert_close(gb.rotor_deltas, _PERROTOR_DELTAS)

    # explicit missing/unexpected accounting on the same rebuild
    composite = build_noise_gen_model(
        "positional_harmonic_gen",
        sample_rate=16000,
        n_harmonics=8,
        use_diff_noise=True,
        cond_dim=int(_PERROTOR_DELTAS.shape[-1]),
        drone_names=list(gb.names),
        learn_rps_jitter_sigma=True,
        z_noise_std=0.0,
        film_spectral_norm=True,
        per_rotor_deltas=True,
        n_rotors=4,
    )
    incompat = composite.load_state_dict(sd, strict=False)
    assert list(incompat.missing_keys) == []
    assert list(incompat.unexpected_keys) == []


def test_per_rotor_deltas_reach_the_emitter_code(tmp_path, fake_geometry):
    """`z_r = z_drone + δz_r`: the sampler's code must gain a rotor axis and
    carry each rotor's own delta (what `_fold` then folds into the batch)."""
    from data_processing.generated_noise import _load_generator, make_sampler

    ck = _tiny_flat_perrotor(tmp_path)
    params = _load_params(ck, interp=_interp_cfg(), gen_batch=3)
    gb = _load_generator(params, "cpu")
    sampler = make_sampler(gb, params, np.random.default_rng(0), "cpu")
    batch = sampler()

    assert batch.z.shape == (3, 4, int(_PERROTOR_DELTAS.shape[-1]))
    z_base = 0.7 * gb.z_map["dregon"] + 0.3 * gb.z_map["michaels"]
    torch.testing.assert_close(batch.z[0], z_base.unsqueeze(0) + _PERROTOR_DELTAS)
    # every clip of the batch shares the batch's code
    torch.testing.assert_close(batch.z[0], batch.z[2])


def test_perrotor_noise_off_keeps_the_checkpoint_deltas(tmp_path, fake_geometry):
    from data_processing.generated_noise import _load_generator, make_sampler

    ck = _tiny_flat_perrotor(tmp_path)
    params = _load_params(ck, interp=_interp_cfg(perrotor_noise=0.0))
    gb = _load_generator(params, "cpu")
    sampler = make_sampler(gb, params, np.random.default_rng(0), "cpu")
    for _ in range(3):
        torch.testing.assert_close(sampler().deltas, _PERROTOR_DELTAS)


def test_perrotor_noise_redraws_deltas_at_the_configured_scale(tmp_path, fake_geometry):
    """Fresh draw per batch, std = perrotor_noise x RMS of the delta norms."""
    from data_processing.generated_noise import _load_generator, make_sampler

    noise = 0.5
    ck = _tiny_flat_perrotor(tmp_path)
    params = _load_params(ck, interp=_interp_cfg(perrotor_noise=noise))
    gb = _load_generator(params, "cpu")
    sampler = make_sampler(gb, params, np.random.default_rng(0), "cpu")
    draws = np.stack([sampler().deltas.numpy() for _ in range(400)])

    assert not np.allclose(draws[0], draws[1])  # consecutive batches differ
    assert np.allclose(draws.mean(axis=0), _PERROTOR_DELTAS.numpy(), atol=0.05)
    unit = float(_PERROTOR_DELTAS.norm(dim=-1).pow(2).mean().sqrt())
    assert draws.std(axis=0).mean() == pytest.approx(noise * unit, rel=0.1)


def test_rotor_jitter_std_perturbs_positions_per_batch(tmp_path, fake_geometry):
    """Independent N(0, sigma) per rotor per coordinate, on top of the alpha
    interpolation, redrawn every batch."""
    from data_processing.generated_noise import _load_generator, make_sampler

    _, rotors = fake_geometry
    base = 0.7 * rotors["dregon"] + 0.3 * rotors["michaels"]
    ck = _tiny_flat_perrotor(tmp_path)
    gb = _load_generator(_load_params(ck), "cpu")

    off = _load_params(ck, interp=_interp_cfg(rotor_jitter_std=0.0))
    sampler = make_sampler(gb, off, np.random.default_rng(0), "cpu")
    for _ in range(3):
        assert np.allclose(sampler().rotor_pos, base)

    on = _load_params(ck, interp=_interp_cfg(rotor_jitter_std=0.2))
    sampler = make_sampler(gb, on, np.random.default_rng(0), "cpu")
    draws = np.stack([sampler().rotor_pos for _ in range(400)])
    assert not np.allclose(draws[0], draws[1])
    assert np.allclose(draws.mean(axis=0), base, atol=0.05)
    assert draws.std(axis=0).mean() == pytest.approx(0.2, rel=0.1)


def test_full_flight_excitation_dispatch(tmp_path, fake_geometry):
    """`rps.kind: full_flight` must reach the generated source's sampler (the
    AGENTS.md 'synthetic_intermittent only' note is stale): the batch spans the
    zero-RPS ground regime, unlike the cruise-only intermittent kind."""
    from data_processing.generated_noise import _load_generator, make_sampler

    ck = _tiny_flat_perrotor(tmp_path)
    gb = _load_generator(_load_params(ck), "cpu")

    def _batch(kind: str) -> np.ndarray:
        params = _load_params(ck, rps_kind=kind, gen_batch=64, chunk_s=0.25, sample_rate=1000)
        return make_sampler(gb, params, np.random.default_rng(0), "cpu")().rps

    full = _batch("full_flight")
    assert full.shape == (64, 4, 250)
    assert full.min() == pytest.approx(0.0, abs=1e-6)  # ground: rotors off
    assert full.max() > 40.0  # ... and cruise
    assert _batch("synthetic_intermittent").min() > 20.0  # cruise-only floor


def test_from_config_accepts_full_flight_rps_kind():
    from data_processing.generated_noise import GeneratedNoisePool

    with pytest.raises(ValueError, match="synthetic_intermittent"):
        GeneratedNoisePool.from_config(
            {"checkpoint": "x.pt", "rps": {"kind": "nope"}}, duration_s=1.0, sample_rate=16000
        )


# --- engine dispatch (no GPU, no spawn) --------------------------------


class _DummyPool:
    def __init__(self, tag, records=None):
        self.tag = tag
        self.records = records or []

    def sample_timeframe(self, rng, duration_s):
        return self.tag


@pytest.mark.slow
def test_build_noise_stream_dispatches_engines(monkeypatch):
    # A generated/static engine spec becomes a random_stream.map(render)
    # sub-stream; an all-engine list is a single-stream pipeline.
    import data_processing.online_mixing as om

    monkeypatch.setattr(om, "_build_engine", lambda cfg, **kw: _DummyPool(str(cfg.get("kind"))))
    stream, ceiling = build_noise_stream(
        [{"kind": "generated", "weight": 3.0}], sample_rate=16000, window_s=1.0, seed=0
    )
    import itertools

    frames = list(itertools.islice(iter(stream), 3))
    assert all(f == "generated" for f in frames)
    assert ceiling == 8
