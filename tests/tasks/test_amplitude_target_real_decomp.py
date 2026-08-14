"""The amplitude path, end to end, on a REAL Vold-Kalman decomposition.

Everything else in the amplitude-target suite runs on synthetic tensors, which
proves shapes and never proves that the published artifacts have the layout the
loader assumes. This test loads one real solve off disk
(``results/vk_decompose_v2/free-flight_nosource_room1``, the v2/v3 artifact
format: ``envelopes.npz`` + ``residual.npz`` + ``report.json``), joins it to the
DREGON geometry and the refined-label carrier exactly as the derivation does,
and runs the whole seam:

    DecompFrameDataset -> frame_collate -> codec.to_inputs -> amp_stats
                       -> codec.to_frame -> AmplitudeTargetLoss

It is skipped when the artifacts are not on this machine (they are gitignored
job outputs, and the DREGON geometry comes from the dload cache).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
RECORDING = "free-flight_nosource_room1"
K_MAX = 80
SR = 16000
STRIDE = 160
CHUNK = 32000  # 2 s — long enough for several emitter control frames


def _decomp_dir() -> Path:
    """Where the local v2/v3 decomposition artifacts are, if anywhere.

    ``results/`` is gitignored and per-checkout, so a worktree usually has none
    of its own; the parent checkout's copy is the one that exists.
    """
    candidates = [
        Path(p)
        for p in (
            os.environ.get("VK_DECOMPOSE_DIR"),
            REPO / "results" / "vk_decompose_v2",
            REPO.parent.parent / "results" / "vk_decompose_v2",
        )
        if p
    ]
    for c in candidates:
        if (c / RECORDING / "envelopes.npz").exists():
            return c
    pytest.skip(f"no local decomposition artifacts (looked in {[str(c) for c in candidates]})")


def _dregon_geometry() -> tuple[np.ndarray, np.ndarray]:
    """The corrected DREGON array geometry, from the local raw tree."""
    from data_processing.sources.dregon import get_geometry

    roots = list((Path.home() / ".cache/dload/materialized/DREGON").glob("*")) + [
        REPO / "data" / "DREGON"
    ]
    for root in roots:
        if (root / "micPos.txt").exists() or (root / "coordinates.mat").exists():
            return get_geometry(root)
    pytest.skip("no local DREGON tree for the array geometry")


def _record() -> dict:
    """One real decomposed chunk, in ``DecompFrameDataset``'s record layout."""
    from scipy.signal import resample_poly

    from data_processing.derivations import _decomp_to_audio_grid, dense_envelopes

    root = _decomp_dir() / RECORDING
    report = json.loads((root / "report.json").read_text())
    with np.load(root / "envelopes.npz") as env:
        a0, a1 = (int(v) for v in np.asarray(env["span_samples"]))
        env_sr, env_stride = int(env["sample_rate"]), int(env["stride"])
        t0 = float(env["t0_offset_s"])
        amp, mask = dense_envelopes(
            np.asarray(env["amp"]), np.asarray(env["valid"]), env["rotor"], env["k"], K_MAX
        )
    with np.load(root / "residual.npz") as res:
        residual = np.asarray(res["residual"], dtype=np.float64)

    # The v2/v3 solve runs at 32 kHz so its line cap reaches the 16 kHz Nyquist;
    # the join decimates, exactly as `derivations._decomp_frame` does.
    decim, rem = divmod(env_sr, SR)
    assert rem == 0 and env_stride == STRIDE * decim, "artifact grid is not a multiple of the spec"
    if decim > 1:
        residual = resample_poly(residual, 1, decim, axis=-1)
        a0, a1 = a0 // decim, a1 // decim

    # The carrier the solve demodulated against: the refined labels, already on
    # the protocol frame grid, put on the audio grid by the derivation's own
    # helper. `ft` is in the untrimmed recording's time base, the loader's is
    # trimmed by `t0_offset_s`.
    labels = np.load(REPO / "src/data_processing/refined_labels" / f"{RECORDING}.npz")
    carrier = _decomp_to_audio_grid(labels["r_refined"], labels["ft"] - t0, a1, SR)[:, a0:a1]

    n_env = amp.shape[-1]
    assert residual.shape[-1] == a1 - a0
    assert carrier.shape[-1] == a1 - a0
    # The two grids must describe the same span, which is the invariant that
    # makes an index cut of both stay aligned.
    assert abs(n_env * STRIDE - (a1 - a0)) < STRIDE

    mic_pos, rotor_pos = _dregon_geometry()
    n_t = min(residual.shape[-1], n_env * STRIDE)
    return {
        "recording_id": report["recording_id"],
        "drone": "dregon",
        "sample_rate": SR,
        "rps": np.ascontiguousarray(carrier[:, :n_t], dtype=np.float32),
        "residual": np.ascontiguousarray(residual[:, :n_t], dtype=np.float32),
        "amp": np.ascontiguousarray(amp[:, :, :, : n_t // STRIDE]),
        "amp_valid": np.ascontiguousarray(mask[:, :, : n_t // STRIDE]),
        "mic_pos": np.asarray(mic_pos, dtype=np.float32),
        "rotor_pos": np.asarray(rotor_pos, dtype=np.float32),
        "span": (0, n_t),
        "k_hi": int(report["k_hi"]),
    }


@pytest.fixture(scope="module")
def real_record() -> dict:
    return _record()


def test_the_published_artifacts_have_the_layout_the_loader_assumes(real_record):
    rec = real_record
    assert rec["amp"].shape[:3] == (8, 4, K_MAX)  # (mic, rotor, k)
    assert rec["residual"].shape[0] == 8
    assert rec["rps"].shape[0] == 4
    # Above the recording's own ceiling nothing is solved; below it, something is.
    k_hi = rec["k_hi"]
    assert not rec["amp_valid"][:, k_hi:].any()
    assert rec["amp_valid"][:, :k_hi].any()
    # A flight recording's carrier is in the cruise band, not at idle.
    assert 40.0 < float(rec["rps"].mean()) < 120.0
    # The envelopes are real amplitudes, not a floor.
    solved = rec["amp"][:, :, :k_hi]
    assert float(solved.max()) > 1e-4


def _model(n_mics: int = 8):
    from models.registry import build_noise_gen_model

    params = {
        "n_harmonics": 100,
        "cond_dim": 16,
        "drone_names": ["dregon", "michaels"],
        "amp_calibration": True,
        "noise_floor_bands": 60,
        "n_mics": n_mics,
        "mic_eq_knots": 16,
    }
    return build_noise_gen_model("positional_harmonic_gen", **params)


def test_codec_round_trip_and_loss_on_the_real_decomposition(real_record):
    """Load, encode, predict, decode, score — on real targets."""
    from data_processing.collate import frame_collate
    from data_processing.frame_datasets import DecompFrameDataset
    from framespec import check_subsumes, spec_of
    from losses import AmplitudeTargetLoss
    from tasks.codecs import NoiseGenerationCodec
    from tasks.task import TASK_FACTORIES

    ds = DecompFrameDataset([real_record], chunk_size=CHUNK, n_samples=2, min_motor_rps=30.0)
    batch = frame_collate([ds[0], ds[1]])
    assert np.asarray(batch["amp"].data).shape == (2, 8, 4, K_MAX, CHUNK // STRIDE)

    codec = NoiseGenerationCodec(conditioned=True, amplitude=True)
    task = TASK_FACTORIES["noise_generation"](conditioned=True, amplitude=True)
    assert check_subsumes(spec_of(batch), task.input_spec) == []

    inputs = codec.to_inputs(batch)
    assert inputs["drone_names"] == ["dregon", "dregon"]
    assert inputs["rel_pos"].shape == (2, 8, 4, 3)

    model = _model()
    model.eval()
    with torch.no_grad():
        pred = codec.to_frame(codec.call_model(model, inputs), batch)
    assert set(pred) == {"amp_pred", "noise_psd", "mic_eq"}
    assert check_subsumes(spec_of(pred), task.output_spec) == []
    amp_pred = torch.as_tensor(np.asarray(pred["amp_pred"].data))
    assert amp_pred.shape[:4] == (2, 8, 4, 100)
    assert torch.isfinite(amp_pred).all()

    loss = AmplitudeTargetLoss()
    with torch.no_grad():
        value = loss(pred, batch)
    assert torch.isfinite(value) and float(value) > 0.0


def test_the_objective_has_a_gradient_on_the_real_targets(real_record):
    """The EQ and the emitter must both receive one, or the head is inert."""
    from data_processing.collate import frame_collate
    from data_processing.frame_datasets import DecompFrameDataset
    from losses import AmplitudeTargetLoss
    from tasks.codecs import NoiseGenerationCodec

    ds = DecompFrameDataset([real_record], chunk_size=CHUNK, n_samples=1, min_motor_rps=30.0)
    batch = frame_collate([ds[0]])
    codec = NoiseGenerationCodec(conditioned=True, amplitude=True)
    model = _model()
    pred = codec.to_frame(codec.call_model(model, codec.to_inputs(batch)), batch)
    AmplitudeTargetLoss()(pred, batch).backward()

    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith("mic_eq.log_eq.dregon") for n in grads)
    assert any(n.startswith("generator.emitter") for n in grads)
    for name, g in grads.items():
        assert torch.isfinite(g).all(), name
    # Michael's curve is untouched by a DREGON-only batch — rig routing is real.
    assert "mic_eq.log_eq.michaels" not in grads


def test_a_two_rig_batch_routes_each_sample_to_its_own_propagation_head(real_record):
    """The combined arm's core claim, exercised on the real record."""
    from data_processing.collate import frame_collate
    from data_processing.frame_datasets import DecompFrameDataset
    from losses import AmplitudeTargetLoss
    from tasks.codecs import NoiseGenerationCodec

    # The second rig stands in for Michael's: same envelopes, a different rig id.
    # What is under test is the ROUTING, so the identical payload is a feature —
    # any difference in the two samples' predictions can only come from the head.
    as_michaels = {**real_record, "drone": "michaels", "recording_id": "FLY124-stand-in"}
    frames = [
        DecompFrameDataset([rec], chunk_size=CHUNK, n_samples=1, min_motor_rps=30.0)[0]
        for rec in (real_record, as_michaels)
    ]
    assert [str(f["meta"]["drone"]) for f in frames] == ["dregon", "michaels"]
    batch = frame_collate(frames)

    codec = NoiseGenerationCodec(conditioned=True, amplitude=True)
    model = _model()
    pred = codec.to_frame(codec.call_model(model, codec.to_inputs(batch)), batch)
    AmplitudeTargetLoss()(pred, batch).backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert "mic_eq.log_eq.dregon" in grads
    assert "mic_eq.log_eq.michaels" in grads
