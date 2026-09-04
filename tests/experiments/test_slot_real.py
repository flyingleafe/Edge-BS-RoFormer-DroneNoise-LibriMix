"""The MONO seam of the C1 slot-comb arm: one microphone, the baselines' protocol.

WHY THIS FILE EXISTS. `SlotCombNet` was trained and scored on eight microphones
power-averaged, while every neural model it is compared with reads ONE
microphone and is scored per mono frame. The comparison was therefore not a
comparison. These tests pin the three pieces that make it one: the mono crop,
the mono scoring of the frozen split, and the dump in the neural models' own
format.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.slot_real import HOP, SR, WindowStream, score_real_mono

REPO_ROOT = Path(__file__).resolve().parents[2]

# The C1 corner at a grid and harmonic count small enough for a unit test, the
# same settings `tests/models/test_comb_slots_partial.py` uses.
NET_KW: dict[str, Any] = dict(
    n_grid=180, k_max=16, floor_hz=60.0, use_checkpoint=False, n_iter=0, multichannel=True
)


class _Series:
    """The one attribute `WindowStream.__next__` reads off a Frame entry."""

    def __init__(self, data: np.ndarray):
        self.data = data


def _chunks(n_mic: int, n: int, n_sample: int = 32000, rps: float = 60.0):
    """``n`` fake policy chunks. Microphone ``m`` is the constant ``m + 1``.

    A constant per microphone is the point: it makes the drawn microphone
    readable off the crop, which is what the mono draw has to be tested on.
    """
    n_t = n_sample // HOP + 1
    for _ in range(n):
        aud = np.repeat(np.arange(1, n_mic + 1, dtype=np.float32)[:, None], n_sample, axis=1)
        yield {"mixture": _Series(aud), "rps": _Series(np.full((4, n_t), rps))}


def _stream(mono: bool, n_mic: int = 8, seed: int = 0, crop_s: float = 1.0) -> WindowStream:
    """A `WindowStream` over fake chunks: `__init__` would load an online-mix policy."""
    st = WindowStream.__new__(WindowStream)
    st.hop, st.sr = HOP, SR
    st.n_crop = int(round(crop_s * SR))
    st.t_crop = st.n_crop // st.hop + 1
    st.r_lo, st.r_hi = 31.0, 99.0
    st.mono = mono
    st.rng = np.random.default_rng(seed)
    st.seen = st.kept = 0
    st._it = _chunks(n_mic, 64)
    return st


def test_mono_stream_yields_one_varying_microphone():
    """``(1, n)`` crops, and not the same microphone every time."""
    st = _stream(mono=True)
    mics = set()
    for _ in range(12):
        a, g = next(st)
        assert a.shape == (1, st.n_crop), a.shape
        assert g.shape == (4, st.t_crop)
        mics.add(int(a[0, 0]) - 1)  # the constant identifies the microphone
    assert mics.issubset(set(range(8)))
    assert len(mics) > 1, mics


def test_eight_channel_stream_is_unchanged():
    """The default is still every microphone, so no measured number moves."""
    a, _ = next(_stream(mono=False))
    assert a.shape == (8, 16000)


def _clip(idx: int, rig: str, seed: int, n: int = 16000, rps: float = 60.0) -> dict:
    """One fake frozen clip: a four-rotor comb on eight microphones, plus labels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    mono = 0.05 * rng.standard_normal(n)
    for r in (rps, rps + 4.0, rps + 8.0, rps + 12.0):
        for k in range(1, 9):
            mono += (1.0 / k) * np.sin(2 * np.pi * k * 2.0 * r * t + rng.uniform(0, 2 * np.pi))
    gains = np.linspace(0.6, 1.4, 8)[:, None]
    audio = mono[None] * gains + 0.02 * rng.standard_normal((8, n))
    n_t = n // HOP + 1
    labels = np.stack([np.full(n_t, r) for r in (rps, rps + 4.0, rps + 8.0, rps + 12.0)])
    return {
        "clip": idx,
        "phase": "cruise",
        "rig": rig,
        "audio": audio.astype(np.float32),
        "rps": labels,
    }


def test_score_real_mono_reads_eight_frames_per_clip():
    """Two clips must give 16 rows — one per microphone — and the usual table."""
    from models.comb_slots import SlotCombNet

    net = SlotCombNet(**NET_KW).eval()
    clips = [_clip(0, "DREGON", seed=1), _clip(1, "FLY124", seed=2)]
    out = score_real_mono(net, clips, name="test", quiet=True)

    assert len(out["rows"]) == 16
    assert [r["mic"] for r in out["rows"]] == list(range(8)) * 2
    assert [r["clip"] for r in out["rows"]] == [0] * 8 + [1] * 8
    for r in out["rows"]:
        assert np.asarray(r["pred"]).shape[0] == 4
        assert np.isfinite(r["mae"])
    # The table aggregates the 296-frame protocol, so its counts are FRAMES.
    assert out["cruise"]["n"] == 16
    assert out["DREGON_cruise"]["n"] == 8 and out["FLY124_cruise"]["n"] == 8
    assert np.isfinite(out["all"]["mean"])
    # The per-clip number stays per clip: the mean over that clip's microphones.
    assert sorted(out["per_clip"]) == [0, 1]
    for c in (0, 1):
        mics = [r["mae"] for r in out["rows"] if r["clip"] == c]
        assert out["per_clip"][c] == pytest.approx(float(np.mean(mics)), abs=1e-3)


def test_score_windows_takes_a_mono_window():
    """Selection must work on ``(1, n)`` without a branch of its own."""
    from experiments.slot_real import score_windows
    from models.comb_slots import SlotCombNet

    net = SlotCombNet(**NET_KW).eval()
    clip = _clip(0, "DREGON", seed=3)
    wins = [(clip["audio"][0:1], clip["rps"].astype(np.float32))]
    assert np.isfinite(score_windows(net, wins))


def test_dump_writes_the_rps_dump_format(tmp_path):
    """``--part comb`` must write ``pred`` / ``n_t`` / ``metric`` as `rps_dump.py` does.

    The zero-parameter corner (``--parts none``) is used because it is the fast
    one, and the three frames are SYNTHESIZED at that size, never sliced out of
    the 256-frame part.
    """
    cmd = [
        sys.executable,
        "scripts/slot_dump.py",
        "--part",
        "comb",
        "--parts",
        "none",
        "--limit",
        "3",
        "--name",
        "test_corner",
        "--out",
        str(tmp_path),
        "--device",
        "cpu",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, timeout=900)

    z = np.load(tmp_path / "comb" / "test_corner.npz")
    assert sorted(z.files) == ["metric", "n_t", "pred"]
    assert z["pred"].shape[:2] == (3, 4) and z["pred"].dtype == np.float32
    assert z["n_t"].shape == (3,) and z["n_t"].dtype == np.int64
    assert z["metric"].shape == (3,) and z["metric"].dtype == np.float64
    assert np.isfinite(z["metric"]).all()
    assert (z["n_t"] == z["pred"].shape[-1]).all()  # equal-length frames, no padding
