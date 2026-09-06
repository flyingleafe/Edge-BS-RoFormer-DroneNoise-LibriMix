"""``scripts/slot_dump.py --v2`` must write exactly what ``rps_dump.py`` writes.

WHY THIS IS THE TEST. The paper's tables come from `scripts/rps_claim_tables.py`
and `scripts/rps_regime_table.py`, which read ``results/rps_dump/<part>/
<model>.npz`` and nothing else. A v2 arm joins those tables only if its dump is
the same three arrays, the same dtypes, the same NaN padding and the same
per-sample PIT MAE. Anything else is a second layout to reconcile, which is the
failure this file exists to prevent.

The checks are therefore against an INDEPENDENT computation of each field, plus
a real read by `rps_regime_table.cells` — not against another call of the same
writer, which would prove nothing. Two samples of the synthetic ``comb`` part
are enough: the format does not depend on the count, and a synthetic part is
built on demand, so the test needs no network.

The cue probe's ``slot:`` source is checked here too, because it needs the same
arm directory and the same two frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

rps_cue_probe = pytest.importorskip("rps_cue_probe")
slot_dump = pytest.importorskip("slot_dump")

from experiments import rps_bench as rb  # noqa: E402
from experiments import slot_v2 as sv  # noqa: E402
from metrics._common import get_array  # noqa: E402

N = 2
#: the smallest model that still decodes: a 90-point grid, 8 harmonic orders
TINY = dict(
    sr=16000,
    n_fft=4096,
    hop_length=512,
    r_lo=30.0,
    r_hi=100.0,
    n_grid=60,
    k_max=6,
    mask_k_max=48,
    f_max=7500.0,
    head_mode="classical",
    floor_hz=60.0,
    n_rot=4,
    n_iter=0,
    use_checkpoint=False,
    emission="classical",
    rate_prior=True,
)


@pytest.fixture(scope="module")
def arm(tmp_path_factory) -> Path:
    """An arm directory as the trainer writes it: ``config.json`` + ``best.pt``."""
    d = tmp_path_factory.mktemp("arms") / "tinyarm"
    net = sv.build_from_config({"model": dict(TINY)})
    sv.save_config(d, {"script": "test", "name": "tinyarm", "model": dict(TINY)})
    torch.save({k: v.detach() for k, v in net.named_parameters() if v.requires_grad}, d / "best.pt")
    return d


@pytest.fixture(scope="module")
def dump(arm: Path, tmp_path_factory) -> Path:
    """``results/rps_dump``-shaped output of the CLI on two ``comb`` frames."""
    root = tmp_path_factory.mktemp("rps_dump")
    code = slot_dump.main(
        [
            "--v2",
            str(arm),
            "--part",
            "comb",
            "--limit",
            str(N),
            "--out",
            str(root),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    return root / "comb"


def test_the_arm_directory_names_the_column(dump: Path):
    """``--name`` defaults to the arm's directory, which is the tables' column."""
    assert (dump / "tinyarm.npz").exists()


def test_the_three_arrays_are_the_rps_dump_ones(dump: Path):
    z = np.load(dump / "tinyarm.npz")
    assert sorted(z.files) == ["metric", "n_t", "pred"]
    assert z["pred"].dtype == np.float32 and z["pred"].ndim == 3
    assert z["pred"].shape[0] == N and z["pred"].shape[1] == 4
    assert z["n_t"].dtype == np.int64 and z["n_t"].shape == (N,)
    assert z["metric"].dtype == np.float64 and z["metric"].shape == (N,)
    # NaN padding, and only past each sample's own length
    for i in range(N):
        assert np.isfinite(z["pred"][i, :, : int(z["n_t"][i])]).all()


def test_the_labels_and_the_metadata_are_written_once(dump: Path):
    """``_gt.npz`` / ``_meta.json`` are the SET's, so the first writer owns them.

    The second arm on a set must leave them alone, or two arms would be scored
    against two label files. The writer is called again here with a DIFFERENT
    frame list: nothing may change.
    """
    from rps_dump import write_set_gt

    z = np.load(dump / "_gt.npz")
    assert sorted(z.files) == ["n_t", "rps"]
    assert z["rps"].dtype == np.float32 and z["rps"].shape[:2] == (N, 4)
    meta = json.loads((dump / "_meta.json").read_text())
    assert len(meta) == N and "channel" in meta[0]

    before = (dump / "_gt.npz").read_bytes(), (dump / "_meta.json").read_bytes()
    write_set_gt(dump, rb.part("comb", n=N)[:1])
    assert (dump / "_gt.npz").read_bytes() == before[0]
    assert (dump / "_meta.json").read_bytes() == before[1]


def test_metric_is_the_per_sample_pit_mae_of_the_stored_prediction(dump: Path):
    """The column the tables average must be recomputable from the dump alone."""
    z = np.load(dump / "tinyarm.npz")
    frames = rb.part("comb", n=N)
    for i, f in enumerate(frames):
        pred = z["pred"][i, :, : int(z["n_t"][i])].astype(np.float64)
        gt = np.asarray(get_array(f, "rps"), dtype=np.float64)
        assert abs(z["metric"][i] - rb.pit_mae(pred, gt)) < 1e-6


def test_the_regime_table_reads_the_dump_unchanged(dump: Path):
    """The real consumer, run over the real files."""
    rrt = pytest.importorskip("rps_regime_table")
    cells = rrt.cells(dump, ["tinyarm"], None)
    assert cells, "the regime table found no cell"
    assert all(np.isfinite(mae) and n > 0 for mae, n in cells.values())


# ─── The cue probe's slot: source ─────────────────────────────────────────────


def test_probe_name_sanitizes_a_slot_spec(arm: Path):
    assert rps_cue_probe.probe_name(f"slot:{arm}") == "slot_tinyarm"
    assert rps_cue_probe.probe_name("r4hb_scv2") == "r4hb_scv2"  # a zoo name is untouched


def test_the_probe_reads_a_slot_arm_like_a_regressor(arm: Path):
    """`load_model` + `speeds` must give ``(4, T)`` rev/s, the regressor shape.

    This is the whole contract the frequency and cutoff probes need: a Frame in,
    four rotor tracks out. The probes deform the audio and carry no label, so
    the arm must read the audio alone.
    """
    import tdseries as td

    from data_processing.frames import audio_series

    fm = rps_cue_probe.load_model(f"slot:{arm}", device="cpu")
    f = rb.part("comb", n=N)[0]
    x = np.asarray(f["mixture"].data, dtype=np.float32).ravel()[: 16000 * 2]
    bare = td.Frame({"mixture": audio_series(x[None], 16000)})  # no labels, as the probe builds
    out = rps_cue_probe.speeds(fm, bare, rps_cue_probe.rate_grid())
    assert out.shape[0] == 4 and out.shape[1] > 1
    assert np.isfinite(out).all()
