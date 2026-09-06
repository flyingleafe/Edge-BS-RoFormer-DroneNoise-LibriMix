"""The seams of ``scripts/train_slot_v2.py``: the flags, the config, the stream.

The trainer itself is a loop over a model two other branches own, so what is
tested here is what THIS script decides:

* the corner. With no v2 flag the constructor call must be the C1 one, because
  every v2 ablation is read against `train_slot_real.py --mono` and a silent
  difference in the grid or the emission would make the comparison meaningless;
* the acceptance window. § 3.1 and § 3.2 exist to put the zero and below-grid
  frames INTO the loss, so ``--off-state`` must open the sampler, and a model
  without an OFF state must keep the C1 window (it cannot express a zero, and
  would learn to put it at the grid edge);
* the config round trip. A dump or a probe rebuilds the arm from
  ``config.json`` alone, so what the trainer writes must build the same model;
* the mono stream. Every table this feeds is per mono frame, so a crop is
  ``(1, N)`` and a batch is ``(B, 1, N)`` with ``(B, 4, T)`` labels, zeros
  included.

The stream test drives a TINY LOCAL POLICY (a static comb plus a silence arm)
and not `conf/online_mix/salv2_comb.yaml`: the salv2 policies pull LibriSpeech
and the real frame datasets, which is a network dependency a unit test must not
have. What the salv2 policies themselves must carry — eight microphones, both
excitations and the silence arm the zero frames come from — is asserted
separately, off the YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

train_slot_v2 = pytest.importorskip("train_slot_v2")

from experiments import slot_v2 as sv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
#: a 90-point grid and 8 orders: enough for a peel, small enough for a unit test
TINY_MODEL = dict(n_grid=90, k_max=8, mask_k_max=64, use_checkpoint=False, n_iter=0)


def _args(argv: list[str]):
    return train_slot_v2.build_parser().parse_args(argv)


def _tiny_policy(tmp_path: Path, duration_s: float = 2.0) -> str:
    """A speechless four-microphone policy: one static comb and one silence arm.

    The silence arm carries an exactly zero rotor-speed label, which is what the
    full-range sampler must stop throwing away.
    """
    cfg = {
        "sample_rate": 16000,
        "duration_s": duration_s,
        "base_seed": 5,
        "sources": {
            "noise": [
                {
                    "kind": "static_comb",
                    "weight": 1.0,
                    "n_harmonics": 20,
                    "n_mics": 4,
                    "n_rotors": 4,
                    "rps": {"kind": "full_flight", "aggressiveness": 1.0, "flight_reuse": 4},
                },
                {"kind": "silence", "weight": 1.0, "n_channels": 4},
            ]
        },
        "policy": {"stages": [{"until": None, "source_prob": 0.0}]},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "tiny_policy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


# ─── The flags ────────────────────────────────────────────────────────────────


def test_no_v2_flag_is_the_c1_corner():
    """The default call is `train_slot_real.py --mono`, keyword for keyword."""
    kw = train_slot_v2.model_kwargs(_args([]))
    assert kw["r_lo"] == 30.0 and kw["r_hi"] == 100.0 and kw["n_grid"] == 700
    assert kw["k_max"] == 40 and kw["floor_hz"] == 60.0 and kw["n_iter"] == 0
    assert kw["head_mode"] == "classical" and kw["emission"] == "partial"
    # mono drops `channels`: a mono input has one channel and it IS the mean
    assert kw["parts"] == ["reliability", "empty_tooth", "floor_mix"]
    assert not any(k in kw for k in ("off_state", "learned_transition", "rate_prior", "v2_parts"))


def test_every_v2_group_reaches_the_constructor():
    a = _args(
        [
            "--off-state",
            "--learned-transition",
            "--rate-prior",
            "--grid-lo",
            "10",
            "--n-grid",
            "900",
            "--emission",
            "v2",
            "--v2-parts",
            "gap,cross_order",
        ]
    )
    kw = train_slot_v2.model_kwargs(a)
    assert kw["off_state"] is True and kw["learned_transition"] is True
    assert kw["rate_prior"] is True
    assert kw["r_lo"] == 10.0 and kw["n_grid"] == 900
    assert kw["emission"] == "v2" and kw["v2_parts"] == ["gap", "cross_order"]


def test_model_kwarg_passes_any_keyword_through():
    """The escape hatch for a group the other branches named differently."""
    kw = train_slot_v2.model_kwargs(_args(["--model-kwarg", "slew=30.0", "--model-kwarg", "x=1"]))
    assert kw["slew"] == 30.0 and kw["x"] == 1
    with pytest.raises(SystemExit):
        train_slot_v2.parse_kv(["no-equals-sign"], "model-kwarg")


def test_check_signature_names_the_missing_keyword():
    """A group that has not landed must fail readably, before anything is built."""
    train_slot_v2.check_signature({"n_grid": 90, "rate_prior": True})  # both exist
    with pytest.raises(SystemExit) as e:
        train_slot_v2.check_signature({"a_group_that_never_landed": True})
    assert "--model-kwarg" in str(e.value)


def test_the_off_state_opens_the_acceptance_window():
    """Zero and below-grid frames reach the loss only with an OFF state."""
    base = _args([])
    assert train_slot_v2.acceptance(base, train_slot_v2.model_kwargs(base)) == (31.0, 99.0)
    on = _args(["--off-state"])
    assert train_slot_v2.acceptance(on, train_slot_v2.model_kwargs(on)) == sv.FULL_RANGE
    forced = _args(["--off-state", "--grid-range"])
    assert train_slot_v2.acceptance(forced, train_slot_v2.model_kwargs(forced)) == (31.0, 99.0)
    opened = _args(["--full-range", "--grid-lo", "10", "--n-grid", "900"])
    assert train_slot_v2.acceptance(opened, train_slot_v2.model_kwargs(opened)) == sv.FULL_RANGE


def test_smoke_shrinks_every_cost():
    a = _args(["--smoke"])
    train_slot_v2.apply_smoke(a)
    assert a.steps == 2 and a.crop_s == 1.0 and a.batch == 1
    assert a.n_grid == 60 and a.select_n == 2 and a.val_clips == 1
    assert a.frozen_every == 0  # the frozen split is the record, not the smoke


# ─── The config round trip ────────────────────────────────────────────────────


def test_config_json_rebuilds_the_same_model(tmp_path: Path):
    """What the trainer writes is what a dump and a probe rebuild."""
    a = _args(["--rate-prior", "--grid-lo", "10", "--n-grid", "90", "--k-max", "8"])
    kw = {**train_slot_v2.model_kwargs(a), "mask_k_max": 64, "use_checkpoint": False}
    net = train_slot_v2.build_model(kw, "cpu")
    sv.save_config(tmp_path, {"script": "test", "name": "t", "model": kw})

    cfg = sv.load_config(tmp_path)  # by directory ...
    assert sv.load_config(tmp_path / "config.json") == cfg  # ... and by file
    assert json.loads((tmp_path / "config.json").read_text())["model"]["r_lo"] == 10.0

    again = sv.build_from_config(cfg)
    assert torch.equal(again.grid, net.grid)
    assert again.rate_prior is not None
    assert list(again.state_dict()) == list(net.state_dict())


def test_load_arm_reads_config_and_best_pt(tmp_path: Path):
    """`load_arm` is the one door a dump or a probe opens an arm through."""
    a = _args(["--rate-prior", "--n-grid", "90", "--k-max", "8"])
    kw = {**train_slot_v2.model_kwargs(a), "mask_k_max": 64, "use_checkpoint": False}
    net = train_slot_v2.build_model(kw, "cpu")
    with torch.no_grad():
        net.rate_prior.v.fill_(0.25)
    sv.save_config(tmp_path, {"script": "test", "name": "t", "model": kw})
    torch.save(train_slot_v2.state_dict_trainable(net), tmp_path / "best.pt")

    back = sv.load_arm(tmp_path)
    assert float(back.rate_prior.v[0]) == 0.25
    assert not back.training  # an arm comes back in eval mode


def test_load_config_without_a_file_is_a_readable_exit(tmp_path: Path):
    with pytest.raises(SystemExit):
        sv.load_config(tmp_path)


def test_warm_start_reports_both_directions(tmp_path: Path, capsys):
    """`--init` loads what matches and names what does not."""
    a = _args(["--n-grid", "90", "--k-max", "8"])
    kw = {**train_slot_v2.model_kwargs(a), "mask_k_max": 64, "use_checkpoint": False}
    net = train_slot_v2.build_model(kw, "cpu")
    state = train_slot_v2.state_dict_trainable(net)
    state["a_key_the_model_does_not_have"] = torch.zeros(1)
    torch.save(state, tmp_path / "best.pt")

    train_slot_v2.warm_start(net, str(tmp_path), "cpu")
    out = capsys.readouterr().out
    assert "a_key_the_model_does_not_have" in out and "missing none" in out


# ─── The data modes and the stream ────────────────────────────────────────────


def test_the_data_modes_name_real_policies():
    """Every mode of the CLI resolves to a policy YAML that is in the repo."""
    for mode in sv.DATA_MODES:
        for name in sv.mode_policies(mode):
            assert (REPO_ROOT / sv.policy_path(name)).exists()
    with pytest.raises(ValueError):
        sv.mode_policies("no_such_mode")
    assert sv.mode_policies("comb+real") == ["comb", "real"]


def test_the_salv2_policies_carry_the_zero_frames():
    """The synthetic modes must supply what § 3.1 trains on.

    A comb arm learns the OFF state only if its stream holds stopped rotors.
    `conf/online_mix/salv2_comb.yaml` says so in prose; this asserts it.
    """
    for name in ("comb", "stoch"):
        cfg = OmegaConf.load(REPO_ROOT / sv.policy_path(name))
        kinds = [str(s["kind"]) for s in cfg.sources.noise]
        assert "silence" in kinds  # the zero-labeled arm
        excitations = {str(s["rps"]["kind"]) for s in cfg.sources.noise if "rps" in s}
        assert excitations == {"full_flight", "synthetic_intermittent"}
        assert all(int(s["n_mics"]) == 8 for s in cfg.sources.noise if "n_mics" in s)


@pytest.mark.parametrize("mono", [True, False])
def test_the_stream_is_mono_and_keeps_the_zero_frames(tmp_path: Path, mono: bool):
    """``(1, N)`` audio and ``(4, T)`` labels, and the silence arm survives."""
    st = sv.windows(_tiny_policy(tmp_path), crop_s=1.0, seed=0, accept=sv.FULL_RANGE, mono=mono)
    crops = [next(st) for _ in range(6)]
    for a, g in crops:
        assert a.shape == ((1, 16000) if mono else (4, 16000))
        assert g.shape == (4, 32) and g.dtype == np.float32
    assert st.accept == 1.0  # full range rejects nothing
    assert any(float(g.max()) < 0.5 for _, g in crops)  # a stopped-rotor crop
    assert any(float(g.max()) > 10.0 for _, g in crops)  # ... and a turning one


def test_a_batch_is_b_one_n(tmp_path: Path):
    """`batch` stacks the mono crops the loss reads: ``(B, 1, N)`` and ``(B, 4, T)``."""
    streams = [sv.windows(_tiny_policy(tmp_path), crop_s=1.0, seed=1, accept=sv.FULL_RANGE)]
    au, gt = train_slot_v2.batch(streams, [0], 1, 2, "cpu")
    assert tuple(au.shape) == (2, 1, 16000)
    assert tuple(gt.shape) == (2, 4, 32)


def test_the_selection_set_splits_across_the_policies_of_the_mode(tmp_path: Path):
    """A two-policy mode selects on both, and never on the frozen split."""
    a = _tiny_policy(tmp_path)
    b = _tiny_policy(tmp_path / "second")
    both = sv.select_set(f"{a}+{b}", n=4, crop_s=1.0, accept=sv.FULL_RANGE, cache=False)
    assert len(both) == 4
    one = sv.select_set(a, n=2, crop_s=1.0, accept=sv.FULL_RANGE, cache=False)
    assert len(one) == 2
    # The set is FIXED: the same call is the same audio.
    again = sv.select_set(a, n=2, crop_s=1.0, accept=sv.FULL_RANGE, cache=False)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(one, again, strict=True))


def test_the_grid_window_rejects_the_zero_crops(tmp_path: Path):
    """The C1 sampler, for the arms that have no OFF state."""
    st = sv.windows(_tiny_policy(tmp_path), crop_s=1.0, seed=0, accept=(31.0, 99.0))
    for _ in range(3):
        _, g = next(st)
        assert float(g.min()) >= 31.0
    assert st.accept < 1.0  # something WAS rejected
