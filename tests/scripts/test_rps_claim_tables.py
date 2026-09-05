"""The matrix mapping and the CLI of ``scripts/rps_claim_tables.py``.

Two things can go wrong in a table generator that nothing else checks. The
mapping can name an experiment that no naming convention of the campaign
produces (a typo turns a real cell into a silently empty one), or it can name
one experiment twice (two cells then carry the same numbers). Both are checked
here against the regexes of the campaign doc.

The CLI itself is checked on a two-set, one-experiment dump in ``tmp_path``,
with no probe cache at all: it must write its five files and leave the probe
columns empty instead of failing.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import rps_claim_tables as rct

REPO_ROOT = Path(__file__).resolve().parents[2]

#: every naming convention of `docs/experiments/paper-regime-matrix.md`
NAME_PATTERNS = (
    r"real_r[1-4]_(sc|scv2|tm|gru|hppnet|hf0)",  # the four real rungs
    r"salv2_(scv2|tr|gru|hppnet|hf0)_(comb|stoch)_(nomix|mix)",  # S1 / S2
    r"(hppnet|hf0)_r2hb_(l4|nomix)",  # the salience ports on the R4 pool
    r"r2hb_(scv2|tm|gru)_nomix(_wu)?",  # the no-speech regressor twins
    r"(hb_scv2_mag_nogate|tm_r2hb_nogate|r2hb_gru_nogate)",  # the old R4 rows
    r"(r4hb_(scv2|gru)|tm_r4hb|hppnet_r4_l4|hf0_r4_l4_v2)",  # C1
    r"r6hb_scv2",  # C2
    r"r7hb_(scv2|tm|gru)",  # M
    r"hb_sal_(multif0|multif0_nsr|multif0_l4|bp|bp_l4|hppnet_orig|hf0_orig)",  # block S
)
NAME_RE = re.compile("|".join(f"(?:{p})" for p in NAME_PATTERNS) + "$")

SETS = ("real", "comb")
EXPERIMENT = "real_r1_scv2"  # one cell of the matrix, so a row must appear
N_SAMPLES, N_ROTORS, T = 4, 4, 6


def make_dump(root: Path) -> Path:
    """A tiny dump: two sets, one experiment, four samples of four rotors."""
    dump = root / "rps_dump"
    rng = np.random.default_rng(0)
    for s in SETS:
        d = dump / s
        d.mkdir(parents=True)
        gt = rng.uniform(40.0, 80.0, size=(N_SAMPLES, N_ROTORS, T)).astype(np.float32)
        n_t = np.full(N_SAMPLES, T, dtype=np.int64)
        np.savez(d / "_gt.npz", rps=gt, n_t=n_t)
        (d / "_meta.json").write_text(
            json.dumps(
                [{"recording_id": f"sample_{i:05d}", "channel": 0} for i in range(N_SAMPLES)]
            )
        )
        np.savez(
            d / f"{EXPERIMENT}.npz",
            pred=gt + 1.0,
            n_t=n_t,
            metric=np.full(N_SAMPLES, 1.0, dtype=np.float32),
        )
    return dump


# ─── The mapping ──────────────────────────────────────────────────────────────


def test_every_mapped_name_obeys_the_doc_naming() -> None:
    names = [e for _, _, e in rct.matrix_rows()]
    names += [e for levels in rct.BLOCK_S.values() for e in levels.values()]
    bad = sorted({n for n in names if not NAME_RE.match(n)})
    assert not bad, f"names outside the campaign's conventions: {bad}"


def test_no_experiment_is_mapped_to_two_matrix_cells() -> None:
    counts = Counter(e for _, _, e in rct.matrix_rows())
    assert [n for n, c in counts.items() if c > 1] == []


def test_the_speech_pairs_only_use_mapped_experiments() -> None:
    known = {e for _, _, e in rct.matrix_rows()}
    paired = {e for per_trunk in rct.SPEECH_PAIRS.values() for p in per_trunk.values() for e in p}
    assert paired <= known, f"unmapped: {sorted(paired - known)}"


def test_every_speech_pair_is_one_trained_without_and_one_with() -> None:
    for family, per_trunk in rct.SPEECH_PAIRS.items():
        for trunk, pair in per_trunk.items():
            assert len(set(pair)) == 2, f"{family}/{trunk} pairs an experiment with itself"


def test_the_trunk_keys_of_every_regime_are_known() -> None:
    for regime, cells in rct.MATRIX.items():
        assert set(cells) <= set(rct.TRUNKS), f"{regime} names an unknown trunk"


# ─── The CLI ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the CLI once on a tiny dump with no probe cache and no rig axis."""
    root = tmp_path_factory.mktemp("claim_tables")
    dump = make_dump(root)
    out = root / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "rps_claim_tables.py"),
            "--dump",
            str(dump),
            "--probe",
            str(root / "no_such_probe_cache"),
            "--out",
            str(out),
            "--no-rigs",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return out


@pytest.mark.parametrize(
    "name", ["ladder.csv", "blocks.csv", "speech_ab.csv", "stochastic.csv", "missing.txt"]
)
def test_the_cli_writes_every_output(cli_out: Path, name: str) -> None:
    assert (cli_out / name).exists()


def test_the_cli_writes_the_markdown(cli_out: Path) -> None:
    md = (cli_out / "claims.md").read_text()
    assert "## Claim 5: the speech A/B" in md
    assert EXPERIMENT in md


def test_a_dumped_cell_gets_its_part_mean_and_an_empty_probe(cli_out: Path) -> None:
    rows = {r["experiment"]: r for r in csv.DictReader((cli_out / "ladder.csv").open())}
    assert set(rows) == {e for _, _, e in rct.matrix_rows()}
    row = rows[EXPERIMENT]
    assert float(row["part_comb"]) == pytest.approx(1.0)
    assert float(row["part_real"]) == pytest.approx(1.0)
    assert row["part_stoch"] == ""  # the set is not in the dump
    assert row["slope_full"] == "" and row["cut10_mae"] == ""  # no probe cache


def test_the_missing_report_lists_the_cells_and_the_probes(cli_out: Path) -> None:
    text = (cli_out / "missing.txt").read_text()
    # the one dumped experiment is missing four of the six sets, never all six
    assert f"{EXPERIMENT} | comb_speech" not in text
    assert "R1 | tm | real_r1_tm | comb,stoch," in text
    assert f"{EXPERIMENT} | freq,cutoff" in text
