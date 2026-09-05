"""The claim tables of `docs/experiments/paper-regime-matrix.md`, from the artifacts.

The campaign states five claims. Each one is a table, and each table was typed
by hand from a different tool's output, so a rerun of one cell meant a manual
edit in three places. This CLI builds every one of them from the artifacts on
disk, in one pass, with nothing typed:

* `ladder.csv` -- one row for each (regime, trunk) cell of "The matrix": the
  seven real-split regime cells, the DREGON cruise microphone split, the six
  part means, and the two cue probes (claims 1-3).
* `speech_ab.csv` -- each (trained without speech / trained with speech) pair
  on the static comb, on the stochastic comb and on real data, read on clean
  input and on input that carries a talker (claim 5).
* `blocks.csv` -- the block-S adaptation ladder, levels L0 to L3.
* `stochastic.csv` -- the error classes and the fan of the stochastic-limit
  rows (claim 4).
* `missing.txt` -- every mapped cell with no dump, and every experiment with
  no probe.
* `claims.md` -- all of the above as markdown tables.

    python scripts/rps_claim_tables.py
    python scripts/rps_claim_tables.py --out results/paper_regime_matrix

Where the numbers come from. A regime cell is the frame-weighted PIT MAE of
`scripts/rps_regime_table.py`, whose functions this CLI imports. A part mean is
the mean of the dump's `metric` column over its samples, which is the value the
training run monitored. A clip-group score (the speech A/B on real data) is the
same mean over the samples of one group of clips. The probes are read from the
JSON cache of `scripts/rps_cue_probe.py`, and the cutoff probe is aggregated by
that script's own `summarize`. The error classes and the fan come from
`scripts/rps_error_profile.py`, which this CLI runs on the stochastic part.

A cell with no dump is empty in every table and is listed in `missing.txt`. The
CLI never fails on a missing cell, because the campaign fills the matrix over
several weeks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT / "src", SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rps_cue_probe as rcp  # noqa: E402
import rps_regime_table as rrt  # noqa: E402

# ─── The matrix (docs/experiments/paper-regime-matrix.md, "The matrix") ───────

TRUNKS = ("sc", "scv2", "tm", "gru", "hppnet", "hf0")

#: regime -> trunk -> experiment. The regime order is the order of the tables.
#: `S1`/`S2` name the transformer cell `salv2_tr_*`; the column stays `tm`.
MATRIX: dict[str, dict[str, str]] = {
    "R1": {t: f"real_r1_{t}" for t in TRUNKS},
    "R2": {t: f"real_r2_{t}" for t in TRUNKS},
    "R3": {t: f"real_r3_{t}" for t in TRUNKS},
    "R4": {
        "sc": "real_r4_sc",
        "scv2": "real_r4_scv2",
        "tm": "real_r4_tm",
        "gru": "real_r4_gru",
        "hppnet": "hppnet_r2hb_l4",
        "hf0": "hf0_r2hb_l4",
    },
    "R4 (warm-up)": {
        "scv2": "hb_scv2_mag_nogate",
        "tm": "tm_r2hb_nogate",
        "gru": "r2hb_gru_nogate",
    },
    "R4 nomix": {
        "scv2": "r2hb_scv2_nomix_wu",
        "tm": "r2hb_tm_nomix_wu",
        "gru": "r2hb_gru_nomix_wu",
        "hppnet": "hppnet_r2hb_nomix",
        "hf0": "hf0_r2hb_nomix",
    },
    "R4 nomix (no warm-up)": {"scv2": "r2hb_scv2_nomix"},
    "S1 nomix": {
        "scv2": "salv2_scv2_comb_nomix",
        "tm": "salv2_tr_comb_nomix",
        "gru": "salv2_gru_comb_nomix",
        "hppnet": "salv2_hppnet_comb_nomix",
        "hf0": "salv2_hf0_comb_nomix",
    },
    "S1 mix": {
        "scv2": "salv2_scv2_comb_mix",
        "tm": "salv2_tr_comb_mix",
        "gru": "salv2_gru_comb_mix",
        "hppnet": "salv2_hppnet_comb_mix",
        "hf0": "salv2_hf0_comb_mix",
    },
    "S2 nomix": {
        "scv2": "salv2_scv2_stoch_nomix",
        "tm": "salv2_tr_stoch_nomix",
        "gru": "salv2_gru_stoch_nomix",
        "hppnet": "salv2_hppnet_stoch_nomix",
        "hf0": "salv2_hf0_stoch_nomix",
    },
    "S2 mix": {
        "scv2": "salv2_scv2_stoch_mix",
        "tm": "salv2_tr_stoch_mix",
        "gru": "salv2_gru_stoch_mix",
        "hppnet": "salv2_hppnet_stoch_mix",
        "hf0": "salv2_hf0_stoch_mix",
    },
    "C1": {
        "scv2": "r4hb_scv2",
        "tm": "tm_r4hb",
        "gru": "r4hb_gru",
        "hppnet": "hppnet_r4_l4",
        "hf0": "hf0_r4_l4_v2",
    },
    "C2": {"scv2": "r6hb_scv2"},
    "M": {"scv2": "r7hb_scv2", "tm": "r7hb_tm", "gru": "r7hb_gru"},
}

#: block S: architecture -> level -> experiment ("Block S", the L3 rows are the ports)
BLOCK_S: dict[str, dict[str, str]] = {
    "LateDeep": {"L0": "hb_sal_multif0", "L1": "hb_sal_multif0_nsr", "L2": "hb_sal_multif0_l4"},
    "Basic Pitch": {"L0": "hb_sal_bp", "L2": "hb_sal_bp_l4"},
    "HarmoF0": {"L0": "hb_sal_hf0_orig", "L3": "hf0_r2hb_l4"},
    "HPPNet": {"L0": "hb_sal_hppnet_orig", "L3": "hppnet_r2hb_l4"},
}
BLOCK_S_LEVELS = ("L0", "L1", "L2", "L3")

#: the speech A/B pairs: family -> trunk -> (trained without speech, trained with speech).
#: On real data the pair is the old R4 row against its `_wu` no-speech twin, so
#: that only the training speech differs; `r2hb_scv2_nomix` pairs with the
#: warm-up-free `real_r4_scv2` instead, which makes that pair a schedule row.
SPEECH_PAIRS: dict[str, dict[str, tuple[str, str]]] = {
    "S1 static comb": {
        "scv2": ("salv2_scv2_comb_nomix", "salv2_scv2_comb_mix"),
        "tm": ("salv2_tr_comb_nomix", "salv2_tr_comb_mix"),
        "gru": ("salv2_gru_comb_nomix", "salv2_gru_comb_mix"),
        "hppnet": ("salv2_hppnet_comb_nomix", "salv2_hppnet_comb_mix"),
        "hf0": ("salv2_hf0_comb_nomix", "salv2_hf0_comb_mix"),
    },
    "S2 stochastic comb": {
        "scv2": ("salv2_scv2_stoch_nomix", "salv2_scv2_stoch_mix"),
        "tm": ("salv2_tr_stoch_nomix", "salv2_tr_stoch_mix"),
        "gru": ("salv2_gru_stoch_nomix", "salv2_gru_stoch_mix"),
        "hppnet": ("salv2_hppnet_stoch_nomix", "salv2_hppnet_stoch_mix"),
        "hf0": ("salv2_hf0_stoch_nomix", "salv2_hf0_stoch_mix"),
    },
    "R4 real (warm-up)": {
        "scv2": ("r2hb_scv2_nomix_wu", "hb_scv2_mag_nogate"),
        "tm": ("r2hb_tm_nomix_wu", "tm_r2hb_nogate"),
        "gru": ("r2hb_gru_nomix_wu", "r2hb_gru_nogate"),
        "hppnet": ("hppnet_r2hb_nomix", "hppnet_r2hb_l4"),
        "hf0": ("hf0_r2hb_nomix", "hf0_r2hb_l4"),
    },
    "R4 real (no warm-up)": {"scv2": ("r2hb_scv2_nomix", "real_r4_scv2")},
}

#: family -> (the clean part, the part that carries a talker)
SPEECH_PARTS = {
    "S1 static comb": ("comb", "comb_speech"),
    "S2 stochastic comb": ("stoch", "stoch_speech"),
}

#: the rows of `stochastic.csv`: the stochastic limit of claim 4
STOCH_REGIMES = ("S2 nomix", "S2 mix", "C1", "C2")

SETS = ("comb", "stoch", "comb_speech", "stoch_speech", "real", "real_nospeech")

#: the seven real-split cells of the doc's row format, as
#: (column, rig, regime, microphone group) into `rps_regime_table.cells`
REAL_CELLS: tuple[tuple[str, str, str, str], ...] = (
    ("zero_frames", "all", "zero-frames", "all"),
    ("below_30", "all", "below-30", "all"),
    ("dregon_ramp", "DREGON", "ramp:in-grid", "all"),
    ("fly124_ramp", "FLY124", "ramp:in-grid", "all"),
    ("dregon_cruise", "DREGON", "cruise:in-grid", "all"),
    ("fly124_cruise", "FLY124", "cruise:in-grid", "all"),
    ("all_frames", "all", "all", "all"),
    # the microphone split rungs 1-2 need
    ("dregon_cruise_ch0", "DREGON", "cruise:in-grid", "ch0"),
    ("dregon_cruise_ch1_7", "DREGON", "cruise:in-grid", "ch1-7"),
)
REAL_CELL_COLS = tuple(c for c, *_ in REAL_CELLS)

PART_COLS = tuple(f"part_{s}" for s in SETS)
PROBE_COLS = ("slope_full", "slope_local")
CUT_COLS = tuple(f"cut{k}_{f}" for k in rcp.K_CUTS for f in ("mae", "frac_true", "frac_half"))

#: the loudspeaker clips of the frozen real split: DREGON room 1 flights where a
#: source played into the room. The other clips are rotor noise only.
LOUDSPEAKER_MARKS = ("speech-low", "whitenoise-low")

FAN_BUCKET_LABELS = ("0-2", "2-5", "5-10", "10-20", "20+")


# ─── Reading the artifacts ────────────────────────────────────────────────────


def part_means(dump: Path) -> dict[str, dict[str, float]]:
    """``set -> experiment -> mean of the dump's monitored ``metric`` column."""
    out: dict[str, dict[str, float]] = {}
    for s in SETS:
        d = dump / s
        if not d.is_dir():
            continue
        out[s] = {
            f.stem: float(np.load(f)["metric"].mean())
            for f in sorted(d.glob("*.npz"))
            if not f.stem.startswith("_")
        }
    return out


def clip_groups(dump: Path, dataset: str) -> dict[str, np.ndarray] | None:
    """Boolean per-frame masks of the real dump, by the origin of the clip.

    ``loudspeaker`` holds the 14 DREGON room-1 clips where a source played into
    the room, ``clean`` the other 23, and the two sub-groups of ``clean`` say
    which rig they came from. A dump frame is one microphone of one clip, so a
    mask selects every microphone of the clips of its group.
    """
    meta_path = dump / "real" / "_meta.json"
    if not meta_path.exists():
        return None
    clips, n_clips = rrt.clip_index(json.loads(meta_path.read_text()))
    recordings = rrt.recording_by_clip(n_clips, dataset)
    if recordings is None:
        return None
    idx = np.asarray(clips)

    def mask_of(keep) -> np.ndarray:
        return np.isin(idx, [i for i, rid in recordings.items() if keep(rid)])

    loud = mask_of(lambda rid: any(m in rid for m in LOUDSPEAKER_MARKS))
    fly = mask_of(lambda rid: "FLY" in rid.upper())
    return {
        "loudspeaker": loud,
        "clean": ~loud,
        "clean_dregon": ~loud & ~fly,
        "clean_fly124": fly,
    }


def clip_group_means(
    dump: Path, groups: dict[str, np.ndarray] | None
) -> dict[str, dict[str, float]]:
    """``group -> experiment -> mean ``metric`` over the clips of that group."""
    if groups is None:
        return {}
    out: dict[str, dict[str, float]] = {g: {} for g in groups}
    for f in sorted((dump / "real").glob("*.npz")):
        if f.stem.startswith("_"):
            continue
        metric = np.load(f)["metric"]
        for g, mask in groups.items():
            if mask.shape[0] == metric.shape[0] and mask.any():
                out[g][f.stem] = float(metric[mask].mean())
    return out


def freq_probes(root: Path) -> dict[str, dict[str, float]]:
    """``experiment -> {slope_full, slope_local}`` from the frequency-probe cache."""
    out: dict[str, dict[str, float]] = {}
    d = root / "freq"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        if f.stem.startswith("summary"):
            continue
        try:
            js = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "slope_full" in js:
            out[f.stem] = {k: float(js[k]) for k in PROBE_COLS}
    return out


def cutoff_probes(root: Path, tag: str) -> dict[str, dict[str, float]]:
    """``experiment -> {cut<k>_<field>}``, aggregated by `rps_cue_probe.summarize`."""
    out: dict[str, dict[str, float]] = {}
    d = root / "cutoff"
    if not d.is_dir():
        return out
    for f in sorted(d.glob(f"*{tag}.json")):
        name = f.name[: -len(f"{tag}.json")]
        if not name or name.startswith("summary"):
            continue
        try:
            js = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "rows" not in js:
            continue
        agg = rcp.summarize(js["rows"], list(rcp.K_CUTS))
        out[name] = {
            f"cut{k}_{field}": float(agg[str(k)][field])
            for k in rcp.K_CUTS
            for field in ("mae", "frac_true", "frac_half")
        }
    return out


def regime_cells(
    dump: Path, experiments: list[str], dataset: str, no_rigs: bool
) -> dict[tuple[str, str, str, str], tuple[float, int]]:
    """The per-cell PIT MAE of `rps_regime_table` over the real split."""
    d = dump / "real"
    present = [e for e in experiments if (d / f"{e}.npz").exists()]
    if not present:
        return {}
    _, n_clips = rrt.clip_index(json.loads((d / "_meta.json").read_text()))
    rigs = None if no_rigs else rrt.rig_by_clip(n_clips, dataset)
    return rrt.cells(d, present, rigs)


def run_error_profile(dump: Path, out: Path, experiments: list[str]) -> Path | None:
    """Run `scripts/rps_error_profile.py` on the stochastic part. Returns its directory."""
    present = [e for e in experiments if (dump / "stoch" / f"{e}.npz").exists()]
    if not present:
        return None
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "rps_error_profile.py"),
        "--dump",
        str(dump),
        "--sets",
        "stoch",
        "--experiments",
        ",".join(present),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit("rps_error_profile.py failed")
    return out


def read_profile(out: Path | None) -> dict[str, dict[str, Any]]:
    """``experiment -> the claim-4 columns`` from a profile directory."""
    if out is None or not (out / "summary.csv").exists():
        return {}
    import pandas as pd

    summary = pd.read_csv(out / "summary.csv")
    classes = pd.read_csv(out / "classes.csv")
    fan = pd.read_csv(out / "fan.csv")
    rows: dict[str, dict[str, Any]] = {}
    for _, r in summary[summary["set"] == "stoch"].iterrows():
        exp = str(r["exp"])
        rows[exp] = {
            "mae": float(r["mean"]),
            "median": float(r["median"]),
            "p90": float(r["p90"]),
            "fan_true": float(r["fan_true"]),
            "fan_pred": float(r["fan_pred"]),
            "fan_slope": float(r["fan_slope"]),
        }
    shares = {("offset", "offset"), ("alias 5/4", "alias_5_4"), ("alias 2", "alias_2")}
    shares |= {("wander", "wander"), ("missed", "missed"), ("dup", "dup")}
    for exp, row in rows.items():
        sub = classes[(classes["exp"] == exp) & (classes["set"] == "stoch")]
        got = dict(zip(sub["cls"], sub["error_share"], strict=True))
        for cls, col in sorted(shares, key=lambda kv: kv[1]):
            row[col] = float(got.get(cls, 0.0))
        sel = cast("pd.DataFrame", fan[(fan["exp"] == exp) & (fan["set"] == "stoch")])
        buckets = sel.sort_values(by="spread_lo")
        for label, (_, b) in zip(FAN_BUCKET_LABELS, buckets.iterrows(), strict=False):
            row[f"bucket_{label}"] = float(b["fan_pred"])
            row[f"bucket_{label}_true"] = float(b["fan_true"])
            row[f"bucket_{label}_n"] = int(b["n_frames"])
    return rows


# ─── Building the tables ──────────────────────────────────────────────────────


def matrix_rows() -> list[tuple[str, str, str]]:
    """Every mapped ``(regime, trunk, experiment)`` of the matrix, in table order."""
    return [(reg, t, cells[t]) for reg, cells in MATRIX.items() for t in TRUNKS if t in cells]


def ladder_table(
    cells: dict[tuple[str, str, str, str], tuple[float, int]],
    parts: dict[str, dict[str, float]],
    freq: dict[str, dict[str, float]],
    cut: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for regime, trunk, exp in matrix_rows():
        row: dict[str, Any] = {"regime": regime, "trunk": trunk, "experiment": exp}
        for col, rig, name, mic in REAL_CELLS:
            hit = cells.get((exp, rig, name, mic))
            row[col] = None if hit is None else hit[0]
        for s in SETS:
            row[f"part_{s}"] = parts.get(s, {}).get(exp)
        row.update({c: freq.get(exp, {}).get(c) for c in PROBE_COLS})
        row.update({c: cut.get(exp, {}).get(c) for c in CUT_COLS})
        rows.append(row)
    return rows


def blocks_table(
    cells: dict[tuple[str, str, str, str], tuple[float, int]],
    parts: dict[str, dict[str, float]],
    freq: dict[str, dict[str, float]],
    cut: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for arch, levels in BLOCK_S.items():
        for level in BLOCK_S_LEVELS:
            exp = levels.get(level)
            if exp is None:
                continue
            row: dict[str, Any] = {"model": arch, "level": level, "experiment": exp}
            for col, rig, name, mic in REAL_CELLS:
                hit = cells.get((exp, rig, name, mic))
                row[col] = None if hit is None else hit[0]
            for s in SETS:
                row[f"part_{s}"] = parts.get(s, {}).get(exp)
            row.update({c: freq.get(exp, {}).get(c) for c in PROBE_COLS})
            row.update({c: cut.get(exp, {}).get(c) for c in CUT_COLS})
            rows.append(row)
    return rows


def speech_table(
    parts: dict[str, dict[str, float]], groups: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """One row for each member of a speech pair, with its clean and talker scores."""
    rows = []
    for family, per_trunk in SPEECH_PAIRS.items():
        clean_set, speech_set = SPEECH_PARTS.get(family, ("real_nospeech", "real:loudspeaker"))
        for trunk in TRUNKS:
            if trunk not in per_trunk:
                continue
            on_parts = family in SPEECH_PARTS
            for exp, trained in zip(per_trunk[trunk], (False, True), strict=True):
                if on_parts:
                    clean = parts.get(clean_set, {}).get(exp)
                    speech = parts.get(speech_set, {}).get(exp)
                else:
                    clean = parts.get("real_nospeech", {}).get(exp)
                    speech = groups.get("loudspeaker", {}).get(exp)
                rows.append(
                    {
                        "family": family,
                        "trunk": trunk,
                        "trained_with_speech": trained,
                        "experiment": exp,
                        "clean_set": clean_set,
                        "speech_set": speech_set,
                        "clean_eval": clean,
                        "speech_eval": speech,
                        "ratio": (None if not clean or speech is None else speech / clean),
                        # the two rigs of the 23 clean clips, for the doc's split.
                        # A synthetic family is not read on real data at all.
                        "clean_dregon": None
                        if on_parts
                        else groups.get("clean_dregon", {}).get(exp),
                        "clean_fly124": None
                        if on_parts
                        else groups.get("clean_fly124", {}).get(exp),
                    }
                )
    return rows


def stochastic_table(profile: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for regime, trunk, exp in matrix_rows():
        if regime not in STOCH_REGIMES:
            continue
        hit = profile.get(exp, {})
        row: dict[str, Any] = {"regime": regime, "trunk": trunk, "experiment": exp}
        for col in ("mae", "median", "p90"):
            row[col] = hit.get(col)
        for col in ("offset", "alias_5_4", "alias_2", "wander", "missed", "dup"):
            row[col] = hit.get(col)
        for col in ("fan_true", "fan_pred", "fan_slope"):
            row[col] = hit.get(col)
        for label in FAN_BUCKET_LABELS:
            row[f"bucket_{label}"] = hit.get(f"bucket_{label}")
        rows.append(row)
    return rows


def missing_report(
    dump: Path, freq: dict[str, dict[str, float]], cut: dict[str, dict[str, float]]
) -> str:
    """Every mapped cell with no dump, and every mapped experiment with no probe."""
    cells = [(r, t, e) for r, t, e in matrix_rows()]
    cells += [
        (f"block S {arch}", level, exp) for arch, ls in BLOCK_S.items() for level, exp in ls.items()
    ]
    seen: OrderedDict[str, list[str]] = OrderedDict()
    lines = ["# cells with no dump (regime | trunk/level | experiment | missing sets)"]
    n_dump = 0
    for regime, trunk, exp in cells:
        seen.setdefault(exp, [])
        gone = [s for s in SETS if not (dump / s / f"{exp}.npz").exists()]
        if gone:
            n_dump += 1
            lines.append(f"{regime} | {trunk} | {exp} | {','.join(gone)}")
    if n_dump == 0:
        lines.append("(none)")
    lines += ["", "# experiments with no cue probe (experiment | which probe)"]
    n_probe = 0
    for exp in seen:
        gone = [n for n, tbl in (("freq", freq), ("cutoff", cut)) if exp not in tbl]
        if gone:
            n_probe += 1
            lines.append(f"{exp} | {','.join(gone)}")
    if n_probe == 0:
        lines.append("(none)")
    lines += [
        "",
        f"# {n_dump} cells with a missing dump, {n_probe} experiments with a missing probe",
    ]
    return "\n".join(lines) + "\n"


# ─── Output ───────────────────────────────────────────────────────────────────


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    path.parent.mkdir(parents=True, exist_ok=True)
    body = [",".join(columns)]
    body += [",".join(cell(r.get(c)) for c in columns) for r in rows]
    path.write_text("\n".join(body) + "\n")


def fmt(v: Any, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def matrix_md(rows: list[dict[str, Any]], column: str, title: str, nd: int = 2) -> str:
    """One matrix table: regime rows, trunk columns, one quantity."""
    by = {(r["regime"], r["trunk"]): r.get(column) for r in rows}
    out = [
        f"### {title}",
        "",
        "| regime | " + " | ".join(TRUNKS) + " |",
        "|---|" + "---|" * len(TRUNKS),
    ]
    for regime in MATRIX:
        vals = [by.get((regime, t)) for t in TRUNKS]
        if all(v is None for v in vals):
            continue
        out.append(f"| {regime} | " + " | ".join(fmt(v, nd) for v in vals) + " |")
    return "\n".join(out) + "\n"


def rows_md(rows: list[dict[str, Any]], head: list[str], cols: list[tuple[str, str, int]]) -> str:
    out = ["| " + " | ".join(head + [c[0] for c in cols]) + " |"]
    out.append("|---|" + "---|" * (len(head) + len(cols) - 1))
    for r in rows:
        cells = [str(r.get(h, "")) for h in head] + [fmt(r.get(c[1]), c[2]) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def claims_md(
    ladder: list[dict[str, Any]],
    speech: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    stoch: list[dict[str, Any]],
    profile: dict[str, dict[str, Any]],
    missing: str,
) -> str:
    parts = [
        "# The claim tables of the paper regime matrix",
        "",
        f"Generated by `scripts/rps_claim_tables.py` on {date.today().isoformat()}. Every",
        "number comes from `results/rps_dump`, `results/rps_probe` and the error profile;",
        "nothing is typed by hand. An empty cell has no dump; see `missing.txt`.",
        "",
        "## Claims 1-3: the ladder",
        "",
        "PIT MAE in rev/s over the frozen real split, all 8 microphones, unless said",
        "otherwise. The S1/S2 transformer cell is `salv2_tr_*`.",
        "",
    ]
    for column, title, nd in (
        ("all_frames", "Real split, all frames", 2),
        ("dregon_cruise", "Real split, DREGON cruise (in-grid)", 2),
        ("fly124_cruise", "Real split, FLY124 cruise (in-grid, unseen drone)", 2),
        ("part_comb", "Static-comb part (mean monitored PIT MAE)", 2),
        ("part_stoch", "Stochastic part (mean monitored PIT MAE)", 2),
        ("slope_local", "Frequency probe, local slope (+-4 %)", 2),
    ):
        parts.append(matrix_md(ladder, column, title, nd))
    parts += [
        "### The rung-1 and rung-2 microphone split, DREGON cruise",
        "",
        rows_md(
            [
                r
                for r in ladder
                if r["regime"] in ("R1", "R2") and r.get("dregon_cruise") is not None
            ],
            ["regime", "trunk", "experiment"],
            [
                ("ch 0", "dregon_cruise_ch0", 2),
                ("ch 1-7", "dregon_cruise_ch1_7", 2),
                ("all", "dregon_cruise", 2),
            ],
        ),
        "## Claim 5: the speech A/B",
        "",
        "`clean_eval` is the score without a talker, `speech_eval` the score with one.",
        "On real data the clean score is the 23 noise-only clips (`real_nospeech`) and",
        "the talker score is the 14 loudspeaker clips of the same split.",
        "",
        rows_md(
            speech,
            ["family", "trunk", "experiment"],
            [
                ("trained with speech", "trained_with_speech", 0),
                ("clean (23)", "clean_eval", 2),
                ("with talker (14)", "speech_eval", 2),
                ("ratio", "ratio", 2),
                ("clean DREGON (8)", "clean_dregon", 2),
                ("clean FLY124 (15)", "clean_fly124", 2),
            ],
        ),
        "## Claim 4: the stochastic limit",
        "",
        "Error classes on the stochastic part: the share of the model's total error",
        "carried by each class of failed rotor track.",
        "",
        rows_md(
            stoch,
            ["regime", "trunk", "experiment"],
            [
                ("MAE", "mae", 2),
                ("median", "median", 2),
                ("p90", "p90", 2),
                ("offset", "offset", 2),
                ("alias 5/4", "alias_5_4", 2),
                ("alias 2", "alias_2", 2),
                ("wander", "wander", 2),
                ("missed", "missed", 2),
                ("dup", "dup", 2),
            ],
        ),
        "",
        "The fan on the cruise time-frames, in buckets of the true rotor spread.",
        "`slope` is 1 for a model that tracks four lines and 0 for a fixed fan.",
        "",
    ]
    truth = next((v for v in profile.values() if "bucket_0-2_true" in v), None)
    fan_rows = list(stoch)
    if truth is not None:
        fan_rows = [
            {
                "regime": "true spread",
                "trunk": "",
                "experiment": f"({int(sum(truth[f'bucket_{b}_n'] for b in FAN_BUCKET_LABELS))} frames)",
                **{f"bucket_{b}": truth[f"bucket_{b}_true"] for b in FAN_BUCKET_LABELS},
                "fan_slope": 1.0,
            }
        ] + fan_rows
    parts += [
        rows_md(
            fan_rows,
            ["regime", "trunk", "experiment"],
            [(b, f"bucket_{b}", 2) for b in FAN_BUCKET_LABELS] + [("slope", "fan_slope", 2)],
        ),
        "## Block S: the adaptation ladder of the multi-pitch baselines",
        "",
        rows_md(
            blocks,
            ["model", "level", "experiment"],
            [
                ("zero", "zero_frames", 2),
                ("below-30", "below_30", 2),
                ("DREGON ramp", "dregon_ramp", 2),
                ("FLY124 ramp", "fly124_ramp", 2),
                ("DREGON cruise", "dregon_cruise", 2),
                ("FLY124 cruise", "fly124_cruise", 2),
                ("all", "all_frames", 2),
                ("probe full", "slope_full", 2),
                ("probe local", "slope_local", 2),
            ],
        ),
        "## Missing cells",
        "",
        "```",
        missing.strip(),
        "```",
        "",
    ]
    return "\n".join(parts)


# ─── Driver ───────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dump", default="results/rps_dump", help="the dump root, one directory per set"
    )
    ap.add_argument("--probe", default="results/rps_probe", help="the cue-probe cache root")
    ap.add_argument("--out", default="results/paper_regime_matrix", help="where the tables go")
    ap.add_argument("--cutoff-tag", default=".fir-n16", help="the cutoff-probe cache variant")
    ap.add_argument("--rig-dataset", default=rrt.RIG_DATASET, help="source of the per-clip rig")
    ap.add_argument("--no-rigs", action="store_true", help="drop the rig axis and the clip groups")
    ap.add_argument("--no-profile", action="store_true", help="reuse the error profile on disk")
    a = ap.parse_args()

    dump, out = Path(a.dump), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    experiments = [e for _, _, e in matrix_rows()]
    block_experiments = [e for ls in BLOCK_S.values() for e in ls.values()]

    parts = part_means(dump)
    freq = freq_probes(Path(a.probe))
    cut = cutoff_probes(Path(a.probe), a.cutoff_tag)
    cells = regime_cells(dump, experiments + block_experiments, a.rig_dataset, a.no_rigs)
    groups = None if a.no_rigs else clip_groups(dump, a.rig_dataset)
    group_means = clip_group_means(dump, groups) if groups else {}

    profile_dir = out / "profile"
    stoch_experiments = [e for r, _, e in matrix_rows() if r in STOCH_REGIMES]
    if not a.no_profile:
        profile_dir = run_error_profile(dump, profile_dir, stoch_experiments) or profile_dir
    profile = read_profile(profile_dir)

    ladder = ladder_table(cells, parts, freq, cut)
    blocks = blocks_table(cells, parts, freq, cut)
    speech = speech_table(parts, group_means)
    stoch = stochastic_table(profile)
    missing = missing_report(dump, freq, cut)

    ladder_cols = [
        "regime",
        "trunk",
        "experiment",
        *REAL_CELL_COLS,
        *PART_COLS,
        *PROBE_COLS,
        *CUT_COLS,
    ]
    write_csv(out / "ladder.csv", ladder, ladder_cols)
    write_csv(out / "blocks.csv", blocks, ["model", "level", "experiment", *ladder_cols[3:]])
    write_csv(
        out / "speech_ab.csv",
        speech,
        [
            "family",
            "trunk",
            "trained_with_speech",
            "experiment",
            "clean_set",
            "speech_set",
            "clean_eval",
            "speech_eval",
            "ratio",
            "clean_dregon",
            "clean_fly124",
        ],
    )
    write_csv(
        out / "stochastic.csv",
        stoch,
        [
            "regime",
            "trunk",
            "experiment",
            "mae",
            "median",
            "p90",
            "offset",
            "alias_5_4",
            "alias_2",
            "wander",
            "missed",
            "dup",
            "fan_true",
            "fan_pred",
            "fan_slope",
            *[f"bucket_{b}" for b in FAN_BUCKET_LABELS],
        ],
    )
    (out / "missing.txt").write_text(missing)
    (out / "claims.md").write_text(claims_md(ladder, speech, blocks, stoch, profile, missing))
    print(
        f"wrote {out}/ladder.csv ({len(ladder)} cells), blocks.csv ({len(blocks)}), "
        f"speech_ab.csv ({len(speech)}), stochastic.csv ({len(stoch)}), missing.txt, claims.md"
    )


if __name__ == "__main__":
    main()
