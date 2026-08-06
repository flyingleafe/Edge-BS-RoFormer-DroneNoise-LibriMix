#!/usr/bin/env python3
"""Generic pivot-table CLI over tidy CSV/JSON result files.

Reads one or more CSV or JSON files (``--in`` accepts globs), builds one
tidy record table, then pivots: ``--rows`` fields down the side, an optional
``--cols`` field across the top, ``--metrics`` aggregated into the cells
(nan-mean by default, ``--agg count`` for composition tables). With
``--cols`` set, one table is emitted per metric; without it, the metrics are
the columns. ``--baseline <row>`` appends delta columns against that row
value, matched within each column group.

JSON files may be a list of records, or nested dicts unwrapped with
``--json-root`` (dotted path to the record mapping; its keys become a
column) and ``--unnest field=column`` (explode a dict-of-dicts field into
one record per key). Leaf dicts flatten to dotted column names.

Primary invocations of the scripts this replaces:

* ``f1_tables.py`` (per-SNR floor table; anchors first)::

    python scripts/table.py \\
        --in 'results/se_eval/noisy__SE-valid-drone__group.csv' \\
             'results/se_eval/wiener__SE-valid-drone__group.csv' \\
             'results/se_eval/f1_*__SE-valid-drone__group.csv' \\
        --rows method --cols input_snr --metrics si_sdr,pesq,estoi \\
        --filter category=all --row-order noisy,wiener

  (the Pass B − Pass A diversity delta is the same table with
  ``--baseline f1_<arch>_a --metrics si_sdr``, per arch; the per-category
  transfer table is ``--rows method --cols category --metrics si_sdr`` on
  the SE-valid-harmonic CSVs.)

* ``f2_ladder_table.py`` (per-SNR ladder with deltas vs the noisy anchor)::

    python scripts/table.py --in 'results/f2_perclip/*.csv' \\
        --rows method --cols input_snr \\
        --metrics si_sdr,sdr,pesq_nb,estoi,gain_db,corr \\
        --baseline noisy --drop-silent

* ``se_valid_composition.py`` (clips per category × SNR, from any per-clip
  CSV of the set — e.g. the ``noisy`` anchor run)::

    python scripts/table.py \\
        --in 'results/se_eval/noisy__SE-valid-harmonic__clip.csv' \\
        --rows category --cols input_snr --agg count

* the WP18 weight-curve table (from the campaign's ``summary.json``; the
  campaign's own reporter is deleted, this call replaces it)::

    python scripts/table.py \\
        --in results/phase_noise_covariance/summary.json \\
        --json-root groups --index-name group --unnest arms=arm \\
        --rows group --cols arm --metrics alpha_signal.median

  (the other WP18 tables are the same pattern over ``cutoffs.*`` and
  ``chan_coherence`` leaves.)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

SILENT_SI_SDR_FLOOR = -70.0


# ── loading ─────────────────────────────────────────────────────────────────
def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys; lists stay opaque values."""
    out: dict[str, Any] = {}
    for k, v in record.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def json_records(
    payload: Any,
    json_root: str | None = None,
    index_name: str = "key",
    unnest: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Turn a JSON payload into flat records.

    ``json_root`` walks a dotted path first. A list is taken as records; a
    dict is taken as a ``name -> record`` mapping whose keys become the
    ``index_name`` column. Each ``(field, column)`` in ``unnest`` then
    explodes that dict-of-dicts field into one record per key.
    """
    for part in (json_root or "").split("."):
        if part:
            payload = payload[part]
    if isinstance(payload, list):
        records = [dict(r) for r in payload]
    elif isinstance(payload, dict):
        records = [{index_name: k, **v} for k, v in payload.items()]
    else:
        raise ValueError(f"cannot tabulate JSON payload of type {type(payload).__name__}")
    for field, column in unnest or []:
        exploded = []
        for r in records:
            sub = r.pop(field, None)
            if not isinstance(sub, dict):
                continue
            for k, v in sub.items():
                child = dict(v) if isinstance(v, dict) else {field: v}
                exploded.append({**r, column: k, **child})
        records = exploded
    return [_flatten(r) for r in records]


def load_inputs(
    patterns: list[str],
    json_root: str | None = None,
    index_name: str = "key",
    unnest: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    paths: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    frames = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"no such input: {p}")
        if path.suffix.lower() == ".json":
            recs = json_records(json.loads(path.read_text()), json_root, index_name, unnest)
            frames.append(pd.DataFrame(recs))
        else:
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no inputs matched --in")
    return pd.concat(frames, ignore_index=True)


# ── transforms ──────────────────────────────────────────────────────────────
def drop_silent(df: pd.DataFrame) -> pd.DataFrame:
    """Drop clips whose reference is digitally silent.

    A silent reference pins ``si_sdr`` at the eps floor for EVERY method at
    once, so a clip is silent iff no method exceeds the floor on it. Dropped
    clips are reported, not silently discarded. Needs per-clip rows with
    ``valid``/``clip_id``/``si_sdr`` columns.
    """
    if not {"valid", "clip_id", "si_sdr"}.issubset(df.columns):
        return df
    best = cast(pd.Series, df.groupby(["valid", "clip_id"])["si_sdr"].max())
    bad: set[tuple[str, str]] = set(best[best < SILENT_SI_SDR_FLOOR].index)  # type: ignore[arg-type]
    if not bad:
        return df
    print(f"dropping {len(bad)} silent-reference clip(s): {sorted(c for _, c in bad)[:8]}")
    keys = list(zip(df["valid"].astype(str), df["clip_id"].astype(str)))
    keep = [k not in bad for k in keys]
    return cast(pd.DataFrame, df[keep]).reset_index(drop=True)


def aggregate(
    df: pd.DataFrame, rows: list[str], cols: str | None, metrics: list[str], agg: str
) -> pd.DataFrame:
    keys = rows + ([cols] if cols else [])
    if agg == "count":
        counts = df.groupby(keys, dropna=False).size()
        counts.name = "n"
        return counts.reset_index()
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise SystemExit(f"metrics not in the data: {missing} (have {sorted(df.columns)})")
    return cast(pd.DataFrame, df.groupby(keys, dropna=False)[metrics].mean()).reset_index()


def pivot(
    grouped: pd.DataFrame,
    rows: list[str],
    cols: str | None,
    metric: str,
    baseline: str | None = None,
) -> pd.DataFrame:
    """One pivoted table for one metric (+ optional deltas vs a baseline row)."""
    if cols:
        table = grouped.pivot_table(index=rows, columns=cols, values=metric, aggfunc="mean")
    else:
        table = grouped.set_index(rows)[[metric]]
    if baseline is not None:
        key = (baseline,) if len(rows) == 1 else tuple(baseline.split("/"))
        flat_key = key[0] if len(rows) == 1 else key
        if flat_key not in table.index:
            raise SystemExit(f"--baseline {baseline!r} not among row values {list(table.index)}")
        deltas = table.subtract(table.loc[flat_key], axis="columns")
        deltas.columns = [f"d_{c}" for c in deltas.columns]
        table = pd.concat([table, deltas], axis=1)
    return cast(pd.DataFrame, table)


def order_rows(table: pd.DataFrame, row_order: list[str]) -> pd.DataFrame:
    if not row_order:
        return table
    idx = [str(i) for i in table.index]
    first = [i for o in row_order for i in table.index if str(i) == o]
    rest = [i for i in table.index if str(i) not in set(row_order)]
    del idx
    return table.loc[first + rest]


def to_markdown(table: pd.DataFrame, title: str) -> str:
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return "—" if v != v else f"{v:.3f}"
        return str(v)

    head = [str(table.index.name or "row"), *[str(c) for c in table.columns]]
    lines = [f"### {title}\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, row in table.iterrows():
        name = " / ".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
        lines.append("| " + " | ".join([name, *[_fmt(v) for v in row]]) + " |")
    return "\n".join(lines) + "\n"


# ── CLI ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="csv|json paths/globs")
    ap.add_argument("--rows", required=True, help="comma-separated row fields")
    ap.add_argument("--cols", default=None, help="single column field (optional)")
    ap.add_argument("--metrics", default="", help="comma-separated metric columns")
    ap.add_argument("--agg", choices=("mean", "count"), default="mean")
    ap.add_argument("--baseline", default=None, help="row value to compute delta columns against")
    ap.add_argument("--format", choices=("md", "csv"), default="md")
    ap.add_argument("--drop-silent", action="store_true", help="drop silent-reference clips")
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="keep only rows where FIELD == VALUE (repeatable)",
    )
    ap.add_argument("--row-order", default="", help="comma list: put these row values first")
    ap.add_argument("--json-root", default=None, help="dotted path to the JSON record mapping")
    ap.add_argument("--index-name", default="key", help="column name for JSON mapping keys")
    ap.add_argument(
        "--unnest",
        action="append",
        default=[],
        metavar="FIELD=COLUMN",
        help="explode a JSON dict-of-dicts field into records (repeatable)",
    )
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    args = ap.parse_args()

    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if args.agg == "mean" and not metrics:
        raise SystemExit("--metrics is required with --agg mean")
    unnest = []
    for spec in args.unnest:
        field, _, column = spec.partition("=")
        unnest.append((field, column or field))

    df = load_inputs(args.inputs, args.json_root, args.index_name, unnest)
    for spec in args.filter:
        field, _, value = spec.partition("=")
        df = cast(pd.DataFrame, df[df[field].astype(str) == value])
    if args.drop_silent:
        df = drop_silent(df)
    if df.empty:
        raise SystemExit("no rows left after filtering")

    grouped = aggregate(df, rows, args.cols, metrics, args.agg)
    row_order = [r.strip() for r in args.row_order.split(",") if r.strip()]
    cell_metrics = ["n"] if args.agg == "count" else metrics

    chunks = []
    for metric in cell_metrics:
        table = pivot(grouped, rows, args.cols, metric, args.baseline)
        table = order_rows(table, row_order)
        if args.format == "csv":
            chunks.append(table.to_csv())
        else:
            chunks.append(
                to_markdown(
                    table,
                    f"{metric} by {'/'.join(rows)}" + (f" × {args.cols}" if args.cols else ""),
                )
            )
    text = "\n".join(chunks)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
