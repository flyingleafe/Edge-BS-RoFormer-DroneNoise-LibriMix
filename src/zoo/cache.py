"""Checkpoint-zoo cache: list the R2 artifact store, remember it locally.

R2 is the source of truth (docs/refactor-2026-08-plan.md § 3.3); the local
index is a **gitignored cache** at ``<repo-root>/.checkpoints-cache.json``.
Bucket/prefix conventions are derived from
:class:`training.artifacts.ArtifactStore` (its constructor defaults and
``_key_root`` layout), so there is exactly one copy of the
``<bucket>/<prefix>/<experiment>/checkpoints/...`` convention in the repo.

Cache file schema (``"schema": 1``)::

    {
      "schema": 1,
      "generated_at": <unix seconds, float>,
      "bucket": "ml-data",
      "prefix": "artifacts",
      "conf_names": ["<experiment yaml stems seen at refresh time>", ...],
      "experiments": {
        "<experiment>": {
          "checkpoints": [
            {"file": "best.ckpt", "key": "artifacts/<exp>/checkpoints/best.ckpt",
             "size": 123, "last_modified": "<iso8601>", "etag": "<hex>"}
          ],
          "has_config": true,          # conf/experiment/<exp>.yaml exists
          "task": "rps_prediction",    # from the referenced conf/model yaml, or null
          "metrics": {...} | null,     # embedded artifacts/<exp>/eval/metrics.json
          "listed_at": <unix seconds>
        }
      }
    }

Incremental refresh (``refresh(full=False)``): the cache remembers the known
experiment prefixes and the ``conf/experiment/*.yaml`` names seen last time.
A refresh diffs cheap signals — the top-level prefix set (one ``Delimiter="/"``
listing) and new conf yaml names — and re-lists only the changed experiments.
Prefixes that disappeared from R2 are dropped. ``full=True`` re-lists
everything. ``has_config``/``task`` are local and cheap, so they are
recomputed for every row on every refresh, without network calls.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.artifacts import ArtifactStore
from utils import checkpoints as _r2env

__all__ = ["CacheInfo", "refresh", "checkpoints", "CACHE_FILENAME", "REPO_ROOT"]

SCHEMA_VERSION = 1
CACHE_FILENAME = ".checkpoints-cache.json"

# src/zoo/cache.py -> src -> repo root; holds for the main checkout and any
# git worktree (conf/ and .gitignore are git-tracked, so they travel along).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Eval metrics JSONs are a handful of scalars; anything bigger than this is
# not a metrics summary and is left out rather than embedded in the cache.
_MAX_METRICS_BYTES = 1_000_000


@dataclass
class CacheInfo:
    """What one :func:`refresh` call did."""

    path: Path
    generated_at: float
    n_experiments: int
    refreshed: list[str]
    full: bool


# ─── Conventions (derived from ArtifactStore, never duplicated) ──────────────


def _conventions() -> tuple[str, str]:
    """The artifact store's default ``(bucket, prefix)``."""
    probe = ArtifactStore(experiment_name="_zoo_probe", enabled=False)
    return probe.bucket, probe.prefix


def _key_root(prefix: str, experiment: str) -> str:
    """``<prefix>/<experiment>`` — via ``ArtifactStore._key_root`` so the key
    layout has exactly one authoritative implementation."""
    return ArtifactStore(experiment_name=experiment, prefix=prefix, enabled=False)._key_root()


def _default_client() -> Any | None:
    """A real boto3 R2 client from ``.env`` creds, or ``None`` when missing —
    built through ``ArtifactStore`` so the endpoint construction stays single-copy."""
    return ArtifactStore(experiment_name="_zoo_probe", enabled=True)._get_client()


# ─── S3 listing helpers (paginated) ──────────────────────────────────────────


def _list_objects(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """All objects under ``prefix`` (follows ``list_objects_v2`` pagination)."""
    out: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        out.extend(resp.get("Contents") or [])
        if not resp.get("IsTruncated"):
            return out
        token = resp.get("NextContinuationToken")


def _list_experiment_prefixes(client: Any, bucket: str, prefix: str) -> set[str]:
    """Top-level experiment names under ``<prefix>/`` (``Delimiter="/"``)."""
    names: set[str] = set()
    root = f"{prefix}/"
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": root, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes") or []:
            name = str(cp.get("Prefix", ""))[len(root) :].strip("/")
            if name:
                names.add(name)
        if not resp.get("IsTruncated"):
            return names
        token = resp.get("NextContinuationToken")


# ─── Per-experiment harvest ──────────────────────────────────────────────────


def _config_info(experiment: str, conf_dir: Path) -> tuple[bool, str | None]:
    """(has ``conf/experiment/<name>.yaml``, task name when resolvable).

    The task comes from the ``override /model: <name>`` default →
    ``conf/model/<name>.yaml``'s ``task:`` field — a plain yaml read, no Hydra
    compose. Best-effort: any parse hiccup degrades to ``task=None``.
    """
    path = conf_dir / "experiment" / f"{experiment}.yaml"
    if not path.is_file():
        return False, None
    try:
        import yaml

        doc = yaml.safe_load(path.read_text()) or {}
        model_name: str | None = None
        for item in doc.get("defaults") or []:
            if isinstance(item, dict):
                for key, value in item.items():
                    if str(key).split()[-1].lstrip("/") == "model":
                        model_name = str(value)
        task: str | None = None
        if model_name:
            model_doc = (
                yaml.safe_load((conf_dir / "model" / f"{model_name}.yaml").read_text()) or {}
            )
            task = model_doc.get("task")
        if task is None:
            model_block = doc.get("model")
            if isinstance(model_block, dict):
                task = model_block.get("task")
        return True, task
    except Exception:
        return True, None


def _harvest_experiment(
    client: Any, bucket: str, prefix: str, experiment: str, conf_dir: Path
) -> dict[str, Any]:
    """One experiment's cache row: checkpoint objects, config presence, metrics."""
    root = _key_root(prefix, experiment)
    ckpts: list[dict[str, Any]] = []
    for obj in _list_objects(client, bucket, f"{root}/checkpoints/"):
        key = str(obj["Key"])
        last_modified: Any = obj.get("LastModified")
        if last_modified is not None and hasattr(last_modified, "isoformat"):
            last_modified = last_modified.isoformat()
        ckpts.append(
            {
                "file": key.rsplit("/", 1)[-1],
                "key": key,
                "size": int(obj.get("Size", 0)),
                "last_modified": last_modified,
                "etag": str(obj.get("ETag") or "").strip('"'),
            }
        )

    metrics: dict[str, Any] | None = None
    eval_objects = {str(o["Key"]): o for o in _list_objects(client, bucket, f"{root}/eval/")}
    metrics_key = f"{root}/eval/metrics.json"
    if metrics_key in eval_objects and int(eval_objects[metrics_key].get("Size", 0)) <= (
        _MAX_METRICS_BYTES
    ):
        try:
            body = client.get_object(Bucket=bucket, Key=metrics_key)["Body"].read()
            metrics = json.loads(body)
        except Exception:
            metrics = None

    has_config, task = _config_info(experiment, conf_dir)
    return {
        "checkpoints": ckpts,
        "has_config": has_config,
        "task": task,
        "metrics": metrics,
        "listed_at": time.time(),
    }


# ─── Cache file I/O ──────────────────────────────────────────────────────────


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cache = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(cache, dict) or cache.get("schema") != SCHEMA_VERSION:
        return None
    return cache


# ─── Public API ──────────────────────────────────────────────────────────────


def refresh(
    full: bool = False,
    *,
    client: Any | None = None,
    cache_path: str | Path | None = None,
    conf_dir: str | Path | None = None,
) -> CacheInfo:
    """(Re)populate the checkpoint cache from the R2 artifact store.

    Incremental by default — see the module docstring for what counts as a
    cheap change signal. ``client``/``cache_path``/``conf_dir`` are
    dependency-injection points for tests; production callers pass nothing.
    """
    bucket, prefix = _conventions()
    cache_file = Path(cache_path) if cache_path else REPO_ROOT / CACHE_FILENAME
    conf_root = Path(conf_dir) if conf_dir else REPO_ROOT / "conf"
    client = client if client is not None else _default_client()
    if client is None:
        raise RuntimeError(
            "zoo.refresh needs R2 credentials in .env "
            "(R2_ACCOUNT_ID + AWS keys) or an injected client"
        )

    cache = _read_cache(cache_file)
    if cache is None or cache.get("bucket") != bucket or cache.get("prefix") != prefix:
        full = True

    current = _list_experiment_prefixes(client, bucket, prefix)
    old_rows: dict[str, Any] = {} if full or cache is None else dict(cache.get("experiments", {}))
    old_conf_names: set[str] = (
        set() if full or cache is None else set(cache.get("conf_names") or [])
    )

    conf_names = {p.stem for p in sorted((conf_root / "experiment").glob("*.yaml"))}
    if full:
        to_list = set(current)
    else:
        to_list = current - set(old_rows)  # new top-level prefixes
        to_list |= (conf_names - old_conf_names) & current  # new experiment yamls

    experiments: dict[str, Any] = {}
    for experiment in sorted(current):
        if experiment in to_list or experiment not in old_rows:
            experiments[experiment] = _harvest_experiment(
                client, bucket, prefix, experiment, conf_root
            )
        else:
            row = dict(old_rows[experiment])
            # Local + free: keep config presence/task current without network.
            row["has_config"], row["task"] = _config_info(experiment, conf_root)
            experiments[experiment] = row

    payload = {
        "schema": SCHEMA_VERSION,
        "generated_at": time.time(),
        "bucket": bucket,
        "prefix": prefix,
        "conf_names": sorted(conf_names),
        "experiments": experiments,
    }
    cache_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return CacheInfo(
        path=cache_file,
        generated_at=float(payload["generated_at"]),  # type: ignore[arg-type]
        n_experiments=len(experiments),
        refreshed=sorted(to_list),
        full=full,
    )


def checkpoints(
    task: str | None = None,
    max_age_s: float = 86400,
    *,
    client: Any | None = None,
    cache_path: str | Path | None = None,
    conf_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Rows for every experiment known to the cache, newest listing wins.

    Auto-refreshes when the cache is missing or older than ``max_age_s`` —
    but only if R2 credentials are available (or a client is injected);
    otherwise it warns and returns whatever the cache holds, so offline use
    never hard-fails. Row keys: ``experiment``, ``task``, ``files``,
    ``sizes``, ``mtimes`` (parallel lists), ``has_config``, ``metrics``.
    """
    cache_file = Path(cache_path) if cache_path else REPO_ROOT / CACHE_FILENAME
    cache = _read_cache(cache_file)
    age = None if cache is None else time.time() - float(cache.get("generated_at", 0.0))
    if cache is None or age is None or age > max_age_s:
        if client is not None or _r2env.load_r2_env() is not None:
            refresh(client=client, cache_path=cache_file, conf_dir=conf_dir)
            cache = _read_cache(cache_file)
        else:
            warnings.warn(
                "zoo: checkpoint cache is "
                + ("missing" if cache is None else f"stale ({age:.0f}s old)")
                + " and R2 credentials are not available — returning cached data as-is; "
                "run zoo.refresh() from a machine with .env creds",
                stacklevel=2,
            )
    if cache is None:
        return []

    rows: list[dict[str, Any]] = []
    for experiment, row in sorted(cache.get("experiments", {}).items()):
        if task is not None and row.get("task") != task:
            continue
        ckpts = row.get("checkpoints") or []
        rows.append(
            {
                "experiment": experiment,
                "task": row.get("task"),
                "files": [c["file"] for c in ckpts],
                "sizes": [c["size"] for c in ckpts],
                "mtimes": [c["last_modified"] for c in ckpts],
                "has_config": bool(row.get("has_config")),
                "metrics": row.get("metrics"),
            }
        )
    return rows
