"""Unit tests for ``zoo.cache`` (refresh + checkpoints) — no network.

Mirrors ``tests/training/test_artifacts.py``'s fake-client pattern: an
in-memory S3 stand-in is dependency-injected via ``refresh(client=...)`` /
``checkpoints(client=...)``, here additionally exposing paginated
``list_objects_v2`` (with ``Delimiter`` support) and ``get_object``, plus a
call log so the incremental-refresh tests can assert what was (not)
re-listed.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import pytest

from zoo import cache as zoo_cache


class FakeS3Client:
    """Paginated in-memory ``boto3.client("s3")`` stand-in (single bucket)."""

    def __init__(self, *, page_size: int = 2) -> None:
        self.objects: dict[str, bytes] = {}
        self.meta: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, ...]] = []
        self.page_size = page_size

    def put(self, key: str, body: bytes = b"x", *, etag: str = "e0") -> None:
        self.objects[key] = body
        self.meta[key] = {
            "Key": key,
            "Size": len(body),
            "LastModified": "2026-08-01T00:00:00+00:00",
            "ETag": f'"{etag}"',
        }

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        Delimiter: str | None = None,
        ContinuationToken: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("list", Prefix, Delimiter or ""))
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        entries: list[tuple[str, str]] = []
        if Delimiter:
            seen: set[str] = set()
            for k in keys:
                rest = k[len(Prefix) :]
                if Delimiter in rest:
                    p = Prefix + rest.split(Delimiter)[0] + Delimiter
                    if p not in seen:
                        seen.add(p)
                        entries.append(("cp", p))
                else:
                    entries.append(("obj", k))
        else:
            entries = [("obj", k) for k in keys]
        start = int(ContinuationToken or 0)
        page = entries[start : start + self.page_size]
        resp: dict[str, Any] = {
            "Contents": [self.meta[k] for kind, k in page if kind == "obj"],
            "CommonPrefixes": [{"Prefix": p} for kind, p in page if kind == "cp"],
            "IsTruncated": start + self.page_size < len(entries),
        }
        if resp["IsTruncated"]:
            resp["NextContinuationToken"] = str(start + self.page_size)
        return resp

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append(("get", Key))
        return {"Body": io.BytesIO(self.objects[Key])}


# ─── fixtures ────────────────────────────────────────────────────────────────


EXP1_METRICS = {"pit_mse": 2.5, "r2": 0.9}


def make_conf(tmp_path: Path) -> Path:
    """A minimal conf tree: exp1 -> model m1 -> task rps_prediction."""
    conf = tmp_path / "conf"
    (conf / "experiment").mkdir(parents=True)
    (conf / "model").mkdir()
    (conf / "experiment" / "exp1.yaml").write_text(
        "# @package _global_\ndefaults:\n  - override /model: m1\nexperiment_name: exp1\n"
    )
    (conf / "model" / "m1.yaml").write_text("task: rps_prediction\n")
    return conf


def make_client() -> FakeS3Client:
    client = FakeS3Client(page_size=1)  # page_size=1 forces pagination everywhere
    client.put("artifacts/exp1/checkpoints/best.ckpt", b"ckpt-bytes-1", etag="a1")
    client.put("artifacts/exp1/checkpoints/ep3_mse_0.1.ckpt", b"ckpt-bytes-22", etag="a2")
    client.put("artifacts/exp1/eval/metrics.json", json.dumps(EXP1_METRICS).encode(), etag="a3")
    client.put("artifacts/exp2/checkpoints/best.ckpt", b"zz", etag="b1")
    return client


def do_refresh(tmp_path: Path, client: FakeS3Client, **kwargs: Any) -> zoo_cache.CacheInfo:
    return zoo_cache.refresh(
        client=client,
        cache_path=tmp_path / ".checkpoints-cache.json",
        conf_dir=make_conf(tmp_path) if not (tmp_path / "conf").is_dir() else tmp_path / "conf",
        **kwargs,
    )


# ─── full refresh ────────────────────────────────────────────────────────────


def test_full_refresh_harvests_everything(tmp_path):
    client = make_client()
    info = do_refresh(tmp_path, client, full=True)

    assert info.full is True
    assert info.n_experiments == 2
    assert info.refreshed == ["exp1", "exp2"]

    cache = json.loads((tmp_path / ".checkpoints-cache.json").read_text())
    assert cache["schema"] == 1
    assert cache["bucket"] == "ml-data"
    assert cache["prefix"] == "artifacts"

    exp1 = cache["experiments"]["exp1"]
    files = {c["file"]: c for c in exp1["checkpoints"]}
    assert set(files) == {"best.ckpt", "ep3_mse_0.1.ckpt"}
    assert files["best.ckpt"]["key"] == "artifacts/exp1/checkpoints/best.ckpt"
    assert files["best.ckpt"]["size"] == len(b"ckpt-bytes-1")
    assert files["best.ckpt"]["etag"] == "a1"
    assert files["best.ckpt"]["last_modified"] == "2026-08-01T00:00:00+00:00"
    assert exp1["has_config"] is True
    assert exp1["task"] == "rps_prediction"
    assert exp1["metrics"] == EXP1_METRICS

    exp2 = cache["experiments"]["exp2"]
    assert exp2["has_config"] is False
    assert exp2["task"] is None
    assert exp2["metrics"] is None
    assert [c["file"] for c in exp2["checkpoints"]] == ["best.ckpt"]


def test_refresh_without_cache_is_full(tmp_path):
    client = make_client()
    info = do_refresh(tmp_path, client)  # full=False, but no cache exists yet
    assert info.full is True
    assert info.refreshed == ["exp1", "exp2"]


# ─── incremental refresh ─────────────────────────────────────────────────────


def test_incremental_lists_only_new_prefixes(tmp_path):
    client = make_client()
    do_refresh(tmp_path, client, full=True)

    # A new experiment appears on R2; exp1/exp2 are unchanged.
    client.put("artifacts/exp3/checkpoints/best.ckpt", b"new", etag="c1")
    client.calls.clear()

    info = do_refresh(tmp_path, client)

    assert info.full is False
    assert info.refreshed == ["exp3"]
    listed_prefixes = {c[1] for c in client.calls if c[0] == "list"}
    # The delimiter listing of the root is the only touch on known prefixes.
    assert "artifacts/exp1/checkpoints/" not in listed_prefixes
    assert "artifacts/exp2/checkpoints/" not in listed_prefixes
    assert "artifacts/exp3/checkpoints/" in listed_prefixes
    # No metrics re-downloads for unchanged experiments.
    assert ("get", "artifacts/exp1/eval/metrics.json") not in client.calls

    cache = json.loads((tmp_path / ".checkpoints-cache.json").read_text())
    assert set(cache["experiments"]) == {"exp1", "exp2", "exp3"}
    # Unchanged rows carried over intact (checkpoints + embedded metrics).
    exp1 = cache["experiments"]["exp1"]
    assert {c["file"] for c in exp1["checkpoints"]} == {"best.ckpt", "ep3_mse_0.1.ckpt"}
    assert exp1["metrics"] == EXP1_METRICS


def test_incremental_relists_on_new_conf_yaml(tmp_path):
    client = make_client()
    do_refresh(tmp_path, client, full=True)

    # exp2 gains a config file (cheap signal) + a new checkpoint on R2.
    (tmp_path / "conf" / "experiment" / "exp2.yaml").write_text(
        "defaults:\n  - override /model: m1\n"
    )
    client.put("artifacts/exp2/checkpoints/ep9.ckpt", b"nine", etag="b2")
    client.calls.clear()

    info = do_refresh(tmp_path, client)

    assert info.refreshed == ["exp2"]
    listed_prefixes = {c[1] for c in client.calls if c[0] == "list"}
    assert "artifacts/exp2/checkpoints/" in listed_prefixes
    assert "artifacts/exp1/checkpoints/" not in listed_prefixes

    cache = json.loads((tmp_path / ".checkpoints-cache.json").read_text())
    exp2 = cache["experiments"]["exp2"]
    assert {c["file"] for c in exp2["checkpoints"]} == {"best.ckpt", "ep9.ckpt"}
    assert exp2["has_config"] is True
    assert exp2["task"] == "rps_prediction"


def test_incremental_drops_removed_prefixes(tmp_path):
    client = make_client()
    do_refresh(tmp_path, client, full=True)

    for key in list(client.objects):
        if key.startswith("artifacts/exp2/"):
            del client.objects[key]
            del client.meta[key]

    do_refresh(tmp_path, client)
    cache = json.loads((tmp_path / ".checkpoints-cache.json").read_text())
    assert set(cache["experiments"]) == {"exp1"}


# ─── checkpoints() ───────────────────────────────────────────────────────────


def test_checkpoints_rows_and_task_filter(tmp_path):
    client = make_client()
    do_refresh(tmp_path, client, full=True)

    rows = zoo_cache.checkpoints(
        cache_path=tmp_path / ".checkpoints-cache.json", conf_dir=tmp_path / "conf", client=client
    )
    by_exp = {r["experiment"]: r for r in rows}
    assert set(by_exp) == {"exp1", "exp2"}
    row = by_exp["exp1"]
    assert set(row["files"]) == {"best.ckpt", "ep3_mse_0.1.ckpt"}
    assert len(row["sizes"]) == len(row["files"]) == len(row["mtimes"])
    assert row["has_config"] is True
    assert row["metrics"] == EXP1_METRICS
    assert by_exp["exp2"]["metrics"] is None

    only_rps = zoo_cache.checkpoints(
        task="rps_prediction",
        cache_path=tmp_path / ".checkpoints-cache.json",
        conf_dir=tmp_path / "conf",
        client=client,
    )
    assert [r["experiment"] for r in only_rps] == ["exp1"]


def test_checkpoints_offline_stale_cache_warns_and_returns(tmp_path, monkeypatch):
    client = make_client()
    do_refresh(tmp_path, client, full=True)

    # Age the cache far past max_age_s, then go offline (no creds).
    cache_file = tmp_path / ".checkpoints-cache.json"
    payload = json.loads(cache_file.read_text())
    payload["generated_at"] = time.time() - 10 * 86400
    cache_file.write_text(json.dumps(payload))
    monkeypatch.setattr("utils.checkpoints.load_r2_env", lambda: None)

    with pytest.warns(UserWarning, match="stale"):
        rows = zoo_cache.checkpoints(cache_path=cache_file, conf_dir=tmp_path / "conf")
    assert {r["experiment"] for r in rows} == {"exp1", "exp2"}


def test_checkpoints_offline_missing_cache_warns_and_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.checkpoints.load_r2_env", lambda: None)
    with pytest.warns(UserWarning, match="missing"):
        rows = zoo_cache.checkpoints(cache_path=tmp_path / "nope.json", conf_dir=tmp_path / "conf")
    assert rows == []


def test_checkpoints_fresh_cache_needs_no_client(tmp_path, monkeypatch):
    client = make_client()
    do_refresh(tmp_path, client, full=True)
    monkeypatch.setattr("utils.checkpoints.load_r2_env", lambda: None)

    # Fresh cache: no refresh attempt, no warning, offline is fine.
    rows = zoo_cache.checkpoints(
        cache_path=tmp_path / ".checkpoints-cache.json", conf_dir=tmp_path / "conf"
    )
    assert len(rows) == 2
