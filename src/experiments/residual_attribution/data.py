"""Fetch the VK-decomposition residual artifacts and the array geometry.

Artifacts live on R2 under ``artifacts/vk-decompose/<recording_id>/`` in the
``ml-data`` bucket (written by ``scripts/vk_decompose.py``). Credentials come
from ``.env`` exactly as :mod:`training.artifacts` loads them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Residual", "list_decompositions", "fetch_residual", "geometry_for"]

BUCKET = "ml-data"
PREFIX = "artifacts/vk-decompose"
DEFAULT_CACHE = Path(".cache/vk_decompose")


def _client():
    from utils.checkpoints import load_r2_env

    env = load_r2_env()
    if env is None:
        raise RuntimeError(
            "R2 credentials missing — `set -a; source .env; set +a` before running, "
            "or export R2_ACCOUNT_ID / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
        )
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def list_decompositions() -> list[str]:
    """Recording ids that have a decomposition on R2."""
    cli = _client()
    out: set[str] = set()
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": PREFIX + "/"}
        if token:
            kw["ContinuationToken"] = token
        resp = cli.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            rest = o["Key"][len(PREFIX) + 1 :]
            if "/" in rest:
                out.add(rest.split("/", 1)[0])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(out)


@dataclass
class Residual:
    recording_id: str
    audio: np.ndarray  # (M, T) float32
    sample_rate: int
    t_start_s: float
    freq_hz: np.ndarray
    psd_residual: np.ndarray  # (M, F)
    psd_original: np.ndarray  # (M, F)
    report: dict


def fetch_residual(recording_id: str, cache_dir: str | Path = DEFAULT_CACHE) -> Residual:
    """Download (once) and load ``residual.npz`` + ``report.json``."""
    d = Path(cache_dir) / recording_id
    d.mkdir(parents=True, exist_ok=True)
    cli = None
    for name in ("residual.npz", "report.json"):
        p = d / name
        if not p.exists():
            cli = cli or _client()
            cli.download_file(BUCKET, f"{PREFIX}/{recording_id}/{name}", str(p))
    with np.load(d / "residual.npz") as z:
        return Residual(
            recording_id=recording_id,
            audio=np.asarray(z["residual"]),
            sample_rate=int(z["sample_rate"]),
            t_start_s=float(z["t_start_s"]),
            freq_hz=np.asarray(z["freq_hz"]),
            psd_residual=np.asarray(z["psd_residual"]),
            psd_original=np.asarray(z["psd_original"]),
            report=json.loads((d / "report.json").read_text()),
        )


#: Recording-id prefixes -> the drone whose geometry applies.
_DRONE_BY_PREFIX = {"free-flight": "dregon", "hovering": "dregon", "static": "dregon"}


def drone_of(recording_id: str) -> str:
    for pref, drone in _DRONE_BY_PREFIX.items():
        if recording_id.startswith(pref):
            return drone
    if recording_id.upper().startswith("FLY"):
        return "michaels"
    raise ValueError(f"cannot infer the drone of {recording_id!r}")


def geometry_for(
    drone: str, cache_dir: str | Path = DEFAULT_CACHE
) -> tuple[np.ndarray, np.ndarray]:
    """``(mic_pos (M,3), rotor_pos (R,3))`` in metres, cached to ``cache_dir``.

    DREGON's geometry is TDOA-validated (``docs`` / stage-0 RTF report).
    Michael's is a **synthetic ring model** — unvalidated; any attribution done
    on it inherits that uncertainty.
    """
    p = Path(cache_dir) / f"{drone}_geometry.npz"
    if p.exists():
        with np.load(p) as z:
            return np.asarray(z["mic_pos"]), np.asarray(z["rotor_pos"])
    if drone == "dregon":
        from data_processing.frame_datasets import _frames_spec_geometry

        mic, rot = _frames_spec_geometry("frames:DREGON-frames")
    elif drone == "michaels":
        from data_processing import sources

        mic, rot = sources.geometry("michaels")
    else:
        raise ValueError(f"unknown drone {drone!r}")
    mic = np.asarray(mic, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, mic_pos=mic, rotor_pos=rot)
    return mic, rot


def env_hint() -> str:
    return "R2_ACCOUNT_ID" in os.environ and "ok" or "missing R2_ACCOUNT_ID"
