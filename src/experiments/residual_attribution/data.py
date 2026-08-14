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


# ─── Local inputs for the band-power instrument (:mod:`power`) ───────────────


def local_residual(recording_id: str, root: str | Path) -> Residual:
    """Load a decomposition written by ``scripts/vk_decompose.py`` from disk.

    Same record as :func:`fetch_residual`, no R2 round trip — the campaign runs
    on a laptop against ``results/vk_decompose_v2/``.
    """
    d = Path(root) / recording_id
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


#: DREGON single-motor bench recordings: one rotor running, seven throttles
#: apart. ``Motor{n}`` is rotor index ``n - 1`` of ``coordinates.mat`` —
#: verified by the per-microphone pattern, whose nearest free-field column is
#: the diagonal one in every band (see ``power.bench_basis``).
BENCH_DIR = "DREGON/DREGON_individual_motors_recordings"
BENCH_SPEEDS = (50, 60, 70, 80, 90)


def bench_clips(
    data_root: str | Path,
    *,
    speeds: tuple[int, ...] = BENCH_SPEEDS,
    seconds: float = 10.0,
) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    """``{(rotor, throttle): (C, N)}`` from the middle of each bench recording.

    The middle avoids the run-up and the shutdown. Returns the clips and the
    sample rate (44.1 kHz as shipped — do NOT resample: the 8-16 kHz band is
    where the rotor patterns separate best).
    """
    import soundfile as sf

    root = Path(data_root) / BENCH_DIR
    clips: dict[tuple[int, int], np.ndarray] = {}
    sr = 0
    for motor in (1, 2, 3, 4):
        for speed in speeds:
            p = root / f"Motor{motor}_{speed}.wav"
            if not p.exists():
                continue
            info = sf.info(str(p))
            sr = int(info.samplerate)
            n = int(seconds * sr)
            start = max(0, (info.frames - n) // 2)
            x, _ = sf.read(str(p), start=start, frames=n, dtype="float64", always_2d=True)
            clips[(motor - 1, speed)] = np.ascontiguousarray(x.T)
    if not clips:
        raise FileNotFoundError(f"no bench recordings under {root}")
    return clips, sr


def bench_combined(data_root: str | Path, *, throttle: int = 70, seconds: float = 10.0):
    """The all-motors bench clip at one throttle — the additivity reference."""
    import soundfile as sf

    p = Path(data_root) / BENCH_DIR / f"allMotors_{throttle}.wav"
    info = sf.info(str(p))
    n = int(seconds * info.samplerate)
    start = max(0, (info.frames - n) // 2)
    x, _ = sf.read(str(p), start=start, frames=n, dtype="float64", always_2d=True)
    return np.ascontiguousarray(x.T), int(info.samplerate)


def rotor_speeds(
    recording_id: str, *, repo_root: str | Path = "."
) -> tuple[np.ndarray, np.ndarray]:
    """``(rps (R, M), t (M,))`` on the DECOMPOSITION's own time base.

    ``t`` is seconds from sample 0 of ``residual.npz``. DREGON reads the
    committed refined-label sidecar (the labels the decomposition itself ran
    on); Michael's replays the alignment of
    ``sources.michaels.load_raw_aligned`` without decoding the audio. Both were
    checked against the per-window ``mean_rev_s`` of ``report.json`` and agree
    to 0.001 rev/s on interior windows.
    """
    root = Path(repo_root)
    if recording_id.upper().startswith("FLY"):
        import pandas as pd
        import soundfile as sf

        from data_processing.sources.michaels import (
            MICHAELS_FILES,
            resolve_raw_root,
            rps_scale_for,
        )

        raw = resolve_raw_root(root / "data/recording_with_motor_speed")
        for wav_rel, csv_rel, offset, dilation in MICHAELS_FILES:
            if Path(csv_rel).stem.upper() != recording_id.upper():
                continue
            duration = sf.info(str(raw / wav_rel)).duration
            csv = pd.read_csv(raw / csv_rel, low_memory=False)
            cols = [c for c in csv.columns if "Motor" in c][:4]
            t_col = "Clock:offsetTime"
            cut = csv[[t_col, *cols]][(csv[t_col] >= offset) & (csv[t_col] <= duration + offset)]
            stamps = np.asarray(cut[t_col], dtype=np.float64)
            jump = int(np.argmax(np.diff(stamps)))
            stamps[0 : jump + 2] = np.linspace(stamps[0], stamps[jump + 2], jump + 2)
            stamps *= dilation
            rps = np.asarray(cut[cols], dtype=np.float64).T / 60 * rps_scale_for(raw / csv_rel)
            return rps, stamps - stamps[0]
        raise KeyError(recording_id)

    with np.load(root / f"src/data_processing/refined_labels/{recording_id}.npz") as z:
        return np.asarray(z["r_refined"]), np.asarray(z["ft"]) - float(z["t0_offset_s"])
