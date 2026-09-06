"""SPCUP19 egonoise source: 10 heterogeneous student-team drone rigs (DREGON/INRIA).

The "the-spcup19-egonoise-dataset" bonus task: 10 teams each recorded their own
drone's ego-noise with their own mic array (1/4/8/16 ch). Packages are wildly
heterogeneous — some ship loose .wav, some ship one big .mat with a nested,
team-specific struct. The builder is per-team + resilient (skips a recording it
can't parse) and bakes the per-team drone model / channel count / condition into
meta; where a .mat exposes mic positions they go into meta too (not a shared-dim
geometry Series — mic count need not match the stored audio channel count).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.sources._common import (
    audio_frame,
    meta_frame,
    read_audio_file,
    safe_key,
)

_TEAMS: dict[str, dict[str, Any]] = {
    "Idea_ssu": {"id": 393, "drone": "DJI Phantom 4 (GL300C)", "channels": 1},
    "Maverick": {"id": 394, "drone": "YH-19HW quadcopter", "channels": 4},
    "Diagonal_Unloading": {"id": 395, "drone": "DJI Phantom 4 PRO", "channels": 16},
    "ChuMS": {"id": 396, "drone": "Skylark M4-680 (Dotterel)", "channels": 8},
    "LEADS_UAV": {"id": 397, "drone": "DJI Phantom 3 Advanced", "channels": 1},
    "NSS_Chellamma": {"id": 398, "drone": "self-assembled UAV", "channels": 1},
    "KumamoTech": {"id": 399, "drone": "enRoute Zion PG560", "channels": 16},
    "AGH": {"id": 400, "drone": "quadrotor (unspecified)", "channels": 8},
    "KU_Leuven": {"id": 401, "drone": "MikroKopter MK EASY Quadro V3", "channels": 8},
    "Shout_COOEE": {"id": 405, "drone": "Intel Aero RTF", "channels": 8},
}
URLS = {
    f"{team}.zip": f"http://dregon.inria.fr/?smd_process_download=1&download_id={info['id']}"
    for team, info in _TEAMS.items()
}
_MIN_AUDIO_LEN = 8000  # below this, a numeric array is metadata (spectrum/SPL), not audio
_SR_FIELDS = ("fs", "samplerate", "sample_rate", "samplingrate", "samplingfrequency")
_MIC_FIELDS = ("micpositions", "micpos", "mic_positions", "micposition")
_AUDIO_TOKENS = ("raw", "audio", "signal", "calibrated", "recording", "data", "wav")


def _orient_audio(arr: np.ndarray) -> np.ndarray:
    """Numeric array → ``(C, T) float32`` with channels (the smaller axis) first."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim == 2:
        return np.ascontiguousarray(a if a.shape[0] <= a.shape[1] else a.T)
    return a.reshape(1, -1)


def _condition(name: str) -> str | None:
    low = name.lower()
    table = (
        ("free_flight", "free_flight"),
        ("free-flight", "free_flight"),
        ("stationary", "stationary"),
        ("static", "stationary"),
        ("hover", "hover"),
        ("spinn", "spinning"),
        ("spin", "spinning"),
        ("up_down", "up_down"),
        ("updown", "up_down"),
        ("takeoff", "takeoff"),
        ("landing", "landing"),
        ("single rotor", "single_rotor"),
        ("single_rotor", "single_rotor"),
        ("calibration", "calibration"),
    )
    return next((cond for key, cond in table if key in low), None)


def _frame(
    team: str,
    info: dict,
    rid_suffix: str,
    audio_ct: np.ndarray,
    sr: int,
    *,
    condition: str | None,
    relpath: str,
    mic_positions: list | None = None,
) -> tuple[str, td.Frame]:
    rid = f"{team}__{safe_key(rid_suffix)}"
    extra: dict[str, Any] = {
        "raw_relpath": relpath,
        "competition": "IEEE SP Cup 2019 ego-noise (bonus task)",
    }
    if mic_positions is not None:
        extra["mic_positions"] = mic_positions
    meta = meta_frame(
        rid,
        "SPCUP19-egonoise",
        system={
            "category": "drone",
            "make_model": info["drone"],
            "team": team,
            "n_channels_expected": info["channels"],
        },
        observation={
            "type": "onboard_array",
            "source_motion": "onboard",
            "relative_trajectory": "none",
            "mic_array": f"{info['channels']}ch",
        },
        operating={"condition": condition},
        label={"team": team, "drone": info["drone"]},
        extra=extra,
    )
    return safe_key(rid), audio_frame(audio_ct, int(sr), meta)


def _walk_mat(obj: Any, path: str, sr_ctx: int | None, out: list) -> None:
    """Collect ``(struct-path, numeric audio array, sr)`` from a loaded .mat,
    propagating the nearest ``Fs`` down the struct tree."""
    from scipy.io.matlab import mat_struct

    if isinstance(obj, mat_struct):
        sr_local = sr_ctx
        fields: list[str] = list(obj._fieldnames)  # pyright: ignore[reportAttributeAccessIssue]
        for f in fields:
            if f.lower() in _SR_FIELDS:
                val = getattr(obj, f)
                if np.isscalar(val):
                    with contextlib.suppress(TypeError, ValueError):
                        sr_local = int(np.asarray(val).astype(float).item())
        for f in fields:
            _walk_mat(getattr(obj, f), f"{path}/{f}", sr_local, out)
        return
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for i, item in enumerate(obj.ravel()):
            _walk_mat(item, f"{path}[{i}]", sr_ctx, out)
        return
    if (
        isinstance(obj, np.ndarray)
        and np.issubdtype(obj.dtype, np.number)
        and obj.ndim in (1, 2)
        and max(obj.shape) >= _MIN_AUDIO_LEN
    ):
        out.append((path, obj, sr_ctx))


def _find_mic_positions(d: dict) -> list | None:
    """First MicPositions-like 2D array anywhere in the loaded .mat, as a list."""
    from scipy.io.matlab import mat_struct

    stack: list = [v for k, v in d.items() if not k.startswith("__")]
    while stack:
        obj = stack.pop()
        if isinstance(obj, mat_struct):
            for f in list(obj._fieldnames):  # pyright: ignore[reportAttributeAccessIssue]
                if f.lower() in _MIC_FIELDS:
                    val = getattr(obj, f)
                    if isinstance(val, np.ndarray) and val.ndim == 2:
                        return np.asarray(val, dtype=np.float64).tolist()
                stack.append(getattr(obj, f))
        elif isinstance(obj, np.ndarray) and obj.dtype == object:
            stack.extend(obj.ravel().tolist())
    return None


def _mat_recordings(
    mat_path: Path, team: str, info: dict, rel_stem: str | None = None
) -> Iterator[tuple[str, td.Frame]]:
    from scipy.io import loadmat

    stem = rel_stem or mat_path.stem

    try:
        d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except Exception as exc:  # noqa: BLE001 - a broken team .mat shouldn't sink the publish
        print(f"  SPCUP19 {team}: cannot load {mat_path.name} ({exc})", flush=True)
        return
    found: list = []
    for k, v in d.items():
        if not k.startswith("__"):
            _walk_mat(v, k, None, found)
    if not found:
        return
    mic = _find_mic_positions(d)
    # Prefer arrays whose struct-path names them audio-like (drops spectrum/SPL).
    named = [t for t in found if any(tok in t[0].lower() for tok in _AUDIO_TOKENS)]
    picks = named or found
    for path, arr, sr in picks:
        try:
            audio = _orient_audio(arr)
        except Exception:  # noqa: BLE001
            continue
        yield _frame(
            team,
            info,
            f"{stem}_{path}",
            audio,
            sr or 48000,
            condition=_condition(path) or _condition(stem),
            relpath=f"{mat_path.name}:{path}",
            mic_positions=mic,
        )


def _dedup_key(seen: dict[str, int], key: str) -> str:
    """Return ``key`` the first time, then ``key_2``/``key_3``/… on collision."""
    n = seen.get(key, 0)
    seen[key] = n + 1
    return key if n == 0 else f"{key}_{n + 1}"


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """10 team packages under ``raw_dir/<team>/``. Loose .wav → one recording
    each; .mat → generic resilient extraction (see helpers).

    Teams lay files out in *scenario subdirs* (``static clean/1.wav``,
    ``ego-noise/single rotors/1.wav``, …) that reuse bare-integer stems, so keys
    are derived from the **team-relative path** (not the stem) — and the subdir
    tokens feed condition detection. A final per-team dedup guard suffixes any
    residual collision so the publish never aborts on a duplicate key."""
    for team, info in _TEAMS.items():
        team_dir = Path(raw_dir) / team
        if not team_dir.exists():
            continue
        seen: dict[str, int] = {}
        for wav in sorted(team_dir.rglob("*.wav")):
            try:
                audio, sr = read_audio_file(wav)
            except Exception:  # noqa: BLE001
                continue
            rel = wav.relative_to(team_dir).with_suffix("")  # e.g. "static clean/1"
            key, frame = _frame(
                team,
                info,
                str(rel),
                audio,
                sr,
                condition=_condition(str(rel)),
                relpath=str(wav.relative_to(raw_dir)),
            )
            yield _dedup_key(seen, key), frame
        for mat in sorted(team_dir.rglob("*.mat")):
            rel_stem = str(mat.relative_to(team_dir).with_suffix(""))
            for key, frame in _mat_recordings(mat, team, info, rel_stem):
                yield _dedup_key(seen, key), frame


PROVENANCE = {
    "source_url": "https://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/",
    "doi": "10.48550/arXiv.1907.04655",
    "license": "free for personal, educational and academic use only",
    "citation": "Deleforge et al., Audio-Based Search and Rescue With a Drone: IEEE SP Cup 2019 (IEEE SPM 36(5), 2019).",
    "collection_method": "10 student teams recorded their own drone's ego-noise with their own on-board mic array (bonus task)",
    "equipment": "heterogeneous: 10 different drones + mic arrays (1/4/8/16 ch); see per-sample system.make_model + team",
    "observation_type": "onboard_array",
    "sample_rate": "varies per team (wav teams 44.1 kHz; .mat teams carry Fs)",
    "channels": "1/4/8/16 varies per team",
    "description": "Drone-variety ego-noise: 10 heterogeneous team rigs (Phantom 3/4, Skylark, Intel Aero, MikroKopter, ...). Loose .wav + team-specific .mat; rich per-team drone/condition meta, mic positions where the .mat exposes them.",
}
