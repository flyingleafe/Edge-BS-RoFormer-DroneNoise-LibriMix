"""AVQ source (audio-visual quadrotor: onboard 8-mic array + speech + DOA/VAD)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.frames import audio_series
from data_processing.sources._common import meta_frame, read_audio_file, safe_key

URL = "https://webspace.eecs.qmul.ac.uk/lin.wang/download/avq.zip"


def _mic_pos(session_dir: Path) -> np.ndarray | None:
    """``misc/mic_pos.mat`` -> ``(mic, 3)`` positions (source is ``(3, 8)``)."""
    from scipy.io import loadmat

    p = session_dir / "misc" / "mic_pos.mat"
    if not p.exists():
        return None
    mp = np.asarray(loadmat(str(p), squeeze_me=True)["mic_pos"], dtype=np.float64)
    if mp.ndim == 2 and mp.shape[0] == 3:  # (3, 8) -> (8, 3)
        mp = mp.T
    return np.ascontiguousarray(mp)


def _session_info(session_dir: Path) -> dict[str, Any]:
    """Session-level calibration (``angle_a = a1*angle_v + a2``) + readme text."""
    from scipy.io import loadmat

    out: dict[str, Any] = {}
    cal_p = session_dir / "misc" / "av_calibration.mat"
    if cal_p.exists():
        d = loadmat(str(cal_p), squeeze_me=True, struct_as_record=False)
        cal = d.get("Calibration")
        if cal is not None and hasattr(cal, "_fieldnames"):
            for f in cal._fieldnames:  # pyright: ignore[reportAttributeAccessIssue]
                with contextlib.suppress(TypeError, ValueError):
                    out[f"av_calib_{f}"] = float(getattr(cal, f))
        if "description" in d:
            out["av_calib_description"] = str(d["description"])
    readme_p = session_dir / "misc" / "readme.txt"
    if readme_p.exists():
        out["readme"] = readme_p.read_text(errors="ignore").strip()
    return out


def _angle_vad(seq_dir: Path) -> tuple[np.ndarray | None, str]:
    """``angle_vad.mat`` -> ``(step, col)`` DOA/VAD ground truth + its column
    description (schema differs per session — preserved verbatim, not unified)."""
    from scipy.io import loadmat

    p = seq_dir / "angle_vad.mat"
    if not p.exists():
        return None, ""
    d = loadmat(str(p), squeeze_me=True)
    angles = np.asarray(d["angles"], dtype=np.float64)
    if angles.ndim == 1:
        angles = angles[:, None]
    return np.ascontiguousarray(angles), str(d.get("description", ""))


def _channels(seq_dir: Path) -> tuple[np.ndarray, int] | None:
    """Stack the ``MONO-000..NNN`` mono files (case-insensitive) into ``(C, T)``,
    truncating to the shortest channel. Returns None if no channel files."""
    wavs = sorted(
        (
            p
            for p in seq_dir.iterdir()
            if p.suffix.lower() == ".wav" and p.stem.lower().startswith("mono")
        ),
        key=lambda p: p.name.lower(),
    )
    if not wavs:
        return None
    chans: list[np.ndarray] = []
    sr0: int | None = None
    for w in wavs:
        audio_ct, sr = read_audio_file(w)  # mono file -> (1, T)
        chans.append(audio_ct[0])
        sr0 = sr0 or sr
    length = min(c.shape[0] for c in chans)
    audio = np.stack([c[:length] for c in chans], axis=0)  # (C, T)
    return np.ascontiguousarray(audio), int(sr0 or 44100)


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """One recording Frame per sequence: 8-ch ``audio`` (native 44.1 kHz) +
    ``mic_pos`` + ``angle_vad`` (labeled sequences) + rich per-session meta (AV
    calibration, readme). The bulky/opaque blobs (video, cameraParams.mat,
    .docx) are kept byte-exact in the companion raw dataset ``AVQ-raw``."""
    root = Path(raw_dir)
    # mic_pos.mat lives at <session>/misc/mic_pos.mat -> session = parent of misc/
    session_dirs = sorted({p.parent.parent for p in root.rglob("misc/mic_pos.mat")})
    for sess_dir in session_dirs:
        session = sess_dir.name  # "S1" / "S2"
        mic_pos = _mic_pos(sess_dir)
        info = _session_info(sess_dir)
        seq_dirs = sorted(
            p for p in sess_dir.iterdir() if p.is_dir() and p.name.lower().startswith("seq")
        )
        for seq_dir in seq_dirs:
            got = _channels(seq_dir)
            if got is None:
                continue
            audio, sr = got
            angles, ang_desc = _angle_vad(seq_dir)
            has_video = any(p.suffix.lower() == ".mp4" for p in seq_dir.iterdir())
            rid = f"{session}_{seq_dir.name}"
            extra: dict[str, Any] = {
                "session": session,
                "sequence": seq_dir.name,
                "sample_rate": int(sr),
                "n_channels": int(audio.shape[0]),
                "duration_s": round(audio.shape[1] / sr, 3),
                "has_video": has_video,
                "has_angle_vad": angles is not None,
                **info,
            }
            if angles is not None:
                extra["angle_vad_columns"] = ang_desc
            meta = meta_frame(
                rid,
                "AVQ",
                system={
                    "category": "drone",
                    "make_model": "quadrotor (AVQ)",
                    "mic_array": f"{audio.shape[0]}ch onboard",
                },
                observation={
                    "type": "onboard_array",
                    "source_motion": "external speech source; onboard ego-noise",
                    "relative_trajectory": "moving source",
                    "mic_array": f"{audio.shape[0]}ch",
                    "video_ground_truth": has_video,
                },
                operating={"content": "speech + rotor ego-noise"},
                label={"has_angle_vad": angles is not None, "has_video": has_video},
                extra=extra,
            )
            entries: dict[str, Any] = {
                "audio": audio_series(audio, int(sr)),
                "meta": meta,
            }
            if mic_pos is not None:
                entries["mic_pos"] = td.wrap(mic_pos, dims=("mic", None))
            if angles is not None:
                entries["angle_vad"] = td.wrap(angles, dims=("avq_step", None))
            yield safe_key(rid), td.Frame(entries)


PROVENANCE = {
    "source_url": "https://webspace.eecs.qmul.ac.uk/lin.wang/download/avq.zip",
    "project_url": "https://webspace.eecs.qmul.ac.uk/lin.wang/",
    "license": "free for academic/research use (courtesy of Lin Wang, QMUL)",
    "citation": "L. Wang and A. Cavallaro, audio-visual quadrotor (AVQ) ego-noise / sound-source localization dataset, QMUL.",
    "collection_method": "onboard 8-mic array on a quadrotor recording an external speech source under rotor ego-noise; synchronized GoPro video gives DOA + VAD ground truth.",
    "equipment": "quadrotor + onboard 8-microphone array (positions in mic_pos.mat) + GoPro camera",
    "observation_type": "onboard_array",
    "sample_rate": 44100,
    "channels": 8,
    "description": "Audio-visual quadrotor: 2 sessions (S1/S2), 12 sequences of 8-ch onboard-array audio (44.1 kHz) with rotor ego-noise + a moving speech source. Labeled sequences carry angle_vad.mat (per-session DOA/VAD schema, preserved verbatim) + mic geometry + AV calibration. Bulky/opaque blobs (video, cameraParams.mat, .docx) live byte-exact in the companion raw dataset AVQ-raw.",
    "companion_raw_dataset": "AVQ-raw",
}
