"""
Train one `GPRotorNoiseModel` per drone (michaels, dregon) on the *swapped*
noise-generation training split (the same data the deep
`PositionalHarmonicNoiseGen` "swapped" run trained on, defined by
`conf/data/noise_rps_dregon_michaels_swapped.yaml` — i.e. per-recording time
hold-out with `val_at_start=True` so train = last 90%, val = first 10% of each
recording's time axis).

Mirrors the deep model's `DroneCodebook` (one code per drone) by training
*separate* GPs per drone, since the microphone geometry differs between
Michael's 8-mic ring and DREGON's 8-mic array.

Saves two checkpoints:
  results/gp_rotor_noise_dregon/best.pt
  results/gp_rotor_noise_michaels/best.pt

Usage
-----
    python -m src.experiments.gp_rotor_noise.train_dregon_michaels \
        --n_harmonics 24 --max_per_source_dur_s 30 --iters 600
"""
# pyright: reportOptionalMemberAccess=false

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_processing import sources
from experiments.gp_rotor_noise.gp_rotor_noise import GPRotorNoiseConfig, GPRotorNoiseModel

SR = 16000
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _recording_frames(source: str, splits: list[str] | None = None) -> list:
    """Published frames for a rig, audio resampled to SR (fixes baked in)."""
    return list(sources.iter_recording_frames(source, splits=splits, sample_rate=SR))


def upsample_rps_to_audio_rate(
    rps_motor: np.ndarray, motor_ts: np.ndarray, audio_ts: np.ndarray
) -> np.ndarray:
    """Linearly interpolate per-rotor RPS (R, M) onto audio timestamps (T,)."""
    R = rps_motor.shape[0]
    out = np.empty((R, audio_ts.shape[0]), dtype=np.float32)
    for r in range(R):
        out[r] = np.interp(audio_ts, motor_ts, rps_motor[r])
    return out


def _val_at_start_cut(dur_sec: float, val_pct: float = 0.1) -> float:
    """`val_at_start=True` swap: validation = [0, val_pct*dur], train =
    [val_pct*dur, dur]. Return the cut time (seconds from start) such that
    train lives in [cut, dur]."""
    return dur_sec * val_pct


def dregon_train_sources(
    splits=("in_flight_noise",), val_pct: float = 0.1
) -> tuple[list[dict], np.ndarray]:
    """Load DREGON `in_flight_noise`, swapped-split training halves.

    Returns (sources list, mic_pos (M,3)). `rotor_pos` is taken from
    `D.get_geometry(dregon_dir)` but the GP only uses mic_pos + rotor factor idx,
    so this is fine.
    """
    frames = [tf for tf in _recording_frames("DREGON", list(splits)) if "motors_measured" in tf]
    if not frames:
        raise RuntimeError("No DREGON in-flight frames with motors_measured in DREGON-frames")
    mic_pos = np.asarray(frames[0]["mic_pos"].data, dtype=np.float32)
    out = []
    for tf in frames:
        dur = tf["audio"].t_end - tf["audio"].t_start
        cut = _val_at_start_cut(dur, val_pct=val_pct)
        train_tf = tf.time[tf["audio"].t_start + cut : tf["audio"].t_end]
        if train_tf["audio"].duration < 1.0:
            continue
        audio = np.asarray(train_tf["audio"].data).astype(np.float32)  # (M, T)
        motor_s = train_tf["motors_measured"]
        motor_ts = np.asarray(motor_s.tindex.abs_stamps)
        motor_data = (
            np.asarray(motor_s.data).astype(np.float32) if motor_s.data is not None else None
        )
        if motor_data is None or motor_data.shape[0] != 4 or motor_data.shape[1] < 2:
            continue
        audio_ts = np.arange(audio.shape[-1]) / SR
        rps_audio = upsample_rps_to_audio_rate(motor_data, motor_ts - motor_ts[0], audio_ts)
        out.append(
            {
                "audio": audio,
                "rps_audio": rps_audio,
                "mic_pos": mic_pos,
                "recording_id": tf["meta"]["recording_id"],
            }
        )
    return out, mic_pos


def michaels_train_sources(val_pct: float = 0.1) -> tuple[list[dict], np.ndarray]:
    """Load both Michael's recordings, swapped-split training halves."""
    mic_pos = sources.geometry("michaels")[0].astype(np.float32)
    tfs = _recording_frames("michaels")
    out = []
    for tf in tfs:
        dur = tf["audio"].t_end - tf["audio"].t_start
        cut = _val_at_start_cut(dur, val_pct=val_pct)
        train_tf = tf.time[tf["audio"].t_start + cut : tf["audio"].t_end]
        if train_tf["audio"].duration < 1.0:
            continue
        audio = np.asarray(train_tf["audio"].data).astype(np.float32)
        motor_ts = np.asarray(train_tf["rps"].tindex.abs_stamps)
        motor_data = np.asarray(train_tf["rps"].data).astype(np.float32)
        audio_ts = np.arange(audio.shape[-1]) / SR
        rps_audio = upsample_rps_to_audio_rate(motor_data, motor_ts - motor_ts[0], audio_ts)
        rid = tf["meta"].get("recording_id", "")
        out.append(
            {"audio": audio, "rps_audio": rps_audio, "mic_pos": mic_pos, "recording_id": rid}
        )
    return out, mic_pos


def _fit_and_save(name: str, sources: list[dict], out_root: Path, cfg: GPRotorNoiseConfig) -> None:
    print(f"\n=== Training GP for drone='{name}' ({len(sources)} sources) ===")
    out = out_root / f"gp_rotor_noise_{name}"
    out.mkdir(parents=True, exist_ok=True)
    m = GPRotorNoiseModel(cfg)
    m.fit(sources)
    ckpt = out / "best.pt"
    m.save(ckpt)
    print(f"[save] {ckpt}  (n_mics={m.n_mics}, sigma_b mean={m.sigma_b_per_mic.mean():.4f})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n_harmonics",
        type=int,
        default=12,
        help="Harmonics per rotor. Caps spectral coverage at ~H*BPF."
        " Memory-bounded defaults (H=12, dur=10s, frames=48) keep peak"
        " <~1 GiB on CPU. Raise H for higher-band coverage.",
    )
    p.add_argument("--win", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument(
        "--max_per_source_dur_s",
        type=float,
        default=10.0,
        help="Cap per-recording training audio (seconds). Controls peak RAM:",
    )
    p.add_argument(
        "--max_total_frames",
        type=int,
        default=48,
        help="Cap total training frames (across mics × sources, after subsample)."
        " Determines GP training-point count = frames * M * (2*nrotors*H+1).",
    )
    p.add_argument(
        "--val_pct",
        type=float,
        default=0.1,
        help="Validation fraction (swapped: train = last 1-val_pct of each recording).",
    )
    p.add_argument("--out_root", type=str, default=str(PROJECT_ROOT / "results"))
    p.add_argument("--only", type=str, default="both", choices=("both", "dregon", "michaels"))
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    cfg = GPRotorNoiseConfig(
        sr=SR,
        win=args.win,
        hop=args.hop,
        n_harmonics=args.n_harmonics,
        iters=args.iters,
        max_per_source_dur_s=args.max_per_source_dur_s,
        max_total_frames=args.max_total_frames,
        verbose=True,
    )
    out_root = Path(args.out_root)
    if args.only in ("both", "michaels"):
        m_src, _ = michaels_train_sources(val_pct=args.val_pct)
        if not m_src:
            print("[warn] no Michael's sources — skipping")
        else:
            _fit_and_save("michaels", m_src, out_root, cfg)
    if args.only in ("both", "dregon"):
        d_src, _ = dregon_train_sources(val_pct=args.val_pct)
        if not d_src:
            print("[warn] no DREGON sources — skipping")
        else:
            _fit_and_save("dregon", d_src, out_root, cfg)


if __name__ == "__main__":
    main(parse_args())
