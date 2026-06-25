"""
Evaluate rotor localization on DREGON, where mic and rotor positions are known.

DREGON gives us the ground-truth ``rotor_positions`` (4, 3) and ``mic_positions``
(8, 3) in a common frame, so we can measure localization error directly.

Usage
-----
    python -m src.localization.eval_dregon                       # default recording/window
    python -m src.localization.eval_dregon --recording free-flight_nosource_room1 \
        --t0 20 --dur 3 --mode both

``--mic_noise`` perturbs the mic positions to simulate the "approximate geometry"
case the method is meant to handle.
"""

from __future__ import annotations

import argparse

import numpy as np

from data_processing.dregon import load_dregon_timeframes

from .rotor_localization import localize_rotors, match_and_score


def _get_window(tf, t0: float, dur: float):
    """Return (audio (C,N), sr, rps (4,M)) for a window starting ``t0`` s into the clip."""
    sub = tf.slice(tf.t_start + t0, tf.t_start + t0 + dur)
    audio = np.asarray(sub["audio"].samples, dtype=np.float32)  # (C, N)
    sr = float(sub["audio"].sr)
    motor_key = "motors_measured" if "motors_measured" in sub.tracks else "motors_command"
    rps = np.asarray(sub[motor_key].values, dtype=np.float64)  # (4, M)
    return audio, sr, rps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--recording", default="free-flight_nosource_room1")
    ap.add_argument("--split", default="in_flight_noise")
    ap.add_argument("--t0", type=float, default=20.0, help="window start (s into recording)")
    ap.add_argument("--dur", type=float, default=4.0, help="window duration (s)")
    ap.add_argument("--mode", choices=["audio", "rps", "both"], default="both")
    ap.add_argument("--coarse_step", type=float, default=0.02)
    ap.add_argument("--fmin", type=float, default=150.0)
    ap.add_argument("--fmax", type=float, default=4000.0)
    ap.add_argument("--n_harmonics", type=int, default=20)
    ap.add_argument(
        "--mic_noise",
        type=float,
        default=0.0,
        help="std (m) of Gaussian perturbation added to mic positions",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tfs = load_dregon_timeframes(args.data_dir, splits=[args.split], download=False)
    tf = next(t for t in tfs if t.tags["recording_id"] == args.recording)

    mic_pos = tf.global_data["mic_positions"].astype(np.float64)  # (8, 3)
    rotor_gt = tf.global_data["rotor_positions"].astype(np.float64)  # (4, 3)

    mic_pos_approx = mic_pos.copy()
    if args.mic_noise > 0:
        rng = np.random.default_rng(args.seed)
        mic_pos_approx = mic_pos + rng.normal(0.0, args.mic_noise, size=mic_pos.shape)

    audio, sr, rps = _get_window(tf, args.t0, args.dur)
    print(
        f"recording={args.recording}  window=[{args.t0:.1f}, {args.t0 + args.dur:.1f}]s  "
        f"audio={audio.shape}@{sr:.0f}Hz  rps∈[{rps.min():.1f},{rps.max():.1f}]Hz"
    )
    print(f"mic_noise={args.mic_noise} m\n")

    def run(mode_rps):
        return localize_rotors(
            audio,
            mic_pos_approx,
            sr,
            n_rotors=4,
            rps=rps if mode_rps else None,
            coarse_step=args.coarse_step,
            fmin=args.fmin,
            fmax=args.fmax,
            n_harmonics=args.n_harmonics,
            device=args.device,
        )

    def report(name, res):
        perm, errors = match_and_score(res.positions, rotor_gt)
        print(f"=== {name} ===")
        for k in range(res.positions.shape[0]):
            est = res.positions[k]
            gt = rotor_gt[perm[k]]
            print(
                f"  est {est.round(3)} -> gt#{perm[k]} {gt.round(3)}  err={errors[k] * 100:5.1f} cm"
            )
        print(
            f"  mean err = {errors.mean() * 100:.1f} cm   max err = {errors.max() * 100:.1f} cm\n"
        )

    if args.mode in ("audio", "both"):
        report("AUDIO-ONLY (top-K peaks)", run(False))
    if args.mode in ("rps", "both"):
        report("RPS-AIDED (per-rotor harmonic isolation)", run(True))

    print("ground-truth rotor positions (cm):")
    for i, g in enumerate(rotor_gt):
        print(f"  #{i}: {(g * 100).round(1)}")


if __name__ == "__main__":
    main()
