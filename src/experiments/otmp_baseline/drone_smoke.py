"""Smoke test of the OT multi-pitch baseline on real 4-rotor drone audio.

Runs the estimator, with the paper's real-data parameters and only the grids
changed (:func:`~experiments.otmp_baseline.estimate.drone_config`), over clips
of the frozen ``DREGON-LM-V4-michaels-valid-full`` split, and scores the four
reported fundamentals against the four rotor-speed targets with per-frame
Hungarian (PIT) matching — the convention of
``results/m3cur_regime_probe/regime_probe.py``.

A rotor turning at ``f`` rev/s radiates a comb whose fundamental is ``f`` Hz,
so estimated pitch in Hz and target rotor speed in rev/s are the same number
and no conversion is needed.

Run it with::

    PYTHONPATH=src python -m experiments.otmp_baseline.drone_smoke --clips 2

This is a *smoke* test: paper defaults, no tuning against these clips.

Result of record, 800 outer iterations, channel 0, 16 windows per clip::

    [cruise]  sample_00001  18.2 s/frame  PIT-MAE 38.3 rev/s  PIT-RMSE 47.2
    [warm-up] sample_00008  16.1 s/frame  PIT-MAE 45.0 rev/s  PIT-RMSE 54.2

Targets run 34-62 rev/s, so those errors are the size of the quantity being
estimated: the method does not find the rotor speeds here. See
:func:`~experiments.otmp_baseline.estimate.drone_config` for why.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.otmp_baseline.estimate import OTMPConfig, drone_config, estimate_frame

__all__ = ["ClipResult", "pit_error", "run_clip", "window_targets"]

DEFAULT_SPLIT = "dload:DREGON-LM-V4-michaels-valid-full"


def window_targets(rps: NDArray, n_samples: int, cfg: OTMPConfig, hop_length: int) -> NDArray:
    """Average the ``(rotor, T_stft)`` targets over each analysis window.

    Returns ``(rotor, n_windows)``. The RPS target is sampled at the STFT rate
    (``sample_rate / hop_length``); an analysis window of ``cfg.frame_len``
    samples covers ``frame_len / hop_length`` of those.
    """
    hop = int(cfg.hop or cfg.frame_len)
    starts = range(0, max(n_samples - cfg.frame_len + 1, 0), hop)
    out = []
    for start in starts:
        lo = start // hop_length
        hi = max(lo + 1, (start + cfg.frame_len) // hop_length)
        out.append(rps[:, lo : min(hi, rps.shape[1])].mean(axis=1))
    return np.asarray(out).T


def regime_of(rps: NDArray) -> str:
    """Clip regime, by the rule of ``results/m3cur_regime_probe/regime_probe.py``.

    ``zero`` = rotors stopped, ``cruise`` = mid-flight, ``warm-up`` = anything
    between (warm-up plus the take-off / landing ramps).
    """
    if float(rps.max()) < 1.0:
        return "zero"
    return "cruise" if float(rps.mean()) >= 45.0 else "warm-up"


def pit_error(pred: NDArray, target: NDArray) -> NDArray:
    """Per-frame Hungarian-matched absolute error, ``(rotor, n_windows)``."""
    from scipy.optimize import linear_sum_assignment

    out = np.empty_like(target)
    for t in range(target.shape[1]):
        cost = np.abs(pred[:, None, t] - target[None, :, t])
        rows, cols = linear_sum_assignment(cost)
        out[:, t] = np.abs(pred[rows, t] - target[cols, t])
    return out


@dataclass
class ClipResult:
    """One clip's outcome."""

    recording_id: str
    regime: str
    pred: NDArray  # (K, n_windows)
    target: NDArray  # (rotor, n_windows)
    abs_err: NDArray  # (rotor, n_windows)
    seconds_per_frame: float

    @property
    def mae(self) -> float:
        return float(np.mean(self.abs_err))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.abs_err**2)))


def run_clip(
    frame, cfg: OTMPConfig, hop_length: int = 512, max_windows: int | None = None
) -> ClipResult:
    """Estimate every analysis window of one dataset Frame and score it."""
    audio = np.asarray(frame["mixture"].data, dtype=np.float64).ravel()
    rps = np.asarray(frame["rps"].data, dtype=np.float64)
    target = window_targets(rps, audio.size, cfg, hop_length)

    hop = int(cfg.hop or cfg.frame_len)
    starts = list(range(0, max(audio.size - cfg.frame_len + 1, 0), hop))[: target.shape[1]]
    if max_windows:
        starts = starts[:max_windows]
    pred = np.empty((cfg.n_pitches, len(starts)), dtype=np.float64)
    start_time = time.perf_counter()
    for i, offset in enumerate(starts):
        est = estimate_frame(audio[offset : offset + cfg.frame_len], cfg.sample_rate, cfg)
        pred[:, i] = est.pitches_hz
    elapsed = time.perf_counter() - start_time

    target = target[:, : len(starts)]
    return ClipResult(
        recording_id=str(dict(frame["meta"].items()).get("recording_id", "?")),
        regime=regime_of(np.asarray(frame["rps"].data, dtype=np.float64)),
        pred=pred,
        target=target,
        abs_err=pit_error(pred, target),
        seconds_per_frame=elapsed / max(len(starts), 1),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args(argv)

    from data_processing.frame_datasets import DregonLMFrameDataset

    over = (
        {}
        if args.max_iter is None
        else {"max_iter": args.max_iter, "debias_max_iter": args.max_iter}
    )
    cfg = drone_config(**over)

    dataset = DregonLMFrameDataset(
        data_dir=args.split,
        n_fft=2048,
        hop_length=512,
        sample_rate=16000,
        channel=args.channel,
    )
    # One cruise clip and one warm-up clip. Clips with the rotors stopped are
    # skipped — there is no rotor speed to estimate in them.
    picked: dict[str, int] = {}
    for idx in range(len(dataset)):
        regime = regime_of(np.asarray(dataset[idx]["rps"].data, dtype=np.float64))
        if regime == "zero":
            continue
        picked.setdefault(regime, idx)
        if len(picked) >= args.clips:
            break

    for regime, idx in picked.items():
        res = run_clip(dataset[idx], cfg, max_windows=args.max_windows)
        print(
            f"\n[{regime}] {res.recording_id}  {res.pred.shape[1]} windows  "
            f"{res.seconds_per_frame:.1f} s/frame\n"
            f"  PIT-MAE {res.mae:7.2f} rev/s   PIT-RMSE {res.rmse:7.2f}\n"
            f"  target  {np.round(res.target.mean(axis=1), 1)}\n"
            f"  pred    {np.round(np.sort(res.pred, axis=0).mean(axis=1), 1)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
