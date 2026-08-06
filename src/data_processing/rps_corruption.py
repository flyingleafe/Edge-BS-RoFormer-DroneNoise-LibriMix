"""Synthetic corruption of clean RPS tracks — training data for the
conditional RPS *refiner* (``simple_conv_v2_ckla_phaseonly_cond``).

Campaign context: classical refinement (VK refine, stage D) cannot pull a
track with 0.5–2 rev/s error toward the truth on real drone audio, although
the comb information is present at high precision (GT-init VK-refine locks to
0.028 MAE). The learned puller is trained on ``(audio, corrupt(GT)) -> GT``
pairs; at inference the conditioning is a real coarse track (blind-VK Viterbi
~0.7 err, or a neural predictor at ~1–2.5 err). This module makes the
corrupted conditioning.

Corruption model (per chunk, per rotor — all knobs config-overridable):

* smooth OU noise with ``sigma ~ U(0.1, 1.5)`` rev/s and correlation time
  ``tau ~ U(0.5, 2.0)`` s (stationary start);
* a constant offset ``~ U(-2.5, 2.5)`` rev/s with probability 0.7;
* with probability 0.15 a *pair-level* event: either swap two rotor rows, or
  set one rotor's conditioning to another rotor's values plus a constant
  ``U(-1, 1)`` offset (models DP twin-capture);
* zero-RPS spans of the ground truth stay exactly zero in the conditioning
  (a coarse tracker emits nothing when the motors are off), and the
  conditioning is clamped at 0 (RPS is non-negative).

Output-identity contract: the refiner's output row ``i`` corresponds to
conditioning row ``i`` and is trained with a plain (non-PIT) MSE, so
:meth:`RPSCorruption.__call__` also returns the ground truth **in
conditioning order** — for the row-*swap* event the returned GT rows are
swapped identically (the swapped conditioning row genuinely tracks the other
rotor). The twin-capture event deliberately keeps the identity GT: the
captured row sits near its own rotor's twin, and pulling it back to its OWN
rotor is exactly the refinement behaviour we want to learn. Row-permuting
the target is invisible to the PIT validation metrics (permutation-invariant
by construction), so the same Frame serves both the plain loss and the
standard ``conf/metrics/rps.yaml`` suite.

Determinism: corruption is a pure function of ``(seed, sample_id, rps)``
via :func:`data_processing.online_mixing.make_rng` — the same convention the
online mixer uses — so validation sees a FIXED corruption per sample and
epochs stay comparable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from data_processing.online_mixing import make_rng

__all__ = [
    "DREGON_TACH_REFRESH_HZ",
    "DREGON_TACH_STEP",
    "RPSCorruption",
    "corrupt_rps",
    "presmooth_track",
    "tachometer_corrupt",
]

#: DREGON's telemetry quantization step (rev/s) — the tachometer reports an
#: integer pulse count, so the label lives on a 0.269 rev/s lattice.
DREGON_TACH_STEP = 0.269
#: DREGON's telemetry refresh rate (Hz) — one new reading every ~20.1 ms,
#: held constant in between (zero-order hold).
DREGON_TACH_REFRESH_HZ = 49.7

#: |GT| at or below this is a zero-RPS span (motors off) — exact zeros in the
#: interpolated targets, but keep a small epsilon for float noise.
ZERO_EPS = 1e-6


# ---------------------------------------------------------------------------
# Tachometer label noise (the telemetry staircase)
# ---------------------------------------------------------------------------


def tachometer_corrupt(
    rps: np.ndarray,
    sample_rate: float,
    *,
    step: float = DREGON_TACH_STEP,
    refresh_hz: float = DREGON_TACH_REFRESH_HZ,
    scale: float = 1.0,
    t_start: float = 0.0,
) -> np.ndarray:
    """Apply the telemetry tachometer's measurement model to a true rotor track.

    The device of DREGON's flight controller, in the order it acts:

    1. **Constant scale.** ``scale`` multiplies the truth (DREGON's telemetry
       over-reports by 0.542 %, so ``scale = 0.99458`` reproduces the measured
       bias). A constant gain is an invertible reparameterization of the label
       and is expected to be *benign* for a conditioned model — it is separable
       from the staircase precisely because it is applied first and alone.
    2. **Refresh interval average.** The tachometer counts shaft pulses over
       one refresh interval, so its reading is the interval *mean*, not a point
       sample. Intervals are ``1 / refresh_hz`` seconds wide and are laid out on
       absolute time (``t_start`` places the window on that grid), so a
       non-integer number of audio samples per interval is handled exactly.
    3. **Quantization.** The pulse count is an integer, so the reading lands on
       the ``step`` rev/s lattice. Rounding is half-to-even via ``np.rint``.
    4. **Zero-order hold.** One reading is held for the whole interval.

    Steps 2-4 together are the *staircase*: a piecewise-constant label whose
    error is dominated by quantization (rms ``step / sqrt(12)`` = 0.078 rev/s
    at DREGON's step) rather than by the hold lag (the shaft's OU dynamics move
    ~0.04 rev/s in one 20 ms interval). The staircase is what scales with the
    harmonic index: a rate error ``e`` displaces harmonic ``k`` by ``k * e`` Hz,
    so 0.078 rev/s is 6.2 Hz at ``k = 80``.

    **No latency.** The reading is held over the *same* interval it averages,
    not the next one. A real device is causal and lags by one interval, but a
    constant 20 ms group delay is a second benign, learnable reparameterization
    and would confound the measurement of the staircase itself; the caller who
    wants it can shift the output.

    Args:
        rps: ``(..., T)`` true rotor speeds (rev/s) on a uniform audio-rate grid.
        sample_rate: grid rate of ``rps`` (Hz).
        step: quantization step (rev/s). ``<= 0`` disables quantization.
        refresh_hz: reading rate (Hz). ``<= 0`` disables the interval average
            and hold (the label then carries only the scale + quantization).
        scale: constant multiplicative bias applied before the device.
        t_start: absolute start time (s) of the window, so consecutive windows
            of one recording see a consistent refresh phase.

    Returns:
        The corrupted label, same shape and dtype-family as ``rps`` (float64).
    """
    arr = np.asarray(rps, dtype=np.float64) * float(scale)
    if arr.ndim == 0:
        raise ValueError("tachometer_corrupt expects an array with a time axis")
    n_t = arr.shape[-1]
    if n_t == 0:
        return arr

    if refresh_hz > 0.0:
        t = (np.arange(n_t, dtype=np.float64) / float(sample_rate)) + float(t_start)
        idx = np.floor(t * float(refresh_hz)).astype(np.int64)
        idx -= idx[0]  # local interval numbering, phase preserved by t_start
        n_int = int(idx[-1]) + 1
        flat = arr.reshape(-1, n_t)
        counts = np.bincount(idx, minlength=n_int).astype(np.float64)
        sums = np.stack([np.bincount(idx, weights=row, minlength=n_int) for row in flat])
        reading = sums / counts  # (N, n_int) interval means
        if step > 0.0:
            reading = np.rint(reading / float(step)) * float(step)
        out = reading[:, idx].reshape(arr.shape)  # zero-order hold
    else:
        out = np.rint(arr / float(step)) * float(step) if step > 0.0 else arr
    return np.ascontiguousarray(out)


def presmooth_track(
    rps: np.ndarray,
    sample_rate: float,
    *,
    cut_hz: float = 5.0,
) -> np.ndarray:
    """Low-pass a rotor track at ``cut_hz`` — the campaign's de-staircasing filter.

    Thin uniform-grid adapter over :func:`tracking.telemetry_refit.presmooth`,
    so "the smoothed telemetry" means the same array here as it does in the
    tracking campaign (detrend, whole-window brickwall, add the trend back).
    5 Hz keeps the shaft dynamics (DREGON free-flight OU modes have
    ``tau`` 0.6-1.0 s, i.e. bandwidth well under 1 Hz) and rejects the 49.7 Hz
    refresh staircase. ``cut_hz <= 0`` is the identity.

    Args:
        rps: ``(..., T)`` track on a uniform grid.
        sample_rate: grid rate (Hz).
        cut_hz: cutoff (Hz).

    Returns:
        The smoothed track, same shape, float64.
    """
    from tracking.telemetry_refit import presmooth

    arr = np.asarray(rps, dtype=np.float64)
    n_t = arr.shape[-1]
    ft = np.arange(n_t, dtype=np.float64) / float(sample_rate)
    flat = arr.reshape(-1, n_t)
    return np.ascontiguousarray(presmooth(flat, ft, float(cut_hz)).reshape(arr.shape))


def corrupt_rps(
    rps: np.ndarray,
    rng: np.random.Generator,
    *,
    frame_rate_hz: float = 16000.0 / 512.0,
    ou_sigma_min: float = 0.1,
    ou_sigma_max: float = 1.5,
    ou_tau_min_s: float = 0.5,
    ou_tau_max_s: float = 2.0,
    offset_prob: float = 0.7,
    offset_max: float = 2.5,
    pair_event_prob: float = 0.15,
    twin_jitter_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a clean ``(R, F)`` RPS track. Returns ``(cond, gt_aligned)``.

    ``cond`` is the corrupted conditioning track; ``gt_aligned`` is the input
    ground truth with its rows permuted to match the conditioning's rotor
    identity (non-identity only when the swap branch of the pair event fired
    — see module docstring). Both are float32 ``(R, F)``.

    All randomness comes from ``rng``; the draw order is fixed (pair event →
    per-rotor OU + offset) so a given generator state maps to exactly one
    corruption.
    """
    rps = np.asarray(rps, dtype=np.float32)
    if rps.ndim != 2:
        raise ValueError(f"corrupt_rps expects a (R, F) track, got shape {rps.shape}")
    n_rotors, n_frames = rps.shape
    gt = rps.copy()
    cond = rps.astype(np.float64).copy()

    # ── pair-level event: swap two rows, or twin-capture one row ──────────
    if n_rotors >= 2 and rng.random() < pair_event_prob:
        i, j = (int(x) for x in rng.choice(n_rotors, size=2, replace=False))
        if rng.random() < 0.5:
            # Row swap: conditioning row i tracks rotor j (and vice versa),
            # so the aligned GT swaps identically.
            cond[[i, j]] = cond[[j, i]]
            gt[[i, j]] = gt[[j, i]]
        else:
            # Twin capture: row i locks onto rotor j's track (plus a constant
            # offset). GT stays identity — the refiner must pull row i back
            # to its OWN rotor.
            cond[i] = cond[j] + rng.uniform(-twin_jitter_max, twin_jitter_max)

    # ── per-rotor smooth OU noise + constant offset ───────────────────────
    dt = 1.0 / float(frame_rate_hz)
    for r in range(n_rotors):
        sigma = float(rng.uniform(ou_sigma_min, ou_sigma_max))
        tau = float(rng.uniform(ou_tau_min_s, ou_tau_max_s))
        a = float(np.exp(-dt / tau))
        innov = rng.normal(0.0, sigma * float(np.sqrt(1.0 - a * a)), size=n_frames)
        innov[0] = rng.normal(0.0, sigma)  # stationary start
        from scipy.signal import lfilter

        ou = lfilter([1.0], [1.0, -a], innov)
        cond[r] += ou
        if rng.random() < offset_prob:
            cond[r] += rng.uniform(-offset_max, offset_max)

    # ── zero-span preservation + physical clamp ───────────────────────────
    # Mask by the ALIGNED GT: conditioning row i tracks gt_aligned row i, so
    # its off-spans are that row's zeros (in practice zero spans are global —
    # all motors off on the ground).
    cond[np.abs(gt) <= ZERO_EPS] = 0.0
    np.clip(cond, 0.0, None, out=cond)
    return cond.astype(np.float32), gt


class RPSCorruption:
    """Config-built, seeded corruption sampler for the dataset adapters.

    ``__call__(rps, sample_id)`` derives a fresh per-sample generator via
    ``make_rng(seed, sample_id)`` (the online mixer's convention) and applies
    :func:`corrupt_rps` — deterministic per ``(seed, sample_id)``,
    independent of worker process and iteration order.
    """

    def __init__(
        self,
        *,
        seed: int = 8451,
        frame_rate_hz: float = 16000.0 / 512.0,
        ou_sigma_min: float = 0.1,
        ou_sigma_max: float = 1.5,
        ou_tau_min_s: float = 0.5,
        ou_tau_max_s: float = 2.0,
        offset_prob: float = 0.7,
        offset_max: float = 2.5,
        pair_event_prob: float = 0.15,
        twin_jitter_max: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.params: dict[str, float] = {
            "frame_rate_hz": float(frame_rate_hz),
            "ou_sigma_min": float(ou_sigma_min),
            "ou_sigma_max": float(ou_sigma_max),
            "ou_tau_min_s": float(ou_tau_min_s),
            "ou_tau_max_s": float(ou_tau_max_s),
            "offset_prob": float(offset_prob),
            "offset_max": float(offset_max),
            "pair_event_prob": float(pair_event_prob),
            "twin_jitter_max": float(twin_jitter_max),
        }

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any] | None, *, frame_rate_hz: float | None = None
    ) -> RPSCorruption | None:
        """Build from a config mapping (``None``/empty -> no corruption).

        ``frame_rate_hz`` (the dataset's exact ``sr / hop``) overrides the
        default unless the config pins its own value.
        """
        if not cfg:
            return None
        kwargs = {str(k): v for k, v in dict(cfg).items()}
        if frame_rate_hz is not None:
            kwargs.setdefault("frame_rate_hz", frame_rate_hz)
        return cls(**kwargs)

    def __call__(self, rps: np.ndarray, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
        rng = make_rng(self.seed, int(sample_id))
        return corrupt_rps(rps, rng, **self.params)  # type: ignore[arg-type]
