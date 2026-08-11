#!/usr/bin/env python3
"""Trajectory-fitness LANDSCAPE benchmark — ``docs/trajectory-fitness-design.md`` §3.

Six candidate measures are scored as LANDSCAPES over trajectories, not as
verdicts on one trajectory. Each entrant is a callable
``(audio, candidate trajectory) -> scalar``; the benchmark asks four questions
about the surface each one draws, and nothing about the number it returns.

The entrants
------------
=================  ========================================  ==============
key                what it is                                better
=================  ========================================  ==============
``fvk_fine``       F_VK, ``k_max`` = 80, ``bw_rps`` default  lower
``fvk_coarse``     F_VK, ``k_max`` = 5 (the coarse rung)     lower
``fvk_alias``      ``fvk_fine`` + ``alias_penalty`` charge   lower
``ridge``          ridge concentration, dB over local floor  HIGHER
``broadband``      out-of-DC envelope-power share (6c)       lower
``harmsum``        plain FFT harmonic sum (published bl.)    HIGHER
=================  ========================================  ==============

``fvk_fine`` and ``fvk_alias`` come out of ONE envelope solve — the charge is a
scoring term, not a different objective — so the pair costs what one costs.
``ridge`` and ``broadband`` likewise share one demodulation
(:func:`tracking.fitness.window_cells`). ``harmsum`` is implemented here
because it is the *baseline*, and a baseline that lives in ``src`` invites
someone to use it.

The four figures of merit
-------------------------
``G``
    global-optimum margin over the structured alias set (sub-multiples,
    scale, constant offset, the DREGON-like ×0.99317 bias, twin swap, time
    shift). Reported as ``wins`` plus the margin in the measure's own units and
    normalized by the measure's own dynamic range (truth → a 5 rev/s error).
``M``
    directional basin profiles ``F(truth + a d)`` for four canonical ``d``, on
    a log amplitude grid; Spearman between ``|a|`` and the score.
``GRA``
    gradient-sign ranking accuracy: at a random perturbation of size ``a``,
    does a further step of ``5 %`` make the score worse?
``CONT``
    continuation validity (F_VK only): the 1-D offset profile's argmin path
    along the ``k_max`` ladder, the ``bw_rps`` ladder and the joint schedule.

Test material
-------------
**Synthetic** (primary): a script-local comb synthesizer — two rotors on an OU
trajectory (:func:`data_processing.rps_synthesis.generate`, low-passed so the
frame grid represents the truth exactly), harmonics ``k = 1..80`` with ``1/k``
amplitudes, per-harmonic Wiener phase noise of variance ``∝ k²`` (the WP18
law), white noise at a chosen comb-to-noise ratio. Two geometries: a DREGON-like
TWIN pair 0.42 rev/s apart, and a well-separated pair.

**Real** (secondary): FLY124 cruise windows from the frozen prep cache
(:func:`tracking.protocols.load_prep_window`), sliced to a short segment; the
recalibrated telemetry is the quasi-truth.

Run
---
Smoke (about a minute)::

    python scripts/fvk_bench.py --quick --jobs 2 --out results/fvk_bench_smoke

Full synthetic grid (this is an omnirun job, not a laptop job)::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 32 --time 4h \\
        --env PYTHONPATH=src -- python scripts/fvk_bench.py --jobs 16

Analysis only (table + the two figures), over an existing output tree::

    python scripts/fvk_bench.py --analyze-only --out results/fvk_bench
"""

from __future__ import annotations

import os

# Cap the BLAS pools BEFORE numpy: process-level parallelism, the harness
# convention (utils.gridrun re-asserts it in every worker).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TRACKING_FFT_WORKERS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

# ---------------------------------------------------------------------------
# the fixed geometry of the benchmark

SR = 16000
FS_FRAME = 100.0  #: candidate trajectories live on this frame grid
FS_TRAJ = 250.0  #: the OU drive rate before the shaft-inertia low pass
TRAJ_FC_HZ = 8.0  #: shaft inertia — also what makes the truth frame-representable
K_SYNTH = 80  #: harmonics rendered per rotor
F_MAX = 7000.0  #: no measure models a line above this (0.45 * SR = 7200)
JITTER_K1 = 0.01  #: rad/sqrt(s) of Wiener phase noise at k = 1; variance ∝ k²

#: (rotor 0, rotor 1) mean rates. ``twin`` is the DREGON separation, 0.42 rev/s.
PAIRS: dict[str, tuple[float, float]] = {"twin": (68.0, 68.42), "sep": (68.0, 61.0)}
SNRS: tuple[float, ...] = (0.0, -10.0, -20.0)
DURS: tuple[float, ...] = (1.0, 4.0)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

MEASURES: tuple[str, ...] = (
    "fvk_fine",
    "fvk_coarse",
    "fvk_alias",
    "ridge",
    "broadband",
    "harmsum",
)
#: The two entrants where a LARGER number is a better fit.
HIGHER_BETTER: frozenset[str] = frozenset({"ridge", "harmsum"})

#: Weight of the alias/order counter-term in ``fvk_alias``. The raw charge
#: travels in every payload as well, so any other lambda is a re-read of the
#: same run rather than a re-run.
ALIAS_LAMBDA = 1.0

#: M: the amplitude grid, rev/s (RMS deviation from truth).
M_AMPS: tuple[float, ...] = tuple(np.round(np.logspace(-2, np.log10(5.0), 15), 5))
M_DIRECTIONS: tuple[str, ...] = ("offset", "scale", "smooth", "shift")
#: GRA: the perturbation sizes, rev/s, and the relative probe step.
GRA_AMPS: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
GRA_EPS = 0.05
#: CONT: the 1-D constant-offset profile grid, rev/s.
CONT_OFFSETS: np.ndarray = np.linspace(-1.5, 1.5, 61)
CONT_K_LADDER: tuple[int, ...] = (5, 10, 20, 40, 80)
CONT_BW_LADDER: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
#: The joint schedule actually proposed in design §2: coarse+wide -> fine+narrow.
CONT_SCHEDULE: tuple[tuple[int, float], ...] = (
    (5, 4.0),
    (10, 2.0),
    (20, 2.0),
    (40, 1.0),
    (80, 1.0),
)
#: The candidate whose distance from truth defines each measure's dynamic range.
RANGE_OFFSET = 5.0
#: The rotor-permutation entry — a degeneracy, reported but never a competitor.
SWAP_KEY = "swap_twin"

FLY124_CRUISE: tuple[str, ...] = ("FLY124__w02", "FLY124__w03", "FLY124__w04", "FLY124__w05")
REAL_SLICE_S = 2.0
REAL_MICS = 2


# ---------------------------------------------------------------------------
# the synthesizer (script-local on purpose: it is test material, not a model)


def _ou_trajectory(seed: int, means: tuple[float, float], span_s: float) -> np.ndarray:
    """``(2, M)`` low-passed OU rotor speeds at :data:`FS_TRAJ`, re-centred.

    A fixed 8 s draw is generated whatever the window length, so the 1 s case is
    a genuine prefix of the 4 s case at the same seed, and the zero-phase low
    pass has the samples it needs. ``TRAJ_FC_HZ`` is the shaft-inertia filter:
    without it the OU drive is white to 250 Hz and point-sampling the truth onto
    the 100 Hz frame grid aliases, which would put an error in the TRUTH.
    """
    from scipy.signal import filtfilt, firwin

    from data_processing.rps_synthesis import generate

    n_keep = int(round(span_s * FS_TRAJ))
    r = generate(8.0, FS_TRAJ, aggressiveness=1.0, rng=np.random.default_rng(1000 + seed))
    taps = firwin(255, TRAJ_FC_HZ / (FS_TRAJ / 2.0), window="hamming")
    r = np.asarray(filtfilt(taps, [1.0], r, axis=1))[:2, :n_keep]
    return np.stack([row - row.mean() + m for row, m in zip(r, means, strict=True)])


def render_comb(
    r_audio: np.ndarray,
    seed: int,
    *,
    snr_db: float,
    k_max: int = K_SYNTH,
    jitter: float = JITTER_K1,
    n_mic: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``(n_mic, T)`` audio from an audio-rate trajectory, plus its provenance.

    Harmonic ``k`` of rotor ``i`` is ``(1/k) cos(k phi_i + psi_ik + w_ik)`` with
    ``w_ik`` a Wiener process whose variance grows as ``k²`` — the WP18 law,
    drawn INDEPENDENTLY per harmonic because WP18 refuted the rank-one (pure
    shaft-jitter) form. White noise is added at ``comb_rms * 10^(-snr/20)``, so
    ``snr_db`` is the comb-to-noise ratio.
    """
    rng = np.random.default_rng(9000 + seed)
    n_t = int(r_audio.shape[-1])
    dt = 1.0 / SR
    comb = np.zeros(n_t)
    for i in range(r_audio.shape[0]):
        phi = 2.0 * np.pi * np.cumsum(r_audio[i]) / SR
        for k in range(1, k_max + 1):
            psi = float(rng.uniform(0.0, 2.0 * np.pi))
            walk = np.cumsum(rng.normal(0.0, jitter * k * math.sqrt(dt), n_t))
            comb += np.cos(k * phi + psi + walk) / k
    comb_rms = float(np.sqrt(np.mean(comb**2)))
    noise = rng.normal(0.0, comb_rms * 10.0 ** (-snr_db / 20.0), (n_mic, n_t))
    meta = {"comb_rms": comb_rms, "k_synth": k_max, "jitter_k1": jitter, "snr_db": snr_db}
    return comb[None, :] + noise, meta


def synth_case(pair: str, snr_db: float, dur: float, seed: int) -> dict[str, Any]:
    """One synthetic case: audio, the exact truth on the frame grid, frame times."""
    ft = np.arange(int(round(dur * FS_FRAME))) / FS_FRAME
    r_lo = _ou_trajectory(seed, PAIRS[pair], dur + 0.5)
    t_lo = np.arange(r_lo.shape[1]) / FS_TRAJ
    r_true = np.stack([np.interp(ft, t_lo, row) for row in r_lo])
    n_t = int(round(dur * SR))
    t_audio = np.arange(n_t) / SR
    # Synthesize from the FRAME-grid truth, so "truth" is exactly the array a
    # candidate is compared against and no interpolation error hides in it.
    r_audio = np.stack([np.interp(t_audio, ft, row) for row in r_true])
    audio, meta = render_comb(r_audio, seed, snr_db=snr_db)
    meta.update({"pair": pair, "dur": dur, "seed": seed, "means": list(PAIRS[pair])})
    return {"audio": audio, "r_true": r_true, "ft": ft, "dur": dur, "meta": meta}


def real_case(key: str, prep_dir: str | None) -> dict[str, Any]:
    """One FLY124 cruise window, sliced to :data:`REAL_SLICE_S` about its centre."""
    from tracking.protocols import load_prep_window

    win = load_prep_window(key, Path(prep_dir) if prep_dir else None)
    audio, ft, r = win["audio"], win["ft"], win["r"]
    n_t = int(round(REAL_SLICE_S * SR))
    i0 = max(0, (audio.shape[-1] - n_t) // 2)
    t0 = i0 / SR
    sel = (ft >= t0) & (ft < t0 + REAL_SLICE_S)
    return {
        "audio": np.ascontiguousarray(audio[:REAL_MICS, i0 : i0 + n_t], dtype=np.float64),
        "r_true": np.ascontiguousarray(r[:, sel], dtype=np.float64),
        "ft": np.ascontiguousarray(ft[sel] - t0, dtype=np.float64),
        "dur": REAL_SLICE_S,
        "meta": {"window": key, "regime": win["regime"], "n_mic": REAL_MICS},
    }


def build_case(params: dict[str, Any]) -> dict[str, Any]:
    if params["material"] == "real":
        return real_case(params["window"], params.get("prep_dir"))
    return synth_case(params["pair"], params["snr_db"], params["dur"], params["seed"])


# ---------------------------------------------------------------------------
# the six measures


class Ctx:
    """Everything pinned per WINDOW: the audio, the reference, the configs.

    The reference is the truth, and it is what pins the harmonic cap, the
    admission gates and the harmonic-sum's ``k`` set — so every candidate of a
    case is scored on the identical degrees of freedom and the comparison
    measures the trajectory rather than the cell count.
    """

    def __init__(self, case: dict[str, Any]):
        from tracking.fitness import FitnessConfig
        from tracking.fitness_vk import FVKConfig

        self.audio = np.ascontiguousarray(case["audio"], dtype=np.float64)
        self.r_true = np.ascontiguousarray(case["r_true"], dtype=np.float64)
        self.ft = np.ascontiguousarray(case["ft"], dtype=np.float64)
        self.dur = float(case["dur"])
        self.fine = FVKConfig(
            sr=SR, k_min=1, k_max=80, f_max=F_MAX, alias_penalty=ALIAS_LAMBDA, max_channels=8
        )
        self.coarse = FVKConfig(sr=SR, k_min=1, k_max=5, f_max=F_MAX, max_channels=8)
        # The two ``fitness`` components want OPPOSITE block lengths and giving
        # them one config would rig the comparison. ``broadband`` is a share of
        # a block's envelope spectrum and keeps the phase-6c count of blocks;
        # ``ridge`` reads a DC line against an annulus, so its floor region only
        # exists once a block resolves the interferer — measured here at 1 s,
        # 8 blocks leaves it 0-9 cells and 1 block leaves it 40. So the ridge
        # gets blocks of the ~2 s the phase-6d calibration used, and the count
        # each component actually scored travels in every payload.
        self.fit_bb = FitnessConfig(
            sr=SR,
            f_max=F_MAX,
            edge_trim_s=0.05,
            n_blocks=int(np.clip(round(2.0 * self.dur + 2.0), 2, 8)),
        )
        self.fit_ridge = FitnessConfig(
            sr=SR, f_max=F_MAX, edge_trim_s=0.05, n_blocks=max(1, int(round(self.dur / 2.0)))
        )
        rate_ref = np.mean(self.r_true, axis=1)
        self.harm_ks = [
            np.arange(1, int(min(K_SYNTH, math.floor(F_MAX / max(float(rr), 1e-6)))) + 1)
            for rr in rate_ref
        ]
        self._spec: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # -- the harmonic-sum baseline ------------------------------------------

    def _spectrogram(self, n_fft: int = 4096, hop: int = 2048) -> tuple[Any, Any, Any]:
        """``(power (F, B), bin freqs, frame centre times)`` — Hann, computed once."""
        if self._spec is None:
            x = self.audio[0]
            win = np.hanning(n_fft)
            starts = list(range(0, max(len(x) - n_fft, 0) + 1, hop)) or [0]
            frames = np.stack(
                [np.pad(x[s : s + n_fft], (0, max(0, n_fft - len(x))))[:n_fft] for s in starts]
            )
            power = np.abs(np.fft.rfft(frames * win, axis=-1)) ** 2
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
            times = (np.asarray(starts, dtype=np.float64) + n_fft / 2.0) / SR
            self._spec = (power, freqs, times)
        return self._spec

    def harmonic_sum(self, r_cand: np.ndarray, k_mult: int = 1) -> float:
        """Sum of ``|X(k f0)|²`` over the pinned ``k`` set, as a share of the total.

        The published baseline, generalized to a trajectory the only way it
        generalizes: per Hann frame, read each rotor's comb at that frame's rate.
        Lines above Nyquist contribute zero rather than folding, which is the
        charitable reading — folding would only sharpen its octave failure.
        ``k_mult`` extends the harmonic set, which is the ONLY thing that makes
        a sub-multiple candidate the nested model of design §1 Fact 2.
        """
        power, freqs, times = self._spectrogram()
        total = float(power.sum())
        acc = 0.0
        for rot in range(r_cand.shape[0]):
            ks = np.arange(1, int(len(self.harm_ks[rot]) * k_mult) + 1)
            rates = np.interp(times, self.ft, r_cand[rot])
            lines = np.outer(rates, ks)  # (B, K)
            ok = (lines > 0.0) & (lines < freqs[-1])
            for b in range(power.shape[0]):
                acc += float(np.interp(lines[b][ok[b]], freqs, power[b]).sum())
        return acc / max(total, 1e-300)

    # -- the whole entrant field at one candidate ---------------------------

    def _cells(self, r_cand: np.ndarray, cfg: Any) -> Any:
        from tracking.fitness import Holdout, score_cells, window_cells

        cells = window_cells(self.audio, self.ft, r_cand, self.r_true, cfg=cfg)
        return score_cells(cells, Holdout.none(), cfg=cfg)

    def evaluate(self, r_cand: np.ndarray) -> dict[str, float]:
        """Every measure at one candidate. Two VK solves, two demods, one FFT."""
        from tracking.fitness_vk import fvk_score

        out: dict[str, float] = {}
        s = fvk_score(self.audio, SR, r_cand, self.ft, self.fine, reference=self.r_true)
        out["fvk_fine"] = float(s["residual"]) / max(float(s["energy"]), 1e-300)
        out["fvk_alias"] = float(s["objective"])
        out["alias_charge"] = float(s["alias_charge"])
        out["k_hi"] = float(s["k_hi"])
        out["n_cells_fvk_fine"] = out["n_cells_fvk_alias"] = float(s["n_cells"])
        c = fvk_score(self.audio, SR, r_cand, self.ft, self.coarse, reference=self.r_true)
        out["fvk_coarse"] = float(c["objective"])
        out["n_cells_fvk_coarse"] = float(c["n_cells"])
        out["n_cells_harmsum"] = float(sum(len(k) for k in self.harm_ks))
        for key, cfg in (("broadband", self.fit_bb), ("ridge", self.fit_ridge)):
            try:  # a gate that empties is a RESULT, not a crash
                sc = self._cells(r_cand, cfg)
                out[key] = float(sc.broadband if key == "broadband" else sc.ridge)
                out[f"n_cells_{key}"] = float(
                    sc.n_cells if key == "broadband" else sc.n_cells_ridge
                )
            except Exception:  # noqa: BLE001
                out[key] = float("nan")
                out[f"n_cells_{key}"] = 0.0
        out["harmsum"] = self.harmonic_sum(r_cand)
        return out

    def nested_submultiple(self, truth: dict[str, float]) -> dict[str, Any]:
        """The Walmsley move: halve the rate AND double the harmonic budget.

        At a FIXED harmonic cap a sub-multiple is not the nested model — it
        covers only half the band and loses the top of the true comb, which is
        why F_VK rejects ``scale_0.5`` in the fixed-DOF alias set. The nesting
        argument of Fact 2 needs ``f0/m`` with ``mK`` harmonics, and that is
        this: the two models then span the same band and the sub-multiple's
        basis CONTAINS the true one. Only the order penalty can break it.
        Reported for the two energy-sum families; the ``fitness`` components
        carry their ``k_max`` in a config and are left out.
        """
        from tracking.fitness_vk import fvk_score

        half = self.r_true * 0.5
        cap = int(truth["k_hi"])
        s = fvk_score(self.audio, SR, half, self.ft, self.fine, reference=self.r_true, k_hi=2 * cap)
        fine = float(s["residual"]) / max(float(s["energy"]), 1e-300)
        return {
            "k_hi": 2 * cap,
            "alias_charge": float(s["alias_charge"]),
            "d_fvk_fine": fine - truth["fvk_fine"],
            "d_fvk_alias": float(s["objective"]) - truth["fvk_alias"],
            "d_harmsum": -(self.harmonic_sum(half, k_mult=2) - truth["harmsum"]),
        }

    def fvk_at(self, r_cand: np.ndarray, k_max: int, bw_rps: float) -> float:
        """One F_VK reading at an arbitrary continuation rung."""
        from dataclasses import replace

        from tracking.fitness_vk import fvk_score

        cfg = replace(self.fine, k_max=int(k_max), bw_rps=float(bw_rps), alias_penalty=0.0)
        s = fvk_score(self.audio, SR, r_cand, self.ft, cfg, reference=self.r_true)
        return float(s["residual"]) / max(float(s["energy"]), 1e-300)


def worse(name: str, value: float) -> float:
    """The score re-oriented so that LARGER always means a worse fit."""
    if not np.isfinite(value):
        return float("nan")
    return -value if name in HIGHER_BETTER else value


# ---------------------------------------------------------------------------
# candidate constructions


def time_shift(r: np.ndarray, ft: np.ndarray, dt: float) -> np.ndarray:
    """``r(t + dt)`` on the same frame grid, clamped at the window edges."""
    return np.stack([np.interp(ft + dt, ft, row) for row in r])


def smooth_noise(
    rng: np.random.Generator, shape: tuple[int, ...], cut_hz: float = 2.0
) -> np.ndarray:
    """Unit-RMS low-passed Gaussian field on the frame grid, one row per rotor."""
    n = int(shape[-1])
    x = rng.normal(0.0, 1.0, tuple(int(v) for v in shape))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS_FRAME)
    spec = np.fft.rfft(x, axis=-1) * np.exp(-((freqs / max(cut_hz, 1e-6)) ** 2))[None, :]
    y = np.fft.irfft(spec, n=n, axis=-1)
    return y / max(float(np.sqrt(np.mean(y**2))), 1e-12)


def alias_set(r_true: np.ndarray, ft: np.ndarray) -> dict[str, np.ndarray]:
    """The structured alias set of design §3 FOM 1.

    Sub-multiples and multiples, the constant offsets a tracker actually makes,
    the DREGON-like ×0.99317 scale bias, the twin swap (the pair of rotors the
    audio cannot separate) and a ±20 ms time shift.

    ``swap_twin`` is a DEGENERACY, not an alias, and :data:`SWAP_KEY` keeps it
    out of the G competition: exchanging two rows of the trajectory leaves the
    SET of carriers untouched, so any permutation-invariant measure returns the
    identical number by construction (measured here at -0.0 for five of the six
    entrants). Rotor identity is certified by the residual pairing, never by the
    fit — the same rule ``tracking.fitness`` states. Scoring it as an alias
    would make every entrant fail G for a reason that has nothing to do with
    its landscape.
    """
    out: dict[str, np.ndarray] = {}
    for s in (0.5, 2.0 / 3.0, 1.5, 2.0):
        out[f"scale_{s:.4g}"] = r_true * s
    for o in (0.1, -0.1, 0.42, -0.42, 1.0, -1.0, 2.0, -2.0):
        out[f"off_{o:+g}"] = r_true + o
    out["scale_dregon"] = r_true * 0.99317
    if r_true.shape[0] >= 2:
        means = r_true.mean(axis=1)
        gap = np.abs(means[:, None] - means[None, :]) + np.eye(len(means)) * 1e9
        i, j = np.unravel_index(int(np.argmin(gap)), gap.shape)
        sw = r_true.copy()
        sw[[i, j]] = sw[[j, i]]
        out[SWAP_KEY] = sw
    for dt in (0.02, -0.02):
        out[f"shift_{dt:+g}"] = time_shift(r_true, ft, dt)
    return out


def direction_candidate(
    kind: str, r_true: np.ndarray, ft: np.ndarray, amp: float, field: np.ndarray, dur: float
) -> np.ndarray | None:
    """``truth + a d`` for one canonical direction; ``None`` when ``a`` is unusable.

    ``amp`` is always in rev/s: for ``scale`` it is the rate error at the mean
    rate, for ``shift`` it is the rate error the shift produces at the
    trajectory's own RMS slope (a shift longer than 40 % of the window is not a
    perturbation of that window, and returns ``None``).
    """
    if kind == "offset":
        return r_true + amp
    if kind == "scale":
        return r_true * (1.0 + amp / max(float(np.mean(r_true)), 1e-9))
    if kind == "smooth":
        return r_true + amp * field
    if kind == "shift":
        slope = float(np.sqrt(np.mean(np.gradient(r_true, axis=1) * FS_FRAME) ** 2))
        slope = max(
            slope, float(np.sqrt(np.mean((np.gradient(r_true, axis=1) * FS_FRAME) ** 2))), 1e-6
        )
        tau = amp / slope
        return None if abs(tau) > 0.4 * dur else time_shift(r_true, ft, tau)
    raise ValueError(f"unknown direction {kind!r}")


# ---------------------------------------------------------------------------
# the four figures of merit


def fom_global(ctx: Ctx) -> dict[str, Any]:
    """G — does truth win against the alias set, and by how much?"""
    truth = ctx.evaluate(ctx.r_true)
    far = ctx.evaluate(ctx.r_true + RANGE_OFFSET)
    aliases = {name: ctx.evaluate(r) for name, r in alias_set(ctx.r_true, ctx.ft).items()}
    rows: dict[str, Any] = {}
    for m in MEASURES:
        t = worse(m, truth[m])
        rng_ = abs(worse(m, far[m]) - t)
        vals = {k: worse(m, v[m]) for k, v in aliases.items()}
        finite = {k: v for k, v in vals.items() if np.isfinite(v) and k != SWAP_KEY}
        if not np.isfinite(t) or not finite:
            rows[m] = {"wins": None, "margin": None, "margin_norm": None, "best_alias": None}
            continue
        best = min(finite, key=lambda k: finite[k])
        margin = finite[best] - t
        rows[m] = {
            "truth": t,
            "range": rng_ if np.isfinite(rng_) else None,
            "wins": bool(margin > 0),
            "margin": margin,
            "margin_norm": (margin / rng_) if np.isfinite(rng_) and rng_ > 0 else None,
            "best_alias": best,
            "swap_delta": vals.get(SWAP_KEY, float("nan")) - t,
            "aliases": {k: (v - t) for k, v in vals.items()},
        }
    return {
        "truth_raw": dict(truth),
        "far_raw": dict(far),
        "nested": ctx.nested_submultiple(truth),
        "measures": rows,
    }


def fom_profiles(ctx: Ctx, seed: int) -> dict[str, Any]:
    """M — directional basin profiles and the distance/score Spearman."""
    from scipy.stats import spearmanr

    field = smooth_noise(np.random.default_rng(4000 + seed), ctx.r_true.shape)
    truth = ctx.evaluate(ctx.r_true)
    out: dict[str, Any] = {"truth_raw": truth, "amps": list(map(float, M_AMPS)), "directions": {}}
    for kind in M_DIRECTIONS:
        amps: list[float] = []
        curves: dict[str, list[float]] = {m: [] for m in MEASURES}
        for a in M_AMPS:
            cand = direction_candidate(kind, ctx.r_true, ctx.ft, float(a), field, ctx.dur)
            if cand is None:
                continue
            vals = ctx.evaluate(cand)
            amps.append(float(a))
            for m in MEASURES:
                curves[m].append(worse(m, vals[m]) - worse(m, truth[m]))
        stats: dict[str, Any] = {}
        for m in MEASURES:
            y = np.asarray(curves[m], dtype=np.float64)
            ok = np.isfinite(y)
            rho = (
                float(np.asarray(spearmanr(np.asarray(amps)[ok], y[ok]))[0])
                if int(ok.sum()) >= 3
                else float("nan")
            )
            mono = 0.0
            for i in range(1, int(ok.sum())):
                if y[ok][i] >= y[ok][i - 1] - 1e-12:
                    mono = float(np.asarray(amps)[ok][i])
                else:
                    break
            stats[m] = {
                "rho": None if not np.isfinite(rho) else rho,
                "monotone_upto": mono,
                "curve": [None if not np.isfinite(v) else float(v) for v in y],
            }
        out["directions"][kind] = {"amps": amps, "measures": stats}
    return out


def fom_gra(ctx: Ctx, seed: int, n_trials: int) -> dict[str, Any]:
    """GRA — at a random error of size ``a``, does a further 5 % step read worse?"""
    rng = np.random.default_rng(7000 + seed)
    hits = {a: {m: 0 for m in MEASURES} for a in GRA_AMPS}
    used = {a: {m: 0 for m in MEASURES} for a in GRA_AMPS}
    for _ in range(n_trials):
        # One random direction per trial: a constant part plus a smooth part,
        # which is the corruption a tracker actually makes (a bias plus wander).
        const = rng.normal(0.0, 1.0, (ctx.r_true.shape[0], 1)) * np.ones((1, ctx.r_true.shape[1]))
        delta = rng.uniform(0.3, 1.0) * const + rng.uniform(0.3, 1.0) * smooth_noise(
            rng, ctx.r_true.shape
        )
        delta = delta / max(float(np.sqrt(np.mean(delta**2))), 1e-12)
        for a in GRA_AMPS:
            near = ctx.evaluate(ctx.r_true + a * delta)
            far = ctx.evaluate(ctx.r_true + a * (1.0 + GRA_EPS) * delta)
            for m in MEASURES:
                dn, df = worse(m, near[m]), worse(m, far[m])
                if not (np.isfinite(dn) and np.isfinite(df)):
                    continue
                used[a][m] += 1
                hits[a][m] += int(df > dn)
    return {
        "n_trials": n_trials,
        "gra": {
            str(a): {m: (hits[a][m] / used[a][m] if used[a][m] else None) for m in MEASURES}
            for a in GRA_AMPS
        },
        "n_used": {str(a): {m: used[a][m] for m in MEASURES} for a in GRA_AMPS},
    }


def _profile_stats(offsets: np.ndarray, vals: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(vals)
    if int(ok.sum()) < 3:
        return {"argmin": None, "n_local_min": None, "curve": None}
    v = np.where(ok, vals, np.inf)
    n_local = int(sum(1 for i in range(1, len(v) - 1) if v[i] < v[i - 1] and v[i] < v[i + 1]))
    return {
        "argmin": float(offsets[int(np.argmin(v))]),
        "n_local_min": n_local,
        "curve": [None if not np.isfinite(x) else float(x) for x in vals],
    }


def fom_continuation(ctx: Ctx) -> dict[str, Any]:
    """CONT — is the argmin path along each continuation ladder continuous?"""
    cands = [ctx.r_true + o for o in CONT_OFFSETS]
    out: dict[str, Any] = {"offsets": [float(o) for o in CONT_OFFSETS], "ladders": {}}
    ladders = {
        "k_ladder": [(k, 1.0) for k in CONT_K_LADDER],
        "bw_ladder": [(80, bw) for bw in CONT_BW_LADDER],
        "schedule": list(CONT_SCHEDULE),
    }
    for name, rungs in ladders.items():
        recs = []
        for k, bw in rungs:
            vals = np.array([ctx.fvk_at(c, k, bw) for c in cands])
            rec = _profile_stats(CONT_OFFSETS, vals)
            rec.update({"k_max": int(k), "bw_rps": float(bw)})
            recs.append(rec)
        argmins = [r["argmin"] for r in recs if r["argmin"] is not None]
        out["ladders"][name] = {
            "rungs": recs,
            "argmin_path": argmins,
            "max_jump": float(np.max(np.abs(np.diff(argmins)))) if len(argmins) > 1 else None,
            "final_argmin": argmins[-1] if argmins else None,
        }
    return out


# ---------------------------------------------------------------------------
# the gridrun worker


def worker(unit: Unit) -> dict[str, Any]:
    """One (case, figure of merit) unit — the restartable quantum of the bench."""
    p = dict(unit.params)
    tic = time.perf_counter()
    case = build_case(p)
    ctx = Ctx(case)
    fom = p["fom"]
    if fom == "G":
        payload: dict[str, Any] = fom_global(ctx)
    elif fom == "M":
        payload = fom_profiles(ctx, int(p.get("seed", 0)) + 13 * len(str(p.get("window", ""))))
    elif fom == "GRA":
        payload = fom_gra(
            ctx, int(p.get("seed", 0)) + 29 * len(str(p.get("window", ""))), int(p["n_trials"])
        )
    elif fom == "CONT":
        payload = fom_continuation(ctx)
    else:
        raise ValueError(f"unknown fom {fom!r}")
    payload.update(
        {
            "uid": unit.uid,
            "fom": fom,
            "case": {k: v for k, v in p.items() if k not in ("fom", "prep_dir", "n_trials")},
            "case_meta": {k: v for k, v in case["meta"].items() if k != "comb_rms"},
            "wall_s": round(time.perf_counter() - tic, 2),
        }
    )
    return payload


# ---------------------------------------------------------------------------
# unit construction


def synth_units(args: argparse.Namespace) -> list[Unit]:
    pairs = args.pairs.split(",")
    snrs = [float(v) for v in args.snrs.split(",")]
    durs = [float(v) for v in args.durs.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    foms = args.foms.split(",")
    units: list[Unit] = []
    for pair in pairs:
        for snr in snrs:
            for dur in durs:
                for seed in seeds:
                    base = {
                        "material": "synth",
                        "pair": pair,
                        "snr_db": snr,
                        "dur": dur,
                        "seed": seed,
                    }
                    tag = f"syn_{pair}_snr{snr:g}_d{dur:g}_s{seed}"
                    for fom in foms:
                        # GRA is the expensive figure (2 evaluations per trial per
                        # amplitude); it runs on the short windows only, which is
                        # where the landscape question is hardest anyway.
                        if fom == "GRA" and dur != min(durs):
                            continue
                        # CONT is F_VK-only and diagnostic: one SNR, two seeds.
                        if fom == "CONT" and (snr != -10.0 or seed >= 2 or dur != min(durs)):
                            continue
                        params = dict(base, fom=fom)
                        if fom == "GRA":
                            params["n_trials"] = args.gra_trials
                        units.append(Unit(uid=f"{tag}__{fom}", params=params))
    return units


def real_units(args: argparse.Namespace) -> list[Unit]:
    foms = [f for f in args.foms.split(",") if f in ("G", "M", "GRA")]
    units: list[Unit] = []
    for key in FLY124_CRUISE[: args.real_windows]:
        for fom in foms:
            params: dict[str, Any] = {"material": "real", "window": key, "fom": fom}
            if args.prep_dir:
                params["prep_dir"] = args.prep_dir
            if fom == "GRA":
                params["n_trials"] = max(4, args.gra_trials // 3)
            units.append(Unit(uid=f"real_{key}__{fom}", params=params))
    return units


def build_preps(dest: Path) -> str:
    """Materialize the FLY124 prep windows from the pinned ``beatvk-valid-raw``.

    A cluster worktree has no pulled prep cache, so the real material has to be
    rebuilt there. This is exactly what ``scripts/rps_eval.py --refine`` does;
    the builder lives in ``beatvk_vk_arms`` and is called, never copied.
    """
    import beatvk_vk_arms as bva

    dest.mkdir(parents=True, exist_ok=True)
    bva.build_preps(dest, {"FLY124": [2, 3, 4, 5]}, None, "dload:DREGON")
    return str(bva.prep_dir(dest))


# ---------------------------------------------------------------------------
# analysis: the FOM table and the two figures

_PALETTE = {
    "fvk_fine": "#1f77b4",
    "fvk_coarse": "#9ecae1",
    "fvk_alias": "#08306b",
    "ridge": "#d62728",
    "broadband": "#ff7f0e",
    "harmsum": "#2ca02c",
}
_INK = "#222222"


def load_rows(out_dir: Path) -> list[dict[str, Any]]:
    raw = out_dir / "raw"
    return [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))] if raw.is_dir() else []


def _mean(vals: list[Any]) -> float | None:
    v = np.asarray([x for x in vals if isinstance(x, (int, float))], dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool every unit into the measures × FOM table the paper quotes."""
    out: dict[str, Any] = {"n_units": len(rows), "by_material": {}}
    for material in ("synth", "real"):
        sel = [r for r in rows if r.get("case", {}).get("material") == material]
        if not sel:
            continue
        table: dict[str, Any] = {m: {} for m in MEASURES}
        g_rows = [r for r in sel if r["fom"] == "G"]
        for m in MEASURES:
            wins = [
                r["measures"][m]["wins"] for r in g_rows if r["measures"][m]["wins"] is not None
            ]
            table[m]["G_win_rate"] = (sum(wins) / len(wins)) if wins else None
            table[m]["G_margin_norm"] = _mean([r["measures"][m]["margin_norm"] for r in g_rows])
            table[m]["G_cells"] = _mean(
                [
                    r["truth_raw"].get(f"n_cells_{m}")
                    for r in g_rows
                    if f"n_cells_{m}" in r["truth_raw"]
                ]
            )
            table[m]["G_swap_delta"] = _mean([r["measures"][m].get("swap_delta") for r in g_rows])
            names = [
                r["measures"][m]["best_alias"] for r in g_rows if r["measures"][m]["best_alias"]
            ]
            table[m]["G_toughest_alias"] = max(set(names), key=names.count) if names else None
        # The nesting test of Fact 2: positive = truth still wins when the
        # sub-multiple is given the harmonic budget that makes it NESTED.
        nested = [r["nested"] for r in g_rows if "nested" in r]
        out.setdefault("nested_submultiple", {})[material] = {
            key: {
                "mean": _mean([n[key] for n in nested]),
                "truth_wins_frac": (
                    sum(1 for n in nested if isinstance(n[key], (int, float)) and n[key] > 0)
                    / len(nested)
                )
                if nested
                else None,
            }
            for key in ("d_fvk_fine", "d_fvk_alias", "d_harmsum")
        }
        m_rows = [r for r in sel if r["fom"] == "M"]
        for m in MEASURES:
            per_dir: dict[str, Any] = {}
            for d in M_DIRECTIONS:
                per_dir[d] = {
                    "rho": _mean(
                        [
                            r["directions"][d]["measures"][m]["rho"]
                            for r in m_rows
                            if d in r["directions"]
                        ]
                    ),
                    "monotone_upto": _mean(
                        [
                            r["directions"][d]["measures"][m]["monotone_upto"]
                            for r in m_rows
                            if d in r["directions"]
                        ]
                    ),
                }
            table[m]["M"] = per_dir
            table[m]["M_rho_mean"] = _mean([v["rho"] for v in per_dir.values()])
            table[m]["M_monotone_mean"] = _mean([v["monotone_upto"] for v in per_dir.values()])
        gra_rows = [r for r in sel if r["fom"] == "GRA"]
        for m in MEASURES:
            table[m]["GRA"] = {
                str(a): _mean([r["gra"][str(a)][m] for r in gra_rows]) for a in GRA_AMPS
            }
        cont_rows = [r for r in sel if r["fom"] == "CONT"]
        cont: dict[str, Any] = {}
        for name in ("k_ladder", "bw_ladder", "schedule"):
            cont[name] = {
                "max_jump": _mean(
                    [r["ladders"][name]["max_jump"] for r in cont_rows if name in r["ladders"]]
                ),
                "final_argmin_abs": _mean(
                    [
                        abs(r["ladders"][name]["final_argmin"])
                        for r in cont_rows
                        if name in r["ladders"] and r["ladders"][name]["final_argmin"] is not None
                    ]
                ),
                "n_local_min_coarse": _mean(
                    [
                        r["ladders"][name]["rungs"][0]["n_local_min"]
                        for r in cont_rows
                        if name in r["ladders"] and r["ladders"][name]["rungs"]
                    ]
                ),
                "n_local_min_fine": _mean(
                    [
                        r["ladders"][name]["rungs"][-1]["n_local_min"]
                        for r in cont_rows
                        if name in r["ladders"] and r["ladders"][name]["rungs"]
                    ]
                ),
            }
        out["by_material"][material] = {
            "n_units": len(sel),
            "n_G": len(g_rows),
            "n_M": len(m_rows),
            "n_GRA": len(gra_rows),
            "n_CONT": len(cont_rows),
            "measures": table,
            "continuation": cont,
        }
    return out


def markdown_table(summary: dict[str, Any], material: str = "synth") -> str:
    blk = summary.get("by_material", {}).get(material)
    if not blk:
        return f"(no {material} units)"
    hdr = (
        "| measure | better | cells | G win rate | G margin (norm) | toughest alias | "
        "M rho (mean) | M monotone to (rev/s) | GRA 0.03 | GRA 0.1 | GRA 0.3 | GRA 1.0 |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]

    def f(v: Any, nd: int = 3) -> str:
        return "—" if v is None else f"{v:.{nd}f}"

    for m in MEASURES:
        t = blk["measures"][m]
        gra = t.get("GRA", {})
        lines.append(
            f"| `{m}` | {'higher' if m in HIGHER_BETTER else 'lower'} | "
            f"{f(t.get('G_cells'), 0)} | "
            f"{f(t['G_win_rate'], 2)} | {f(t['G_margin_norm'])} | "
            f"{t.get('G_toughest_alias') or '—'} | {f(t['M_rho_mean'])} | "
            f"{f(t['M_monotone_mean'])} | "
            + " | ".join(f(gra.get(str(a)), 2) for a in GRA_AMPS)
            + " |"
        )
    return "\n".join(lines)


def _style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": "#888888",
            "axes.labelcolor": _INK,
            "text.color": _INK,
            "xtick.color": _INK,
            "ytick.color": _INK,
        }
    )
    return plt


def fig_profiles(
    rows: list[dict[str, Any]], dest: Path, snr: float = -10.0, dur: float = 1.0
) -> Path | None:
    """(a) The M panel: directional basin profiles, one row per rotor geometry."""
    plt = _style()
    pairs = ["twin", "sep"]
    sel = [
        r
        for r in rows
        if r["fom"] == "M"
        and r["case"].get("material") == "synth"
        and float(r["case"].get("snr_db", 0)) == snr
        and float(r["case"].get("dur", 0)) == dur
    ]
    if not sel:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.6), sharex=True)
    for ri, pair in enumerate(pairs):
        sub = [r for r in sel if r["case"].get("pair") == pair]
        for ci, d in enumerate(M_DIRECTIONS):
            ax = axes[ri, ci]
            for m in MEASURES:
                amps, curves = None, []
                for r in sub:
                    if d not in r["directions"]:
                        continue
                    amps = np.asarray(r["directions"][d]["amps"], dtype=np.float64)
                    c = r["directions"][d]["measures"][m]["curve"]
                    curves.append([np.nan if v is None else v for v in c])
                if amps is None or not curves:
                    continue
                arr = np.asarray(curves, dtype=np.float64)
                med = np.nanmedian(arr, axis=0)
                scale = np.nanmax(np.abs(med))
                if not np.isfinite(scale) or scale <= 0:
                    continue
                ax.plot(amps, med / scale, color=_PALETTE[m], lw=1.8, label=m, marker="o", ms=2.6)
            ax.set_xscale("log")
            ax.set_xlim(min(M_AMPS) * 0.7, max(M_AMPS) * 1.4)
            ax.axhline(0.0, color="#bbbbbb", lw=0.8)
            ax.set_ylim(-1.15, 1.15)
            if ri == 0:
                ax.set_title(d, fontsize=12)
            if ci == 0:
                ax.set_ylabel(f"{pair} pair\nworseness (own max = 1)")
            if ri == 1:
                ax.set_xlabel("perturbation |a|, rev/s")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        f"Directional basin profiles — synthetic, SNR {snr:g} dB, {dur:g} s window "
        "(median over seeds; rising = the measure knows it is wrong)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = dest / "fig_m_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_gra(rows: list[dict[str, Any]], dest: Path) -> Path | None:
    """(b) GRA against perturbation amplitude, per geometry and SNR."""
    plt = _style()
    sel = [r for r in rows if r["fom"] == "GRA" and r["case"].get("material") == "synth"]
    if not sel:
        return None
    pairs, snrs = ["twin", "sep"], sorted({float(r["case"]["snr_db"]) for r in sel}, reverse=True)
    fig, axes = plt.subplots(
        2, len(snrs), figsize=(4.4 * len(snrs), 6.6), sharey=True, squeeze=False
    )
    for ri, pair in enumerate(pairs):
        for ci, snr in enumerate(snrs):
            ax = axes[ri][ci]
            sub = [
                r
                for r in sel
                if r["case"].get("pair") == pair and float(r["case"]["snr_db"]) == snr
            ]
            for m in MEASURES:
                y = [_mean([r["gra"][str(a)][m] for r in sub]) for a in GRA_AMPS]
                ax.plot(
                    GRA_AMPS,
                    [np.nan if v is None else v for v in y],
                    color=_PALETTE[m],
                    marker="o",
                    ms=4,
                    lw=1.8,
                    label=m,
                )
            ax.axhline(0.5, color="#888888", ls="--", lw=1.2)
            ax.set_xscale("log")
            # Explicit, so a panel whose cases are all NaN still lays out.
            ax.set_xlim(min(GRA_AMPS) * 0.7, max(GRA_AMPS) * 1.4)
            ax.set_ylim(0.0, 1.03)
            if ri == 0:
                ax.set_title(f"SNR {snr:g} dB", fontsize=12)
            if ci == 0:
                ax.set_ylabel(f"{pair} pair\ngradient-sign ranking accuracy")
            if ri == 1:
                ax.set_xlabel("perturbation |a|, rev/s")
    axes[0][0].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    fig.suptitle(
        "Gradient-sign ranking accuracy — does a 5 % further step read worse? (0.5 = a coin flip)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = dest / "fig_gra.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def analyze(out_dir: Path) -> int:
    rows = load_rows(out_dir)
    if not rows:
        print(f"[fvk_bench] no unit JSONs under {out_dir}/raw", flush=True)
        return 1
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    md = [
        "## Synthetic\n",
        markdown_table(summary, "synth"),
        "\n## Real (FLY124 cruise)\n",
        markdown_table(summary, "real"),
    ]
    (out_dir / "table.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    for fn in (fig_profiles, fig_gra):
        path = fn(rows, out_dir)  # type: ignore[operator]
        print(f"[fvk_bench] figure: {path}", flush=True)
    if summary.get("nested_submultiple"):
        print(
            "\nnested sub-multiple (f0/2 with 2K harmonics; >0 = truth still wins):\n"
            + json.dumps(summary["nested_submultiple"], indent=1),
            flush=True,
        )
    cont = summary.get("by_material", {}).get("synth", {}).get("continuation")
    if cont:
        print("\ncontinuation:\n" + json.dumps(cont, indent=1), flush=True)
    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", default="results/fvk_bench", help="output tree (gridrun raw/ + summary.json)"
    )
    ap.add_argument("--material", default="synth", help="comma list: synth,real")
    ap.add_argument("--foms", default="G,M,GRA,CONT", help="comma list: G,M,GRA,CONT")
    ap.add_argument("--pairs", default=",".join(PAIRS), help="comma list of rotor geometries")
    ap.add_argument("--snrs", default=",".join(f"{v:g}" for v in SNRS))
    ap.add_argument("--durs", default=",".join(f"{v:g}" for v in DURS))
    ap.add_argument("--seeds", default=",".join(str(v) for v in SEEDS))
    ap.add_argument("--gra-trials", type=int, default=40, help="random perturbations per amplitude")
    ap.add_argument("--real-windows", type=int, default=4, help="FLY124 cruise windows to score")
    ap.add_argument(
        "--prep-dir", default=None, help="frozen prep cache (default: protocols.resolve_prep_dir)"
    )
    ap.add_argument(
        "--build-preps", action="store_true", help="rebuild the FLY124 prep windows first"
    )
    ap.add_argument("--quick", action="store_true", help="one tiny case, for the smoke test")
    ap.add_argument(
        "--analyze-only", action="store_true", help="re-read <out>/raw and rebuild the report"
    )
    add_gridrun_args(ap, jobs=4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.analyze_only:
        return analyze(out_dir)

    if args.quick:
        args.pairs, args.snrs, args.durs, args.seeds = "twin", "-10", "1", "0"
        args.gra_trials = min(args.gra_trials, 2)

    materials = args.material.split(",")
    if "real" in materials and args.build_preps:
        args.prep_dir = build_preps(out_dir / "preps")
        print(f"[fvk_bench] prep cache: {args.prep_dir}", flush=True)

    units: list[Unit] = []
    if "synth" in materials:
        units += synth_units(args)
    if "real" in materials:
        units += real_units(args)
    print(f"[fvk_bench] {len(units)} units -> {out_dir}", flush=True)

    result = gridrun_from_args(
        args,
        units,
        worker,
        out_dir,
        summarize=summarize,
        mp_context="spawn",  # torch in the workers; fork after import is a trap
    )
    analyze(out_dir)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
