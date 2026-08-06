"""Goodness of fit of a CANDIDATE rotor-speed trajectory against the audio.

GitHub issue 17, sections A-D. The campaign that measured the DREGON telemetry
bias chose its best setting by eye in the comb explorer, and the issue says
plainly why that is not good enough: *a fitted trajectory has far more freedom
than fixed telemetry, so it will fit better whether or not it is more correct*.
This module is the numerical answer to that. It scores a trajectory, it does so
at FIXED degrees of freedom, and it carries the four controls that make the
number mean something.

The statistic
-------------
A candidate ``r(t)`` is scored by demodulating the audio at the carriers it
implies (``tracking.phase_increment_tracker.demod_bank``, through
``comb_displacement.demod_comb_bank``) and reporting THREE components, never
one:

1. **Broadband noise** (:attr:`FitnessScore.broadband`). The share of
   demodulated envelope power that sits OUTSIDE the near-DC region the comb can
   explain. A correct carrier puts the rotor line at exactly DC and leaves the
   band otherwise empty; a wrong one leaves the line rotating, which reads as
   in-band residual. This is the issue's "envelope concentration", written as a
   residual so that all three components point the same way (less is better).
2. **Phase noise** (:attr:`FitnessScore.phase_noise`). The mean square of the
   per-harmonic rate opinion ``dr_k(t) = arg(z_k[n] conj(z_k[n-1])) fs_env /
   (2 pi k)``, about ZERO (not about its own mean — a scale error is a constant
   ``dr``, and a variance about the mean would be blind to exactly the error we
   are hunting). Harmonics are weighted ``k^2``, which is WP18's measured
   ``1 / v_k`` shape (``docs/experiments/rps-refine-precision.md``); the
   aggregation is per microphone because WP18 also measured the common jitter
   term to be per-MIC rather than per-shaft.
3. **Magnitude roughness** (:attr:`FitnessScore.roughness`). The share of
   ``|z_k(t)|`` power above ``rough_cut_hz``. A correct carrier makes the
   amplitude envelope slowly varying; a wrong one beats against the line it
   missed.

Every one of the three is reported separately. "A more correct trajectory
admits less variance" is a claim about all three at once, and collapsing them
into one number would hide the case where it holds for one and fails for
another.

Fixed degrees of freedom
------------------------
The comparison is only fair if the candidate is the ONLY thing that changes.
So, per (window, rotor), the following are pinned to the window's REFERENCE
trajectory and never re-derived from the candidate:

* the per-harmonic band ``B_k = min(b0 k, band_frac rate_ref)`` Hz,
* the envelope rate, the block partition and the edge trim,
* the harmonic set, and
* the **admission mask** — the conditioning gate of issue 17 §D9. Near-coincident
  cross-rotor pairs are excluded by
  :func:`tracking.comb_displacement.nearest_interloper_hz`, evaluated at the
  reference carriers. An admission rule that read the candidate (an envelope-SNR
  gate, say) would silently give a flexible trajectory a different, easier cell
  set — the very failure mode the issue is about.

What the harness deliberately does NOT use
------------------------------------------
The coupled VK envelope solve. Phase 5 of this campaign measured it and rejected
it (``docs/experiments/vk-frontend-probe.md``, issue 15): on contested harmonics
it is degenerate (amplitude 179-1156x, correlation with the data 0.000-0.007)
and it costs 15-110x the demodulation. Fixed-carrier demodulation is THE
estimator here.

Held-out scoring (§A)
---------------------
Every component is computed per **cell** ``(channel, harmonic, time block)``,
and a :class:`Holdout` is nothing but a mask over those axes: held-out harmonics
(fit even ``k``, score odd), held-out channels (fit 1 mic, score 7), held-out
time (fit some blocks, score the gaps). The fitter itself arrives in phase 6b;
here the "fit" side is recorded and masked out of the score.

Controls (§B)
-------------
:func:`apply_control` produces the carriers of all four:

``none``
    the measurement.
``offcomb``
    the half-integer comb ``(k + 0.5) g(t)``, where no rotor line can exist.
``mismatch``
    a partner window's trajectory on this window's audio.
``permute``
    the candidate's rotor rows cyclically permuted.

One property of the permutation control has to be stated up front, because it
is a fact about the statistic rather than a defect. **The acoustic components
are invariant under a rotor permutation, by construction.** The three
components are functions of the carrier SET only, and permuting rows does not
change that set; the gate follows the carrier (``skip`` below), so it does not
change either. What the permutation does break is the per-rotor CORRESPONDENCE
with telemetry, which is what :func:`residual_decompose` measures — and that is
exactly the reading the issue's own comment arrived at ("a permutation null
needs a quantity attached to a rotor independently of the carrier"). So the
permutation control is run against the residual half of the report, and its
invariance on the acoustic half is an assertion the tests check rather than a
result to interpret.

Residual decomposition (§C) and uncertainty (§D)
------------------------------------------------
:func:`residual_decompose` splits ``candidate - telemetry`` into a smooth
systematic part (reported as a scale in percent and a lag in seconds, from one
joint least-squares fit against ``[r, dr/dt, 1]``) and a residual, then tests
the residual against the DREGON tachometer's known signature: bounded by half a
quantisation step (0.135 rev/s), roughly flat in spectrum up to the 24.85 Hz
refresh Nyquist, structure at the 49.7 Hz refresh rate. It reports the spectrum
statistics, never a verdict.

:func:`bootstrap_scores` resamples cells over microphone subsets, harmonic
subsets and time blocks, giving a confidence interval on any scalar derived
from the scores — the error bar the issue notes no current estimate has.

Purity: numpy plus tracking siblings only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from tracking.comb_displacement import (
    DisplacementConfig,
    demod_comb_bank,
    nearest_interloper_hz,
    pulse_pair_bank,
)
from tracking.phase_noise import brickwall

__all__ = [
    "CONTROLS",
    "Cells",
    "FitnessConfig",
    "FitnessScore",
    "Holdout",
    "apply_control",
    "bootstrap_scores",
    "default_holdouts",
    "fitness_stage",
    "residual_decompose",
    "score_cells",
    "score_window",
    "window_cells",
]

#: The four controls of issue 17 section B (FLY124, the fourth, is a DATA
#: choice — every function here is recording-agnostic, so running the identical
#: procedure on FLY124 is one flag of the driver).
CONTROLS: tuple[str, ...] = ("none", "offcomb", "mismatch", "permute")

#: DREGON tachometer signature (``docs/experiments/dregon-telemetry-forensics.md``):
#: quantisation step at 80 rev/s, refresh rate, and the implied bound on the
#: quantisation part of a residual (half a step).
TACH_STEP_REV_S = 0.269
TACH_REFRESH_HZ = 49.7
TACH_BOUND_REV_S = 0.5 * TACH_STEP_REV_S


@dataclass(frozen=True)
class FitnessConfig:
    """Measurement geometry. Every field is pinned per window, not per candidate.

    The two shapes that matter:

    * ``B_k = min(b0_revs * k, band_frac_of_rate * rate_ref)`` Hz — a CONSTANT
      capture in rev/s at every harmonic (a rate error ``dr`` displaces harmonic
      ``k`` by ``k dr`` Hz), clamped away from the neighbouring tooth. This is
      the k-scaled band the displacement campaign's only identity-preserving
      refinement used.
    * ``dc_revs`` is the half-width, in rev/s, of the near-DC region counted as
      "the comb". It is floored at ``dc_bins`` FFT bins of one block, because a
      region narrower than the block's own resolution measures nothing.
    """

    sr: int = 16000
    fs_env: float = 250.0
    k_min: int = 2
    k_max: int = 40
    b0_revs: float = 1.0
    band_frac_of_rate: float = 0.45
    dc_revs: float = 0.10
    dc_bins: int = 2
    rough_cut_hz: float = 5.0
    phase_weight_exp: float = 2.0
    n_blocks: int = 8
    guard_hz: float = 1.0
    #: The conditioning gate admits harmonic ``k`` when the nearest foreign
    #: rotor line is farther than ``gate_band_frac * B_k + guard_hz``. At 1.0
    #: (the default) no interferer may be inside the band at all, which on a
    #: DREGON twin pair — separation 0.42 rev/s against a capture of ``b0``
    #: rev/s — empties almost the whole harmonic set. Lowering it is the
    #: coverage-versus-purity trade, and it must be reported with the number.
    gate_band_frac: float = 1.0
    min_clear_frac: float = 0.9
    edge_trim_s: float = 0.5
    min_rate: float = 5.0
    f_max: float = 6000.0
    probe_frac: float = 0.5
    #: Low-pass cutoff separating the smooth part of a trajectory residual.
    smooth_cut_hz: float = 1.0
    #: Channel cap (the multichannel-fusion convention of the tracking stack).
    max_channels: int = 8

    @property
    def stride(self) -> int:
        return int(round(self.sr / self.fs_env))

    @property
    def ks(self) -> tuple[int, ...]:
        return tuple(range(self.k_min, self.k_max + 1))

    def band_hz(self, rate_ref: float) -> np.ndarray:
        """``(K,)`` demodulation half-band in Hz at a pinned reference rate."""
        return np.array(
            [min(self.b0_revs * k, self.band_frac_of_rate * rate_ref) for k in self.ks],
            dtype=np.float64,
        )

    def displacement(self) -> DisplacementConfig:
        """The sibling geometry object ``demod_comb_bank`` takes."""
        return DisplacementConfig(sr=self.sr, fs_env=self.fs_env, f_max=self.f_max)


# ---------------------------------------------------------------------------
# hold-out specification


@dataclass(frozen=True)
class Holdout:
    """Which cells are the FIT side; the score runs on the complement.

    ``kind``:

    ``none``
        everything is scored (no fit side).
    ``harmonics``
        ``fit`` holds one parity: ``(0,)`` fits even ``k`` and scores odd.
    ``channels``
        ``fit`` holds the microphone indices used for fitting.
    ``blocks``
        ``fit`` holds the time-block indices used for fitting.

    Phase 6a has no fitter, so ``fit`` is only recorded and excluded. Phase 6b
    hands the same object to the fitter, which is why the fit side is named
    rather than implied.
    """

    kind: str = "none"
    fit: tuple[int, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("none", "harmonics", "channels", "blocks"):
            raise ValueError(f"unknown holdout kind {self.kind!r}")
        if not self.name:
            object.__setattr__(self, "name", self.kind if not self.fit else self._auto_name())

    def _auto_name(self) -> str:
        return f"{self.kind}:{'+'.join(str(v) for v in self.fit)}"

    @classmethod
    def none(cls) -> Holdout:
        return cls(kind="none", name="none")

    @classmethod
    def harmonics(cls, fit_parity: int) -> Holdout:
        tag = "even" if fit_parity == 0 else "odd"
        return cls(kind="harmonics", fit=(int(fit_parity),), name=f"fit_k_{tag}")

    @classmethod
    def channels(cls, fit: Sequence[int]) -> Holdout:
        f = tuple(int(v) for v in fit)
        return cls(kind="channels", fit=f, name=f"fit_ch_{'+'.join(map(str, f))}")

    @classmethod
    def blocks(cls, fit: Sequence[int]) -> Holdout:
        f = tuple(int(v) for v in fit)
        return cls(kind="blocks", fit=f, name=f"fit_blk_{'+'.join(map(str, f))}")

    def score_mask(self, n_ch: int, ks: Sequence[int], n_blocks: int) -> np.ndarray:
        """``(C, K, B)`` bool: the cells this hold-out SCORES."""
        m = np.ones((n_ch, len(ks), n_blocks), dtype=bool)
        if self.kind == "harmonics":
            parity = self.fit[0] % 2
            keep = np.array([(k % 2) != parity for k in ks], dtype=bool)
            m &= keep[None, :, None]
        elif self.kind == "channels":
            keep = np.ones(n_ch, dtype=bool)
            keep[[c for c in self.fit if 0 <= c < n_ch]] = False
            m &= keep[:, None, None]
        elif self.kind == "blocks":
            keep = np.ones(n_blocks, dtype=bool)
            keep[[b for b in self.fit if 0 <= b < n_blocks]] = False
            m &= keep[None, None, :]
        return m

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "fit": list(self.fit)}


def default_holdouts(n_ch: int) -> tuple[Holdout, ...]:
    """The §A backbone: no hold-out, both harmonic parities, mic 0, half the blocks."""
    return (
        Holdout.none(),
        Holdout.harmonics(0),
        Holdout.harmonics(1),
        Holdout.channels((0,)),
        Holdout.blocks((0, 2, 4, 6)) if n_ch else Holdout.none(),
    )


# ---------------------------------------------------------------------------
# the per-cell measurement


@dataclass
class Cells:
    """One rotor's per-cell statistics on a ``(channel, harmonic, block)`` grid.

    Every array is ``(C, K, B)`` except where noted. ``admit`` is ``(K, B)`` —
    the conditioning gate is a property of the geometry, not of a microphone —
    and it is derived from the REFERENCE trajectory so that it is identical for
    every candidate and every control.
    """

    rotor: int
    ks: tuple[int, ...]
    band_hz: np.ndarray  # (K,)
    rate_ref: float
    broadband: np.ndarray  # (C, K, B) out-of-DC share of envelope power
    phase_ms: np.ndarray  # (C, K, B) mean square rate opinion, (rev/s)^2
    roughness: np.ndarray  # (C, K, B) high-pass share of |z| power
    pp_dr: np.ndarray  # (C, K, B) coherent pulse-pair centre, rev/s
    pp_coh: np.ndarray  # (C, K, B) pulse-pair coherence
    snr: np.ndarray  # (C, K, B) in-band signal / off-comb noise power
    admit: np.ndarray  # (K, B) bool
    block_t: np.ndarray  # (B, 2) block time spans, seconds
    diag: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.broadband.shape  # type: ignore[return-value]


def _blocks(n_env: int, cfg: FitnessConfig) -> list[slice]:
    """Equal-length contiguous blocks over the edge-trimmed envelope grid."""
    trim = max(1, int(round(cfg.edge_trim_s * cfg.fs_env)))
    lo, hi = trim, max(trim + 1, n_env - trim)
    span = hi - lo
    length = span // max(cfg.n_blocks, 1)
    if length < 32:
        raise ValueError(
            f"blocks of {length} envelope samples are too short "
            f"(n_env={n_env}, n_blocks={cfg.n_blocks}); shorten n_blocks or the trim"
        )
    return [slice(lo + b * length, lo + (b + 1) * length) for b in range(cfg.n_blocks)]


def admission(
    reference: np.ndarray,
    ft: np.ndarray,
    rot: int,
    blocks: Sequence[slice],
    band_hz: np.ndarray,
    *,
    cfg: FitnessConfig,
    rate_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``((K, B) admit, (K, N) nearest-interloper Hz)`` from the REFERENCE only.

    The conditioning gate of issue 17 §D9: harmonic ``k`` is admitted in a block
    when no other rotor's real line comes within
    ``gate_band_frac * B_k + guard_hz`` of it for at least ``min_clear_frac``
    of the block's frames, and when ``k rate_ref`` is
    still inside the audio band. Nothing here reads the candidate, which is what
    keeps the degrees of freedom identical across candidates.
    """
    ks = cfg.ks
    nearest = nearest_interloper_hz(
        reference, reference[rot], rot, ks, f_max=cfg.f_max, min_rate=cfg.min_rate
    )
    clear = nearest > (cfg.gate_band_frac * band_hz[:, None] + cfg.guard_hz)
    stride_s = cfg.stride / cfg.sr
    admit = np.zeros((len(ks), len(blocks)), dtype=bool)
    for b, sl in enumerate(blocks):
        t0, t1 = sl.start * stride_s, (sl.stop - 1) * stride_s
        sel = (ft >= t0) & (ft <= t1)
        if not sel.any():
            sel = np.zeros(ft.size, dtype=bool)
            sel[min(int(t0 / max(ft[1] - ft[0], 1e-9)), ft.size - 1)] = True
        admit[:, b] = clear[:, sel].mean(axis=1) >= cfg.min_clear_frac
    in_band = np.array([k * rate_ref <= cfg.f_max for k in ks], dtype=bool)
    admit &= in_band[:, None]
    return admit, nearest


def demod_cells(
    audio: np.ndarray,
    ft: np.ndarray,
    carrier: np.ndarray,
    reference: np.ndarray,
    rot: int,
    skip: int,
    *,
    cfg: FitnessConfig = FitnessConfig(),
    half: bool = False,
) -> Cells:
    """One rotor's :class:`Cells` for one carrier.

    Args:
        audio: ``(C, T)`` at ``cfg.sr``.
        ft: ``(N,)`` frame times, audio-relative seconds.
        carrier: ``(N,)`` the candidate rate for this rotor slot, rev/s.
        reference: ``(R, N)`` the window's reference trajectories (telemetry).
            They pin the band, the admission gate and the interferer geometry.
        rot: the rotor slot being scored (for the report).
        skip: the reference rotor whose line the CARRIER sits on, excluded from
            the interferer set. It differs from ``rot`` only under the
            rotor-permutation control, where following the carrier is what
            keeps the gate honest (see the module docstring).
        half: the off-comb null carrier ``(k + 0.5) g(t)``.
    """
    ks = cfg.ks
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))[: cfg.max_channels]
    n_t = x.shape[-1]
    n_env = n_t // cfg.stride
    blocks = _blocks(n_env, cfg)
    rate_ref = float(np.mean(reference[skip]))
    band_hz = cfg.band_hz(rate_ref)
    dcfg = cfg.displacement()

    z_on, z_off, _ = demod_comb_bank(
        x,
        np.asarray(carrier, dtype=np.float64),
        ft,
        ks,
        cfg=dcfg,
        half=half,
        band_hz_k=band_hz,
        probe_off_hz=cfg.probe_frac * rate_ref,
        return_probe=True,
    )
    noise_pow = np.maximum(np.mean(np.abs(z_off) ** 2, axis=-1), 1e-30)  # (C, K)

    admit, nearest = admission(reference, ft, skip, blocks, band_hz, cfg=cfg, rate_ref=rate_ref)

    n_ch, n_k = z_on.shape[0], len(ks)
    n_b = len(blocks)
    length = blocks[0].stop - blocks[0].start
    kf = np.asarray(ks, dtype=np.float64)

    # Fixed per-block frequency geometry: the "comb" region and the roughness
    # cut are the same for every candidate because the block length is.
    freqs = np.fft.fftfreq(length, d=1.0 / cfg.fs_env)
    res_hz = cfg.fs_env / length
    dc_hz = np.minimum(np.maximum(cfg.dc_revs * kf, cfg.dc_bins * res_hz), 0.9 * band_hz)
    dc_mask = np.abs(freqs)[None, :] <= dc_hz[:, None]  # (K, L)
    rfreq = np.fft.rfftfreq(length, d=1.0 / cfg.fs_env)
    hi_mask = rfreq >= cfg.rough_cut_hz
    pos_mask = rfreq > 0

    bb = np.empty((n_ch, n_k, n_b))
    ph = np.empty((n_ch, n_k, n_b))
    ro = np.empty((n_ch, n_k, n_b))
    pp = np.empty((n_ch, n_k, n_b))
    coh = np.empty((n_ch, n_k, n_b))
    snr = np.empty((n_ch, n_k, n_b))

    for b, sl in enumerate(blocks):
        zb = np.asarray(z_on[:, :, sl], dtype=np.complex128)
        # 1. broadband: power outside the near-DC comb region
        pw = np.abs(np.fft.fft(zb, axis=-1)) ** 2
        tot = pw.sum(axis=-1)
        inn = (pw * dc_mask[None, :, :]).sum(axis=-1)
        bb[:, :, b] = 1.0 - inn / np.maximum(tot, 1e-300)
        # 2. phase noise: mean square rate opinion about ZERO
        lag = zb[..., 1:] * np.conj(zb[..., :-1])
        dr = np.angle(lag) * cfg.fs_env / (2.0 * np.pi * kf[None, :, None])
        ph[:, :, b] = np.mean(dr**2, axis=-1)
        off, cc = pulse_pair_bank(zb, ks, fs_env=cfg.fs_env, sum_channels=False)
        pp[:, :, b], coh[:, :, b] = off, cc
        # 3. magnitude roughness: high-pass share of |z| power
        amp = np.abs(zb)
        amp = amp - amp.mean(axis=-1, keepdims=True)
        pa = np.abs(np.fft.rfft(amp, axis=-1)) ** 2
        ro[:, :, b] = pa[..., hi_mask].sum(-1) / np.maximum(pa[..., pos_mask].sum(-1), 1e-300)
        snr[:, :, b] = np.mean(np.abs(zb) ** 2, axis=-1) / noise_pow

    stride_s = cfg.stride / cfg.sr
    block_t = np.array([[sl.start * stride_s, (sl.stop - 1) * stride_s] for sl in blocks])
    return Cells(
        rotor=rot,
        ks=ks,
        band_hz=band_hz,
        rate_ref=rate_ref,
        broadband=bb,
        phase_ms=ph,
        roughness=ro,
        pp_dr=pp,
        pp_coh=coh,
        snr=snr,
        admit=admit,
        block_t=block_t,
        diag={
            "skip": int(skip),
            "half": bool(half),
            "n_env": int(n_env),
            "block_len": int(length),
            "res_hz": round(float(res_hz), 4),
            "dc_hz": np.round(dc_hz, 3).tolist(),
            "admit_frac": round(float(admit.mean()), 4),
            "median_nearest_hz": round(float(np.median(nearest[np.isfinite(nearest)])), 2)
            if np.isfinite(nearest).any()
            else None,
        },
    )


# ---------------------------------------------------------------------------
# controls


def apply_control(
    candidate: np.ndarray,
    control: str = "none",
    partner: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(carriers (R, N), skip (R,), half)`` for one of :data:`CONTROLS`.

    ``skip[i]`` names the REFERENCE rotor whose line carrier ``i`` sits on, so
    the collision gate excludes the carrier's own line rather than a slot index.
    Under ``permute`` that is the permuted index — without it the carrier would
    collide with a line the gate is not allowed to skip and every unit would
    return NaN, which is the degeneracy the issue's own comment describes.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    n_r = cand.shape[0]
    if control == "none":
        return cand, np.arange(n_r), False
    if control == "offcomb":
        return cand, np.arange(n_r), True
    if control == "mismatch":
        if partner is None:
            raise ValueError("control 'mismatch' needs a partner trajectory")
        p = np.asarray(partner, dtype=np.float64)
        if p.shape != cand.shape:
            raise ValueError(f"partner shape {p.shape} != candidate shape {cand.shape}")
        return p, np.arange(n_r), False
    if control == "permute":
        perm = np.roll(np.arange(n_r), 1)
        return cand[perm], perm, False
    raise ValueError(f"unknown control {control!r}; known: {CONTROLS}")


# ---------------------------------------------------------------------------
# scoring


@dataclass(frozen=True)
class FitnessScore:
    """The three components plus the provenance needed to read them."""

    broadband: float
    phase_noise: float
    roughness: float
    pp_dr: float
    pp_abs: float
    snr_median: float
    n_cells: int
    per_rotor: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "broadband": _r(self.broadband, 6),
            "phase_noise": _r(self.phase_noise, 6),
            "roughness": _r(self.roughness, 6),
            "pp_dr": _r(self.pp_dr, 5),
            "pp_abs": _r(self.pp_abs, 5),
            "snr_median": _r(self.snr_median, 4),
            "n_cells": self.n_cells,
            "per_rotor": self.per_rotor,
        }

    def component(self, name: str) -> float:
        return {
            "broadband": self.broadband,
            "phase_noise": self.phase_noise,
            "roughness": self.roughness,
        }[name]


def _r(v: float, nd: int) -> float | None:
    return None if not np.isfinite(v) else round(float(v), nd)


def _wmean(vals: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(vals) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(w[ok] * vals[ok]) / np.sum(w[ok]))


def _cell_weights(
    cells: Cells, holdout: Holdout, cfg: FitnessConfig, extra: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(uniform weight, k^exp weight)`` over the admitted, scored cells."""
    n_ch, n_k, n_b = cells.shape
    w = holdout.score_mask(n_ch, cells.ks, n_b).astype(np.float64)
    w *= cells.admit[None, :, :].astype(np.float64)
    if extra is not None:
        w = w * extra
    kw = np.asarray(cells.ks, dtype=np.float64)[None, :, None] ** cfg.phase_weight_exp
    return w, w * kw


def score_cells(
    cells: Sequence[Cells],
    holdout: Holdout = Holdout.none(),
    *,
    cfg: FitnessConfig = FitnessConfig(),
    extra: Sequence[np.ndarray] | None = None,
) -> FitnessScore:
    """Aggregate per-rotor :class:`Cells` into the three components.

    Rotors are combined by an unweighted mean of their per-rotor aggregates,
    which is the pooling convention of the tracking protocols. ``extra`` is an
    optional per-rotor multiplicative cell weight — the bootstrap's resampling
    counts, and nothing else should use it.
    """
    per_rotor: dict[str, dict[str, Any]] = {}
    acc: dict[str, list[float]] = {k: [] for k in ("bb", "ph", "ro", "pp", "ppa", "snr")}
    n_cells = 0
    for i, c in enumerate(cells):
        w, kw = _cell_weights(c, holdout, cfg, None if extra is None else extra[i])
        vals = {
            "bb": _wmean(c.broadband, w),
            "ph": _wmean(c.phase_ms, kw),
            "ro": _wmean(c.roughness, w),
            "pp": _wmean(c.pp_dr, kw),
            "ppa": _wmean(np.abs(c.pp_dr), kw),
            "snr": _wmean(c.snr, w),
        }
        n_cells += int(np.count_nonzero(w > 0))
        for k, v in vals.items():
            acc[k].append(v)
        per_rotor[str(c.rotor)] = {
            "broadband": _r(vals["bb"], 6),
            "phase_noise": _r(vals["ph"], 6),
            "roughness": _r(vals["ro"], 6),
            "pp_dr": _r(vals["pp"], 5),
            "n_cells": int(np.count_nonzero(w > 0)),
        }

    def m(key: str) -> float:
        v = np.asarray(acc[key], dtype=np.float64)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    return FitnessScore(
        broadband=m("bb"),
        phase_noise=m("ph"),
        roughness=m("ro"),
        pp_dr=m("pp"),
        pp_abs=m("ppa"),
        snr_median=m("snr"),
        n_cells=n_cells,
        per_rotor=per_rotor,
    )


# ---------------------------------------------------------------------------
# bootstrap (§D8)


def bootstrap_scores(
    cells: Sequence[Cells],
    holdout: Holdout = Holdout.none(),
    *,
    cfg: FitnessConfig = FitnessConfig(),
    n_boot: int = 200,
    axes: Sequence[str] = ("channels", "harmonics", "blocks"),
    seed: int = 0,
    stat: Callable[[FitnessScore], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Confidence intervals by resampling cells over the requested axes.

    Each axis is resampled WITH REPLACEMENT and the draw enters as an integer
    cell weight, so a resample is a re-aggregation of statistics already
    computed — the demodulation runs once. ``stat`` maps a score to the scalars
    to bound; the default bounds the three components and the pulse-pair centre.
    """
    rng = np.random.default_rng(seed)
    stat = stat or _default_stat
    n_ch, n_k, n_b = cells[0].shape
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        extra = []
        cnt_c = _counts(rng, n_ch) if "channels" in axes else None
        cnt_k = _counts(rng, n_k) if "harmonics" in axes else None
        cnt_b = _counts(rng, n_b) if "blocks" in axes else None
        w = np.ones((n_ch, n_k, n_b))
        if cnt_c is not None:
            w = w * cnt_c[:, None, None]
        if cnt_k is not None:
            w = w * cnt_k[None, :, None]
        if cnt_b is not None:
            w = w * cnt_b[None, None, :]
        extra = [w] * len(cells)
        s = stat(score_cells(cells, holdout, cfg=cfg, extra=extra))
        for k, v in s.items():
            draws.setdefault(k, []).append(v)
    out: dict[str, Any] = {"n_boot": int(n_boot), "axes": list(axes), "seed": int(seed)}
    for k, v in draws.items():
        a = np.asarray(v, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size < 8:
            out[k] = None
            continue
        out[k] = {
            "mean": round(float(a.mean()), 6),
            "sd": round(float(a.std(ddof=1)), 6),
            "lo": round(float(np.percentile(a, 2.5)), 6),
            "hi": round(float(np.percentile(a, 97.5)), 6),
        }
    return out


def _counts(rng: np.random.Generator, n: int) -> np.ndarray:
    idx = rng.integers(0, n, size=n)
    return np.bincount(idx, minlength=n).astype(np.float64)


def _default_stat(s: FitnessScore) -> dict[str, float]:
    return {
        "broadband": s.broadband,
        "phase_noise": s.phase_noise,
        "roughness": s.roughness,
        "pp_dr": s.pp_dr,
    }


# ---------------------------------------------------------------------------
# residual decomposition (§C)


def residual_decompose(
    candidate: np.ndarray,
    reference: np.ndarray,
    ft: np.ndarray,
    *,
    cfg: FitnessConfig = FitnessConfig(),
) -> dict[str, Any]:
    """Split ``candidate - reference`` into systematic + residual, and test it.

    The systematic part is one joint least squares of ``d = candidate -
    reference`` on ``[reference, d(reference)/dt, 1]``: the first coefficient is
    a rate SCALE (reported in percent), the second is minus a LAG in seconds
    (a period counter reports the previous revolution, so a lag shows up as a
    term proportional to the derivative), the third an offset in rev/s.

    The residual is then tested against the DREGON tachometer's known
    signature — bounded by half a quantisation step, roughly flat in spectrum
    up to the refresh Nyquist, structure at the refresh rate. All three are
    reported as numbers: a binary verdict would hide that the frame grid can be
    too coarse to resolve the refresh line at all (``f_tach_resolved``).

    Read ``design_cond`` before reading ``scale_pct``. On a cruise window the
    rate column and the intercept are nearly collinear (a rotor holds 85 rev/s
    to about 1 %), so the systematic part is split between them arbitrarily and
    ``scale_pct`` alone means nothing — ``d_mean`` and ``d_rms`` still do, and
    :func:`tracking.telemetry_refit.scale_summary` gives the well-posed scale.
    """
    cand = np.atleast_2d(np.asarray(candidate, dtype=np.float64))
    ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    ft = np.asarray(ft, dtype=np.float64)
    dt = float(np.median(np.diff(ft))) if ft.size > 1 else 1.0
    fs = 1.0 / max(dt, 1e-9)
    trim = int(round(cfg.edge_trim_s / max(dt, 1e-9)))
    sel = slice(trim, max(trim + 1, ft.size - trim))

    per_rotor: list[dict[str, Any]] = []
    for r in range(cand.shape[0]):
        if float(np.mean(ref[r])) < cfg.min_rate:
            continue
        d = cand[r] - ref[r]
        drdt = np.gradient(ref[r], ft)
        des = np.stack([ref[r], drdt, np.ones_like(ref[r])], axis=1)[sel]
        y = d[sel]
        coef, *_ = np.linalg.lstsq(des, y, rcond=None)
        model = des @ coef
        resid = y - model
        # The split between the scale column and the intercept is only
        # identified when the rate actually varies. On a cruise window it does
        # not (85 rev/s +- 1 %), the two columns are collinear, and least
        # squares divides the systematic part between them arbitrarily —
        # DREGON w01 reads scale -31.9 % with offset +27.1 rev/s, which is one
        # number, not two. ``design_cond`` says when that has happened;
        # ``tracking.telemetry_refit.scale_summary`` is the well-posed reading.
        cond = float(np.linalg.cond(des / np.maximum(np.abs(des).max(axis=0), 1e-30)))
        smooth = brickwall(resid, cfg.smooth_cut_hz, fs)
        per_rotor.append(
            {
                "rotor": r,
                "mean_rev_s": round(float(np.mean(ref[r])), 3),
                "d_mean": round(float(np.mean(y)), 5),
                "d_rms": round(float(np.sqrt(np.mean(y**2))), 5),
                "scale_pct": round(float(100.0 * coef[0]), 5),
                "lag_s": round(float(-coef[1]), 5),
                "offset_rev_s": round(float(coef[2]), 5),
                "design_cond": round(cond, 1),
                "resid_rms": round(float(np.sqrt(np.mean(resid**2))), 5),
                "resid_max_abs": round(float(np.max(np.abs(resid))), 5),
                "resid_slow_share": round(
                    float(np.sum(smooth**2) / max(np.sum(resid**2), 1e-30)), 4
                ),
                **_tach_signature(resid, fs),
            }
        )
    if not per_rotor:
        return {"per_rotor": [], "pooled": {}}

    def pool(key: str) -> float | None:
        v = np.asarray([p[key] for p in per_rotor if p.get(key) is not None], dtype=np.float64)
        return round(float(np.mean(v)), 5) if v.size else None

    pooled = {
        k: pool(k)
        for k in (
            "d_mean",
            "d_rms",
            "scale_pct",
            "lag_s",
            "design_cond",
            "resid_rms",
            "tach_bound_frac",
            "tach_flatness",
            "tach_line_ratio",
            "tach_acf",
        )
    }
    pooled["fs_frame_hz"] = round(fs, 3)
    pooled["f_tach_resolved"] = bool(fs > 2.0 * TACH_REFRESH_HZ)
    return {"per_rotor": per_rotor, "pooled": pooled}


def _tach_signature(resid: np.ndarray, fs: float) -> dict[str, Any]:
    """The three tachometer-signature readings of §C on one residual series."""
    out: dict[str, Any] = {
        "tach_bound_frac": round(float(np.mean(np.abs(resid) <= TACH_BOUND_REV_S)), 4),
        "fs_frame_hz": round(float(fs), 3),
        "f_tach_resolved": bool(fs > 2.0 * TACH_REFRESH_HZ),
    }
    n = resid.size
    if n < 32:
        return {**out, "tach_flatness": None, "tach_line_ratio": None, "tach_acf": None}
    x = resid - resid.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    # flat-ish up to the refresh Nyquist: spectral flatness (geometric/arithmetic)
    top = min(0.5 * TACH_REFRESH_HZ, 0.95 * fs / 2.0)
    band = (f > 0) & (f <= top)
    if band.sum() >= 8:
        pb = np.maximum(p[band], 1e-300)
        out["tach_flatness"] = round(float(np.exp(np.mean(np.log(pb))) / np.mean(pb)), 4)
        out["tach_band_top_hz"] = round(float(top), 3)
    else:
        out["tach_flatness"] = None
    # the refresh line itself, only where the grid can carry it
    if out["f_tach_resolved"]:
        j = int(np.argmin(np.abs(f - TACH_REFRESH_HZ)))
        near = (np.abs(f - TACH_REFRESH_HZ) > 2.0) & (np.abs(f - TACH_REFRESH_HZ) < 10.0)
        base = float(np.median(p[near])) if near.sum() >= 4 else float(np.median(p[1:]))
        out["tach_line_ratio"] = round(float(p[j] / max(base, 1e-300)), 4)
        lag = int(round(fs / TACH_REFRESH_HZ))
        if 1 <= lag < n // 2:
            a, b = x[:-lag], x[lag:]
            den = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
            out["tach_acf"] = round(float(np.sum(a * b) / max(den, 1e-300)), 4)
        else:
            out["tach_acf"] = None
    else:
        out["tach_line_ratio"] = None
        out["tach_acf"] = None
    return out


# ---------------------------------------------------------------------------
# the front door


def window_cells(
    audio: np.ndarray,
    ft: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    cfg: FitnessConfig = FitnessConfig(),
    control: str = "none",
    partner: np.ndarray | None = None,
) -> list[Cells]:
    """The expensive half: one demodulation per rotor under one control."""
    carriers, skip, half = apply_control(candidate, control, partner)
    ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    out: list[Cells] = []
    for rot in range(carriers.shape[0]):
        if float(np.mean(ref[int(skip[rot])])) < cfg.min_rate:
            continue
        out.append(
            demod_cells(
                audio,
                ft,
                carriers[rot],
                ref,
                rot,
                int(skip[rot]),
                cfg=cfg,
                half=half,
            )
        )
    return out


def score_window(
    audio: np.ndarray,
    ft: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    cfg: FitnessConfig = FitnessConfig(),
    holdouts: Sequence[Holdout] | None = None,
    control: str = "none",
    partner: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """One (window, candidate, control) unit: every hold-out, plus §C and §D.

    The demodulation runs once; hold-outs, the bootstrap and the residual
    decomposition are all re-aggregations of the same cells, which is why the
    unit of the driver is a control and not a hold-out.
    """
    cells = window_cells(audio, ft, candidate, reference, cfg=cfg, control=control, partner=partner)
    if not cells:
        return {"control": control, "failed": "no rotor above min_rate"}
    n_ch = cells[0].shape[0]
    hos = list(holdouts) if holdouts is not None else list(default_holdouts(n_ch))
    carriers, _skip, _half = apply_control(candidate, control, partner)
    scores = {h.name: score_cells(cells, h, cfg=cfg).as_dict() for h in hos}
    boots = {
        h.name: bootstrap_scores(cells, h, cfg=cfg, n_boot=n_boot, seed=seed)
        for h in hos
        if h.kind == "none"
    }
    return {
        "control": control,
        "holdouts": [h.as_dict() for h in hos],
        "scores": scores,
        "bootstrap": boots,
        "residual": residual_decompose(carriers, reference, ft, cfg=cfg),
        "cells": {
            "n_rotors": len(cells),
            "n_channels": int(n_ch),
            "n_harmonics": int(cells[0].shape[1]),
            "n_blocks": int(cells[0].shape[2]),
            "admit_frac": round(float(np.mean([c.admit.mean() for c in cells])), 4),
            "rate_ref": [round(c.rate_ref, 3) for c in cells],
            "diag": [c.diag for c in cells],
        },
    }


# ---------------------------------------------------------------------------
# Stage adapter


def fitness_stage(
    reference_entry: str = "rps_meas",
    *,
    cfg: FitnessConfig = FitnessConfig(),
    holdouts: Sequence[Holdout] | None = None,
    control: str = "none",
) -> Any:
    """A :data:`tracking.stages.Stage` that scores the frame's current ``rps``.

    The trajectory is not changed; the stage appends a ``{"stage": "fitness",
    ...}`` diagnostics entry, so it can be dropped anywhere into a refinement
    ladder to record how the fit moved. The reference (which pins the bands and
    the gate) is read from ``reference_entry``, defaulting to the frame's
    untouched ``rps_meas``.
    """
    from tracking.stages import get_audio, get_rps, with_rps

    def run(frame: Any) -> Any:
        audio, sr = get_audio(frame)
        r, ft = get_rps(frame)
        ref, _ = get_rps(frame, reference_entry)
        t0 = float(ft[0]) if ft.size else 0.0
        use = replace(cfg, sr=int(round(sr)))
        info = score_window(
            audio,
            ft - t0,
            r,
            ref,
            cfg=use,
            holdouts=holdouts,
            control=control,
            n_boot=0,
        )
        return with_rps(frame, r, ft, stage="fitness", info=info)

    return run
