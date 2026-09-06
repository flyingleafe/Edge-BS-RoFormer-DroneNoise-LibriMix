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
4. **Ridge concentration** (:attr:`FitnessScore.ridge`, phase 6d, and the only
   component where MORE is better). The line power on the carrier's own ridge
   over the LOCAL floor, in dB: ``10 log10(mean power in |f| <= dc_hz(k) /
   floor density)``, the floor read from an annulus of the same envelope
   spectrum with every known interferer offset excised. This is the statistic
   the eye uses on a spectrogram — is there a line here, and how far does it
   stand above the noise around it — and it is the Phase-7 generator readout
   (``docs/experiments/generator-label-sensitivity.md``, self-test flat to
   0.008 dB) moved into the demodulation domain: same fixed band, same local
   floor, same refusal to peak-search. Components 1-3 are all SHARES of the
   in-band power, so they saturate once the envelope is noise dominated (the
   phase increments of a noise-dominated phasor are uniform whether or not the
   carrier is right); component 4 does not, which is why phase 6d added it.

Every one of the four is reported separately. "A more correct trajectory admits
less variance" is a claim about all of them at once, and collapsing them into
one number would hide the case where it holds for one and fails for another.

Fixed degrees of freedom
------------------------
The comparison is only fair if the candidate is the ONLY thing that changes.
So, per (window, rotor), the following are pinned to the window's REFERENCE
trajectory and never re-derived from the candidate:

* the per-harmonic band ``B_k = min(b0 k, band_frac rate_ref)`` Hz,
* the envelope rate, the block partition and the edge trim,
* the harmonic set, and
* the **admission masks** — the conditioning gate of issue 17 §D9. Near-coincident
  cross-rotor pairs are excluded by
  :func:`tracking.comb_displacement.nearest_interloper_hz`, evaluated at the
  reference carriers. An admission rule that read the candidate (an envelope-SNR
  gate, say) would silently give a flexible trajectory a different, easier cell
  set — the very failure mode the issue is about.

  There are TWO gates, because the two kinds of statistic need different
  protection (phase 6d). Components 1-3 measure the shape of everything inside
  the band, so an interferer anywhere in the band corrupts them:
  ``admit`` requires the nearest foreign line to be outside
  ``gate_band_frac * B_k``. Component 4 measures the power at DC against a floor
  it reads elsewhere, so it only needs the interferer to be RESOLVED away from
  DC and excised from the floor region: ``admit_ridge`` requires the nearest
  foreign line to be outside ``ridge_clear * dc_hz(k)``. On a DREGON twin pair
  the difference is 6.6 % of the cells against 96 % — and the first gate throws
  away almost all of the comb's line energy, which is what phase 6c was blind
  with (``docs/experiments/telemetry-fitness.md`` § "Phase 6d").

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
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tracking.comb_displacement import (
    DisplacementConfig,
    demod_comb_bank,
    interloper_offsets_hz,
    nearest_interloper_hz,
    pulse_pair_bank,
)
from tracking.phase_noise import brickwall

__all__ = [
    "CONTROLS",
    "HIGHER_IS_BETTER",
    "Cells",
    "FitnessConfig",
    "FitnessScore",
    "Holdout",
    "LinePower",
    "apply_control",
    "bootstrap_scores",
    "default_holdouts",
    "line_bins",
    "line_masks",
    "line_power",
    "measure_tdoa",
    "residual_decompose",
    "score_cells",
    "score_window",
    "window_cells",
]

#: The components whose GOOD direction is up. Everything else in
#: :class:`FitnessScore` is a residual share or a mean square: less is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset({"ridge"})

#: The four controls of issue 17 section B (FLY124, the fourth, is a DATA
#: choice — every function here is recording-agnostic, so running the identical
#: procedure on FLY124 is one flag of the driver).
CONTROLS: tuple[str, ...] = ("none", "offcomb", "mismatch", "permute")

#: DREGON tachometer signature (``docs/experiments/dregon-telemetry-forensics.md``):
#: quantisation step at 80 rev/s, refresh rate, and the implied bound on the
#: quantisation part of a residual (half a step).
#: Median-to-mean factor of an exponential (periodogram) bin distribution. The
#: ridge floor is a MEDIAN over the annulus (robust to a line that slipped the
#: excision) divided by this, which makes it an unbiased mean density — so the
#: ridge of a pure-noise cell reads 0 dB rather than +1.6.
_LN2 = float(np.log(2.0))

TACH_STEP_REV_S = 0.269
TACH_REFRESH_HZ = 49.7
TACH_BOUND_REV_S = 0.5 * TACH_STEP_REV_S


# ---------------------------------------------------------------------------
# the line-against-a-local-floor reading (promoted from the phase-7 readout)


@dataclass(frozen=True)
class LinePower:
    """One fixed band read against a local floor.

    THE one implementation of "how much line is here" in this project. It was
    written for the phase-7 generator readout
    (``docs/experiments/generator-label-sensitivity.md``, where ``--self-test``
    pushes the true comb through it and reads back flat to 0.008 dB) and is
    promoted here so the tracking harness's ridge component and that readout
    are the same estimator rather than two of them.

    The band and the annulus are FIXED regions around a given centre. There is
    deliberately no peak search: a peak-pick inside a window of half-width ``W``
    returns about ``W / 2`` on pure noise and has already withdrawn two claims
    in this project (``docs/experiments/dregon-comb-displacement.md``).

    Attributes:
        total: floor-subtracted band power, each bin clipped at zero. The
            phase-7 reading — an amplitude, so a negative excursion is not an
            amplitude.
        raw: floor-subtracted band power without the clip. The unbiased one,
            which is what a RATIO against the floor needs.
        floor: the floor power DENSITY per bin.
        n_bins: bins in the band.
        spread_hz: rms offset of the clipped residual from the centre.
    """

    total: Any
    raw: Any
    floor: Any
    n_bins: int
    spread_hz: Any


def line_masks(
    freqs: np.ndarray,
    center: float,
    half_bw: float,
    *,
    annulus: tuple[float, float] = (3.0, 8.0),
    exclude: Sequence[tuple[float, float]] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """``(band, annulus)`` bin masks around ``center``.

    The annulus is ``annulus * half_bw`` on both sides, minus every interval of
    ``exclude`` (absolute frequencies, low/high). ``exclude`` is how a contested
    cell is scored instead of gated: the sibling's line is removed from the
    FLOOR region rather than the cell from the report.
    """
    off = np.asarray(freqs, dtype=np.float64) - center
    band = np.abs(off) <= half_bw
    ann = (np.abs(off) > annulus[0] * half_bw) & (np.abs(off) <= annulus[1] * half_bw)
    for lo, hi in exclude:
        ann &= (freqs < lo) | (freqs > hi)
    return band, ann


def line_power(
    power: np.ndarray,
    freqs: np.ndarray | None = None,
    center: float = 0.0,
    half_bw: float = 0.0,
    *,
    annulus: tuple[float, float] = (3.0, 8.0),
    exclude: Sequence[tuple[float, float]] = (),
    floor_scale: float = 1.0,
    masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> LinePower:
    """Read the band ``|f - center| <= half_bw`` of ``power`` against its annulus.

    ``power`` may carry any number of leading axes; the frequency axis is the
    last one. The floor is the MEDIAN power density of the annulus divided by
    ``floor_scale`` — a fixed region, so the estimate is unbiased whether or not
    a line is present. For an exponential (periodogram) bin distribution the
    median is ``ln 2`` of the mean, so ``floor_scale = ln 2`` turns the median
    into an unbiased mean-density estimate; the phase-7 default of 1.0 keeps the
    raw median, which is what a line-dominated band wants.
    """
    p = np.asarray(power, dtype=np.float64)
    band, ann = (
        masks
        if masks is not None
        else line_masks(
            np.asarray(freqs, dtype=np.float64),
            center,
            half_bw,
            annulus=annulus,
            exclude=exclude,
        )
    )
    n_bins = int(np.count_nonzero(band))
    if n_bins == 0:
        nan = np.full(p.shape[:-1], np.nan)
        return LinePower(nan, nan, nan, 0, nan)
    floor = (
        np.median(p[..., ann], axis=-1) / floor_scale
        if np.count_nonzero(ann)
        else np.zeros(p.shape[:-1])
    )
    inb = p[..., band] - floor[..., None]
    clipped = np.clip(inb, 0.0, None)
    total = clipped.sum(axis=-1)
    off = (np.asarray(freqs, dtype=np.float64) - center)[band] if freqs is not None else None
    spread = np.full(p.shape[:-1], np.nan)
    if off is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.sqrt((clipped * off**2).sum(axis=-1) / np.where(total > 0, total, np.nan))
    return LinePower(total, inb.sum(axis=-1), floor, n_bins, spread)


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
    #: The RIDGE gate (phase 6d). The ridge component reads DC against a floor
    #: it takes from elsewhere in the band, so it does not need an empty band —
    #: it needs the interferer RESOLVED away from DC and excised from the floor
    #: region. Harmonic ``k`` is admitted when the nearest foreign line is
    #: farther than ``ridge_clear * dc_hz(k)``, which on a DREGON twin pair
    #: (0.42 rev/s against a DC region of 0.10 rev/s) admits nearly everything
    #: above the block's own resolution floor.
    ridge_clear: float = 2.0
    #: The ridge floor annulus, as a fraction of the band half-width. It stays
    #: inside the band: outside it the zoom filter has rolled off and the floor
    #: there is the filter's shape, not the recording's noise. Its inner edge is
    #: pushed out to ``dc_hz + res_hz`` as well, so the floor region and the line
    #: region never overlap — at low ``k`` the resolution floor makes ``dc_hz``
    #: comparable to the band, and an annulus inside the line is a floor
    #: estimate made of the line.
    floor_frac: tuple[float, float] = (0.25, 0.9)
    #: A ridge cell needs a floor to be read against. Fewer annulus bins than
    #: this (BEFORE the interferer excisions, so the count does not follow the
    #: candidate) and the cell leaves the ridge gate.
    min_floor_bins: int = 4
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
    ridge: np.ndarray  # (C, K, B) dB, DC line density over the local floor
    line_pow: np.ndarray  # (C, K, B) floor-subtracted line power, absolute
    admit: np.ndarray  # (K, B) bool — components 1-3
    admit_ridge: np.ndarray  # (K, B) bool — component 4 (phase 6d)
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
    dc_hz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``((K, B) admit, (K, B) admit_ridge, (K, N) nearest-interloper Hz)``.

    Both gates read the REFERENCE only, which is what keeps the degrees of
    freedom identical across candidates, and both require ``k rate_ref`` to be
    inside the audio band.

    ``admit`` is the conditioning gate of issue 17 §D9: no other rotor's real
    line within ``gate_band_frac * B_k + guard_hz`` for at least
    ``min_clear_frac`` of the block's frames. Components 1-3 read the whole
    band, so they need the whole band clean.

    ``admit_ridge`` (phase 6d) is the same rule at the ridge component's own
    scale: no foreign line within ``ridge_clear * dc_hz(k)``. The ridge reads DC
    against a floor taken from elsewhere in the band, so an interferer that is
    RESOLVED away from DC (and excised from the floor region, see
    :func:`demod_cells`) costs it nothing. ``dc_hz`` is the near-DC half-width
    already floored at the block's own resolution, so a block too short to
    resolve the interferer gates itself out.
    """
    ks = cfg.ks
    nearest = nearest_interloper_hz(
        reference, reference[rot], rot, ks, f_max=cfg.f_max, min_rate=cfg.min_rate
    )
    clear = nearest > (cfg.gate_band_frac * band_hz[:, None] + cfg.guard_hz)
    clear_ridge = (
        nearest > cfg.ridge_clear * np.asarray(dc_hz, dtype=np.float64)[:, None]
        if dc_hz is not None
        else clear
    )
    stride_s = cfg.stride / cfg.sr
    admit = np.zeros((len(ks), len(blocks)), dtype=bool)
    admit_ridge = np.zeros_like(admit)
    for b, sl in enumerate(blocks):
        t0, t1 = sl.start * stride_s, (sl.stop - 1) * stride_s
        sel = (ft >= t0) & (ft <= t1)
        if not sel.any():
            sel = np.zeros(ft.size, dtype=bool)
            sel[min(int(t0 / max(ft[1] - ft[0], 1e-9)), ft.size - 1)] = True
        admit[:, b] = clear[:, sel].mean(axis=1) >= cfg.min_clear_frac
        admit_ridge[:, b] = clear_ridge[:, sel].mean(axis=1) >= cfg.min_clear_frac
    in_band = np.array([k * rate_ref <= cfg.f_max for k in ks], dtype=bool)
    admit &= in_band[:, None]
    admit_ridge &= in_band[:, None]
    return admit, admit_ridge, nearest


# ---------------------------------------------------------------------------
# the inter-microphone delay (issue 17 phase 6e) — the instrument that can
# actually resolve a fraction of a millisecond
#
# A propagation delay is a phase ramp across the comb, not a rate error. Sound
# from rotor ``j`` reaches microphone ``c`` after ``d_cj / 343``, so harmonic
# ``k`` arrives with phase ``-2 pi k rate_j d_cj / 343`` relative to the
# reference microphone, and the mean harmonic-to-harmonic phase INCREMENT of
# the cross-spectrum gives the inter-mic delay directly,
#
#     delay_cj = -mean_k wrap(psi_{k+1} - psi_k) / (2 pi rate_j)
#
# with no unwrapping ambiguity while the delay stays under ``1 / (2 rate)`` =
# 6.25 ms, against delays of at most 0.5 ms. The ridge of phase 6d cannot see
# this at all: through its ``tau dr/dt`` channel the whole DREGON inter-mic
# spread (0.156 ms) is a twentieth of the ridge window. This estimator can, and
# it reads slope 1.013 against the michaels rig geometry
# (``docs/experiments/telemetry-fitness.md`` § "Phase 6e").


def line_bins(
    pw: np.ndarray, freqs: np.ndarray, ks: np.ndarray, dr_step: float, n_step: int = 60
) -> tuple[float, np.ndarray]:
    """``(the residual rate offset in rev/s, (K,) bin index of each line)``.

    The comb's residual rate error is ONE number per (rotor, block): harmonic
    ``k`` sits at ``k dr`` Hz. So the offset is found by a joint scan that sums
    the harmonics' power along each candidate comb — never by a peak search per
    harmonic, which returns about half a window width on pure noise and which
    has already cost this campaign two published claims.
    """
    grid = np.arange(-n_step, n_step + 1) * dr_step  # rev/s
    kk = np.arange(ks.size)
    idx = np.stack(
        [np.abs(freqs[None, :] - (ks * dr)[:, None]).argmin(axis=1) for dr in grid]
    )  # (G, K)
    score = np.asarray([float(np.sum(pw[:, kk, idx[g]])) for g in range(grid.size)])
    best = int(np.argmax(score))
    return float(grid[best]), idx[best]


def measure_tdoa(
    audio: np.ndarray,
    ft: np.ndarray,
    carrier: np.ndarray,
    reference: np.ndarray,
    *,
    cfg: FitnessConfig,
    half: bool = False,
    dr_step: float = 0.02,
    ref_ch: int = 0,
    gate: bool = True,
) -> dict[str, Any]:
    """``delay_ms (R, C)`` relative to ``ref_ch``, plus the weights behind it.

    One rotor at a time. Per block the envelope spectrum is taken with the same
    Hann taper the ridge uses, the comb's own line bin is found jointly over
    harmonics, and the cross-spectrum against ``ref_ch`` is accumulated across
    blocks COHERENTLY — the geometry-induced phase is the one thing that does
    not change from block to block, so summing complex values averages
    everything else down.

    ``half=True`` demodulates at the half-integer comb, where no rotor line can
    exist: the null control of the same estimator.
    """
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))[: cfg.max_channels]
    ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    cand = np.atleast_2d(np.asarray(carrier, dtype=np.float64))
    ks = np.asarray(cfg.ks, dtype=np.float64)
    n_env = x.shape[-1] // cfg.stride
    blocks = _blocks(n_env, cfg)
    length = blocks[0].stop - blocks[0].start
    freqs = np.fft.fftfreq(length, d=1.0 / cfg.fs_env)
    taper = np.hanning(length)
    taper /= np.sqrt(np.mean(taper**2))

    delay_ms: list[list[float | None]] = []
    weight: list[list[float]] = []
    n_pairs: list[list[int]] = []
    for rot in range(cand.shape[0]):
        rate_ref = float(np.mean(ref[rot]))
        if rate_ref < cfg.min_rate:
            continue
        band = cfg.band_hz(rate_ref)
        z, _ = demod_comb_bank(
            x, cand[rot], ft, cfg.ks, cfg=cfg.displacement(), half=half, band_hz_k=band
        )
        res_hz = cfg.fs_env / length
        dc_hz = np.minimum(np.maximum(cfg.dc_revs * ks, cfg.dc_bins * res_hz), 0.9 * band)
        _, admit_ridge, _ = admission(
            ref, ft, rot, blocks, band, cfg=cfg, rate_ref=rate_ref, dc_hz=dc_hz
        )
        if not gate:
            # The ungated arm. DREGON's ridge gate leaves 4-8 harmonic pairs,
            # and the delay's own error falls as 1/sqrt(pairs), so the gate is
            # worth turning off ONCE to see whether coverage or the estimator
            # is the limit. It admits contested harmonics, whose phase is a
            # mixture of two rotors at two directions — a bias, reported as a
            # separate arm rather than folded into the measurement.
            admit_ridge = np.ones_like(admit_ridge)
        n_ch = z.shape[0]
        # (C, K) complex cross-spectrum against the reference microphone,
        # accumulated over blocks at each block's own line bin.
        cross = np.zeros((n_ch, ks.size), dtype=np.complex128)
        power = np.zeros(ks.size)
        for b, sl in enumerate(blocks):
            zb = np.asarray(z[:, :, sl], dtype=np.complex128) * taper
            Z = np.fft.fft(zb, axis=-1)  # (C, K, L)
            pw = np.abs(Z) ** 2
            _, bins = line_bins(pw, freqs, ks, dr_step)
            val = Z[:, np.arange(ks.size), bins]  # (C, K)
            ok = admit_ridge[:, b]
            cross += np.where(ok[None, :], val * np.conj(val[ref_ch])[None, :], 0.0)
            power += np.where(ok, np.abs(val[ref_ch]) ** 2, 0.0)
        psi = np.angle(cross)  # (C, K)
        w = np.minimum(np.abs(cross)[:, :-1], np.abs(cross)[:, 1:])  # (C, K-1)
        # Harmonic-to-harmonic increment: 2 pi rate * delay per unit k, wrapped
        # into (-pi, pi]. Unambiguous while |delay| < 1 / (2 rate) = 6.25 ms.
        dpsi = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1])))
        good = np.asarray(power > 0)[:-1] & np.asarray(power > 0)[1:]
        w = w * good[None, :]
        tot = w.sum(axis=1)
        mean_dpsi = np.where(tot > 0, (w * dpsi).sum(axis=1) / np.maximum(tot, 1e-300), np.nan)
        delay = -mean_dpsi / (2.0 * np.pi * rate_ref)
        delay_ms.append([None if not np.isfinite(v) else round(float(v) * 1e3, 5) for v in delay])
        weight.append([round(float(v), 6) for v in tot])
        n_pairs.append([int(v) for v in (w > 0).sum(axis=1)])
    return {
        "delay_ms": delay_ms,
        "weight": weight,
        "n_pairs": n_pairs,
        "ref_ch": ref_ch,
        "half": bool(half),
    }


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

    #: Hann on the ridge spectrum only. A rectangular block leaks a strong DC
    #: line at -13 dB into the first sidelobe and decays 6 dB/octave, which puts
    #: the LINE into the floor region and compresses the very ratio the
    #: component reports; Hann is -31 dB and 18 dB/octave. Components 1-3 are
    #: shares of the same total and keep the untapered spectrum.
    taper = np.hanning(length)
    taper /= np.sqrt(np.mean(taper**2))

    # The ridge component's floor region, per harmonic. The annulus is a fixed
    # fraction of the band; the excisions are the foreign lines that land in it.
    # Their POSITIONS are (reference fact) - (this carrier), so a candidate whose
    # band sits elsewhere excises the interferer where it actually is — the only
    # candidate-dependent quantity here is the one that defines the candidate.
    rate_car = float(np.mean(np.asarray(carrier, dtype=np.float64)))
    offs = interloper_offsets_hz(
        np.mean(np.atleast_2d(reference), axis=1),
        rate_car,
        skip,
        ks,
        band_hz=band_hz,
        half=half,
        f_max=cfg.f_max,
        min_rate=cfg.min_rate,
    )
    ridge_masks: list[tuple[np.ndarray, np.ndarray]] = []
    floor_ok = np.zeros(n_k, dtype=bool)
    for i in range(n_k):
        lo_hz = max(cfg.floor_frac[0] * band_hz[i], dc_hz[i] + res_hz)
        hi_hz = cfg.floor_frac[1] * band_hz[i]
        d = max(dc_hz[i], 1e-12)
        band_m, ann_m = line_masks(
            freqs, 0.0, float(dc_hz[i]), annulus=(float(lo_hz / d), float(hi_hz / d))
        )
        floor_ok[i] = np.count_nonzero(ann_m) >= cfg.min_floor_bins
        # The excisions are applied on top, and only where something survives:
        # a floor made of the sibling's line is worse than a wider floor, but an
        # EMPTY floor is worse than both, and the fallback is the conservative
        # direction (a contaminated floor can only lower the ridge).
        keep = ann_m.copy()
        for o in offs[i]:
            keep &= (freqs < o - dc_hz[i]) | (freqs > o + dc_hz[i])
        ridge_masks.append((band_m, keep if np.count_nonzero(keep) >= 2 else ann_m))

    admit, admit_ridge, nearest = admission(
        reference, ft, skip, blocks, band_hz, cfg=cfg, rate_ref=rate_ref, dc_hz=dc_hz
    )
    admit_ridge &= floor_ok[:, None]

    bb = np.empty((n_ch, n_k, n_b))
    ph = np.empty((n_ch, n_k, n_b))
    ro = np.empty((n_ch, n_k, n_b))
    pp = np.empty((n_ch, n_k, n_b))
    coh = np.empty((n_ch, n_k, n_b))
    snr = np.empty((n_ch, n_k, n_b))
    ridge = np.empty((n_ch, n_k, n_b))
    line_pw = np.empty((n_ch, n_k, n_b))

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
        # 4. ridge concentration: DC line density over the local floor, in dB
        pwt = np.abs(np.fft.fft(zb * taper, axis=-1)) ** 2
        for i in range(n_k):
            lp = line_power(pwt[:, i, :], masks=ridge_masks[i], floor_scale=_LN2)
            dens = (lp.raw + lp.floor * lp.n_bins) / max(lp.n_bins, 1)
            ridge[:, i, b] = 10.0 * np.log10(
                np.maximum(dens, 1e-300) / np.maximum(lp.floor, 1e-300)
            )
            line_pw[:, i, b] = lp.total

    # A harmonic with no floor region has no ridge reading, and NaN is the only
    # honest value: a ratio against an empty annulus is a division by the clamp.
    ridge[:, ~floor_ok, :] = np.nan
    line_pw[:, ~floor_ok, :] = np.nan

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
        ridge=ridge,
        line_pow=line_pw,
        admit=admit,
        admit_ridge=admit_ridge,
        block_t=block_t,
        diag={
            "skip": int(skip),
            "half": bool(half),
            "n_env": int(n_env),
            "block_len": int(length),
            "res_hz": round(float(res_hz), 4),
            "dc_hz": np.round(dc_hz, 3).tolist(),
            "admit_frac": round(float(admit.mean()), 4),
            "admit_frac_ridge": round(float(admit_ridge.mean()), 4),
            # What share of the comb's line energy the phase-6c gate could see.
            # This is the number that made phase 6c blind, so it travels with
            # every unit rather than living in a one-off analysis.
            "line_share_gated": _line_share(line_pw, admit),
            "line_share_ridge": _line_share(line_pw, admit_ridge),
            "median_nearest_hz": round(float(np.median(nearest[np.isfinite(nearest)])), 2)
            if np.isfinite(nearest).any()
            else None,
        },
    )


def _mean_or_none(vals: Sequence[Any]) -> float | None:
    v = np.asarray([x for x in vals if isinstance(x, (int, float))], dtype=np.float64)
    v = v[np.isfinite(v)]
    return round(float(v.mean()), 4) if v.size else None


def _line_share(line_pw: np.ndarray, mask: np.ndarray) -> float | None:
    """Share of the window's total floor-subtracted line power inside ``mask``."""
    tot = float(np.nansum(line_pw))
    if not np.isfinite(tot) or tot <= 0:
        return None
    return round(float(np.nansum(line_pw * mask[None, :, :]) / tot), 4)


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
    #: dB, and the ONE component where more is better (:data:`HIGHER_IS_BETTER`).
    ridge: float = float("nan")
    #: Cells behind ``ridge``. It has its own gate, so it is its own count.
    n_cells_ridge: int = 0
    per_rotor: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "broadband": _r(self.broadband, 6),
            "phase_noise": _r(self.phase_noise, 6),
            "roughness": _r(self.roughness, 6),
            "ridge": _r(self.ridge, 5),
            "pp_dr": _r(self.pp_dr, 5),
            "pp_abs": _r(self.pp_abs, 5),
            "snr_median": _r(self.snr_median, 4),
            "n_cells": self.n_cells,
            "n_cells_ridge": self.n_cells_ridge,
            "per_rotor": self.per_rotor,
        }

    def component(self, name: str) -> float:
        return {
            "broadband": self.broadband,
            "phase_noise": self.phase_noise,
            "roughness": self.roughness,
            "ridge": self.ridge,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(uniform, k^exp, ridge-gated)`` weights over the scored cells.

    The third uses the ridge gate rather than the conditioning gate — the two
    statistics need different protection, see :func:`admission`.
    """
    n_ch, n_k, n_b = cells.shape
    base = holdout.score_mask(n_ch, cells.ks, n_b).astype(np.float64)
    if extra is not None:
        base = base * extra
    w = base * cells.admit[None, :, :].astype(np.float64)
    wr = base * cells.admit_ridge[None, :, :].astype(np.float64)
    kw = np.asarray(cells.ks, dtype=np.float64)[None, :, None] ** cfg.phase_weight_exp
    return w, w * kw, wr


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
    acc: dict[str, list[float]] = {k: [] for k in ("bb", "ph", "ro", "rd", "pp", "ppa", "snr")}
    n_cells = 0
    n_cells_ridge = 0
    for i, c in enumerate(cells):
        w, kw, wr = _cell_weights(c, holdout, cfg, None if extra is None else extra[i])
        vals = {
            "bb": _wmean(c.broadband, w),
            "ph": _wmean(c.phase_ms, kw),
            "ro": _wmean(c.roughness, w),
            "rd": _wmean(c.ridge, wr),
            "pp": _wmean(c.pp_dr, kw),
            "ppa": _wmean(np.abs(c.pp_dr), kw),
            "snr": _wmean(c.snr, w),
        }
        n_cells += int(np.count_nonzero(w > 0))
        n_cells_ridge += int(np.count_nonzero(wr > 0))
        for k, v in vals.items():
            acc[k].append(v)
        per_rotor[str(c.rotor)] = {
            "broadband": _r(vals["bb"], 6),
            "phase_noise": _r(vals["ph"], 6),
            "roughness": _r(vals["ro"], 6),
            "ridge": _r(vals["rd"], 5),
            "pp_dr": _r(vals["pp"], 5),
            "n_cells": int(np.count_nonzero(w > 0)),
            "n_cells_ridge": int(np.count_nonzero(wr > 0)),
        }

    def m(key: str) -> float:
        v = np.asarray(acc[key], dtype=np.float64)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    return FitnessScore(
        broadband=m("bb"),
        phase_noise=m("ph"),
        roughness=m("ro"),
        ridge=m("rd"),
        pp_dr=m("pp"),
        pp_abs=m("ppa"),
        snr_median=m("snr"),
        n_cells=n_cells,
        n_cells_ridge=n_cells_ridge,
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
        "ridge": s.ridge,
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
            "admit_frac_ridge": round(float(np.mean([c.admit_ridge.mean() for c in cells])), 4),
            "line_share_gated": _mean_or_none([c.diag.get("line_share_gated") for c in cells]),
            "line_share_ridge": _mean_or_none([c.diag.get("line_share_ridge") for c in cells]),
            "rate_ref": [round(c.rate_ref, 3) for c in cells],
            "diag": [c.diag for c in cells],
        },
    }
