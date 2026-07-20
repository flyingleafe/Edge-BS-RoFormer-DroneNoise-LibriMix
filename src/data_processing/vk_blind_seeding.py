"""Blind seeding v2 for the coupled VK tracker — shared-comb prior.

Implements ``docs/vk-order-tracking-design.md`` §7: four independently
composable arms on top of the whitened base-speed comb scan that
``scripts/vk_blind_annotation.py`` validated (its ``_whitened_spec`` /
``_comb_scan`` / ``_scan_peaks`` now delegate here — this module is the
shared home of that machinery):

- **arm T** (§7.1) — shared-template matched-filter scan: estimate the
  drone's harmonic amplitude profile ``â_k`` from the most confident rotor
  (flat scan → LP-demodulated envelope magnitudes, ``â_k = median_t |z_k(t)|``
  normalized), then re-scan with the template-weighted score
  ``S(f0) = Σ_k â_k W(k f0) / Σ_k â_k`` instead of the flat mean.
- **arm C** (§7.2) — alias/completeness rejection: a candidate base whose
  fraction of teeth with energy above the local floor is below ``c_min``
  (default 0.5) is an alias (the FLY124 (2/3)×91 ≈ 60.7 comb has energy only
  at every 3rd tooth), a true weak comb passes.
- **arm N** (§7.3) — rotor-count prior with duplicate seeding: dedup scan
  peaks at ``dedup_rps`` (4 rev/s), and when fewer than R peaks survive, seed
  the missing rotors AT the strongest surviving bases (±``split_nudge``);
  the coupled solve + ``_break_symmetry`` separates true twins. Never invent
  an extra base; never leave a rotor unseeded.
- **arm K** (§7.4) — auto-knobs: per-recording ``update_gate`` from the
  noise floor of the gate's own statistic (periodogram peak/median ratio of
  detuned-comb demod bands with predicted comb lines masked, μ + 3σ) and
  capture ``bw_hz`` from the scan-peak width. Replaces the two hand-retuned
  per-regime knobs (gate 30↔8, bw 1.5↔7).

Documented interpretations of §7 (loose points in the sketch):

1. ``W`` is the whitened **log**-magnitude spectrum (running-median-over-
   frequency subtracted), not linear magnitude — consistent with the
   validated flat scan, which becomes the ``â_k = const`` special case of the
   matched filter; the L1 normalization makes matched and flat scores
   commensurate (both are weighted means over valid teeth).
2. §7.1's "capture the single most confident rotor" is realized as
   narrowband demodulation at the flat-scan primary base (the same quantity
   a 1-rotor VK envelope solve returns, minus the smoothing prior) — no full
   VK solve inside seeding.
3. §7.4 says "``update_gate`` from the scan-score noise floor", but
   ``VKConfig.update_gate`` thresholds a *periodogram peak/median ratio*, not
   a scan score — so the calibration measures the null distribution of that
   exact statistic at detuned bases (with predicted true-comb lines masked
   out of the periodogram, since for base speeds below ``fs_env`` every demod
   band contains some true harmonic), gate = μ + 3σ, clipped.

Conventions match ``vk_tracking``: trajectories in rev/s, harmonics at
``k * r`` Hz, audio ``(T,)`` or ``(C, T)``. Seeds carry no rotor identity
(downstream metrics are PIT-aligned).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data_processing.vk_tracking import VKConfig, demodulate

__all__ = [
    "SeedConfig",
    "SeedResult",
    "whitened_logmag",
    "comb_scan",
    "scan_peaks",
    "estimate_template",
    "completeness",
    "count_prior",
    "auto_knobs",
    "blind_seed",
]


@dataclass(frozen=True)
class SeedConfig:
    """Knobs for blind seeding (defaults = the validated blind-scan values)."""

    # base-speed scan (identical to scripts/vk_blind_annotation.py's scan)
    scan_lo: float = 30.0
    scan_hi: float = 120.0
    scan_step: float = 0.05
    k_scan: int = 40  # harmonics sampled by the scan
    f_min: float = 60.0
    f_max: float = 6000.0
    scan_f_max: float | None = 1200.0  # scan/completeness tooth-band cap (Hz).
    # A k-count-only cap makes a small base average its 40 teeth over the
    # energetic low band while a large base dilutes into the dead high band —
    # the small-base subharmonic bias that put FLY124's primary at 30.35
    # (= 91.8/3). Capping the BAND scores every base on comparable spectrum;
    # None restores the historical k-only behaviour (vk_blind_annotation).
    whiten_hz: float = 150.0  # running-median window subtracted from log-mag
    peak_min_sep: float = 1.5  # rev/s between scan peaks
    peak_prominence_frac: float = 0.03  # of the score range
    octave_rel: float = 0.9  # prefer the half base when it scores >= this
    octave_tol: float = 1.0  # rev/s tolerance for the half-base peak match
    octave_up_rel: float = 0.7  # subharmonic-up promotion: when a peak near
    # m*base (m = 2, 3) scores >= this fraction of the top peak, the larger
    # base is the physical rotor (the submultiple inherits every m-th tooth)
    # baseline (legacy) R=4 init — two rotors per harmonically-unrelated peak
    harm_guard: float = 1.5  # rev/s: 2nd peak must not be a half/double of 1st
    pair_nudge: float = 0.5  # rev/s: 2 rotors per chosen peak at peak -/+ nudge
    blind_offsets: tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5)  # 1-peak fallback
    # arm T — shared-template matched filter
    template_floor: float = 0.02  # â_k < floor * max(â) → unvalidated tooth
    t_conf_min: float = 6.0  # template gate: min primary-peak z-score (vs the
    # scan-score distribution) — a low-contrast primary yields a noise
    # template that poisons the matched re-scan (FLY124/whitenoise failure)
    t_cos_min: float = 0.7  # template gate: min cosine between the top-2
    # confident candidates' templates (shared-blades premise check); on
    # disagreement arm T falls back to the flat scan
    template_fs_env: float = 40.0  # envelope rate for template demod: the
    # 0.45 * fs_env LP cutoff (18 Hz) must exclude the *neighbouring* tooth
    # (>= scan_lo = 30 Hz away) or its envelope bleeds into every weak tooth
    # arm C — completeness
    c_min: float = 0.5  # min fraction of teeth above the local floor
    tooth_sigma: float = 2.0  # tooth present iff whitened value > this * σ_floor
    tooth_quantile: float = 0.6  # per-frequency time-quantile of the whitened
    # spectrogram used as the completeness presence spectrum: high enough to
    # catch teeth a wandering line only visits, low enough that an alias's
    # occasionally-grazed teeth do not count (0.8 lets 3:2 aliases through)
    # arm N — rotor-count prior
    dedup_rps: float = 4.0  # near-duplicate peak suppression (rev/s)
    split_nudge: float = 0.1  # duplicate-seed nudge (± around the pair base)
    # arm K — auto-knobs
    gate_offsets: tuple[float, ...] = (-3.7, -2.3, 2.3, 3.7)  # detuned bases
    gate_k_max: int = 30  # harmonics used for the gate calibration
    gate_mask_hz: float = 3.0  # base half-width of the predicted-line mask
    gate_mask_cap_hz: float = 8.0  # cap on the k * wander widening of the
    # mask (a smeared line raises the periodogram median rather than its
    # peak; without the cap, dense candidate sets mask entire demod bands)
    gate_max_masked: float = 0.7  # drop tracks with more of the band masked
    gate_clip: tuple[float, float] = (5.0, 60.0)
    bw_k_ref: float = 6.0  # bw_hz ≈ scan-peak FWHM (rev/s) * this
    bw_clip: tuple[float, float] = (1.5, 8.0)


@dataclass
class SeedResult:
    """Output of :func:`blind_seed`."""

    bases: np.ndarray  # (R,) sorted seed base speeds (rev/s); no rotor identity
    candidates: list[dict[str, Any]]  # per scan candidate: base, scores, flags
    template: np.ndarray | None  # (k_scan,) normalized â_k (arm T), else None
    update_gate: float | None  # arm K, else None
    bw_hz: float | None  # arm K, else None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# scan machinery (shared with scripts/vk_blind_annotation.py)


def whitened_logmag(
    audio: np.ndarray, fs: float, cfg: SeedConfig | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """``(F, N)`` channel-averaged whitened log-mag + ``(bin_hz, frame_times)``.

    The raw comb score is envelope-dominated on real drone audio (low-
    frequency rumble makes smaller bases score higher) — whitening subtracts
    a running median over frequency (``cfg.whiten_hz`` window) so comb scores
    measure line evidence above the local background.
    """
    from scipy.ndimage import median_filter

    from data_processing.rps_refinement import RefineConfig, compute_logmag

    cfg = cfg or SeedConfig()
    rcfg = RefineConfig(sample_rate=int(round(fs)), device="cpu")
    spec = compute_logmag(audio, rcfg)
    lm = spec.logmag.cpu().numpy()  # (C, F, N)
    bin_hz = float(spec.bin_hz)
    win = int(round(cfg.whiten_hz / bin_hz)) | 1
    white = (lm - median_filter(lm, size=(1, win, 1))).mean(axis=0)  # (F, N)
    st = np.asarray(spec.frame_times, dtype=np.float64)
    return white, bin_hz, st


def _tooth_values(
    white_vec: np.ndarray, bin_hz: float, base: float, cfg: SeedConfig
) -> tuple[np.ndarray, np.ndarray]:
    """``(K,)`` interpolated whitened values at ``k * base`` + validity mask."""
    n_f = len(white_vec)
    freqs_max = (n_f - 1) * bin_hz
    f_hi = min(cfg.f_max, freqs_max)
    if cfg.scan_f_max is not None:
        f_hi = min(f_hi, cfg.scan_f_max)
    ks = np.arange(1, cfg.k_scan + 1)
    f = ks * base
    valid = (f >= cfg.f_min) & (f <= f_hi)
    idx = np.clip(f, 0.0, freqs_max) / bin_hz
    j = np.floor(idx).astype(int)
    frac = idx - j
    vals = (1 - frac) * white_vec[j] + frac * white_vec[np.minimum(j + 1, n_f - 1)]
    return np.where(valid, vals, 0.0), valid


def comb_scan(
    white_vec: np.ndarray,
    bin_hz: float,
    grid: np.ndarray,
    cfg: SeedConfig | None = None,
    *,
    template: np.ndarray | None = None,
) -> np.ndarray:
    """Comb score of each constant base in ``grid``.

    Flat scan (``template=None``): mean whitened value over valid teeth —
    numerically identical to the validated ``vk_blind_annotation`` scan.
    Matched filter (arm T): template-weighted mean
    ``S(f0) = Σ_k â_k W(k f0) / Σ_k â_k`` over valid teeth.
    """
    cfg = cfg or SeedConfig()
    w_all = (
        np.ones(cfg.k_scan)
        if template is None
        else np.asarray(template, dtype=np.float64)[: cfg.k_scan]
    )
    scores = np.empty(len(grid))
    for gi, b in enumerate(grid):
        vals, valid = _tooth_values(white_vec, bin_hz, float(b), cfg)
        w = w_all[: len(valid)] * valid
        wsum = float(w.sum())
        scores[gi] = float((w * vals).sum() / wsum) if wsum > 0 else -np.inf
    return scores


def scan_peaks(grid: np.ndarray, scores: np.ndarray, cfg: SeedConfig | None = None) -> np.ndarray:
    """Indices of local maxima (>= ``peak_min_sep`` apart), fallback to argmax."""
    from scipy.signal import find_peaks

    cfg = cfg or SeedConfig()
    step = float(grid[1] - grid[0]) if len(grid) > 1 else cfg.scan_step
    rng = float(scores.max() - scores.min())
    idx_pk, _ = find_peaks(
        scores,
        prominence=max(1e-4, cfg.peak_prominence_frac * rng),
        distance=max(1, int(cfg.peak_min_sep / step)),
    )
    if len(idx_pk) == 0:
        idx_pk = np.array([int(np.argmax(scores))])
    return idx_pk


def _pick_primary(
    grid: np.ndarray, peak_speeds: np.ndarray, peak_scores: np.ndarray, cfg: SeedConfig
) -> tuple[float, bool, bool]:
    """Top peak with subharmonic-up promotion + octave-down disambiguation.

    Up promotion first: an integer submultiple of the true base inherits
    every m-th tooth of the true comb (plus whatever its extra low teeth
    graze) and can top the flat scan — FLY124's 30.35 = 91.8/3. When a peak
    near ``m * base`` (m = 3, 2) scores at least ``octave_up_rel`` of the
    top, the larger base is the physical rotor. Then the octave-down guard
    (a true base at b/2 implies the b comb is its even-harmonic subset).
    Returns ``(base, octave_down, promoted_up)``.
    """
    base, score0 = float(peak_speeds[0]), float(peak_scores[0])
    promoted = False
    for m in (3.0, 2.0):
        up = m * base
        if up > float(grid[-1]) + cfg.harm_guard:
            continue
        d = np.abs(peak_speeds - up)
        j = int(np.argmin(d))
        if float(d[j]) <= cfg.harm_guard and float(peak_scores[j]) >= cfg.octave_up_rel * score0:
            base, promoted = float(peak_speeds[j]), True
            break
    octave = False
    half = base / 2.0
    if half >= float(grid[0]):
        d = np.abs(peak_speeds - half)
        j = int(np.argmin(d))
        base_score = float(peak_scores[int(np.argmin(np.abs(peak_speeds - base)))])
        if float(d[j]) <= cfg.octave_tol and float(peak_scores[j]) >= cfg.octave_rel * base_score:
            base, octave = float(peak_speeds[j]), True
    return base, octave, promoted


# ---------------------------------------------------------------------------
# arm T — shared harmonic-amplitude template


def estimate_template(audio: np.ndarray, fs: float, base: float, cfg: SeedConfig) -> np.ndarray:
    """``(k_scan,)`` normalized harmonic amplitude profile ``â_k`` (§7.1).

    Demodulates every harmonic of the (captured) primary base and takes the
    time-median envelope magnitude, averaged over channels:
    ``â_k = median_t |z_k(t)| / Σ_k median_t |z_k(t)|`` over the validated
    band; teeth outside ``[f_min, 0.45 fs]`` or below
    ``template_floor * max(â)`` are zeroed (unvalidated).
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    ks = np.arange(1, cfg.k_scan + 1, dtype=np.float64)
    t = np.arange(y.shape[-1], dtype=np.float64) / fs
    phase = ks[:, None] * (2.0 * np.pi * base) * t[None, :]
    vkcfg = VKConfig(fs=float(fs), fs_env=cfg.template_fs_env)
    z = demodulate(y, phase, vkcfg)  # (C, K, T_env)
    amp = np.median(np.abs(z), axis=-1).mean(axis=0)  # (K,)
    f = ks * base
    amp[(f < cfg.f_min) | (f > min(cfg.f_max, 0.45 * fs))] = 0.0
    if amp.max() > 0:
        amp[amp < cfg.template_floor * amp.max()] = 0.0
    s = float(amp.sum())
    return amp / s if s > 0 else amp


# ---------------------------------------------------------------------------
# arm C — alias/completeness rejection


def completeness(
    white_vec: np.ndarray,
    bin_hz: float,
    base: float,
    cfg: SeedConfig | None = None,
    *,
    template: np.ndarray | None = None,
) -> tuple[float, int, int]:
    """Fraction of the comb's teeth with energy above the local floor (§7.2).

    A tooth is present when its value in the supplied presence spectrum
    exceeds the spectrum's median by ``tooth_sigma`` robust sigmas. The
    presence spectrum should be a per-frequency time-*quantile* of the
    whitened spectrogram for wandering audio (see :func:`blind_seed`, which
    passes the ``tooth_quantile`` slice): under base wander the time-mean at
    a high-k tooth frequency smears toward the floor even for a true comb —
    a quantile catches "the line visits this tooth", which is what
    completeness means. A time-mean vector is fine for near-constant bases.
    With a template, only teeth the template expects (``â_k > 0``) are
    counted — a blade-pass-dominated drone legitimately has silent odd
    teeth. Returns ``(fraction, n_present, n_teeth)``.
    """
    cfg = cfg or SeedConfig()
    center = float(np.median(white_vec))
    sigma = 1.4826 * float(np.median(np.abs(white_vec - center)))
    vals, valid = _tooth_values(white_vec, bin_hz, base, cfg)
    counted = valid.copy()
    if template is not None:
        counted &= np.asarray(template, dtype=np.float64)[: len(counted)] > 0
    n_teeth = int(counted.sum())
    if n_teeth == 0:
        return 0.0, 0, 0
    present = counted & (vals > center + cfg.tooth_sigma * max(sigma, 1e-12))
    return float(present.sum() / n_teeth), int(present.sum()), n_teeth


# ---------------------------------------------------------------------------
# arm N — rotor-count prior with duplicate seeding


def count_prior(
    bases: np.ndarray | Iterable[float],
    scores: np.ndarray | Iterable[float],
    n_rotors: int,
    cfg: SeedConfig | None = None,
) -> np.ndarray:
    """``(R,)`` sorted seeds from candidate peaks under the count prior (§7.3).

    Dedup at ``dedup_rps`` (score order), trim to R; if fewer survive, seed
    the missing rotors AT the strongest surviving bases with alternating
    ±``split_nudge`` (growing every full duplication cycle) — the coupled
    solve + ``_break_symmetry`` separates true twins from there. Never a base
    beyond the surviving peaks; never an unseeded rotor.
    """
    cfg = cfg or SeedConfig()
    b = np.asarray(list(bases), dtype=np.float64)
    s = np.asarray(list(scores), dtype=np.float64)
    if len(b) == 0:
        raise ValueError("count_prior needs at least one candidate base")
    kept: list[float] = []
    for i in np.argsort(s)[::-1]:
        bi = float(b[i])
        if all(abs(bi - kb) >= cfg.dedup_rps for kb in kept):
            kept.append(bi)
        if len(kept) == n_rotors:
            break
    seeds = list(kept)
    dup = 0
    while len(seeds) < n_rotors:
        anchor = kept[dup % len(kept)]  # strongest surviving bases first
        cycle = dup // len(kept) + 1
        sign = 1.0 if cycle % 2 == 1 else -1.0
        seeds.append(anchor + sign * cfg.split_nudge * ((cycle + 1) // 2))
        dup += 1
    return np.sort(np.asarray(seeds))


# ---------------------------------------------------------------------------
# arm K — auto-knobs


def _gate_null_ratios(
    audio: np.ndarray,
    fs: float,
    anchor_bases: np.ndarray,
    mask_bases: np.ndarray,
    sigma_wander: float,
    cfg: SeedConfig,
) -> list[float]:
    """Null samples of the ``update_gate`` statistic at detuned bases.

    For each detuned base (``anchor ± gate_offsets``) the audio is
    demodulated per harmonic exactly as ``vk_tracking._freq_update`` sees it;
    periodogram bins within a predicted comb line's neighbourhood
    (``gate_mask_hz + k * sigma_wander`` — the lines wander with the rotors;
    lines of EVERY detected comb in ``mask_bases``) are masked, and the
    max/median ratio of the rest is the line-free null. One sample per
    detuned base: the max over its usable tracks and channels (the gate
    thresholds a max, so the null must be of the max).
    """
    y = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    vkcfg = VKConfig(fs=float(fs))
    stride = max(1, int(round(fs / vkcfg.fs_env)))
    fs_env = fs / stride
    t = np.arange(y.shape[-1], dtype=np.float64) / fs
    ks = np.arange(1, min(cfg.k_scan, cfg.gate_k_max) + 1)
    ratios: list[float] = []
    for b0 in anchor_bases:
        for off in cfg.gate_offsets:
            bd = float(b0) + float(off)
            if bd <= 0:
                continue
            f_teeth = ks * bd
            sel = (f_teeth >= cfg.f_min) & (f_teeth <= min(cfg.f_max, 0.45 * fs))
            ks_use = ks[sel]
            if len(ks_use) == 0:
                continue
            phase = ks_use[:, None].astype(np.float64) * (2.0 * np.pi * bd) * t[None, :]
            z = demodulate(y, phase, vkcfg)  # (C, K_use, T_env)
            n_env = z.shape[-1]
            win = np.hanning(n_env)
            freqs = np.fft.fftfreq(n_env, d=1.0 / fs_env)
            best = 0.0
            usable = False
            for ti, k in enumerate(ks_use):
                mask = np.zeros(n_env, dtype=bool)
                half = cfg.gate_mask_hz + min(float(k) * sigma_wander, cfg.gate_mask_cap_hz)
                for bt in mask_bases:
                    k_near = int(round(k * bd / float(bt)))
                    for k2 in range(max(1, k_near - 2), k_near + 3):
                        delta = k2 * float(bt) - k * bd
                        if abs(delta) <= 0.6 * fs_env:
                            mask |= np.abs(freqs - delta) < half
                if float(mask.mean()) > cfg.gate_max_masked:
                    continue
                pxx = np.abs(np.fft.fft(z[:, ti] * win[None, :], axis=-1)) ** 2
                pxx = pxx[:, ~mask]
                med = np.median(pxx, axis=-1)
                best = max(best, float(np.max(pxx.max(axis=-1) / np.maximum(med, 1e-30))))
                usable = True
            if usable:
                ratios.append(best)
    return ratios


def auto_knobs(
    audio: np.ndarray,
    fs: float,
    bases: np.ndarray,
    grid: np.ndarray,
    scores: np.ndarray,
    primary_idx: int,
    cfg: SeedConfig | None = None,
    *,
    mask_bases: np.ndarray | None = None,
) -> tuple[float | None, float, dict[str, Any]]:
    """Per-recording ``(update_gate, bw_hz, diagnostics)`` (§7.4).

    ``update_gate`` = μ + 3σ of the gate statistic's null (see
    :func:`_gate_null_ratios`; detune anchors = ``bases``, predicted-line
    mask = ``mask_bases`` — pass every detected comb there, defaults to the
    anchors), clipped to ``gate_clip``; ``bw_hz`` = scan-peak FWHM (rev/s) ×
    ``bw_k_ref``, clipped to ``bw_clip`` — a wide peak means the base
    wanders, so the capture band must admit ``k · wander``.
    """
    from scipy.signal import peak_widths

    cfg = cfg or SeedConfig()
    step = float(grid[1] - grid[0]) if len(grid) > 1 else cfg.scan_step
    try:
        widths = peak_widths(scores, [int(primary_idx)], rel_height=0.5)[0]
        fwhm_rps = float(widths[0]) * step
    except ValueError:  # primary not a strict local max (degenerate scan)
        fwhm_rps = 2.0 * step
    bw_hz = float(np.clip(fwhm_rps * cfg.bw_k_ref, cfg.bw_clip[0], cfg.bw_clip[1]))

    uniq: list[float] = []
    for b in np.sort(np.asarray(bases, dtype=np.float64)):
        if all(abs(float(b) - u) > 1.0 for u in uniq):
            uniq.append(float(b))
    masks = np.asarray(uniq) if mask_bases is None else np.asarray(mask_bases, dtype=np.float64)
    ratios = _gate_null_ratios(audio, fs, np.asarray(uniq), masks, fwhm_rps / 2.0, cfg)
    gate: float | None = None
    if ratios:
        mu, sd = float(np.mean(ratios)), float(np.std(ratios))
        gate = float(np.clip(mu + 3.0 * sd, cfg.gate_clip[0], cfg.gate_clip[1]))
    diag = {"fwhm_rps": fwhm_rps, "gate_null_ratios": ratios, "gate_bases": uniq}
    return gate, bw_hz, diag


# ---------------------------------------------------------------------------
# entry point


def _harmonic_alias_filter(
    bases: np.ndarray, scores: np.ndarray, cfg: SeedConfig, primary: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Drop candidates at a low-order ratio (3:2, 2 or 3) of a stronger one.

    The legacy init applied the half/double guard when picking its second
    peak (a peak at b/2 of a stronger peak is its subharmonic alias, not a
    rotor); the count-prior path (arm N) needs the same guard over the whole
    candidate list — extended to ratio 3 (reachable inside the scan band,
    e.g. 91/3 ≈ 30.3) and to 3:2 (FLY124's 60.7 ≈ (2/3)×91; a 3:2-related
    comb shares half/a-third of its teeth and sits exactly on arm C's
    completeness knife-edge). Rotors of one drone in cruise lie within
    ~1.25× of each other (DREGON 75/86, FLY124 74/91), so a 3:2 ratio
    between scan candidates is an alias prior, not a plausible rotor pair.
    ``primary`` (the elected base, possibly promoted over a higher-scoring
    submultiple) is kept first so its subharmonics are shadowed even when
    they out-score it. Relatedness is checked against every *stronger*
    candidate — kept or itself dropped — because dropping an alias must not
    un-shadow the alias's own multiples (e.g. 57.6 = (2/3)x86 dropped, its
    double 115.2 must stay dropped too).
    """
    order = list(np.argsort(scores)[::-1])
    if primary is not None and len(bases):
        pi = int(np.argmin(np.abs(np.asarray(bases) - primary)))
        if abs(float(bases[pi]) - primary) <= cfg.peak_min_sep and pi in order:
            order.remove(pi)
            order.insert(0, pi)
    keep: list[int] = []
    seen: list[int] = []
    for i in order:
        b = float(bases[i])
        related = any(
            abs(m * b - float(bases[j])) < cfg.harm_guard
            or abs(b - m * float(bases[j])) < cfg.harm_guard
            for j in seen
            for m in (1.5, 2.0, 3.0)
        )
        seen.append(int(i))
        if not related:
            keep.append(int(i))
    keep_sorted = sorted(keep)
    return bases[keep_sorted], scores[keep_sorted]


def _legacy_init4(
    base: float, cand_bases: np.ndarray, cand_scores: np.ndarray, cfg: SeedConfig
) -> np.ndarray:
    """The validated pre-§7 R=4 init: two rotors per harmonically-unrelated
    peak (quadrotor = two tight pairs), offsets fallback with a single peak."""
    second = None
    for i in np.argsort(cand_scores)[::-1]:
        c = float(cand_bases[i])
        if abs(c - base) <= 2.0 * cfg.pair_nudge:
            continue
        if abs(2.0 * c - base) < cfg.harm_guard or abs(c - 2.0 * base) < cfg.harm_guard:
            continue
        second = c
        break
    if second is None:
        init4 = base + np.asarray(cfg.blind_offsets)
    else:
        init4 = np.sort(
            np.array(
                [
                    base - cfg.pair_nudge,
                    base + cfg.pair_nudge,
                    second - cfg.pair_nudge,
                    second + cfg.pair_nudge,
                ]
            )
        )
    return np.clip(init4, cfg.scan_lo, cfg.scan_hi)


def _gated_template(
    audio: np.ndarray,
    fs: float,
    base: float,
    peak_speeds: np.ndarray,
    peak_scores: np.ndarray,
    scores_flat: np.ndarray,
    cfg: SeedConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Confidence-gated shared template for arm T (coordinator fix #3).

    A template estimated from a wrong or weak primary *restricts* the
    matched re-scan and poisons every downstream arm, so T only engages when
    (a) the primary's scan contrast is confident (z-score of its peak score
    against the whole scan-score distribution >= ``t_conf_min``) and (b) the
    top-2 confident candidates' templates agree (cosine >= ``t_cos_min`` —
    the shared-blades premise §7.1 rests on, checked instead of assumed).
    When both top-2 are confident and agree, their templates are averaged
    (stabilises the estimate). Returns ``(template | None, gate diag)``.
    """
    med = float(np.median(scores_flat))
    mad = 1.4826 * float(np.median(np.abs(scores_flat - med)))
    mad = max(mad, 1e-12)

    def zscore(speed: float) -> float:
        j = int(np.argmin(np.abs(peak_speeds - speed)))
        return (float(peak_scores[j]) - med) / mad

    z1 = zscore(base)
    diag: dict[str, Any] = {"z_primary": z1, "z_second": None, "cos": None, "combined": False}
    if z1 < cfg.t_conf_min:
        diag.update(applied=False, reason=f"primary contrast z={z1:.1f} < {cfg.t_conf_min}")
        return None, diag
    tpl1 = estimate_template(audio, fs, base, cfg)
    second = None
    for spd in peak_speeds:  # score-descending already
        if abs(float(spd) - base) >= cfg.dedup_rps:
            second = float(spd)
            break
    if second is not None:
        z2 = zscore(second)
        diag["z_second"] = z2
        if z2 >= cfg.t_conf_min:
            tpl2 = estimate_template(audio, fs, second, cfg)
            n1, n2 = float(np.linalg.norm(tpl1)), float(np.linalg.norm(tpl2))
            cos = float(np.dot(tpl1, tpl2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
            diag["cos"] = cos
            if cos < cfg.t_cos_min:
                diag.update(
                    applied=False, reason=f"template disagreement cos={cos:.2f} < {cfg.t_cos_min}"
                )
                return None, diag
            s = tpl1 + tpl2  # both sum to 1 -> equal-weight average
            tpl1 = s / max(float(s.sum()), 1e-12)
            diag["combined"] = True
    diag.update(applied=True, reason="")
    return tpl1, diag


def blind_seed(
    audio: np.ndarray,
    fs: float,
    n_rotors: int,
    cfg: SeedConfig | None = None,
    arms: Iterable[str] = (),
) -> SeedResult:
    """Blind per-rotor base-speed seeding with the §7 arm set.

    ``arms`` ⊆ {"T", "C", "N", "K"} (empty = the validated pre-§7 baseline
    on the band-capped scan — see ``SeedConfig.scan_f_max``). Returns sorted
    seed bases (constant per rotor — the capture phase adds the time
    dimension), per-candidate scores/flags, the template (arm T), auto knobs
    (arm K) and a diagnostics dict.
    """
    cfg = cfg or SeedConfig()
    arm_set = frozenset(arms)
    unknown = arm_set - {"T", "C", "N", "K"}
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)} (expected subset of T/C/N/K)")

    white, bin_hz, _ = whitened_logmag(audio, fs, cfg)
    wvec = white.mean(axis=1)
    grid = np.arange(cfg.scan_lo, cfg.scan_hi + cfg.scan_step / 2, cfg.scan_step)
    scores_flat = comb_scan(wvec, bin_hz, grid, cfg)
    pk = scan_peaks(grid, scores_flat, cfg)
    order = np.argsort(scores_flat[pk])[::-1]
    peak_speeds, peak_scores = grid[pk][order], scores_flat[pk][order]
    base, octave, promoted = _pick_primary(grid, peak_speeds, peak_scores, cfg)

    template: np.ndarray | None = None
    tgate: dict[str, Any] = {}
    scores_used = scores_flat
    if "T" in arm_set:
        # The gate's top-2 comparison must not pick an integer-ratio alias of
        # the primary as the "second rotor" — filter the flat peaks first.
        tg_b, tg_s = _harmonic_alias_filter(peak_speeds, peak_scores, cfg, primary=base)
        tg_o = np.argsort(tg_s)[::-1]
        template, tgate = _gated_template(audio, fs, base, tg_b[tg_o], tg_s[tg_o], scores_flat, cfg)
        if template is not None:
            scores_used = comb_scan(wvec, bin_hz, grid, cfg, template=template)
            pk_used = scan_peaks(grid, scores_used, cfg)
            cand_b, cand_s = grid[pk_used], scores_used[pk_used]
        else:  # gate tripped -> flat-scan fallback
            cand_b, cand_s = peak_speeds, peak_scores
    else:
        cand_b, cand_s = peak_speeds, peak_scores
    if not np.any(np.abs(cand_b - base) <= cfg.peak_min_sep):
        # the matched scan must not lose the flat scan's primary
        gi = int(np.argmin(np.abs(grid - base)))
        cand_b = np.append(cand_b, base)
        cand_s = np.append(cand_s, scores_used[gi])

    # Completeness presence spectrum: per-frequency time-quantile (wander-
    # robust — see the completeness docstring); the scan keeps the time-mean.
    qvec = np.quantile(white, cfg.tooth_quantile, axis=1)
    # Integer-ratio alias guard against the FULL candidate list, primary
    # pre-kept (a promoted primary must shadow the higher-scoring submultiple
    # it was promoted over); a stronger peak shadows its sub/superharmonic
    # even when C later rejects the stronger peak itself.
    filt_b, _ = _harmonic_alias_filter(cand_b, cand_s, cfg, primary=base)
    filt_set = {float(v) for v in filt_b}

    candidates: list[dict[str, Any]] = []
    for b, s in zip(cand_b, cand_s):
        frac, n_present, n_teeth = completeness(qvec, bin_hz, float(b), cfg, template=template)
        is_primary = abs(float(b) - base) <= cfg.peak_min_sep
        accepted, reason = True, ""
        if float(b) not in filt_set and not is_primary:
            accepted, reason = False, "integer-ratio alias of a stronger peak"
        elif "C" in arm_set and frac < cfg.c_min and not is_primary:
            accepted, reason = False, f"completeness {frac:.2f} < {cfg.c_min}"
        gi = int(np.argmin(np.abs(grid - float(b))))
        candidates.append(
            {
                "base": float(b),
                "score": float(s),
                "score_flat": float(scores_flat[gi]),
                "completeness": frac,
                "teeth_present": n_present,
                "teeth_total": n_teeth,
                "is_primary": is_primary,
                "accepted": accepted,
                "reason": reason,
            }
        )
    surv = [c for c in candidates if c["accepted"]]
    surv_b = np.array([c["base"] for c in surv])
    surv_s = np.array([c["score"] for c in surv])

    if "N" in arm_set or n_rotors != 4:
        bases_out = count_prior(surv_b, surv_s, n_rotors, cfg)
    else:
        bases_out = _legacy_init4(base, surv_b, surv_s, cfg)

    update_gate: float | None = None
    bw_hz: float | None = None
    knob_diag: dict[str, Any] = {}
    if "K" in arm_set:
        primary_gi = int(pk[np.argmin(np.abs(grid[pk] - base))])
        # Mask prediction covers every PLAUSIBLE comb (accepted candidates +
        # seeds) — an unmasked true line masquerades as a huge "noise" ratio,
        # while masking every raw local max blankets the demod bands and
        # leaves no null tracks at all.
        mask_bases = np.unique(np.round(np.concatenate([surv_b, bases_out]), 3))
        update_gate, bw_hz, knob_diag = auto_knobs(
            audio, fs, mask_bases, grid, scores_flat, primary_gi, cfg
        )

    return SeedResult(
        bases=np.asarray(bases_out, dtype=np.float64),
        candidates=candidates,
        template=template,
        update_gate=update_gate,
        bw_hz=bw_hz,
        diagnostics={
            "arms": sorted(arm_set),
            "grid": grid,
            "scores_flat": scores_flat,
            "scores_used": scores_used,
            "primary": base,
            "octave": octave,
            "promoted_up": promoted,
            "template_gate": tgate,
            "peak_speeds": peak_speeds,
            "peak_scores": peak_scores,
            **knob_diag,
        },
    )
