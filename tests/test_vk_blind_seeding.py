"""Synthetic tests for blind seeding v2 (design §7, ``vk_blind_seeding``).

All fast (short mono clips, no ``vk_track`` runs): (a) template recovery,
(b) matched-filter scan finds the weak comb the flat scan misses and
out-scores the (2/3)x alias (the E12/FLY124 failure), (c) completeness
rejects the every-3rd-tooth alias and passes a full weak comb, (d) rotor-count
prior with duplicate seeding of a twin pair, (e) auto-gate between the
hand-tuned 8 and 30.

Run:  pytest tests/test_vk_blind_seeding.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_processing.vk_blind_seeding import (  # noqa: E402
    SeedConfig,
    auto_knobs,
    blind_seed,
    comb_scan,
    completeness,
    count_prior,
    estimate_template,
    scan_peaks,
    whitened_logmag,
)

FS = 16000.0


def synth_combs(
    dur: float,
    bases: list[float],
    scales: list[float],
    profile: np.ndarray,
    noise_scale: float,
    seed: int,
) -> np.ndarray:
    """Mono mixture of harmonic combs sharing one amplitude ``profile`` + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    sig = np.zeros_like(t)
    for base, scale in zip(bases, scales):
        phase = 2 * np.pi * base * t
        for k in range(1, len(profile) + 1):
            a = scale * profile[k - 1]
            if a > 0:
                sig += a * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
    return sig + noise_scale * rng.standard_normal(len(t))


def profile_even(k_max: int = 40) -> np.ndarray:
    """Blade-pass-dominated profile: strong even teeth, near-silent odd."""
    p = np.full(k_max, 0.01)
    p[1::2] = 1.0 / np.sqrt(np.arange(2, k_max + 1, 2))  # k even
    return p


def profile_flat(k_max: int = 40) -> np.ndarray:
    return 1.0 / np.sqrt(np.arange(1, k_max + 1, dtype=np.float64))


def peak_score_near(grid: np.ndarray, scores: np.ndarray, base: float, tol: float = 1.0) -> float:
    """Max score within ``tol`` rev/s of ``base``."""
    sel = np.abs(grid - base) <= tol
    return float(scores[sel].max())


def test_template_matches_generating_profile():
    """(a) Template estimated from a synthetic comb matches its profile."""
    cfg = SeedConfig()
    profile = profile_even()
    y = synth_combs(5.0, [45.0], [1.0], profile, noise_scale=0.05, seed=0)
    tpl = estimate_template(y, FS, 45.0, cfg)

    f = np.arange(1, cfg.k_scan + 1) * 45.0
    valid = (f >= cfg.f_min) & (f <= min(cfg.f_max, 0.45 * FS))
    ref = np.where(valid, profile, 0.0)
    ref = ref / ref.sum()
    # cosine similarity on the validated band — shape is what the matched
    # filter uses, absolute normalization cancels
    cos = float(np.dot(tpl, ref) / (np.linalg.norm(tpl) * np.linalg.norm(ref)))
    assert cos > 0.95, f"template/profile cosine {cos:.3f} below 0.95"
    # strong (even) teeth must dominate the template as they do the profile
    assert tpl[1::2].sum() > 5.0 * tpl[0::2].sum()


def _fly124_scenario(seed: int = 1) -> np.ndarray:
    """E12/FLY124-style mixture: strong comb at 91, weak comb at 81 (shared
    blade-pass profile). The (2/3)x91 = 60.67 alias comb's every-3rd teeth
    land on 91's even harmonics (and, as in the real recording, some of its
    other teeth graze 81's — 4x60.67 = 3x80.9)."""
    profile = profile_even()
    return synth_combs(5.0, [91.0, 81.0], [1.0, 0.12], profile, noise_scale=0.12, seed=seed)


def test_matched_filter_finds_weak_comb_flat_misses():
    """(b) Flat scan scores the 60.7 alias above the real weak 81 comb; the
    matched-filter scan reverses the order (alias out-scored)."""
    cfg = SeedConfig()
    y = _fly124_scenario()
    white, bin_hz, _ = whitened_logmag(y, FS, cfg)
    wvec = white.mean(axis=1)
    grid = np.arange(cfg.scan_lo, cfg.scan_hi + cfg.scan_step / 2, cfg.scan_step)

    flat = comb_scan(wvec, bin_hz, grid, cfg)
    alias, weak = 2.0 / 3.0 * 91.0, 81.0
    assert peak_score_near(grid, flat, alias) > peak_score_near(grid, flat, weak), (
        "flat scan should be fooled by the alias in this scenario"
    )

    tpl = estimate_template(y, FS, 91.0, cfg)
    matched = comb_scan(wvec, bin_hz, grid, cfg, template=tpl)
    s_alias = peak_score_near(grid, matched, alias)
    s_weak = peak_score_near(grid, matched, weak)
    assert s_weak > s_alias, f"matched filter: weak {s_weak:.3f} <= alias {s_alias:.3f}"


def test_completeness_rejects_alias_passes_weak_comb():
    """(c) The every-3rd-tooth alias comb fails the completeness gate; a full
    weak comb passes."""
    cfg = SeedConfig()
    profile = profile_flat()
    # weak comb at 77 (not 81) so the alias stays a PURE every-3rd-tooth comb:
    # with a weak 81, 4x60.67 = 3x80.9 also lights the alias's k=4j teeth,
    # pushing its completeness to exactly 1/3 + 1/4 - 1/12 = 0.5
    y = synth_combs(5.0, [91.0, 77.0], [1.0, 0.25], profile, noise_scale=0.08, seed=2)
    white, bin_hz, _ = whitened_logmag(y, FS, cfg)
    wvec = white.mean(axis=1)

    frac_alias, _, _ = completeness(wvec, bin_hz, 2.0 / 3.0 * 91.0, cfg)
    frac_weak, _, _ = completeness(wvec, bin_hz, 77.0, cfg)
    assert frac_alias < cfg.c_min, f"alias completeness {frac_alias:.2f} not rejected"
    assert frac_weak >= cfg.c_min, f"weak-comb completeness {frac_weak:.2f} rejected"
    assert frac_alias < 0.45, f"alias should have ~1/3 of teeth, got {frac_alias:.2f}"


def test_count_prior_duplicates_twin_pair():
    """(d) 2 true twins 0.65 apart (one merged scan peak) + 2 distinct rotors,
    R=4 -> 4 seeds with the twins duplicated at the pair base."""
    cfg = SeedConfig()
    # direct unit check: near-duplicate peak suppressed, strongest base duplicated
    seeds = count_prior([74.3, 75.0, 81.0, 91.0], [5.0, 4.5, 3.0, 2.8], 4, cfg)
    assert len(seeds) == 4
    near_pair = np.abs(seeds - 74.3) <= 2.0 * cfg.split_nudge
    assert near_pair.sum() == 2, f"expected twin duplicate at 74.3, got {seeds}"
    assert np.any(np.abs(seeds - 81.0) < 0.01) and np.any(np.abs(seeds - 91.0) < 0.01)

    # end-to-end: twins 74.0/74.65 resolve as ONE scan peak; C+N must seed 4.
    # C is included because (2/3)x aliases of the distinct rotors are genuine
    # scan peaks — rejecting them is arm C's §7.2 job, while N's §7.3 contract
    # (dedup + duplicate-at-strongest) is covered above. c_min is raised to
    # 0.65 for this synthetic only: zero-wander constant-base combs give the
    # aliases rational near-miss teeth (Hann main lobe ~8 Hz at n_fft=8192)
    # that lift their completeness to ~0.55-0.65 — still inside the safety
    # band below the real single-rotor calibration (77-100% teeth above
    # floor, §7.2); real wandering audio does not pin lines onto the lobes.
    from dataclasses import replace

    cfg65 = replace(cfg, c_min=0.65)
    profile = profile_flat()
    y = synth_combs(
        6.0, [74.0, 74.65, 89.0, 101.0], [1.0, 1.0, 0.8, 0.8], profile, noise_scale=0.1, seed=3
    )
    res = blind_seed(y, FS, 4, cfg65, arms={"C", "N"})
    assert len(res.bases) == 4
    pair_mean = 0.5 * (74.0 + 74.65)
    assert (np.abs(res.bases - pair_mean) <= 1.0).sum() == 2, (
        f"twins not duplicated at the pair base: {res.bases}"
    )
    assert np.any(np.abs(res.bases - 89.0) <= 1.0), f"89 rotor unseeded: {res.bases}"
    assert np.any(np.abs(res.bases - 101.0) <= 1.0), f"101 rotor unseeded: {res.bases}"


def test_auto_gate_between_hand_tuned_values():
    """(e) Detuned-comb noise-floor calibration lands between the hand-tuned
    gates 8 and 30 for a mid-SNR synthetic case."""
    cfg = SeedConfig()
    profile = profile_flat()
    y = synth_combs(5.0, [45.0], [1.0], profile, noise_scale=0.5, seed=4)  # mid SNR
    white, bin_hz, _ = whitened_logmag(y, FS, cfg)
    wvec = white.mean(axis=1)
    grid = np.arange(cfg.scan_lo, cfg.scan_hi + cfg.scan_step / 2, cfg.scan_step)
    scores = comb_scan(wvec, bin_hz, grid, cfg)
    pk = scan_peaks(grid, scores, cfg)
    primary_idx = int(pk[np.argmax(scores[pk])])
    assert abs(float(grid[primary_idx]) - 45.0) < 1.0

    gate, bw_hz, diag = auto_knobs(y, FS, np.array([45.0]), grid, scores, primary_idx, cfg)
    assert gate is not None and len(diag["gate_null_ratios"]) > 0
    assert 8.0 < gate < 30.0, f"auto gate {gate:.1f} outside the hand-tuned (8, 30) range"
    assert cfg.bw_clip[0] <= bw_hz <= cfg.bw_clip[1]
