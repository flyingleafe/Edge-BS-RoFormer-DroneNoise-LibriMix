# Narrative — The rotor comb, three ways: generate it, read it, borrow it
kind: slides
audience: You / advisor — already saw this deck's original generator arc (geometry fix, per-rotor sub-embeddings, wind channel). This UPDATE keeps that spine intact and folds in the two other threads from the week: 2026-07-15 Vold–Kalman order tracking, 2026-07-16 JASA-GP replication.
through-line: The harmonic comb is the one object all week's work shares. The generator *writes* the comb — its structure forced by DREGON's data (the existing arc). Vold–Kalman order tracking *reads the comb back* with no telemetry, and its one failure (twin rotors) is exactly what the generator's per-rotor prior is built to fix — the two threads close a loop. JASA-GP is a faithful physics+GP *alternative* comb model from the literature, now reproducible in-repo as a listening baseline. Three activities, one comb, feeding each other.

## THIS IS AN UPDATE, NOT A NEW DECK
- Target: `writing/slides/2026-07-18_dregon-analysis-and-generator-design/slides.typ` (already exists, 8 content sections, renders 12 pages).
- KEEP the existing 8 generator sections essentially as they are (light touch only: you MAY add a one-line opening "map" and retitle the deck — see below). Do NOT rewrite the geometry/per-rotor/wind content; it is already reviewed and good.
- ADD the new material: ~3 slides on Vold–Kalman blind order tracking, ~2 on JASA-GP, and revise the final Takeaways slide to synthesise all three threads.
- Retitle (default, apply it): title → "The rotor comb, three ways"; subtitle → something like "Generate it · read it back · borrow the literature's baseline". Keep author/date lines. Add a single opening bullet or sentence on slide 1 framing the comb as the shared object across the three threads.

## Figure pipeline
- Existing figures are copied by `prepare.py` from the companion report's `assets/`. For the NEW figures, EXTEND `prepare.py` to also copy from `results/`:
  - VK / blind re-annotation overlays: `results/vk_tracking/blind_annotation/blind_vs_gt_nosource.png` (DREGON, the clean success), and `results/vk_tracking/blind_annotation/blind_hard.png` and/or `results/vk_tracking/blind_annotation/fix_blindfixA_free-flight_nosource_room1.png` (FLY124 / harder case). Inspect these PNGs first and pick the clearest one or two.
  - JASA-GP: `results/jasa_gp/eval_V7.png` (held-out flyover fit) and `results/jasa_gp/loudness.png` (loudness map). `results/jasa_gp/eval_metrics.json` has per-point corr/loudness numbers.
- If a copied figure is unreadable at slide size (dense multi-panel), prefer regenerating a slimmed single-panel version with a small script rather than shrinking it. Use `improve-plot-visibility` judgement.
- Run `make figures` (which runs `prepare.py`) then `make check` to render `check/page-*.png`; VIEW every changed/new page before declaring done.

## Sections (ordered)

Spine — existing generator arc, KEEP (deck sections 1–8):
1. A physics-structured generator, answerable to data — three testable assumptions — existing propagation eqn.
2. Geometry: a small error is diagnostic — freq-proportional phase error — `assets/geo_propagation_phase.png`.
3. Geometry: the 180° frame mismatch — anti-correlation, peak +0.93 at 183° — `assets/geo_frame_alignment.png`.
4. Geometry: the fine fix + honest limit — bundle adjustment (DREGON), non-identifiability (Michael's) — `assets/geo_summary.png`.
5. Are the four rotors one source? — 6.8 dB RMS timbre difference — `assets/fig_per_rotor.png`.
6. …so: per-rotor sub-embeddings — learnable zero-init per-rotor delta.
7. A wind channel — incoherent flow noise needs its own additive channel — `assets/fig_wind_schema.png`.
8. From analysis to variants — 3 variants scored on free-flight MR-STFT; geometry vindicated (OLD 4.51 → v1 5.22, v2 4.82, v3 3.44).

NEW thread A — reading the comb back (Vold–Kalman order tracking):
9. Reading the comb backwards — message: can we recover per-rotor RPS from audio ALONE, no telemetry? Coupled Vold–Kalman (VK) order tracking, built from published math (Vold & Leuridan 1993; Tuma 2005). PyVKF rejected: GPL-3 + full T·M audio-rate system ≈16M unknowns, frequencies are inputs (no tracking loop). Our design: demodulate+decimate to an envelope grid (unknowns T·M → ~200k), sparse coupling groups; the off-diagonal coupling term makes overlapping tracks COMPETE (explaining-away) — the fix for the twin-capture bias that killed the earlier heuristic comb refinement. — evidence: functional / coupling schematic (a clean 1-panel diagram; can be a small matplotlib or typst-drawn figure — the coupling term J[a] = Σ|y − Σ Re[a_m c_m]|² + Σρ²‖Δ²a_m‖²). Source: `docs/vk-order-tracking-design.md`, `scripts/vk_blind_annotation.py`.
10. Blind re-annotation: where it works, where it breaks — message: DREGON free-flight (nosource, 25 s): all 4 rotors recovered, pooled err 0.68 rev/s (spatial-DP arm), combs ride the ridges. FLY124 free-flight CRUISE (~74/74/81/91 rev/s — NOT idle): 3 of 4 rotors to ~1 rev/s (91→0.8, 74→1.0, 81→1.3); the 4th fails because a TWIN pair (~74/74) seeds only ONE track from a buried comb, and the second twin's init lands on a spurious alias peak (~60 Hz). So the real story is TWIN SEEDING, not total failure — and its fix is a shared per-rotor comb-shape prior, i.e. the generator's own per-rotor amplitude template (PositionalHarmonicNoiseGen). The two threads close a loop. — evidence: `results/vk_tracking/blind_annotation/blind_vs_gt_nosource.png` (DREGON success) + FLY124 harder overlay. Source: memory blind-reannotation-dregon-vs-fly124.
11. The honest limit: fine jitter is below the floor — message: 0.68 rev/s is the achievable FLOOR, not a tuning gap. The objective's minimum is NOT at true RPS: the smooth-carrier + flexible-envelope fit out-explains the real 0.74 rev/s jitter (data residual LOWER at the smooth blind answer 0.99196 than at true jittery telemetry 0.99376). Comb is <1% of full-band energy; strip wind (high-pass) and it is ~19% of mid-band, residual 0.990 → 0.874 (hp>0.5k) → 0.810 (bp 1–3k), but the smooth path still wins in every band. Jitter is invisible to the comb model, not hidden by wind. — evidence: the residual-vs-band numbers as a tiny table or inline. (If the deck runs long, MERGE this into §10 as the closing caveat.)

NEW thread B — a physics baseline we can now reproduce (JASA-GP):
12. Borrowing the literature's best: JASA-GP — message: faithful replication of Lee et al. (JASA 159(4):3418, 2026) GP rotor-noise model on the `jasa-flyovers` dataset (10 CONA quadrotor flyovers, V=1..10 m/s, 256 ground mics). Pipeline: per-(mic,V) de-Doppler → estimate f0 → lstsq Fourier coeffs at k·f0 (H=24) → Matérn-5/2 GP over (x,y,V) predicting the coeff vector; broadband synthesised separately. A physics+GP flyover comb model — a complement/comparator to our own emitter+propagation generator. — evidence: `results/jasa_gp/eval_V7.png` (held-out flyover, V=7 not in training).
13. Getting it faithful + a listening deliverable — message: three fixes took amplitude prediction from ~4× under to faithful, matching the paper's Fig 5/7 (held-out AC corr 0.41→0.70, loudness err 9.3→3.0 dB): (1) a np.roll SIGN bug in phase alignment → sign-mixed phasors → GP shrinkage, replaced with exact coefficient-space alignment; (2) per-signal f0 estimation (a pinned 33.55 Hz comb drifts off the bin by k≈12); (3) measure loudness on the AC part (CONA "tonal" has a large inaudible DC). Deliverable = interactive 3D listening notebook (`notebooks/jasa_gp_interactive.ipynb`: plotly 3D + rotor-speed FM + audio/spectrogram). — evidence: `results/jasa_gp/loudness.png` + optional notebook screenshot.

Close:
14. Takeaways (REVISE the existing takeaways slide) — message: one comb, three activities that feed each other. (a) Generator structure is data-forced and validated (geometry fix permanent; per-rotor + wind well-motivated, payoff regime-dependent). (b) VK reads per-rotor RPS to 0.68 rev/s blind — at the achievable floor — and its twin-seeding failure directly motivates the generator's per-rotor comb prior. (c) JASA-GP gives a faithful physics baseline + a listening tool. Next: shared-comb prior to close the twin gap; free-flight airspeed + coherence-aware loss for the wind channel.

## Cut (considered, excluded)
- The KILLED Kalman harmonic tracker (07-10, pre-window, superseded by VK) — one line max only if it aids the VK motivation; do not give it a slide.
- SPCup annotation outputs — a deliverable detail; fold into §10 as a byproduct if space allows.
- Full VK normal-equations math — one schematic (§9) suffices; equations belong in a report.

## Notes
- Narrative was NOT user-reviewed (10-min checkpoint elapsed with no reply). Defaults applied: retitle to "The rotor comb, three ways"; 3 VK slides + 2 JASA-GP slides.
- Every new numeric claim must trace to a source above; mark anything you cannot verify against a file/results dir with `[TODO verify]`.
