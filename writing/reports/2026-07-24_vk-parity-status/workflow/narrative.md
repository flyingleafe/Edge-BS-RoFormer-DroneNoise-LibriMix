# Narrative — VK parity: the tracker, its precision, and the road to neural parity
kind: report
audience: supervisor + research group + any technically literate reader
(undergrad-followable); they know the project premise (RPS-informed speech
enhancement) but must NOT need any prior VK knowledge — the tutorial sections
are the point of the report.
through-line: We built a coupled Vold–Kalman order tracker as the precision
reference for rotor-speed (RPS) annotation from audio alone. This report (1)
teaches the algorithm at REIMPLEMENTATION fidelity — a reader should be able
to rebuild the whole blind-annotation pipeline from these pages, (2) reports
its current precision against ground-truth telemetry on both drones with
trajectory-overlay evidence, and (3) documents the neural-parity program:
what test-time smoothing and longer training context did (both refuted as
parity levers), and the front-end arms now in build.

**USER SPEC (binding, 2026-07-24): full undergrad-level tutorial of the VK
tracking algorithm aiming at reimplementation-grade understanding; current
precision metrics + illustrations of results against ground truth; current
approaches at neural-parity modifications — what is already done, what is
next. Style laws carried from the deck rework: pseudocode blocks explicitly
welcome; ZERO unbacked claims (every number visible or explicitly marked
pending); no wasted space; figures/formulas anchor the text.**

## Sections (ordered)

1. **Problem & objective** — message: per-rotor RPS trajectories from audio
   alone, scored by how well they explain the recording (generative
   residual); why: telemetry is absent at inference and imperfect as labels —
   evidence: signal model formula y(t) = Σ_a Re[env_a(t)·exp(jφ_a(t))] +
   noise, residual functional; the DREGON/FLY124 spectrogram-with-comb figure
   (remake from results npz or crib recipe from
   writing/slides/2026-07-18_dregon-analysis-and-generator-design/prepare.py)
   — sources: docs/vk-order-tracking-design.md §1.

2. **Tutorial I — envelopes are a linear solve (VP), demodulate + decimate**
   — message: given frequencies, amplitudes/phases come from least squares
   (variable projection); envelopes are narrowband, so demodulating by the
   carrier and decimating to an envelope-rate grid shrinks 16 M unknowns to
   ~200 k — evidence: VP/lstsq formula; demod formula z = LP[y·conj(c)],
   fs_env grid diagram; concrete shapes for a 20 s / 16 kHz recording —
   sources: design doc §1–2.1, src/data_processing/vk_tracking.py.

3. **Tutorial II — the coupled banded solve and the frequency update** —
   message: nearby harmonics of different rotors compete for the same energy
   (explaining-away) ⇒ solve envelopes JOINTLY: block-banded Hermitian PD
   normal equations, time-major interleave index = t·g + a, bandwidth p·g,
   banded Cholesky; then update frequencies by the envelope phase slope
   δ̂ = angle(x_{t+1}·conj(x_t))·fs_env/(2πk), Fisher-weighted across
   harmonics, smoothness prior, k_max annealing (capture wide/coarse →
   refine narrow/precise) — evidence: coupling predicate, matrix-structure
   figure (banded blocks — remake), both update formulas, the
   capture-vs-refine config table (bw_hz, k_max schedule, update_gate) —
   sources: design doc §2.2–2.3 + §8 items 1–2 (banded solver, pruning),
   vk_tracking.py.

4. **Tutorial III — blind annotation end-to-end (with FULL pseudocode)** —
   message: no telemetry at all: whiten spectrum → band-capped comb
   matched-filter scan (scan_f_max kills low-f0 aliases) → dedup/alias
   rejection → seed → capture → residual re-scan (arm R) recovers combs
   shadowed by stronger neighbours → spatial-DP ladder resolves twin rotors
   → per-track stage guard reverts refinement stages that destroy a track
   (detected by comb-occupancy drop — raw confidence RISES during the
   failure) — evidence: ONE full-page pseudocode block of the whole
   pipeline (crib the deck's pseudocode slide, expand to report depth);
   whitened-scan score illustration; the alias arithmetic example
   (60.7 ≈ (2/3)·91: teeth k=3,6,9 alias the 91-rotor's even harmonics) —
   sources: design doc §7 (incl. §7.5), src/data_processing/vk_blind_seeding.py,
   scripts/vk_blind_annotation.py, scripts/vk_blind_sweep.py.

5. **Precision results vs ground truth** — message: telemetry-init refine
   sits at the label jitter floor (DREGON refine 0.604 vs command 0.609
   rev/s); blind: DREGON 0.680 / 0.701 / 0.744 pooled (twins resolved 2/3
   recordings), FLY124-cruise 1.027 pooled, per-rotor 0.67/1.19/1.22/1.03,
   capture 4/4 — and the FLY124 evolution story 4.0 → 3.24 → 1.027 (what
   each fix bought: band cap, arm R, stage guard) — evidence: results table;
   TWO trajectory-overlay figures (blind estimate over spectrogram with
   dashed GT: one DREGON, one FLY124 — remake at report quality from
   results/vk_tracking/blind_annotation/*__blindvit2dsp.npz and
   results/vk_blind_sweep_r6/omnirun-outputs/python-4a015c/results/vk_blind_sweep/FLY124-cruise__vit2dsp__R.npz;
   figure recipe in the 07-18 deck's prepare.py — NOTE the `edge` array in
   sweep npz is a KEEP mask, not an exclude mask); evolution mini-table —
   sources: results/vk_eval/vk_valid_comparison.csv,
   results/vk_blind_sweep_r{4,5,6}/sweep_report.csv, design doc §7.5.

6. **Fast inference** — message: CPU path optimized ~10×: banded Hermitian
   Cholesky (2.9×) + far-pair pruning (1.7×) + fixes ⇒ refine rtf
   0.037 → 0.36–0.40, blind 0.34 → 0.95, RSS halved, results
   regression-exact (≤1e-3 gate); FIR polyphase alternative measured and
   REJECTED (slower than pocketfft brickwall + 2.5e-3 trajectory
   perturbation); GPU torch adapter (block-tridiagonal formulation of the
   banded solve) IN BUILD — mark pending — evidence: per-phase before/after
   bar or rtf table from results/vk_bench/profile_report.*; one line on the
   solver structure — sources: design doc §8, commit 1ca7581.

7. **Neural parity program: done, refuted, and in build** — message: the
   criterion is an audio-only model trained on real data + augmentations
   matching blind VK on the SAME clips (protocol:
   results/vk_eval/vk_valid_comparison.csv, per-clip PIT-MAE). Phase A
   (test-time smoothing of the E12 transformer): best DREGON cruise 2.62,
   FLY124 1.55 — helps but saturates; error is systematic within-window.
   Phase B (native 4 s / 8 s training context): best 2.87 / 1.90 — WORSE
   than phase A; context length REFUTED as the lever; `last` checkpoints
   degrade sharply (7.07). Blind VK (0.68–0.74 / 1.03) currently beats every
   neural variant on both pools. In build NOW (mark pending, zero numbers):
   G2a harmonic-stacked HCQT front-end, G2b instantaneous-frequency phase
   channels — hypothesis: the magnitude-STFT front-end lacks harmonic
   aggregation and sub-bin precision. Parked levers: VK-distilled labels,
   VK-annotated unlabeled data — evidence: the phase-A/B table (numbers from
   omnirun-outputs/python-267f52/results/rps_predictor_vk_eval/report.json +
   results/rps_predictor_vk_eval/), a speed-accuracy quadrant or bar figure
   (VK vs neural arms); G2 hypothesis stated as hypothesis — sources:
   docs/experiments/g1-vk-parity.md (incl. the Phase B result section),
   scripts/rps_predictor_vk_eval.py docstring.

8. **Verdict & roadmap** — message: blind VK is the precision reference
   (criterion 2.2 closed — every non-learning lever measured, adopted or
   refuted); neural parity is OPEN with a quantified gap (2.87 vs 0.68–0.74
   DREGON; 1.90 vs 1.03 FLY124) and a concrete, evidence-motivated next
   step already in build — sources: all above.

## Cut (considered, excluded)
- SE-baseline (f1) results — separate 2026-07-22 report.
- Generator/GP 4-way comparison — belongs to the generator thread, not VK
  parity; one cross-reference line only.
- Kalman-tracker post-mortem — superseded design; one line in §1 at most.

## Open questions for the user
- none — the user spec above is binding; deviations only where evidence
  demands, flagged in the creator log.

## NON-NEGOTIABLE REVIEW CRITERIA (inherited from the deck rework)
1. Tutorial sections at REIMPLEMENTATION fidelity — a reader rebuilds the
   tracker from this report alone; one honest full pseudocode block beats
   prose.
2. ZERO unbacked statements — every claim shows its number/figure on the
   page or is explicitly marked pending (G2 arms, GPU adapter).
3. No wasted space; figures and formulas anchor, text explains.
4. Typst gotcha: term-list colons in Typst markup must be escaped/avoided
   (known trap); build with `make check` and visually inspect every page.
