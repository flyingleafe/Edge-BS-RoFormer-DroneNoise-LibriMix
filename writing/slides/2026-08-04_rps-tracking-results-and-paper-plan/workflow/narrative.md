# Narrative — RPS tracking: results, the ceiling, and the paper plan

kind: slides
audience: PhD supervisor. Last deck seen: 2026-07-27 (DCUNet generalization).
Has read weekly reports but is lost in them — this deck must stand alone.
Knows the project goal (SE under rotating-machinery noise) and the December
transfer gate.

through-line: We answered "how well can rotor speeds be read from audio alone,
and what limits it?" The limits are now measured facts, not hunches; they
dictate the noise model, the estimator, and the architecture. A combined paper
(classical VK chain + neural trackers on one frozen protocol) is in draft and
submits in ~4 weeks. In parallel, the SE side got a trustworthy foundation and
the RPS-conditioned-SE question starts now.

## GLOBAL BUILD REQUIREMENTS (binding for the creator)

0. **Read `workflow/style-guide.md` (same directory as this file) in full
   before every editing pass.** It is distilled from this user's past deck
   feedback and overrides your habits. Its top recurring bounce-causes: >40
   words on a slide; orphan/continuation pages and dead whitespace; content
   hidden in speaker notes that belongs on the slide (key formulas ON the
   slide); tiny figures; unbacked claims; internal experiment designators on
   slides; fake narrative drama about retracted results; em-dash+bold slide
   splits and `~`-eats-approx Typst traps.

1. **As few words as possible on slides.** Every slide: a title, a figure or
   table or 1–3 short lines, nothing else. The full explanatory narrative for
   each slide goes into **speaker notes** (Touying `config-common` speaker-note
   mechanism or pdfpc-style notes — pick the template's supported way and use
   it on EVERY slide). The speaker-note text for each slide is given below —
   adapt freely for flow but keep the content.
2. Numbers policy: use ONLY post-recalibration numbers from
   `docs/experiments/beat-vk.md`. The FLY124 neural rows are being re-scored
   right now (job in flight); before finalizing, check `beat-vk.md` for a
   dated "post-recalibration neural re-score" subsection. If it has not
   landed, mark those cells `*` with a footnote "re-score in flight" — do NOT
   quote the stale pre-recalibration values as current.
3. Figures: generate from real data via a `prepare_figs.py` INSIDE the deck
   dir (pattern: existing decks' sibling prepare scripts). Plot RPS
   comparisons with PIT alignment via `align_rps_to_gt`
   (`src/tasks/rps_prediction.py`) — unaligned plots show fake rotor swaps.
   Image assets stay gitignored; only .typ/.csv/.py are tracked.
4. You have read access everywhere and write access to a few extra repo dirs
   (hook-permitted for other agents' sake) — do NOT write outside the deck
   directory anyway. Log blockers in workflow/creator-log.md.
5. Renders must be visually checked (the check/ pages), especially the
   algorithm-step slides and any two-column layouts.

## Sections (ordered; ≈18–20 slides)

1. **Title + agenda** — message: results / what limits them / the paper /
   next 4 weeks. Speaker notes: one-paragraph frame — two months went into
   "how well can RPS be read from audio alone"; this deck shows the answer,
   why it is a hard limit, and the path to submission.

2. **Headline results table** — message: one table, MAE (rev/s, fixed
   15-window raw protocol, both drones, post-recalibration). Rows: blind VK
   chain (blind_fullrange + gated refine), VK telemetry-init (oracle),
   CKLA phase-only 4 s, KLA, transformer, uni_gru128, (scv2 optional),
   hybrid (neural init + pi_kalman). Columns: DREGON cruise / FLY124 cruise.
   Evidence: `docs/experiments/beat-vk.md` current tables (+ re-score
   subsection when it lands). Speaker notes: the protocol (15 frozen 16 s
   windows, PIT-MAE vs raw telemetry); the telemetry recalibration story in
   two sentences (drifting clock + 0.7% scale, measured, validated with
   held-out and audio-only controls, everything re-scored); what each row is;
   the oracle row is ~0.8 not 0 — foreshadows the ceiling.

3.–7. **The VK tracking algorithm, step by step** (~5 slides, "slider"
   style: same layout each slide, one stage highlighted per slide, like a
   stepper). Each slide: one pipeline diagram with the current step
   highlighted + one small real-data panel for that step + at most ONE
   formula + one line of text. Steps:
   (a) Input & global stage: whitened multi-channel spectrogram; comb-matched
       frame Viterbi over the full rate range (emission = comb-matched score,
       smooth transition penalty). Evidence: generate the spectrogram/ridge
       panel from a DREGON window (prep cache `results/beatvk_vk_arms/`,
       `src/data_processing/vk_blind_*.py` utilities).
   (b) Ramp handling: full-range coarse Viterbi + octave disambiguation via
       blade-pass line-ratio check + energy-timed bridge across spool-up.
       Speaker notes: warmup spectra carry only even/blade-pass lines, the
       seeder used to octave-promote; ramps went 15–36 → 2.9–4.0 MAE.
   (c) Per-rotor decoupling: corridor split around the global track + gated
       residual re-seed (span + dedup guards). Speaker notes: twin pairs
       0.5–0.65 rev/s apart are the hard case; the seeding guards fixed a
       coin-flip failure worth 1.57 rev/s on FLY124.
   (d) Coupled VK envelope solve: the Vold-Kalman data + smoothness
       equations, all rotors jointly (one formula: the two-equation VK
       system). Speaker notes: why joint — shared bins between overlapping
       combs.
   (e) Phase-increment refinement (pi_kalman): measurement Δψ_k =
       arg(z_k(t)·z̄_k(t−1)), variance from envelope SNR + per-harmonic
       random-walk q_k; pair-joint mode; RTS smoother. ONE pass, not a loop
       (speaker notes: iterating injects noise — measured).
   Evidence for the chain: paper draft Sec. IV (`writing/papers/
   2026-07_coupled-vk-blind-rps/main.tex`), `docs/experiments/beat-vk.md`,
   `src/data_processing/vk_tracking.py`, `phase_increment_tracker.py`.

8. **Output comparison: DREGON slice** — message: best VK chain vs CKLA-4s
   on one DREGON cruise window; RPS-vs-time, 4 rotors, dotted GT,
   PIT-aligned; spectrogram column if layout allows (two_columns pattern).
   Speaker notes: VK tracks fluctuations tightly (~1.0/window steady), CKLA
   shows the anchor-collapse residual (one rotor off by pair spacing);
   DREGON is where the classical chain still wins.
   Evidence: NPZs in `results/beatvk_vk_arms/` runs + neural NPZs
   (re-score job artifacts under `omnirun-outputs/`, or
   `omnirun-outputs/bash-cbfe15` __nseed NPZs as fallback).

9. **Output comparison: FLY124 slice** — same layout, one FLY124 cruise
   window. Speaker notes: here the neural model is the better tracker
   (matches the telemetry-init oracle on steady windows) and the hybrid
   (neural init + pi_kalman) is best overall — neither family dominates,
   which is the paper's empirical spine.

10. **Noise model v0** — message: the textbook assumption — harmonic comb,
    random initial phases, deterministic evolution from the shaft. One
    schematic + the phase equation φ_k(t) = k·φ_shaft(t) + φ_k(0). Speaker
    notes: this is what our generator and all standard methods assume.

11. **Model v1: uncertain shaft** — message: admit imprecise shaft IF; all
    harmonics stay LOCKED together (error at k = k × shaft error). One-line
    testable consequence. Speaker notes: one hidden scalar process; if you
    know the phase error at one harmonic you know it everywhere — this
    coherence is measurable.

12. **The lock measurement + realism ladder (MONEY SLIDE)** — message: bar
    chart, lock at k=1–2 across: synthetic / single real motor (0.72–0.88) /
    four motors static bench (0.02–0.09) / hover / free flight (same floor).
    One line: "running four rotors at once destroys phase coherence."
    Evidence: `results/vk_phase_validation_decomp`, ladder docs in
    `docs/experiments/beat-vk.md`. Speaker notes: define lock (demodulate by
    telemetry shaft phase; resultant of phase increments); the collapse
    arrives with multi-rotor operation itself — staggered setpoints included
    — not with flight and not per-rotor aero noise.

13. **"Use more microphones?" — no** — message: bar chart single mic vs
    delay-and-sum vs ORACLE-steered combiner (~0.10) vs needed (~0.7+).
    One line: "the incoherence arrives identically at every microphone."
    Speaker notes: the oracle combiner is given the best per-mic alignment
    after the fact — an upper bound no physical beamformer can beat; even it
    fails → the loss is common-mode, at the source; arrays cannot restore
    coherence that was never emitted. Evidence: beamform lock probe results
    (`scripts/beamform_lock_probe.py`, results referenced in beat-vk.md).

14. **The displaced comb** — message: per-harmonic offset figure (or compact
    table): k=2–13 sit 0.3–0.5 rev/s BELOW mechanical; k≥16 exactly on-grid
    but weak; hover 3–4× weaker. One line: "at low harmonics, acoustic truth
    ≠ mechanical truth." Speaker notes: the four verification probes (per-k
    structure; clock-skew refuted — audio/telemetry ratio 1.0000±0.0005 on
    steady clips; estimator exonerated on synthetic; hover contrast); why
    proportional mechanisms (Doppler/calibration) cannot explain sparing
    high k; leading hypothesis = forward-flight loading modulation skewing
    strong low lines; mechanism open. If no figure-able artifact exists,
    schematic + numbers and flag [TODO verify] for the prep script.

15. **Model v2: what the data forces** — message: φ_k = k·φ_shaft + b_k,
    b_k per-harmonic random walk, measured budgets (τ_k ≈ 0.4–1.7 s at
    k=8–40). One line: "we built the matching estimator (pi_kalman) — and
    with it we sit at the measured limit." Speaker notes: the full chain
    moves the blind baseline <2% (1.825→1.819 DREGON); the estimator that
    models the phase noise is the best this physics admits; this measured
    model is also the design brief for the neural architecture (CKLA's
    phase-first structure is its mirror).

16. **Significance** — message: three bullets, ≤6 words each:
    (1) blind per-rotor acoustic tracking: no prior art (twin combs);
    (2) limits are measured facts → dictate models;
    (3) required RPS precision for SE now known. Speaker notes: the sharp
    novelty claim that survived this week's adversarial literature crawl
    (scratchpad baseline-crawl.md): no prior method recovers four per-rotor
    rate trajectories blind from audio when combs overlap within a filter
    bandwidth; nearest prior art (2012 supervised helicopter NN; aggregate
    BPF estimators; ARC 2026 single-shaft tacholess) and why each falls
    short; significance 3 sets up the SE question.

17. **The paper (WIP)** — message: combined paper skeleton as a compact
    diagram/list: methods = VK chain + neural trackers (CKLA family) +
    hybrid; protocol = 3 synthetic tiers (S1 single rotor / S2 twins 0 dB /
    S3 measured phase noise −5 dB) + frozen 15-window real protocol;
    baselines = IAVKF, VKF+SWT, tacholess order tracking (iterated warp),
    multi-pitch salience (multif0, Basic Pitch — already tested), ablations.
    Status line: draft WIP; baselines + figures = the remaining work.
    Speaker notes: the plan IS the baseline list + validation of each
    proposed method against all baselines on the same protocol; the current
    8-page draft covers the VK side; the neural section extends it into the
    combined paper; venue input wanted.

18. **Next 4 weeks** — message: 4-row calendar: wk1 baselines + figures;
    wk2 neural section + full results; wk3 polish + internal review; wk4
    buffer + SUBMIT. Parallel lane: RPS-conditioned SE starts on the fixed
    grid (oracle RPS into MP-SENet / Edge-BS-RoFormer via comb subtraction,
    harmonic-aligned bands, FiLM; first read end of August). Speaker notes:
    the SE foundation was fixed this week (mixing bug that zeroed ~6% of
    samples found + valid sets rebuilt + verified clean; TF-GridNet
    "failure" diagnosed as compute starvation; all four baselines retraining
    with clean curves, NaN guard added); the conditioning question is the
    core of the next paper.

19. **Asks** — message: three bullets: venue for paper 1; reaction to the
    ceiling argument (it becomes the thesis backbone); agree
    RPS-conditioned SE = next-paper question. Speaker notes: one sentence
    each.

## Cut (considered, excluded)

- Full thesis program / Aug–Dec tracks (user directive: not in this deck;
  lives in docs/thesis-program.md).
- Two-month four-line inventory as its own section (folded into speaker
  notes where needed).
- Wind-channel/likelihood post-mortem, generator arc, WP16–21 dead ends —
  omitted entirely (report material).
- CKLA kernel/infra engineering, data-layer refactor.
- F1/F2 details beyond the one speaker-note paragraph in section 18.

## Open questions for the user

- None blocking; venue question is deliberately ON a slide (asks).
