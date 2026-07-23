# Narrative — RPS prediction and drone noise generation progress (REWORK)
kind: slides
audience: supervisor + research group; know the project premise (RPS-informed SE),
saw the 2026-07-13 deck (sim2real RPS story) and 2026-07-06 (GP/FWH). General
technical audience — every method slide must be followable by an undergrad.
through-line: We audited the physics-structured generator against real data
(geometry fixed for good; per-rotor + wind well-motivated but not yet wins),
built a principled blind RPS tracker (coupled Vold–Kalman) that we then made
10× faster and systematically de-biased, reproduced the JASA-GP literature
baseline and trained it on setup-matched CONA auralizations — and the 4-way
comparison (real/CONA/deep/GP on the same RPS) crowns the deep generator while
exposing the ONE dominant realism gap all synthetics share: spatial coherence
(wind). Meanwhile two new fronts opened: modern blind-SE baselines (f1) and a
VK-parity RPS predictor plan.

**BINDING SOURCE: the speaker notes in slides.typ (backup:
workflow/slides-notes-source.typ). Each note is an INSTRUCTION for what its
slide must show. Style law: figures + formulas dominate; text only as anchors.
The notes remain in the deck as the speaker script (may be lightly edited for
accuracy, never deleted). This REWORK supersedes the previous narrative in this
file — rearrangement and figure remakes are expected, the old "keep as is"
constraint no longer applies.**

## Sections (ordered)

1. **This week: the rotor comb, three ways (+ two initiated)** — message: three
   work threads + two initiated programs — evidence: text overview only (the
   one text-allowed slide) — sources: existing slide 1, extend with initiated
   work bullets.
2. **Generator: the model and the mid-harmonic symptom** — message: physics
   structure makes assumptions testable; weak mid-harmonics started the audit —
   evidence: propagation formula + (existing) geo_propagation_phase.png —
   sources: 2026-07-18 report §1–2.
3. **Geometry bug: DREGON 183°** — message: TDOA anti-correlation → frame
   rotation — evidence: geo_frame_alignment.png + NEW 3-D mic+rotor positions
   figure (before/after frame fix) per note — sources: 07-15 report assets +
   dregon.get_geometry.
4. **Geometry bug: Michael's plane swap** — message: vertical↔horizontal ring
   coding bug, found visually; TDOA method not identifiable here (no
   single-rotor recs) — evidence: NEW mic+rotor 3-D positions figure (wrong vs
   fixed plane) — sources: 07-15 report §5, michaels.get_geometry.
5. **Fine calibration (DREGON)** — message: coherence-weighted phase bundle
   adjustment, ≤2.2 cm moves, synthetic control 0.36 cm — evidence: objective
   formula + geo_summary.png or before/after positions figure — sources: 07-15
   report §4.
6. **Rotors are individuals** — message: 6.8 dB RMS timbre spread — evidence:
   fig_per_rotor.png (existing) — sources: 07-18 report §3.
7. **Per-rotor sub-embeddings** — message: z_r = z_drone + δz_r — evidence:
   NEW generator schema figure with the sub-embedding block added+highlighted,
   + the δz regularization formula per note — sources: gen_v2 config,
   models/generative.
8. **Wind channel** — message: physics places flow, learned head transduces —
   evidence: fig_wind_schema.png (existing) + gate validation number (Spearman
   0.92) — sources: 07-17/07-18 reports.
9. **Variants: what the data said** — message: v1 (geometry) wins in flight;
   idle-heavy set inverts ranking — evidence: NEW spectrogram grid per note:
   rows DREGON/Michael's, columns real | OLD | v1 | v2 | v3, mrstft under each
   generated panel (assets from 07-17 report / its prepare.py can be re-run;
   free-flight clips) — sources: 07-17 report.
10. **Generator discussion** — message: WHY v2/v3 underperformed: v2 helps only
    idle asymmetries (regime-dependent); v3 dormant at hover AND invisible to a
    single-channel magnitude loss; today's coherence measurement (real MSC 0.05
    vs deep 0.69 low-band) proves wind is the dominant residual gap — evidence:
    small MSC bar/table from the 4-way notebook — sources: 07-17 discussion +
    notebooks/noise_four_way_comparison.ipynb.
11. **VK thread opener: score a trajectory by how well it explains the audio** —
    message: generative residual as the objective — evidence: signal model
    formula y = Σ Re[a·exp(jφ)], residual functional — sources:
    docs/vk-order-tracking-design.md §1.
12. **Step 1: envelopes by least squares (VP)** — message: given frequencies,
    amplitudes/phases are a linear solve — evidence: VP/lstsq formula, tiny
    schematic — sources: design doc predecessor + §1.
13. **Step 2: demodulate + decimate** — message: envelopes are narrowband →
    16M unknowns become 200k — evidence: demod formula z=LP[y·conj(c)] +
    fs_env grid cartoon — sources: design §2.1.
14. **Step 3: coupled solve** — message: nearby tracks compete for shared
    energy (explaining-away); block-banded Hermitian system — evidence:
    coupling predicate + matrix-structure figure (banded blocks) — sources:
    design §2.2 (the commented-out vk_coupling_schematic.png may be revived or
    remade).
15. **Step 4: frequency update by phase slope + annealing** — message: phase
    slope δ̂ = angle(x_{t+1}·conj(x_t))·fs_env/(2πk), Fisher-weighted fused,
    k_max annealed — evidence: the two formulas + capture-basin cartoon —
    sources: design §2.3.
16. **VK results** — message: telemetry-init refinement at the jitter floor;
    blind re-annotation numbers — evidence: table (DREGON refine err 0.604 vs
    command 0.609; blind DREGON 0.68 [spatial-DP arm]; FLY124 cruise pooled 4.0
    with 3/4 rotors ~1) + NEW trajectory-overlay figure (refined vs telemetry
    over spectrogram, one DREGON + one FLY124 panel; npz in results/vk_eval +
    results/vk_tracking/blind_annotation) — sources: vk_valid_comparison.csv,
    blind_annotation fix_summary.
17. **Today: 10× faster VK** — message: profile → banded Hermitian Cholesky
    (2.9×) + pair pruning (1.7×) + fixes ⇒ refine rtf 0.037→0.38, blind
    0.34→0.95, RSS halved, results bit-identical — evidence: before/after
    per-phase bar + rtf table — sources: results/vk_bench/profile_report.*,
    commit 1ca7581.
18. **Today: blind seeding v2** — message: FLY124's failure was a scan-band
    bias (alias 91/3 outscored the real comb); band-capped matched-filter scan
    fixes seeding (both twin pairs found); arms T/C/N/K + spatial-DP ladder in
    a 64-run sweep NOW RUNNING — evidence: seed-before/after table
    ([30.0,30.85,91.3,92.3] → [74.7,75.7,91.3,92.3] vs truth
    [73.9,74.5,81.3,91.0]) + round-1 arm table; mark round-2 cell "(sweep in
    flight)" — sources: a894501 commit message, round-1
    omnirun-outputs/python-27f6d4 sweep_report.csv.
19. **JASA-GP: the model** — message: GP over (mic pos, condition, time) with
    BPF-informed Fourier kernel — evidence: kernel formulas — sources:
    src/experiments/gp_rotor_noise/jasa_gp.py + 07-06 report.
20. **CONA + original data** — message: how CONA auralizes (FW-H F1A tonal +
    BPM broadband + Griffin-Lim); NASA 1-Pax quad, 256 ground mics — evidence:
    pipeline schematic + jasa_gp_eval_slim.png (existing) — sources: auraflow
    jasa module docs, 07-06 report.
21. **Adapting to our rigs** — message: DREGON + Matrice-100 reconstructed in
    CONA; prescribed constant RPS 40–85 (static-stand protocol); 64 shell mics;
    40-case dataset generated TODAY — evidence: NEW figure: the two drones'
    geometry + mic shells + rps grid; dataset card (40 cases, 2 s, 44.1 kHz) —
    sources: auraflow drone_egonoise.py docstring, drone-egonoise@2e6644161ce1.
22. **GP trained on it** — message: dregon GP: seed-holdout corr 0.75 /
    rel_resid 3.7%; held-out rps-60 interpolation corr 0.87 / 1.8% (matrice100
    training queued) — evidence: overlay PNG from
    omnirun-outputs/python-fbd20f/results/gp_egonoise/dregon/ — sources:
    eval_metrics.json.
23. **RESULTS: the 4-way comparison** — message: same GT RPS, same mic — deep
    generator closest by far — evidence: spectrogram grid real|CONA|deep|GP
    with comb-mask + msSTFT numbers under each (deep 7.0/8.2/9.5 dB, msSTFT
    6.9; GP 39.1/26.8/19.1, 15.6; CONA 50.1/52.9/52.0, 30.9) — export figures
    from notebooks/noise_four_way_comparison.ipynb / four_way_lib.py —
    sources: notebook (commit 41a941a).
24. **Discussion: is the GP bad, or its data?** — message: the gap is largely
    CONA's, inherited: BPF-only (even-k) harmonics from identical-blade FW-H,
    no jitter linewidth, resolution-limited comb, broadband truncation bug;
    plus GP high-harmonic shrinkage; AND the coherence finding again — all
    three synthetics far too coherent (wind channel = the lever) — evidence:
    small table of gaps (CONA-structural vs GP-regression vs deep-residual) —
    sources: notebook verdict cells.
25. **Initiated: f1 blind-SE baselines** — message: modern no-RPS floor before
    any RPS-informed claim; 5 archs × 2 passes (drone-only vs all-harmonic
    category-uniform); valid sets published; anchors done; runs queued —
    evidence: arch table (from f1 batch doc: Edge-BS-RoFormer, TF-GridNet
    8.38M, MP-SENet 1.71M, DCUNet, SGMSE+ 65.6M from-scratch) + protocol
    one-liner + anchor floor row — sources:
    docs/experiments/f1-se-blind-baselines.md, results/f1_tables/f1_tables.md.
26. **Initiated: RPS predictor to VK parity** — message: VK is accurate but
    (pre-optimization) slow and non-causal; predictor ideas: longer temporal
    context (VK integrates 20 s vs 1 s chunks), comb-structured front-end,
    VK-distilled training targets, VK-annotated unlabeled data — evidence:
    speed/accuracy quadrant sketch (VK vs neural now vs target) — sources:
    campaign notes; keep to one slide.
27. **Takeaways** — message: geometry fixed for good; wind = the one shared
    realism gap (now measured); VK now fast + de-biased, sweep closing; deep
    generator confirmed best-in-class vs literature baseline; two programs
    running toward the SE decision — sources: all above.

## Cut (considered, excluded)
- Koopman/bidirectional-latent survey — literature only, no results; one
  spoken mention at takeaways if at all.
- Kalman-tracker post-mortem, E5–E12 sim2real story — covered by the 07-13 deck.
- Gulli-paper reconciliation + oracle-SE negative evidence — belongs to the SE
  decision deck once f1 floors exist; here only as one motivation line on
  slide 25.
- The commented-out jitter-floor slide — keep commented (speaker may revive).

## Open questions for the user
- none — speaker notes are binding and complete.

## NON-NEGOTIABLE REVIEW CRITERIA (user-added mid-build; creator and critic MUST enforce)
1. The VK method slides (sections 11–15) must explain the process at
   REIMPLEMENTATION fidelity: a reader should be able to rebuild the tracker
   from the slides alone. Pseudocode blocks are explicitly welcome (e.g. the
   outer loop: demod → coupled banded solve → phase-slope update → anneal;
   with shapes/grids annotated). Prefer one honest pseudocode block over prose.
2. No large unused white space on any slide — fill with the figure/formula at
   larger scale, or merge sparse slides.
3. ZERO unbacked statements: every claim on a slide must be supported by a
   number or plot VISIBLE on that slide (or explicitly marked as pending, e.g.
   "sweep in flight"). If a claim has no demonstrable number/figure, either
   produce the figure or cut the claim.
