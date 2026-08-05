# Thesis Program — The Probabilistic Rotating Latent State

Status: draft v1, 2026-08-03. Decomposed from the user's statement of direction
(discussion of 2026-08-03). This document holds the *stretch program* (§2–§4),
the *minimal defendable thesis* (§5), and the gates that connect them (§6).
GOALS.md stays the authority on constraints and deadlines; this document is the
candidate C5 thesis crystallization.

## 1. The target, stated precisely

Build a method that reads audio of rotating machinery and infers a
**probabilistic rotating latent state**: a per-source latent stochastic process
that carries, at minimum, the rotational phase/rate and the source timbre, with
calibrated uncertainty. One representation, several downstream heads:

- **RPS prediction** — read the rate component of the latent.
- **Noise reconstruction** — decode the latent back to the coherent noise field.
- **Noise separation / SE** — subtract or mask the reconstruction; enhance the residual.
- **Fault detection** (optional) — anomaly detection on the latent trajectory.
- **Machine-type classification** (optional) — classify the timbre component.

Training: mostly self-supervised on unannotated recordings of rotating
machinery. Labels (telemetry) serve for evaluation and for a small supervised
calibration head, not as the primary training signal.

This target satisfies C1/C3/C4 by construction: nothing in the formulation is
drone-specific, and MIMII-class industrial audio is a natural second domain.

### 1.1 The claim decomposed

The program is a conjunction of five claims. Each one is falsifiable on its own:

- **T1 — Identifiability.** The audio determines the rotating state to a usable
  precision. *Largely established*: the VK program measured what is and is not
  recoverable (comb frequency to ~0.2 rev/s short-window; long-horizon phase
  unrecoverable; acoustic comb displaced 0.3–0.5 rev/s from the mechanical
  shaft under translation; coherence times τ_k ≈ 0.4–1.7 s).
- **T2 — Learnability.** A neural encoder can extract this state, and
  structure helps (CKLA's comb-aware recurrence vs generic architectures).
  *Partially established*: CKLA phase-only is the best neural tracker; it does
  not yet beat the classical chain on DREGON.
- **T3 — Self-supervision suffices.** The state is learnable without telemetry
  at scale. *Open.* Candidate levers already scoped: VK pseudo-labels
  (annotator exists, AVQ/SPCUP annotated), equivariance SSL
  (pred(scale_α x) ≈ α·pred(x)), reconstruction likelihoods. Warning from the
  generator line: naive probabilistic objectives collapse the comb into
  variance ("being uncertain is cheap") — any SSL likelihood needs a
  comb-structure term and a harmonic-structure metric in the eval.
- **T4 — Downstream utility.** Knowing the rotating state to the achievable
  precision improves at least one downstream task over a blind model. *Open
  and load-bearing.* The RPS-conditioned SE experiment (strong backbones,
  structural conditioning) is the first honest test.
- **T5 — Generality.** T1–T4 transfer to a second RPM regime / machine class
  (C3: MIMII or AeroSonicDB). *Open; required before submission.*

## 2. What each existing line contributes to the program

| Line | Contribution to the program | Status |
|---|---|---|
| VK / classical tracking | T1 measurements (τ_k, displaced comb, k² weight law, capture ranges); the classical baseline every learned method must beat; the paper | Done; paper in flight |
| CKLA + neural predictors | T2 evidence; the architecture family for the encoder; the training-regime lessons (window length, full-envelope data, time-warp) | Active |
| Noise generators | The decoder half of T3; the likelihood-degeneracy lesson (what a probabilistic objective must not do); silence/ramp coverage | Paused, banked |
| SE baselines (F1/F2) | The clean yardstick for T4; the corrected model ranking; leakage protocol | Fix done; sets to rebuild |
| RPS-conditioned SE (next) | The T4 gate itself | Starts now |
| Benchmarks / datasets | The evaluation substrate for everything; DREGON + Michael's + AVQ + (later) MIMII | Continuous |

The honest one-line summary of two months: *the T1 half of the program is now
measured rather than assumed — and that measurement is what tells us the
latent state must be probabilistic (phase is a short-coherence stochastic
process, not a deterministic angle).*

## 3. The stretch program, Aug–Dec 2026 (revised 2026-08-03)

Four parallel tracks, not three sequential stages. P2 and P3 are split into
gated sub-stages; each sub-stage carries its own thesis-reduction contingency,
so a failure reduces the thesis instead of stalling it.

**Scheduling model (revised 2026-08-04).** Implementation is not the
constraint: with agentic development, each stage's build is 1–2 days. The
binding resources are (1) GPU wall-clock and (2) *debug-and-retry cycles* in
uncharted territory. One cycle = implement → train → diagnose ≈ 1 week.
Charted stages (Track B: known models, known data, known metrics) get 1 cycle
each. Uncharted stages (Track C: new training loops, new objectives; Track D:
new model class) get an explicit multi-cycle budget below — that budget IS the
schedule, and an unused cycle rolls forward as slack, never as scope creep.

### Track A (Aug → early Sep) — finalize the tracking efforts

- Paper 1 (coupled-VK blind RPS) to submission: baselines, Table VII, figures.
- CKLA wrap-up: the post-recalibration re-score, the 8s-ft arm verdict, and a
  frozen "best neural tracker" checkpoint + number for all later tracks.
- Exit: paper 1 submitted; the tracking lines go into maintenance.

### Track B (Aug → Sep, charted: 1 cycle per stage) — RPS-conditioned SE

- **B1 (running, gate ~08-10)** — the clean grid: F1-corrected valid sets
  rebuilt (done, repinned 08-04), F2 leakage discipline (unseen-noise split;
  unseen-drone hard tier), all four baselines retrained without anomalies.
  Gate G-B1: trustworthy blind reference numbers exist. *Contingency: none
  needed — this work is mandatory for every thesis shape.*
- **B2 (~08-11 → 08-24)** — oracle conditioning on MP-SENet +
  Edge-BS-RoFormer via structural mechanisms: (a) coherent comb
  reconstruction/subtraction front-end (VK envelopes), (b) harmonic-aligned
  band splitting, (c) FiLM per band. Implementation 1–2 days; the rest is
  training wall-clock. Gate G-B2 (early read of G-P1, ~08-31): any mechanism
  beats blind at −10/−15 dB.
- **B3 (~09-01 → 09-21)** — pseudo-RPS conditioning: replace oracle with the
  blind VK chain and with the best neural tracker at their *measured*
  precisions; measure the gain-vs-precision curve (closes the loop with
  paper 1's ceiling; the thesis's central figure if positive).
- **Gate G-P1 (expected ~09-30; hard backstop 2026-11-30)**: does RPS
  knowledge at achievable precision improve SE on a strong backbone? The
  Oct–Nov window between expected and backstop is *reserve*, spent only on
  re-runs a murky verdict demands — by default it goes to Tracks C/D.
  *Contingency (negative)*: the result publishes as a rigorous negative with
  the ceiling explanation (chapter 5 of §5); paper 2 falls back per §3.5.

### Track C (Sep → mid-Nov, uncharted: budgeted cycles) — P2 refined:
self-supervised rotating-state learning

Track C starts ~09-01 (infrastructure prep — corpus assembly, pseudo-label
runs on DroneAudioSet/MIMII — can start in August on CPU while Track B holds
the GPUs).

- **C1 (Sep, budget 2 cycles)** — VK pseudo-label distillation at scale
  (annotator exists; AVQ/SPCUP done; add DroneAudioSet, MIMII). Gate G-C1
  (~09-30): distilled model matches the supervised predictor on the fixed
  raw protocol after small supervised calibration. *Contingency: fails →
  thesis keeps telemetry-at-train supervised training; drop the
  "unannotated" claim; C2/C3 descoped to exploratory appendix work.*
- **C2 (Oct, budget 2 cycles)** — scale-equivariance SSL on top of C1.
  Gate G-C2 (~10-15): measurable gain over C1 alone. *Contingency: fails →
  C1-only ("distillation from a classical expert"), still a coherent
  chapter.*
- **C3 (Oct → mid-Nov, budget 3–4 cycles — the largest single allocation in
  the program)** — comb-structure-guarded reconstruction likelihood as an
  SSL objective. This is a *new training loop with a known failure mode*
  (the §1.1-T3 variance degeneracy): expect the first cycle to fail on the
  objective, the second on the metric, and budget accordingly. Requires a
  harmonic-structure metric in the eval and an objective that cannot buy its
  way out through variance. Gate G-C3 (first read ~10-31, final ~11-15): the
  latent carries calibrated phase/rate information (probe-measurable).
  *Contingency: fails → the "probabilistic" qualifier reduces to
  rate-uncertainty only; the reconstruction head is dropped from the thesis
  claim.*

### Track D (Sep → Nov) — P3 refined: the joint model + generality

- **D3 first (Sep–Oct, MANDATORY, charted, ~1 cycle + data work)** — the
  C3-constraint generality result: blind VK chain + generator on MIMII-class
  industrial audio + speech mixing. Required for *every* thesis shape, uses
  only existing, proven components — so it is charted work and moves EARLY
  (September data assembly alongside C1, October run), not into the November
  crunch. Whatever of C/D survives later re-runs on it as a bonus.
- **D1 (mid-Oct → mid-Nov, budget 2–3 cycles)** — two heads (RPS +
  reconstruction) on drone data, encoder from the best of Track C (or
  supervised if C1 failed). New model class = uncharted. Gate G-D1 (~11-15):
  both heads within noise of their incumbents. *Contingency: fails → no
  joint-model chapter; heads stay separate chapters; program §1 becomes the
  "future work" section of the transfer document.*
- **D2 (Nov, budget 1–2 cycles, first casualty of overrun)** — add the
  separation head (consumes reconstruction; competes with Track B's
  conditioned models). Gate G-D2 (~11-30): separation head competitive with
  B2/B3's best. *Contingency: fails or unstarted → joint model claims RPS +
  reconstruction only; D2 moves past the transfer into the dissertation
  proper.*

### 3.5 Paper 2 (submission early Dec, selected ~2026-11-15)

Decision ladder, first satisfied wins:

1. **G-P1 positive** → "RPS-conditioned speech enhancement under harmonic
   noise: gain vs conditioning precision" (Track B; the strongest paper).
2. **G-C1/C2 positive** → "Self-supervised rotating-state estimation from
   audio, distilled from a classical tracker" (Track C).
3. **Fallback (always available)** → the benchmark-integrity paper: F1
   anatomy + F2 leakage + the corrected baseline suite + non-leaked protocol
   (Track B1 material; largely exists by October).

The ladder guarantees an early-December submission exists in every branch.

### What December needs from this

The +7-month transfer gate (~early Dec 2026): paper 1 submitted (Track A),
paper 2 submitted or in final draft (§3.5), G-P1 resolved either way (Track
B), Track C gates resolved at least through G-C1, D3's generality result in
hand, and the transfer document assembling §5's chapters with the surviving
§1 program as the forward plan.

## 4. Why this program is credible from where we stand

- The latent state's *form* is dictated by measurement, not taste: short-
  coherence phase (τ_k measured), per-harmonic amplitude envelopes (VK solve),
  a per-source displacement between acoustic and mechanical rate (displaced
  comb), and heavy-tailed rate dynamics (OU fits). A deterministic-phase
  model is refuted before we build it — this is the direct architectural
  payoff of the VK program.
- CKLA already is a rotating-latent encoder in miniature (complex recurrence
  at comb-locked rates; the phase-only readout winning is evidence the latent
  carries phase structure profitably).
- The generator already is the decoder in miniature, and its failure mode is
  mapped (variance degeneracy), which de-risks the P2/P3 likelihood design.
- The risks are named: T4 may fail for SE (G-P1), SSL may not suffice (G-P2),
  and the joint model may lose to specialists everywhere (G-P3). Each gate
  has a survivable exit.

## 5. The minimal defendable thesis (fallback, decided at G-P1/G-P2)

**Title shape:** "RPS-informed speech enhancement under harmonic noise from
rotating sources: limits, methods, and benchmarks."

Chapters buildable from results that exist today:

1. **Background & datasets** — DREGON-LM, Michael's corpus with recalibrated
   telemetry, AVQ; the mixing/streaming pipeline. (Exists.)
2. **Blind acoustic RPS tracking** — the paper: the coupled-VK chain, the
   neural family, the hybrid, and the measured information ceiling. (Paper in
   flight; strongest chapter.)
3. **Trustworthy evaluation at ultra-low SNR** — F1 bug + corrected baselines;
   F2 leakage anatomy + non-leaked protocol; the corrected model ranking.
   (Exists; needs the valid-set rebuild to finalize.)
4. **Synthetic rotating-noise modeling** — FWH/GP physics baseline, the neural
   generator with linewidth/silence findings, the likelihood post-mortem as a
   methodological contribution ("proper scores are not automatically good
   training objectives" — with the measured degeneracy). (Exists.)
5. **RPS-conditioned SE** — whatever G-P1 produces, positive or negative, plus
   pseudo-RPS conditioning at measured tracker precision. (The one chapter
   that needs new results; the experiment runs regardless of which thesis
   shape wins.)
6. **Generalization** — the C3 result on MIMII-class audio: at minimum, the
   blind tracker + generator transferred to industrial rotating machinery.
   (Needed in every scenario; schedule in Oct–Nov.)

This is defendable with **zero** further architectural breakthroughs: chapter 5
can carry a rigorous negative ("conditioning at achievable precision does not
help strong backbones — and here is the information-theoretic reason, from
chapter 2's ceiling"). It is not the preferred outcome, but it is a coherent
dissertation, and every experiment in §3 feeds it even when its gate fails.

**Rejected fallback:** pivoting to machinery diagnostics as the primary frame.
Expressly undesirable (supervisor discussion); diagnostics stays a *motivation
and outlook* item, never the thesis subject.

## 6. Decision calendar (revised 2026-08-04)

| Date | Decision | Input |
|---|---|---|
| ~2026-08-10 | G-B1: clean SE grid + retrained baselines accepted | Track B1 (running) |
| ~2026-08-31 | G-B2: early oracle-conditioning read | Track B2 |
| ~2026-09-05 | Paper 1 submission | Track A: baselines, Table VII, figures |
| ~2026-09-30 | **G-P1 expected** (pseudo-RPS gain-vs-precision curve) | Track B3 |
| ~2026-09-30 | G-C1: distillation matches supervised? | Track C1 |
| ~2026-10-15 | G-C2 | Track C2 |
| ~2026-10-31 | G-C3 first read + D3 generality result in hand | Tracks C3, D3 |
| ~2026-11-15 | **Paper 2 selection** (ladder §3.5) + thesis shape (stretch vs §5) + G-C3 final + G-D1 | all gates to date |
| ~2026-11-30 | G-D2 (or its declared descope) + G-P1 backstop if reserve was used | Tracks D2, (B) |
| ~2026-12 (early) | Paper 2 submission + transfer write-up submission | everything above |
