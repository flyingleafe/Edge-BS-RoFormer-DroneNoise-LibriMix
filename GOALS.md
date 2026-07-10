# Goals & Constraints

Last reviewed: 2026-05-04
Status: bootstrap (durable sections frozen; portfolio section live, next review at week 7–8 of experimental work)

## Project Goal

Develop and defend, by submission, a PhD dissertation on **speech enhancement under harmonic noise from rotating sources**. Approach: exploit the *structural* properties of harmonic noise (periodicity, RPS-driven harmonic combs) rather than treating it as generic non-stationary noise. Operating regime: ultra-low SNR (0 to −30 dB).

The unifying technical thesis ("bigger" claim) is **deliberately deferred** — see C5. It crystallizes from whichever experimental bet shows signal at the mid-point review.

## Deadline Structure

| Offset | Gate | What's required | Failure mode |
|---|---|---|---|
| **+7 months** (~2026-12) | Write-up transfer | Credible thesis direction + preliminary results convincing to panel | **Sponsorship expires → project ends.** Hard cliff. |
| +24 months (~2028-05) | Submission | Full dissertation | Soft. 17 months for write-up and finishing experiments. |

The **+7-month gate is the binding constraint**. All experimental planning is indexed to it. The submission deadline is informational.

## Constraints (durable)

**C1 — Framing.** Dissertation framed as SE under harmonic noise from rotating machinery / rotating sources of harmonic noise. Drones are case studies, not the subject. Vocabulary in dissertation prose follows: "rotating machinery", "tonal/harmonic noise sources", "RPS-instrumented periodic interference" — not "UAV", "drone", "quadcopter" except inside the case-study chapter.

**C2 — Integrity.** C1 was the basis of ATAS approval (granted on second application, after first was rejected for being "painfully dual-use"). The general claim must be honestly defensible at viva — not a relabelling exercise.

**C3 — Generalization evidence.** At least one experimental result on non-drone rotating machinery is required before submission. Primary target: **MIMII + speech mixing** (industrial fans, pumps; 16 kHz; no native speech or RPS labels — both must be added). Secondary: AeroSonicDB (propeller aircraft).

**C4 — Method generality.** No silent drone-specific assumptions in architecture. The RPS-handling components (RotorEncoder, range assumptions, harmonic priors, rotor count) must work across at least two RPM regimes: drone ~3000–10000 RPM and industrial ~600–3600 RPM.

**C5 — Deferred thesis.** The unifying dissertation thesis is **not** chosen at bootstrap. It crystallizes from whichever bet shows signal at the mid-point review (~month 4). Candidate framings under consideration:

- RPS as universal latent for rotating-source SE
- Physics-informed analysis-by-synthesis at extreme SNR
- Auxiliary-task disentanglement for structured noise

Do not commit to a framing before the data demands it.

**C6 — Deadline cliff.** +7 months = write-up transfer = sponsorship gate. Binding constraint. All bets, kill criteria, and capacity decisions are indexed to this date.

**C7 — Portfolio discipline.** At most 3 concurrent experimental bets. Each has hypothesis, MVP, kill criterion, and time budget *before* starting. Mid-point review at week 7–8 of experimental work is **mandatory** and gates further investment. **Plan D** (acceptable-failure fallback — a clean benchmark study of existing SE methods on DN-LM/DREGON-LM, transfer-doc-grade) decision is deferred until mid-point; operate as if Plan D is acceptable until then.

**C8 — Workflow.** Agentic dev is assumed (project conducted in 2026). Division of labour:

- **Agent owns:** scaffolding, data pipelines, eval harnesses, plotting, sweeps, refactoring, dataset preprocessing.
- **Human owns:** hypothesis design, debugging novel research anomalies, result interpretation, prioritization, choosing what to do next.

Compute/wallclock is the binding throughput constraint, not code-writing speed. Bounded, well-specified sub-tasks (e.g., MIMII+speech dataset construction) may be delegated to the supervisor's students; "help me with research" delegations are not.

## Current Portfolio (live)

Last bets review: 2026-05-04 (**pre-supervisor discussion** — to be revisited)
Next mandatory review: week 7–8 from start of experimental work (mid-point)

Three concurrent bets, all RPS-clustered, on purpose: that is where months of prior work give momentum and the unique research angle. Headlines below; per-bet detail cards live under `docs/experiments/bets/` once started.

### Bet 1 — Pseudo-RPS replaces oracle telemetry

- **Hypothesis.** The conditioning signal that helps SE is *RPS itself*, not *telemetry per se*. Replacing oracle RPS with SP-extracted pseudo-RPS (cepstral / harmonic-product / pYIN-on-noise) recovers most of the oracle-conditioned gain.
- **MVP.** In current P2 architecture on DREGON-LM, swap telemetry RPS for SP-extracted pseudo-RPS at inference. Compare SI-SDR / STOI vs (a) telemetry-conditioned upper bound, (b) telemetry-blind baseline. At −10 and −20 dB.
- **Kill criterion.** Pseudo-RPS yields <20% of the (oracle − blind) gap at −10 dB SNR.
- **Time budget.** ~3 weeks.
- **If it works.** Telemetry-free deployment becomes plausible → cross-domain generalization plausible → C5 (RPS-as-latent thesis) stays alive.

### Bet 2 — Multi-task RPS regression head (telemetry-at-train-only)

- **Hypothesis.** Forcing the SE model to also predict RPS as an auxiliary task during training improves the SE branch even when telemetry is unavailable at inference. Disentanglement-via-supervision.
- **MVP.** Add an RPS regression head to the existing SE model on DREGON-LM, train with telemetry, evaluate without it. Compare to telemetry-blind baseline and telemetry-conditioned upper bound.
- **Kill criterion.** Multi-task SE matches or underperforms the RPS-blind baseline on SE metrics.
- **Time budget.** ~2–3 weeks.
- **If it works.** Independent paper; most deployable setup (telemetry at train, none at inference). Independent of Bet 1's outcome.

### Bet 3 — Analysis-by-synthesis with a parametric harmonic-comb noise model

- **Hypothesis.** At extreme SNR (−20 to −30 dB) where neural SE collapses, a parametric generative model of the noise (sum of RPS-driven harmonic combs with learned/estimated amplitudes and decay) subtracted from the mixture recovers speech better than learned methods, because the noise has very low intrinsic dimensionality.
- **MVP.** Implement parametric comb model with oracle RPS on DREGON-LM. Subtract from mixture. Evaluate at −20, −30 dB. Compare to a strong neural baseline.
- **Kill criterion.** Parametric subtraction at −20 dB SNR is no better than spectral subtraction.
- **Time budget.** ~3–4 weeks.
- **If it works.** Strong "physics-informed methods beat ML when noise is structured" narrative. If it loses to neural, still a defensible interpretable lower bound and a transfer-doc figure.

### Parked alternatives

Explicitly considered and not chosen for this round; re-evaluate at mid-point review:

- Diffusion / flow-matching SE conditioned on RPS — strong, but ~4–6 weeks for a baseline; no infra reuse with current code.
- Multimodal (visual / video) — DREGON video unreliable; different PhD.
- Foundation-model SE fine-tuning — guaranteed mediocre, no novel angle. Reserved as Plan D ingredient.
- Active noise cancellation using RPS — different problem (intervention, not post-hoc).
- RPS-driven Kalman harmonic tracker ("complex-KLA" filter) — **tested and killed at its Phase-0 gate (2026-07-10, K2)**: causal per-harmonic Kalman tracking matches framed lstsq_VP under oracle RPS (needs the per-order joint update) but degrades *faster* than lstsq under RPS drift — process-noise widening cannot absorb systematic rotation error. Full post-mortem + salvage lessons for any learned (KLA-layer) revival: `docs/experiments/bets/kalman-harmonic-tracker.md`.

### Sequencing (indicative)

```
Wk 1–3   Finish P2 telemetry-given baseline (must land regardless)
Wk 1–3   Bet 3 in parallel (classical, low-coupling to neural pipeline)
Wk 4–6   Bets 1 and 2 in parallel (share P2 infrastructure)
Wk 7–8   Mid-point review — pick the winner, decide on Plan D
Wk 8–12  Double down on winner; extend to MIMII / second domain (C3)
Wk 13–18 Crystallize C5 thesis; draft transfer doc
Wk 19–28 Buffer + transfer doc finalization + viva prep
```

## Document Discipline

- **Durable sections** ("Project Goal", "Deadline Structure", "Constraints") edit only when the underlying situation changes (new bureaucratic constraint, missed deadline, scope shift). Each edit is dated in the header.
- **Live section** ("Current Portfolio") is updated whenever a bet is killed, modified, or added. **Mandatory rewrite at the mid-point review.**
- Strategic routing and directory structure live in `AGENTS.md`. Concrete how-to lives in skills under `.pi/skills/`. This file answers *what* and *why*, not *how*.
- Completed/background experiment write-ups (motivation/results/conclusion) live in `docs/experiments/`; the live bet detail cards below get their own cards under `docs/experiments/bets/` once a bet starts.
