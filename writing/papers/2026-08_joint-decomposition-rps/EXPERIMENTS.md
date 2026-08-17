# Experimental section — what remains to be run

Status date: 2026-08-17 (evening). One line per experiment; grep the draft for
`\pending{` / `\wip{` to see where each lands in the text. "Dev-measured" means
the number exists from a development campaign and needs a consolidated re-run
or only typesetting; "not run" means real compute is missing.

Legend: **[H]** heavy compute (cluster, hours+) · **[L]** light (local or
minutes) · **[W]** mainly a writing/decision task · **(dep: …)** blocked by.

## D — Decomposition (Sec. results/res-decomp)

| ID | What | Status |
|----|------|--------|
| D1 | Headline excess-retained table, per band × recording, at telemetry AND at refined labels | telemetry column DONE at v3e (DREGON 6.9/8.4/15.4/7.6 %, FLY124 0.5/6.3/36.6/47.2 %, FLY125 0.3/5.0/36.3/52.8 %). **Caveat: the DREGON arm is already refined-conditioned** (the driver applies the sidecar override); a RAW-telemetry DREGON arm is needed for the contrast, and refined-label FLY arms need the FLY sidecars (dep: refine-fly job, running). [L per arm — one vk_decompose run each] |
| D2 | Component-wise SNR on synthetic S3 (exact ground truth for the split) | not run under the joint solver. Needs: S1–S3 synthesis at the frozen seeds → joint decompose → component SNR. [H, ~hours] |
| D3 | The three-channel (regime-3) story in the paper | **[W]** the draft predates the stochastic channel entirely: the model section needs the coherence-ladder regimes and the per-bin √(S/P̃) extractor; the metric section needs `residual_final` as the gate reading. All supporting measurements exist (linewidth law, v3c flat-gain failure autopsy, v3e numbers). |
| D4 | FLY high-band miss vs the ≤10 % target | **[W + decision]** 37–53 % retained at k≥25 on FLY at ≤0.09 dB absolute contrast — the estimation-floor argument (width-factor sweep insensitivity) is measured. Either (a) restate the gate as retained % OR absolute-contrast cap, with the floor argument, or (b) run synchronous shaft-angle averaging to push it down [H, unbuilt]. |
| D5 | Figure F-D1 (before/after residual spectrograms + inset numbers) | assets exist from the v3e artifact build; needs paper-grade regeneration. [L] |
| D6 | Floor-seam fix (time-interpolated floor) | measured today (boundary jumps 16.9×→1.5×); one sentence + optional supplementary figure. [W] |

## M — The measure (Sec. res-measure)

| ID | What | Status |
|----|------|--------|
| M1 | J-rescoring table: telemetry / refined / blind / fan on the 5 frozen windows | dev-measured through the whole ladder: profiled 2/5 → +marginal 2/5 (null) → +H-aware 4/5; refined beats telemetry on the nosource window (−17.620 vs −17.611 Jh/cell). **The pre-registered criterion (refined ≤ telemetry < fan everywhere) currently FAILS on 1/5** — the DREGON nosource fan wins via a floor-fit artifact (fully autopsied: −186k at 3.2–8 kHz where residuals are identical; blanket-band H absorption). Next lever queued: hypothesis-independent floor (rent cancels; kills the tilt channel exactly) [L — observer-side rescore change + one job]. Also: refined hypothesis on the remaining 4 windows (dep: FLY sidecars). Paper decision if the lever also nulls: report 4/5 with the autopsy as a finding (arguably stronger than a clean pass). |
| M2 | The least-squares failure exhibit (fan windows, numbers + figure) | dev-measured; typeset only. [W] |
| M3 | Independent-instrument consistency (−0.596 % scale vs comb-scan CI) | measured; typeset only. [W] |
| M4 | Profiling-vs-marginalization discussion | **[W]** the draft says "if profiling proves insufficient, marginalization is the escalation" — that experiment has now been run (exact Gaussian envelope marginalization: measured null on rankings). Rewrite the discussion around the measured ladder; also fold in the two lever nulls (adaptive floor / Lorentzian H) and the dense-comb defeat measured in the acceptance fixture. |

## R — Refinement (Sec. res-refine)

| ID | What | Status |
|----|------|--------|
| R1 | Capture curves: ε ∈ {0.1,0.3,0.5,1.0,2.0} on S1–S3, ours + all baselines (IAVKF, VKF+SWT, tacholess OT, VK-loop ablation) | **not run under the joint solver — the largest remaining compute item.** Baselines are implemented from the previous campaign; the sweep is windows × seeds × ε × methods. [H, cluster, likely the overnight centerpiece] |
| R2 | Real-data refinement table (telemetry → refined per recording, cruise-scale numbers) | sidecars now exist for nosource + speech-low + whitenoise-low (scales −0.53 %/−0.43 % measured today); FLY124/125 running (dep: refine-fly). Table assembly after. [L] |
| R3 | Downstream generator test (raw / refined / exact labels; consolidated spectra figure) | dev-measured on the old labels (washout + 8.6 dB result). A consolidated retrain on v3e-era labels is desirable but interacts with the user-held amplitude-target training decision (task #28). [H, GPU; decision needed on scope] |
| R4 | Iteration-harm re-measurement under the joint solver (the 0.167→0.268 anchor) | not re-run. [L–H depending on window count] |

## B — Blind (Sec. res-blind)

| ID | What | Status |
|----|------|--------|
| B1 | End-to-end blind table vs baselines on the frozen protocol | ours measured (0.688 / 1.027 rev/s); **baseline columns not run** (same baseline implementations as R1 — plan together). [H] |
| B2 | Tachometer-referenced blind numbers | measured; typeset with the bias-floor caveat. [W] |
| B3 | Ramp figure F-B1 | regenerate from the final chain. [L] |

## O — Cross-cutting

| ID | What | Status |
|----|------|--------|
| O1 | Runtime table (per stage, per 16 s window, single core) | not assembled; timings largely exist in job logs. [L] |
| O2 | Hyperparameter freeze: single-rule forms for b_ψ(k), b_A(k); final ladder/iteration constants | **[W + L]** the current implementation values are stable; needs the reduction pass and Table `tab:constants` fill. |
| O3 | Hard-EM floor bias materiality | not measured. [L — one proper-EM arm on one recording] |
| O4 | Model-section figures: linewidth-vs-k law, rig-commonness correlation | probe data exists; paper-grade regeneration. [L] |
| O5 | Sample-rate statement | **[W]** the draft says f_s = 16 kHz / K = 80 "below Nyquist"; the shipped v3e runs at 32 kHz with the 8 kHz modelling ceiling and K to the band edge — reconcile the numbers in Secs. model-skeleton and setup. |

## Suggested overnight batching

1. **R1 + B1 together** (shared baselines, shared windows) — the big sweep;
   everything else is small against it.
2. **D2** (synthetic tiers) — independent, parallelizable with 1.
3. **M1's floor lever** — one small rescore job; decides the shape of the
   measure section.
4. **D1's missing arms** (raw-telemetry DREGON; refined FLY) — two vk_decompose
   runs once the FLY sidecars land.
5. Everything marked [W] is writing and can proceed without compute.

## Addendum (2026-08-17 late): k-scaling of the coverage vulnerability

Rescoring the failing window at k = 65 instead of the campaign's k = 40
(same audio, same hypotheses) keeps the ranking and GROWS both margins:
the fan's from 0.135 to 0.178 per cell (its k 41–65 bands absorb
3.6–5.9 kHz), the refined-over-telemetry margin from 0.009 to 0.017.
The measure's fan-vulnerability scales with the modelled harmonic count
— a capped comb acts as a regularizer of the measure. This belongs in
the M1/M4 discussion and possibly as its own small results exhibit
(results/joint_rescore_ref65; band attribution in the w01 probe).

## Addendum 2 (2026-08-17 late): the carve, and the v4 unification path

Measured on both FLY recordings: in the band owned solely by rotor 0
(6.3–7.4 kHz), the coherent channel absorbs ~20 % of original power as a
quasi-continuum and the broadband channel sits ~2 dB below its level in
the adjacent track-free band. Root cause chain: floor smoothness is a
projection OUTSIDE the likelihood (masked/median/cepstral) → S
extrapolates low in blanket bands → whitened data term overprices that
band → the amplitude-free envelope prior lets the envelope bank absorb
the excess. Two-step repair, now in the Discussion as design-in-progress
and to be planned as experiments:
- **F1**: penalized-Whittle floor INSIDE J (explicit R(log S); no mask —
  comb explained by the model). Kills the carve + the measure's floor
  artifacts at the source; supersedes the hypothesis-independent-floor
  patch.
- **F2**: proper (finite-variance) envelope priors at high k — Gaussian
  line processes with per-harmonic power H_k and the measured 0.6k
  coherence bandwidth. MAP = Wiener smoother; unifies the stochastic
  channel into block A, estimates H_k jointly (the generator's amplitude
  targets), de-singularizes overlapping bands (ridge), with the honest
  caveat that twin-shared energy splits by prior, not data.
Also added to the draft: Sec. "The model as a joint distribution"
(explicit OU forms, the linewidth relation γ_k = 2π σ_k² τ_ν, the
improper-prior statement, the floor-constraint statement) and the
derivation of the D₂ penalties from the window-local OU limit
(D₂θ = 2πΔe · rate increments) in the MAP section.
