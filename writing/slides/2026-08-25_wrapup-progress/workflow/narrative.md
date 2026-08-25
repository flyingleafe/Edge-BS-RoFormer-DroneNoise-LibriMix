# Narrative — Rotor Speeds from Ego-Noise: the Wrap-Up, One Table
kind: slides
audience: supervisor + group; they know the project's history (VK tracking,
generators, curricula) from previous decks; they have NOT seen the unified
leaderboard, the honest-silence regime, the gates, or the test set.
through-line: The wrap-up paper now has one frozen protocol that every
method — classical, neural, blind — meets, and the campaign of the last week
answered its central open question: the zero-regime failure was a training
honesty problem, and fixing the data regime (silence with content + an SNR
floor + a voicing gate) closes most of the gap synthetic curricula used to
cover. The blind classical tracker still owns cruise precision at 10x
realtime cost. Everything else in flight lands into the same table.

## Sections (ordered)
1. The question and the paper — message: one paper, six-bullet arc,
   per-rotor speeds from onboard audio with no published direct method —
   evidence: paper abstract structure — sources:
   writing/papers/2026-08_wrapup/src/index.tex (abstract, contributions).
2. Data reality and the three-way split — message: ~1 h of labeled audio,
   two drones; train/frozen-valid/reserved-test now formal, including two
   newly calibrated MD2 flights (a shared 1.19% session clock dilation was
   the catch) — evidence: splits table; calibration numbers — sources:
   paper sec:splits; docs (michaels-test); dload michaels-test-frames.
3. One protocol, five regimes — message: every number in the paper comes
   from 37 clips x 8 channels under per-frame Hungarian PIT with per-regime
   pooling; training regimes are named R1-R5 and nested — evidence: protocol
   box + R1-R5 list — sources: docs/experiments/unified-baseline-eval.md.
4. The training-free floor — message: classical pitch methods fail 2-3
   orders of magnitude on the real task; NMF is the best training-free
   cruise number (8.1 MAE) and beats the 2026 OT method (16.3); no
   classical method can say "stopped" — evidence: classical five + OT
   table — sources: unified-baseline-eval.md (results), the restored
   2026-05-29 report for provenance.
5. What the models actually read — message: a 2% frequency scaling moves
   predictions 0.03%; the zero-regime failure traced to a level shortcut
   (every training silence was globally quiet; only 26 s of unique silence,
   one room) — evidence: probe number; regime distribution of training
   time; the rumble-clip failure — sources: paper sec:augmentation;
   honest-base-frontends.md (motivation).
6. The honest regime R2 + voicing gates — message: three fixes (zero-labeled
   silence arm with content at 17% of chunks, reference-power floor on the
   speech scaling, sigmoid voicing gate on the head) — evidence: design
   sketch; policy shares — sources: honest-base-frontends.md.
7. PUNCHLINE: the HB 3x3 grid — message: the honest regime alone closes
   most of the zero gap (11.8 -> 3.7 MAE) at almost no cruise cost; the
   front-end winner is architecture-dependent (IF for scv2, magnitude for
   the Transformer, synchrosqueezed for the causal GRU, where it repairs
   the zero deficit); best cell Transformer 31.7 vs real-only 42.3 —
   evidence: 3x3 table with per-regime rows; the rumble-clip panel
   (0.51 vs 19.6 MAE) — sources: unified-baseline-eval.md (HB rows),
   writing/papers/2026-08_wrapup/figures/qual_zero.*.
8. The blind tracker meets the same protocol — message: ungated it beats
   every neural cell on cruise MAE (2.27) and fails silence; gated it
   decides silence perfectly and loses half its cruise windows (gates were
   precision-calibrated); 10 CPU-s per audio-second — evidence: two-row
   table + compute number — sources: unified-baseline-eval.md (blind rows).
9. Synthetic data: coverage, not realism — message: mixed-in synthetic hurts
   (1.8-3x), curricula helped mainly where real data lacked coverage
   (silence, ramps); with R2 honest coverage the remaining synthetic edge is
   being re-measured by nested reruns — evidence: R5 vs real-only numbers;
   m3cur/comb regime attribution; "in flight" marker — sources:
   generator-refined-labels.md, unified-baseline-eval.md (rerun design).
10. Seeing the errors — message: three qualitative panels (rumble silence,
    stop-start transition, cruise) make the failure modes inspectable —
    evidence: the three generated figures — sources:
    writing/papers/2026-08_wrapup/figures/qual_*.png + sidecar JSON MAEs.
11. In flight tonight — message: nested R3-R5 reruns, salience baselines
    retrained (with a zero-decode fix and a widened narrow-SR grid whose
    round-trip floor drops 7.2 -> 2.25), ebsrof + CKLA rows on R2, cloud
    lanes added — evidence: fleet table with pending cells — sources:
    TODO.md, unified-baseline-eval.md (remaining rows).
12. Next bet: HG-CKLA — message: the CKLA layer becomes a true pi_kalman
    pass by moving the measurement inside the recurrence (state-conditioned
    harmonic gathers, innovation phasors, WP18 k^2 weights); gates G1-G3 —
    evidence: one architecture sketch — sources: docs/pikalman-ckla-design.md.

## Cut (considered, excluded)
- FLY103/FLY108 calibration mechanics — one line on slide 2; the full lag
  fit story is report material, not deck material.
- The classical-baselines report archaeology (restored from git) — a
  footnote on slide 4 at most.
- Infra (omnirun lanes, sentinels, gridrun) — process, not results.
- The May single-rotor SimpleConv-collapse finding — belongs to the paper's
  related-work discussion; the deck's structural-prior point is carried by
  slide 5 already.

## Open questions for the user
- none blocking; --no-user run.
