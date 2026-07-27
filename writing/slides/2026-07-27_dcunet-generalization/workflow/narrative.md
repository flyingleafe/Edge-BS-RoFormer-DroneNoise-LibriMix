# Narrative — deck extension: baseline replication + the CKLA campaign
kind: slides
audience: supervisor-level; has seen previous decks; the existing Part 1 (DCUNet generalization) stays as-is
through-line: Part 1 (existing, KEEP INTACT): why DCUNet wins some benchmarks and loses ours — leakage measured. Part 2 (NEW, from writing/reports/2026-07-27_ckla-campaign/report.typ + docs/experiments/ckla.md): the augmentation regime and its silent failure, the freq-scale lever against collapse-to-the-mean, the CKLA architecture, and the results ladder — with final numbers explicitly PENDING because the staging-bug discovery reset the protocol (legacy-equivalent epochs/batch restored; protocol-B reruns in flight).

USER INSTRUCTION (2026-07-27): update this existing deck with the last
report's narrative — augmentations + CKLA explanation + results (pending).
Existing Part-1 slides must NOT be rewritten (light touch only: a part
divider / agenda slide is allowed). Style per the standing deck rules:
figure- and formula-dominated, minimal anchoring text, speaker notes carry
the words.

## Sections (ordered; NEW sections only — insert after the existing "What this means"/"Not established" slides as Part 2, keep Backup last)
1. Part 2 divider — message: from SE baselines to the RPS-parity campaign: what we trained, what we measured, what broke — evidence: none (divider) — sources: —
2. The augmentation regime, before and after — message: mixture-level augs only for weeks; the 6-transform noise-level family failed BUNDLED (dilution ~11% per-transform strength); and the staging bug: flatten_channels made stage boundaries land at epoch ~80, so staged augs NEVER fired in any E12-family run (proof: bit-identical cross-policy runs; 8.0 frames/chunk measured) — evidence: policy YAML fragment figure, the dilution bar, the bit-identity curve overlay — sources: report §1, docs/experiments/ckla.md § THE STAGING BUG
3. Freq-scale vs collapse-to-the-mean — message: models respond 0.05% to a 2% frequency shift — predictions anchored to the label prior, not the comb; freq_scale (noise+RPS pair rescaled, labels ×α) is the one transform that makes spacing load-bearing — evidence: scale-response table (all three models ~1.00), freq-scaled spectrogram pair — sources: report §2, activation analysis §A6 + probes2 numbers
4. CKLA in one slide: the filter as a layer — message: information-form Kalman recursion (den/λ/η equations), complex ā = e^(−γ+iω_t), |ā|² keeps the precision algebra real, input-dependent rotation = closed-loop tracking — evidence: the three-line recursion + layer block diagram — sources: report §3, docs/ckla-design.md
5. What the trained head actually does — message: accumulator degeneration (gain → 1e-6) explains the FLY124-win/DREGON-loss split; the p_init fix restores gain (measured 3000×); rotation redundant when gain collapsed, +21% when gain alive (single seed, pre-reset protocol — pending confirmation) — evidence: λ/gain trajectory figure, rotation-attribution matrix — sources: report §4-5, activation analysis
6. Results — PENDING — message: the protocol reset (legacy-equivalent epochs/batch: 5000 chunks/epoch, 128-frame batches, stage 2 at true epoch 10) invalidates cross-era comparisons; what stands: P0 synthetic (7× convergence, locks where attention never does), mechanistic measurements; what's rerunning: pnoise/norot/g2_if/freqscale under protocol-B on kaggle — mark every number cell "updating" — evidence: the standing-vs-pending table with job handles — sources: docs/experiments/ckla.md § Results + § THE STAGING BUG
7. Stream sanity protocol — message: new gate: scripts/check_stream.py measures frame-expansion, effective stage boundaries in epochs, and EMPIRICAL fire rates (generate-with/without diffing) before any training submission — evidence: example check output table — sources: the check_stream worktree (if merged by build time; else describe the design)

## Cut
- Full activation-analysis six-question detail — report material, deck gets the two load-bearing figures only
- Edge-BS-RoFormer adaptation — not yet learning (val flat); mention only in the pending table
