# docs/experiments/ — Past-Experiment Log

One markdown file per completed (or in-progress) experiment: **Motivation /
Results / Conclusion**, plus a status/dates header. This replaced a scattered
"zoo" of `.pi/checkpoints/*.md` session handoffs, raw `autoresearch/*.md`
sweep logs, and duplicate legacy `writing/papers/*/report.md` drafts
(consolidated 2026-07-03; originals deleted, content preserved in git
history and in these docs).

This is a **history/reference log**. Its `bets/` subdirectory (`docs/experiments/bets/`)
holds GOALS.md's live, currently-active portfolio bets — one detail card per
bet, created when a bet starts (none started as of this writing). A concluded
bet's card stays under `bets/`; its write-up (motivation/results/conclusion)
lands as a top-level file here.

## Convention

- One file per experiment cluster, kebab-case filename.
- Header line: `**Status:** done | in progress | blocked` + date range +
  (if one exists) a link to the full `writing/reports/<date>_<slug>/`
  Typst report (`make` in that dir builds the PDF).
- If a full Typst report already exists, keep this doc **short** — a
  pointer/summary, not a re-publication (see `channel-generalization-pit-loss.md`).
  If no polished report exists, this doc **is** the full record (see
  `dregon-lm-v2-v3-baseline.md`).
- Only state numbers/claims present in source material — no invented
  figures, no rounding away uncertainty the source didn't have.

## Two tiers

- **Batch docs** (the files in this directory) — one motivation/results/conclusion
  narrative per *group* of related experiments.
- **Per-experiment docs** — one `conf/experiment/<name>.md` **beside every**
  `conf/experiment/<name>.yaml`, with YAML frontmatter linking the config to its
  batch doc (`batch:`) plus a `## Motivation` / `## Conclusion` body. Enforced by
  `scripts/validate_experiment_docs.py` (a pre-commit hook): every experiment
  config must have a valid sibling doc pointing at an existing batch doc here.

## Contents (batch docs)

| File | Experiments (conf/experiment prefix) | Experiment batch |
|---|---|---|
| `paper1-edge-bs-roformer-dn-lm.md` | `a1_*` | Paper-1 Edge-BS-RoFormer + SE baselines on DN-LM |
| `diffusion-buffer-se.md` | `a2_*` | Diffusion-Buffer (BBED) SE baseline |
| `rps-conditioned-se-dregon.md` | `b1_*` | Paper-2 oracle-RPS-conditioned DCUNet/DCCRN on DREGON-LM |
| `refactored-decoder-rps-fusion.md` | `b2_*` | Refactored decoder-side RPS fusion + auxiliary RPS head |
| `simpleconv-rps-architecture-search.md` | `c1_c3_c6_*`, `c10_*`, `rps_simple_conv_v2_v4` | SimpleConv encoder/temporal-head sweeps (offline + online-mixed) |
| `dregon-lm-v2-v3-baseline.md` | `c4_*`, `c5_*` | DREGON-LM V2→V3 dataset evolution + baseline RPS training (superseded by V4) |
| `channel-generalization-pit-loss.md` | `c9_*` | RPS models don't generalize across mic channels without PIT loss |
| `cross-drone-generalization-fly125.md` | `c11_*` | Adding Michael's FLY125 closes the cross-drone RPS gap on FLY124 |
| `salience-map-rps-tracking.md` | `c7_*`, `c8_*` | Multi-F0 salience-map tracking as an alternative to direct RPS regression |
| `noise-generation-augmentation.md` | `e2_*`, `e3_*`, `e4_*` | Learned harmonic noise generator as an online-mixing augmentation source |
| `generator-label-sensitivity.md` | `p7_labelsens_*` | Does the tachometer staircase (0.269 rev/s / 49.7 Hz) explain the generator's high-harmonic underfit? Synthetic A/S/B/C/B0 on exact vs biased vs staircased vs presmoothed labels, per-k line readout. No: the 0.542 % constant bias carries the effect (-8.8 dB at k50-80); the staircase alone costs 0.45 dB |
| `vk-frontend-probe.md` | — (driver `scripts/vk_frontend_probe.py` **deleted** in the 2026-08 R2 consolidation; the document is the record) | Coupled VK envelopes vs independent demodulation as the tracker's front end (issue #15): rejected |
| `telemetry-fitness.md` | — (standalone driver `scripts/telemetry_fitness.py`, no conf/experiment entry) | The goodness-of-fit HARNESS for a candidate rotor-speed trajectory (issue #17 phases 6a-6e): FOUR components at fixed degrees of freedom, held-out harmonics/channels/time, all four §B controls, residual decomposition + bootstrap. Design, synthetic acceptance tests, the 6c campaign, and the 6d sensitivity fix — **read § "Phase 6d" first if you are citing a number**: the first three components are shares of one in-band power and saturate, the ridge component is the one that reads a line against a local floor. § "Phase 6e" adds the TIME axis: DREGON's telemetry runs early by -42 ms [-85,-31] (the proposed tachometer lag is excluded by sign), the per-microphone differential is ~500x below the ridge's resolution, and the propagation delay itself is confirmed on the michaels rig at slope 1.013 by a cross-channel phase estimator |
| `kalman-harmonic-tracker-phase0.md` | — (standalone runner `src/experiments/kalman_harmonic/`, no conf/experiment entry) | RPS-driven Kalman harmonic tracker vs framed lstsq_VP; bet killed at K2 (drift robustness refuted); mitigation list for any learned revival |
| `blind-corpus-annotation.md` | — (driver `scripts/blind_corpus.py` on branch `blind-corpus`, no conf/experiment entry) | Blind rotor-speed annotation of corpora that have NO telemetry, and the label-free instruments that judge one. The calibrated gate is two numbers from two calibration sets — `fvk_ratio_double` >= 1.065 (DREGON room2, separates correct from halved/ramp 17/17) and a NEGATIVE half-margin (DREGON motor bench, separates doubled from correct 46/46 by sign) — plus the per-rotor octave test that restores the undiluted per-track reading on a quadrotor. Read § "Full single-motor bench" for the throttle ladder that validates the corrected annotation without any label, and § "The corpus plan" for what is annotatable and in what order |

Related: `docs/fwh_rotor_acoustic_simulator_plan.md` (a tool build with its
own implementation-status section, not a comparative experiment, so it stays
in `docs/` rather than here).
