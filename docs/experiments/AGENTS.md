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
| `kalman-harmonic-tracker-phase0.md` | — (standalone runner `src/experiments/kalman_harmonic/`, no conf/experiment entry) | RPS-driven Kalman harmonic tracker vs framed lstsq_VP; bet killed at K2 (drift robustness refuted); mitigation list for any learned revival |

Related: `docs/fwh_rotor_acoustic_simulator_plan.md` (a tool build with its
own implementation-status section, not a comparative experiment, so it stays
in `docs/` rather than here).
