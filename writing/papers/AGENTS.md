# papers/

Self-contained paper drafts. Each subdirectory is one paper.

| Directory | Status | What it is |
|-----------|--------|------------|
| `rps-from-drone-sound/` | draft (May 2026) | Short paper on audio-only rotor speed estimation. Uses only locally-stored results; builds with `latexmk` via the project flake. |
| `2026-07_coupled-vk-blind-rps/` | superseded (Aug 2026) | Blind multi-rotor VK tracking paper. Demoted: its method sections survive as the search section + capture-range ablation of the 2026-08 paper. Builds with **Tectonic v2** (see its `README.md`). |
| `2026-08_joint-decomposition-rps/` | draft, structure-first (Aug 2026) | The v4 restructure: joint harmonic+broadband decomposition with a built-in measure of fit — one solver, three readouts (decomposition / refinement / measure). Full prose, WIP results behind `\pending{}`/`\wip{}`. Tectonic v2, `src/index.tex`, output `build/main/main.pdf`. |
| `2026-08_wrapup/` | draft, prose final v0.2 (Aug 2026) — **git submodule** (`github.com/flyingleafe/drone-rps-wrapup-paper`, private; run `git submodule update --init` after clone) | The wrap-up paper: audio-only motor-speed estimation with scarce annotations — the frequency-scaling probe, label-transforming augmentations, the structured noise generator as training data, and adjacent-task baselines (multi-pitch + tacholess order tracking incl. our two-stage blind method). Narrative fixed by the author 2026-08-20; pending results behind `\wip{}`. Tectonic v2; `draft.md` is the frozen markdown source of record, `inventory.md` the claim→experiment map. |

## Conventions

- **One paper per subdirectory.** Each has `main.tex`, `refs.bib`, `Makefile`,
  `make_figures.py`, `figures/`, and a `README.md` describing scope, build
  steps, and data provenance.
- **No fresh experiments at paper-build time.** Figures and tables are
  regenerated from existing results in `../results/` (and other already-
  produced artifacts). If a paper needs a new number, run the experiment
  separately (`python train.py experiment=<name>` locally, or via
  `omnirun submit ... -- python train.py experiment=<name>` on a remote GPU)
  and let the result land in `../results/` before regenerating figures.
- **LaTeX toolchain comes from the flake.** `nix develop` provides
  `pdflatex`, `latexmk`, `biber`, and IEEEtran via a custom `texlive.combine`
  on top of `scheme-medium`. Do not introduce a TeX-from-pip or per-paper
  toolchain.
- **Honesty over polish.** Where a number is from a 5-clip subset, say so.
  Where a ratio is computed against a level-matched baseline rather than the
  headline number, say that in-text. The supervisor reviews drafts; quiet
  glossing over caveats wastes a review cycle.
