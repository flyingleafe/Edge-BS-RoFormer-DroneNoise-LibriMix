# Estimating Rotor Speeds from Drone Ego-Noise with Scarce Annotated Data

## Scope

The wrap-up paper of the rotor-speed line: audio-only estimation of the four
motor speeds of a quadrotor, under an annotated corpus of about one hour
across two drone models. It carries four threads — the frequency-scaling
probe that shows what the neural regressors actually read, the
label-transforming augmentations (time-warping and frequency scaling) that
raise their use of the frequency cue, a structured neural noise generator
used as synthetic training material, and baselines borrowed from two adjacent
tasks (multi-pitch tracking and tacholess order tracking, the latter
including our own two-stage blind method). The narrative was fixed by the
author on 2026-08-20 and is final; from now on this directory holds the paper
itself, not the plan for it.

## Build (Tectonic v2)

    tectonic -X build      # output: build/main/main.pdf
    tectonic -X watch      # rebuild on change

Layout follows `../2026-08_joint-decomposition-rps/`: `Tectonic.toml` at the
paper root, the whole document in the self-contained `src/index.tex`,
bibliography at `src/references.bib` (copied from that paper — no entry was
added), figures under `src/figures/`. Outputs land in `build/main/`.
`tectonic` comes from the project flake; if the shell does not have it, use
`nix develop -c tectonic -X build`.

The two bibliography lines at the end of `src/index.tex` are commented out.
Every citation is still a `\pending{cite: ...}` marker, so the document has
no `\cite` command, and BibTeX would emit an empty bibliography — a LaTeX
error. Uncomment the two lines when the first real citation lands.

## Status

**Structure-first draft.** The prose is final at v0.2 and was typeset
verbatim; the results are partially pending. Two macros mark the gaps, both
copied from the sibling paper:

- `\pending{...}` — a citation that has no bibkey yet
- `\wip{...}` — a queued experiment (PENDING), a planned baseline (TO RUN), a
  figure to prepare (FIGURE), or a detail to verify (TODO)

Grep for `\wip` and `\pending` before any submission step. Appendix B is
internal bookkeeping and carries its own note to strip it before submission.

## Figures

`make_figures.py` renders the qualitative per-regime comparison figures — for
one validation clip, a 0-4 kHz spectrogram on top, then one panel for each
method with the four predicted rotor tracks (solid) over the truth (dotted).
Regenerate all three with:

    python writing/papers/2026-08_wrapup/make_figures.py --out-dir src/figures

The default `--out-dir` is `figures/` (which `.gitignore` covers for PDFs);
pass `src/figures` to write where the document reads. Each clip gives
`qual_<regime>.pdf`, a `.png` for quick viewing, and a `.json` with the clip
PIT MAE (per-frame Hungarian) of each method, overall and for each target
regime — the numbers for the captions.

Clips come from `dload:DREGON-LM-V4-michaels-valid-full`, channel 0. The three
defaults were selected by their target-regime frame counts, of 251 frames each:

| Regime | Clip | zero / low / flight |
|--------|------|---------------------|
| `zero` | 36 | 251 / 0 / 0 |
| `transition` | 8 | 87 / 59 / 105 |
| `cruise` | 20 | 0 / 0 / 251 |

The script prints these counts and an OK/MISMATCH verdict on every run.
`--clip <regime>:<index>` overrides them. `--method "<label>=<source>"`
overrides the method list; a source is `zoo:<experiment>` (a checkpoint
through `zoo.load`), `classical:<name>` (a key of
`experiments.classical_rps.predictors.CLASSICAL_TRACKERS`), or
`npz:<path>[#<key>]` (a precomputed `(4, T)` trajectory on the 2048/512 grid —
the route for the blind tracker and the order-tracking rows).

## Files

- `src/index.tex` — the paper. **All future edits happen here.**
- `make_figures.py` — the qualitative-figure driver (see above).
- `draft.md` — the markdown source of record, frozen at v0.2. It is the
  origin of the prose in `index.tex`, and it is not maintained after this
  point.
- `inventory.md` — the claim-to-experiment mapping that Appendix B points to.
