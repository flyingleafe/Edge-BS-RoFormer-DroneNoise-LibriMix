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

## Files

- `src/index.tex` — the paper. **All future edits happen here.**
- `draft.md` — the markdown source of record, frozen at v0.2. It is the
  origin of the prose in `index.tex`, and it is not maintained after this
  point.
- `inventory.md` — the claim-to-experiment mapping that Appendix B points to.
