# Joint Decomposition of Quadcopter Ego-Noise

The v4 restructure of the rotor-speed paper. The claim: one joint
harmonic-plus-broadband model gives (1) the decomposition, (2) trajectory
refinement, and (3) a principled measure of trajectory fit — one solver,
three readouts. The 2026-07 draft (`../2026-07_coupled-vk-blind-rps/`)
is demoted to the search section and the capture-range ablation of this
paper; it stays untouched as the source of record for those parts.

Status: **structure-first draft** — full prose, WIP results. Every
unmeasured number sits behind `\pending{...}` and every passage that
waits for an experiment behind `\wip{...}`. Grep for both before any
submission step. The style guide for the prose is the author's Stage 2
report (`~/Research/PhD/Stage2_report_Dmitrii_Mukhutdinov.pdf`).

Gate before the results land (see `docs/experiments/vk-decomposition.md`
and memory `vk-paper-v4-structure`): the decomposition must pass the
excess-retained target on every recording with telemetry, and the
J-rescoring test must rank telemetry above the coverage-fan solutions
(`results/fvk_arms/raw/s5__*`).

## Build (Tectonic v2)

    tectonic -X build      # output: build/main/main.pdf
    tectonic -X watch      # rebuild on change

Layout follows the 2026-07 paper: `Tectonic.toml` at the paper root,
the whole document is `src/index.tex` (self-contained), bibliography at
`src/references.bib` (copied from the 2026-07 paper; entries marked
`\pending{cite: ...}` in the text still need to be added), figures under
`src/figures/`. Outputs land in `build/main/`.
