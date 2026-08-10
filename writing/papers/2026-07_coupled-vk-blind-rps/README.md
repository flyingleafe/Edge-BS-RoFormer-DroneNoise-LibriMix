# Coupled VK + Blind RPS

Short paper on blind tracking of four rotor rates from quadrotor
ego-noise with a capture-range-explicit Vold–Kalman pipeline
(IEEEtran, ~8 pp).

## Build (Tectonic v2)

Tectonic comes from the project flake (pinned at 0.15.0):

    tectonic -X build      # output: build/main/main.pdf
    tectonic -X watch      # rebuild on change

## Layout — read this before moving anything

- `Tectonic.toml` at the paper root declares the document as
  `src/index.tex`, self-contained (`\documentclass` … `\end{document}`).
- The engine's filesystem root is `src/`: **every input the document
  reads must live under `src/`** — the bibliography
  (`src/references.bib`), figures (`src/figures/` — note
  `\graphicspath{{figures/}}` resolves there), any `\input`s. Files at
  the paper root are invisible to the engine and to bibtex.
- Outputs land in `build/main/` (pdf plus transient aux/bbl/log).
- Gotcha: the `inputs = "..."` key in `[doc]` exists only in Tectonic
  ≥ 0.16; 0.15.0 rejects it (`unknown field 'inputs'`) and hardcodes the
  `_preamble.tex` / `index.tex` / `_postamble.tex` triple under `src/`.
  Since `index.tex` is the whole document, `Tectonic.toml` sets
  `preamble = ""` / `postamble = ""` to skip the wrappers.
