# physical-modeling-report

Educational LaTeX subproject: *Blade Element Momentum Theory (BEMT) vs Vortex Lattice Method (VLM)* for drone rotor acoustic simulation.

## What it is

A self-contained LaTeX report explaining BEMT and VLM from first principles,
with validated plots and a benchmark table derived from real DREGON motor recordings.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | LaTeX document (IEEE QD 2026 class) |
| `make_figures.py` | Regenerates all PDF figures from `results/spectrum_comparison.json` |
| `compute_spectrum_comparison.py` | Computes spectrum distances and timing benchmarks |
| `figs/` | PDF figures for the paper |
| `results/` | `spectrum_comparison.json` with numeric results |
| `refs.bib` | Bibliography (DREGON, aerodynamics, FWH papers) |

## Build

```bash
# Inside nix develop (provides pdflatex, biber, IEEEtran)
cd papers/physical-modeling-report
make figures  # regenerate figures from JSON
make pdf      # build main.pdf
```

## Source data

- **Simulated spectra** — computed live by `compute_spectrum_comparison.py` using
  `fwh_rotor_sim/` modules
- **Real recordings** — `../../data/DREGON/DREGON_individual_motors_recordings/`
- **Mic positions** — `../../data/DREGON/micPos.txt`

## Main findings

| Metric | BEMT | VLM |
|--------|------|-----|
| Spectral MSE to real | ~41 dB | ~42 dB (1.4 dB worse) |
| Pearson r (shape match) | 0.752 | 0.752 |
| Speed (single GPU) | 3360 ms/call | 515 ms/call (6.7× faster) |
| End-to-end differentiability | partial | full |

**Decision for project:** VLM is preferred (6.7× faster, full differentiability).
BEMT is retained as a reference model for validation.

## Status

- [x] `compute_spectrum_comparison.py` — all 12 motor/RPM combos + benchmark
- [x] `make_figures.py` — 5 figures generated from results JSON
- [x] `main.tex` — full skeleton with theory sections and figures
- [x] Result figures: spectra grid, MSE bar, correlation bar, speed bar
- [x] Architecture diagrams (fig\_arch\_compact, fig\_arch\_detailed)
- [ ] Build with `make pdf` (needs nix develop + QD_2026.cls template)
