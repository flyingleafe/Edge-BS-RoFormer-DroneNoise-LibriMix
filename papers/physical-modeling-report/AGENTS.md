# physical-modeling-report/

Educational LaTeX report: *Blade Element Momentum Theory (BEMT) vs Vortex Lattice Method (VLM)*
for drone rotor acoustic simulation.

## What it is

A self-contained LaTeX report (IEEE QD 2026 class) explaining BEMT and VLM from first
principles, with validated plots and a benchmark table derived from real DREGON motor recordings.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | LaTeX document |
| `make_figures.py` | Regenerate all PDF figures from `results/spectrum_comparison.json` |
| `compute_spectrum_comparison.py` | Tasks 1+2: spectrum distances + speed benchmark |
| `figures/` | PDF figures for the paper |
| `results/spectrum_comparison.json` | Numeric results |
| `refs.bib` | Bibliography (DREGON, aerodynamics, FWH papers) |
| `Makefile` | Build: `make figures`, `make pdf` |

## Build

```bash
# Inside nix develop (provides pdflatex, biber, IEEEtran class)
cd papers/physical-modeling-report
make figures   # regenerate figures from results JSON
make pdf       # build main.pdf
```

## Main findings

### Spectral accuracy
- **VLM and BEMT produce near-identical waveforms** (Pearson r = 0.9996, RMS diff < 0.04 dB).
  This is not a coincidence: VLM's Kutta–Joukowski Γ = L′/(ρU) recovers exactly the same
  lift as thin-airfoil BEMT for the same angle of attack.
  The only algorithmic difference is the retarded-time solver (joint vs strip-by-strip).
- **Neither model captures broadband noise** — both are inviscid potential flow.
  The ~41 dB MSE gap vs real recordings comes from motor/ESC noise and turbulence,
  not from model differences.
- **An initial result claiming "VLM is 1.4 dB worse" was a resampling artefact.**
  With n_t=200 (fs=40 Hz, Nyquist=20 Hz << BPF=180 Hz), harmonics were completely aliased.
  With n_t=2000 (Nyquist > BPF), the VLM–BEMT MSE collapses to 0.04 dB².

### Speed (single GPU)
| Method | Time (ms/call) | Rate | Ratio |
|--------|----------------|------|-------|
| BEMT | 3361 ms | 0.30 calls/s | 1.00× |
| VLM | 515 ms | 1.94 calls/s | **6.7× faster** |

### Differentiability
- **VLM wins**: full autograd chain (Biot–Savart = matmul + cross + norm, all native autograd).
- BEMT: partial autograd through thin-airfoil polars.

### Decision
**VLM is preferred** for this project (6.7× faster, full differentiability, same spectral accuracy).
BEMT is retained as a fast prototyping tool and validation baseline.

## Data sources

- **Simulated spectra** — computed by `compute_spectrum_comparison.py` using `fwh_rotor_sim/`
- **Real recordings** — `../../data/DREGON/DREGON_individual_motors_recordings/`
- **Mic positions** — `../../data/DREGON/micPos.txt`

## Status

- [x] `compute_spectrum_comparison.py` — 12 motor/RPM combos + benchmark
- [x] `make_figures.py` — 5 figures generated from results JSON
- [x] `main.tex` — full skeleton with theory sections and corrected results
- [x] Architecture diagrams (fig_arch_compact, fig_arch_detailed)
- [ ] Build with `make pdf` (needs nix develop + QD_2026.cls template)
