# Rotor Speed Estimation from Drone Sound — paper draft

A short (4-page, IEEE conference-style) paper presenting initial results on
audio-only multi-rotor RPS estimation from noisy mixtures, using only locally
available results from this repository.

## What this paper says

- **Headline result.** A lightweight convolutional regressor predicts per-frame
  rotor speed for all four rotors of a DREGON quadcopter from a single-channel
  mixture (LibriSpeech + DREGON noise, SNR in `[-30, 0]` dB), reaching
  R² = 0.84, MAE = 0.56 rev/s, MSE = 5.15 (rev/s)² on the held-out validation
  set.
- **Framing.** This is a passive acoustic monitoring paper, *not* a
  speech-enhancement paper. Downstream uses (flight-state inference,
  payload/fault diagnostics, source separation, SE conditioning) are listed
  but not evaluated.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Paper source (IEEEtran conference format). |
| `refs.bib` | Bibliography. |
| `make_figures.py` | Regenerates all figures from local results in `../../results/`. |
| `figures/` | Compiled PDF figures referenced by `main.tex`. |
| `Makefile` | `make pdf` rebuilds figures and PDF; `make clean` / `make veryclean`. |
| `main.pdf` | Compiled output (4 pages). |

## How to build

From the project root, enter the dev shell once:

```bash
nix develop
```

The flake provides `pdflatex`, `latexmk`, `biber`, and IEEEtran (a custom
`texlive.combine` extending `scheme-medium`; see `../../flake.nix`).

Then either run `latexmk` directly:

```bash
cd papers/rps-from-drone-sound
latexmk -pdf main.tex
```

or use the Makefile (which also regenerates figures from local result JSON
and `.npy` files first):

```bash
cd papers/rps-from-drone-sound
make pdf
```

## Data sources

Everything the figures depend on is already in this repository; no remote
fetch is needed:

| Figure | Source |
|--------|--------|
| `fig_training_curves.pdf` | `../../results/rps_predictor/rps_predictor/training_log.csv` (SimpleConv standalone training). |
| `fig_qualitative_combined.pdf` | `../../results/rps_eval_specific_samples/sample_{00000,00149,00599}/` (mixture WAVs and `*_rps.npy` predictions). |
| `fig_full_sequence.pdf` | `../../results/rps_full_sequence/` (full-sequence predictions on `DREGON_free-flight_speech-high_room1`). |

Headline metrics (R² = 0.84, MAE = 0.56, MSE = 5.15) are the best validation
epoch from `training_log.csv` (the per-sample full-set R² is 0.835; the CSV
records batch-level R² which peaks at 0.949).

## Known caveats

These are stated in the paper but worth being explicit about for review:

1. **Single platform.** Training and evaluation are on a single quadcopter
   (DREGON drone). Cross-airframe generalisation is untested.
2. **Validation overlap.** DREGON-LM splits at the clip level, not the flight
   level, so train and val share underlying flights. The high-SNR free-flight
   eval on the full recording is the cleanest evidence we have against trivial
   memorisation.
3. **Takeoff and landing in the high-SNR recording.** The free-flight recording
   includes takeoff and landing phases (rotor speeds below $50~\mathrm{rev/s}$)
   not represented in DREGON-LM training data. These are highlighted as grey
   bands in Fig.~\ref{fig:fullsequence} and dominate the global MSE; the
   in-flight MSE ($19.9~(\mathrm{rev/s})^2$) is a fairer comparison with the
   held-out synthetic value ($5.15~(\mathrm{rev/s})^2$).
