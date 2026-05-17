# Rotor Speed Estimation from Drone Sound — paper draft

A short (4-page, IEEE conference-style) paper presenting initial results on
audio-only multi-rotor RPS estimation from noisy mixtures, using only locally
available results from this repository.

## What this paper says

- **Headline result.** A lightweight convolutional regressor predicts per-frame
  rotor speed for all four rotors of a DREGON quadcopter from a single-channel
  mixture (LibriSpeech + DREGON noise, SNR in `[-30, 0]` dB), reaching
  R² = 0.95, MAE = 0.56 rev/s, MSE = 5.15 (rev/s)² on the held-out validation
  set.
- **Comparison.** Larger complex-valued encoders adapted from DCUNet and DCCRN
  are *better* on the in-distribution subset (MSE 3.10 / 2.63 vs 6.84) but
  *worse* on out-of-distribution high-SNR free-flight recordings (factors of
  3–5 vs 1.2 for SimpleConv). The simpler model is more robust to distribution
  shift; the in-distribution ranking reverses out of distribution.
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
| `fig_highsnr_per_sample.pdf` | `../../results/rps_high_snr_analysis.json` (10 free-flight clips from `DREGON_free-flight_speech-high_room1`). |

Table II uses aggregate metrics from
`../../results/rps_eval_specific_samples/evaluation_results.json`.
Headline metrics (R² = 0.95, MAE = 0.56, MSE = 5.15) are the best validation
epoch from `training_log.csv`.

## Known caveats

These are stated in the paper but worth being explicit about for review:

1. **Single platform.** Training and evaluation are on a single quadcopter
   (DREGON drone). Cross-airframe generalisation is untested.
2. **Validation overlap.** DREGON-LM splits at the clip level, not the flight
   level, so train and val share underlying flights. The high-SNR free-flight
   eval is the cleanest evidence we have against trivial memorisation.
3. **Subset comparison.** Table II's three-way comparison uses 5 clips with
   matched input normalisation, because that is the only place we have
   apples-to-apples predictions stored locally for all three models. A
   full 600-clip three-way comparison is a natural follow-up.
4. **Outlier handling in the high-SNR analysis.** The 10th clip
   (`t ≈ 38.6 s`) is a drone-landing phase (RPS → 0) not represented in
   training data; we treat it explicitly as a distribution-shift failure mode
   and report the mean over the other 9 clips.
