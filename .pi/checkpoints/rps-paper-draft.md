# RPS Prediction Paper Draft

**Goal:** Submit a short (4-page IEEE conference style) paper "Rotor Speed Estimation from Drone Sound" by **2026-05-22** (6 days).
**Status:** in-progress
**Last touched:** 2026-05-16
**Resume on:** any

## Done

- Title/abstract finalised after two rounds of supervisor feedback. Final version pasted into `papers/rps-from-drone-sound/main.tex` and matches the version sent to the supervisor with the level-matched-baseline clarification added to the OOD claim.
- Paper directory created: `papers/rps-from-drone-sound/`.
  - `main.tex` — 4-page IEEEtran conference, ~287 lines. Builds clean (no overfull boxes; only harmless `caption` package warning).
  - `refs.bib` — 10 entries (DREGON, LibriSpeech, DCUNet, DCCRN, Liu 2025, Gulli 2025, YIN, CREPE, Al-Emadi UAV detection, AdamW unused but kept).
  - `make_figures.py` — regenerates all three figures from local results in `../../results/`.
  - `Makefile` — `make pdf` / `make clean` / `make veryclean`.
  - `README.md` — paper scope, build steps, data provenance per figure, known caveats.
  - `figures/fig_training_curves.pdf` — SimpleConv train/val MSE + R² vs epoch.
  - `figures/fig_qualitative_combined.pdf` — 3 samples × (spectrogram + GT/pred RPS overlay).
  - `figures/fig_highsnr_per_sample.pdf` — per-clip MSE bars for 10 OOD free-flight clips, 3 models, with landing outlier annotated.
- `papers/AGENTS.md` — directory conventions (one paper per subdir, no fresh experiments at build time, toolchain from flake, honesty over polish).
- `flake.nix` modified — added `texlive.combine` extending `scheme-medium` with `ieeetran`, `biblatex`, `biber`, `cm-super`, `cmap`, `latexmk`. Verified `IEEEtran.cls` and `biber` resolve.
- PDF compiled to 4 pages, 696 KB: `papers/rps-from-drone-sound/main.pdf`.

## Pending

(Ordered by likely review feedback risk → reward.)

1. **Get supervisor review of the current draft.** Send `papers/rps-from-drone-sound/main.pdf` and ask for feedback specifically on: (a) is the "passive acoustic monitoring" framing strong enough or should we still angle this toward SE? (b) does Table II's 5-clip subset feel underpowered? (c) are the OOD claims solid?
2. **Address: full 600-clip three-way comparison.** Currently Table II uses a 5-clip subset because that's the only place we have apples-to-apples normalised predictions for all three models locally. The full-validation eval of DCUNet-enc and DCCRN-enc has *not been run* with consistent input normalisation. Decide whether to (a) run it (needs `datasets/DREGON-LM/valid/` which is **not on disk** — currently only `datasets/DREGON-LM-test/` exists; would need `dvc pull` or rebuild), or (b) keep the subset and defend it.
3. **Decide on author block.** Currently anonymised. Real names + affiliations need to go in before submission.
4. **Pick a venue.** Candidates the abstract is compatible with: Interspeech short paper, WASPAA, ICASSP-SP-L, Drones journal (open access), arXiv-first if no conference fits the timeline. Check Interspeech 2026 / WASPAA 2026 deadlines vs 22 May.
5. **Polish pass:** scan for `et~al.` rendering issues (verified clean), check Strauss DREGON citation key vs DBLP, confirm Liu/Gulli have correct first authors (currently placeholders: `Liu, Yifan and others`, `Gulli, Andrea and others`).
6. **Optional:** add a 1-figure architecture diagram (TikZ) if reviewers find SimpleConv hard to picture from Table I. Not strictly needed.
7. **Optional:** if supervisor pushes back on encoder comparison, fall back plan is to make Table II SimpleConv-only and move the encoder comparison entirely into the OOD section.

## State

- Working tree (paper-related only):
  - `M flake.nix` — texlive addition, not yet committed
  - `?? papers/` — entire directory untracked
- Auxiliary LaTeX files in `papers/rps-from-drone-sound/` (`*.aux`, `*.bbl`, `*.log`, `*.fls`, `*.fdb_latexmk`, `*.out`, `*.blg`) are present but should be gitignored, not committed.
- `papers/rps-from-drone-sound/.venv/` is a stray uv venv from the `nix develop` shell hook — should also be gitignored.
- Numbers used in `main.tex` (all verified against local JSONs, do not relitigate):
  - Headline (full validation, SimpleConv): R² = 0.949, MAE = 0.56 rev/s, MSE = 5.15 (rev/s)². Source: `results/rps_predictor/rps_predictor/training_log.csv` best epoch.
  - 5-clip subset (matched normalisation, peak 0.9): SimpleConv MSE 6.84, DCUNet 3.10, DCCRN 2.63. Source: `results/rps_eval_specific_samples/evaluation_results.json`.
  - High-SNR free-flight (9 of 10 clips, excluding landing): SimpleConv 7.91, DCUNet 16.44, DCCRN 10.01 (means in (rev/s)²). Source: `results/rps_high_snr_analysis.json`. Landing outlier is sample at t=38.6s where RPS→0.
  - Naive baseline MSE ≈ 365 (rev/s)² from training log epoch 1.
- No data was generated in this session; only existing JSONs/CSVs and saved `.npy` predictions were used.

## Decisions (do not relitigate)

- **Framing:** drone-acoustics / passive monitoring paper, **not** an SE paper. SE is one downstream use among several. This was the supervisor's pushback in this session — locked.
- **Title:** "Rotor Speed Estimation from Drone Sound" (supervisor's suggestion). Locked.
- **Abstract:** finalised in this session after the supervisor's three points (add method description; spell out metrics; drop ego-noise framing). Locked unless reviewer feedback says otherwise.
- **No `\textit` for paper "case study" framing.** This is its own paper, not a chapter; we don't need to keep "rotating machinery" language. Locked for this paper only; thesis framing (C1 in GOALS.md) is separate.
- **Toolchain:** `texlive.combine` from the project flake. **Do not** add a per-paper TeX install or `tlmgr` usage.
- **Table II uses the 5-clip subset honestly.** Caveat stated in-text and in README. Do not silently expand or mask the subset size.
- **Outlier handling:** landing clip (t=38.6s) is reported and explained as distribution shift, not excluded silently.

## Open questions

- **First-author names** for `liu2025bandsplit` (`Liu, Yifan` is a placeholder) and `gulli2025rps` (`Gulli, Andrea` is a placeholder). Need real names before submission.
- **Venue + deadline.** Interspeech 2026 / WASPAA 2026 / Drones journal — user has not chosen. The 22 May deadline implied a specific submission; confirm which.
- **Author block.** Real authors + affiliation to replace "Anonymous Authors / Affiliation withheld for review".
- **Do we want a `.gitignore` in `papers/rps-from-drone-sound/`** for `*.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out main.pdf .venv/`? Or commit `main.pdf` for review convenience?

## Resume

```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
nix develop

# Rebuild figures + PDF
cd papers/rps-from-drone-sound
make pdf   # equivalent to: python ../../make_figures.py && latexmk -pdf main.tex

# Inspect
xdg-open main.pdf   # or any PDF viewer

# Most likely next steps:
#  - Open main.tex, address supervisor's review comments (see Pending #1).
#  - If running the full-validation 3-way comparison (Pending #2):
#      cd ../..
#      ls datasets/DREGON-LM/valid 2>/dev/null  # check if val set is on disk
#      # If not:  dvc pull   (or rebuild via create_dregon_librimix.py)
#      python evaluate_rps_predictor_samples.py --output_dir results/rps_eval_full \
#             --all_samples   # may need a flag tweak; not run in this session
#  - If picking a venue, check 2026 deadlines for: Interspeech, WASPAA, Drones (journal).
```

If reviewer comments are minimal, the path to submission is: address comments → fill in author block → run `latexmk` once more → submit `main.pdf` + source.
