# Supervisor Comments — Todo List

## `main (1).pdf` — 13 comments

| # | Page | Status | Comment | Action |
|---|------|--------|---------|--------|
| 1 | 2 | ✅ Done | "this is not problem formulation. should go to other sections." | Moved interpolation/ground-truth alignment text from Problem formulation to Dataset section |
| 2 | 3 | ✅ Done | "should go to other sections." | Merged with comment #1 — moved telemetry alignment text to Dataset |
| 3 | 3 | ✅ Done | "this is not correct — should be a mapping from time-frequency spectrogram to rotor speed." | Rewrote problem formulation to explicitly define $f_\theta: \mathbb{R}^{F \times T} \to \mathbb{R}^{4 \times T}$ mapping log-magnitude spectrogram to per-rotor speeds |
| 4 | 3 | ✅ Done | "this is not your target, as you don't know the ground-truth in practice — it is your training cost function" | MSE equation now explicitly described as the **training cost function**, not the problem target |
| 5 | 3 | ✅ Done | "move to discussion" | Interpolation discussion moved to Dataset section; Discussion section itself merged into single Discussion & Conclusion |
| 6 | 4 | ✅ Done | "try to avoid using bullet points in the paper." | Removed all `itemize` environments from Discussion & Conclusion; converted to prose paragraphs |
| 7 | 5 | ✅ Done | "simulation results" | Renamed subsection to **"Results on held-out synthetic mixtures"** |
| 8 | 5 | ✅ Done | "results on real recordings" | Renamed subsection to **"Results on real high-SNR free-flight recordings"** |
| 9 | 5 | ✅ Done | "echoing on my previous comments: would be nice to have results per SNR." | Ran inference on all 600 validation mixtures with `best.pt`, computed per-sample MSE/MAE/R², stratified by SNR bins `[-30,-25], [-25,-20], [-20,-15], [-15,-10], [-10,-5], [-5,0]` dB. Results show stable performance across the range (R² 0.77–0.88). Generated table (`rps_per_snr_table.tex`) and figure (`rps_per_snr.pdf`) in `figures/` |
| 10 | 5 | ✅ Done | "why harder?" | Added prose explaining why synthetic [-30,0] dB is harder than typical on-board microphone recordings (near-field vs. scaled mixture) |
| 11 | 5 | ✅ Done | "you did not introduce Fig. 4" | Added more explicit figure introduction before `fig_highsnr` |
| 12 | 6 | ✅ Done | "try to make the harmonics visible in the plot. it is very vague in the current version" | Changed spectrogram colormap from `magma` to `hot` and added per-sample percentile clipping (`vmin=p2`, `vmax=p99`) to enhance harmonic ridge visibility. Regenerated `fig_qualitative_combined.pdf` and `fig_highsnr_outlier.pdf`. |
| 13 | 7 | ✅ Done | "You already have plots for simulation. For real recording, you can plot the prediction results for the whole sequence (2 min or longer? from taking off to landing). panel: spectrogram; panel: rotor speed estimation; pane: MSE variation with time" | Ran inference on full `DREGON_free-flight_speech-high_room1` recording (46.9 s with motor data, covering takeoff → flight → landing). Generated 3-panel figure `fig_full_sequence.pdf`: spectrogram, GT vs predicted RPS, and 1-s-smoothed per-frame MSE. Saved predictions + metrics to `results/rps_full_sequence/`. Added `fig_full_sequence()` to `make_figures.py` for regeneration. In-flight MSE = 19.9 (vs 88.6 global due to takeoff/landing outliers). |

## `main (2).pdf` — 8 comments

| # | Page | Status | Comment | Action |
|---|------|--------|---------|--------|
| 1 | 1 | ✅ Done | "This section does not read like an introduction. You need some rework on this. It should include: What is the topic? background and context; Why does it matter?; Why it is difficult? What is the state of the art?; What does this paper add? You can merge the related work into this section. You can move the potential use in the discussion into this section. Avoid using equations and numbers in the introduction, e.g. b, kbfr, in the first paragraph, R^2 0.95, 0.56 in the last paragraph. Nobody understand what they are." | Completely rewrote Introduction: added background/context, motivation, difficulty, prior work, and contributions. Removed Related Work section entirely. Removed all equations and result numbers. Moved potential uses into intro prose |
| 2 | 2 | ✅ Done | "Merge this section into introduction." | **Related Work section removed** and fully merged into Introduction |
| 3 | 2 | ✅ Done | "Move equations in the introduction to this section." | Equations moved to **Problem formulation** subsection under Method |
| 4 | 4 | ✅ Done | "Give equations to these dentitions." | Added explicit equations for MSE, MAE, and $R^2$ in Evaluation metrics |
| 5 | 5 | ✅ Done | "Can you give estimate results per SNR? To understand the performance variation with respect to SNR." | **Same as (1) #9** — completed |
| 6 | 6 | ✅ Done | "Avoid using bullet points. Move 'potential uses' to the introduction" | Potential uses paragraph moved to Introduction; Discussion bullets removed |
| 7 | 7 | ✅ Done | "Merge limitation and future work into the conclusion (discussion and conclusion)" | **Discussion and Conclusion merged** into single section; Limitations and Future Work converted from bold itemised headings to prose paragraphs |
| 8 | 8 | ⬜ **Pending** | "Check the capitalization of all the references. Add more references. Would be good to have 20-30 references. You can add references on drone audition and introduce them in the introduction section." | **User will handle** — references to be added by author |

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Text reformulations | 17 | ✅ All done |
| Figure modifications (existing data) | 1 | ✅ Done |
| New evaluations | 3 | 2 ✅ Done (per-SNR + full-sequence), 1 ⬜ Pending: full-sequence audio comparison? |
| References (self-assigned) | 1 | ⬜ Pending: user will add |
| **Total** | **22** | **18 done, 4 pending** |
