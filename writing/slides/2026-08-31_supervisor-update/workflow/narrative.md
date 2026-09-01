# Narrative — supervisor update, 2026-08-31
kind: slides
audience: the supervisor. Knows the wrap-up paper's thread (scarce annotations,
frequency-scaling probe, generator, baselines). Has NOT seen: the stochastic-comb
family results, the scaling ablations read as a structural verdict, or any of the
gather/slot work.
through-line: the regressors do not read harmonic positions; augmentation makes
them react to a GLOBAL frequency shift but no more, and only synthetic comb data
makes them track harmonics reliably. Every synthetic family we then built exposes
the same degenerate solution — predict the mean, splay a memorized fan — and
scaling does not touch it, because it is the model structure and the loss, not
capacity. A hypothesis-scoring architecture with no trained parameters solves the
synthetic case outright; on real data it is not there yet, and three multi-pitch
architectures from the literature are being ported to test whether the idea
generalizes.

**NARRATIVE FIXED BY THE USER** — nine slides, specified verbatim in the request.
Do not re-order, re-scope or add slides. Every slide except the title is figures
and/or tables; prose only as a one-line takeaway per slide.

## Sections (ordered)

1. **Augmentation makes them react; only synthetic data makes them track** —
   evidence: `assets/freq_probe_nophase.pdf` (ALREADY GENERATED — the paper's
   Fig. 3 with the green "phase-increment readout" curve removed, as requested).
   Slopes near alpha=1: no label-transforming augmentation **0.16**, augmentation
   only (R2) **0.14**, augmentation + comb pre-training (R4) **1.02**; ideal 1.00.
   Full-range slopes 0.03 / 0.89 / 1.04 — the R2 curve reacts globally (0.89) yet
   is locally flat (0.14), which IS the point of the slide.
   sources: `writing/papers/2026-08_wrapup/plot_freq_probe.py`, draft.md Sec. 6.

2. **What we had before stochastic combs** — two columns.
   LEFT: three spectrograms over the SAME RPS trajectory — real drone noise,
   neural-generated noise, static comb. NEEDS GENERATING.
   RIGHT: the curriculum/mixing table + how bad synthetic-only validation was.
   | arrangement | all-MAE | val PIT-MSE |
   | comb stage 1 -> real stage 2 (R4) | 2.67 | 17.59 |
   | stochastic stage 1 -> real stage 2 (R6) | 3.04 | 23.78 |
   | one stage, stochastic pooled (R7) | 5.99 | 94.97 |
   | real stage 2 + 29.4% stochastic, warm (R8) | 5.81 | 104.85 |
   | one stage, generator pooled (R5) | — | 147.6 |
   Plus from draft.md Sec. 8: naive addition of generated data made validation
   PIT-MSE **27% worse**; generated-only training reached 17.8-25.4 PIT-MSE
   against ~7.3 for real-data training, and a short real fine-tune 11.1-14.1.
   sources: `docs/experiments/stochastic-transfer.md` (~line 1956),
   `writing/papers/2026-08_wrapup/draft.md` Sec. 8.

3. **The stochastic comb family** — the generative formula plus samples.
   Formula (from `src/data_processing/stochastic_rotor_noise.py` docstring):
   S(f,t) = B(f,t) + sum_r sum_k P_rk(t) * L(f - k*rps_r(t); gamma_rk),
   L(d;g) = (1/pi) * g / (d^2 + g^2)   [Cauchy], gamma_rk = gamma0_r + slope_r*k,
   10log10 P_rk(t) = harm_mean_db + profile_db[r,k] + h_rk(t), h_rk a
   squared-exponential Gaussian process; the floor B is a smooth random curve in
   log frequency with its own GP. NEEDS: several sample spectrograms over the
   SAME RPS with different random amplitude profiles, next to real noise.

4. **Results after stochastic-comb training, split by regime** — one table.
   | model | trained on | all-MAE | zero | low | flight |
   | `r4hb_scv2` | real | **2.67** | **2.87** | **3.48** | **2.49** |
   | `stoch_s1g_scv2` | synthetic | 8.08 | 20.27 | 16.20 | 4.50 |
   | `m3abl_comb_unigru128_s1` | synthetic | 8.30 | 4.73 | 24.24 | 6.00 |
   | `stoch_s1h_scv2` | synthetic | 9.07 | 27.98 | 26.77 | **2.60** |
   Best synthetic-only is 3.03x the real target overall: 7.06x on stopped rotors,
   4.66x on ramps, but only 1.81x at cruise.
   source: `docs/experiments/stochastic-transfer.md` (~line 24).

5. **Synthetic-only, trained to convergence** — table.
   | run | epochs | val RMSE min | at epoch |
   | `stoch_long_scv2` (1.5M) | 229 | 10.304 | 202 |
   | `stoch_long_trxxl` (38M) | 157 | 13.207 | 31 |
   **CAVEAT THAT MUST APPEAR ON THE SLIDE:** these are HALF-REAL, HALF-SYNTHETIC
   validation numbers and are not comparable to synthetic-only or real-only
   figures elsewhere in the deck. Mark any multif0-salience-on-synthetic row
   `[TODO verify]` — those runs are still in flight.
   source: `docs/experiments/synthetic-solvability-limits.md` (~line 497).

6. **Every model outputs an evenly-spaced fan around the mean** — output plots.
   Best synthetic-trained model on synthetic data, and best real+aug model on real
   data, predicted vs true trajectories. NEEDS GENERATING; `writing/papers/
   2026-08_wrapup/make_figures.py` renders exactly this kind of panel and
   `writing/papers/2026-08_wrapup/figures/qual_*.pdf` are existing examples —
   reuse that script rather than writing a new one.
   The measured statement: true spread varies over **42.6 rev/s**, predicted
   spread over **1.85**; the model splays four rotors by ~9 rev/s when they turn
   in unison and by only 10.8 when they are 42.75 apart, settling on ~9.4 rev/s —
   the generator's own mean spread. It learned the marginal distribution of the
   quantity, not the signal that determines it.
   source: `docs/experiments/synthetic-solvability-limits.md` (~line 230-256).

7. **Scaling does not touch it** — two columns.
   LEFT: results. Width scaling HURTS: 38M `trxxl` is 28% worse than the 1.5M
   trunk (13.207 vs 10.304) and climbs away from its epoch-31 minimum. The
   comb-floor study found the same axis harmful on a different family.
   RIGHT: schematic of how the transformers were scaled (wider vs more layers).
   NEEDS DRAWING — a simple two-panel schematic is fine.
   Quote for the takeaway: "The model is not capacity-limited; it has found a
   degenerate solution the loss rewards."
   source: `docs/experiments/synthetic-solvability-limits.md` (~line 250, 511).

8. **The gather + salience + slot method** — two columns.
   LEFT: architecture scheme — STFT power -> local floor -> GATHER at k*r over a
   candidate-rate grid -> score head -> salience -> CRF best path, with the slot
   loop (four slots divide the bins) underneath. Only the head has parameters
   (137); the rest is fixed algorithm.
   RIGHT: the numbers.
   Synthetic static comb, geomean of per-cell PIT-RMSE: previous peel **0.772**,
   this family untrained **0.487**, trained **0.432**.
   Synthetic stochastic comb, geomean: regressors trained on that family
   **5.962** (coherent) / **5.765** (Rayleigh); this family untrained **2.800** /
   **3.834**.
   REAL data, PIT-MAE on the 15-window beat-VK protocol: blind Viterbi+pi_kalman
   bars **1.825** (dregon_cruise) / **3.992** (fly124_cruise); ours **8.881** /
   **5.971**. STATE PLAINLY that neither real bar is cleared.
   sources: `docs/experiments/comb-slot-crf.md`, `comb-salience-family.md`.

9. **Adapting three multi-pitch architectures — running** — scheme + parallels.
   HarmoF0's `MRDConv` and HPPNet's `HarmonicDilatedConv` both gather harmonic
   evidence as a SHIFT by log2(k)*B bins on a log-frequency axis; hFT-Transformer
   cross-attends from one output token per note to frequency tokens. The port
   replaces the shift with an explicit gather at k*r on the LINEAR STFT, because
   under a log grid a rotor pair's separation-to-bandwidth ratio is
   D/(r*(2^(1/B)-1)) — k cancels, so a pair is resolved at every harmonic or none.
   Mark all results `[TODO verify]` — experiments are in flight.
   source: `docs/harmonic-ports-design.md` (read it; it has the tables).

## Cut (considered, excluded)
- The blind-tracker / classical baseline comparison — not in the user's nine.
- The OTMP optimal-transport baseline — not in the user's nine.
- The CRF loss derivation — slide 8 shows the scheme, not the maths.

## Open questions for the user
- None; the narrative was specified verbatim.
