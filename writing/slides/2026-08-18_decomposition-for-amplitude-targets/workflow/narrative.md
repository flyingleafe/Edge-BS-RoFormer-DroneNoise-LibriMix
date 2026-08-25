# Narrative — Why the generator needs a decomposition, and how we built one
kind: slides
audience: PhD supervisor; knows the RPS-conditioned generator program and the
July decks; has NOT seen the refined-label A/B, the decomposition campaign, or
the v4 model.
through-line: Refined labels made the generator slightly better, but the mid
and high harmonics stay washed out — and we can prove why: the harmonic lines
decohere inside the loss's own analysis windows, so no audio-domain spectral
loss can reward a sharp high harmonic. The fix is to train on per-harmonic
amplitudes instead of synthesized audio — which requires a decomposition of
real recordings. We built that decomposition as MAP estimation under one
Gaussian noise model (v4: smooth floor + Lorentzian comb, marginal Whittle
objective). Per-harmonic powers are parameters of that model — exactly the
training targets we set out to get.

## HARD STYLE CONSTRAINTS (from the user — these override all defaults)

1. All slide prose MUST follow the ASD-STE100 Simplified Technical English
   standard: sentences of 20 words or fewer, active voice, simple present /
   simple past / imperative only, no jargon metaphors. FORBIDDEN words include
   "load-bearing", "unglamorous", "honest", "story", and every figurative
   idiom. Write plain technical statements.
2. Few words. Formulas and figures carry each slide; text supports them.
   Bullets of at most ~8 words. No paragraph text on slides.
3. Use the full slide area. Prefer two-column layouts (figure + formula,
   figure + table). Avoid large white space.
4. Dense but short: target 14-16 content slides.

## Sections (ordered; ≈1-2 slides each)

1. Recap + this week's question — refined telemetry labels were the last
   label-side fix; did they save the audio-domain generator? Source:
   docs/experiments/generator-refined-labels.md.
2. Refined-label generators: better, not sufficient — refined labels put the
   learned lines ON the comb; harmonics above k≈25 still not learned.
   Evidence: A/B per-band comb readout table/figure.
   Sources: generator-refined-labels.md (§ verdict + CORRECTION).
3. Cause: phase decoherence defeats the loss — shaft wander ≈0.6 rev/s gives
   ≈0.24·k rad of phase drift inside one 2048-sample loss window; log-L1 fits
   the fluctuating band magnitude's low median → permanent downward gradient
   on steady lines above k≈25. Data-side check: order-averaged tooth contrast
   along refined tracks = 6.76 / 1.36 / 0.13 / 0.01 dB by band.
   Sources: docs/experiments/amplitude-target-training.md ("The problem this
   objective removes"), generator-perrotor-dynamics.md finding 6. Figure: the
   0.6k linewidth law.
4. The fix: fit amplitudes, not waveforms — train on per-(rotor, harmonic)
   amplitude envelopes + broadband floor. The training path exists and ran
   end-to-end on v1/v2 targets (plumbing validation: perrotor 0.818 vs
   codebook 0.856 val). Missing piece: targets from a decomposition we trust.
   Sources: amplitude-target-training.md, conf/experiment/gen_a*_amp*.
5. The requirement — decompose real free-flight audio into comb + broadband
   exactly (channels sum back sample for sample), per harmonic, on both rigs.
   One spectrogram of the difficulty: interleaved twin combs + strong wash.
   Assets: results/vk_decompose_v3e originals.
6. The noise model (v4) — one Gaussian model: PSD M = S + Σ H·L; S = smooth
   floor, H = per-line power, L = Lorentzian at k·r_i(t) with half-width
   0.6k Hz. Phase noise lives in rate increments (integrated OU): lines
   decohere in phase, stay anchored in frequency. Show the model equation and
   a floor-plus-bumps sketch. Sources: docs/v4-unified-model-design.md.
7. Objective part 1 — the Whittle cost — each spectrogram cell costs
   P/S + log S. Figure: the U-curve (cost vs claimed S; minimum at S = P) and
   the two-bar example (line cell priced as floor: cost 10 + log S0; priced
   as line: 3.3 + log S0). Source for figure design: the "tariff" figure of
   the Whittle explainer (regenerate in matplotlib for slides; do NOT use the
   word "tariff" on the slide — call it "the cost of one cell").
8. Objective part 2 — Lorentzian width + marginalization — exponential phase
   forgetting in time = Lorentzian width in frequency (measured 0.6k Hz law,
   three-bump figure at k=5/20/40). Envelopes are Gaussian processes;
   averaging them out gives J = Σ [P/M + log M], M = S + Σ H·L. H sits inside
   log M, so claimed line power pays the same cost as the floor. No envelope
   variable remains in J.
9. Minimization — block-coordinate loop diagram: (1) fit (S, H) by penalized
   Whittle + nonnegative line fit; (2) envelope posterior by one banded solve
   (Wiener shrinkage); (3) trajectory corrections from phase increments of
   coherent harmonics, coarse-to-fine in k. Output channels: comb = posterior
   reconstruction; broadband = recording − comb (exact).
10. RESULTS: the decomposition (PUNCHLINE, 2-3 slides) — original / comb /
    broadband spectrogram panels for DREGON, FLY124, FLY125 at 0-8 kHz;
    per-band retained-excess table (DREGON 6.9/8.4/15.4/7.6 %; FLY124
    0.5/6.3/36.6/47.2 %; FLY125 0.3/5.0/36.3/52.8 % — quote the ≤0.09 dB
    absolute contrast beside the high-k FLY numbers). REGENERATE figures for
    16:9 slides from results/vk_decompose_v3e/<rid>/residual.npz (+ original
    audio via the loader; see scratchpad split_demo3 assets script as the
    recipe: /tmp/claude-1000/-home-flyingleafe-Research-PhD-projects-harmonic-noise-suppression/5a88d51c-adfa-4ffa-951d-2f560860cb3c/scratchpad/split_demo3/assets_recordings_v3e.py).
    LABELING: these panels are the shipped v3e decomposition. If files named
    results/vk_decompose_v4/... exist by build time, PREFER them and label
    "v4"; otherwise label "current (v3e); v4 refit running". Leave one
    placeholder slide "v4 first light" with a clearly marked TODO if v4
    results are absent.
11. The objective is also a label-quality measure — the converged J ranks
    trajectory hypotheses. Fresh result: refined labels rank above raw
    telemetry on 3 of 5 frozen test windows; never below except one 0.009
    margin. Table from results/joint_rescore_refall (pulled at
    omnirun-outputs/jr-refall-1485fb/results/joint_rescore_refall/summary.json).
    Message: this is the tool for blind annotation of unlabeled corpora.
12. Next steps — v4 validation gates; retrain the generator on decomposition
    amplitude targets; blind pseudo-labels for unlabeled corpora.

## Cut (considered, excluded)
- v1→v3 decomposition iterations; seam/carve autopsies (present v4 directly).
- The adversarial-fan measure saga (the positive refined-ranking carries it).
- Per-rotor attribution, blind-corpus campaign, paper status (one line max).
