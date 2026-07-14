#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [Synthetic data for RPS prediction: status update],
  subtitle: [Noise generator improvements, drone-texture interpolation, time-warp, and where synthetic data still falls short],
  author: [Dmitrii Mukhutdinov],
  date: [2026-07-13],
)

= Where we left off (July 6)

- Rotor-speed (RPS) prediction from drone audio: predict all 4 rotor speeds from the mic signal, scored by permutation-invariant MSE (PIT-MSE).
- We have a deep noise generator that can turn any RPS trajectory into synthetic multi-mic drone noise — unlimited training data, in principle.
- First attempt at using it as training augmentation *made things worse* (+27% PIT-MSE): the generator's clean, telemetry-exact combs were off-distribution vs. real recordings, and predictors latched onto that as a shortcut.
- This fortnight: make the generator (and the recipe around it) realistic enough that synthetic data actually helps.

= Generator fix 1: harmonic linewidth

#figure(
  table(
    columns: 5, align: (left, right, right, right, right), stroke: 0.5pt,
    table.header([Generator], [$k<10$], [$k$ 10--25], [$k$ 25--40], [msSTFT]),
    [clean oscillator (baseline)], [9.00], [7.31], [7.82], [5.67],
    [with RPS-jitter injection], [*7.04*], [*6.40*], [*6.36*], [*4.85*],
  ),
  caption: [Comb-masked mean $|Delta "log-mag"|$ (dB) vs. real DREGON recordings, lower is better. Real rotors jitter; injecting that jitter into the generator (per-drone $sigma$ learned: $approx$0.63 DREGON, $approx$0.61 Michael's) closes only *part* of the mid-frequency gap (9.00 → 7.04 for $k<10$; still 6.3--6.4 dB of gap left at $k$ 10--40).],
)

- Real rotor harmonics have finite width because the true RPS jitters fast (label-invisible); a telemetry-conditioned oscillator bank previously rendered perfectly clean tones instead, and the loss suppressed those as "wrong".
- Fix: perturb the generator's conditioning fundamental with the same jitter statistics (OU process, calibrated per drone) — reduces the gap, doesn't close it; see next slide for exactly where the dB improvement lands.

= Generator fix 1, in detail: linewidth, per rotor (1)

#figure(
  image("assets/jitter_decompose.png", height: 75%),
  caption: [Same DREGON code and RPS trajectory, per rotor. Row 2: full spectrum, jitter off (blue) vs. on (red) vs. real (grey) vs. the broadband-residual floor (orange dashed) — rotors 1/3 (facing away from this mic) show the comb barely clearing the broadband floor, even with jitter on. Row 3: zoomed on one harmonic — jitter fills in the deep inter-harmonic notches. Row 4: where the loss actually wins — per-(freq, time) $|Delta|_"off" - |Delta|_"on"$ vs. real (3-frame-smoothed; red = jitter has smaller error there). Note row 2's grey "real" curve sits *above* jitter-on almost everywhere below 1 kHz — jitter narrows the gap, it does not close it.],
)

= Generator fix 1, in detail: linewidth, per rotor (2)

- Two separate effects, same fix: (a) linewidth — jitter smears each harmonic from a delta into a finite-width bump (row 3); (b) level — even broadened, some rotors' harmonics still barely poke out of the broadband floor (row 2, rotors 1/3), matching what real recordings look like.
- Row 4 shows the win is diffuse, not a clean "harmonics only" story: red/blue speckle everywhere, with a mild low-frequency red band (jitter helps the notch-filling most where notches are deepest) and no single dominant band that "explains" the whole 9.00→7.04 dB gain. The mid/high-$k$ gap (6.3--6.4 dB) is still open — jitter is a partial fix, not the missing texture.

= Does it actually look right? Real vs. generated

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  figure(
    image("assets/real_vs_gen_dregon.png", width: 100%),
    caption: [DREGON: real recording vs. generator output, same STFT settings, shared dB normalization per pair.],
  ),
  figure(
    image("assets/real_vs_gen_michaels.png", width: 100%),
    caption: [Michael's M100: same comparison.],
  ),
)

- The harmonic ladder matches in both cases, but overall brightness/energy balance still differs for DREGON (generated is globally dimmer/quieter, under the shared normalization above) and the real ~6 kHz whine is missing; Michael's matches noticeably better on both counts.
- Remaining texture gap is what the linewidth-jitter fix (previous slide) is chipping away at.

= Generator fixes 2--3: silence + full flight

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  figure(
    image("assets/silence_fade.png", width: 100%),
    caption: [A stopped rotor is now exactly silent. Before: a numerical DC pedestal meant "silence" never really meant zero RPS.],
  ),
  figure(
    image("assets/fullflight.png", width: 100%),
    caption: [Synthetic RPS trajectories now cover a whole flight (ground → warm-up → take-off → cruise → landing), not just cruise.],
  ),
)

- Loudness now scales physically with speed (pressure $prop "rps"^2.5$), so amplitude and frequency tell the same story across the whole flight.
- Purpose: cover regimes real cruise recordings never show.

= What makes interpolation work: two regularizers

- The generator only ever sees 2 conditioning codes at train time (DREGON, Michael's) — nothing constrains its behaviour *between* them. Two training-time additions fix that:

$ z' = z + epsilon dot (sigma_z dot "RMS"(z)), quad epsilon tilde cal(N)(0, I), quad sigma_z = 0.1 $

- *Vicinal z-noise*: perturb each sample's drone code by 10% of its own RMS magnitude (`z_noise_std=0.1`), train-only, RMS detached (no gradient through the noise scale). Forces the decoder to behave sensibly in a small ball around each of the 2 codes, not just exactly at them.

$ (gamma, beta) = "SpectralNorm"(W_"FiLM") dot z + b_"FiLM" $

- *Spectral-normed FiLM*: the linear map from code $z$ to the FiLM $(gamma,beta)$ that modulates the generator's backbone is Lipschitz-bounded (`film_spectral_norm=true`, power-iteration spectral norm on $W_"FiLM"$). Caps how fast the generator's output can change as $z$ moves — the smoothness precondition for the walk on the next 2 slides to look like anything physical.
- Free lunch: this combination (`jitter_latreg`) didn't cost anything on the real-data comb-fidelity numbers from the previous slide (7.04/6.40/6.36/4.85, same as jitter alone) — it's a training-time-only addition to the loss landscape, not a capacity trade.

= Interpolating drone textures (1/2)

#figure(
  image("assets/interp_strip_dregon_to_michaels.png", height: 58%),
  caption: [Same real RPS trajectory (DREGON), generator embedding code walked from the DREGON code ($alpha=0$) to Michael's ($alpha=1$), bracketed by the two REAL recordings (red border) it's interpolating between, all panels sharing one dB normalization. The texture drifts smoothly — no jumps, no artifacts. The generated $alpha=1$ endpoint matches real Michael's noticeably better than the generated $alpha=0$ endpoint matches real DREGON.],
)

- The per-drone conditioning code lives in a continuous embedding space; the generator was never trained on anything but the two endpoints ($alpha in {0,1}$).
- Interpolating between them gives physically smooth, novel "in-between drones".

= Interpolating drone textures (2/2)

#figure(
  image("assets/interp_strip_michaels_to_dregon.png", height: 58%),
  caption: [Same walk, other direction: Michael's RPS trajectory, embedding code walked from Michael's ($alpha=1$) back to DREGON ($alpha=0$), again bracketed by both real recordings, shared dB normalization throughout. Again, the Michael's endpoint ($alpha=1$) matches its real bracket better than the DREGON endpoint ($alpha=0$) matches real DREGON.],
)

- We exploit this directly: training draws a *random* $alpha$ each batch (plus a little embedding noise), so predictors see a continuum of drone timbres, not two fixed points — this "vicinal" sampling is how the sim-trained curriculum later gets its diversity.

= Time-warp augmentation

#figure(
  image("assets/timewarp_before_after.png", width: 50%),
  caption: [One real DREGON clip, resampled by a slowly-varying rate $alpha(t) = c + a sin(2 pi f t + phi.alt)$ ($|alpha - 1| <= 0.12$) and relabelled consistently. Same real spectral content, a different, valid speed profile.],
)

- Cheap way to multiply real trajectories, without inventing spectral texture (unlike generated noise). Validation PIT-MSE (cruise-only validation split): transformer 11.76 → *8.74*, SimpleConv-v2 (scv2) 9.71 → *8.85*, uni-GRU-128 tie.
- Helps most where overfitting was worst (the highest-capacity model, which had the worst baseline).
- But *not the best real-data result on this split*: the architecture sweep's online uni-GRU-128 (no time-warp, just wider online mixing) already scored *7.33* PIT-MSE on the same cruise validation set — the number to actually beat.

= But: still not better than the best real-data models

- Best known real-data number on the (old) cruise-only validation set: *7.33* PIT-MSE, online uni-GRU-128 from the architecture sweep — not the time-warp transformer's 8.74 quoted above. Every number below names its validation set; the cruise-only split and the full-envelope split used from here on are *not* interchangeable.
- All of this — jitter, full-flight coverage, silence, interpolation, time-warp — is real, measurable progress on the generator and the recipe.
- None of it has yet beaten training on real data alone, on a uniform evaluation.
- So we stopped asking "does synthetic data help on average" and asked *where, specifically, do predictors fail* — broken down by flight regime.

= What is the "analytic static comb" (E8)?

#figure(
  image("assets/static_comb_vs_generator.png", width: 72%),
  caption: [Static comb: crisp harmonic ridges at $k dot "rps"$, everything else fixed. Neural generator: realistic, textured spectrum.],
)

- *What it is*: no neural network. Fixed harmonics at $k dot "rps"(t)$ per rotor, amplitude $prop "rps"^2.5$ only (stopped rotor = silent), fixed broadband floor underneath.
- *Why*: predictors trained on the neural generator learn to read *loudness*, not *harmonic position* — a shortcut that fails on real audio. The comb removes that shortcut: amplitude carries no RPS info, only ridge position does.
- Used as *half* of E9's sim-pretrain mix (50/50 with the neural generator) — see next slide.

= Training data: what's actually in each recipe

- Every recipe mixes noise + LibriSpeech speech (`train-clean-100`) online, SNR $tilde U(-30, 0)$ dB, per-channel independent; augmentation (50% of samples: random gain $plus.minus$6 dB, polarity flip, or channel drop) turns on after the first 50k samples.
- *real-only (cruise)*: noise = real DREGON in-flight + Michael's FLY125, filtered to $"rps" > 30$ rev/s (drops warm-up/take-off/landing entirely).
- *real full-flight*: same real sources, filter removed ($"rps" > 0$) — keeps the real warm-up/take-off ramp Michael's/DREGON recordings actually contain.
- *sim full-flight curriculum*, 2 stages:
  + *Pretrain*: 50% neural generator + 50% analytic static-comb, both driven by *synthetic* full-flight RPS trajectories (ground→warm-up→take-off→cruise→landing); generator drone code $z(alpha)=(1-alpha) z_"dregon" + alpha z_"michaels" + cal(N)(0, 0.15 dot ||z_1-z_0||)$, $alpha tilde U(0,1)$ — a novel "in-between" drone every batch (the vicinal sampling from the previous slides, used for real training here).
  + *Fine-tune*: warm-started from the pretrained checkpoint, then trained on the *real full-flight* recipe above (time-warp augmentation included, $|alpha-1| <= 0.12$, `dev_const=0.08, dev_sine=0.04`).
- All 3 recipes validate on the same fixed real full-envelope split — never seen in training by any of them.

= Per-regime evaluation: setup

- Full-flight validation set (Michael's M100): *27 cruise / 6 warm-up / 4 ground* clips — same clips, same scoring, for every recipe below.
- Three training recipes compared, × 3 model sizes (uni-GRU-128, SimpleConv-v2, Transformer):

#figure(
  table(
    columns: 3, align: (left, left, left), stroke: 0.5pt,
    table.header([Recipe], [Training data], [Low-RPS coverage]),
    [real-only (cruise)], [real recordings, 30 rev/s floor], [none (dropped)],
    [sim full-flight curriculum], [synthetic full-flight, then real fine-tune], [full (synthetic ramp)],
    [real full-flight], [real recordings, no floor], [full (real ramp)],
  ),
)

- Metric: PIT-MSE per regime; RMSE $= sqrt("PIT-MSE")$ quoted in rev/s where noted.

= Per-regime results: where predictors fail

#text(size: 0.8em)[#include "assets/regime_mae_table.typ"]

#text(size: 0.85em)[
- Cruise: everyone fine (MAE 2--5 rev/s). Ground/warm-up: cruise-only baseline far worse (up to 34 rev/s) — never saw a slow rotor.
- Best: *real full-flight Transformer* (PIT-MSE 79.6, a *quarter* of cruise-only's 338.4). Sim curriculum beats cruise-only but real ramp still wins (real fine-tune is what makes it work).
- *Not the old 7.33 PIT-MSE headline*: different, easier validation set (cruise-filtered `rps>30`). This table's `-valid-full` has no filter — different test sets, not a regression.
]

= What the predictions actually look like: cruise

#figure(
  image("assets/regime_overlay_cruise.png", width: 56%),
  caption: [One cruise clip, all 3 conditions (transformer). Solid = prediction (PIT-aligned), dotted = ground truth, one line per rotor.],
)

- Everyone tracks cruise well — this is the easy regime, and it dominates the validation set (216 of 296 windows) — except the sim curriculum still shows brief dropout transients (down to $approx$58 rev/s in the first second), consistent with its 2x MAE here (5.2 vs. 2.5--2.8 rev/s).

= What the predictions actually look like: warm-up

#figure(
  image("assets/regime_overlay_warmup.png", width: 56%),
  caption: [One warm-up clip — ground truth here is a near-constant idle ($approx$30--41 rev/s), not a ramp; all 3 conditions.],
)

- The cruise-only baseline reads it with a modest offset (MAE 10.0); the sim curriculum badly under-reads this idle speed (predicts $approx$15--18 rev/s, MAE 20.2 — twice the baseline); real full-flight tracks it best (MAE 7.7).

= What the predictions actually look like: ground

#figure(
  image("assets/regime_overlay_ground.png", width: 46%),
  caption: [One ground clip (rotors near-stopped), all 3 conditions.],
)

- The baseline overshoots badly (it never saw a near-zero rotor in training); both full-flight recipes come much closer to the true near-zero speed, though neither is perfect.

= What the predictions actually look like: full flight

#figure(
  image("assets/regime_overlay_fullflight.png", height: 55%),
  caption: [Whole validation set walked in time order (37 windows, ground→warm-up→cruise→ground, repeated across concatenated recordings), all 3 conditions (transformer).],
)

- Ground truth (dotted) really does span the whole envelope — the flat-zero GT in an earlier draft was a bug (channel duplicates of one window grouped as if consecutive time steps; fixed).
- Same story as the single-regime slides: cruise is easy for all three, the sim curriculum is twitchier on ramps, ground dips are the hardest.

= Sim curriculum predictions are twitchier, not just wronger

- Looking across cruise/warm-up/ground (3 slides back): the sim curriculum's prediction is visibly *twitchier* than the other two everywhere, not just where it's wrong — small fast up-down wiggles riding on top of the right overall trend.
- That's not noise in the estimate: it's the signature of the *training* data. `generate_full_flight`'s synthetic RPS trajectories carry fast synthetic dynamics that real recordings don't; the sim-pretrained model learned to expect (and reproduce) that texture, then only partially unlearns it during real fine-tuning.

= Mean-tracking sanity check

#figure(
  image("assets/tracking.png", width: 50%),
  caption: [Mean predicted vs. mean true rotor speed, per regime (Transformer). All three recipes track the true speed — none has collapsed to guessing the global average ($approx 49$ rev/s).],
)

- Worth checking explicitly, since the validation set is mostly cruise: a model could "cheat" by predicting the average speed and ignoring regime. Not what's happening — cruise reads $approx 79$ against a true 78.9 for every recipe; models differ only at the low-speed end.

= Conclusion

- We have not yet found a synthetic-data training recipe that beats the best real-data-trained result.
- Be honest about what "tracks well" means here: *nobody* tracks RPS to high precision — the best cruise MAE is 2.5--2.8 rev/s (real full-flight / baseline transformer), not sub-rev/s. The sim curriculum additionally inherits its training data's fast synthetic twitchiness as a visible artifact in its predictions (previous slide).
- What synthetic data still uniquely provides: true-silence / zero-RPS coverage that no real recording gives us cleanly (even the best real model over-reads a stopped drone by 10--15 rev/s).
- Next: fold time-warp + the jitter-broadened generator together; per-drone jitter calibration for out-of-train drones; smoother (less twitchy) synthetic RPS trajectories before trying generated-noise augmentation again.

= Bonus: RPS label refinement -- the idea

#figure(
  image("assets/method_displacement.png", height: 52%),
  caption: [A labelling error of $delta$ rev/s displaces harmonic $k$ by $k dot delta$ Hz — small at $k=1$, huge by $k=40$. High harmonics punish a wrong label, which is what makes them useful for correcting one.],
)

- Motivation: refine RPS labels from audio alone → better ground truth, and automatic annotation of drone-only recordings with no label at all.
- Method: treat the label as a guess, track each rotor's harmonic comb in the spectrogram, nudge the guess onto the bright ridges (synthetic test: mean error 0.46 → 0.03 rev/s).

= Bonus: two ways to read the spectrogram

#figure(
  image("assets/method_comb_alignment.png", width: 72%),
  caption: [Before (left) vs. after (right) refinement, one rotor's high harmonics: dashed guess track moves off the valleys and onto the ridges.],
)

- *Stages B+C (magnitude)*: fit the comb to spectrogram *brightness* -- grid-search a coarse correction, then a smooth per-rotor spline, both by maximizing average log-magnitude along the harmonic comb. Cheap, but only sees "is there energy here", not whose.
- *Stage D (phase)*: fit the comb *coherently* -- solve for each harmonic's complex amplitude and read the residual phase drift, which pins down each rotor individually even when two rotors' magnitude ridges overlap.

= Bonus: validating against a hidden truth

#figure(
  image("assets/basin.png", width: 62%),
  caption: [Refinement recovers the true speed anywhere *inside* the initial search range, but never outside it -- a sanity check that the method interpolates, it does not hallucinate.],
)

- DREGON logs both the *commanded* speed (what we normally train on) and a *measured* one, hidden from refinement. Natural validation: start from command, see if refinement moves toward measured.
- Protocol: run stages B+C and D from the same command-track initialization, score all three (command / refined / measured) against the hidden measured track on 5 real flights -- see next slide for what happened.

= Bonus: RPS label refinement -- status

#figure(
  image("assets/val_overlay.png", width: 72%),
  caption: [Validating against DREGON's hidden `measured` track. Left: one rotor, 12 s. Right: per-rotor signed bias. The magnitude stage (red) is systematically biased low; only the phase-based stage (blue) tracks truth as well as telemetry (gray).],
)

- Honest negative-ish result: telemetry labels are already nearly unbiased (error 0.633 rev/s, bias $-0.057$) — little room to gain. On tightly-paired twin rotors the amplitude-based stages are actively biased ($approx -0.44$): low/mid harmonics of the pair merge into one ridge.
- Only the phase-based stage is trustworthy so far — still worth pursuing for the annotation use-case.
