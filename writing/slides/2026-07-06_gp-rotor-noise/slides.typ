#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [Training on generated noise (first try)],
  subtitle: [\+ Investigating GP rotor noise model from the Quiet Drones paper (Lee et al.)],
  author: [Dmitrii Mukhutdinov],
  date: [2026-07-06],
)

// ===========================================================================
// Part 1 — The negative result: generated-noise augmentation
// ===========================================================================

= The hypothesis

- When we use all the annotated data we have, even with online mixing and augmentation models start to overfit.
- Best models do temporal modeling, so apparently they memorize the few real RPS trajectories.
- So let's train on generated data.

#v(0.4em)
*What was done:*
+ Deep generator was trained on real data (DREGON + Michael's), with per-drone conditioning.
+ Live noise generator wired into the online-mix training stream: sampled RPS -> noise model -> audio + rps.

= The deep generator: architecture

#figure(
  image("assets/noise_gen_diagram_positional.png", width: 88%),
  caption: [`PositionalHarmonicNoiseGen`. A *per-drone conditioning code* $z$
    FiLM-conditions *one shared-weight single-rotor emitter*, applied to each
    rotor independently (rotor axis folded into the batch). The per-rotor sources
    are then *mixed by geometry* — $1/r$ attenuation + $r/c$ fractional delay,
    summed over rotors — to give the noise at a mic.],
)

= The deep generator: real vs. generated

#grid(
  columns: 1fr,
  rows: auto,
  gutter: 0.6em,
  figure(
    image("assets/noise_gen_spec_dregon.png", width: 55%),
    caption: [DREGON — low-BPF comb, weak mid-freq harmonics.],
  ),
  figure(
    image("assets/noise_gen_spec_michaels.png", width: 55%),
    caption: [Michael's — per-drone conditioning, fixed filter-envelope fingerprint.],
  ),
)

= Not good enough (actually makes things worse)

#grid(
  columns: (1.5fr, 1fr),
  gutter: 0.8em,
  figure(
    image("assets/e4_aug_degradation.png", width: 95%),
    caption: [+1 generated `michaels` source ($approx 1/3$ of noise batches)
      degrades PIT MSE +27% (uni-GRU) and +26% (Transformer).],
  ),
  [
    *Why:*
    - Imperfect mid-freq harmonics + fixed filter-envelope → the predictor
      latches onto the generator's *fingerprint* as a shortcut.
    - RPS-trajectory diversity too narrow.
    - The transformer memorises: train $9.9 arrow.r 3.7$ while val PIT
      $10.6 arrow.r 43.6$ after epoch 9.
    - *Not* a plumbing failure (42 tests pass).
  ],
)

= Loss curves: why it fails?

#figure(
  image("assets/e4_pit_curves.png", width: 92%),
  caption: [Train/val PIT MSE vs. epoch (wandb `rps-prediction`). Log axis; dashed
    line = no-generator baseline.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  [
    - *Transformer overfits* even with augmentation: train $9.9 arrow.r 3.7$ while
      val turns *up* after epoch 9 ($10.6 arrow.r 43.6$), across 3 seeds.
  ],
  [
    - *NaN instability:* 9/11 jobs died (divergence / OOM); plain simple-conv blows
      up by epoch 4. The 2 survivors both stay *above* baseline.
  ],
)

= Part I — take-away

- Generative noise model learns only Michael's drone setup reasonably well, on DREGON harmonics wash away
- That is probably because of wind noise in DREGON
- Adding generated noise to training mix with just single conditioning vector can still lead to overfitting - wow
- Need the noise to be way more diverse in texture
- But let's check out how others generate noise first, see pros and cons

// ===========================================================================
// Part 2 — The GP rotor-noise model
// ===========================================================================

= So how do other people generate rotor noise?

You pointed me to this at Quiet Drones:

#align(center)[
  *Lee, Ko, Seshadri & Rauleder.*
  *Bayesian ML framework for time-domain prediction of multirotor vehicle noise.*
  *JASA* 159(4), 3418–3435 (2026). DOI: 10.1121/10.0043469.
]

#v(0.3em)
- It's a Gaussian process that predicts the whole pressure waveform of a
  multirotor — even at flight conditions it never actually measured — and it
  hands you uncertainty too.
- The bit I like: the tonal part is *pinned to known physics* (the blade-pass
  frequencies). Nothing is left loose for a predictor to latch onto — the exact
  opposite of our deep generator's problem.

= How it works, roughly

- Split each frame of audio into two parts with a wavelet: the *tonal* hum and
  the *broadband* hiss.
- The tonal part is just sinusoids sitting *exactly* at the blade-pass
  frequencies — and physics gives us those straight from the RPS. So we never
  learn *where* the tones are, only *how loud* they are.
- Fit those loudnesses per frame with plain least-squares. Then a GP smooths
  them across microphones (and conditions).
- The hiss is just Gaussian noise with a level learned per mic.
- To make new audio: predict the loudnesses -> drop the sinusoids back at the
  known frequencies -> add the hiss.

#figure(
  image("assets/gp_overview.png", width: 92%),
)

= How our attempt differs from theirs

- *Generated audio vs real.* They train on *generated* audio — a physics
  simulator auralised over a clean grid of hundreds of virtual mics, millions of
  perfectly-labelled points. We train on the same tiny pile of *real* recordings
  we've had all along: 6 DREGON + 2 Michael's flights, 8 real mics, wind and all.
- *Four free motors vs a locked trim.* Their quadcopter flies *in trim*, so every
  rotor speed is pinned to one number (flight speed) — the tonal part collapses to
  just *two* combs (front + aft rotors), fixed ahead of time. Ours are *four*
  motors each doing their own thing, so four combs overlap and we
  least-squares-separate them, frame by frame.
- *One GP per drone.* Mic geometry differs between drones, so we fit a separate
  GP per drone — same idea as the per-drone code in the deep model.

= Does it actually work? (one recording so far)

#figure(
  image("assets/gp_faithful_spectrum.png", width: 52%),
)

- Tried it on one Michael recording as a first sanity check.
- The blade-pass peaks land *exactly* where they should (149/146/156/160 Hz) —
  no more washed-out tones.
- Coefficient fit is tight (RMSE 0.10). It reaches up to about 1.7 kHz with 24
  harmonics; above that (2–6 kHz) it under-covers. More harmonics would fix it,
  but RAM grows as $H^2$.

= What it costs, and what's next

- *Cheap.* About 46 min per drone on a laptop CPU, about 8.6 GB RAM, basically
  two knobs (harmonics $H$, inducing points). No GPU. And no artifacts to
  exploit — the tones are fixed physics.
- *Next:* train one GP per drone on the *full swapped split* — the exact same
  data the deep generator used — so we can compare like-for-like.

#v(0.4em)
*Results on the full dataset are pending.*
