#import "@preview/touying:0.7.4": *
#import "/writing/templates/typst/slides.typ": hns-slides

#show: hns-slides.with(
  title: [From Overfitting to Synthetic Drone Noise],
  subtitle: [RPS prediction → drone-noise generation → realistic RPS trajectories],
  author: [Dmitrii Mukhutdinov],
  date: [2026-06-30],
)

// ===========================================================================
// Part 1 — RPS prediction and the overfitting pattern
// ===========================================================================

= Where we are: predicting rotor speeds (RPS) from sound

- *Task:* recover the 4 rotor speeds (RPS, Hz) of a drone from its noise — the
  conditioning signal for RPS-aware speech enhancement and noise synthesis.
- *Data:* DREGON-LM-V4 + Michael's (FLY125 train / FLY124 valid), 8-channel,
  evaluated permutation-invariant (PIT) on a fixed validation set.
- *Recent work:* a *26-variant architecture sweep* around the `SimpleConvV2`
  baseline — temporal heads (BiGRU, GRU, TCN, Transformer), causal/streaming
  variants, input-feature and pooling changes — trained under *two regimes*:
  - *offline* fixed train set (50 epochs), and
  - *online mixing* — fresh speech/SNR/augmentation per sample (200 epochs).

#v(0.4em)
*Question that drove the sweep:* are causal/streaming temporal heads viable,
and does online mixing change the answer?

= Architecture sweep — offline leaderboard

#figure(
  image("assets/s1_offline_leaderboard.png", height: 80%),
  caption: [PIT MSE (log), 26 variants, fixed offline train set. Best: the
    `SimpleConvV2` + SMoLnet-refinement family (≈ 7.9). Causal/unidirectional-GRU
    variants are unstable and scattered (several diverged to NaN).],
)

= Online mixing rescues the overfit models...

#grid(
  columns: (1.45fr, 1fr),
  gutter: 1em,
  figure(
    image("assets/s1_offline_vs_online.png", height: 85%),
    caption: [PIT MSE, offline vs. online-mix (log–log). Points below the
      diagonal improve under online mixing.],
  ),
  [
    Fresh-sample gradients each step break memorisation:
    - Transformer #h(0.2em) $43.5 arrow 8.5$
    - uni-GRU family #h(0.2em) $39$–$228 arrow 7.3$–$8.7$

    Online mixing *compresses the field by lifting the failures*, and the wide
    unidirectional GRU (`uni_gru128`, *7.33*) takes the top spot — so model
    selection must be done *under the online regime*.
  ],
)

= ...but a residual pattern remains: sequence models still overfit

- Online mixing diversifies the *acoustic dressing* (speech, SNR, augmentation)
  of each trajectory — but *not the underlying set of RPS trajectories*, which is
  still just our handful of real flights.
- Models with explicit *temporal* structure (causal RNNs, the BiGRU baseline)
  can still *memorise the small menu of trajectories* instead of reading
  harmonics:
  - `causal_gru` stays the weakest GRU online ($R^2 = 0.77$, below baseline);
  - the BiGRU baseline only reaches its best epoch at epoch 42 — a slow grind on
    limited sequence structure.

#v(0.5em)
#align(center)[
  #block(fill: rgb("#fef3c7"), inset: 0.8em, radius: 0.3em, width: 92%)[
    *Acoustic mixing cannot fix this.* The fix is to *augment the RPS
    trajectories themselves* — generate novel, realistic rotor-speed curves (and
    the corresponding noise). The next two parts are the two halves of that.
  ]
]

// ===========================================================================
// Part 2 — the drone-noise generative model
// ===========================================================================

= Drone-noise model: harmonic + broadband synthesis from RPS

#grid(
  columns: (1.15fr, 1fr),
  gutter: 1.2em,
  figure(
    image("assets/s2_model_diagram.png", height: 86%),
    caption: [Architecture (Stage-2 report, Fig. 3.1).],
  ),
  [
    *Spectral-modelling synthesis* (DDSP-style), per drone:
    - A causal Conv1d encoder maps interpolated motor speeds to two heads.
    - *Harmonic* head → per-rotor harmonic amplitudes → *sinusoidal synthesiser*
      (driven by the rotor fundamentals).
    - *Broadband* head → a diffuse spectral shape → *time-variant noise filter*
      on white noise.
    - The two are summed and trained against the real recording with a
      *multi-resolution spectral loss*.

    #v(0.3em)
    The deployed model wraps this *per-rotor emitter* with positional propagation
    and per-drone conditioning (next slide).
  ],
)

= How it is designed to generalise

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    *Across microphone positions*
    - Each rotor is an isotropic *point source*, synthesised once at the rotor.
    - Rendering at a mic = pure free-field propagation: $1/r$ amplitude + an
      *exact fractional delay* $r/c$ (Fourier phase ramp).
    - $M$ mics cost $R$ forward + $M$ inverse transforms → *any array geometry*,
      and differentiable w.r.t. position.
  ],
  [
    *Across drones*
    - Per-drone conditioning is *external*: a `DroneCodebook` maps a drone name
      to a FiLM code $z$ that conditions the emitter.
    - Model parameters *never resize* with the number of drones.
    - An *unseen drone* is few-shot-adaptable: freeze the model, fit just its
      code $z$.
  ],
)

#v(0.5em)
Trained jointly on DREGON (8-mic) + Michael's DJI Matrice 100 (8-mic), ≈ 100
harmonics, multi-scale STFT loss.

= Real vs. generated — Michael's (DJI Matrice 100)

#figure(
  image("assets/s2_spec_michaels.png", width: 100%),
  caption: [The broadband shape and low-order harmonics transfer; the dense
    *mid-frequency harmonic comb* (≈ 1–3 kHz) is under-reproduced.],
)

= Real vs. generated — DREGON

#figure(
  image("assets/s2_spec_dregon.png", width: 100%),
  caption: [DREGON is reproduced *poorly*: the model captures broadband energy
    but loses most discrete harmonic lines.],
)

= Where the noise model struggles

- *DREGON is hard.* Its recordings carry strong *wind / aeroacoustic noise* that
  looks broadband; the model appears to fit that texture and *misses the discrete
  harmonics*, so DREGON is replicated much worse than Michael's.
- *Mid-frequency harmonics* are systematically under-restored (a loss-design
  issue — the multi-scale STFT loss is already log-dominated and averages over
  bins, so faint mid-band lines contribute little).
- *Takeaway:* the emitter is a good multi-mic / multi-drone *structure*, but
  fidelity is limited by (a) the wind-vs-harmonic confound and (b) the loss.

#v(0.4em)
Both halves — better noise *and* a richer supply of trajectories to drive it —
feed the same goal: training data that does not overfit.

// ===========================================================================
// Part 3 — realistic RPS trajectory synthesis (the new work)
// ===========================================================================

= New: generating realistic RPS trajectories

- *Goal:* an unlimited supply of novel, realistic 4-rotor RPS curves to break the
  trajectory-set overfitting from Part 1 (and to drive the noise model in Part 2).
- *Key idea 1 — control modes.* Work in the quadrotor's
  *common / roll / pitch / yaw* modes, not the 4 rotor channels. The $plus.minus 1$
  control-allocation mixer recovers the strong inter-rotor correlation *for free*.
- *Key idea 2 — control is intermittent.* The real telemetry says a piloted drone
  is *not* in constant motion.

= Real control is intermittent (the pilot mostly holds still)

#figure(
  image("assets/s3_rc_sticks.png", height: 76%),
  caption: [Raw DJI RC stick inputs: centred almost throughout, deflecting only in
    brief maneuver bursts. Measured on telemetry, the drone is *active only
    3.5–15.7 % of the time*, with maneuvers every 5–14 s.],
)

= A "pilot + airframe" model beats continuous noise

#figure(
  image("assets/s3_model_comparison.png", height: 78%),
  caption: [Real (top) vs. continuous-OU (middle) vs. *intermittent pilot +
    airframe* (bottom). Each mode = an intermittent setpoint (Poisson maneuver
    pulses) through a first-order *motor lag* + small cruise jitter. The
    intermittent model reproduces "hold, then a brief maneuver"; OU just wanders.],
)

= Two interpretable knobs

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  figure(
    image("assets/s3_intermittent_agg.png", width: 100%),
    caption: [*Aggressiveness* — scales how often / how hard the drone maneuvers,
      preserving the hold-then-burst texture.],
  ),
  figure(
    image("assets/s3_drone_profile_sweep.png", width: 100%),
    caption: [*Drone dynamics* — blends DREGON (small, fast) ↔ Michael's DJI M100
      (large, slow `motor_tau`); same maneuvers, different airframe response.],
  ),
)

= Summary & next steps

*Story so far*
- Architecture sweep + online mixing: online mixing is the right default, but
  *sequence models still overfit the small set of real RPS trajectories*.
- A *multi-mic / multi-drone noise generator* exists; it fits Michael's
  reasonably but *struggles on DREGON (wind) and mid-frequency harmonics*.
- *New:* a calibrated *intermittent "pilot + airframe"* RPS-trajectory generator
  with *aggressiveness* and *drone-dynamics* knobs — unlimited realistic curves.

#v(0.5em)
*Next*
- Train RPS predictors with synthetic-trajectory augmentation → test whether the
  residual temporal overfitting disappears.
- Drive the noise model with synthetic trajectories; revisit the loss to recover
  mid-frequency harmonics; separate wind from harmonics for DREGON.
