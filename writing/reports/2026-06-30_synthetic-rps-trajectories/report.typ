#import "/writing/templates/typst/report.typ": report, author-meta

#let fig(path, caption, width: 100%) = figure(image(path, width: width), caption: caption)

#show: report.with(
  title: [Synthetic RPS Trajectories: a Pilot-and-Airframe Model in Control-Mode Space],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    Training and evaluating RPS-conditioned models needs a steady supply of
    realistic rotor-speed (RPS) trajectories, but the real corpus is small —
    six DREGON in-flight recordings and two of Michael's flights, about ten
    minutes total. This report describes a lightweight generative model that
    produces unlimited novel 4-rotor RPS trajectories statistically similar to
    real flights, with two interpretable knobs: an *aggressiveness* dial and a
    *drone-dynamics* dial that morphs the airframe from the small DREGON quad to
    Michael's larger, slower DJI Matrice 100 (or anything in between).

    The model works in the quadrotor's *control modes* — collective thrust,
    roll, pitch, yaw — rather than the four rotor channels directly: projected
    through the inverse of the standard control-allocation mixer, real rotor
    speeds decompose into four weakly-coupled scalar signals, and recombining
    them through the mixer reproduces the strong inter-rotor correlation for
    free. The pilot logs reveal that real control is *intermittent*: the sticks
    sit centred for seconds at a time and deflect only in brief bursts, so the
    drone holds steady and maneuvers occasionally. We model each mode as an
    intermittent setpoint (Poisson maneuver events) passed through a first-order
    motor/airframe lag plus a small cruise jitter — the lag being the
    drone-dynamics knob. A simpler continuous Ornstein–Uhlenbeck (OU) variant is
    retained as a baseline and shown to be unrealistically "twitchy" by
    comparison. We calibrate both from DREGON (929 Hz telemetry) and Michael's
    flights and validate the marginal, joint, intermittency and per-mode
    statistics against the real data.
  ],
  keywords: ("RPS", "data augmentation", "intermittent control", "quadrotor", "control allocation"),
)

#set heading(numbering: "1.")

= Motivation

RPS-conditioned speech enhancement and the RPS-prediction task both consume
rotor-speed trajectories: as conditioning signals, as prediction targets, and
(increasingly) as the driver of harmonic-noise synthesis in the generative
noise pipeline. The supply of *real* trajectories is tiny — six DREGON
`in_flight_noise` recordings and Michael's two DJI flights, on the order of ten
minutes of in-flight telemetry. Online mixing already recycles these aggressively;
any model trained on them risks memorising the handful of real maneuver patterns.

We want a generator that (i) emits unlimited novel trajectories, (ii) is
statistically faithful to real flights, (iii) exposes a small number of
interpretable controls — in particular a difficulty/aggressiveness dial so we can
curriculum-train or stress-test, and a knob that morphs the airframe between our
two real drones — and (iv) is cheap and dependency-free.

Two design choices follow from looking at the data. First, we work in the
quadrotor's *control modes* rather than the four rotor channels, so a generator of
independent scalar signals recovers the real inter-rotor correlation through the
mixer (§2). Second — and this is what makes the trajectories *look* real — we make
each mode *intermittent*: the pilot logs (§3) show the drone holds steady for
seconds and maneuvers only in brief bursts, so a continuous random process is the
wrong primitive. The main model (§4) is therefore an intermittent setpoint passed
through a motor/airframe lag; a continuous Ornstein–Uhlenbeck (OU) variant is kept
as an instructive baseline (§6). A *multivariate AR(p)* fit to the concatenated
traces was also considered but rejected as uninterpretable and bounded by the
$tilde$10 minutes of data.

= Control-mode decomposition

A quadrotor actuates four degrees of freedom through its four motors via a fixed
linear control-allocation mixer $B$. With rotor order
(RFront, LFront, LBack, RBack) and mode order (common, roll, pitch, yaw),

$ w = B m, quad B = mat(
  1, +1, +1, +1;
  1, -1, +1, -1;
  1, -1, -1, +1;
  1, +1, -1, -1) $

where $w in RR^4$ are rotor speeds (rev/s) and $m in RR^4$ are the mode
coefficients. The columns of $B$ are mutually orthogonal with squared norm 4, so
$B^T B = 4 I$ and the inverse projection is simply $m = B^T w \/ 4$. The *common*
mode is the mean rotor speed (collective thrust / altitude); *roll* and *pitch*
are the front–back and left–right differentials; *yaw* is the diagonal-pair
differential (CW vs CCW rotor torque balance).

Projecting real flights onto these modes (@fig-modes, left column) shows why the
decomposition is useful: the four mode signals are far closer to mutually
independent than the four rotor channels (which are all strongly positively
correlated through the dominant common mode). We can therefore model each mode
*independently* and let the mixer reintroduce the correct correlation. The
question is only what process to put on each mode.

= Real control is intermittent <sec-intermittent>

A human-piloted drone is not in constant motion. Michael's DJI logs record the
raw RC stick inputs — the actual pilot commands — and these
(@fig-rc) sit centred near zero for the great majority of the flight, deflecting
only in short bursts when the pilot commands a maneuver. The drone consequently
holds a near-constant attitude most of the time and moves only occasionally.

#fig("assets/rc_sticks.png",
  [Raw DJI RC stick inputs (normalised) for the two Michael's flights, against the
   logger clock. The sticks are centred almost throughout; maneuvers are brief
   deflections. Large spikes at the ends are takeoff/landing. (Aligning these to
   audio would apply the same whole-table `time_offset` / `time_dilation`
   correction as `data_processing/michaels.py`; not needed to show structure.)])
  <fig-rc>

We quantify this on the rotor telemetry itself by projecting each recording onto
the control modes and flagging *active* frames — those whose 0.5 s rolling
differential-mode std exceeds the per-recording median by $3 dot 1.4826 dot$MAD
(@tbl-intermit). Across the eight real flights the drone is *active only
3.5–15.7 % of the time*, with maneuver onsets every 5–14 s. This is the single
most important fact for realism, and a continuous process — which spreads its
variance evenly across all time — cannot reproduce it.

#figure(
  text(size: 8pt)[
    #table(
      columns: 5,
      align: (left, right, right, right, right),
      inset: 4pt,
      stroke: 0.35pt + luma(180),
      table.header([recording], [dur (s)], [active %], [maneuver rate (1/s)], [mean hold (s)]),
      [DREGON/free-flight r1], [60.2], [11.4], [0.166], [5.3],
      [DREGON/free-flight r2], [74.7], [10.7], [0.187], [4.8],
      [DREGON/hovering], [33.5], [4.1], [0.090], [10.7],
      [DREGON/updown], [43.4], [15.7], [0.138], [6.1],
      [DREGON/rectangle], [34.1], [4.8], [0.088], [10.8],
      [DREGON/spinning], [28.1], [3.5], [0.071], [13.6],
      [michaels/124], [105.7], [5.5], [0.095], [10.0],
      [michaels/125], [170.9], [3.7], [0.105], [9.1],
    )
  ],
  caption: [Maneuver-activity intermittency of the real flights. The drone holds
    steady 84–96 % of the time, maneuvering in brief bursts.],
) <tbl-intermit>

= The pilot-and-airframe model <sec-model>

We model each control mode with a two-layer "pilot + airframe" structure that
mirrors how the trajectory is actually produced.

+ *Intermittent setpoint* (the pilot). A piecewise-constant signal $s_k (t)$ that
  holds at the mode's trim value. Maneuver onsets arrive as a Poisson process of
  rate $lambda_k$; each adds a rectangular deflection of amplitude
  $a tilde cal(N)(0, sigma_k^2)$ (random sign) for a duration
  $tilde "Exp"(macron(d)_k)$ before returning to trim. The active fraction is then
  $approx lambda_k macron(d)_k$.

+ *Motor / airframe lag* (the airframe). The setpoint is low-passed by a causal
  first-order filter with time constant $tau_"motor"$, rounding the rectangular
  step edges into the smooth ramps that rotor inertia produces. Because the lag is
  linear it commutes with the mixer, so applying it per mode is exact. *This is the
  drone-dynamics knob*: a small, snappy quad has a short $tau_"motor"$; a large,
  sluggish one a long $tau_"motor"$.

+ *Cruise jitter.* A small-amplitude OU term (the same primitive as the baseline,
  §6) is added so that holds are never perfectly flat.

Recombining via $w = B m$ and clipping gives the trajectory. Two knobs control it:

- *aggressiveness* $a$ scales every mode's onset rate $lambda_k$ and amplitude
  $sigma_k$ — gentle near-hover at $a < 1$, busy maneuvering at $a > 1$;
- *`drone_profile`* $in [0, 1]$ linearly blends a DREGON profile (0) and a
  Michael's profile (1) — every trim, amplitude, rate, duration and $tau_"motor"$
  is interpolated, so $0.5$ is a genuine in-between airframe.

The two profiles are calibrated from the data: maneuver structure
($lambda$, $macron(d)$) from @tbl-intermit, trim biases and amplitudes from the
mode projection (with $sigma_k$ set so the overall per-mode std — diluted by the
$tilde 8 %$ active fraction — matches the measured spread), and $tau_"motor"$ set
short (0.15 s) for the small DREGON quad and long (0.35 s) for the Matrice 100.
The implementation (`generate_intermittent`, `DroneProfile`, `blend_profiles` in
`data_processing/rps_synthesis.py`) is pure NumPy and covered by unit tests
asserting the bursty active fraction, monotonic aggressiveness, profile
interpolation, and that a larger $tau_"motor"$ measurably smooths the response.

= Results <sec-results>

@fig-compare is the central result: a real DREGON cruise window, the continuous-OU
baseline, and the intermittent model side by side. The intermittent trajectory
holds steady plateaus with a fixed rotor ordering and breaks into brief maneuvers
that settle to a new level — the texture of the real trace — whereas OU wanders
continuously at all times.

#fig("assets/model_comparison.png",
  [Real (top) vs. continuous-OU (middle) vs. intermittent pilot-and-airframe
   (bottom) trajectories. The intermittent model reproduces the "hold, then a
   brief maneuver" structure; OU does not.])
  <fig-compare>

The *aggressiveness* knob (@fig-iagg) scales how often and how hard the drone
maneuvers while preserving the intermittent character at every level: gentle
($a = 0.4$) is almost pure hold with rare small corrections; aggressive
($a = 2.5$) maneuvers frequently and to the clamp.

#fig("assets/intermittent_agg.png",
  [Intermittent trajectories at three aggressiveness levels. The hold-then-burst
   texture is preserved; only the frequency and size of maneuvers change.])
  <fig-iagg>

The *drone-dynamics* knob (@fig-drone) morphs the airframe with the maneuver event
times held fixed (shared seed), isolating the dynamics: from DREGON (top) to
Michael's (bottom) the hover level drops $approx 80 -> 68$ rev/s and the same
maneuvers render progressively more rounded and sluggish as $tau_"motor"$ grows.

#fig("assets/drone_profile_sweep.png",
  [The drone-dynamics knob with maneuver times fixed: DREGON (small, fast) ->
   in-between -> Michael's (DJI M100, large, slow). Lower hover and slower,
   rounder maneuvers as `drone_profile` increases.])
  <fig-drone>

= The continuous-OU baseline <sec-ou>

The intermittent model's cruise jitter, and a useful point of comparison, is the
simpler model we started with: drive each mode with a continuous
Ornstein–Uhlenbeck (OU) process — the simplest stationary, mean-reverting,
temporally-correlated Gaussian process,

$ dif m = 1/tau (mu - m) dif t + sigma_"drive" dif W , $

parametrised by its *stationary mean* $mu$, *stationary std*
$sigma = sigma_"drive" sqrt(tau\/2)$ and *correlation time* $tau$, and sampled by
the exact discrete transition (valid for any step $Delta t$):

$ m_(i+1) = mu + phi (m_i - mu) + epsilon_i, quad
  phi = e^(-Delta t \/ tau), quad
  epsilon_i tilde cal(N)(0, sigma^2 (1 - phi^2)) . $

This needs just twelve numbers (`DEFAULT_CONFIG`), exposed through `generate` /
`generate_batch` with the same `aggressiveness` knob (scaling every $sigma$). It
reproduces the real *marginal* and *joint* statistics well — it is only the
*temporal* character (continuous wander vs. intermittent holds) that it gets
wrong, which is exactly what motivated the model of §4. Its parameters are
calibrated by projecting each recording onto the modes and estimating
$(mu, sigma, tau)$ from the lag-1 autocorrelation $rho_1 = e^(-Delta t\/tau)$
(@tbl-calib).

#figure(
  text(size: 7.5pt)[
    #table(
      columns: 13,
      align: (left,) + (right,) * 12,
      inset: 3.2pt,
      stroke: 0.35pt + luma(180),
      table.header(
        [recording], [c·μ], [c·σ], [c·τ], [r·μ], [r·σ], [r·τ],
        [p·μ], [p·σ], [p·τ], [y·μ], [y·σ], [y·τ],
      ),
      [free-flight r1], [79.8], [4.5], [0.99], [0.0], [0.58], [1.19], [0.3], [0.55], [0.81], [5.1], [1.28], [2.88],
      [free-flight r2], [80.2], [3.5], [0.57], [1.2], [0.71], [0.45], [-0.3], [0.90], [0.73], [2.8], [1.17], [0.73],
      [hovering], [79.9], [5.1], [0.47], [1.4], [0.55], [0.25], [-0.7], [1.02], [0.91], [2.0], [1.38], [0.82],
      [updown], [80.5], [5.0], [0.38], [1.3], [1.12], [0.36], [-0.7], [1.42], [0.50], [2.5], [1.63], [0.74],
      [rectangle], [80.5], [5.2], [0.64], [1.4], [0.83], [0.65], [-0.2], [0.93], [0.88], [2.5], [1.77], [1.56],
      [spinning], [80.2], [5.7], [0.40], [1.4], [0.63], [0.26], [-0.4], [0.75], [0.38], [1.6], [1.34], [0.63],
      table.hline(stroke: 0.6pt),
      [*DREGON median*], [*80.2*], [*5.0*], [*0.52*], [*1.3*], [*0.67*], [*0.41*], [*-0.4*], [*0.92*], [*0.77*], [*2.5*], [*1.36*], [*0.78*],
      table.hline(stroke: 0.3pt + luma(200)),
      [michaels/124], [66.9], [19.8], [34.4], [1.7], [3.63], [2.74], [0.8], [2.45], [3.99], [3.9], [2.87], [9.31],
      [michaels/125], [76.0], [12.8], [12.7], [2.2], [1.40], [0.83], [1.5], [2.32], [1.98], [5.2], [1.99], [3.83],
    )
  ],
  caption: [Per-recording OU mode parameters. Columns are mean (μ), stationary
    std (σ, rev/s) and correlation time (τ, s) for the common (c), roll (r),
    pitch (p) and yaw (y) modes. The bold row is the DREGON median used for
    `DEFAULT_CONFIG`.],
) <tbl-calib>

Three observations from this table inform both models (the trim biases and
amplitudes are shared; only the temporal primitive differs):

+ *Common mode* sits at a stable $approx 80$ rev/s hover level across all DREGON
  flights, wandering by 3.5–5.7 rev/s with a sub-second-to-second correlation
  time. Its variance is largest for `updown` and `spinning` (altitude/throttle
  activity), as expected.

+ *Maneuver modes carry persistent trim biases.* Roll sits near $+1.3$, yaw near
  $+2.5$ rev/s — the drone holds a static differential to balance net torque
  (the CW/CCW rotor-drag asymmetry), then fluctuates about it. The model keeps
  these as fixed means and only scales the fluctuation.

+ *Sample rate matters for $tau$.* DREGON's 929 Hz telemetry resolves the
  sub-second maneuver dynamics cleanly; Michael's 29 Hz logs alias the short time
  constants into implausibly long values (e.g. $tau = 34$ s for 124's common
  mode reflects slow drift over a 100 s flight, not the maneuver bandwidth).
  We therefore calibrate `DEFAULT_CONFIG` from DREGON only.

The hand-set defaults and the `fit_config` DREGON-median agree closely
(common $sigma$ 4.0 vs 5.0, yaw $sigma$ 1.40 vs 1.36, $tau$'s within a factor of
$tilde$1.3), confirming the fit is stable.

== OU baseline validation

The OU baseline matches the real data on every *static* statistic; the figures
below establish that, so that the only thing left to fix — the temporal texture —
is clearly attributable to the continuous-process assumption (and addressed by §4).

@fig-traj shows three OU batches at increasing aggressiveness. All four rotors
track together (common mode) while their spread grows with $a$ — gentle hover
stays within a few rev/s, the aggressive profile swings $plus.minus 25$ rev/s and
approaches the physical clamp.

#fig("assets/traj_examples.png",
  [OU trajectories at three aggressiveness levels (shared axes). Rotors move
   together via the common mode; differential spread scales with the knob. Note
   the continuous wander — contrast the intermittent model, @fig-iagg.])
  <fig-traj>

@fig-realsynth places a real DREGON free-flight next to a calibrated OU
trajectory. The OU flight reproduces the qualitative texture: a dominant
common level, a steady rotor ordering from the trim biases, and second-scale
wander. (`free-flight_room2` is one of the calmer recordings — closer to
$a approx 0.7$, cf. @fig-sweep — so the $a = 1.0$ synthetic shows somewhat more
activity, matching the DREGON *average* rather than this specific calm flight.)

#fig("assets/real_vs_synth.png",
  [Real DREGON flight (left) vs. calibrated OU trajectory at
   aggressiveness 1.0 (right). Takeoff ramp trimmed from the real trace.])
  <fig-realsynth>

@fig-modes confirms the per-mode dynamics match: OU common, roll, pitch
and yaw modes have the same amplitudes and timescales as the real ones (after the
takeoff transient). The real common mode here is flatter because the `rectangle`
flight held constant altitude, whereas the synthetic common carries the
across-flight median variance.

#fig("assets/mode_decomposition.png",
  [Control-mode decomposition, real `rectangle` flight (left) vs. synthetic
   (right). Per-mode amplitude and timescale match; the real common mode is
   flatter (constant-altitude flight).])
  <fig-modes>

@fig-dist validates the aggregate statistics. The centred rotor-speed marginal
of the synthetic batch overlaps the pooled real DREGON distribution, and the
inter-rotor correlation matrix reproduces the real pattern: all pairs strongly
positive, with diagonal pairs (RFront–LBack) more correlated than adjacent ones.
The synthetic correlations are slightly *less* asymmetric than real (0.92 vs 0.81
diagonal/adjacent, against the real 0.91 vs 0.63), because real flights excite the
yaw (diagonal) mode relatively more than our isotropic knob does — see Discussion.

#fig("assets/distributions.png",
  [Marginal centred rotor-speed distribution and inter-rotor correlation
   matrices, real (pooled DREGON) vs. synthetic.])
  <fig-dist>

Finally, @fig-sweep shows the aggressiveness knob is monotonic and linear in
maneuver activity (mean temporal std of the roll/pitch/yaw modes), passing
through zero at $a = 0$. The real DREGON flights land between $a approx 0.7$
(`free-flight`) and $a approx 1.4$ (`updown`), so $a = 1$ is a typical
maneuvering flight and the knob comfortably brackets — and extrapolates beyond —
the observed range.

#fig("assets/aggressiveness_sweep.png",
  [Maneuver activity vs. the aggressiveness knob (synthetic curve), with real
   flights as horizontal references. $a = 1$ matches a typical DREGON flight.],
  width: 80%)
  <fig-sweep>

= Discussion and limitations

- *Isotropic maneuvering.* The `aggressiveness` knob scales all modes equally, but
  real flight types are anisotropic — `spinning` is yaw-heavy, `updown` is
  common-heavy, `rectangle` is roll/pitch-heavy. A natural extension is a small set
  of named maneuver *presets* — per-mode rate/amplitude profiles for
  hover / translate / spin / climb — calibrated per real flight type, layered on
  top of the existing per-mode parameters.

- *Calibration from RC sticks.* We calibrate maneuver *amplitudes* in RPS
  mode-space (directly matchable) and use the raw RC sticks (@fig-rc) only to
  establish the intermittent *structure*. Mapping the normalised stick units to
  RPS through an identified stick→thrust gain would let the pilot command itself
  drive the setpoint, tightening the model; it needs the whole-table time
  alignment of `michaels.py` extended to the RC columns.

- *Rectangular maneuver pulses.* Each maneuver is a rectangular setpoint pulse
  (smoothed by the airframe lag). Real maneuvers have structured rise/hold/fall and
  occasionally *re-trim* to a new attitude rather than returning; a richer pulse
  shape or an occasional setpoint step would capture this.

- *Drone profiles from two airframes.* The `drone_profile` knob interpolates
  exactly two calibrated drones. It extrapolates plausibly but is only validated at
  the endpoints; more airframes would make the axis meaningful as a continuum.

- *No takeoff/landing.* Both generators produce in-flight cruise only; the RC logs
  show takeoff/landing as large stick excursions (@fig-rc) that we exclude. A
  non-stationary common-mode envelope would add ramps and spindown.

- *Mixer assumptions.* We assume the idealised $plus.minus 1$ allocation and a
  rigid hover trim; real allocation has small asymmetries (the residual
  correlation mismatch in @fig-dist) that do not affect marginal fidelity.

= Reproducibility

Generate a realistic intermittent batch, or the OU baseline:

```python
from data_processing.rps_synthesis import (
    generate_intermittent_batch, generate_batch, blend_profiles,
)
# intermittent, in-between airframe, busy maneuvering:
batch = generate_intermittent_batch(
    64, duration=8.0, fs=100.0, drone_profile=0.5, aggressiveness=1.5, rng=0,
)  # -> (64, 4, M) rev/s, rotor order (RFront, LFront, LBack, RBack)

# continuous-OU baseline (same knobs minus drone_profile):
ou = generate_batch(64, duration=8.0, fs=100.0, aggressiveness=1.5, rng=0)
```

Rebuild this report's figures and tables (real data found under `$DATA_ROOT` or
the repo's `data/DREGON`):

```bash
cd writing/reports/2026-06-30_synthetic-rps-trajectories
make all      # prepare.py -> figures + CSV tables, then typst compile
```
