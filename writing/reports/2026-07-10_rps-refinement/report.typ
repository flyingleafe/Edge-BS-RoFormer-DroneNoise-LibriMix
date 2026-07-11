// RPS trajectory refinement by comb alignment.
#import "/writing/templates/typst/report.typ": report, author-meta

#let fig(path, caption) = figure(image(path, width: 92%), caption: caption)

#show: report.with(
  title: [How Good Are Our RPS Labels? A Comb-Alignment Audit
    (and Why Refinement Buys Little)],
  authors: (
    "Dmitrii Mukhutdinov": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    We suspected our rotor-speed (RPS) labels of sabotaging generator
    training: a label error of a few tenths of rev/s displaces the
    #emph[k]-th rotor harmonic by #emph[k]-times that error in Hz — enough
    to knock mid-frequency harmonics off their spectral peaks. To test
    this we built a comb-alignment refinement stack (a home-grown instance
    of #emph[tacholess order tracking]) and audited the labels against the
    audio itself, using DREGON recordings that log both #emph[commanded]
    and #emph[measured] rotor speeds as a natural experiment. *The verdict
    goes against the hypothesis*: our telemetry labels are already
    essentially unbiased (+0.02 rev/s), the residual error is fast zero-mean
    jitter that no spectral refinement can recover on this airframe, and
    the magnitude-based refiner is actively #emph[biased] (−0.44 rev/s) by
    DREGON's tightly-paired rotors. The practical outputs are therefore a
    #emph[negative result with consequences] — the mid-harmonic washout in
    generator training is better explained by label-invisible #emph[jitter
    linewidth] than by label error, moving the fix into the generator — plus
    a modest toolbox: per-recording clock alignment, a comb-presence
    verifier that honestly refuses non-harmonic audio, and a quantified
    operating envelope. Blind #emph[trajectory] annotation of unlabeled
    recordings is *not* demonstrated: on maneuvering SPCup recordings the
    refiner recovers mean base speeds, not maneuvers; a Vold-Kalman-class
    coupled order tracker is the right instrument if that is ever needed.
  ],
  keywords: ("RPS estimation", "harmonic comb", "label refinement",
             "order tracking", "variable projection"),
)

#set heading(numbering: "1.")

= Why refine labels at all?

Every model in this project that touches rotor speed — the RPS predictors,
the harmonic noise generator — is trained against telemetry-derived
trajectories $r_i (t)$ (rev/s, one per rotor). Telemetry is close to the
truth, but not equal to it:

- *Clock offset.* The telemetry logger and the audio interface have separate
  clocks; a constant (or slowly drifting) offset $tau$ misaligns every label,
  and during a manoeuvre with slew $dot(r)$ the induced speed error is
  $dot(r) dot tau$.
- *Command $eq.not$ actual.* DREGON logs both the flight controller's
  #emph[commanded] speeds and (for some recordings) the #emph[measured] ones;
  most derived datasets use the command track, which the motors only follow
  approximately.
- *Sparse sampling.* Michael's DJI logs tick at $tilde 29$ Hz; anything the
  rotors do between ticks is interpolation.

== The multiplication that makes small errors matter

A rotor spinning at $r$ rev/s radiates at harmonics of $r$: the $k$-th
harmonic sits at $k r$ Hz. If the label is wrong by $delta$, the model's idea
of harmonic $k$ sits at $k (r + delta)$ — displaced by $k delta$ Hz. With the
spectrogram resolution used throughout this project (2048-point FFT at
16 kHz, i.e. 7.8 Hz per bin):

$ delta = 0.3 "rev/s": quad k = 2 arrow.r 0.6 "Hz (invisible)", quad
  k = 40 arrow.r 12 "Hz" approx 1.5 "bins (off the peak entirely)". $

#fig("assets/method_displacement.png", [A single label error $delta$ displaces
  the $k$-th harmonic by $k delta$ Hz. With $delta = 0.3$ rev/s the first few
  harmonics stay within their spectrogram peak; beyond $k approx 25$ the
  labelled track has left the peak entirely.])

So label error is not one number — it is a #emph[frequency-proportional]
corruption. Low harmonics stay aligned; mid and high harmonics land in the
valleys next to their true peaks. For a generative model whose oscillators
are #emph[placed] at label-implied frequencies and can only learn amplitudes,
the optimal response to a misaligned harmonic is to #emph[suppress] it — a
candidate explanation for the washed-out mid-frequency harmonics we observed
when training the noise generator.

#emph[If] labels were substantially wrong, refining them would pay three
times over: cleaner targets for the generator, a lower error floor for the
RPS predictors, and — because the refinement objective doubles as a
#emph[verifier] — a route to annotating unlabeled recordings. That was the
hypothesis. The rest of this report builds the instrument, and then reports
honestly that the hypothesis fails: the labels were already good, and the
instrument's real yield is the audit itself.

= The method

== One idea, three stages

Fix a spectrogram $L(f, t)$ (log-magnitude, averaged over microphones). If a
trajectory guess $hat(r)(t)$ is correct, then the curves
$f = k hat(r)(t)$ for $k = 1, 2, 3, dots$ run along bright ridges of $L$;
if the guess is off, they run through the dark valleys beside them. Define
the *comb score*

$ S(hat(r)) = "mean"_(k, t, "mic") L(k hat(r)(t), t), $

the average brightness of the spectrogram sampled along the whole comb
(linear interpolation between frequency bins; harmonics outside
$[60, 6000]$ Hz excluded; the #emph[mean] rather than the sum makes a
half-empty comb score poorly, which suppresses subharmonic impostors).
Refinement is then just: adjust $hat(r)$ to maximise $S$
(@method_comb shows the before/after on a synthetic comb). Because the true
comb is a sharp, repeated structure, $S$ has a pronounced maximum at the true
trajectory — but also local ripples away from it, so we go coarse-to-fine:

+ *Stage A — clock alignment.* Scan a single time offset
  $tau in [-0.5, 0.5]$ s applied to the whole telemetry track; keep the
  best-scoring $tau$ (parabolic interpolation gives sub-step precision).
  One number per recording, catches the biggest systematic error first.
+ *Stage B — coarse capture.* Split the recording into 2 s windows; in each
  window and for each rotor, grid-search a #emph[constant] correction
  $delta in [-3, 3]$ rev/s (step 0.05) using only harmonics $k <= 20$. Low
  harmonics move little per unit $delta$, so their score basins are wide —
  ideal for capture; precision comes later.
+ *Stage C — fine refinement.* Parameterise each rotor's correction
  $delta_i (t)$ as a spline (knots every 0.25 s) and maximise $S$ with
  gradient descent (Adam, 300 steps) over all rotors jointly, now using the
  full comb ($k$ up to 80). Two regularisers keep the solution honest: a
  smoothness penalty on the knots' second differences (rotors have inertia)
  and a weak anchor toward the initialisation.

#figure(
  image("assets/method_comb_alignment.png", width: 100%),
  caption: [High harmonics ($k approx 23 dots 33$) of a synthetic rotor.
    Left: labels with a $approx 0.5$ rev/s error — the labelled tracks
    (dashed) run through the valleys beside the true ridges. Right: after
    refinement the tracks lock onto the ridges; mean error drops to
    0.028 rev/s.],
) <method_comb>

== Why this is easy: the linear/nonlinear split

The underlying signal model is
$ x(t) = sum_(i, k) a_(i k)(t) cos(2 pi k integral_0^t r_i + phi.alt_(i k)) + "broadband", $
which looks daunting until you notice that #emph[given the trajectories], all
amplitudes $a_(i k)$ and phases $phi.alt_(i k)$ enter #emph[linearly] (a
cosine with unknown amplitude and phase is a linear combination of a cosine
and a sine). This is the classical #emph[variable projection] structure:
the linear parameters solve in closed form, and the genuinely nonlinear
search is only over the low-dimensional spline corrections — a few numbers
per rotor per quarter-second, not a function-space problem.

We exploit the linear half twice:

- *Fit metric.* After refinement we solve, block by block (0.25 s), the
  joint linear least-squares problem for all harmonic amplitudes of all
  rotors at once, and report the *residual energy ratio*
  $norm(x - hat(x))^2 slash norm(x)^2$. Jointness matters: quadrotor speeds
  sit within a few rev/s of each other, so low harmonics of different rotors
  share spectrogram bins and #emph[interfere]; fitting tracks independently
  double-counts that shared energy, while the joint solve attributes it
  correctly.
- *Confidence.* For each window we report the #emph[comb contrast]: the score
  at $hat(r)$ minus the median score at deliberately de-tuned copies
  ($hat(r) plus.minus 0.75 dots 2$ rev/s). A real locked comb has high
  contrast; noise without harmonic structure has none. This is the gate that
  makes refined labels trustworthy: windows with low contrast are rejected,
  not guessed.

= Validation on annotated data: a natural experiment that bit back

DREGON's five `free-flight_*_room1` recordings log both the flight
controller's #emph[commanded] rotor speeds and the #emph[measured] ones.
That makes a perfect validation: initialise refinement from the command
track, never show it the measured track, and check whether it moves toward
the truth. It did not — and the forensics of #emph[why] taught us more than a
success would have.

== The magnitude stages fail here — with a systematic bias

#table(
  columns: (1.5fr, 0.8fr, 0.8fr, 0.8fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center, center),
  [*trajectory*], [*err*], [*bias*], [*err vs sm.*], [*bias vs sm.*],
  [*LSQ resid*],
  [command (telemetry)], [0.633], [−0.057], [0.484], [*+0.017*], [0.44–1.08],
  [stage B+C (magnitude)], [*0.848*], [*−0.440*], [0.749], [−0.366], [0.44–0.54],
  [stage D (coherent)], [0.638], [−0.078], [0.491], [*−0.004*], [0.43–0.63],
  [measured (truth)], [—], [—], [—], [—], [*0.42–0.46*],
)
#align(center)[#text(size: 9pt)[Pooled over the five room1 recordings; err/bias
in rev/s against `motors_measured` (raw and 0.25 s-smoothed); LSQ resid =
joint harmonic fit residual ratio (range across recordings, k ≤ 40).]]

#figure(
  image("assets/val_overlay.png", width: 100%),
  caption: [Refining DREGON command labels toward the #emph[hidden] measured
    truth. #emph[Left:] one rotor over a 12 s window — the magnitude-only
    stage B+C (red) drifts consistently #emph[below] the measured speed
    (black), while the coherent stage D (blue) and the raw command track
    (gray) stay on it. #emph[Right:] averaged over the whole recording, every
    rotor's signed error tells the same story: stage B+C carries a systematic
    downward bias (≈ −0.41 rev/s here, −0.44 pooled across the five
    recordings above), whereas command and stage D are essentially unbiased.
    #emph[Takeaway:] chasing spectrogram #emph[brightness] alone pulls the
    estimate off the true speed on this twin-paired airframe; reading the
    #emph[phase] (stage D) does not.],
) <val_overlay>

#figure(
  image("assets/val_jitter.png", width: 90%),
  caption: [Why even the unbiased stage D cannot shrink the #emph[unsigned]
    error. A 4 s ultra-zoom of one rotor: the measured speed (black) jitters
    rapidly around its local mean, but neither the smooth telemetry (gray) nor
    the refined coherent track (blue) follows those fast wiggles — they are
    #emph[label-invisible]. This irreducible jitter, not a calibration offset,
    is what remains after refinement; it is also the better explanation for
    the washed-out mid-frequency harmonics in generator training.],
) <val_jitter>

Across all five recordings, stage B+C refinement moved the labels *away*
from measured truth (@val_overlay; pooled |error| 0.63 → 0.85 rev/s), with a
consistent *downward* bias of ≈ 0.44 rev/s — while simultaneously
#emph[improving] the joint least-squares audio fit relative to command. Three
observations pin the mechanism:

+ The command track is already nearly #emph[unbiased]: its signed offset
  from measured is only ≈ −0.04 rev/s; its 0.63 rev/s unsigned error is
  zero-mean, fast fluctuation ("jitter"), not calibration error.
+ This drone's four rotors fly as #emph[two tight pairs] (≈ 85.4/84.8 and
  75.2/74.5 rev/s — about 0.65 rev/s apart within a pair). Below
  $k approx 13$ the paired harmonics fall within one spectrogram peak: the
  magnitude ridge sits #emph[between] the two true frequencies, and
  per-rotor tracks have no per-rotor peak to lock onto.
+ Band-restricted scoring shows the truth signal lives at high $k$: for
  $k in [45, 80]$ the measured trajectory out-scores everything, but for
  $k <= 25$ a down-shifted comb scores markedly better (non-rotor structure
  and merged peaks) — and under uniform averaging the low band, with its
  larger score differences, #emph[outvotes] the resolved high harmonics.

(A metric caveat, kept honest: on two badly-misaligned windows the LSQ
residual ratio exceeded 1 — the windowed overlap-add reconstruction is
calibrated for approximately-correct bases and over-subtracts when the
basis is wrong. The #emph[ordering] of residuals remained informative in
every case; absolute values above ~1 just mean "very wrong".)

The failure is instructive precisely because our synthetic tests could not
produce it: synthetic combs had no confounding low-band structure and no
tight pairs, so uniform harmonic weighting was harmless there. Restricting
or re-weighting the magnitude objective toward high $k$ does *not* fix the
real data — the resolved high harmonics of the quiet pair member are then
captured by its louder twin's comb 0.65 rev/s away (bias flips to +0.8).
Magnitude ridges alone cannot arbitrate twin-paired rotors.

== Stage D: read the phase, not the ridge

The coherent fix follows from the same variable-projection insight as the
fit metric. Demodulate the signal by the current track's phase
($z_k (t) = "lowpass"(x dot e^(-i 2 pi k integral hat(r)))$): if the track
is off by $delta$, the complex envelope $z_k$ rotates at $k delta$ Hz — the
error becomes a #emph[phase slope], read locally within a narrow band
(3 Hz). Combining slopes across harmonics with Fisher weights
($k^2 |z_k|^2$) gives an update whose precision grows with $k$, and whose
narrowband locality #emph[structurally excludes] the twin's comb for
$k gt.eq 13$. On the synthetic twin-pair fixture this cuts error
0.26 → 0.033 rev/s with per-rotor bias under 0.03; on the real recordings it
is unbiased (|bias| < 0.08 rev/s) and improves the audio fit where the
magnitude stages degraded it.

== What refinement can and cannot buy on this data

Stage D does #emph[not] shrink the unsigned error against measured — because
what remains is fast, zero-mean jitter (@val_jitter), and recovering jitter
through phase-slope tracking needs #emph[low] harmonics (small
phase-modulation index), while twin rejection needs #emph[high] ones. On twin-paired
airframes those requirements collide; the jitter is structurally
unrecoverable from magnitude or narrowband phase alike. The honest summary:

- DREGON's command labels are, at recoverable timescales, already excellent
  (bias −0.06 rev/s raw, +0.02 against smoothed measured); there was far
  less headroom than the project assumed. The same holds for Michael's
  manual alignment: stage D moves those trajectories by only
  0.03–0.08 rev/s on average — telemetry is within the refiner's noise
  floor — while stage B+C drifts them by its characteristic 0.7–1.3 rev/s
  and, on one idle-to-ramp segment, mis-locks outright (fit residual 4.0:
  worse than fitting silence). Where the refiners disagree, trust the
  coherent one.
- The refinement machine still pays for itself as a #emph[clock aligner]
  (stage A found per-recording offsets up to 0.21 s — during a manoeuvre at
  10 rev/s², that alone is a 2 rev/s label error), as a #emph[verifier]
  (comb confidence + LSQ residual), and as a #emph[blind annotator]
  (next section).
- The unrecoverable jitter reframes the mid-frequency harmonic washout in
  generator training: at $k = 30$, ±0.6 rev/s of label-invisible jitter
  smears the true harmonic over ±18 Hz while the generator, conditioned on
  the smooth label, renders a clean tone — the loss then prefers to
  suppress it. The fix belongs in the #emph[generator]: per-harmonic
  linewidth (stochastic phase diffusion growing with $k$), not in ever
  better labels.

= Unlabeled recordings (SPCup): comb detection, not annotation

For unlabeled data there is no telemetry to start from, so the pipeline
gains a blind initialisation: scan a constant base speed $r_0$ over
30–120 rev/s with the comb score, take the peaks (recording both octave
candidates — $2 r_0$'s harmonics are a subset of $r_0$'s, so the scan alone
cannot always tell them apart), then select the rotor count $R in {1, 2, 4}$
by the residual-improvement elbow of the joint LSQ fit.

Let us be precise about what this achieves — and what it does not. On seven
recordings spanning six different SPCup team rigs (1–8 channels):

- *Comb detection and mean base speed work.* Five recordings yield sharp
  scan peaks at plausible quad base speeds (28–60 rev/s), one resolving two
  overlapping combs (KU Leuven, 42.5 + 46.2 rev/s). One monotone recording
  with no comb structure is #emph[refused] (confidence 0.018) rather than
  guessed, and a calibration-tone control that fools the confidence gate is
  caught by its poor LSQ residual (0.90) — the two gates are complementary.
- *Trajectory tracking through maneuvers does #emph[not] work* — and the
  spectrograms make this plain (@spcup). The KU Leuven recording visibly
  maneuvers, yet the refined tracks move by only ±1.3 rev/s peak-to-peak:
  starting from a #emph[constant] init, the coarse stage's ±2 rev/s window
  and the spline stage's smoothness prior simply cannot follow real flight
  dynamics. What is delivered is a #emph[mean operating point with a
  presence certificate], not a trajectory label.

So the honest use of this machinery on unlabeled data is #emph[triage]:
finding which recordings contain harmonic rotor noise, at what base speed,
with a refusal mode — useful for corpus curation (e.g. selecting external
recordings for generator-training codebooks, where near-stationary segments
with a known operating point are exactly what is wanted). It is #emph[not]
an annotation pipeline for maneuvering flight. If per-frame trajectories on
unlabeled maneuvering data are ever genuinely needed, the right instrument
is a coupled multi-order tracker of the Vold-Kalman family (next section),
initialised by this scan — not more iterations of this refiner.

#figure(
  image("assets/spcup.png", width: 100%),
  caption: [What blind "annotation" actually delivers, honestly rendered.
    #emph[Left:] the base-speed scan — KU Leuven (blue) produces a sharp
    peak (a real comb, annotated with its mean base speed); the mono
    Idea_ssu clip (red) has no peak and is #emph[refused] (confidence
    0.02). #emph[Right:] the KU Leuven spectrogram with the refined combs
    overlaid: the recording clearly maneuvers (the ridges bend), while the
    refined tracks stay nearly straight — mean base speed is recovered,
    the maneuvers are not. Detection works; trajectory tracking does not.],
) <spcup>

= Operating envelope

Three sweeps on controlled synthetics (4 rotors with realistic trim splits,
harmonics to $k = 40$), plus wall-clock cost:

- *Capture basin = the coarse grid, exactly* (@basin). Initialisation errors
  up to `delta_max` (3 rev/s default, 6 with a wider grid) are recovered
  regardless of sign pattern; beyond it, total failure. The gradient stage
  adds precision, never range — by design, and now by measurement.
- *Noise floor* (@noise_floor): white and pink noise are tolerated down to
  ≈ 0 dB harmonic SNR (error < 0.15 rev/s); speech-shaped noise, which sits
  right on the mid-frequency harmonics, already bites at +5 dB. Below the
  floor, errors grow smoothly rather than catastrophically.
- *The confidence gate works — with one blind spot* (@confidence_gate).
  Across 195 trials, gating at confidence > 0.17 keeps 95% precision at
  80% recall against
  noise-induced failures. What it #emph[cannot] catch is rotor-identity
  capture (a track locked confidently onto a #emph[different] rotor's
  comb): those failures are high-confidence by construction. A per-rotor
  uniqueness check (flag |r̂_i − r̂_j| < 0.15 rev/s) is the complementary
  gate, and the joint LSQ residual arbitrates.
- *Cost* (single core, Intel Core Ultra 7 165U): front-end + coarse capture
  are negligible (< 0.12× realtime); spline refinement runs ≈ 0.2–0.4×
  realtime mono (1.0–1.4× for 8 channels); the joint LSQ fit metric
  ≈ 1.6× realtime. Annotating an hour of audio is a coffee break, not a
  cluster job.

#figure(
  image("assets/basin.png", width: 82%),
  caption: [Capture basin = the coarse grid range, exactly. Refinement recovers
    an initial constant label error up to $delta_"max"$ (3 rev/s by default,
    6 with a wider grid) whatever the sign pattern across rotors, and fails
    completely beyond it (shaded). The gradient stage sharpens precision
    #emph[inside] the basin but never extends its #emph[range] — a
    designed-in property, here confirmed by measurement. (Curves average the
    same- and opposite-sign offset trials; the in-basin plateau below 1 is the
    fixed fraction of hard trials, not degradation.)],
) <basin>

#figure(
  image("assets/noise_floor.png", width: 82%),
  caption: [Noise floor: how much non-harmonic energy the method tolerates.
    Mean RPS error (log scale) as broadband noise is added at decreasing
    harmonic SNR. White and pink noise are tolerated down to ≈ 0 dB before the
    error crosses the 0.15 rev/s tolerance (dashed); speech-shaped noise —
    whose energy sits right on the mid-frequency harmonics the method leans on
    — bites earlier, already near tolerance at +5 dB. Below the floor the
    error grows #emph[smoothly], not catastrophically.],
) <noise_floor>

#figure(
  image("assets/confidence_gate.png", width: 90%),
  caption: [The confidence gate and its one blind spot. Each point is a trial:
    comb confidence (x) vs RPS error (y, log scale), pooled across the
    initialisation and noise sweeps. Gating at confidence > 0.171 (the
    Youden-optimal threshold, dashed) cleanly rejects the noise-induced
    failures (high error #emph[and] low confidence, upper-left). What it
    #emph[cannot] catch is #emph[identity capture] — a track locked
    confidently onto a #emph[different] rotor's comb (orange stripe): those
    are high-confidence by construction. A per-rotor uniqueness check and the
    joint LSQ residual are the complementary gates.],
) <confidence_gate>

= Discussion

A blunt ledger first. As a #emph[tool for better labels] — the original
goal — this failed: annotated corpora did not get better-fitting labels
(telemetry was already at the recoverable optimum), and unlabeled
maneuvering recordings did not get trajectories (only operating points).
What the exercise actually paid for is the #emph[audit result] and its
consequences:

+ *Label quality was not the bottleneck we thought.* DREGON command
  telemetry is bias-free to 0.04 rev/s; Michael's manual alignment holds to
  ≈ 0.1 s (stage A found only small residual offsets). The washed-out
  mid-frequency harmonics in generator training are better explained by
  #emph[jitter linewidth] — label-invisible fast speed fluctuation that
  broadens real harmonics in a way a cleanly-conditioned oscillator bank
  cannot imitate. The actionable fix is a linewidth/phase-diffusion
  parameter in the generator's emitter, increasing with harmonic index —
  and #emph[not] further label work. Testing this hypothesis cost one day;
  building generator v2 on the wrong premise would have cost weeks.
+ *Twin-paired airframes set a hard limit* on per-rotor label precision
  from audio alone: below the pair-resolution harmonic, rotors are
  acoustically one object, and magnitude-ridge methods are actively biased
  there. Any future per-rotor claim (localisation, per-rotor RPS at low k)
  has to reckon with this.
+ *What survives as tooling* is modest: the clock aligner (stage A; found
  offsets up to 0.21 s, which during maneuvers is a multi-rev/s label
  error), and the presence/refusal gates for corpus triage. A bootstrap
  loop feeding a predictor with #emph[refined trajectories] is *not*
  supported by these results — at most a loop feeding it
  #emph[near-stationary segments with verified operating points].

== Relation to existing work

This module is a home-grown instance of #emph[tacholess / computed order
tracking] from rotating-machinery diagnostics, with a telemetry prior. The
established tool for the part that defeated us — jointly tracking multiple
close, time-varying, crossing orders through maneuvers — is the
*Vold-Kalman order filter* (coupled multi-order formulation) and its
descendants; our stage D is essentially the classical phase-based tacho
correction, and stages B–C a simplistic magnitude variant that the
validation showed to be biased in exactly the regime VK was designed for.
We did not benchmark against a proper VK implementation; the honest
statement is that this report characterises a #emph[baseline-grade]
reimplementation, and any future need for blind trajectory annotation
should start from the VK literature (no maintained Python implementation
exists, which is a porting task, not a research one), initialised by our
base-speed scan and gated by our verifier. The variable-projection
structure (linear amplitudes/phases, closed-form inner solve) is standard
separable least squares and is the one piece worth keeping regardless — it
now also powers the fit metric in the generator's own basis.
