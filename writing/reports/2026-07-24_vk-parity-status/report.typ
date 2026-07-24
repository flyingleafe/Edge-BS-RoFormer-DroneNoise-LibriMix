#import "/writing/templates/typst/report.typ": report, author-meta

#show: report.with(
  title: [VK Parity Status: the Coupled Vold–Kalman Tracker, its Precision, and the Road to Neural Parity],
  authors: (
    "Harmonic Noise Suppression Project": author-meta("project"),
  ),
  affiliations: (
    "project": "Harmonic Noise Suppression Project",
  ),
  abstract: [
    We built a coupled Vold–Kalman (VK) order tracker as a precision reference
    for rotor-speed (RPS) annotation from audio alone. This report has three
    jobs. First, it teaches the algorithm at reimplementation depth: a reader
    who follows it should be able to rebuild the whole blind-annotation
    pipeline. Second, it reports the tracker's current precision against
    ground-truth telemetry on two drones, with trajectory-overlay evidence.
    Third, it documents the "neural parity" program — training an audio-only
    RPS predictor to match VK's accuracy — including two levers that did
    *not* work (test-time smoothing, longer training context) and the
    front-end redesign now in build. Bottom line: blind VK is accurate but
    slow (DREGON 0.68–0.74 rev/s pooled, FLY124 1.03 rev/s pooled, real-time
    factor below 1); the best neural predictor so far is fast but roughly
    4x coarser, and the gap has not closed with either lever tried.
  ],
  keywords: ("Vold-Kalman", "order tracking", "RPS", "rotor speed", "harmonic noise"),
)

= Problem and objective

The project needs the per-rotor rotation speed (RPS) trajectory of a drone
from its recorded audio alone, at inference time when no telemetry is
available. Telemetry itself is an imperfect label even at *training* time:
DREGON's onboard tachometer readings jitter by roughly 0.6 rev/s and are not
perfectly synchronized to the audio, so "ground truth" is itself noisy. What
we actually want is a trajectory estimate scored by how well it *explains the
recording*, independent of any telemetry channel.

The signal model: a drone with $R$ rotors produces audio

$ y(t) = sum_(a) "Re"[ "env"_a (t) dot exp(i phi.alt_a (t)) ] + "noise" $

where each harmonic $a = (r, k)$ (rotor $r$, harmonic order $k$) has its own
slowly-varying complex envelope $"env"_a (t)$ and a phase $phi.alt_a(t) = 2 pi
k integral r(tau) d tau$ set entirely by the rotor's instantaneous speed
$r(t)$. Given a candidate trajectory, we can compute a *residual*: how much
of $y$ is left unexplained after removing the best-fit harmonic
reconstruction. The VK tracker is exactly the machinery that (a) fits the
envelopes given a trajectory guess, and (b) uses the fit to correct the
trajectory — iterated to convergence. Sections 2–4 build this up from
scratch; sections 5–8 report where it stands.

#figure(
  image("assets/vk_blind_dregon_full.png", width: 88%),
  caption: [
    DREGON, blind annotation (no telemetry used at any point): top panel,
    four tracked rotor speeds (colour) against measured ground truth (black,
    jittery); bottom panel, the per-sample absolute error on a log scale.
    This is the finished pipeline (§4); the next three sections build the
    machinery that produces it.
  ],
) <fig-dregon-preview>

= Tutorial I — envelopes are a linear solve, demodulation shrinks it

*Given the frequencies, the envelopes are a least-squares problem.* Freeze
every phase trajectory $phi.alt_a(t)$ (i.e. assume the rotor speeds are known).
The signal model becomes *linear* in the unknown envelopes: at each audio
sample $t$, $y(t) = sum_a "Re"[a_a (t) c_a(t)]$ with $c_a(t) = exp(i phi.alt_a
(t))$ known. Stacking samples into a matrix $bold(C)(t)$, the envelope
estimate is ordinary variable-projection least squares with a smoothness
prior:

$ bold(a) = arg min_(bold(a)) norm(y - bold(C) bold(a))_2^2 + sum_a rho_a^2 norm(Delta^p a_a)^2 $

with $Delta^p$ a $p$-th order finite-difference operator (we use $p=2$) and
$rho_a$ a per-track regularization weight set from a target envelope
bandwidth (the Tuma relation — see Vold & Leuridan 1993, Tuma 2005). This is
the textbook second-generation VK filter.

*Why not solve it at audio rate.* A 20-second recording at 16 kHz, 4 rotors,
40 harmonics each, is $T times M approx 16"e6" times 160$ real unknowns —
solving that dense linear system per outer iteration is not tractable, and
was one of two reasons we could not simply vendor an existing VK
implementation (the other being license terms).

*The fix: demodulate, then decimate.* Envelopes are narrowband by
construction — a rotor doesn't change speed audibly fast — so instead of
solving for $a_a(t)$ at 16 kHz, demodulate the signal by each candidate
carrier and low-pass it:

$ z_a = "LP"[y dot overline(c_a)] $

using a zero-phase (filtfilt) low-pass, then decimate onto a coarse envelope
grid $f_s^"env"$ (default 100–200 Hz). $z_a$ *is* $a_a$ sampled on that grid,
up to the low-pass transfer function. Concretely, for a 20 s / 16 kHz / 4
rotor / 40 harmonic problem at $f_s^"env" = 100$ Hz: unknowns drop from
$approx 16"M"$ to $(20 dot 100) dot 160 approx 320"k"$ — solvable in seconds,
not hours. This demodulate + decimate step is the first of the three
deviations from textbook VK that make the method practical (design doc
§2.1).

= Tutorial II — the coupled banded solve and the frequency update

*Why coupling.* Two rotors spinning close in speed (DREGON's mechanical twin
pairs sit about 0.65 rev/s apart) have harmonics that land almost on top of
each other in frequency — at harmonic $k$, the frequency gap is $k$ times the
rotor-speed gap, so by the 3rd–5th harmonic the two combs are essentially
coincident. Solved *independently*, both trackers try to claim the same
spectral energy and both converge toward the pair's mean speed ("twin
capture" — exactly the failure mode of the predecessor stage-B/C tracker;
see design doc's post-mortem). The fix is to solve them *jointly*: whenever
two tracks' instantaneous frequencies come within the envelope bandwidth
$B_"env"$ of each other, couple them in one linear system so the least-squares
fit has to split the shared energy between them (explaining-away) instead of
letting each one grab all of it.

#figure(
  image("assets/vk_coupling_schematic.png", width: 82%),
  caption: [
    A twin pair's harmonics (red, blue) overlap almost completely in the
    observed spectrum (grey fill) by the 3rd–5th harmonic. An uncoupled
    per-track solve cannot tell them apart; the coupled solve treats the
    overlap as shared energy that must be split.
  ],
)

*The coupled system.* For a group of tracks that mutually satisfy $|f_a(t) -
f_b(t)| < B_"env"$ at some point in time, the normal equations of the joint
least-squares problem form a block system

$ bold(G)_(a,b) = cases(
  bold(C)_a^* bold(C)_a & a = b,
  bold(C)_a^* bold(C)_b & "coupled",
  0 & "otherwise",
) $

Using a *time-major* interleave of the unknowns (index $= t dot g + a$, where
$g$ is the group size), $bold(G)$ is Hermitian positive-definite and
*banded*: the smoothness prior contributes bands at offsets $g$ and $2g$
(neighbouring time samples of the same track), the coupling term contributes
bands at offsets $1 .. g-1$ (same time sample, different track), and the
bandwidth is exactly $p dot g$ — no fill-in. This is assembled directly in
LAPACK banded storage and factorized with a banded Cholesky
(`cholesky_banded` / `cho_solve_banded`), 5–10x faster than the naive sparse
`splu` reference path (see §6). Groups are found by union-find on the
coupling predicate and re-computed every outer iteration; singleton groups
(no nearby track) fall back to plain single-track VK.

*Frequency update.* VK gives envelopes assuming frequencies are known — it
doesn't track frequency on its own. We alternate an envelope solve with a
frequency correction read off the fitted envelope's phase. If the assumed
frequency for harmonic $k$ is off by $delta$ (rev/s), the demodulated
envelope $x_(r,k)(t)$ rotates at a residual rate proportional to $k dot
delta$ — so the phase slope between consecutive envelope samples estimates
the error, and dividing by $k$ removes the harmonic's amplification:

$ hat(delta)_(r,k)(t) = angle(x_(r,k)(t{+}1) dot overline(x_(r,k)(t))) dot
  frac(f_s^"env", 2 pi k) $

Each harmonic gives its own noisy estimate of the same trajectory error;
these are fused with Fisher weights $w_(r,k) = k^2 |x_(r,k)|^2$ (a harmonic
that carries more energy and a higher $k$ is a more informative — but also
more easily aliased — witness) into one smoothness-regularized 1-D solve per
rotor:

$ min_delta sum_(k,t) w_(r,k)(t) (hat(delta)_(r,k)(t) - delta(t))^2 + lambda norm(Delta^2 delta)^2 $

a tridiagonal-plus-banded solve at per-sample resolution (no splines). The
correction is clipped to $|delta| lt.eq$ 0.5 rev/s per iteration and added to
the trajectory; then the demodulation carriers are recomputed and the loop
repeats.

*Capture then refine — annealing $k_max$.* Early in the loop, before the
tracker has locked on, only low harmonics are safe to use: high-$k$ harmonics
amplify a large initial frequency error $delta_0$ into aliasing once $k dot
delta_0$ exceeds half the envelope sample rate. So the outer loop runs in two
coarseness regimes: *capture* (few harmonics $k = 6..12$, wide envelope
bandwidth $B_"env" approx 7$ Hz — a large, forgiving basin of attraction) then
*refine* (many harmonics $k = 6..30$, narrow bandwidth $approx 1.5$ Hz — high
precision once locked). $k_max$ grows as the track's residual improves. Each
outer iteration's convergence is monitored by the joint residual ratio
$norm(y - hat(y))^2 slash norm(y)^2$ (the reconstruction-quality number used
throughout this report) plus the size of the last frequency update.

= Tutorial III — blind annotation end-to-end

Everything above assumes an initial trajectory guess to iterate from. At
inference there is no telemetry to seed from, so the pipeline needs a
*blind seeding* stage before VK can run at all. The full pipeline, in
execution order:

```python
def blind_annotate(audio[M, T], fs, R=4) -> r[R, T_env], confidence:
    # -- steps 1-2: whiten + scan (once) ------------------------------------
    W       = whiten(mean_spectrogram(audio))            # peaks vs local floor
    cands   = [(f0, mean(W[k*f0] for k*f0 <= 1.2kHz))     # comb score
               for f0 in grid(30..120, step 0.05)]
    cands   = alias_filter(dedup(cands, min_sep=4))       # drop subsets/ratios
    # -- step 3: seed R rotors ------------------------------------------------
    seeds   = top_R_distinct(cands)
    if len(seeds) < R:
        seeds += rescan(W minus teeth(seeds))             # residual re-scan (arm R)
        seeds += duplicate(lowest_scoring(seeds), plus.minus 0.1)  # merged-twin fallback
    r       = constant_tracks(seeds)
    # -- steps 4-5: capture then refine (same loop, two coarseness levels) --
    for cfg in [CAPTURE(bw=7Hz, k=6..12), REFINE(bw=1.5Hz, k=6..30)]:
        for it in 1..N_outer:
            z[r,k,m] = decimate(LP(audio * exp(-i*2*pi*k*integral(r))))  # demodulate
            a        = banded_cholesky(coupled_normal_eqs(z))            # envelopes
            delta[r] = sum_k fisher_w(k,z) * phase_slope(a[r,k])         # freq error
            r       += clip(delta, max_step);  k_max = anneal(it)
        if stage_destroys_track(r, prev_r, W):   # comb-occupancy drop
            r = prev_r                            # per-track stage guard: revert
    # -- step 6: twin resolution across windows -----------------------------
    r = spatial_DP(windowed_candidates(r), mic_weights(geometry))
    return r, comb_confidence(r, W)   # low confidence => stay at init
```

*Whiten (step 1).* A rotor's harmonics are peaks in the spectrum, but their
absolute height depends on broadband tilt and room colouration we don't care
about. Whitening divides a smoothed spectral envelope out of the mean
spectrogram, $W(f) = macron(S)(f) slash "smooth"(macron(S))(f)$, so a comb
becomes a row of *equal-height* teeth regardless of overall spectral shape.

*Scan (step 2), band-capped.* For each candidate fundamental $f_0$ in a
plausible range, score it by the mean whitened tooth height, restricted to
harmonics landing under 1.2 kHz:

$ S(f_0) = "mean"_(k : k f_0 lt.eq "1.2 kHz")\ W(k f_0) $

The band cap turned out to matter more than anything else in this section:
without it, a small $f_0$ packs *all* of its many teeth into the loud
low-frequency region and can outscore a real, higher-frequency comb whose
harmonics dilute into the quieter high band. On FLY124 this uncapped-scan bug
was the single largest source of blind-annotation error (§5).

#figure(
  image("assets/vk_alias_illustration.png", width: 82%),
  caption: [
    The concrete alias that misled FLY124's blind seeding before the band
    cap: $60.7 approx (2/3) times 91$. Every 3rd tooth of the alias comb
    (teeth $k = 3, 6, 9, ...$) coincides with an even harmonic of the true
    91 rev/s rotor, so the alias steals real energy and can out-score a
    weak true comb in an uncapped scan.
  ],
) <fig-alias>

*Seed (step 3).* Take the top-$R$ distinct scan peaks (deduplicated at 4
rev/s — closer peaks are the same rotor). If fewer than $R$ bases are found,
the *residual re-scan* (arm R) masks the teeth already claimed and re-scans
what's left — a weak rotor hidden behind a louder neighbour becomes the
dominant comb in the residual. If still short, the lowest-scoring used base
is duplicated with a small offset ($plus.minus 0.1$ rev/s); a genuinely
merged twin pair spreads its energy across two nearly-identical combs and
under-scores as a single candidate, so this fallback gives the coupled solve
(§Tutorial II) two starting points to split apart.

*Capture, refine (steps 4–5).* The Tutorial-II loop, run twice: coarse then
tight, as described there.

*Per-track stage guard.* A refinement stage can occasionally re-capture a
weak track onto a stronger neighbouring comb — this is a *failure*, but the
raw confidence score actually *rises* when it happens (the tracker is now
sitting on louder energy). The guard instead watches comb *occupancy*
(fraction of expected teeth present) and reverts any stage that drops it for
a given track, using the pre-stage trajectory instead. This directly fixed
FLY124's worst failure mode (§5).

*Twin resolution (step 6).* Spatial evidence (multi-mic amplitude ratios
consistent with rotor geometry) resolves the remaining ambiguity between
near-identical twin tracks across windows via dynamic programming
(spatial-DP).

= Precision results vs ground truth

#figure(
  table(
    columns: 4,
    align: (left, center, center, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*task (blind unless noted)*], [*pooled err (rev/s)*], [*twins*], [*reading*]),
    [DREGON refine, telemetry-init], [0.604], [—], [vs. 0.609 without VK refinement — at the label jitter floor],
    [DREGON nosource, blind], [*0.680*], [resolved], [all 4 rotors, no telemetry],
    [DREGON speech-low, blind], [*0.701*], [resolved], [speech interference],
    [DREGON whitenoise, blind], [*0.744*], [not resolved], [auto-knobs arm (K) wins this recording],
    [FLY124 cruise, blind], [*1.027*], [resolved], [per-rotor 0.67 / 1.19 / 1.22 / 1.03; capture 4/4],
  ),
  caption: [
    Precision reference numbers (`results/vk_eval/vk_valid_comparison.csv`,
    `results/vk_blind_sweep_r{4,5,6}/sweep_report.csv`). "err" is pooled
    unsigned error against measured/smoothed telemetry, PIT-aligned.
  ],
) <tab-precision>

Telemetry-init refinement sits almost exactly at the label jitter floor
(0.604 vs. the raw command trajectory's 0.609) — VK refinement doesn't hurt,
and there isn't much more accuracy to extract from telemetry-init on DREGON
specifically, because the telemetry is already close to true.

Blind annotation (no telemetry at all, @fig-dregon-preview and below) is the
number that matters for inference, and it is close to the refine number:
0.680–0.744 rev/s pooled on DREGON, twin pairs resolved on 2 of 3
recordings.

#figure(
  image("assets/vk_blind_fly124.png", width: 84%),
  caption: [
    FLY124-cruise blind annotation (current pipeline, arm R) against ground
    truth telemetry (dashed). All four rotors captured, including the twin
    pair near 74–76 rev/s.
  ],
) <fig-fly124>

*The FLY124 evolution story.* FLY124 pooled error moved 4.0 -> 3.24 ->
1.027 rev/s across three fixes, each targeting a distinct failure:

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*fix*], [*pooled err after*], [*what it fixed*]),
    [(none) plain band-uncapped scan], [4.0], [baseline blind seeding],
    [band-capped scan (`scan_f_max=1200`)], [3.24], [the alias bug: uncapped scan let a
      low-$f_0$ alias comb outscore the true comb (@fig-alias) — this was a scan-band
      *bias*, not an SNR floor],
    [\+ residual re-scan (arm R)], [1.03 (best stage)], [recovers a weak rotor (82.7
      rev/s) that a strong neighbour was shadowing in the first-pass scan],
    [\+ per-track stage guard], [*1.027 (final)*], [without the guard, refinement
      re-captured the recovered weak rotor back onto the strong 91 Hz comb
      (raw confidence went *up* during this failure — occupancy is the
      correct signal, not confidence)],
  ),
  caption: [FLY124 fix-by-fix evolution (design doc §7.5).],
)

The "all four rotors under 2 rev/s" success bar is now cleared on FLY124; the
baseline arm's persistent 3.24 rev/s error traces entirely to its 4th rotor
never being seeded in the first place (a guard cannot repair a seed that was
never made) — arm R is therefore load-bearing, and arm K (auto-knobs) is
adopted as a free, drone-agnostic win with zero accuracy cost. Two arms from
the original design (T: shared-template matched filter, C: alias/completeness
rejection, N: count-prior duplicate seeding) were tried and refuted — no
composition beat the band-capped scan baseline.

= Fast inference

Profiling (`scripts/vk_bench.py`) found the CPU path spending 54–58% of wall
time in the coupled-group normal-equation solve (`splu`) and 29% in
full-length demodulation FFTs. Two algorithmic changes, not more threads
(no measurable thread scaling was found), bought roughly 10x:

1. *Banded Hermitian solver.* With the time-major interleave from Tutorial
   II, the coupled system is exactly banded with zero fill-in; assembling it
   directly into LAPACK banded storage and factorizing with
   `cholesky_banded`/`cho_solve_banded` agrees with the reference `splu`
   solve to $approx 10^(-8)$ relative and is 2.9x faster.
2. *Far-pair coupling pruning.* Track pairs whose instantaneous frequency
   separation never drops close to the demodulation cutoff can only pick up
   negligible spectral leakage ($approx 10^(-3)$) and are skipped; 1.7x on
   top of (1), with no measurable trajectory change ($< 10^(-8)$ rev/s RMS in
   synthetic A/B).

#figure(
  image("assets/vk_speedup_bars.png", width: 78%),
  caption: [
    Real-time factor before/after the fast paths. Refine (telemetry-init)
    reaches 0.36–0.40x real time; blind reaches 0.95x — close to real time,
    still not under it.
  ],
) <fig-speedup>

*Rejected alternative: FIR polyphase decimation.* A two-stage linear-phase
Kaiser decimator was measured *slower* than the batched pocketfft brickwall
filter used by default (4.3–5.6 ms vs 3.3 ms per 20 s signal — scipy's
`upfirdn` can't beat pocketfft's SIMD path even at a fraction of the nominal
multiply count), and its transition band perturbs blind-capture trajectories
by $approx 2.5 times 10^(-3)$ rev/s RMS versus the brickwall filter. Kept
behind a flag for A/B only, not adopted.

*In build, not yet measured:* a GPU torch adapter using a
block-tridiagonal reformulation of the banded solve. All numbers above are
CPU-path, regression-gated to within $10^(-3)$ rev/s of the recorded
reference trajectories.

= Neural parity program: done, refuted, and in build

The parity criterion is precise: an audio-only neural RPS predictor, trained
on real data plus augmentation, evaluated against blind VK on the *same*
clips, using the same per-clip PIT-MAE protocol
(`results/vk_eval/vk_valid_comparison.csv`, `scripts/rps_predictor_vk_eval.py`).
Blind VK's numbers from @tab-precision (0.68–0.74 DREGON, 1.03 FLY124) are
the bar.

#figure(
  image("assets/vk_parity_bars.png", width: 84%),
  caption: [
    Pooled PIT-MAE, blind VK vs. the two neural levers tried. Neither phase
    closes the gap; phase B (longer context) is worse than phase A
    (smoothing) on DREGON.
  ],
) <fig-parity>

*Phase A — test-time smoothing (no training).* The E12 real-full-flight
transformer checkpoint, evaluated with sliding-window stitching plus
2–20 s moving-average / running-median aggregation on top of its raw
per-window output. Smoothing helps — best DREGON cruise 2.62 rev/s (from a
raw 3.19), best FLY124 1.55 rev/s (already under the 3.24 pre-fix blind-VK
bar without any smoothing) — but *saturates* well short of blind VK's 0.68–0.74
DREGON floor: past roughly a 10 s window, more smoothing makes things worse
(mixing distinct flight regimes). The residual error is *systematic within a
window*, not zero-mean jitter that averaging would remove — smoothing was
never going to be a full fix, and the sweep confirms it.

*Phase B — longer native training context (refuted).* Hypothesis: E12 was
trained on 1 s chunks but evaluated on 8 s clips, while VK integrates over
the whole trajectory — maybe giving the model native long context helps.
Same recipe, `duration_s` set to 4 and 8 (batch size scaled down to fit a
T4/P100). Result: best DREGON 2.87 rev/s, best FLY124 1.90 rev/s — *worse*
than phase A on both pools. Native context length is refuted as a parity
lever. A secondary finding: the `last`-epoch checkpoints of both new arms
degrade sharply relative to `best` (e.g. up to 8.27 rev/s raw, DREGON,
8s/ch0/last) — early stopping matters more than context length here.

*In build now, no numbers yet:* the working hypothesis is
that the front-end is the bottleneck, not training length or smoothing. The
current magnitude-STFT front-end has no built-in harmonic aggregation and no
sub-bin frequency precision — exactly the two properties the VK tracker's
whitened-comb scan and phase-slope update rely on. Two arms:

- *G2a — harmonic-stacked HCQT front-end*, giving the model per-harmonic
  channels analogous to VK's per-track envelopes.
- *G2b — instantaneous-frequency phase channels*, giving the model
  sub-bin frequency information analogous to VK's phase-slope update.

*Parked, not being pursued right now:* VK-distilled training labels, and
using VK to annotate otherwise-unlabeled data for training. Both are
plausible but orthogonal to the front-end hypothesis and are deferred until
G2 either closes the gap or is itself refuted.

= Verdict and roadmap

Blind VK is the accuracy reference and campaign criterion 2.2 (exhaust
non-learning accuracy levers) is closed: every arm in the original design
(T, C, N, K) plus three additions found during the sweeps (band-capped scan,
residual re-scan, per-track stage guard) has been measured, and each is
either adopted or explicitly refuted with a number attached (§5). Blind VK
reaches 0.68–0.74 rev/s pooled on DREGON and 1.03 rev/s pooled on FLY124, at
just under real time (0.95x rtf, §6).

Neural parity (criterion 2.3) is *open*, with a quantified and now
better-understood gap: the best neural predictor so far reaches 2.62 vs.
0.68–0.74 rev/s on DREGON, 1.55 vs. 1.03 rev/s on FLY124 — and the other
lever tried (longer native context) is worse still (2.87 / 1.90). Two
training-side levers (test-time smoothing, longer
native context) have been tried and neither closes it — smoothing saturates
because the error is systematic, not jitter; longer context actively hurts.
The next step in build (G2: harmonic-stacked HCQT + instantaneous-frequency
phase channels) targets the front-end directly, on the hypothesis that the
model currently lacks the two structural ingredients — harmonic aggregation,
sub-bin frequency resolution — that make VK itself accurate. No results yet;
this is the open item to report on next.
