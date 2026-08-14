# VK maximum-likelihood decomposition of real drone audio

**Date**: 2026-08-12 · **Branch**: `tracking-opt` · **Status**: instrument
built + full-recording decomposition DONE; amplitude-target training is the
follow-up. Motivated by the MSSTFT erosion verdict
(`generator-perrotor-dynamics.md`): the audio-domain magnitude loss cannot
defend mid/high-k lines because real lines decohere within the loss's STFT
windows — so decompose the audio ONCE, correctly, and train on the
decomposition instead.

## The decomposition

`scripts/vk_decompose.py`: windowed coupled Vold-Kalman solve
(`tracking.fitness_vk.solve_envelopes`) on REFINED labels →

    x_mic(t) = Σ_{rotor,k} Re[ env_{rotor,k,mic}(t) · e^{j·k·φ_rotor(t)} ] + residual_mic(t)

per-track amplitude = |env|, per-timestamp PHASE ERROR = arg(env), initial
phase = arg(env(t0)); the sum is EXACT by construction (measured resynthesis
error 6e-8). The ML content is the VK solve: penalized LS = MAP under a
Gaussian residual + per-track bandwidth prior. Broadband floors are per MIC
(per-rotor attribution of the residual needs cross-mic structure — the
wind-channel lesson — and stays out of scope here).

Full run (`vk-decompose-6bdede` + remote stitch): free-flight_nosource_room1,
64 s, 8 mics, k_hi 62 (f_max 6 kHz), 248 tracks, fs_env 100 Hz. Artifacts on
R2 `artifacts/vk-decompose/free-flight_nosource_room1/` (envelopes.npz 103 MB,
residual.npz, report.json, bw_sweep.json).

## Findings

1. **Energy ledger**: tracks 28.2 %, residual 67.8 %, cross 4 %. Track energy
   is k1-9-dominated (97.5 %); k10-24 1.6 %, k25-49 0.7 %, k50-80 0.3 % — the
   quantified reason MSSTFT never defends high-k lines.
2. **Phase model — the pi-kalman independence assumption WINS.** Rank-one
   share of phase-error increments across k within a rotor: λ1/Σλ =
   0.027-0.028 (chance for 62 tracks ≈ 0.016), mean pairwise correlation ≈ 0,
   all four rotors. The shaft-wander (rank-1) model is refuted at track level;
   per-harmonic independent drift is the right prior. (Smoke at K=20 gave
   0.19 vs noise bar 0.075 — a weak common component exists but is tiny.)
3. **Drift statistics are prior-limited for weak tracks.** Measured drift std
   is flat with k (4.4-5.1 rad/s in every band) — but the ρ sweep shows drift
   scales directly with the achieved bandwidth (2.5 / 4.9 / 9.5 rad/s at
   0.5 / 1 / 2 Hz), and the per-frame unwrap step already touches π at
   fs_env=100 Hz. So the flat curve reflects the 1 Hz clamp, not physics; the
   dense comb floors the per-group bandwidth clamp at 1 Hz regardless of
   bw_rps (only rho_scale moves it).
4. **Amplitude estimates of weak bands absorb floor noise ∝ bandwidth**
   (k10-24 mean amp 3.2e-4 / 4.5e-4 / 6.2e-4 at 0.5 / 1 / 2 Hz). Targets
   derived from these envelopes must either fix one bandwidth everywhere
   (consistent bias) or debias via the noise-equivalent bandwidth
   (`Envelopes.bw_track` exists for exactly this).

## Consequence: the amplitude-target training path

The decomposition gives per-(rotor, k, mic, t) amplitude envelopes and
per-mic residual PSDs — direct supervision targets for the generator's
`harm_amps` and noise branch. A loss on these targets never synthesizes
audio, so phase decoherence exits the training problem, and high-k amplitude
information survives (VK demodulation is comb-coherent, unlike the fixed
STFT). Design cautions carried over: debias weak-band amps (finding 4);
supervise log-amplitudes; keep the OU jitter for rendering only.

Gotchas: omnirun output collection silently drops ~25 MB files — upload big
artifacts to R2 from inside the job; the coupled group chains ~all tracks
into one banded system (memory ≈ 1e-4·k²·window_s GB/worker — `group_plan`
forecasts it, `--mem-budget-gb` guards it; this is what OOM-killed a laptop).

## v2: linewidth-matched bandwidth

**Date**: 2026-08-12 · **Status**: built + tuned + verified on one window;
the full re-decomposition is the ready command at the end. Instrument:
`--bw-schedule`, `--sr`, `--f-max` on `scripts/vk_decompose.py`;
`tracking.decompose.BandwidthSchedule`.

### The diagnosis: v1 solved every track with 1 Hz

`VKConfig.bw_rps` asks for a k-scaled band, but a coupled group clamps every
track to `max(bw_hz, 6 · minimum pair separation)`, and the DREGON comb is
dense enough to floor that at `bw_hz` = 1 Hz. So v1 filtered harmonic 40 with
the same 1 Hz passband as harmonic 1, while the shaft jitter widens harmonic
`k` by `k` times the rate error. The comb therefore LEAKED into the residual.

Measured with the order-contrast probe (2048/512 Hann spectrogram, power
averaged over mics, each frame interpolated onto the order grid `f / f0_r(t)`
per rotor, contrast = 10·log10(mean on-order ±0.06 / mean half-order ±0.06)),
on three 4 s windows of `free-flight_nosource_room1` (t0 = 12/30/50 s, 2 mics,
k_hi 40, 16 kHz). Residual contrast in decibels — **zero is the target in
every band**; positive is comb leak, negative is over-subtraction:

| arm | t12 k1-9/k10-24/k25-40 | t30 | t50 | Σ\|·\| |
|---|---|---|---|---|
| original audio | 11.07 / 1.08 / 0.24 | 7.97 / 2.34 / 0.24 | 7.34 / 1.36 / 0.18 | 31.81 |
| v1 flat 1 Hz | 2.44 / 0.68 / 0.00 | 2.19 / 1.68 / −0.06 | 0.96 / 0.90 / −0.10 | 9.00 |
| **v2 `3,0,1.5,3`** | 1.30 / −0.02 / −0.51 | 2.11 / 0.72 / −0.60 | −0.02 / 0.06 / −0.60 | **5.93** |

### The tuned schedule

    --bw-schedule 3,0,1.5,3
    bw_k = clip(3.0 + 0.0·k, base, min(1.5 · line separation, 3.0)) Hz

Achieved bands (`Envelopes.bw_track`, recorded in `report.json` and
`envelopes.npz`): 1.9 Hz at k1-9, 2.9 Hz at k10-24 and k25-40.

Three findings the grid produced, none of them the expected one:

1. **The slope tunes to ZERO.** The declared grid (72 arms of `bw0 ∈ {1,2,3}
   × slope ∈ {0.05…0.2} × capfrac ∈ {0.6,0.9,1.2} × absmax ∈ {6,10}`) was
   screened first, then three refinement stages — 103 distinct arms, 149
   solves in all. The k-shaping the linewidth law wants is already supplied
   by the SEPARATION CAP: at high k the four rotors' lines interleave, so
   `1.5 · sep` is itself a function of k (1.9 → 2.9 Hz here). An explicit
   slope on top of that over-widens. `slope_hz_per_k` is kept in the API
   because a SPARSE comb (one rotor, or Michael's two) never triggers the cap
   and would need the law itself.
2. **Window 30 alone is a trap.** Screening on one window picked
   `3,0.2,1.5,6` (k10-24 leak 1.68 → −0.01). On all three windows that arm
   scores 9.00 — a tie with flat — because windows 12 and 50 leak far less
   under flat (0.68, 0.90) and the wide band over-subtracts them (−0.82,
   −0.92). Three windows, then rank.
3. **Over-subtraction is the binding constraint, and it is symmetric.**
   Both bands respond to the achieved bandwidth with the same slope
   (≈ −0.28 dB/Hz), and they are offset by ≈ 1.7 dB, so widening trades leak
   at k10-24 for notching at k25-40 almost one for one. Arms that score
   marginally better on Σ\|·\| (`3,0,4,3` → 5.34) do it by driving k1-9 to
   −1.33: they eat the low band instead of the comb. `3,0,1.5,3` is the best
   arm that leaves no band's sign flipped.

### The low-k stripes are NOT comb leak

k1-9 residual contrast barely moves with bandwidth (2.19 → 2.11 across 72
arms on window 30) and stays large under every schedule, so it is not a band
the schedule can close. `residual_tones` says why. Run on the tuned residual
of the three windows (one 4 s segment each), the top ten peaks split into two
populations by `order_dist`, and only the second one is foreign:

| f (Hz) | prom | order at t12 | at t30 | at t50 | reading |
|---|---|---|---|---|---|
| 1554-1565 | 15-16 dB | 21.083 | 20.995 | — | near-integer: comb leak |
| 1178-1213 | 10-13 dB | 14.032 | 14.050 | 13.967 | near-integer: comb leak |
| **488.3 / 488.3 / 490.2** | 9-10 dB | **5.751** | **5.703** | **5.814** | **foreign** |
| 43.0 | 16 dB | 0.579 | — | — | foreign |
| 634.8 | 9 dB | — | — | 8.350 | foreign |

The 489 Hz line is the clean case: it holds its FREQUENCY to 0.4 % over 38 s
while its ORDER moves 1.9 % with the rotor speed, and it sits a quarter of an
order away from any harmonic. That is a structural or aerodynamic resonance,
not a comb line, and a smooth-PSD noise model cannot represent it. Hence the
measurement: `report.json → residual_tones` (per ~8 s segment, top ten peaks
below 2 kHz, Welch 8192, prominence ≥ 6 dB, each with its frequency,
prominence and distance to the nearest rotor order). **Measurement only —
nothing removes them**, and `order_dist` is the field that separates leftover
comb from foreign tone.

### 32 kHz / k 80: the sample rate was the real harmonic cap

`f_max` was never the binding ceiling. The geometry also holds every line
under `0.375 · sr`, so at 16 kHz the cap is 6 kHz whatever `--f-max` says:
`free-flight_nosource_room1` peaks at 91.56 rev/s and caps at **k_hi 62** at
16 kHz for both `--f-max 6000` and `--f-max 8000`. At `--sr 32000 --f-max
8000` it reaches the requested **k_hi 80** (320 tracks).

One verification solve, 2 s window at t0 = 30 s, 2 mics, k_hi 80, 32 kHz
(a 4 s window at k 80 does not fit the 6 GB local cap):

| arm | k1-9 | k10-24 | k25-49 | k50-80 | Σ\|·\| | residual frac |
|---|---|---|---|---|---|---|
| original | 7.55 | 1.94 | 0.12 | −0.25 | — | 1.000 |
| flat 1 Hz | 1.16 | 1.40 | −0.07 | −0.37 | 3.01 | 0.448 |
| v2 `3,0,1.5,3` | 0.73 | 0.63 | −0.44 | −0.53 | **2.33** | 0.410 |

Achieved bands 2.32 / 2.97 / 2.94 / 2.75 Hz. The schedule tuned at k ≤ 40 and
16 kHz transfers to k 80 at 32 kHz without retuning.

**Memory**: at k_hi 80 the four rotors put all 320 tracks in ONE coupling
group, so a 16 s window needs 10.50 GB per worker and a 12 s window 7.88 GB.
v2 therefore runs **12 s windows at a 9 s hop** (the v1 0.75 overlap), not
16/12.

### The ready re-decomposition

Note that only ONE recording currently survives the loader:
`src/data_processing/refined_labels/` holds a refined-label sidecar for
`free-flight_nosource_room1` alone, and the decomposition is conditioned on
those labels. The command below takes every surviving recording, so it widens
by itself as sidecars are added.

The upload arguments travel through `sys.argv`, so the snippet carries no
quote or `$` that the two shell levels would have to escape (a heredoc does
not survive `bash -c '...'` — its terminator never matches):

    omnirun submit --backend uni-cpu --gpus 0 --cpus 4 --mem 32 --time 6h \
      --name vk-decompose-v2 --outputs "results/vk_decompose_v2/**" \
      --env PYTHONPATH=src -- \
      bash -c "set -e
    python scripts/vk_decompose.py --mode all --jobs 2 \
      --out results/vk_decompose_v2 \
      --sr 32000 --f-max 8000 --k-max 80 \
      --window-s 12 --hop-s 9 --mem-budget-gb 9 \
      --bw-schedule 3,0,1.5,3
    python -c 'import sys, pathlib
    from training.artifacts import ArtifactStore
    store = ArtifactStore(experiment_name=sys.argv[1])
    root = pathlib.Path(sys.argv[2])
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != sys.argv[3]):
        for f in sorted(d.iterdir()):
            print(store.upload_file(f, d.name + sys.argv[4] + f.name), flush=True)' \
      vk-decompose-v2 results/vk_decompose_v2 raw /
    "

`ArtifactStore(experiment_name="vk-decompose-v2").upload_file(f, "<rid>/<name>")`
writes `r2://ml-data/artifacts/vk-decompose-v2/<recording_id>/<name>` — its key
root is `<prefix>/<experiment_name>`. The in-job upload is not optional:
omnirun output collection silently drops files above ~25 MB, and
`envelopes.npz` at 8 mics / 320 tracks / 64 s is about 260 MB.

The daemon was down when this was written, so the command is prepared and NOT
submitted.

### Full-recording v2 verification (DREGON, 2026-08-13)

The full-dataset v2 run (job `vk-decompose-v2-ddaf81`) completed all 7
windows of `free-flight_nosource_room1`: 320 tracks, k_hi 80, resynthesis
max error 6.0e-8, energy split 35.0 % tracks / 60.0 % residual / 5.0 %
cross-term, band shares of track energy 93.3 / 3.4 / 1.7 / 1.6 %.

The order-contrast probe on the FULL recording (all 8 mics, 64 s) confirms
the de-striping — v2 residual holds no positive comb contrast in any band:

| signal | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|
| original audio | 4.60 | 1.10 | 0.10 | 0.10 |
| v1 residual (16 kHz, k≤62) | 0.67 | 0.69 | — | — |
| **v2 residual** | **−0.29** | **+0.01** | **−0.42** | **−0.19** |

The small negative values are the known mild over-subtraction of the tuned
schedule (−0.28 dB/Hz slope, section above); the k1-9 magnitude also carries
the foreign tones. Gotcha for consumers: `vk_decompose.get_recording` takes
`sr=` as a DEFAULT-VALUE parameter bound at import — setting the module
global `SR` after import does nothing and silently yields 16 kHz audio
against 32 kHz sample spans.

Michael's run (`vk-decompose-v2-m`) finished 30/32 windows; two FLY125 units
died in banded factorization ("Not enough memory") under `--jobs 2` and are
re-running as a gridrun resume with `--jobs 1` (job `vk-decompose-v2-m-fix`),
followed by the same stitch + R2 upload.

### The two "OOM" units were a PD failure, not a memory shortage

`FLY124__f003091` and `FLY125__f005178` failed identically at 24 GB and at
48 GB. The message "Not enough memory to perform factorization" is
SuperLU's: on these two windows the banded Cholesky loses positive
definiteness to decimation rounding, and the splu REFERENCE path it falls
back to needs fill-in memory no node has. Fix in
`tracking.vk_tracking`: two PD-repair retries with a relative diagonal
inflation (1e-6, then 1e-4) before the splu fallback; the default path
multiplies the diagonal by exactly 1.0 and stays bit-identical.

### Full-recording v2 verification (Michael's, 2026-08-13)

With the PD-repair fix the two failing units solved on the first retry
(`diag_scale` 1+1e-6) and the full runs completed: FLY124 12/14 windows
(k_hi 76, tracks 56.9 % / residual 38.6 %), FLY125 20/20 (k_hi 75, 60.3 %
/ 35.2 %), resynthesis max error 3e-8. Residual order contrast (probe on
all mics, full recordings):

| signal | k1-9 | k10-24 | k25-49 | k50-80 |
|---|---|---|---|---|
| FLY124 original | 0.72 | 1.49 | 0.23 | 0.03 |
| FLY124 v2 residual | 0.58 | −0.17 | −0.67 | −0.57 |
| FLY125 original | 1.64 | 1.83 | 0.21 | 0.02 |
| FLY125 v2 residual | 0.94 | 0.19 | −0.83 | −0.66 |

The comb is gone above k10; the somewhat stronger over-subtraction at
k25-80 (vs DREGON's −0.2 to −0.4) is the sparse-comb regime — two twin
pairs rarely trigger the separation cap, so the achieved bands sit at the
3 Hz absmax. Both recordings' `{envelopes,residual}.npz + report.json`
are on R2 under `artifacts/vk-decompose-v2/`.

## v3: the JOINT decomposition

**Date**: 2026-08-14 · **Branch**: `vk-v3` · **Status**: built + tested +
smoked; the three full-recording jobs are in flight. Design and every measured
number: [`docs/vk-decompose-v3-design.md`](../vk-decompose-v3-design.md).
Module: `src/tracking/joint_decompose.py`; driver `scripts/vk_decompose.py
--joint`.

### What v2 could not do

v2 pushes EVERY timing deviation through one envelope band and takes the
leftover to be white. A shaft that wanders about 0.6 rev/s makes harmonic `k`
about `0.6·k` Hz wide, so from about `k` 5 up the flanks of every line become
"residual" by construction, and an unweighted misfit is tolerant of comb
structure exactly where the floor is loud. v3 alternates three
linear-Gaussian blocks: a whitened VK solve with the shaft correction folded
into the CARRIER, a `k`-weighted phase split (rig-common shaft, per-rotor
shaft, per-track remainder), and a smooth floor fitted BETWEEN the masked comb
lines.

### The instrument, corrected

`depth_db` (folded cell peak over cell median) is a RATIO, and it fails in two
ways this campaign now has measured: it can rise while the comb falls, because
the floor it is measured against falls faster; and on a four-rotor rig the
other rotors' lines put a floor under it. On the synthetic fixture the ORIGINAL
audio reads 2.54 dB at k10-24 and an almost perfect decomposition reads
1.62 dB — no discrimination at all. `order_cell_profile` therefore also
returns **`excess_db`**, the summed absolute `peak − median` over the band's
cells in the input's own power units. That number is comparable between two
signals, so "original minus residual" is decibels of comb removed. Read it
first. The narrow slot contrast stays retired.

### Measured on the synthetic fixture (20 s, 16 kHz, 4 rotors, 3 mics, k ≤ 20)

| arm | residual fraction | k1-9 depth / excess dB | k10-24 depth / excess dB |
|---|---|---|---|
| original audio | 1.0000 | 28.17 / 62.03 | 2.54 / 35.93 |
| v2 (flat carrier) | 0.0551 | 6.09 / 43.65 | 2.02 / 34.18 |
| **v3, 3 rounds** | **0.0025** | **1.21 / 24.86** | **1.38 / 20.24** |
| oracle (true shaft folded in) | 0.0024 | 0.89 / 22.90 | 1.62 / 19.85 |

v2 removes 18.4 dB of comb excess at k1-9 and **1.75 dB** at k10-24. v3 removes
37.2 dB and 15.7 dB, and lands inside 2 dB of the oracle in both bands. The
recovered shaft phase correlates 1.000 with the truth, and the fitted log floor
is within 0.5 to 1.0 dB rms of truth away from the lines.

### Three findings that changed the design

1. **The annealing ladder is limited by the envelope BAND, not by the unwrap.**
   Harmonic `k` of a shaft wandering `sigma_r` rev/s is a frequency modulation
   of about `k·sigma_r` Hz, and a band of `B` Hz distorts its phase once
   `k·sigma_r` is more than `B/2` — at 0.6 rev/s and 3 Hz that is `k` 2.5. A
   ladder that starts at `k` 6 recovers 43 % of the true shaft phase in three
   rounds; a ladder that starts at 3 recovers all of it. Default `3,12,80`.
2. **Whitening must be bandwidth-neutral.** A track whose floor is 15 dB loud
   has its data term scaled down but keeps its curvature prior, so its
   effective band narrows by the same factor. That alone left 12.6 dB of
   residual comb at k1-9 against 4.3 dB unwhitened. The per-track mean weight
   now goes into `rho^2` as well.
3. **The floor mask must be about three linewidths wide, and capped.** The
   log-floor error against truth reads 3.5 dB rms at `(1.5, 3 Hz)`, **0.6 dB**
   at `(3, 10 Hz)` and 6.5 dB again at `(4, 30 Hz)` — too wide is as bad as too
   narrow, because the fit then bridges gaps instead of seeing the floor.

### The jobs

`vk-decompose-v3-dregon-1b0249`, `vk-decompose-v3-fly124-66a2e0`,
`vk-decompose-v3-fly125-71193b` on `uni-cpu`, 12 s windows at a 9 s hop,
32 kHz, `--k-max 80`, `--bw-schedule 3,0,1.5,3`, `--iters 3 --k-trust
3,12,80`, one recording each, all 8 microphones. Artifacts go to R2 under
`artifacts/vk-decompose-v3/<recording_id>/` from INSIDE the job, because
omnirun output collection silently drops files above about 25 MB.

### The "half-order comb" was the instrument, and the audit that found it

**Date**: 2026-08-14. The first v3 production run put the residual's k1-9
order-cell peak at **−0.4962 orders on all four DREGON rotors** and on three of
four FLY124 rotors, which reads as a sub-harmonic comb. It is not one. Two
tests, both cheap, both decisive:

1. **The two ends of a folded cell are the SAME physical half-integer
   position** (order `m−0.5` and order `(m+1)−0.5`). A line must appear at
   both. On the DREGON residual the low end read **+1.6 dB** and the high end
   **−0.4 dB**, in nearly every cell.
2. The cell profile is **monotone** from its low edge through the integer to
   its high edge. That is a ramp, not a peak.

The cause: one unit cell spans a whole order, which is 70 to 85 Hz of
frequency, and the broadband floor falls steeply across that span at low
harmonics. `cell_profile` normalized each cell by its own SCALAR median, which
removes the level but not the slope, so a cell of pure smooth floor folds into
a ramp and `argmax` lands on the low edge — the half-integer position — by
construction. This is the **third** instrument of this campaign to fail in the
same direction, after the narrow slot contrast and the rendered-comb metric.

**The fix** is `_order_trend`: divide the order profile by its running median
over one order before the fold (`order_cell_profile(detrend_orders=1.0)`, on by
default). A smooth tilt goes to unity, a line of much less than one order in
width passes through. Re-read of the SAME production residuals:

| residual, k1-9 | before (cell median) | after (running median) |
|---|---|---|
| DREGON | 1.427 dB at −0.4962 | **0.305 dB at −0.030** |
| DREGON k10-24 / k25-49 / k50-80 | 0.333 / 0.196 / 0.159 | **0.423 / 0.104 / 0.073** |
| FLY124 | — | **1.029 dB at +0.062** |

The DREGON per-rotor peak offsets scatter after the detrend (−0.50, +0.28,
−0.215, +0.315) — noise, not a comb.

### H_sub against H_cross, decided

Two hypotheses survived the instrument audit for whatever structure is LEFT.
`H_sub`: a genuine per-rotor `r/2` comb. `H_cross`: the leftover skirts of
ANOTHER rotor's already-modelled integer line, which fold near half-order in
the reference rotor's grid.

**`H_sub` is refuted on both rigs, and `H_cross` is confirmed on FLY124.**

- Per cell on DREGON, the detrended `−0.5` and `+0.5` ends collapse to about
  ±0.5 dB with no systematic sign in any cell. `H_sub` predicts every cell
  keeps the peak. Nothing is left to explain.
- On FLY124 two cells DO survive the detrend, and only on the twin pair
  (rotors 1 and 3, cells k2 and k3). There cell k2's high end and cell k3's low
  end — the same physical order 2.5 — agree (+5.90 and +5.50 dB on rotor 1),
  so that IS a line.
- The absolute-frequency check names it. Rotor 1's order-2.5 energy sits at
  **168.89 Hz**, and rotor 0's rate is 84.483 rev/s, so the line is
  **2.0000 × r_0** — off by 0.0009 orders. It is rotor 0's own `k` 2 harmonic,
  not a sub-harmonic of rotor 1.

This also explains why FLY124's rotor 0 read +0.06 while rotors 1 to 3 read
−0.50 under the old instrument: rotor 0 has a strong GENUINE integer-order
leftover (2.24 dB detrended) that outweighs the tilt ramp in the `argmax`,
while on the other three the real leftover is weak (0.49 to 0.71 dB) and the
ramp wins.

Consequence: **no half-order track grid**. The rotor-speed labels are the
SHAFT rate, so an `r/2` line needs a period-two-revolution mechanism, and the
measurement says there is none. The indicated correction for `H_cross` is a
FLOOR on the per-track phase-correction bandwidth at low `k`
(`--bw-psi slope,max,min`, default `0.6,8,1.5`): the law alone allows a `k` 2
line only 1.2 Hz, which is narrower than its true incoherent linewidth, so the
line keeps a skirt the model cannot follow.

### The theta stitch spike: an edge frame, and a metric that overstated it

`joint.theta_stitch_max_rate_hz` read 46.08 on the production run against
0.003 on the single-window smoke. Reproduced locally with two overlapping
estimated windows: the maximum sits at **frame 0 or the last frame of a
window**, where the cross-fade weight is 0.0025 to 0.005 — that is, where the
window contributes almost nothing to the stitch. Weighted by that fade, the
same reproduction reads **0.077 Hz** against a raw 1.86 Hz. The metric was
measuring a rotation that never reaches the bank.

Three changes, all small:

1. `split_phases` now weights the shaft smoother by the solver's own
   `edge_taper`, so the estimate EXTRAPOLATES over the span where the data term
   was faded instead of fitting the transient there.
2. `theta_rate` holds its first and last value instead of taking
   `np.gradient`'s one-sided difference, so no frame carries a different
   estimator from the interior.
3. The report now carries `theta_stitch_max_rate_hz` (fade weighted, the number
   to read) beside `theta_stitch_max_rate_hz_raw`.

### A real bug the same pass found

`r2_ref_mic` was NEGATIVE on nearly every production window (DREGON −0.16,
FLY124 −0.90 to −0.00) while the stitched ledger was healthy. The per-window
check rebuilt the plain label carrier instead of reading the solver's own
`env.phase`, which on the joint path carries the shaft correction — so it
scored the bank against a carrier it was never fitted to. Fixed; the stitch
itself always used the right phase.

### v3b: the re-run with the fixed instrument (2026-08-14)

Jobs `vk-decompose-v3b-{dregon,fly124,fly125}`, same settings as v3 plus
`--bw-psi 0.6,8,1.5`. Artifacts on R2 under `artifacts/vk-decompose-v3b/`.
Every number below comes from the DETRENDED instrument, so it is not
comparable to the v3 table above — read this one against itself.

**DREGON** (`free-flight_nosource_room1`, 7 windows, k_hi 80, tracks 37.6 % /
residual 59.3 %, resynthesis 6.0e-08):

| band | original | residual | excess original -> residual |
|---|---|---|---|
| k1-9 | 3.423 dB at +0.071 | **0.308 dB at −0.020** | 35.62 -> 21.96 (−13.66 dB) |
| k10-24 | 1.131 dB at −0.010 | **0.386 dB at +0.031** | 15.17 -> 10.83 (−4.34 dB) |
| k25-49 | 0.191 dB at −0.013 | **0.078 dB at −0.035** | 7.56 -> 5.05 (−2.52 dB) |
| k50-80 | 0.076 dB at −0.014 | **0.069 dB at +0.022** | 7.87 -> 6.25 (−1.63 dB) |

**FLY124** (12 of 14 windows, k_hi 76, tracks 61.9 % / residual 37.3 %,
resynthesis 3.0e-08):

| band | original | residual | excess original -> residual |
|---|---|---|---|
| k1-9 | 4.161 dB at +0.061 | 1.028 dB at +0.065 | 37.97 -> 20.57 (−17.41 dB) |
| k10-24 | 0.808 dB at +0.033 | **0.187 dB at +0.100** | 18.60 -> 13.66 (−4.94 dB) |
| k25-49 | 0.083 dB at −0.170 | 0.099 dB at −0.386 | 10.95 -> 9.86 (−1.09 dB) |
| k50-80 | 0.033 dB at −0.098 | 0.075 dB at −0.421 | 6.96 -> 6.17 (−0.79 dB) |

Readings:

- **No band peaks at a half-integer order any more**, on either rig. The
  per-rotor offsets scatter, which is what noise does.
- DREGON is at or near the acceptance bar in every band (0.069 to 0.386 dB).
- FLY124 keeps ONE real leftover: rotor 0 at k1-9, 2.22 dB at **+0.055** — an
  integer-order comb, not a half-order one. The `bw_psi_min` floor of 1.5 Hz
  did not close it (2.24 dB before, 2.22 dB after), so the next lever for that
  rotor is not the phase band.
- FLY124's k25-80 residual depths (0.075 to 0.099 dB) sit at the instrument's
  own noise floor and slightly exceed the original's, while the absolute
  `excess_db` still falls. Read the excess there, not the depth.
- `r2_ref_mic` is POSITIVE on every window now (DREGON 0.21 to 0.79, FLY124
  0.35 to 0.65) against −0.90 to +0.46 before the carrier fix.
- `theta_stitch_max_rate_hz` (fade weighted) 15.5 on DREGON and 6.8 on FLY124,
  against 46.6 and 48.6 raw — both inside the 50 Hz envelope Nyquist.
- The phase unwrap saturates at pi in the WORST track of every band
  (`max_step_rad_by_band_worst` 3.11 to 3.141). It does not reach the trusted
  set: `split_phases` drops any track whose step reaches pi, and `n_trust` was
  the full 12 tracks at `k` <= 3 and the full 48 at `k` <= 12, so no trusted
  track was ever gated out.
