# The joint decomposition (v3) — design

**Module**: `src/tracking/joint_decompose.py` · **Driver**: `scripts/vk_decompose.py --joint` ·
**Continues**: `docs/experiments/vk-decomposition.md` (v1, v2)

The v2 decomposition (`tracking.decompose`) splits a recording into per-(rotor, harmonic,
microphone) Vold-Kalman envelopes plus a per-microphone residual. It does this under two
conditions that it never states, and both conditions are incorrect. First, one fixed envelope
band holds every timing deviation. A shaft wanders by about 0.6 rev/s, so the identity
`k (phi + theta)` makes harmonic `k` about `0.6 k` Hz wide. That is more than the 3 Hz envelope
band from about `k` 5 up, and the flanks of every line become residual by construction. Second,
an unweighted least squares takes the floor to be white. Drone noise is strongly colored, so the
unweighted fit is tolerant of comb structure exactly where the floor is loud. v3 removes both
conditions.

## 1. The model

```
y_c(t) = sum_{r,k} Re[ g_{r,k,c}(t) e^{j(k phi_r(t) + k theta_r(t) + psi_{r,k}(t))} ] + n_c(t)
```

**`g_{r,k,c}`** is the residual envelope. Every timing deviation moves into the two phase terms,
so `g` only needs AMPLITUDE bandwidth. That is what lets the tuned v2 bandwidth law stay
unchanged.

**`theta_r`** is a slow coherent shaft correction, in radians, common to every harmonic of one
rotor. It has two parts. A rig-common part comes from every rotor's tracks. A small per-rotor
part comes from that rotor's own tracks. This hierarchy is the measured structure of the
deviation.

**`psi_{r,k}`** is a slow per-track phase correction. It holds what one harmonic does that
`k theta_r` cannot explain. Its allowed band increases with `k` (`bw_psi_hz`), because a high
harmonic wanders more.

**`n_c`** is colored noise with a smooth log spectrum `S_c(f, t)`. Smooth is the important word:
because `S` is smooth and the comb lines are sparse, `S` stays identifiable from BETWEEN the
lines.

There is NO per-microphone arrival term in this model. It was measured, and it is not there.
Every microphone sees the same phase deviation up to one constant of its own, and
`_combine_channels` removes that constant before the phase split.

## 2. The three blocks

`joint_solve_window` alternates three blocks. Each block is linear-Gaussian given the other two,
so the alternation is block-coordinate descent on one MAP objective. One iteration is about one
v2 solve, because the other two blocks are very cheap.

### Block A — the whitened VK solve

The coherent phases fold into the CARRIER, which is an exact reparametrization: `k (phi +
theta)` keeps the rotor-major power recursion, so a corrected carrier needs one array add only.
The three hooks of `tracking.vk_tracking.vk_envelopes` are the whole seam — `phase_offset`
(theta at audio rate), `env_rotation` (psi on the envelope grid) and `data_weight` (the
whitening). Every hook is `None` on the v2 path, and the arithmetic is then bitwise the v2
arithmetic.

The Whittle likelihood of colored noise weights each frequency by `1 / S`. Because `S` is smooth
and one line is narrow, that whole weighting becomes the ONE scalar `1 / S(k r(t), t)` per track
and per envelope frame (`whiten_weights`). Thus the banded structure of the solver stays
unchanged, and the whitening adds no work to the solve. `_whiten_weights` splits the weight into
its diagonal form `u^2` and its cross form `u_a u_b`, because a cross block is a product of two
DIFFERENT tracks' weighted bases.

`bandwidth_neutral` is the correction that makes the whitening safe. Without it a track whose
floor is 15 dB loud gets its data term scaled down and its curvature prior unchanged, so its
effective band becomes narrow by the same factor and its envelope becomes too smooth. The fix
puts each track's mean weight into `rho^2` as well, so the ACHIEVED bandwidth stays at the tuned
v2 value. The whitening then does only what it is for — the relative trust between coupled
tracks and across time.

### Block B — the phase split

`split_phases` reads the solved envelope bank of the CURRENT carrier, so its angle is the phase
error that is left. By the model that angle is `k theta_r + psi_{r,k}` plus noise. Every
harmonic measures `theta` with precision `k^2` times its own, so the shaft estimate is the
`k`-weighted mean of `arg x / k` over the trustable tracks. That estimate is far better
determined than any one harmonic or than the telemetry.

The smoother is `wh_smooth`, a Whittaker-Henderson smoother. Its transfer is
`1 / (1 + lam (2 sin(w/2))^4)`, which IS the VK-2 transfer. Thus `lam = rho^2`, and the
bandwidth relation is the solver's own (`_tuma_rho`). There is one calibration in the package,
not two.

The fit has two levels. The rig-common `theta_rig` comes first, from every rotor's trustable
tracks. Then each rotor's own tracks give a small per-rotor increment on top of it. What is left
per track becomes `psi`, smoothed with a wider band at higher `k`.

Two gates select the trustable set. The annealing cap `k_trust` is the first. The second is a
CONCENTRATION gate: `|mean exp(j d arg x)|` more than `conc_min` — a scale-free signal-to-noise
proxy that reads about 0 for a noise-dominated envelope and near 1 for a locked one. A track
whose phase increment reaches pi at any frame leaves the set whatever its concentration,
because its unwrap is a guess.

### Block C — the masked smooth floor

`masked_smooth_psd` computes a Welch spectrum of the current residual with every predicted comb
line masked out, per short frame, so the mask stays on a moving line. It then fits a smooth log
spectrum through the gaps — a moving median across frequency, then a cepstral lift to
`psd_n_cep` coefficients.

The mask must be several linewidths wide, and it must also be capped. A line whose skirts are
25 dB more than the floor still lifts the fit two linewidths out, so one linewidth is not
sufficient. The cap `mask_frac_of_rate` keeps the rule usable at `k` 80, where
`3 * 0.6 * k` alone is wider than the distance between one rotor's adjacent lines. Too wide is
as bad as too narrow, because the fit must then bridge gaps instead of reading the floor.

The mask is what makes the estimate honest. An unmasked floor fit increases under every line,
and a floor that increases under the lines tells block A not to fit them. That failure mode
hides itself, which is why the mask is not optional.

## 3. The annealing ladder

The ladder `k_trust` starts at 3, and this is the important correction to the original plan.

The limit on which harmonics can measure the shaft is the ENVELOPE BAND, not the phase unwrap. A
shaft that wanders by `sigma_r` rev/s makes harmonic `k` a frequency modulation of bandwidth
about `k sigma_r` Hz. A band of `B` Hz distorts that phase once `k sigma_r` is more than
`B / 2`. At `sigma_r` 0.6 and `B` 3 that is `k` 2.5. Thus the ladder starts at 3 and not at 10.

Each fold decreases the residual wander everywhere, which brings higher harmonics under the
ceiling, so the next rung can be far higher. The shipped ladder is `(3, 12, 80)`. `psi` starts
at iteration `psi_from_iter` (2 by default), after `theta` takes the coherent part.

## 4. The instruments

A verdict is only worth what its instrument is worth, so both instruments live in the same
module.

**`order_cell_profile`** is THE probe. It takes the power spectrogram, averaged over
microphones, and re-expresses each frame's frequency axis in ORDERS of one reference rotor
(frequency over that rotor's instantaneous rate). The comb then stops drifting and its teeth sit
on the integers. The result is averaged onto a fixed order grid, and every unit cell
`[m - 0.5, m + 0.5)` of a harmonic band is folded into ONE profile (`cell_profile`). Each cell
is first normalized by its own median, so the fold measures modulation and not the spectral tilt
across the band. Every rotor is the reference in turn, and the band reading is the mean over
them. `exclude_others` removes every bin near ANY other rotor's line before the order mapping,
which is what makes the reading meaningful on a multi-rotor rig.

Two readings come back:

- `depth_db` — the folded peak over the folded median. It is a RATIO, so it can increase while
  the residual decreases toward the broadband floor. On a four-rotor rig the other rotors' lines
  also put a floor under it.
- `excess_db` — ten times the log of the summed ABSOLUTE excess `peak - median` over the band's
  cells, before the per-cell normalization. It is in power units of the input, so it is
  comparable ACROSS signals. The original audio's `excess_db` minus the residual's `excess_db`
  is how many decibels of comb the decomposition removed, and it does not move when the floor
  moves.

**Read `excess_db` for the verdict.** `depth_db` is the v2 instrument of record and it stays in
the report, but at mid `k` it gives almost no difference between two signals (see § 9).

**`whitened_flatness`** is the second instrument: the spectral flatness of `|N(f)|^2 / S(f)` per
microphone, beside the flatness of `|N|^2` itself. A correct floor model leaves a flat whitened
residual, so the pair (raw, whitened) is the reading.

**Standing policy.** Never use a narrow on-order against half-order slot contrast as a verdict.
That instrument reads about zero for a comb whose linewidth is more than the slot, or whose peak
sits outside it. It has already given one reported verdict, and that verdict was then removed.

## 5. The knobs

| Name | Default | What it does |
|---|---|---|
| `JointConfig.iters` | 3 | Alternation rounds. One round is about one v2 solve. |
| `JointConfig.k_trust` | `(3, 12, 80)` | The annealing ladder — one trustable harmonic cap per iteration. |
| `JointConfig.psi_from_iter` | 2 | First iteration (1 based) that estimates the per-track `psi`. |
| `JointConfig.bw_theta_hz` | 1.5 | Bandwidth of the shaft correction `theta`, in Hz. |
| `JointConfig.bw_psi_slope` | 0.6 | Slope of `bw_psi_hz(k) = min(slope * k, max)` — the measured linewidth law. |
| `JointConfig.bw_psi_max` | 8.0 | Cap of the same law. It prevents a high-`k` correction that takes in the floor. |
| `JointConfig.conc_min` | 0.5 | Concentration gate on a track's phase increments. |
| `JointConfig.per_rotor_theta` | `True` | Fit the small per-rotor part of `theta` on top of the rig-common part. |
| `JointConfig.whiten` | `True` | Weight block A by `1 / sqrt(S)`. |
| `JointConfig.whiten_clamp_db` | 15.0 | Clamp on the weight, so one quiet band cannot get too much of a solve. |
| `JointConfig.bandwidth_neutral` | `True` | Put each track's mean weight into `rho^2`, so the whitening does not retune the band. |
| `JointConfig.psd_n_fft` | 4096 | Transform length of the floor fit. |
| `JointConfig.psd_blocks` | 4 | Time blocks of the per-window floor. |
| `JointConfig.psd_n_cep` | 40 | Cepstral coefficients kept — how smooth the log floor is. |
| `JointConfig.profile_n_fft` | 8192 | Transform length of the order-cell probe. |
| `JointConfig.profile_order_step` | 0.005 | Order grid step of the probe. |
| `JointConfig.profile_every_iter` | `True` | Profile every iteration's residual, not the last one only. |
| `--joint` | off | Turn on the v3 alternation. Off IS the v2 path, call for call. |
| `--iters` | 3 | Sets `JointConfig.iters`. |
| `--k-trust` | `3,12,80` | Sets the ladder. |
| `--bw-psi` | `0.6,8` | Sets `bw_psi_slope,bw_psi_max`. |
| `--bw-theta` | 1.5 | Sets `bw_theta_hz`. |
| `--no-whiten` | off | Runs block A on the unweighted misfit. |

`masked_smooth_psd` carries its own mask geometry: `mask_factor` 3.0, `mask_min_hz` 10.0 and
`mask_frac_of_rate` 0.45. The half-width per line is
`clip(mask_factor * LINEWIDTH_HZ_PER_K * k, mask_min_hz, mask_frac_of_rate * r(t))` Hz.

## 6. Windows and the stitch

One window holds its own shaft correction `theta`, and a phase has an additive constant of its
own in each window. Two overlapping windows would then hold one physical correction at two
origins, which is exactly the failure that the envelope stitch already guards against.

Thus the stitch moves the RATE across a window boundary, and not the phase. `theta_rate` gives
`d theta / dt / (2 pi)` in rev/s, which is gauge-free because the additive constant
differentiates away. `global_rate_correction` cross-fades the per-window rates onto one global
envelope grid. `corrected_phase` integrates that into one global corrected carrier
`r_corrected = r_labels + dr_global`. `window_extra_phase` then gives the extra rotor phase that
moves one window onto that global carrier, and each track `m` is multiplied by
`exp(j k_m e[rotor_m])` before the usual cross-fade.

The rotation is slow by construction, because the two carriers are the same trajectory up to the
blend between adjacent windows. That is REPORTED and not taken for granted:
`theta_stitch_max_rate_hz` in `report.json` measures how fast the fastest track's rotation is,
so a reader can see whether it stayed inside the 100 Hz envelope grid. At high `k` a large
disagreement would cause aliasing on that grid.

## 7. Outputs

`--joint` adds these to the v2 output set:

- Per unit, in `raw/<uid>.npz`: `theta` (the total shaft correction, radians), `dr` (the same
  correction as a rate, rev/s), `psi` (the total per-track correction), and the floor model as
  `psd_freq`, `psd_t` and `psd_log_s`.
- Per recording, `joint.npz`: `dr_global`, `r_corrected` and `r_labels` on the envelope grid.
- In `report.json`: a `joint` section (the stitch rate statistics), `order_cell` with both a
  `residual` and an `original` band table, and `flatness`.
- Per unit JSON, under `joint.iterations`: every iteration's diagnostics — `residual_fraction`,
  `track_fraction`, `psd_masked_frac`, `flatness`, the phase-split diagnostics and the
  order-cell band table of that iteration's residual.

## 8. Measured results on the synthetic fixture

The fixture is 20 s, 16 kHz, 4 rotors, 3 microphones, `k` up to 20. Its shaft rate wanders by
0.5 rev/s at a bandwidth of 0.5 Hz (a rig-common part plus a per-rotor fifth). Per-track phase
noise is 0.02·k radians rms at a bandwidth of min(0.6·k, 8) Hz. The colored floor is smooth and
34 dB less than the comb.

| arm | residual fraction | k1-9 depth / excess dB | k10-24 depth / excess dB |
|---|---|---|---|
| original audio | 1.0000 | 28.17 / 62.03 | 2.54 / 35.93 |
| v2 (flat carrier) | 0.0551 | 6.09 / 43.65 | 2.02 / 34.18 |
| **v3 (`--joint`, 3 rounds)** | **0.0025** | **1.21 / 24.86** | **1.38 / 20.24** |
| oracle (the true shaft folded in) | 0.0024 | 0.89 / 22.90 | 1.62 / 19.85 |

The reading: v2 removes 18.4 dB of comb excess at k1-9 and only 1.75 dB at k10-24. v3 removes
37.2 dB and 15.7 dB, and it is within 2 dB of the oracle in both bands. The correlation of the
recovered shaft phase against the truth is 1.000 (0.999 to four places, error 0.15 rad rms
against a true 3.65 rad rms). The fitted log floor is within 0.5 to 1.0 dB rms of the truth away
from the lines. The whitened residual flatness increases from 0.010 to 0.36.

Three design decisions were measured on the same fixture:

1. **The ladder must start low.** A ladder that starts at `k` 6 recovers only 43 % of the true
   shaft phase in three rounds. A ladder that starts at 3 recovers all of it.
2. **The whitening needs the bandwidth-neutral correction.** Without it the residual comb at
   k1-9 is 12.6 dB, against 4.3 dB unwhitened. A down-weighted track keeps its curvature prior,
   so its band becomes narrow.
3. **The floor mask must be about three linewidths wide.** The log floor error against truth is
   3.5 dB rms at `(1.5, 3 Hz)`, 0.6 dB at `(3, 10 Hz)`, and 6.5 dB again at `(4, 30 Hz)`.

## 9. Open questions and limits

- **`depth_db` is weak at mid `k` on a four-rotor rig.** The original audio reads 2.54 dB at
  k10-24 and a near-perfect decomposition reads 1.62 dB, so the ratio reading gives almost no
  difference there. `excess_db` gives a clear difference, which is why it is the verdict.
- **The whitening weight is pooled over microphones.** `SmoothPSD.pooled` takes the geometric
  mean, so the per-microphone floor LEVEL does not reach the solve. Only the shape does. A
  per-microphone weight makes the banded system channel dependent, and it needs one banded
  factorization for each microphone. The per-microphone surface is still estimated and still
  reported, because a downstream noise model wants it.
- **Foreign non-comb tones still cause damage to the floor fit locally.** The v2 record found
  quasi-stationary structural or aerodynamic resonances at non-integer orders (for example the
  489 Hz line of `free-flight_nosource_room1`). The mask does not cover them, and a smooth log
  spectrum cannot represent them. `report.json -> residual_tones` measures them, and nothing
  removes them.
- **The per-window floor is fitted in a small number of time blocks** (`psd_blocks` 4 by
  default). The fit does not follow a floor that moves faster than one block of a window.
