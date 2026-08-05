# Null controls for the displaced-comb measurement

Measured 2026-08-05 on the same 15 windows of `beatvk-valid-raw@54849c13ed3a` that
`measure_displacement.py` used. Scripts: `nullcontrol.py` (measurement + nulls + traces),
`summarize_null.py` (the tables below), `make_figs2.py` (figures). Raw numbers:
`nullcontrol.json`, `prominence.json`, `summary_null.txt`. Figure: `figs/F4_nullcontrol.png`.

---

## VERDICT

**The claim "the high-$k$ acoustic comb tracks telemetry to ~0.09 rev/s" is REFUTED.
It is a peak-search-window artifact.**

The identical pipeline, run at a carrier where **no rotor line can exist**, returns the
same number:

| DREGON cruise, high $k$ (16-40) | MAE vs telemetry (rev/s) |
|---|---|
| MEASURED, carrier $k\,g_r(t)$ | **0.0856** |
| NULL, carrier $(k{+}0.5)\,g_r(t)$ — off-comb, no line exists | **0.0857** |
| NULL, carrier $k\,g_{\text{partner}}(t)$ — telemetry from a different window | **0.0845** |
| analytic: peak drawn uniformly in the search window | 0.1537 |

Ratio measured / off-comb null = **1.00**. On FLY124 high $k$: 0.0654 measured vs 0.0648
off-comb null vs 0.0702 mismatched — ratio **1.01**. Neither dataset's high-$k$ number
carries any information about the acoustic comb.

The mechanism is exactly the one suspected: the search half-width is $\min(1.5k, 8)$ Hz,
which in shaft-rate units is $\le 8/k$ rev/s and shrinks as $1/k$. Combining ~25
harmonics per frame averages the per-$k$ peak-picks down, so the *combined* series lands
near 0.086 whether or not there is a line. The number measures the window, not the physics.

**The corroborating fact is stronger than the null itself: on DREGON there is no
high-$k$ line to measure.** Pooled over 9 cruise windows x 4 rotors x 25 harmonics
(900 units), only **3 of 900** harmonics with $k \ge 16$ reach 6 dB over their own in-band
noise floor — **fewer than the off-comb null's 8 of 900**. Median prominence at high $k$ is
1.79 dB on-comb against 1.73 dB for the null (excess **+0.06 dB**), and median demod SNR is
3.86 dB on-comb against 3.73 dB for the null. The "3-5 dB SNR" in the bottom panel of F2 is
the peak-pick bias of a noise floor, not a signal.

### Consequences for what was published

1. **Delete the three-way error decomposition claim.** "high-$k$ comb MAE 0.086 vs
   flagship blind track 1.854, therefore 20x headroom, therefore the remaining error is
   estimator error not physics" does not follow. The 0.086 is what the estimator returns
   on noise. F3 must be withdrawn or relabelled.
2. **The honest high-$k$ statement**: on DREGON cruise the acoustic comb above $k = 16$ is
   **not measurable at all** by this demodulation. There is no evidence for or against
   agreement with telemetry. The only defensible bound is the search half-width itself —
   $\pm 0.31$ rev/s at $k = 16$ falling to $\pm 0.20$ rev/s at $k = 40$ — and even that is
   not a bound, because a line outside the window would simply be missed.
   On FLY124 a trace of high-$k$ signal survives (14 of 400 units over the bar, against 2
   for the null); restricted to those units the mean offset is **-0.055 rev/s**
   (MAE 0.071, n = 14). That is the *only* high-$k$ number in either dataset with a signal
   behind it, and it rests on 3.5% of the harmonics.
3. **The "returns to the grid by $k \ge 16$" story is unsupported.** The high-$k$ offsets
   are small because the window is small, not because the comb is on the grid.

---

## What SURVIVES: the low-$k$ displacement

The low-$k$ finding is real, and it is **larger** than reported.

| | DREGON cruise | FLY124 cruise |
|---|---|---|
| mean $|\delta|$, low $k$, measured | 0.4094 | 0.2634 |
| mean $|\delta|$, low $k$, off-comb null | 0.3449 | 0.4848 |
| signed mean $\delta_k$, measured | **-0.1239** | -0.0605 |
| signed mean $\delta_k$, off-comb null | +0.0084 | -0.0732 |
| median prominence, on-comb / null | 2.72 / 2.31 dB | 5.62 / 3.54 dB |
| units over the 6 dB bar, on / null | **34 / 2** of 432 | 74 / 30 of 192 |

Two independent things separate the measurement from the null at low $k$:

- **The null's signed mean is 0** (+0.008 rev/s), the measurement's is not (-0.124 rev/s).
- **The bar test discriminates**: 34 DREGON low-$k$ units clear 6 dB where the null clears
  it twice. Those are real lines.

Restricted to the 34 DREGON units that clear the bar, the offset is
**-0.424 rev/s** (MAE 0.442), and the window-independent pulse-pair estimator on the same
units gives **-0.231 rev/s** (pulse-pair is biased toward zero at low SNR, so it is a lower
bound in magnitude). The corresponding FLY124 figure is **-0.051 rev/s** over 74 units —
i.e. FLY124's comb sits on its telemetry, DREGON's does not.

Two further facts about the DREGON displacement, from the trace analysis (see `wiggle.md`):

- It is **flat in $k$**: for the strongest rotor of `free-flight_nosource_room1__w01`,
  $\delta_k$ = -0.51, -0.51, -0.53, -0.34 rev/s at $k$ = 2, 4, 6, 8. A fixed-frequency
  artifact would fall as $1/k$ (-0.51, -0.25, -0.17, -0.13). Flat wins.
- As a **fraction of rate** it is -0.54% (mean over the 34 bar-clearing units,
  rate range 56-86 rev/s); per rotor -0.55%, -0.34%, -0.54%, -0.45%. FLY124's is
  **-0.063%**, consistent with zero and with the -0.054 rev/s residual left by the
  2026-07-31 Michael's calibration (WP14). This is the signature of a **telemetry scale
  error of roughly -0.5 to -0.6% on DREGON** — the same class of defect that was measured
  and corrected on Michael's rig, and never checked on DREGON.

**The pooled -0.145 rev/s previously reported for DREGON low $k$ is diluted**, because it
SNR-weights over rotors and harmonics that carry no measurable line and therefore
contribute noise centred on zero. Where a line exists the displacement is -0.42 rev/s.

### Where a line actually exists (from `prominence.json`, figure F5)

Median ridge prominence over the in-band floor, on-comb and at the off-comb null, pooled over cruise windows x rotors.

| $k$ | DREGON on | DREGON null | excess | over 6 dB | FLY124 on | FLY124 null | excess | over 6 dB |
|---|---|---|---|---|---|---|---|---|
| 2 | 9.1 | 3.0 | +6.1 | 89% | 24.5 | 6.1 | +18.4 | 94% |
| 3 | 3.0 | 2.7 | +0.3 | 6% | 8.8 | 3.7 | +5.0 | 88% |
| 4 | 4.2 | 2.6 | +1.5 | 17% | 15.1 | 4.0 | +11.1 | 94% |
| 6 | 2.9 | 2.2 | +0.7 | 0% | 10.1 | 5.6 | +4.5 | 100% |
| 8 | 2.6 | 2.0 | +0.6 | 0% | 9.2 | 3.0 | +6.2 | 81% |
| 10 | 3.1 | 1.7 | +1.3 | 6% | 4.1 | 2.8 | +1.3 | 31% |
| 13 | 3.6 | 1.5 | +2.2 | 33% | 1.9 | 2.1 | -0.2 | 0% |
| 16 | 2.6 | 1.9 | +0.7 | 0% | 3.0 | 2.3 | +0.7 | 19% |
| 20 | 1.8 | 2.1 | -0.4 | 0% | 2.3 | 2.6 | -0.3 | 19% |
| 25 | 1.6 | 1.8 | -0.2 | 0% | 2.1 | 2.0 | +0.1 | 0% |
| 30 | 2.0 | 1.9 | +0.1 | 0% | 2.7 | 2.1 | +0.6 | 0% |
| 40 | 1.5 | 1.5 | -0.0 | 0% | 2.1 | 1.5 | +0.6 | 0% |

DREGON has one usable harmonic, $k = 2$ (+6.1 dB over the null, 89% of units over the bar), a weak second group around $k = 4$-13, and **nothing** above $k = 14$.
FLY124 carries a real harmonic set out to $k \approx 20$.

---

## Method of the controls

All four controls share the measurement's demod band ($\min(3k,\,0.45\,\bar r)$ Hz), its
search half-width ($\min(1.5k, 8)$ Hz, further capped at 0.9 x band), its collision gate,
its per-frame SNR weighting, and its weighted combination over $k$. Only the carrier
changes.

- **(a) Off-comb null.** Carrier $(k{+}0.5)\,g_r(t)$: the same trajectory, the same band,
  but a rate at which no rotor line exists. Implemented by halving the phase and asking the
  tracker's own integer recursion for $2k{+}1$
  (`exp(-i(2k{+}1)\phi/2) = exp(-i(k{+}1/2)\phi)`), so the demod path is bit-for-bit the
  measurement's.
- **(b) Mismatched-telemetry null.** Carrier $k\,g_r(t)$ taken from a **different window**
  (DREGON: the same window index of another recording; FLY124: another cruise window).
  Real spectra, broken correspondence.
- **(c) Analytic baseline.** A peak drawn uniformly in $[-W, W]$ gives $E|\delta| = W/2$.
  Mean over the band: 0.549 rev/s (low $k$), 0.154 rev/s (high $k$). The measured window
  fill $|\delta| / W$ is 0.365-0.395 rather than the uniform 0.5, on-comb *and* on the
  nulls alike — the smoothing and parabolic refinement concentrate a noise peak slightly,
  which is why the raw analytic figure overestimates the null and the empirical nulls are
  the ones to compare against.
- **(d) Window-independent estimator.** Pulse-pair / phase increment,
  $\arg\left(\sum_n \sum_c z[c,n]\,\overline{z[c,n{-}1]}\right) / (2\pi k\,\Delta t)$. Its
  unambiguous range is $f_{\text{env}}/2k$ rev/s, far outside the demod band, so it is
  band-limited but **not** search-limited (the band is 4-9x wider than the search window at
  high $k$).

  DREGON high $k$: on-comb mean **+0.0061** rev/s (MAE 0.1199); off-comb null mean
  **+0.0395** (MAE 0.1489). FLY124 high $k$: +0.0006 (MAE 0.145) vs +0.0005 (MAE 0.229).
  The coherent estimator **does not discriminate either** — it returns near-zero on the
  null as well, because in-band noise is symmetric and pulls the phase-increment mean to
  zero. Agreement between the peak-pick and the pulse-pair at high $k$ is therefore *not*
  evidence: both estimators return "≈ 0" on pure noise. This is stated explicitly because
  it is the trap that made 0.086 look meaningful in the first place.

  At **low** $k$, where lines exist, the pulse-pair does separate: DREGON on-comb mean
  -0.025 vs null +0.146; restricted to bar-clearing units, -0.231 vs the peak-pick's -0.424.

### One methodological bug found and fixed

The first pass gated the nulls with the tracker's own twin rule, which assumes the carrier
is $k\,r_i$ of a rotor present in the telemetry array. That is false for both nulls, so
they were being contaminated by real neighbouring-rotor lines that the measurement is
protected from. `carrier_collision_mask` re-derives the rule for an arbitrary carrier
against the **audio's** real rotor lines; it reproduces `_twin_collision_mask` exactly for
the on-comb case (verified bit-identical on all 4 rotors of a window). All numbers above
are post-fix. The fix moved the DREGON low-$k$ null from 0.357 to 0.294 and left the
high-$k$ verdict unchanged (0.0796 -> 0.0857 against 0.0856 measured).

### Caveats

- The nulls are empirical, not analytic, so they inherit the same in-band noise statistics
  as the measurement. That is the point, but it means "measurement = null" is evidence of
  *no detectable signal*, not proof that the comb is on the grid.
- The collision gate removes 61-73% of (harmonic, frame) pairs on DREGON. The surviving
  frames are not a random sample.
- Everything is cruise-regime. FLY124 warm-up windows are excluded throughout.
