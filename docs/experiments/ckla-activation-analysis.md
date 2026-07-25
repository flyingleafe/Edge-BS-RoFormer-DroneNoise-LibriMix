# CKLA activation analysis — what the trained head actually computes

**Status:** done — 2026-07-26. Analysis note (no training); companion to the
CKLA batch doc [`ckla.md`](./ckla.md) and the design-§6 diagnostics kit in
`docs/ckla-design.md`. Tooling: `scripts/ckla_activation_analysis.py` +
the `return_state`/`capture_state` instrumentation added to
`src/models/ckla.py`. Raw numbers/figures: `results/ckla_activation_analysis/`
(gitignored; regenerate with the command below).

```
python scripts/ckla_activation_analysis.py \
    --data datasets/DREGON-LM-V4-michaels-full/valid   # or dload: URI
```

Protocol: 12 seeded clips (8 dregon_cruise + 4 fly124_cruise) from the
vk_valid_comparison table, mic ch0, 8 s, CPU fp32 no_grad. Models:
`ckla_p1_if` best (the P1 model), `ckla_p0_staticcomb` best (synthetic
reference for A1–A3), `g2_if_transformer` best (comparator for A5/A6).
Intact 12-clip PIT-MAE on this subset: ckla_p1 2.712, g2_if 2.779,
ckla_p0 5.064 (off-distribution, as expected).

## Mechanistic summary

The trained CKLA head is **a bank of fixed very-long-horizon evidence
integrators with a regime-level (not frame-level) input gate, plus a small
but real rotation contribution on DREGON**. Uncertainty gating in the
Kalman sense is essentially unused: within cruise clips the evidence
precision λ_v is near-constant (CV 0.03–0.17) and only weakly tracks
acoustics; its one strong behaviour is a step change at flight-regime
transitions (idle→cruise). State precisions λ never converge within an 8 s
clip — they climb monotonically to 10³–10⁵ so the effective Kalman gain
φ/λ decays to 10⁻⁷–10⁻⁴, i.e. by mid-clip each new frame nudges the state
by parts-per-million: the layer behaves as a monotone clip-scale
accumulator, not an adaptive filter. The readout slot mix is nearly static
and concentrated on the longest-horizon slots (entropy ≈1.9 of 4 bits).
The rotation path — flatly null in P0 — is *weakly load-bearing on real
DREGON*: ω excursions in layer 2 correlate strongly with GT RPS level
(r = 0.82), and zeroing rotation costs +0.31 dregon PIT-MAE (+9%), while
FLY124 is untouched. Against the cross-drone hypothesis, CKLA is **more**
recoloring- and gain-sensitive than the transformer (−6 dB gain shifts its
predictions by 3.6 rev/s), and *both* architectures ignore a ×1.02
frequency scaling of the input (respond ≈0.03% vs the ideal 2%): neither
model is a comb-reader at the margin — predictions are anchored to the
training RPS prior, and CKLA's FLY124 advantage is not explained by
scale-faithful comb reading.

## A1 — precision-gating adaptivity (λ_v)

Within-clip statistics of the frame-wise channel-mean λ_v, per layer
(mean over 12 clips):

| model | layer | CV(λ_v) | r(λ_v, log-energy) | r(λ_v, speech-band frac) |
|---|---|---|---|---|
| ckla_p1 | 1 | 0.117 | +0.144 | −0.157 |
| ckla_p1 | 2 | 0.058 | −0.215 | +0.121 |
| ckla_p0 | 1 | 0.174 | −0.131 | +0.076 |
| ckla_p0 | 2 | 0.028 | +0.072 | −0.043 |

Structural note: because k is L2-normalised over slots (QK-norm), the
per-frame *total* evidence precision Σ_n φ_t[n,d] = λv_t[d] exactly — so
CV(φ) ≡ CV(λ_v) and the only evidence-weighting lever the layer has is
λ_v itself.

Verdict: within cruise, λ_v is close to a constant (CV ≤ 0.17, acoustic
correlations |r| ≤ 0.22) — no frame-level SNR weighting. The exception is
*regime* gating: in the one clip containing a takeoff ramp
(sample_00001, `fig_a1_lamv_sample_00001.png`), layer-1 λ_v steps ~0.3 → 2.0
exactly at spin-up (t ≈ 1.9 s) and layer-2 λ_v does the inverse. The gate
is used as a flight-regime switch, not as a per-frame Kalman evidence
weight.

## A2 — state precision λ_t dynamics

Per-slot λ trajectories (mean over channels and clips), slot horizons from
the trained decay ā: 13–219 frames (0.4–7.0 s) in both layers — the
static-forensics multi-scale bank survives training. Numbers (ckla_p1):

| layer | med t_sat (s) | max t_sat (s) | CV(λ) post-sat | median end gain φ/λ |
|---|---|---|---|---|
| 1 | 7.23 | 7.65 | 0.015 | 4.4e−06 |
| 2 | 7.34 | 7.74 | 0.015 | 1.3e−06 |

(t_sat = first t with λ ≥ 0.95·λ_final; ckla_p0 is indistinguishable:
7.25–7.78 s, CV 0.015.) λ *never converges inside the clip* — trajectories
rise monotonically through the full 8 s to 10³–10⁵
(`fig_a2_lambda_gain_ckla_p1.png`), even for slots whose nominal horizon
is <1 s, and the effective Kalman gain collapses within ~1–2 s to
10⁻⁷–10⁻⁴ per step. Post-rise λ is flat (CV 0.015) and λ_v is flat (A1),
so nothing is being tracked: the layer degenerates to a fixed bank of
clip-scale accumulators whose outputs are increasingly frozen as the clip
proceeds. This is *consistent with the P0 capture-boundary result*
(graceful degradation, locks slowly) and with the vk_eval finding that
CKLA benefits from stitched long context: its computation is accumulation,
not adaptation.

## A3 — readout horizon selection

Readout mass p_t[n] = normalised |q_t[n]·‖μ_t[n,:]‖| over the 16 slots
(slots sorted by horizon), per layer:

| model | layer | entropy (bits, max 4) | ±std | mean std_t(p) | r(long-horizon mass, speech frac) |
|---|---|---|---|---|---|
| ckla_p1 | 1 | 1.93 | 0.57 | 0.020 | +0.10 |
| ckla_p1 | 2 | 1.89 | 0.34 | 0.019 | −0.01 |
| ckla_p0 | 1 | 1.71 | 0.35 | 0.024 | −0.02 |
| ckla_p0 | 2 | 2.74 | 0.22 | 0.020 | +0.02 |

The slot mix is static: per-slot mass fluctuates by only ±0.02 over a clip
and has no speech coupling (|r| ≤ 0.10). Mass concentrates on the two or
three *longest*-horizon slots (heatmaps `fig_a3_slotmass_*.png`; in
sample_00001 the dominant slot switches once, at the takeoff transition,
then stays put). The multi-scale bank is effectively used as one or two
long integrators + a residue; q does not re-select horizons frame-to-frame.

## A4 — rotation usage on real data (ckla_p1)

Excursions ω_t − ω0 (per layer, over 12 clips):

| layer | median std_t per slot (rad) | max slot (rad) | r(mean \|exc\|, GT RPS) | r(mean \|exc\|, \|dGT/dt\|) |
|---|---|---|---|---|
| 1 | 0.013 | 0.108 | +0.29 | −0.16 |
| 2 | 0.006 | 0.136 | **+0.82** | −0.33 |

Excursions are small in absolute terms (≤0.14 rad std; ω0 spans 0–π) and
concentrated in 3–4 slots per layer, but in layer 2 they track the GT RPS
*level* strongly (r = 0.82 pooled over frames) — the input-dependent
rotation W_ω h does encode an RPS-dependent signal (a closed-loop
frequency belief), rather than noise. Correlation with the RPS derivative
is weakly negative, i.e. it is a level code, not a slew code.

Causal 3-arm test (per-pool mean PIT-MAE over the 12 clips):

| arm | dregon_cruise | fly124_cruise | all |
|---|---|---|---|
| intact | **3.481** | 1.174 | **2.712** |
| rotation zeroed (s=ω0=W_ω=0) | 3.788 | **1.156** | 2.910 |
| imaginary readout zeroed (mix im-half=0) | 3.647 | 1.165 | 2.820 |

Unlike the P0 static-comb null, rotation IS load-bearing on real DREGON:
zeroing it costs +0.31 (+9%), consistently across all 8 dregon clips
(the takeoff-ramp clip sample_00001 contributes most, +1.67; excluding it
the delta is still +0.11 over 7/7 clips). Zeroing only the imaginary
readout half costs +0.17 — about half the rotation effect is carried
through the im channels, half through rotation's effect on η_re. On
FLY124 both ablations are exact nulls (±0.02) — the cross-drone win does
**not** come from the complex path.

## A5 — where does RPS become decodable (ridge probes)

Ridge probe (closed-form, standardised features, α picked on an inner
80/20 train split) at three taps, target = per-frame **sorted (ascending)
GT RPS vector** (documented choice: removes rotor permutation without
per-clip PIT bookkeeping; equivalent to PIT up to frame-wise ties). Fit on
6 clips, evaluated on the 6 held-out clips (split stratified per pool,
seed 1). R² is harsh here because cruise-frame variance is tiny — read
MAE (rev/s) as the primary column:

| model | tap | R² | MAE |
|---|---|---|---|
| ckla_p1 | trunk (freq_pool out, 128d) | −6.90 | 3.82 |
| ckla_p1 | after CKLA block 1 (128d) | −1.91 | 2.40 |
| ckla_p1 | after CKLA block 2 (128d) | −1.11 | 1.95 |
| ckla_p0 | trunk | −13.5 | 5.45 |
| ckla_p0 | block 1 / block 2 | −5.9 / −5.6 | 3.90 / 3.86 |
| g2_if | trunk (freq_pool out, 128d) | −14.5 | 4.42 |
| g2_if | after transformer layer 1 (64d) | +0.03 | 1.45 |
| g2_if | after transformer layer 2 (64d) | +0.05 | 1.43 |

RPS is **not** linearly decodable from the shared conv trunk (MAE
3.8–4.4): the temporal head does the real work in both architectures. The
transformer makes RPS almost fully linearly explicit after ONE layer
(1.45) with no gain from layer 2; CKLA refines gradually (3.82 → 2.40 →
1.95) and never reaches the transformer's linear readability — yet the
full models score the same (2.71 vs 2.78). CKLA's final RMSNorm+linear
readout evidently extracts more than a plain ridge on the block-2 features
can, i.e. its RPS code stays partly non-linear (plausibly ratio-coded in
the slowly-frozen accumulator states) where the transformer's is an
explicit per-frame feature.

## A6 — amplitude-shortcut sensitivity (cross-drone hypothesis)

Per-clip mean |ΔRPS| (rev/s) under input perturbations; scale row =
response to resampling all frequencies ×1.02, ideal answer is predictions
×1.02:

| perturbation | ckla_p1 | g2_if |
|---|---|---|
| spectral tilt +6 dB (0→8 kHz, ±3 dB at edges) | 0.80 | 0.46 |
| spectral tilt −6 dB | 0.57 | 0.27 |
| gain +6 dB | 0.71 | 0.65 |
| gain −6 dB | **3.64** | 0.99 |
| freq scale ×1.02 — mean pred ratio | 1.0003 | 1.0006 |
| freq scale — deviation from ideal ×1.02 | −1.93% | −1.90% |

The hypothesis (transformer = timbre-reader, CKLA = scale-faithful
comb-reader) is **refuted on both ends**: CKLA is *more* sensitive to
recoloring (0.80/0.57 vs 0.46/0.27) and dramatically sensitive to −6 dB
gain (3.64 rev/s — the log1p-magnitude front-end is not scale-invariant
and the CKLA head amplifies that), and BOTH models respond ≈0.03–0.06% to
a 2% comb-frequency shift — i.e. a genuine ×1.02 shift of every rotor line
moves neither model's prediction. Both are amplitude/timbre-pattern
readers whose outputs are anchored to the training RPS distribution;
CKLA's FLY124 advantage must come from something else (most plausibly the
long-integrator temporal prior of A2 suppressing the transformer's
per-frame overfitting to DREGON texture, cf. the A5 one-layer-explicit
transformer code).

## Conclusions

1. The Kalman machinery is mostly unused as *uncertainty* machinery:
   constant λ_v within regime, gain → 0, static slot mix. What survives is
   a **multi-horizon leaky-accumulator bank with a regime gate** — that,
   not adaptivity, is the P0/FLY124 advantage.
2. Rotation is now *quantifiably* load-bearing on real DREGON cruise
   (+0.31 MAE when removed, r = 0.82 RPS-level code in layer 2) — the P0
   "rotation null" does not transfer to real data, but the effect is far
   too small to close the 2.87 → 2.481 gap, and null on FLY124.
3. Neither CKLA nor the transformer reads comb spacing at the margin (2%
   scale test ≈ ignored). Any push toward VK parity via architecture must
   first break the amplitude anchor — e.g. scale-equivariant targets/augs
   or explicit comb front-ends — before head-internal dynamics matter.
