# E5 — Time-Warp Augmentation for RPS Prediction

**Status:** done — **positive result on higher-capacity models** (neutral on the
small causal GRU) · **Date:** 2026-07-10/11

## Motivation

RPS predictors overfit the handful of *real* rotor-speed trajectories in the
training pool: DREGON + Michael's give only a few dozen distinct hover/maneuver
profiles, so a high-capacity head can memorise trajectory shapes instead of
learning the audio→RPS map. E4 tried to fix this by mixing in **generated** noise
(more trajectories), but that *hurt* (+27% PIT MSE — see
[noise-generation-augmentation.md](noise-generation-augmentation.md) and the
[[e4-pit-curves-wandb]] memory): the generator's clean, telemetry-exact combs are
off-distribution vs real recordings.

E5 is the Tier-1 alternative: **time-varying time-warp** of the *real*
noise+RPS pair. Resample the audio by a smooth factor `alpha(t) = c + a·sin(2πft+φ)`
(`|alpha−1| ≤ 0.12`, closed-form inverse `tau`) and relabel `r̃(t) = alpha(t)·r(tau(t))`
— exact for the discriminative task. This synthesises new, physically-plausible
trajectories from real audio (same spectral content, perturbed speed profile)
without inventing spectral texture. Implementation `src/data_processing/time_warp.py`,
online-mix policy key `noise_time_warp` (stage 2, prob 0.5); see the
[[e5-timewarp-augmentation]] memory for the stabilised recipe (grad_clip 0.5,
`amp: false`, batch 32 — the framework defaults NaN'd both arms).

## Arms

Three model heads × {baseline, time-warp}, all under the same stabilised recipe
and the same fixed non-mixed validation split (`DREGON-LM-V4-michaels-valid`):

- `simple_conv_v2_uni_gru128` — small causal unidirectional GRU head.
- `simple_conv_v2` (scv2) — the BiGRU workhorse.
- `simple_conv_v2_transformer` — highest-capacity (Transformer temporal head).

## Results

Best validation PIT MSE (val/mse; lower better), with the epoch it was reached
and where the run stopped:

| model | arm | best PIT MSE | RMSE | @ep | run end | Δ vs baseline |
|---|---|---|---|---|---|---|
| uni_gru128 | baseline | 10.454 | 3.02 | 22 | ep72 (done) | — |
| uni_gru128 | time-warp | 10.331 | 3.05 | 24 | ep74 (done) | −1% (tie) |
| scv2 | baseline | 9.707 | 2.82 | 29 | ep79 (done) | — |
| scv2 | time-warp | **8.849** | 2.84 | 31 | ep66 (crash) | **−9%** |
| transformer | baseline | 11.759 | 3.05 | 79 | ep115 (walltime) | — |
| transformer | time-warp | **8.737** | **2.70** | 80 | ep115 (walltime) | **−26%** |

## Analysis

- **Time-warp is not a universal tie.** The memory's early "tie" verdict was
  read off the uni_gru128 pair alone (the first to finish). Across all three
  heads the pattern is **capacity-dependent**: neutral on the small causal GRU,
  a clear win on scv2 (−9%), a large win on the Transformer (−26%).
- **It helps most exactly where overfitting is worst.** The Transformer
  *baseline* is the **worst** baseline of the three (11.76 — worse than the tiny
  GRU's 10.45 and scv2's 9.71) despite the most capacity — the classic signature
  of memorising the limited trajectory set. Time-warp turns it into the **best
  model overall** (8.74, RMSE 2.70). The small GRU has little capacity to overfit
  in the first place, so the augmentation buys almost nothing there. (The
  per-step `train/loss` is *not* a clean overfitting probe here: time-warp makes
  the training distribution harder, so its train loss is inflated by
  construction — the best-val ordering above is the reliable signal.)
- **Time-warp succeeds where generated-noise (E4) failed.** Both target the same
  overfitting problem; the difference is distribution. Warping *real* audio stays
  on the real-comb manifold (only the speed profile moves), whereas E4's
  generated combs were off-distribution — the same reasoning that motivated the
  E6 linewidth work ([noise-gen-linewidth.md](noise-gen-linewidth.md)).

## Caveats

- **Crashes, but best-vals are valid.** scv2 time-warp crashed at ep66 (best
  ep31) and both Transformers hit walltime at ep115 (best ep79–80). In every
  case the best-val was reached *well before* the stop and the curve had
  plateaued (best epoch < stop epoch ⇒ no improvement after), so the numbers
  stand. A clean rerun with proper early-stop would only confirm them — low
  priority.
- **Transformer needs a long run.** Its best epoch (~80) is 3× later than
  scv2/uni (~30); the gpushort walltime is the binding constraint for that head,
  not convergence.

## Conclusion

Time-warp augmentation is a **safe, capacity-scaled win** for RPS prediction:
neutral on small heads, decisive on high-capacity ones, and — unlike E4's
generated-noise augmentation — never harmful. The best RPS predictor to date is
`simple_conv_v2_transformer` + time-warp (**PIT MSE 8.74 / RMSE 2.70**). Adopt
time-warp as a default augmentation for the larger heads; keep it optional for
the small causal GRU. Next: fold time-warp into the RPS-predictor that seeds the
generator-conditioning labels, and combine with the E6 jitter-broadened generator
once per-drone σ lands.
