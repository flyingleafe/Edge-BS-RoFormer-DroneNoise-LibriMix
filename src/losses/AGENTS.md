# src/losses — consolidated training losses

Every loss in the repo lives here, once. Extracted from the deleted
`train.py`/`train_rps_predictor.py`/`train_noise_generation.py` and
`src/models/generative/losses.py` during the 2026-07 refactor.

## Structure

Each module has (a) pure tensor functions (the math) and (b) thin **Frame
adapters** — classes with `requires_pred: FrameSpec`, `requires_target:
FrameSpec` and `__call__(pred: td.Frame, target: td.Frame) -> Tensor`.
The specs are what pre-run validation checks (`src/training/validate.py`).

| Module | Contents |
|---|---|
| `spectral.py` | `MultiScaleSTFT` (DDSP-style, was generative/losses.py) + `AuraMRSTFTLoss` (auraloss wrapper, was inlined in old train.py) |
| `amplitude_target.py` | `AmplitudeTarget` / `AmplitudeTargetLoss` — the Vold-Kalman objective: log-L1 on the per-`(mic, rotor, k, frame)` amplitude envelopes of `decomp-frames-v*`, a hinge barrier on the harmonics above each recording's decomposition ceiling (unsupervised directions run away without it — measured 46 dB), and log-L1 on the residual's per-mic band power. Never synthesizes audio, so line decoherence leaves the training problem |
| `spectral_likelihood.py` | `rice_nll` / `SpectralLikelihood` — the objective for generators with a **stochastic** branch. `spectral.py` compares one *realization* to another, which is right for the harmonic bank and wrong for gusts and broadband residual: the L1 magnitude minimizer of a Rayleigh bin is its median, so any purely stochastic component is fitted **-1.6 dB low** (a factor `ln 2`) at any capacity, and its gradient is draw-to-draw noise. Here the model predicts a *distribution* — coherent mean + analytic variance, nothing sampled — and the unidentifiable absolute rotor phase is marginalized out, giving a Rice likelihood that degenerates to the **Whittle** likelihood on pure-noise bins and to magnitude matching as the variance vanishes. `split_coherence` moves partially-decohered harmonic power from mean to variance (the generator-side counterpart of the VK decoherence budget). |
| `pit.py` | `pairwise_mse`, `pit_mse_loss`, `segmented_pit_mse` + adapters. **Guard: k ≤ 8** — PIT materializes k! permutations; an unbatched `(K, T)` tensor read as `(B, K)` once inferred k=T and OOM-froze a machine. Shape validation is load-bearing; do not remove. |
| `masked.py` | Quantile-masked MSE (robust clipping, was old train.py fallback). Adapter normalizes shapes to `(G=1, B, rest…)` — see docstring; the naive port produced a constant-zero loss. |
| `salience.py` | BCE-on-salience with pos_weight (salience RPS models) |
| `regularizers.py` | `smoothness_penalty` / second differences — THE one implementation (was duplicated in two trainers) |
| `composite.py` | Weighted sum-of-losses combinator (replaces the old `choice_loss` flag menu); conf/loss entries compose through it |
| `_common.py` | `get_tensor`, canonical entry names, `Loss` protocol |

## Fitting a stochastic component: three traps

`spectral_likelihood.py` carries the scars; read these before adding any
likelihood-style objective.

1. **Never put a `sqrt`/square round trip in the graph.** It is the identity
   analytically, but autograd walks it stepwise and `d(sqrt)/dx` is infinite at
   zero while `d(x^2)/dx` is zero — `inf * 0 = NaN` wherever the model predicts
   exact silence. This is why the model/loss interface is **power**
   (`noise_psd`), not magnitude. The loss *value* stays finite when this bites,
   so it presents as a NaN gradient on one branch only.
2. **A learned variance needs a floor, relative to the data's own scale.** The
   NLL is unbounded below: where the mean fits, `log sigma2 -> -inf` beats
   `(r-a)^2/sigma2`. An *absolute* floor is inert on loud clips and dominant on
   quiet ones, so the floor is a fraction of each clip's mean observed power.
3. **`beta`-NLL is not a proper scoring rule.** It rescales the loss value, so
   it preserves the optimum only for a per-bin-flexible sigma and shifts the
   argmin when sigma is shared. Train with `beta=0.5`; **score with `beta=0`.**

Separately, the quadratic term is unbounded *above* when the mean is far off
(~1e12 at a random init), which no floor fixes — such objectives need a
warm start from a magnitude-trained model, not a better floor.

## Adding a loss

New module or extend an existing one; expose a Frame adapter declaring its
specs; add a `conf/loss/<name>.yaml`. Nothing else — the trainer discovers
it through config composition. Future SSL objectives follow exactly this
path (design doc §"Future expansions").
