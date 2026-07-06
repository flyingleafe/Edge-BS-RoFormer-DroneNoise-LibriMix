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
| `pit.py` | `pairwise_mse`, `pit_mse_loss`, `segmented_pit_mse` + adapters. **Guard: k ≤ 8** — PIT materializes k! permutations; an unbatched `(K, T)` tensor read as `(B, K)` once inferred k=T and OOM-froze a machine. Shape validation is load-bearing; do not remove. |
| `masked.py` | Quantile-masked MSE (robust clipping, was old train.py fallback). Adapter normalizes shapes to `(G=1, B, rest…)` — see docstring; the naive port produced a constant-zero loss. |
| `salience.py` | BCE-on-salience with pos_weight (salience RPS models) |
| `regularizers.py` | `smoothness_penalty` / second differences — THE one implementation (was duplicated in two trainers) |
| `composite.py` | Weighted sum-of-losses combinator (replaces the old `choice_loss` flag menu); conf/loss entries compose through it |
| `_common.py` | `get_tensor`, canonical entry names, `Loss` protocol |

## Adding a loss

New module or extend an existing one; expose a Frame adapter declaring its
specs; add a `conf/loss/<name>.yaml`. Nothing else — the trainer discovers
it through config composition. Future SSL objectives follow exactly this
path (design doc §"Future expansions").
