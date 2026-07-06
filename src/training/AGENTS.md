# src/training — unified training framework

The machinery behind the single root `train.py` / `eval.py` (Hydra). Design
contract: `docs/refactor-unified-framework.md`. There is exactly ONE
training loop in this repo; task differences live in configs, task codecs
(`src/tasks/codecs.py`), and loss/metric selection — never in forked
trainer scripts.

## Files

| File | Role |
|---|---|
| `config.py` | Dataclass structured configs registered in Hydra ConfigStore (`RootConfig`, Data/Model/Loss/Metrics/Optim/Wandb/Artifacts/Lora). `instantiate_target` resolves `_target_` entries and normalizes rate params to reduced `(num, den)` tuples — required for exact equality with `GridIndex` rates. |
| `validate.py` | Pre-run spec validation: dataset ⊇ model input; model output ∪ dataset ⊇ every loss/metric requirement; monitor metric exists; one-batch CPU smoke test. Runs at the start of train/eval; `validate_only=true` exits after it. |
| `loop.py` | The generic loop: map-style + iterable (online-mix, `samples_per_validation`) datasets, AMP, grad accumulation/clipping, optimizer factory, ReduceLROnPlateau + early stop on the monitor metric, checkpointing, wandb logging. |
| `artifacts.py` | `ArtifactStore` — uploads checkpoints + selected validation samples to Cloudflare R2 (bucket `ml-data`, `artifacts/<experiment>/...`), env from `.env` (`R2_ACCOUNT_ID` + AWS keys), via `s3fs`. Uploads never crash training; disabled/missing-env → no-op. |
| `val_logging.py` | SNR-stratified validation-sample logging: audio triples (mixture/target/output) for speech tasks, mixture + RPS-overlay figure for `rps_prediction`; goes to wandb AND R2. Takes a logger interface — reusable by future multi-model training schemes. |
| `lora.py` | LoRA config seam (`maybe_apply_lora`). Disabled by default; enabling raises NotImplementedError pointing at the legacy implementation (`git show d94ce9f:train.py`). |

## Future-expansion seams (see design doc §"Future expansions")

- Multi-source under-annotated datasets: keep dataset construction behind
  DataConfig; nothing outside it may assume specific Frame entries.
- SSL objectives: losses are declarative; per-source loss applicability is
  the planned extension — don't hardcode loss↔dataset couplings.
- Joint/adversarial schemes: keep build→step→validate behind the current
  narrow seam; checkpointing/logging/artifacts must stay scheme-agnostic.

## Gotchas

- Dropped on purpose: multi-GPU `DataParallel` validation (decision
  2026-07-03), the 8-way `choice_loss` flag menu (→ `losses.composite`).
- Tests: run ONE file at a time under a hard cap —
  `bash -c "ulimit -v 3000000; timeout 180 uv run pytest tests/training/<file> -q -x"`.
  Development machines here are small; unbounded pytest has frozen a box.
- `tests/training/test_artifacts_r2_integration.py` hits the real bucket
  (skips without `R2_ACCOUNT_ID`); it cleans up after itself.
