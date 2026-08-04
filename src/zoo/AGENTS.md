# src/zoo/ — Model Zoo

Checkpoint discovery + one-call model loading (docs/refactor-2026-08-plan.md
§ 3.3). R2 (the `training.artifacts.ArtifactStore` bucket/prefix) is the
source of truth; the local index is the **gitignored**
`<repo-root>/.checkpoints-cache.json` (schema 1).

## Public API (`import zoo`)

| Call | What it does |
|------|--------------|
| `zoo.model_types()` | Re-export of `models.registry.model_types()` — every registered model type. |
| `zoo.refresh(full=False)` | List the R2 artifact store into the cache. Incremental by default: only new top-level prefixes and experiments with new `conf/experiment/*.yaml` names are re-listed; `full=True` re-lists everything. Returns `CacheInfo`. |
| `zoo.checkpoints(task=None, max_age_s=86400)` | Cached rows (`experiment`, `task`, `files`/`sizes`/`mtimes`, `has_config`, `metrics`). Auto-refreshes when stale **and** R2 creds are in `.env`; otherwise warns and returns the cached data — never hard-fails offline. |
| `zoo.load(name, ckpt="best", device="cpu")` | Hydra-compose `experiment=<name>`, instantiate the model, resolve the checkpoint (local `results/<name>/`, else `r2://` via `utils.checkpoints.resolve_checkpoint_uri`), return a `FrameModel`. |
| `zoo.FrameModel` | `(model, codec, task)` as one callable: `td.Frame` in → `td.Frame` out; accepts a single unbatched sample (collate → codec → slice back) or an already-batched Frame. |

## Files

| File | Purpose |
|------|---------|
| `cache.py` | Cache file schema + `refresh`/`checkpoints`; R2 listing (the repo's one `list_objects_v2` call site). Bucket/prefix/key-root conventions are **derived from `ArtifactStore`**, never re-declared. |
| `frame_model.py` | `FrameModel` + `load` (the `scripts/rps_predictor_vk_eval.py::load_model` recipe, packaged). |

## Gotchas

- Eval metrics appear in `checkpoints()` rows only after `eval.py` has run
  for that experiment (it uploads `eval/metrics.json` next to the
  checkpoints) **and** a subsequent `zoo.refresh()` of that prefix.
- Incremental refresh does not detect new checkpoint files inside an
  already-known experiment prefix — run `refresh(full=True)` (or wait for a
  cheap signal) when you need those picked up.
- Tests inject `client=`/`cache_path=`/`conf_dir=` — no network in
  `tests/zoo/`.
