# Plan: DN-LM + DREGON-LM as dload derived datasets

Status: **designed, not implemented** (2026-07-07). Blocked on one small
upstream dload change (below). Companion to `docs/data-and-artifacts.md`.

## The feature (dload 0.2.0)

`Repository.derive(name, pipeline, *, tag=None)` memoizes a finite,
deterministic `Pipeline`: it fingerprints the DAG (source dataset *versions*,
all params/seeds, transform functions by module+qualname) and keeps a
derivation ref at `datasets/<name>/derived/<fingerprint>` → version id.
Fingerprint hit = instant `Dataset`, no recompute; miss = run once, commit as
a normal content-addressed version, publish the ref. Python-API only (no
CLI). Unseeded randomness, unbounded repeat, lambdas, and non-JSON params are
rejected at fingerprint time. `dload.from_iterable(partial(module_level_fn,
json_spec))` is a legal fingerprintable source — the bridge for our
generation scripts, with resolved parent pins carried *inside* the spec dict
(from_iterable pipelines have no SourceNodes, so `derived_from` is empty).

## Design (agreed shape)

- `src/data_processing/derivations.py`: module-level generator functions
  (`generate_dregon_lm_split(spec)`, `generate_dn_lm(spec)`) yielding
  sample-dir-v1 `(key, {field: bytes})` samples (incl. the `_meta` sample),
  reading parents via `resolve_source()`/dload URIs — never `data/` paths —
  plus a registry of JSON specs (all flags + seed + `recipe_version` +
  resolved parent pins). Specs are the durable, reviewed derivation
  declarations.
- `scripts/derive.py`: registry lookup → `repo.derive` → `dload pin`.
- `scripts/create_dregon_librimix.py` / `create_dataset.py`: split the
  generation core (yields samples) from the disk-writing CLI wrapper; sort
  all glob listings. Biggest chunk (~2 days).
- **Migration of the existing 15 DREGON-LM-\* pins: adopt-in-place, not
  re-derive.** Re-running would NOT dedup (shard bytes depend on sample
  bytes/order/cut points; our RNG is not byte-stable across environments) and
  would upload a near-full second copy. Instead write the derivation ref by
  hand (`repo.remote.put_bytes(datasets/<name>/derived/<fp>, pinned_version)`)
  for the actively-used four (DREGON-LM-V4-{train,valid},
  -V4-michaels-{train,valid}), marked "adopted, unverified" in the registry;
  leave V2/V3/test/rps_* as plain pinned historical datasets.
- DN-LM (currently absent from the bucket): pure win — declare specs for
  DN-LM-{train,valid}; first `scripts/derive.py DN-LM-train` materializes and
  shares it; unblocks the A1/A2 replication rows.

## Blocker (upstream, dload 0.2.1)

`derive()` hardcodes the manifest meta to `{derived_from, fingerprint, tag}`
and forwards no `meta=`/`recipe=` to `commit()`. Our consumers *require*
manifest meta: `meta["fields"]`/`meta["meta_sample"]` (ensure_local file
reconstruction) and `meta["layout"]` (tdframe decode dispatch). Needed
upstream: merge user `meta=` under the derivation keys + forward `recipe=`.
Half a day in dload, then this plan is unblocked.

## Gotchas to respect when implementing

- Function identity is by *name*: editing a generator's behavior silently
  serves the stale snapshot. Bump the spec's `recipe_version` on any
  behavioral change (review-enforced convention).
- Cross-machine byte-determinism of our generators is not guaranteed
  (numpy RNG order, librosa versions). Benign for memoization (ref settles
  once), but derive from one designated box and keep listings sorted.
- We are NOT trying to reproduce historic dataset bytes — new fingerprints
  mean new datasets; adoption-in-place is the bridge to the historical pins.

Effort: ~4–5 days total, upstream change first.
