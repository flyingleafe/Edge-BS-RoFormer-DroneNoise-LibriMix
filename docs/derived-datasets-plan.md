# Plan: DN-LM + DREGON-LM as dload derived datasets

Status: **implemented** (2026-07-08). The upstream blocker landed in
**dload-ml 0.3.0** (`derive()` now forwards `meta=`/`recipe=` to `commit()`,
merging user meta under the reserved derivation keys — see dload issue #2).
Companion to `docs/data-and-artifacts.md` § "Derived datasets".

## What shipped (2026-07-08)

- `src/data_processing/derivations.py` — module-level generator functions
  (`generate_dregon_lm_split`, `generate_dn_lm_split`) yielding `sample-dir-v1`
  samples, + the `SPECS` registry (frozen JSON specs with params/seed/
  `recipe_version`/resolved parent pins), + `build_pipeline`/`dataset_meta`/
  `fingerprint` helpers. Kept torch-free (heavy cores imported lazily) so
  fingerprinting/adoption run on any box.
- Shared per-sample cores factored out of the disk-writing CLIs and reused by
  the generators (no duplicated mixing math, RNG order preserved so the CLIs'
  output is unchanged): `render_multichannel_sample` in
  `scripts/create_dregon_librimix.py`, `mix_dn_lm` in `scripts/create_dataset.py`
  (whose `torchcodec` import is now deferred to the HF path).
- `scripts/derive.py` — `list` / `derive` / `adopt` driver.
- `tests/data_processing/test_derivations.py` — specs integrity, fingerprint
  stability/uniqueness, `recipe_version` sensitivity, wav encoding, and a
  streams round-trip proving the emitted convention decodes + reconstructs.
- **Adopted the four active V4 pins in place** (`DREGON-LM-V4-{train,valid}`,
  `-michaels-{train,valid}`): derivation refs now point at the historical
  uploads. Verified each pinned manifest is `layout='sample-dir-v1'` first.
- **DN-LM specs declared but PROVISIONAL** (`recipe_version 1`): the drone-noise
  source (`drone_audio`) / any drone-only filtering must be reviewed before the
  first `scripts/derive.py derive DN-LM-train`, which must run on a big box.

Deviations from the original design below: the pure cores were **reused from**
the CLIs via a small `sys.path` shim rather than hoisted into the package (the
~2-day refactor); a future cleanup can hoist them and drop the shim. The
michaels/real-valid splits are **adopt-only** (the synthesized generator does
not reproduce composed-noise-pool / raw-clip splits) — fine, since adoption
never runs the generator.

---

_Original plan (2026-07-07), kept for context:_

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

## Blocker (RESOLVED in dload-ml 0.3.0)

`derive()` used to hardcode the manifest meta to `{derived_from, fingerprint,
tag}` and forward no `meta=`/`recipe=` to `commit()`. Our consumers *require*
manifest meta: `meta["fields"]`/`meta["meta_sample"]` (ensure_local file
reconstruction) and `meta["layout"]` (decode dispatch). **Fixed upstream**
(dload issue #2): `derive(name, pipe, *, meta=None, recipe=None, ...)` now
forwards both to `commit()`, merging user `meta` under the reserved derivation
keys (`derived_from`/`fingerprint`/`tag` win on conflict). `derivations.py`
passes the `sample-dir-v1` meta via `dataset_meta(name)`.

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
