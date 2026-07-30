# Data & Artifacts — R2 via dload + wandb Artifacts

Single page. Read once, bookmark, move on.

## What lives where

| Thing                | Tool              | Where                                   |
|----------------------|-------------------|-----------------------------------------|
| Datasets (raw/processed) | **dload** (PyPI `dload-ml`) + R2 remote | `s3://ml-data-new/` + `dload.toml`/`dload.lock` in git |
| Checkpoints          | **wandb Artifacts** | wandb servers, linked to the training run |
| Metrics / logs / run metadata | wandb (already in use) | wandb |
| Eval audio samples   | local `results/<job>/eval/`; push to wandb or R2 manually if needed | |

One bucket (`ml-data-new`), one Cloudflare account. The remote endpoint and
bucket are committed in `dload.toml`; per-dataset version pins live in
`dload.lock` (managed by `dload pin` / `dload unpin`).

## One-time setup per machine

```bash
# 1. Install deps (uv does this via pyproject.toml; dload-ml is a direct dep).
uv sync

# 2. Put credentials in .env (never committed).
cp .env.example .env
# edit: fill AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_ACCOUNT_ID,
#       WANDB_API_KEY, and keep AWS_DEFAULT_REGION=auto (R2 rejects real
#       AWS region names; only "auto" works).

# 3. Sanity check: resolved config, cache usage, remote reachability.
dload status
dload ls          # list datasets in the remote
```

On the laptop, `direnv` loads `.env` automatically (see `.envrc`). On a GPU
server, source it yourself (`set -a; . ./.env; set +a`) before invoking
`dload pull` or your training command.

## Workflow

### Overnight on CPU server — process and publish a dataset

**Raw trees** are committed once with the CLI: `dload commit NAME --from
data/NAME` (key = relpath minus extension, field = the extension; the CLI does
not skip hidden files), then referenced by a `data_processing.sources` registry
entry's `raw_dataset`.

**Everything else is a derivation** — declared as a frozen spec in
`data_processing.derivations.SPECS` and materialized only through the single
driver (full detail: `src/data_processing/AGENTS.md` § "Publishing datasets to
dload", architecture: `docs/refactor-data-pipelines.md`):

```bash
python scripts/derive.py list --check-remote -v   # specs + fingerprints + refs
python scripts/derive.py derive <NAME>            # materialize + commit + pin
python scripts/derive.py adopt  <NAME> --commit   # ref an existing historical pin
```

Manifest layouts: `sample-dir-v1` (DREGON-LM / DN-LM mixes), `tdframe-v1`
(rich per-recording Frames), `raw-files` / `pcm16-mono-v1` (byte passthrough /
decoded PCM cache).

Whatever the route, finish by pinning:

```bash
dload pin DREGON-LM-V4-train
git add dload.lock
git commit -m "dataset: DREGON-LM-V4-train new version"
git push
```

Every commit on the same dataset name produces a new version; the git diff on
`dload.lock` shows the version change. `dload info NAME` shows the manifest
and version history; `--recipe FILE` (or the Python-API equivalent) stores the
producing script/config verbatim alongside the version.

### Derived datasets (memoized generation pipelines)

Our mixed datasets (DREGON-LM, DN-LM) are also expressible as dload **derived
datasets** (`Repository.derive`, dload ≥ 0.3.0): a finite deterministic
pipeline that dload runs once, commits as a normal content-addressed version,
and memoizes by *fingerprint* — every later caller of the identical pipeline
hits the same snapshot instead of recomputing. The durable, reviewed
declarations live in `src/data_processing/derivations.py` (`SPECS`): one frozen
JSON spec per dataset (all generation params + seed + `recipe_version` +
resolved parent pins), plus module-level generator functions that reuse the
same per-sample mixing cores the disk-writing CLIs use. Driver:
`scripts/derive.py`.

```bash
python scripts/derive.py list --check-remote -v   # specs, fingerprints, ref status
python scripts/derive.py derive DN-LM-train        # materialize + pin a new dataset
python scripts/derive.py adopt DREGON-LM-V4-train  # dry-run: adopt an existing pin
python scripts/derive.py adopt DREGON-LM-V4-train --commit
```

Two modes:

- **`derive`** materializes a genuinely new dataset (e.g. **DN-LM**, absent from
  the bucket) — runs the generator, commits, forwards the `sample-dir-v1`
  manifest meta + the spec as recipe, and pins it. Must run on a box with the
  parent data + enough disk (materialization streams LibriSpeech/DREGON via
  dload); the small-RAM dev box cannot. Submit it via omnirun if needed.
- **`adopt`** (adopt-in-place) points a dataset's *derivation ref*
  (`datasets/<name>/derived/<fingerprint>`) at its existing `dload.lock` pin
  instead of re-materializing. Used for the historical `DREGON-LM-V4-*` uploads:
  re-deriving would upload a near-duplicate (our RNG is not byte-stable across
  environments). Offline except the ref write (additive, reversible, uploads
  nothing); **dry-run by default**, writes with `--commit`. The four active V4
  pins (`DREGON-LM-V4-{train,valid}`, `-michaels-{train,valid}`) are adopted.

**Bump a spec's `recipe_version`** on any behavioral change to a generator — the
fingerprint keys on the spec, so an unbumped edit silently serves the stale
snapshot. `derivations.py` stays importable without torch, so fingerprinting /
`adopt` run anywhere. Full design + gotchas: `docs/derived-datasets-plan.md`.

### Morning on GPU server — train

```bash
git pull                           # get the latest dload.lock pins
python train.py experiment=b1_dccrn_rps_dregon
# Remote GPU (Slurm / Colab / Kaggle): submit the same command via omnirun —
# see "Job running (omnirun)" below.
```

There is usually no separate download step: training code consumes
dload-managed datasets directly (next section), fetching shards lazily into
the local cache. `dload pull NAME` still exists for eagerly prefetching the
pinned version or inspecting data by hand.

### How training code consumes dload datasets

`src/data_processing/streams.py` is the project's only seam between dload's
`(key, {field: bytes})` sample world and the `td.Frame` data model (full API
in its module docstring):

- **`DloadFrameDataset`** — a torch `IterableDataset` usable directly as a
  Hydra `_target_` in `conf/data/*.yaml`. Streams a dataset by name; the
  version resolves via the `dload.lock` pin (override with `version:`).
  Decoding dispatches on the manifest `meta.layout`: `tdframe-v1` datasets
  decode via the generic Frame codec (`sample_to_frame`), everything else via
  the DREGON-LM file-stem convention (`decode_dregon_lm`). Stream knobs:
  `shuffle` (False / True / int seed), `shuffle_buffer`, `prefetch`, `take`,
  `repeat`. Reference config: `conf/data/dregon_lm_v4_stream.yaml` — train is
  an infinite shuffled stream (`repeat: true`), so the experiment **must set
  `samples_per_validation`** (same contract as online-mix). RAM note: the
  shuffle buffer holds *raw* samples (~0.8 MB each for V4) — 512 ≈ 400 MB;
  keep it ≤512 on small machines.
- **Pipeline combinators** — `to_frames` / `frame_windows` / `mix_frames` /
  `resample_frames` compose dload `Pipeline`s into windowed/mixed Frame
  streams; `iter_published_frames(name)` iterates a published `tdframe-v1`
  dataset as decoded Frames.
- **`dload:` URIs** — `resolve_source()` lets any path-shaped config knob
  (`data_dir`, `root`, `dregon_dir`, `michaels_dir`) accept
  `dload:NAME[@VERSION][/subpath]`: the dataset is materialized once into the
  cache (`ensure_local`, version-addressed, idempotent) and the local path is
  returned. Path-based loaders need no dload awareness.
- **`frames:NAME[@VERSION]` specs** — `noise_rps_dataset` and online-mix
  `kind: frames` noise sources consume the published rich-frame datasets
  (`DREGON-frames`, `michaels-frames`) directly; see
  `src/data_processing/AGENTS.md`.

Measured on the V4 stream (laptop ↔ R2): cold time-to-first-sample ~6 s per
shard, warm epochs ~18× faster than cold, ~48 MB/s sustained download;
bounded-budget cache eviction verified working.

At training end, the best checkpoint is automatically logged as a wandb
artifact (aliased `best` and `latest`).

### Evening on laptop — analyze

```bash
# Pull the dataset (only needed if you want to inspect raw data locally).
git pull
dload pull DREGON-LM

# Pull trained checkpoints from wandb (cached by wandb under ~/.cache/wandb).
python -c "
import wandb
api = wandb.Api()
for run_id in ['abc123', 'def456']:
    art = api.artifact(f'flyingleafe/harmonic-noise-suppression/model-{run_id}:best')
    print(run_id, '→', art.download())
"

# Or, programmatically inside a notebook:
# run = wandb.init(...)
# art = run.use_artifact('flyingleafe/harmonic-noise-suppression/model-abc123:best')
# ckpt_path = Path(art.download()) / 'best_model.ckpt'
```

## Disk management

**dload shard cache** — content-addressed, deduplicated, eviction-bounded.
`dload status` shows usage; `dload cache` subcommands manage it. `dload gc`
deletes *remote* shards referenced by no manifest of any dataset (destructive
housekeeping — use deliberately). Two env vars control the cache (put them in
`.env`, see `.env.example`):

- `DLOAD_CACHE_DIR` — cache root; put it on a big/fast partition on GPU boxes.
- `DLOAD_CACHE_BUDGET` — local eviction budget (e.g. `20G`, or `unlimited`).

Per-machine recommendations:

| Machine | Setting |
|---|---|
| Laptop / small box | `DLOAD_CACHE_BUDGET` ≥ `1.5G` — the floor for streaming (must hold the `prefetch=3` shard window); keep `shuffle_buffer` ≤ 512 |
| Apocrita | `DLOAD_CACHE_DIR=/gpfs/scratch/acw592/dload-cache`, `DLOAD_CACHE_BUDGET=unlimited` (3 TB scratch) |
| Colab / Kaggle | ephemeral disks — forward overrides per job via `omnirun submit --env DLOAD_CACHE_BUDGET=...` if needed |

**wandb cache** (default `~/.cache/wandb/artifacts/`). Override with
`WANDB_CACHE_DIR=/big/disk/wandb`. To cap size:

```bash
export WANDB_CACHE_SIZE=50GB   # wandb enforces LRU above this
```

**On-the-fly streaming** — landed; it replaces the old rclone-mount trick.
`DloadFrameDataset` (see "How training code consumes dload datasets" above)
fetches shards lazily as the DataLoader reads and evicts under
`DLOAD_CACHE_BUDGET`. Keep the budget ≥ `1.5G` locally so the prefetch window
fits; on big scratch set it `unlimited` and warm epochs run at local-disk
speed (~18× faster than cold, measured).

## Job running (omnirun)

Remote GPU jobs are submitted with **omnirun** (installed as a `uv tool`).
Backend definitions live in the user-global `~/.config/omnirun/config.toml`:

| Backend | What it is |
|---|---|
| `apocrita-short` | Apocrita Slurm, `gpushort` partition, ≤ 1 h |
| `apocrita-long` | Apocrita Slurm, `sae` partition (GPU), long jobs — GPU-only, don't use for CPU work |
| `apocrita-cpu` | Apocrita Slurm, `compute` partition — **CPU-only**, open-access, ≤ 10 days. For dataset generation / preprocessing (`scripts/derive.py derive`, etc.); submit with `--gpus 0` so GPU partitions stay free |
| `colab` | Colab T4 — needs the local keep-alive daemon running; T4 allocation is a lottery (503s) |
| `kaggle` | Kaggle P100 — kernel source cap ~1 MB, needs the slim-snapshot clone recipe (strip `notebooks/ writing/ tests/ docs/ .pi/ scripts/ uv.lock`, orphan commit, no origin, `env kind = system`) |

The committed repo-level `omnirun.toml` holds job *defaults* only
(`outputs = results/**`, 1 GPU, 1 h, `env kind = auto`).

```bash
omnirun submit --backend apocrita-short --gpus 1 --time 30m --yes -- \
    python train.py experiment=<name>
omnirun ps                    # all jobs;  omnirun status/logs <job> for one
omnirun pull <job>            # collect the job's results/** locally
omnirun backends check        # re-establish the SSH ControlMaster after expiry
```

- Requires a **clean, pushed HEAD** (`--push`/`--dirty` exist; prefer clean —
  `train.py` errors on a dirty tree anyway).
- On the cluster each job runs in a shared worktree
  `$PROJECT_ROOT/.trees/<sha12>`, reusing the checkout's `.venv`
  (`uv sync --frozen`).
- `.env` ships with every job automatically — R2 + WANDB credentials travel,
  so dload streaming and wandb logging work on any backend.
- The `sae` partition (`apocrita-long`) **requires** `account =
  "pilot_sae_gpu"` in the backend config — `sbatch` rejects the submission
  otherwise. `sae` is GPU and account-gated — reserve it for real GPU jobs.
- **CPU-only jobs go to `apocrita-cpu`** (the open-access `compute` partition,
  no account, ≤ 10 days), submitted with `--gpus 0`. Dataset materialization is
  the canonical case — e.g. build DN-LM from its derived-dataset spec:
  ```bash
  omnirun submit --backend apocrita-cpu --gpus 0 --time 3h --yes -- \
      bash -lc 'python scripts/derive.py derive DN-LM-train --no-pin && \
                python scripts/derive.py derive DN-LM-valid --no-pin'
  # then locally, once it succeeds: pull the new versions from R2 into the lock
  dload pin DN-LM-train && dload pin DN-LM-valid && git add dload.lock
  ```
- Shared same-SHA worktree warts: a crashed run's `results/<exp>` dir
  persists in the worktree and **poisons retries at the same SHA**
  (`FileExistsError`) — work around with a `results_root=...` override. Also
  `outputs = results/**` scoops *sibling* jobs' results dirs into every
  `omnirun pull`.
- After the local SSH ControlMaster dies, jobs show **LOST** from stale
  heartbeats — run `omnirun backends check` and verify via `sacct` on the
  cluster; a completed job may stay "lost" in `omnirun ps` while
  `omnirun pull` still works.
- Legacy fallback on an Apocrita login node: `./scripts/sbatch.sh` (and
  `./scripts/sync_results.sh` to rsync results back). Prefer omnirun.

## Training artifacts (checkpoints + val samples) → R2

Since the unified `train.py`, checkpoints **and** a selection of validation
samples (audio + figures) also upload directly to the Cloudflare R2 bucket
`ml-data`, in addition to the wandb-artifact checkpoint flow above. This is
handled by `src/training/artifacts.py::ArtifactStore` — not dload, and not the
`dload pull`/`wandb.use_artifact` flow described above.

- **Where**: `s3://ml-data/artifacts/<experiment_name>/checkpoints/<filename>.ckpt`
  and `s3://ml-data/artifacts/<experiment_name>/val_samples/epoch_<N>/...`
  (`bucket`/`prefix` are configurable; defaults are `ml-data`/`artifacts`).
- **Client**: `boto3.client("s3")` (same library dload uses; boto3 is a
  direct project dependency) pointed at the same R2 S3-compatible endpoint
  used by dload (`https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`).
  Formerly `s3fs`, dropped because its `aiobotocore` pin capped `botocore`
  below dload-ml's `boto3` floor, forcing a `[tool.uv]`
  `override-dependencies` pin that made the project unresolvable by plain
  pip.
- **Credentials**: the same `.env` vars as the dload setup above —
  `R2_ACCOUNT_ID`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. If any are
  missing, or `artifacts.enabled=false` in the Hydra config, every
  `ArtifactStore` method becomes a no-op (one log line, no exception) — a
  broken/absent artifact store never fails training.
- **wandb linkage**: each training run's wandb summary carries `r2/*` URIs
  pointing at the uploaded checkpoint/val-sample objects, so a wandb run page
  is still the starting point for finding an R2 artifact.
- Config seam: `conf/artifacts/r2.yaml` (selected via the `artifacts:` Hydra
  default group in `conf/config.yaml`).

## Gotchas

- **`dload commit` uploads immediately, but the pin is git's job.** After
  `dload commit` + `dload pin`, remember to `git add dload.lock` and push —
  the data remote and git are two separate things.
- **Reproducibility**: a job's dataset version is fully captured by the git
  commit of `dload.lock`. Pin experiments to a commit in the job record and
  you can always recreate.
- **wandb artifact size**: if checkpoints get huge (~GBs) and push free-tier
  limits, switch to *reference artifacts* that point at R2:
  `artifact.add_reference("s3://hns-research/checkpoints/...")`.
