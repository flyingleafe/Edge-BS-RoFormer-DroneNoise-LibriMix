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

```bash
# (whatever produces the data)
python scripts/create_dregon_librimix.py --output datasets/DREGON-LM ...

# Ingest as a new version of the dataset (content-addressed shards → R2).
dload commit DREGON-LM --from datasets/DREGON-LM

# Pin the repo to the new version and commit the lock change.
dload pin DREGON-LM
git add dload.lock
git commit -m "dataset: DREGON-LM new version"
git push
```

Every `dload commit` on the same dataset name produces a new version; the git
diff on `dload.lock` shows the version change. `dload info NAME` shows the
manifest and version history; `--recipe FILE` on commit stores the producing
script/config verbatim alongside the version.

### Morning on GPU server — train

```bash
git pull                           # get the latest dload.lock pins
dload pull DREGON-LM               # fetch shards into the local cache
python train.py experiment=b1_dccrn_rps_dregon
# On a Slurm cluster, wrap the last line:
#   ./scripts/sbatch.sh -- python train.py experiment=b1_dccrn_rps_dregon
```

`dload pull NAME` fetches the pinned (or latest) version of a dataset into the
local shard cache. How training code consumes dload-managed datasets (the
streaming/`Pipeline` layer, `dload.Repository.open()`) is still under
construction; until it lands, the datasets under `data/` are used as plain
local directories.
<!-- TODO(dload-docs): expand after streams layer lands -->

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

**dload shard cache** — content-addressed, deduplicated. `dload status` shows
usage; `dload cache` subcommands manage it. `dload gc` deletes *remote* shards
referenced by no manifest of any dataset (destructive housekeeping — use
deliberately).
<!-- TODO(dload-docs): expand after streams layer lands -->

**wandb cache** (default `~/.cache/wandb/artifacts/`). Override with
`WANDB_CACHE_DIR=/big/disk/wandb`. To cap size:

```bash
export WANDB_CACHE_SIZE=50GB   # wandb enforces LRU above this
```

**On-the-fly streaming** — dload's streaming layer (lazy shard fetch as the
DataLoader reads, LRU-capped local cache) is the intended replacement for the
old rclone-mount trick; it is still under construction.
<!-- TODO(dload-docs): expand after streams layer lands -->

## Training artifacts (checkpoints + val samples) → R2

Since the unified `train.py`, checkpoints **and** a selection of validation
samples (audio + figures) also upload directly to the Cloudflare R2 bucket
`ml-data`, in addition to the wandb-artifact checkpoint flow above. This is
handled by `src/training/artifacts.py::ArtifactStore` — not dload, and not the
`dload pull`/`wandb.use_artifact` flow described above.

- **Where**: `s3://ml-data/artifacts/<experiment_name>/checkpoints/<filename>.ckpt`
  and `s3://ml-data/artifacts/<experiment_name>/val_samples/epoch_<N>/...`
  (`bucket`/`prefix` are configurable; defaults are `ml-data`/`artifacts`).
- **Client**: `s3fs.S3FileSystem` (not `boto3`; s3fs is a direct project
  dependency) pointed at the same R2 S3-compatible endpoint used by dload
  (`https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`).
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
