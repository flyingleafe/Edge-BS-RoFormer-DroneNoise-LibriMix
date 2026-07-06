# Data & Artifacts — R2 via DVC + wandb Artifacts

Single page. Read once, bookmark, move on.

## What lives where

| Thing                | Tool              | Where                                   |
|----------------------|-------------------|-----------------------------------------|
| Datasets (raw/processed) | **DVC** + R2 remote | `s3://ml-data/datasets/` + `.dvc` files in git |
| Checkpoints          | **wandb Artifacts** | wandb servers, linked to the training run |
| Metrics / logs / run metadata | wandb (already in use) | wandb |
| Eval audio samples   | local `results/<job>/eval/`; push to wandb or R2 manually if needed | |

One bucket (`ml-data`, prefix `datasets/`), one Cloudflare account.

## One-time setup per machine

```bash
# 1. Install deps (uv does this via pyproject.toml).
uv sync

# 2. Put credentials in .env (never committed).
cp .env.example .env
# edit: fill AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_ACCOUNT_ID,
#       WANDB_API_KEY, and keep AWS_DEFAULT_REGION=auto (R2 rejects real
#       AWS region names; only "auto" works).

# 3. Point DVC at your R2 endpoint (gitignored via .dvc/config.local).
dvc remote modify --local r2 endpointurl \
    "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# 4. Sanity check.
dvc remote list                 # should show 'r2 s3://ml-data/datasets (default)'
dvc status --cloud              # should connect without errors
```

On the laptop, `direnv` loads `.env` automatically (see `.envrc`). On a GPU
server, source it yourself (`set -a; . ./.env; set +a`) before invoking
`dvc pull` or your training command.

> **Gotcha**: `.dvc/config` has a `region` field, but **dvc-s3 ignores it**.
> The region must come from the `AWS_DEFAULT_REGION` env var (= `auto` for
> R2). Hence the entry in `.env`.

## Workflow

### Overnight on CPU server — process and publish a dataset

```bash
# (whatever produces the data)
python scripts/create_dregon_librimix.py --output datasets/DREGON-LM ...

# Track it with DVC (writes datasets/DREGON-LM.dvc, ignores the dir in git).
dvc add datasets/DREGON-LM

# Push data to R2 and the pointer to git.
dvc push
git add datasets/DREGON-LM.dvc datasets/.gitignore
git commit -m "dataset: DREGON-LM v1"
git push
```

Every subsequent `dvc add` on the same dataset dir produces a new content hash
→ git diff on the `.dvc` file shows the version change. `dvc push` uploads
**only the changed blobs** (content-addressed, deduplicated).

### Morning on GPU server — train

```bash
git pull                           # get the latest .dvc pointers
dvc pull                           # fetch any datasets missing locally
python train.py experiment=b1_dccrn_rps_dregon
# On a Slurm cluster, wrap the last line:
#   ./scripts/sbatch.sh -- python train.py experiment=b1_dccrn_rps_dregon
```

Run `dvc pull` for any dataset that is missing locally but has a `.dvc` file in
the repo. At training end, the best checkpoint is automatically logged as a
wandb artifact (aliased `best` and `latest`).

### Evening on laptop — analyze

```bash
# Pull the dataset (only needed if you want to inspect raw data locally).
git pull
dvc pull datasets/DREGON-LM.dvc

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

**DVC cache** (default `.dvc/cache/` per project, shared via `dvc cache dir`).
Content-addressed, deduplicated. When it grows:

```bash
dvc gc --workspace          # remove blobs not referenced by HEAD
dvc gc --all-branches       # remove blobs not referenced by any branch
```

**wandb cache** (default `~/.cache/wandb/artifacts/`). Override with
`WANDB_CACHE_DIR=/big/disk/wandb`. To cap size:

```bash
export WANDB_CACHE_SIZE=50GB   # wandb enforces LRU above this
```

**True on-the-fly streaming** — if a dataset is too big to fit locally and you
want files fetched lazily as the DataLoader reads them, skip `dvc pull` and
mount R2 directly with `rclone`:

```bash
# once per machine
cat >> ~/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = $AWS_ACCESS_KEY_ID
secret_access_key = $AWS_SECRET_ACCESS_KEY
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
EOF

# per-session
rclone mount r2:hns-research/ ./r2-mount \
    --vfs-cache-mode full \
    --vfs-cache-max-size 50G \
    --daemon
# then symlink/point datasets at ./r2-mount/<path>
```

`--vfs-cache-max-size` gives you the LRU-capped local mirror behavior
natively, no custom code.

## Training artifacts (checkpoints + val samples) → R2

Since the unified `train.py`, checkpoints **and** a selection of validation
samples (audio + figures) also upload directly to the Cloudflare R2 bucket
`ml-data`, in addition to the wandb-artifact checkpoint flow above. This is
handled by `src/training/artifacts.py::ArtifactStore` — not DVC, and not the
`dvc`/`wandb.use_artifact` flow described above.

- **Where**: `s3://ml-data/artifacts/<experiment_name>/checkpoints/<filename>.ckpt`
  and `s3://ml-data/artifacts/<experiment_name>/val_samples/epoch_<N>/...`
  (`bucket`/`prefix` are configurable; defaults are `ml-data`/`artifacts`).
- **Client**: `s3fs.S3FileSystem` (not `boto3`) pointed at the same R2
  S3-compatible endpoint used by DVC (`https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`).
- **Credentials**: the same `.env` vars as the DVC setup above —
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

- **`dvc push` is per-commit, not per-session.** After `dvc add`, remember to
  `git add <name>.dvc` AND push both git and DVC — they're two separate remotes.
- **DVC cache type** is configured to `reflink,hardlink,symlink,copy` in
  `.dvc/config`. On NFS or cross-filesystem setups it falls back to `copy`
  (and doubles disk). Put `datasets/` and `.dvc/cache/` on the same filesystem.
- **Reproducibility**: a job's dataset version is fully captured by the git
  commit of its `.dvc` files. Pin experiments to a commit in the job record
  and you can always recreate.
- **wandb artifact size**: if checkpoints get huge (~GBs) and push free-tier
  limits, switch to *reference artifacts* that point at R2:
  `artifact.add_reference("s3://hns-research/checkpoints/...")`.
