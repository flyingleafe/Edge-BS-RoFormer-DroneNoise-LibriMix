# postdoc — Job Orchestration Status

> Read this at the start of any session on job orchestration.
> Last updated: 2026-04-22.

---

## Where things stand

### Code (`src/postdoc/`)

Architecture: **SkyPilot persistent cluster + `sky exec`** per job. One pod mounts `/root/harmonic-noise-suppression` and `/root/.ssh` from the host via k8s hostPath volumes. All jobs share the single venv/datasets/results on disk.

```
postdoc pool-up           # sky ssh up (k3s on vast-server, one-time)
postdoc cluster-up        # sky launch -c postdoc <bootstrap.yaml>
postdoc submit <cmd>      # git preflight → sky exec -d postdoc <exec.yaml>
postdoc list|logs|cancel  # sky queue|logs|cancel postdoc
```

git preflight: require clean tree → push HEAD → remote does `git reset --hard $SHA && uv sync --no-dev`.

### Bootstrap state on vast-server (as of session end)

`postdoc cluster-up` ran and printed the success banner:

```
Cluster name: postdoc
├── To log into the head VM:  ssh postdoc
└── …
```

**But the job queue setup (bootstrap job ID 1) status is unknown** — the watcher process was killed when the user aborted the session. Before submitting any real job, verify:

```bash
postdoc list --all        # check job 1 status: should be SUCCEEDED
postdoc cluster-status    # should be UP
```

If job 1 is FAILED_SETUP, re-run `postdoc cluster-up` (it will teardown and retry).

### Known issues that were fixed in this session

All fixed in code, all in the git history:

| Issue | Fix |
|-------|-----|
| `accelerators: '*:N'` crashes sky (regex invalid) | Use `RTX4070-TI:N` (concrete type) |
| `sky launch` without `--detach-run` hits FileNotFoundError on log tail | Added `--detach-run` |
| `uv sync` builds `pygraphviz` → `libgraphviz-dev` missing in pod | `uv sync --no-dev` |
| `git fetch` fails in pod (no creds) | Mount host `/root/.ssh` read-only via hostPath |
| `git reset --hard` fails ("dubious ownership") | `git config --global safe.directory $REPO_DIR` in bootstrap |
| Ray session mismatch on re-launch after abort | `sky down -y` before retrying |
| `socat`/`nc`/`kubectl` not in Nix env | Added to `flake.nix` (netcat-gnu, not netcat) |
| `~/.ssh/config` read-only symlink (home-manager) | User added `programs.ssh.includes` in home-manager config |
| k3s disk-pressure taint | Deleted old epoch checkpoints (>14d) → freed 10 G |

---

## Honest reflection on the approach

### What works well

- **git-native submit** (clean tree → push → remote reset) is the right invariant. Reproducibility is guaranteed; accidental drift is impossible.
- **Shared hostPath volume** is the correct answer to "how do jobs share 31 GB of datasets" on a single server.
- **`sky exec` on a persistent cluster** is the right SkyPilot primitive for a fixed on-prem host (vs `sky jobs launch` which spawns a fresh pod per job, copying nothing).

### What is wrong with the chosen approach

**SkyPilot on an SSH node pool uses k3s (Kubernetes) as its runtime. This is the wrong level of abstraction for a single fixed GPU server with 2 GPUs.**

The friction we hit is not incidental — it is structural:

- `sky ssh up` installs k3s, takes minutes, requires `socat` + `nc` + `kubectl`
- Every `sky launch` pulls a 2 GB Ray+SkyPilot container image on first run
- The cluster pod runs a full Ray head node (GCS server, dashboard, monitor, raylet, workers)
- Pod/Ray state is fragile: aborting a `sky launch` mid-flight leaves orphaned Ray processes that corrupt the next launch (session ID mismatch)
- k3s enforces disk-pressure thresholds that blocked pod scheduling for hours
- Container image doesn't have system libs the project needs (`libgraphviz-dev`), requiring `--no-dev`
- hostPath ownership/git safe.directory gymnastics for UID namespace differences

**None of this complexity buys anything for a single-server workflow.** SkyPilot's value is multi-cloud portability, spot-instance recovery, and heterogeneous queue management — none of which apply to vast-server today.

### Project philosophy check

From root `AGENTS.md`:

> **Off-the-shelf first.** Before building infra, name ≥2 existing tools and state why each is rejected.

We did pick an off-the-shelf tool. But we failed at the next step: **the tool must fit the scale**. SkyPilot is off-the-shelf for "manage ML workloads across 20 clouds." It is not off-the-shelf for "queue two training jobs on a single SSH server."

> **Bespoke code goes where it's novel.** Queueing, GPU allocation, env setup — delegate to mature tools.

The intention is right. But k3s + Ray is "mature tools" at the wrong abstraction level. A POSIX job queue (e.g., task-spooler, GNU parallel --jobserver, even a tmux FIFO) is also a mature tool — and one that matches the actual workload size.

---

## Decision needed next session

Before writing more code, resolve this fork:

**Option A — Stay with SkyPilot.**
The bootstrap pain is one-time (image cached, k3s up). Day-to-day `postdoc submit` is fast once the cluster is healthy. SkyPilot pays off the moment you want to burst to cloud GPUs or use spot instances. Accept the operational complexity; document the recovery procedures.

**Option B — Revert to thin bespoke queue + keep SkyPilot for cloud-only.**
For vast-server: a 100-LOC shell queue (task-spooler or a tmux FIFO) that SSH-execs jobs. No k8s, no Ray, no container images. For cloud: `sky jobs launch` when needed. Both paths share the git-native submit logic and the POSTDOC_REPO_DIR convention; the transport layer differs.

**My honest read:** Option B is more aligned with the philosophy. The infrastructure cost of SkyPilot on a single on-prem host is not paid back by the scheduling features. If vast-server is ever replaced by a cloud spot instance, revisit.

But this is a design call the user should make explicitly, not the agent by default.

---

## Quick-recovery checklist (next session)

```bash
# 1. Verify cluster health
postdoc cluster-status
postdoc list --all         # job 1 should be SUCCEEDED

# 2. If job 1 FAILED or cluster INIT:
postdoc cluster-down -y
postdoc cluster-up         # image cached → ~2 min

# 3. Smoke test
postdoc submit python -c "import torch; print(torch.cuda.is_available())"
postdoc list               # wait for SUCCEEDED
postdoc logs <id>          # read output on pod

# 4. Real job
git add -A && git commit -m "..."
postdoc submit python train.py --model_type dccrn --config configs/...
```

## Disk headroom (vast-server, as of session end)

~35 G free. DVC-managed `data/` and `datasets/` (31 G) are still present.
Old epoch checkpoints >14d old deleted; best+early-stop kept per run.
If free drops below ~20 G, k3s will re-taint the node. At that point:
`ssh vast-server 'rm -rf ~/harmonic-noise-suppression/data ~/harmonic-noise-suppression/datasets'`
(DVC will repull on next `postdoc submit`).
