# DVC: push raw `data/` to R2 + commit DREGON-LM as a pipeline artifact

**Goal:** All raw `data/*` subdirs live in `s3://ml-data/datasets` via `dvc add`, and `datasets/DREGON-LM` is regenerable + tracked as a `dvc.yaml` pipeline output. End state: `git pull && dvc pull` on any machine reproduces everything.
**Status:** in-progress
**Last touched:** 2026-05-17
**Resume on:** any — but check `vast-server-2` first (push may have completed autonomously).

## Done

- All raw data transferred to `vast-server-2:/root/harmonic-noise-suppression/data/`:
  - `data/DREGON` (2.8 GB, zips removed) ✓
  - `data/librispeech` (6.3 GB, train-clean-100) ✓
  - `data/drone_audio` (768 MB) ✓
  - `data/new-drone-noises` (36 MB) ✓
  - `data/recording_with_motor_speed` (266 MB) ✓
  - `data/zenodo_drone_noises` (186 MB) ✓
- `dvc repro dregon-lm` completed on server → `datasets/DREGON-LM` (725 MB, 6000 train + 600 valid) ✓
- `dvc add` ran on server for all 6 raw dirs → `.dvc` files created ✓
- `dvc push -j 8` **IN PROGRESS on vast-server-2** (tmux session `downloads`, window `dvc-add`). ~86.8k cache entries, was at ~40k/87k when we left it. Upload ~27 Mbps. ETA ~50 min from last check.
- `dvc.yaml` written (cmd uses `.venv/bin/python`), `dvc.lock` generated ✓
- `.gitignore` patched locally (exposes `data/*.dvc` etc.) and committed as `268aaaf` (WIP: dvc init) — **not yet pushed to origin**
- New `.dvc` files + `dvc.lock` + `dvc.yaml` pulled to local machine and staged

## Pending

1. **Wait for `dvc push` to finish on `vast-server-2`.**
   - Check: `ssh vast-server-2 "tmux capture-pane -t downloads:dvc-add -p | tail -5"`
   - Done when: `ps aux | grep 'dvc push' | grep -v grep` returns nothing AND tmux shows "PUSH DONE" or a shell prompt.

2. **Git commit + push** (run locally):
   ```bash
   cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
   git add data/DREGON.dvc data/librispeech.dvc data/drone_audio.dvc \
           data/new-drone-noises.dvc data/recording_with_motor_speed.dvc \
           data/zenodo_drone_noises.dvc \
           dvc.yaml dvc.lock
   # datasets/.gitignore may not exist yet — skip if absent
   git add datasets/.gitignore 2>/dev/null || true
   git commit --amend --no-edit   # fold into the existing WIP: dvc init commit
   git push
   ```

3. **Pull on server** so server HEAD matches origin:
   ```bash
   ssh vast-server-2 "cd /root/harmonic-noise-suppression && git pull"
   ```

4. **Delete this checkpoint.**

## State

- **vast-server-2**: `dvc push` running in tmux `downloads:dvc-add`. Server HEAD is `9826ca7` (hasn't been git-pulled yet).
- **Local**: HEAD is `268aaaf` (WIP: dvc init, not yet pushed). Working tree has updated `data/DREGON.dvc` (M) and 4 new untracked `.dvc` files + `dvc.yaml` + `dvc.lock`.
- **DREGON.dvc on server** — 355 files, 3.0 GB (no zips). **Local** `data/DREGON.dvc` was pulled from server and reflects this.
- **dvc.lock** — deps include `data/DREGON` (md5 `8180528f...`, 334 files, 3.05 GB) and `data/librispeech/LibriSpeech/train-clean-100` (md5 `9931d25a...`, 29124 files). Out is `datasets/DREGON-LM` (md5 `7c638a0d...`, 26401 files, 739 MB).
- `.dvc/config.local` on server has `endpointurl` for R2. This file is gitignored — must be present on any machine doing `dvc pull/push`.

## Decisions (do not relitigate)

- **R2 bucket = `ml-data`**, prefix `datasets/`. `AWS_DEFAULT_REGION=auto` mandatory in `.env`.
- **`.dvc/config.local`** holds `endpointurl = https://d064390efe0e59d764d4f701b59d7b71.r2.cloudflarestorage.com` (gitignored). Must be set up manually on any new machine.
- **`dvc.yaml` cmd** uses `.venv/bin/python` (not bare `python`) — the server's PATH doesn't activate the project venv automatically.
- **DREGON zips deleted** from server before final `dvc add` — the INRIA download script left zip artifacts; they were removed so the tracked directory is clean extracted data only.
- **`drone_audio` is tracked** even though it has no current pipeline dep — included for completeness; DN-LM pipeline can reference it later.

## Open questions

- None blocking. `drone_audio` question (DN-LM pipeline dep?) can be deferred.

## Resume

```bash
# 1. Check if dvc push finished on server
ssh vast-server-2 "ps aux | grep 'dvc push' | grep -v grep; \
  tmux capture-pane -t downloads:dvc-add -p | tail -5"

# 2. Once done — commit locally
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
git add data/DREGON.dvc data/librispeech.dvc data/drone_audio.dvc \
        data/new-drone-noises.dvc data/recording_with_motor_speed.dvc \
        data/zenodo_drone_noises.dvc dvc.yaml dvc.lock
git add datasets/.gitignore 2>/dev/null || true
git commit --amend --no-edit
git push

# 3. Sync server
ssh vast-server-2 "cd /root/harmonic-noise-suppression && git pull"

# 4. Delete checkpoint
rm .pi/checkpoints/dvc-push-raw-data-and-dregon-lm.md
```
