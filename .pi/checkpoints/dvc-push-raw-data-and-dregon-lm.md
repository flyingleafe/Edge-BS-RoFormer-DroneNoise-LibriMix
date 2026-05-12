# DVC: push raw `data/` to R2 + commit DREGON-LM as a pipeline artifact

**Goal:** All raw `data/*` subdirs live in `s3://ml-data/datasets` via `dvc add`, and `datasets/DREGON-LM` is regenerable + tracked as a `dvc.yaml` pipeline output. End state: `git pull && dvc pull` on any machine reproduces everything.
**Status:** in-progress
**Last touched:** 2026-05-11
**Resume on:** any machine with fat upload bandwidth — `vast-server-2` recommended (server→R2 is fast; laptop→R2 is hours).

## Done

- `.gitignore` patched to expose `data/*.dvc`, `datasets/*.dvc`, and the auto-generated `data/.gitignore` / `datasets/.gitignore` (replaces the blanket `data/` and `datasets/` ignores). **Uncommitted in tree.**
- `dvc add data/DREGON` → `data/DREGON.dvc` (3.1 GB hashed). Uncommitted.
- `dvc add data/drone_audio` → `data/drone_audio.dvc` (1.1 GB hashed). Uncommitted.
- `.dvc/cache/` ≈ 7.4 GB of content-addressed blobs locally. **Not pushed.**

## Pending

1. **Get raw data onto the chosen machine.** If resuming on `vast-server-2`:
   - LibriSpeech train-clean-100 (~13 GB, public): `cd /root/harmonic-noise-suppression/data && mkdir -p librispeech && wget -O - https://www.openslr.org/resources/12/train-clean-100.tar.gz | tar -xz -C librispeech/`
   - DREGON (3.1 GB): `python -c "from data_processing.dregon import load_dregon_dataset; load_dregon_dataset()"` — pulls from HuggingFace into `data/DREGON`.
   - Bespoke subdirs (~1.5 GB total) — `rsync -avh data/{new-drone-noises,recording_with_motor_speed,zenodo_drone_noises,drone_audio} vast-server-2:/root/harmonic-noise-suppression/data/` from the laptop.
2. **`dvc add` the remaining four subdirs**: `librispeech`, `new-drone-noises`, `recording_with_motor_speed`, `zenodo_drone_noises`. (`DREGON` and `drone_audio` already added on the laptop; if redone on the server the hashes will match because DVC is content-addressed.)
3. **Write `dvc.yaml`** with one stage `dregon-lm`:
   - `cmd`: `python create_dregon_librimix.py`
   - `deps`: `create_dregon_librimix.py`, `data_processing/dregon.py`, `data/DREGON`, `data/librispeech/LibriSpeech/train-clean-100`
   - `outs`: `datasets/DREGON-LM`
   - Defaults in the script already match these paths and use `seed=42` → deterministic.
4. **`dvc repro dregon-lm`** to materialize `datasets/DREGON-LM` (6000 train + 600 valid samples; ~10–30 min CPU).
5. **`dvc push -j 8`** — uploads everything in cache to `s3://ml-data/datasets`.
6. **Commit**: `git add data/*.dvc data/.gitignore datasets/*.dvc datasets/.gitignore dvc.yaml dvc.lock .gitignore && git commit -m "dvc: track raw data + DREGON-LM pipeline" && git push`.
7. **Delete this checkpoint**: `rm .pi/checkpoints/dvc-push-raw-data-and-dregon-lm.md`.

## State

- Working tree dirty. **In scope** for this job (touch / commit only these): `.gitignore`, `data/DREGON.dvc`, `data/drone_audio.dvc`. **Out of scope** (pre-existing user WIP, leave alone): `AGENTS.md`, `data_processing/AGENTS.md`, `models/AGENTS.md`, and the untracked files `configs/noise_gen.yaml`, `data_processing/michaels.py`, `data_processing/noise_rps_dataset.py`, `models/generative/`, `train_noise_gen.py`.
- HEAD on `origin/main`: `9826ca7` — postdoc migration to vast-server-2 + DVC remote switch to `ml-data/datasets`.
- `.dvc/cache/` (7.4 GB) is laptop-local and not pushed. If resuming on the server, it's faster to re-`dvc add` from scratch than to rsync this cache.
- **DO NOT commit `data/DREGON.dvc` / `data/drone_audio.dvc` until their blobs are pushed to R2.** Otherwise `dvc pull` will fail for anyone else cloning the repo.

## Decisions (do not relitigate)

- **R2 bucket = `ml-data`**, prefix `datasets/`. Old `s3://hns-research` never existed.
- **`AWS_DEFAULT_REGION=auto`** is mandatory in `.env` — R2 quirk; `region` in `.dvc/config` is silently ignored by dvc-s3. Already in `.env` and `.env.example`.
- **Per-subdir granularity** for raw data (one `.dvc` per `data/<subdir>`) — chosen so the DREGON-LM stage's `deps:` can name just `data/DREGON` + LibriSpeech, not pull all 18 GB.
- **DREGON-LM is a DVC pipeline output** (`dvc.yaml outs:`), not `dvc add`-ed. Re-running the create script reruns the stage; the artifact stays content-addressed.
- **LibriSpeech goes into R2** (13 GB). Storage cost ≈ $0.20/month; ingress free; gives offline reproducibility.

## Open questions

- Is `data/drone_audio` (DroneAudioDataset, used by DN-LM / Paper 1) actually a dep of any current pipeline, or legacy? If only relevant to the DN-LM dataset (which has no pipeline file yet), confirm whether to keep tracking it here or defer until a DN-LM stage is written.

## Resume

```bash
# On vast-server-2 (or whichever machine you chose). Adjust paths if not server.
ssh vast-server-2
cd /root/harmonic-noise-suppression
git pull
# .env should already have AWS_*, R2_ACCOUNT_ID, AWS_DEFAULT_REGION=auto, WANDB_API_KEY.

# 1. Bring in the raw data (see "Pending" step 1 — three sources).

# 2. dvc-add everything (idempotent for already-added subdirs).
.venv/bin/dvc add data/DREGON data/drone_audio data/librispeech \
    data/new-drone-noises data/recording_with_motor_speed data/zenodo_drone_noises

# 3. Write dvc.yaml (deps + outs as listed in "Pending" step 3).

# 4. Repro DREGON-LM.
.venv/bin/dvc repro dregon-lm

# 5. Push everything.
.venv/bin/dvc push -j 8

# 6. Commit + push, then delete this checkpoint.
git add data/*.dvc data/.gitignore datasets/*.dvc datasets/.gitignore \
        dvc.yaml dvc.lock .gitignore
git commit -m "dvc: track raw data + DREGON-LM pipeline"
git push
rm .pi/checkpoints/dvc-push-raw-data-and-dregon-lm.md
```
