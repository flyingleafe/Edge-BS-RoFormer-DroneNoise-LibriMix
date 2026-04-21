# Vast-Server Training

Run training scripts on the vast-server GPU machine using tmux sessions.

## Prerequisites

- SSH access: `ssh vast-server` (configured in `~/.ssh/config`)
- Project synced at `~/harmonic-noise-suppression/` on vast-server
- Code must be pushed to git first (if local changes need to reach server)

## Steps

### 1. Sync local code to server (if needed)

```bash
git push origin <branch>
ssh vast-server "cd ~/harmonic-noise-suppression && git pull"
```

Note: if the server can't pull from github directly, the code must already be present on the server.

### 2. Start a training run

Use tmux to keep the job running after disconnecting.

```bash
ssh vast-server "cd ~/harmonic-noise-suppression && tmux new-session -d -s <SESSION_NAME> '<COMMAND> 2>&1 | tee <LOG_PATH>'"
```

- Pick a descriptive `<SESSION_NAME>` (e.g. `dcunet_rps`, `dccrn_rps`)
- Use `--device cuda:0` or `--device cuda:1` to select GPU
- Always `tee` into a log file under `results/` for later inspection

### 3. Check that training started correctly

```bash
ssh vast-server "tmux capture-pane -t <SESSION_NAME> -p | tail -20"
```

Or attach to watch live:

```bash
ssh -t vast-server "tmux attach -t <SESSION_NAME>"
```

Detach with `Ctrl+B, D`.

### 4. Monitor training progress

```bash
ssh vast-server "tail -5 ~/harmonic-noise-suppression/<LOG_PATH>"
```

### 5. List running tmux sessions

```bash
ssh vast-server "tmux list-sessions"
```

### 6. Kill a training run if needed

```bash
ssh vast-server "tmux kill-session -t <SESSION_NAME>"
```

## Common training commands

### train_rps_predictor.py

```bash
# DCUNet encoder RPS on GPU 0
tmux new-session -d -s dcunet_rps 'python train_rps_predictor.py --model dcunet_enc_rps --device cuda:0 2>&1 | tee results/rps_predictor_comparison/dcunet_enc_rps.log'

# DCCRN encoder RPS on GPU 1
tmux new-session -d -s dccrn_rps 'python train_rps_predictor.py --model dccrn_enc_rps --device cuda:1 2>&1 | tee results/rps_predictor_comparison/dccrn_enc_rps.log'

# SimpleConv baseline
tmux new-session -d -s simple_conv_rps 'python train_rps_predictor.py --model simple_conv --device cuda:0 2>&1 | tee results/rps_predictor_comparison/simple_conv.log'

# DCCRNLite
tmux new-session -d -s dccrn_lite_rps 'python train_rps_predictor.py --model dccrn_lite_rps --device cuda:1 2>&1 | tee results/rps_predictor_comparison/dccrn_lite_rps.log'

# Train all models sequentially
tmux new-session -d -s rps_all 'python train_rps_predictor.py --train_all --device cuda:0 2>&1 | tee results/rps_predictor_comparison/all.log'
```

### Training via postdoc

```bash
postdoc job submit experiments/<experiment>.yaml
postdoc job status <job_id>
postdoc job logs <job_id> --tail
```

## Syncing results back

```bash
./sync_results.sh
```