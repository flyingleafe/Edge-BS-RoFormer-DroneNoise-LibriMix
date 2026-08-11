# Hands-on tutorial: fitness, optimization, and tracking in a notebook

This tutorial shows how to do the F_VK campaign work by hand, cell by cell:

1. Measure the fitness of a given RPS trajectory against given audio.
2. Run the L-BFGS optimizer on an (audio, RPS) pair, with stop criteria.
3. Run a baseline method, our method, or a neural method, with or without an
   initial RPS estimate, and get RPS predictions.
4. Run any method stage by stage or iteration by iteration, and see how the
   RPS estimates change. This includes direct optimization (L-BFGS).

Every code cell in this document ran on the laptop against the frozen
`beatvk` windows (2026-08-11). Related documents:
`docs/trajectory-fitness-design.md` (why F_VK), `src/tracking/AGENTS.md`
(the stage vocabulary), `docs/notebook-primitives-tutorial.md` (plots).

## 0. Setup

Start the notebook in the project venv. All imports below resolve directly.

```python
import numpy as np
import matplotlib.pyplot as plt
import tracking as trk
from tracking.fitness_vk import FVKConfig, FVKStage, fvk_score, optimize_trajectory
from tracking.protocols import load_prep_window, pit_align
```

### Load a frozen protocol window

`load_prep_window` reads one frozen `beatvk` window. The prep directory
resolves automatically (`TELEMETRY_PREP_DIR`, then the pulled cache, then the
built cache). The 15 window keys: `FLY124__w00` … `FLY124__w05` and
`free-flight_{nosource,speech-low,whitenoise-low}_room1__w00` … `__w02`.

```python
w = load_prep_window("free-flight_speech-low_room1__w01")
sr = 16000
audio = w["audio"]      # (8, 256000) float64, 16 s at 16 kHz
ft = w["ft"]            # (500,) frame times, s, window-relative
r_telem = w["r"]        # (4, 500) raw telemetry, rev/s
w["regime"]             # 'cruise'
```

For audio outside the protocol, load it with `soundfile` or
`plots.explore.pick`, and build the arrays in the same shapes.

### Compute cost, and how to control it

One F_VK evaluation is one full coupled-VK envelope solve. Anchor numbers on
this laptop, 16 s of 8-mic audio: approximately 15-20 s at `k_max` 40, and
some minutes for a full L-BFGS schedule. Three levers:

- **Slice.** A 4 s, 2-mic slice makes every cell interactive (seconds).
- **Threads.** The stack uses 1 CPU thread by default. Opt in with
  `with trk.thread_pool(4): ...` or `TRACKING_FFT_WORKERS=4`.
- **GPU.** Set `TRACKING_DEVICE=cuda` before import for the 18-32x demod/peel
  speedups.

WARNING: L-BFGS at `k_max` 40 holds an autograd graph of ~12.6 GB for a full
16 s window. Use a slice, or 2 mics, on the laptop. Submit full sweeps
through omnirun (`no-heavy-compute` rule).

The interactive slice this tutorial uses:

```python
n = 4 * sr
sel = ft < 4.0
audio_s, ft_s, r_s = audio[:2, :n], ft[sel], r_telem[:, sel]
```

## 1. Measure the fitness of a trajectory

`fvk_score` scores one candidate trajectory against one window's audio. It
returns a plain dict. Compare candidates by `objective` (scale-free, lower is
better).

```python
cfg = FVKConfig(k_max=40)          # k_max=80 for campaign-grade readings

s_tel = fvk_score(audio_s, sr, r_s, ft_s, cfg, reference=r_s)
s_off = fvk_score(audio_s, sr, r_s + 0.5, ft_s, cfg, reference=r_s)

s_tel["objective"], s_off["objective"]   # 0.599 < 0.720 at k_max=10
s_tel["r2"]                              # energy fraction the comb captures
s_tel["k_energy"]                        # per-harmonic captured energy
s_tel["n_cells"]                         # the fixed cell count
```

Rules that keep a comparison honest:

- **Pin `reference`.** The reference trajectory sets the harmonic cap and thus
  the cell set. Pass the SAME reference for every candidate of a window.
  Make sure `n_cells` agrees across candidates.
- **The measure is relative.** One `objective` value alone says little. The
  differences between candidates on the same window and config carry the
  information.
- **Alias degeneracy.** A sub-multiple comb (f0/2 with 2K harmonics) scores as
  well as truth. The order penalty is not built yet, so do not read F_VK as a
  global verdict across octave candidates. The ridge detector
  (`trk.fitness_stage`, `tracking/fitness.py`) is the calibrated lock/no-lock
  verdict instrument. Use F_VK for optimization and local comparison.

For gradient work by hand, `tracking.fitness_vk.fvk_loss` gives the same
objective as a differentiable torch scalar (envelope theorem, `a*` detached).

## 2. Run the L-BFGS optimizer

`optimize_trajectory` refines whole trajectories by L-BFGS on F_VK under a
`k_max` annealing schedule. The parameterization is a coarse cubic B-spline
(knots each `knot_s` seconds) plus a per-rotor constant offset.

```python
schedule = (
    FVKStage(k_max=5,  max_iter=20),
    FVKStage(k_max=10, max_iter=20),
    FVKStage(k_max=20, max_iter=20),
    FVKStage(k_max=40, max_iter=20),
)   # DEFAULT_SCHEDULE = the same ladder up to k_max=80

r_refined, diag = optimize_trajectory(
    audio_s, sr, r_s, ft_s, cfg,
    schedule=schedule,
    knot_s=0.25,
    smooth_lambda=1.0,      # cruise-calibrated; see the warning below
    reference=r_s,          # pins the cell set, as in section 1
)

diag["stages"][0]["loss_trace"]   # loss at every closure evaluation, rung 0
diag["stages"][0]["move_rms"]     # how far the argmin moved in that rung
diag["move_total_max"]            # total movement, rev/s
```

Stop criteria:

- **Per rung**: `FVKStage.max_iter` caps the L-BFGS iterations of the rung.
  Torch L-BFGS also stops on its own gradient/change tolerances.
- **Custom** (no improvement in the last N chunks, wall-clock cap, target
  loss): run the optimizer in small chunks and test between chunks — see
  section 4.

WARNING: `smooth_lambda=1.0` is calibrated for CRUISE windows. On a ramp or
warmup window the smoothness term swamps the data term (measured
`loss_start` 244 against 0.80). Decrease it to match the trajectory's dynamic
range before any non-cruise use.

Two campaign facts to keep in mind:

- From telemetry init, L-BFGS is the only arm that recovers a corrupted init
  (0.598 → 0.182 rev/s rms at 0 dB). From random init, use multi-start plus
  the blind seed — a single start 8 rev/s off converges to an alias.
- The oracle sanity result: DREGON moves by −0.596 % scale, FLY124 cruise by
  −0.0007 %. If your run disagrees strongly, look for a config difference.

## 3. Get RPS predictions from any method

Every tracking method is a `Stage: td.Frame -> td.Frame`. Build the frame
once, then apply any method.

```python
frame = trk.tracking_frame(
    audio, sr,
    rps=r_telem, frame_times=ft,   # the current candidate (optional for blind)
    rps_meas=r_telem,              # the untouched reference; never modified
    meta={"recording_id": "free-flight_speech-low_room1"},
)
```

### Blind (no initial estimate) — our method

```python
run = trk.blind_fullrange()        # seed -> coarse Viterbi -> calibrated ladder
out = run(frame)
r_pred, ft_pred = trk.get_rps(out)
[e["stage"] for e in out["meta"]["tracking"]]
```

`trk.vit2dsp()` is the cruise-only variant without the full-range coarse
pass. Both need 4 rotors and the full window (they are calibrated recipes).

### From an initial estimate — refinement arms

```python
out = trk.flagship(4)(frame)                  # ours: 4 x (peel -> pi_kalman)
out = trk.refit_stage()(frame)                # the telemetry refit recipe
out = trk.fvk_refine_stage(cfg)(frame)        # L-BFGS (section 2) as a stage
out = trk.pipeline(                            # IAVKF-style baseline: plain VK
    trk.vk_stage(trk.VKConfig()),
)(frame)
r_pred, ft_pred = trk.get_rps(out)
```

`refit_stage` and `fvk_refine_stage` read their init from `rps_meas` /
`rps` respectively — see their docstrings in `src/tracking/top.py`.

### Classical baselines

`tasks.classical_rps_predictors` holds the non-learned baselines
(`cepstral_tracker`, `hps_tracker`, `pyin_single_f0`, `nmf_tracker`,
`matched_filter_tracker`). They
are array functions: `cepstral_tracker(audio_mono, sr) -> (4, T_frames)`.

### Neural methods

The quick one-clip path goes through the zoo:

```python
import zoo, tdseries as td

zoo.checkpoints(task="rps_prediction")     # list what is on R2
fm = zoo.load("c11_dregon_fly125_retrain")   # -> FrameModel, td.Frame in/out

mono = trk.tracking_frame(audio.mean(axis=0), sr)
pred = fm(td.Frame({"mixture": mono["audio"]}))
rps_pred = pred["rps_pred"].data.numpy()     # (4, T) at the model frame rate
```

The PUBLISHED numbers come from the frozen stitched-chmean inference
(`scripts/rps_predictor_vk_eval.py`, sliding windows + per-mic permutation
alignment + stitch). For protocol-wide runs use the generic evaluator:

```bash
python scripts/rps_eval.py --protocol beatvk --pred model:c11_scv2_best
python scripts/rps_eval.py --protocol beatvk --pred telem --refine pi_kalman
```

### Score a prediction

Rotor identity is arbitrary — always PIT-align before an error reading:

```python
aligned, order = pit_align(r_pred, r_telem)       # Hungarian assignment
err = np.abs(aligned - r_telem).mean()
```

Or score by fitness, on the same fixed cells as any other candidate:

```python
score = fvk_score(audio, sr, aligned, ft, cfg, reference=r_telem)
```

## 4. Watch a method stage by stage

### Any composed method

A pipeline is only a `for` loop over stages. Keep every intermediate frame:

```python
stages = [
    trk.blind_seed_stage(4, arms=("K", "R")),
    trk.coarse_init_stage(),
    trk.vit2dsp_stage(),
]
frames = [frame]
for stg in stages:
    frames.append(stg(frames[-1]))

history = [trk.get_rps(f)[0] for f in frames[1:]]        # (4, N) per stage
labels = [f["meta"]["tracking"][-1]["stage"] for f in frames[1:]]
```

Note: `vit2dsp_stage` fuses the four calibrated ladder steps into one stage
on purpose (they are one calibrated unit). The seed, the coarse pass, and
the ladder are the natural observation points of the blind method.

### The flagship, iteration by iteration

`peel_alternation` returns every intermediate frame directly:

```python
frames = trk.peel_alternation(frame, n_apps=6, arm="peeled")   # [init, app1..app6]
history = [trk.get_rps(f)[0] for f in frames]
diag = [f["meta"]["tracking"][-1] for f in frames[1:]]         # step + wall per app
```

### Score every step with the F_VK judge

`trk.fvk_stage` appends a score and does NOT change the trajectory, so it
composes with anything:

```python
judge = trk.fvk_stage(cfg)
objectives = [
    judge(f)["meta"]["tracking"][-1]["objective"] for f in frames
]
```

### L-BFGS, chunk by chunk, with a custom stop criterion

Run the optimizer in small chunks and feed each result back as the next
init. Each chunk re-parameterizes around its init, which is exact — the
basis spans the same space. The L-BFGS curvature memory resets per chunk,
so use this loop to OBSERVE and to STOP, and the single-call form of
section 2 for the final number.

```python
from dataclasses import replace

r_cur = r_s + 0.3                      # a corrupted init to watch recover
history, losses = [r_cur], []
for st in schedule:                    # the section-2 schedule
    best, stall = np.inf, 0
    for chunk in range(6):             # up to 6 x 5 iterations per rung
        r_cur, d = optimize_trajectory(
            audio_s, sr, r_cur, ft_s, cfg,
            schedule=(replace(st, max_iter=5),),
            reference=r_s,
        )
        history.append(r_cur)
        loss = d["stages"][0]["loss_end"]
        losses.append((st.k_max, loss))
        # stop rule: no improvement > 1e-4 in the last 2 chunks
        stall = 0 if loss < best - 1e-4 else stall + 1
        best = min(best, loss)
        if stall >= 2:
            break
```

Inside one call, `diag["stages"][i]["loss_trace"]` already records the loss
at every closure evaluation — plot it when the per-chunk granularity is too
coarse.

### Plot the evolution

```python
fig, ax = plt.subplots(figsize=(9, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
for i, r_h in enumerate(history):
    ax.plot(ft_s, r_h[0], color=colors[i], lw=1,
            label=f"step {i}" if i in (0, len(history) - 1) else None)
ax.plot(ft_s, r_s[0], "k--", lw=1.5, label="reference")
ax.set(xlabel="time, s", ylabel="rotor 0 rev/s")
ax.legend()
```

For aligned side-by-side frame comparisons, `plots.dwym({label: frame})`
takes a dict of frames — see `docs/notebook-primitives-tutorial.md`.
