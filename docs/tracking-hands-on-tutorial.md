# Hands-on tutorial: fitness, optimization, and tracking in a notebook

This tutorial shows how to do the F_VK campaign work by hand, cell by cell:

1. Measure the fitness of a given RPS trajectory against given audio.
2. Run the L-BFGS optimizer on an (audio, RPS) pair, with stop criteria.
3. Run a baseline method, our method, or a neural method, with or without an
   initial RPS estimate, and get RPS predictions.
4. Run any method stage by stage or iteration by iteration, and see how the
   RPS estimates change. This includes direct optimization (L-BFGS).
5. Reproduce the three whole-recording campaigns: telemetry refinement, blind
   annotation, and the VK decomposition.

Every code cell in sections 0-4 ran on the laptop against the frozen `beatvk`
windows (2026-08-11); the section-5 numbers are the campaigns' own outputs.
Related documents: `docs/trajectory-fitness-design.md` (why F_VK),
`src/tracking/AGENTS.md` (the stage vocabulary),
`docs/experiments/vk-decomposition.md` (what the decomposition measured),
`docs/notebook-primitives-tutorial.md` (plots).

## 0. Setup

Start the notebook in the project venv. All imports below resolve directly.

```python
import numpy as np
import matplotlib.pyplot as plt
import tracking as trk
from tracking.fitness_vk import FVKConfig, FVKStage, fvk_score, optimize_trajectory
from tracking.protocols import load_prep_window, pit_align
```

### Load data through dload (the common path)

Every recording lives in a pinned dload dataset — `plots.explore` is the
front door. `beatvk-valid-raw` holds the four protocol recordings (44.1 kHz
audio + `rps_raw` telemetry + the per-recording window manifest in meta), and
the rich frame datasets (`DREGON-frames`, `michaels-frames`, `AVQ`, …) hold
full recordings with all telemetry entries. `explore.datasets()` lists the
catalog.

```python
from plots import explore
from data_processing.frames import meta_dict, resample_audio_series
from tracking.protocols import WindowSpec, slice_window

rec = explore.pick("beatvk-valid-raw", "free-flight_speech-low")
meta = meta_dict(rec)
meta["windows"]                      # the protocol window manifest

audio16 = np.asarray(resample_audio_series(rec["audio"], 16000).data, np.float64)
rps = rec["rps_raw"]                 # (rotor, time) Series on telemetry stamps

w = meta["windows"][1]
spec = WindowSpec(protocol="beatvk", recording_id=meta["recording_id"],
                  index=1, start_s=w["start_s"], end_s=w["end_s"],
                  regime=w["regime"])
audio, ft, r_telem, edge = slice_window(
    audio16, 16000, spec,
    ts=np.asarray(rps.timestamps, np.float64), vals=np.asarray(rps.data),
)
sr = 16000
```

This reproduces the frozen prep window EXACTLY (audio and telemetry, max
diff 0.0 against the cache — measured 2026-08-11). For an arbitrary
(non-protocol) clip from any frames dataset, slice the frame in time and go
through the same resample:

```python
rec = explore.pick("DREGON-frames", "free-flight_speech-low",
                   rps="motors_measured")          # audio + rps, coerced
t_rec = float(rec.t_start)            # DREGON stamps are epoch-absolute
clip = rec.time[t_rec + 10.0 : t_rec + 26.0]      # tick-exact time slice
audio = np.asarray(resample_audio_series(clip["audio"], 16000).data, np.float64)
r_ts = np.asarray(clip["rps"].timestamps, np.float64)
r_vals = np.asarray(clip["rps"].data, np.float64)
t0 = float(clip.t_start)
ft = np.arange(0.0, clip.duration, 0.032)
r_telem = np.stack([np.interp(ft + t0, r_ts, row) for row in r_vals])
```

### Shortcut: the frozen prep cache

`load_prep_window` reads the same window from the campaign's local `.npz`
cache with no network and no resample. Use it when the cache is present
(`TELEMETRY_PREP_DIR`, then the pulled cache, then the built cache). The 15
window keys: `FLY124__w00` … `FLY124__w05` and
`free-flight_{nosource,speech-low,whitenoise-low}_room1__w00` … `__w02`.

```python
w = load_prep_window("free-flight_speech-low_room1__w01")
sr = 16000
audio = w["audio"]      # (8, 256000) float64, 16 s at 16 kHz
ft = w["ft"]            # (500,) frame times, s, window-relative
r_telem = w["r"]        # (4, 500) raw telemetry, rev/s
w["regime"]             # 'cruise'
```

Note on the purity rule: `load_prep_window` does not touch
`data_processing` — it reads pre-built `.npz` files with numpy (the builder
that WROTE them, `scripts/beatvk_vk_arms.py --build-preps`, lives outside
the package). The dload cells above run in the notebook, on the
`data_processing` side of the seam, so the import-linter contract holds.

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

## 5. Reproduce this yourself

Sections 1-4 work on ONE window. The three campaign pipelines below run over
WHOLE recordings: refine the labels, annotate blind, decompose. Each is a
`scripts/` driver on the `utils.gridrun` harness — one window is one unit, one
unit is one `<out>/raw/<uid>.json`, and a re-run skips the units that exist.
Each has a `--smoke` arm that fits on the laptop and a cluster arm that does
not.

### 5.1 Refine the telemetry (`scripts/refine_dregon_rps.py`)

L-BFGS on F_VK (section 2) from the telemetry init, window by window over the
generator's own DREGON recordings, then stitched into a committed sidecar.

```bash
# smoke: one 4 s cruise window, k_max 10, 1 channel — about a minute.
# Redirect BOTH outputs: --mode defaults to "all", so a bare --smoke would
# stitch its single window over the committed sidecars.
PYTHONPATH=src python scripts/refine_dregon_rps.py --smoke \
  --out /tmp/refine-smoke --label-dir /tmp/refine-smoke/labels

# the real run: refine on a CPU node, stitch locally after the pull
omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 64 --time 24h \
  --name refine-dregon --outputs "results/refine_dregon_rps/**" \
  --env PYTHONPATH=src -- \
  python scripts/refine_dregon_rps.py --mode refine --jobs 6
omnirun pull refine-dregon
PYTHONPATH=src python scripts/refine_dregon_rps.py --mode stitch
```

Two knobs decide whether the run is honest, and both are already defaults:

- `smooth_lambda="auto"` (`trk.auto_smooth_lambda`). The refiner's fixed 1.0 is
  cruise-calibrated; a takeoff-ramp window reads a prior of 261 and cannot move
  at all under it. The unit reports `prior_init` and the weight it derived.
- Acceptance is **per rotor**. A window keeps a rotor's refinement only if the
  objective improved AND that rotor moved less than `MAX_MOVE_REV_S` = 3 rev/s.
  One alias-captured rotor must not discard the other three.

Read the sidecar and its report:

```python
import json, numpy as np
rid = "free-flight_nosource_room1"
rep = json.load(open(f"src/data_processing/refined_labels/{rid}.report.json"))
rep["cruise_scale_pct"]                 # -0.55356  <- THE headline
rep["cruise_scale_pct_raw_optimizer"]   # -1.04239  <- before the crossfade
rep["windows"][0]["used_per_rotor"]     # [True, False, True, True]
rep["windows"][0]["prior_init"]         # 260.94 -> smooth_lambda 0.00192

z = np.load(f"src/data_processing/refined_labels/{rid}.npz")
z["ft"], z["r_telemetry"], z["r_refined"]     # (1996,), (4, 1996), (4, 1996)
float(z["t0_offset_s"])                       # 5.480264
```

`ft` is seconds from the **published** recording's audio `t_start`, so the
labels apply to the untrimmed frame; the loader's telemetry-overlap trim
(`t0_offset_s`) is already added back. The stitched cruise shift is
**-0.554 %** (per rotor -0.25 / -0.61 / -0.70 / -0.68), inside the § 6d ridge
confidence interval [-0.877, -0.533] and next to the L-BFGS oracle's -0.596 —
which is how you know a re-run agrees. `cruise_scale_pct_raw_optimizer` is
larger because it averages the raw optimizer movement of accepted windows,
rejected rotors included; the headline reads the stitched labels. Full record:
`docs/experiments/generator-refined-labels.md`.

### 5.2 Annotate blind (`scripts/rps_eval.py`)

The blind ladder itself is section 3 (`trk.blind_fullrange()`). Over a whole
protocol it is the generic evaluator — prediction source x refinement x
protocol, one window per unit:

```bash
PYTHONPATH=src python scripts/rps_eval.py --protocol beatvk --pred telem
PYTHONPATH=src python scripts/rps_eval.py --protocol beatvk --pred telem --refine pi_kalman
PYTHONPATH=src python scripts/rps_eval.py --protocol beatvk --pred model:c11_scv2_best
```

`--pred` takes `telem`, `npz:<path-or-dir>` (a blind annotation you produced
yourself) or `model:<key>`; `--refine` is `none|pi_kalman|vk|warp`. Bars to
beat: blind DREGON 0.688 and FLY124 cruise 1.027 rev/s (`docs/experiments/
beat-vk.md`). Score a single window by hand with `trk.protocols.pit_align`
(section 3) — rotor identity is arbitrary and an unaligned error reading is
meaningless.

### 5.3 Decompose (`scripts/vk_decompose.py`)

One coupled-VK solve per window on the REFINED labels, stitched into one
envelope bank:

    x_mic(t) = sum_{rotor,k} Re[ env(t) e^{j k phi_rotor(t)} ] + residual_mic(t)

The sum is EXACT because the residual is DEFINED as the unexplained part; only
the split is estimated, and it is the MAP estimate under the VK bandwidth
prior. The array core is `tracking.decompose`; the script is the data, the
units and the file formats.

```bash
# smoke: one 4 s window, k_max 20, 2 mics, solve + stitch — about ten seconds
PYTHONPATH=src python scripts/vk_decompose.py --smoke --out /tmp/decomp-smoke

# the full recording (8 mics, k 80) on a CPU node
omnirun submit --backend uni-cpu --gpus 0 --cpus 4 --mem 32 --time 4h \
  --name vk-decompose --outputs "results/vk_decompose/**" \
  --env PYTHONPATH=src -- \
  python scripts/vk_decompose.py --mode solve --jobs 2 --bw-sweep
omnirun pull vk-decompose
PYTHONPATH=src python scripts/vk_decompose.py --mode stitch
```

WARNING: **size the job before you run it.** Coupling is transitive, so at
`k_hi` 62 the four rotors' 248 lines chain into ONE banded system and a 16 s
window needs 6.3 GB per worker. The law is

    bytes = 2 (2g + 1) g n_env 16    ~  1e-4 * k_hi^2 * window_s  GB

with `g ~ 4 k_hi` tracks per group and `n_env ~ 100 window_s`. Channels are
right-hand sides and cost nothing. `--mem-budget-gb` (8 GB) refuses a window
that does not fit, so the unit fails with the arithmetic instead of the pool
being killed — this is what OOM-killed a laptop once. Forecast it first:

```python
import numpy as np, tracking as trk
from tracking import decompose as D

z = np.load("src/data_processing/refined_labels/free-flight_nosource_room1.npz")
cfg = D.solve_config(80, sr=16000, mics=8)
k_hi = trk.k_cap(cfg, z["r_refined"])         # 62 — the cap of the WHOLE
                                              # recording, so every window
                                              # holds the same track set
r = D.to_audio_grid(z["r_refined"][:, :500], z["ft"][:500] - z["ft"][0],
                    16 * 16000, 16000)        # the first 16 s window
D.group_plan(r, k_hi, cfg)
# {'n_tracks': 248, 'n_groups': 1, 'max_group': 248, 'n_env': 1600,
#  'banded_gb': 6.311}
```

One group of 248: the whole comb. Feed it four rotors at exactly the same rate
and the lines coincide instead of chaining, which reads 62 tiny groups and
0.002 GB — so forecast on the REAL labels, never on a flat toy.

WARNING: omnirun output collection silently drops files above ~25 MB. The
stitched `envelopes.npz` is 103 MB for 64 s of 8-mic audio, so either stitch
locally after pulling the small per-window units (the recipe above) or upload
the big artifact to R2 from **inside** the job.

Load the results:

```python
import numpy as np
d = "results/vk_decompose/free-flight_nosource_room1"
z = np.load(f"{d}/envelopes.npz")
z["rotor"], z["k"]        # (248,) track table: which rotor, which harmonic
z["t_env"]                # (N,) s from the published recording's t_start
amp = z["amp"]            # (mic, track, N) = |env|      -> the amplitude
pherr = z["phase_err"]    # (mic, track, N) = arg(env)   -> the PHASE ERROR
z["bw_track"]             # (248,) the bandwidth each track really got
```

`phase_err` is the phase of harmonic `k` against the label-driven carrier, so
it is an error signal and not an absolute phase; the initial phase of a track
is `pherr[mic, m, 0]`. Both arrays are float32, and `phase_err` is stored
already unwrapped along time.

```python
r = np.load(f"{d}/residual.npz")
r["residual"]             # (mic, T) float32, the broadband remainder
r["freq_hz"], r["psd_residual"], r["psd_original"]     # the Welch pair
```

The exact-resynthesis check, which is the one property the decomposition
guarantees:

```python
rep = json.load(open(f"{d}/report.json"))
rep["resynthesis_max_abs"]        # 6e-8 — audio - (tracks + residual)
rep["energy"]["track_fraction"]   # 0.282
rep["energy"]["residual_fraction"]# 0.678   (cross term 4 %)
rep["energy"]["band_share_of_tracks"]
# {'k1-9': 0.975, 'k10-24': 0.016, 'k25-49': 0.007, 'k50-80': 0.003}
rep["phase_model"]["per_rotor"]["0"]["rank_one"]["lambda1_share"]   # 0.027
```

Two readings worth knowing before you use the envelopes as targets:

- **The rank-one (shaft-wander) phase model is refuted.** `lambda1_share` is
  0.027-0.028 against a chance level of 0.016 for 62 tracks, in all four
  rotors. Per-harmonic independent drift — the pi-kalman prior — is right.
- **Weak-band amplitudes absorb floor noise proportionally to the bandwidth,
  and the bandwidth you asked for is not the one you got.** A coupled group
  clamps every track to `max(bw_hz, 6 x line separation)`, and a dense comb
  floors that at 1 Hz, so the `--bw-grid` arms mostly measure the clamp;
  `--rho-grid` is the axis that survives it. Always compare an arm against its
  own `bw_track_hz_by_band` first. Record: `docs/experiments/vk-decomposition.md`.

To decompose one window in a notebook instead, the whole thing is a Stage —
it does not change the trajectory, so it composes after any method of
section 3:

```python
out = trk.decompose_stage(D.solve_config(40, sr=sr, mics=2))(frame)
out["meta"]["tracking"][-1]["track_fraction"]     # the ledger, as a log entry
env = out["meta"]["decompose"]["envelopes"]       # the seam: the bank itself
recon = out["meta"]["decompose"]["recon"]
residual = trk.get_audio(out)[0] - recon
```
