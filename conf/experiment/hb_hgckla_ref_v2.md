---
experiment: hb_hgckla_ref_v2
training_config: conf/experiment/hb_hgckla_ref_v2.yaml
batch: docs/experiments/rps-trajectory-refinement.md
---

# `hb_hgckla_ref_v2`

## Motivation

Probe P4 measured the trained `hb_hgckla_ref` refiner on the frozen real
split (37 clips x 8 mics, `experiments.rps_bench.part("real")`). It is safe
and it helps a little — the best regressor's cruise PIT MAE goes from 2.278
to 2.088 rev/s in one pass and no frame gets worse — but three measurements
say that it is not a precision stage:

- **Its fixed point is not the truth.** Given the true labels as
  conditioning, it moves 0.41 rev/s away at cruise, on 96.6 % of frames
  (median 0.32, p90 0.84, signed mean +0.02). That is the size of the error a
  precision stage exists to remove, and about 15x the 0.028 rev/s that
  GT-initialized classical VK refinement reaches on the same audio.
- **Iteration walks out.** Three passes on the best regressor give cruise
  2.278 -> 2.088 -> 2.124 -> 2.169 rev/s, with 71 % of cruise frames worse at
  pass 2 than at pass 1. A pure +2 rev/s offset parks at 1.0 rev/s, not at 0.
- **The pull is a fixed fraction, not a lock.** Inside +-2 rev/s the model
  removes ~40 % of any offset per call, the same fraction at 0.25 rev/s as at
  2 rev/s, and the fraction collapses outside that band (22 % at 4, 4 % at 8).

Each defect has a named cause in the architecture, and this arm turns on the
one flag that repairs it (`models/hg_ckla.py`, the "v2" paragraph):

1. `state_features` — the cell is memoryless. It never sees its own state
   difference, the correction the previous cells already applied, or how
   reliable the previous cell's reading was, so it cannot tell "already
   corrected" from "not yet corrected" and applies the same fraction every
   time. Four scalars per (rotor, frame) supply those differences (the
   KalmanNet F2/F4 inputs).
2. `kalman_gain` — the gain is a per-frame `sigmoid` with no memory of the
   evidence that came before it. It becomes a scalar random-walk Kalman
   filter whose measurement variance starts at the PHYSICAL scatter of the
   per-harmonic rate errors, with a zero-initialized learned correction on
   its log, so an untrained cell is the classical filter of design section 2
   step 3.
3. `smoother` — the scan is forward only, and a forward-only filter lags
   every turn. The backward Rauch-Tung-Striebel recursion on the same scalar
   state is the "smoother (RTS) over frames" the classical design asks for.
   It has no parameters.

## Setup

Model `hg_ckla_refiner_v2` (`conf/model/hg_ckla_refiner_v2.yaml`): the
`hg_ckla_refiner` stack with `state_features`, `kalman_gain` and `smoother`
all true, every other constructor argument unchanged (`k_caps` 10 / 25 / 40,
`d_model` 64, `n_state` 16, `max_delta` 5.0). The flags add 4 input features
per cell plus one zero-initialized variance head and two scalars, so the
parameter count moves by less than 0.5 %.

Data, loss, metrics and optimizer are `hb_hgckla_ref` verbatim: the
`ckla_refiner` corruption seam on the honest-base R2 policy
`conf/online_mix/hb_silence_dload.yaml`, plain non-PIT MSE
(`conf/loss/mse_cond.yaml`), batch 128 frames, spv 40000, grad clip 1.0,
AdamW 1e-3 / wd 1e-4. Only the schedule differs: `epochs: 40`,
`patience: 8`, because the v1 run was flat from epoch 1 to epoch 21.

Train: `python train.py experiment=hb_hgckla_ref_v2`.

Score with the same probe as v1, now a library:
`python scripts/refiner_bench.py --experiment hb_hgckla_ref_v2 --out results/refiner_bench/hb_hgckla_ref_v2`.

## Gate

Two numbers decide this arm, both from `scripts/refiner_bench.py` on the
frozen real split, and both against the v1 row measured in P4:

1. **A fixed point at the truth.** With the true labels as conditioning, the
   cruise PIT MAE must fall well below v1's 0.41 rev/s. This is the
   measurement that disqualified v1 as a precision stage.
2. **An iteration that does not walk out.** Over three passes, the cruise PIT
   MAE on the best regressor's conditioning must not grow after the pass that
   minimizes it, and the constant-offset case must approach the truth instead
   of parking about 1 rev/s away.

If both hold, C2 becomes the precision stage that a C1 seed feeds. If the
oracle drift stays near 0.41 rev/s, the defect is not in these three parts
and the phase measurement belongs inside the seed model, not after it.

## Conclusion

_Pending run._
