# G3 — GP-Generated Rotor Noise in the RPS Curriculum

**Status:** built, not yet run — **Date:** 2026-07-24

## Hypothesis

Campaign criterion 3.4: adding **GP-generated rotor noise** (the per-drone
egonoise GPs from the blind-eval campaign,
`src/experiments/gp_rotor_noise/train_egonoise_gp.py`) to the RPS-predictor
training curriculum improves real-data performance. The GP is a clean,
controllable comb with **exact synthetic RPS labels** and a *physics-anchored*
timbre — a third point on the synthetic-realism axis between the two families
that already failed or half-worked:

- **E7** (neural `PositionalHarmonicNoiseGen`, vicinal interp): no sim2real
  transfer — real val PIT MSE ~222, R² ≈ −10 across all archs; the predictor
  reverse-engineers the generator's RPS-correlated amplitude dynamics, a
  shortcut absent in real data.
- **E8** (analytic static comb, amplitudes RPS-independent): helped the
  **transformer only** (225 → 189), not the smaller heads.
- **E9** (50/50 neural + static comb, "hard" task): still far from the real
  baseline.

The GP shares the static comb's key property — coefficients are **frozen at the
chunk-mean rps** (no within-chunk amplitude→RPS shortcut; the only cue is the
comb's instantaneous frequency) — but its per-harmonic amplitudes and their
across-mic pattern come from a *fitted physical model* (Fourier coefficients
regressed over mic-xyz × rps on the CONA-auralized ego-noise sweep, rps 40–85),
not from a hand-designed profile family. Unlike E7/E8 (synthetic-**only**
probes), G3 tests the *augmentation* configuration directly on top of the best
real-data recipe (E12 full flight), mirroring how
`online_mix_generated_augment` added deep-gen noise in E4 (~1/3 share).

## Implementation

- `src/data_processing/gp_noise.py` — `GPRotorNoisePool` (`kind: gp`), a
  sibling of `StaticCombNoisePool`: pure-CPU synthesis in the DataLoader
  workers. The GP itself is evaluated **once at init**: the posterior mean is
  batch-queried on a dense rps grid (35–92, step 0.5) at the rig mic positions
  and kept as a `(G, M, 2H+1)` coefficient table (~1.6 MB, picklable); per
  chunk, coefficients are rps-interpolated, the comb is FM-synthesized at
  24 kHz (anti-aliased) from the `rps_synthesis` intermittent trajectory, the
  checkpoint's σ_b(rps) colored broadband floor is added, and the result is
  decimated to 16 kHz. ~200 ms per 1 s 8-mic chunk.
- Checkpoints: `r2://ml-data/artifacts/gp_egonoise/{dregon,matrice100}/best.pt`
  (uploaded from omnirun jobs `python-fbd20f` / `python-a87752`; sha256-verified
  round-trip via `training.artifacts.resolve_checkpoint_uri`).
- Policy `conf/online_mix/g3_gp_aug_dload.yaml`: E12 real full-flight policy
  with real shares unchanged + two GP sources (DREGON rig + Matrice-100 rig,
  weight 0.5 each vs 2.0 real ⇒ ~1/3 GP chunks).

### Design deviations (measured, not assumed)

1. **Mics projected onto the GP's training shell** (`mic_mode: shell`,
   default). The GP was trained on a mic shell *around* the airframe (DREGON
   |r| ∈ [0.45, 0.70] m; Matrice [0.60, 0.95] m) while the real rig arrays sit
   inside it (DREGON |r| = 0.085 m ≈ 3 xyz-lengthscales from the nearest
   training point). At the native positions the posterior mean-reverts and the
   across-mic coefficient spread collapses (0.121 → 0.049); radial projection
   (directions preserved) keeps the query in-support. Frame meta still carries
   the native rig geometry.
2. **Per-rotor 1/R decomposition** (`rotor_mode: per_rotor`, default). The
   sweep drives all 4 rotors at one shared rps, so the GP models the total
   field S(mic, rps). Rendering `Σ_r S(mic, rps_r)/R` keeps per-rotor labels
   exact and non-degenerate (exactly S when the rotors coincide, mild beating
   otherwise). `rotor_mode: mean` (four_way_lib's convention) is available but
   makes the 4 labels degenerate.
3. **Cruise-only RPS** (`synthetic_intermittent`). The GP has no amplitude
   model below rps 40, so `full_flight` excitation is rejected; the real E12
   sources still cover the whole envelope, so the arm remains a clean
   "E12 + cruise-timbre augmentation" comparison.

## Arm

| Experiment | Data | Model |
|---|---|---|
| `g3_gp_aug_transformer` | `g3_gp_aug` (E12 stream + 1/3 GP) | `simple_conv_v2_transformer` |

Transformer first — the arch E8 helped. Valid split identical to E12
(`dload:DREGON-LM-V4-michaels-valid-full`, FLY124 — no leakage; the GP was
fitted on CONA-simulated sweeps, never on held-out real recordings).

```bash
python train.py experiment=g3_gp_aug_transformer
```

## Success metric

Protocol eval `scripts/rps_predictor_vk_eval.py` against the E12 transformer
baseline on the same clips: **DREGON-cruise per-clip PIT MAE 3.186 (raw) /
2.62 (phase-A smoothed)**. G3 succeeds if the GP-augmented model beats those
numbers (and does not regress FLY124-cruise 1.766/1.55).

## Conclusion

_Pending run._

## Result (2026-07-24) — definite negative

Training 9wwaa7vb (38 epochs, best ep 18, val/mse 65.3). Protocol eval
`python-e1e448`: raw DREGON 3.960 / FLY124 1.747; best smoothed DREGON
3.132 (chmean/stitchmed) vs the E12 baseline's 3.186 raw / 2.62 smoothed
and FLY124 1.766/1.55. Verdict: adding GP-generated noise at the E4 1/3
share does NOT improve the RPS predictor on real data — DREGON degrades
(smoothed 2.62→3.13), FLY124 is a wash (1.75 vs 1.77 raw). Consistent
with the E7/E8 pattern: synthetic noise families whose fine structure
diverges from real (GP: even-harmonics-only, frozen-at-mean-rps
amplitudes) dilute rather than augment. Campaign criterion 3.4 is closed
with this negative.
