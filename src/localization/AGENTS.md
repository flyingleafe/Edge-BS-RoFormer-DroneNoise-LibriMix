# src/localization/ — Rotor position estimation from mic-array audio

## Purpose

Given an **N-channel recording** of drone noise and the **approximate microphone
array geometry**, estimate the **3-D position of each rotor** relative to the
array. DREGON is the instrumented test case (mic + rotor positions known), so we
can measure error directly.

## What's here

| File | Purpose |
|------|---------|
| `srp_phat.py` | Near-field SRP-PHAT engine: multichannel STFT → PHAT-whitened cross-spectrum `R[i,j,f]` → steered-response power over a 3-D **position** grid, with peak extraction (non-max suppression) and coarse-to-fine refinement. |
| `rotor_localization.py` | High-level `localize_rotors(audio, mic_positions, sr, rps=None, ...)`. Two modes (below) + `match_and_score` (Hungarian) for evaluation. |
| `eval_dregon.py` | CLI: load a DREGON recording, localize, report per-rotor error vs ground truth. `python -m src.localization.eval_dregon`. |

## Why near-field (not classical DOA)

Classical SRP-PHAT and `pyroomacoustics.doa.SRP` assume **plane waves** and
search over **direction** (azimuth/elevation). That model is invalid here: the
DREGON array aperture (~0.17 m) is the same order as the rotor range (~0.22–0.40
m), so wavefront curvature carries the *position*. We therefore search a 3-D grid
of candidate **positions** `p` and use the true per-mic propagation delay
`d_i(p) = ||p − m_i|| / c`. `pyroomacoustics` is not a dependency (and isn't
installed); the near-field steering is the only novel bit, so we own it.

## Two modes

- **Audio-only** (`rps=None`): one combined SRP map, take the `n_rotors`
  strongest spatially-separated peaks.
- **RPS-aided** (`rps=(n_rotors, M)` in rev/s): build a soft harmonic STFT mask
  `k·f0_r(t)` per rotor, apply to every channel, localize each isolated rotor
  with its own map. Uses the rotor-speed telemetry the rest of the project is
  built on. Only separates rotors when their speeds differ enough for the masks
  to be distinct.

## Results on DREGON (and the physical limit) — IMPORTANT

The engine is **correct**: a single **broadband** free-field source is recovered
to **0.3 cm** (synthetic check). But on real (and clean-simulated) drone audio,
per-rotor error is **~25–30 cm** (≈ the source range), with large angular error.

This is **physics, not a bug**:

1. **Low-frequency harmonic content.** Rotor noise is dominated by blade-pass
   harmonics at 150 Hz–a few kHz (wavelengths 0.34–2.3 m ≫ 0.17 m aperture). At
   these wavelengths a compact array has weak phase differences → poor angular
   resolution and almost no **range** observability. Broadband high-frequency
   energy is what makes near-field range estimable, and rotor noise lacks it.
2. **Four simultaneous coherent sources** ~30 cm away in a compact array
   interfere; SRP peaks don't cleanly map to individual rotors. A clean,
   no-reverb 4-source simulation still gives ~27–44 cm.
3. **Separation is the dominant lever, but it has a hard floor.** A controlled
   sweep (clean no-reverb 4-rotor sim, *perfect* RPS fed in) over harmonic-mask
   sharpness shows monotonic improvement as the mask is pushed to *high-harmonic-
   only* bands — where the inter-rotor spacing `k·ΔBPF` exceeds the mask linewidth:
   `(kmin=1, rel=0.02, minw=15 Hz)` → **55.8 cm**; `(kmin=8, rel=0.01, minw=8 Hz)`
   → **36.0 cm**; `(kmin=20, rel=0.006, minw=5 Hz)` → **27.9 cm**. So tightening +
   raising the harmonic floor genuinely isolates rotors, but even the most
   aggressive config floors at ~28 cm — two orders of magnitude above the 0.7 cm
   single-source accuracy. Masking-based isolation of coherent rotors has a
   ceiling no mask tuning crosses; the residual cross-rotor leakage is the wall.

So with the DREGON 8-mic array this is **near the resolution floor**. Take-away
for the project: rotor localization with this geometry is fundamentally limited;
meaningful improvement needs a larger aperture, higher-frequency content, or a
model that exploits the *known* rigid quad geometry as a strong prior rather than
free 3-D search. The module is the right substrate for any of those.

## Conventions

- Audio is **channel-first `(C, N)`**; `mic_positions` is `(C, 3)`, same frame as
  the returned rotor positions.
- Runs in torch on CPU or CUDA. Grid points are chunked; coarse-to-fine keeps a
  fine grid from ever covering the whole volume.
- DREGON audio is native 44.1 kHz, 8-channel — do **not** downsample for
  localization (higher SR = finer TDOA).
