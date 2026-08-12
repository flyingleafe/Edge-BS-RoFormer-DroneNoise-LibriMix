# VK maximum-likelihood decomposition of real drone audio

**Date**: 2026-08-12 · **Branch**: `tracking-opt` · **Status**: instrument
built + full-recording decomposition DONE; amplitude-target training is the
follow-up. Motivated by the MSSTFT erosion verdict
(`generator-perrotor-dynamics.md`): the audio-domain magnitude loss cannot
defend mid/high-k lines because real lines decohere within the loss's STFT
windows — so decompose the audio ONCE, correctly, and train on the
decomposition instead.

## The decomposition

`scripts/vk_decompose.py`: windowed coupled Vold-Kalman solve
(`tracking.fitness_vk.solve_envelopes`) on REFINED labels →

    x_mic(t) = Σ_{rotor,k} Re[ env_{rotor,k,mic}(t) · e^{j·k·φ_rotor(t)} ] + residual_mic(t)

per-track amplitude = |env|, per-timestamp PHASE ERROR = arg(env), initial
phase = arg(env(t0)); the sum is EXACT by construction (measured resynthesis
error 6e-8). The ML content is the VK solve: penalized LS = MAP under a
Gaussian residual + per-track bandwidth prior. Broadband floors are per MIC
(per-rotor attribution of the residual needs cross-mic structure — the
wind-channel lesson — and stays out of scope here).

Full run (`vk-decompose-6bdede` + remote stitch): free-flight_nosource_room1,
64 s, 8 mics, k_hi 62 (f_max 6 kHz), 248 tracks, fs_env 100 Hz. Artifacts on
R2 `artifacts/vk-decompose/free-flight_nosource_room1/` (envelopes.npz 103 MB,
residual.npz, report.json, bw_sweep.json).

## Findings

1. **Energy ledger**: tracks 28.2 %, residual 67.8 %, cross 4 %. Track energy
   is k1-9-dominated (97.5 %); k10-24 1.6 %, k25-49 0.7 %, k50-80 0.3 % — the
   quantified reason MSSTFT never defends high-k lines.
2. **Phase model — the pi-kalman independence assumption WINS.** Rank-one
   share of phase-error increments across k within a rotor: λ1/Σλ =
   0.027-0.028 (chance for 62 tracks ≈ 0.016), mean pairwise correlation ≈ 0,
   all four rotors. The shaft-wander (rank-1) model is refuted at track level;
   per-harmonic independent drift is the right prior. (Smoke at K=20 gave
   0.19 vs noise bar 0.075 — a weak common component exists but is tiny.)
3. **Drift statistics are prior-limited for weak tracks.** Measured drift std
   is flat with k (4.4-5.1 rad/s in every band) — but the ρ sweep shows drift
   scales directly with the achieved bandwidth (2.5 / 4.9 / 9.5 rad/s at
   0.5 / 1 / 2 Hz), and the per-frame unwrap step already touches π at
   fs_env=100 Hz. So the flat curve reflects the 1 Hz clamp, not physics; the
   dense comb floors the per-group bandwidth clamp at 1 Hz regardless of
   bw_rps (only rho_scale moves it).
4. **Amplitude estimates of weak bands absorb floor noise ∝ bandwidth**
   (k10-24 mean amp 3.2e-4 / 4.5e-4 / 6.2e-4 at 0.5 / 1 / 2 Hz). Targets
   derived from these envelopes must either fix one bandwidth everywhere
   (consistent bias) or debias via the noise-equivalent bandwidth
   (`Envelopes.bw_track` exists for exactly this).

## Consequence: the amplitude-target training path

The decomposition gives per-(rotor, k, mic, t) amplitude envelopes and
per-mic residual PSDs — direct supervision targets for the generator's
`harm_amps` and noise branch. A loss on these targets never synthesizes
audio, so phase decoherence exits the training problem, and high-k amplitude
information survives (VK demodulation is comb-coherent, unlike the fixed
STFT). Design cautions carried over: debias weak-band amps (finding 4);
supervise log-amplitudes; keep the OU jitter for rendering only.

Gotchas: omnirun output collection silently drops ~25 MB files — upload big
artifacts to R2 from inside the job; the coupled group chains ~all tracks
into one banded system (memory ≈ 1e-4·k²·window_s GB/worker — `group_plan`
forecasts it, `--mem-budget-gb` guards it; this is what OOM-killed a laptop).
