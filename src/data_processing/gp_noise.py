"""GP rotor-noise source pool (``kind: gp``) — the setup-matched egonoise GP
(inference core :mod:`data_processing.egonoise_gp`; trained by
:mod:`experiments.gp_rotor_noise.train_egonoise_gp`) as an online-mix noise
source.

Motivation (G3). Neither previous synthetic family transfers to real RPS
prediction on its own: the neural ``PositionalHarmonicNoiseGen`` fails outright
(E7: real val PIT MSE ~222, R^2 ~ -10 — the predictor reverse-engineers its
amplitude dynamics), and the analytic static comb helps only the transformer
(E8). The per-drone egonoise GP is a third point on the realism axis: a
*physics-anchored* comb (Fourier coefficients regressed over mic-xyz x rps on
the CONA-auralized ego-noise sweep) with none of the neural generator's
RPS-correlated amplitude dynamics — coefficients are frozen at the chunk's mean
rps, so within a chunk the only RPS cue is (as in E8) the comb's instantaneous
frequency, but the *timbre* is the fitted drone's, not an arbitrary synthetic
profile.

Design — mirrors :class:`data_processing.rotor_spectral_model.StaticCombNoisePool`
(cheap synthesis directly in the DataLoader workers; no GPU, no producer
process), with the GP itself evaluated only **once, at pool construction**:

* **Coefficient table, not per-chunk GP queries.** A warm GP posterior query is
  ~0.1-0.3 s — too slow per chunk. At init we batch-query the posterior mean on
  a dense rps grid at the (fixed) rig mic positions and keep only the resulting
  ``(G, M, 2H+1)`` table; per chunk the coefficients are a linear interpolation
  in rps. The gpytorch model is then dropped, so the pool pickles/forks as
  plain numpy.
* **In-support mic positions.** The GP was trained on a spherical *shell* of
  mics around the airframe (DREGON |r| in [0.45, 0.70] m, Matrice-100
  [0.60, 0.95] m), while the real rig arrays sit well inside it (DREGON
  |r| = 0.085 m ~ 3 xyz-lengthscales from the nearest training point, where the
  posterior mean-reverts and the across-mic coefficient spread collapses
  0.121 -> 0.049). Default ``mic_mode: shell`` therefore projects each rig mic
  radially onto the nearest training-shell radius (directions preserved);
  ``mic_mode: native`` keeps the raw rig positions. Frame metadata always
  carries the *native* rig geometry.
* **Per-rotor FM decomposition.** The sweep drives all 4 rotors at one shared
  rps, so the GP models the *total* field S(mic, rps). To keep per-rotor labels
  exact and non-degenerate we render ``sum_r S(mic, rps_r) / R`` — each rotor's
  synthetic trajectory gets its own comb at 1/R amplitude (exactly S when all
  rotors coincide, mild beating otherwise). ``rotor_mode: mean`` instead
  renders one comb at the rotor-mean trajectory and replicates that mean as the
  label of all rotors (audio/label still exact, but degenerate across rotors).
* **Cruise-only RPS.** The GP's operating-point support is rps 40..85, so only
  ``rps.kind: synthetic_intermittent`` (cruise trajectories, DREGON-calibrated)
  is allowed — a ``full_flight`` excitation would drive the GP far outside its
  domain (no amplitude model at low/zero rps).
* **Anti-alias rendering.** Synthesis runs at ``render_fs`` (24 kHz): harmonic
  frequencies reach ``H * n_blades * rps`` ~ 11 kHz > the 16 kHz-Nyquist, and
  only components above ``2*render_nyq - out_nyq`` = 16 kHz (rps > 133 —
  unreachable) could alias back into the output band before the polyphase
  decimation to ``sample_rate`` filters everything above 8 kHz.

Broadband uses the checkpoint's sigma_b(rps) table + global colored magnitude
shape (one incoherent draw per chunk at the chunk-mean rps). A per-rotor random
initial phase decorrelates chunks at identical RPS (phase offsets on a comb are
time shifts — the E4-style ``random_phase`` variety knob, always on). The chunk
is finally normalized to ``target_rms`` **globally across mics** (inter-mic
level ratios preserved), matching the static-comb convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td
from scipy.signal import resample_poly

from data_processing import rps_synthesis
from data_processing.frames import make_recording_frame
from data_processing.generated_noise import _DRONE_PROFILE_BLEND, load_geometry


class GPRotorNoisePool:
    """Trained :class:`~data_processing.egonoise_gp.EgonoiseGPModel`
    exposed as a ``sample_timeframe`` noise source (``kind: gp``)."""

    def __init__(
        self,
        checkpoint: str | Path,
        drone: str,
        *,
        sample_rate: int = 16000,
        duration_s: float = 1.0,
        broadband: bool = True,
        rotor_mode: str = "per_rotor",
        mic_mode: str = "shell",
        aggressiveness: float = 1.0,
        drone_profile: float | None = None,
        render_fs: int = 24000,
        table_rps_min: float = 35.0,
        table_rps_max: float = 92.0,
        table_rps_step: float = 0.5,
        query_chunk: int = 256,
        target_rms: float = 0.1,
        dregon_dir: str | Path = "data/DREGON",
    ):
        if rotor_mode not in ("per_rotor", "mean"):
            raise ValueError(f"rotor_mode must be 'per_rotor' or 'mean', got {rotor_mode!r}")
        if mic_mode not in ("shell", "native"):
            raise ValueError(f"mic_mode must be 'shell' or 'native', got {mic_mode!r}")
        self.drone = str(drone)
        self.sample_rate = int(sample_rate)
        self.chunk_s = float(duration_s)
        self.broadband = bool(broadband)
        self.rotor_mode = str(rotor_mode)
        self.mic_mode = str(mic_mode)
        self.aggressiveness = float(aggressiveness)
        self.blend = (
            float(drone_profile)
            if drone_profile is not None
            else _DRONE_PROFILE_BLEND.get(self.drone, 0.0)
        )
        self.render_fs = int(render_fs)
        if self.render_fs <= self.sample_rate:
            # need headroom above the output Nyquist for the decimation filter
            raise ValueError("render_fs must exceed sample_rate")
        self.target_rms = float(target_rms)

        # Native rig geometry (carried in frame meta; synthesis may project).
        self.mic_pos, self.rotor_pos = load_geometry(self.drone, dregon_dir)
        self.n_mics = int(self.mic_pos.shape[0])
        self.n_rotors = int(self.rotor_pos.shape[0])

        self.rps_grid = np.arange(
            float(table_rps_min),
            float(table_rps_max) + float(table_rps_step) / 2.0,
            float(table_rps_step),
        )
        self._build_tables(checkpoint, int(query_chunk))

    # -- one-time GP evaluation --------------------------------------------------

    def _build_tables(self, checkpoint: str | Path, query_chunk: int) -> None:
        """Load the GP once, project mics if needed, batch-query the posterior
        mean over ``rps_grid`` x rig mics, stash numpy tables, drop the GP."""
        import gpytorch
        import torch

        from data_processing.egonoise_gp import FS as GP_FS
        from data_processing.egonoise_gp import EgonoiseGPModel
        from utils.checkpoints import resolve_checkpoint_uri

        gp = EgonoiseGPModel.load(resolve_checkpoint_uri(checkpoint))
        assert gp.model is not None and gp._tx is not None
        assert gp.x_mean is not None and gp.x_std is not None
        assert gp.y_mean is not None and gp.y_std is not None
        self.n_blades = int(gp.cfg.n_blades)
        self.n_harm = int(gp.cfg.n_harm)

        # Synthesis mic positions: radial projection onto the training shell.
        tx_phys = gp._tx[0].numpy() * gp.x_std + gp.x_mean  # (N, 4)
        r_shell = float(np.linalg.norm(tx_phys[:, :3], axis=1).min())
        rig_r = np.linalg.norm(self.mic_pos, axis=1, keepdims=True)
        if self.mic_mode == "shell":
            self.synth_mics = self.mic_pos / np.maximum(rig_r, 1e-9) * r_shell
        else:
            self.synth_mics = np.asarray(self.mic_pos, dtype=np.float64)

        # Batched posterior-mean table over (rps_grid x mics).
        G, M = len(self.rps_grid), self.n_mics
        pts = np.concatenate(
            [
                np.repeat(self.synth_mics[None, :, :], G, axis=0).reshape(G * M, 3),
                np.repeat(self.rps_grid, M)[:, None],
            ],
            axis=1,
        )  # (G*M, 4)
        n_tasks = int(np.asarray(gp.y_mean).shape[0])
        mu_rows = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for lo in range(0, len(pts), query_chunk):
                chunk = pts[lo : lo + query_chunk]
                xqs = torch.tensor((chunk - gp.x_mean) / gp.x_std, dtype=torch.float32)
                xqb = xqs.unsqueeze(0).expand(n_tasks, len(chunk), 4).contiguous()
                post = gp.model(xqb)  # MultivariateNormal (batched over tasks)
                mu_rows.append(post.mean.numpy().T)  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]  # (n, T)
        mu = np.concatenate(mu_rows, axis=0) * gp.y_std + gp.y_mean  # (G*M, 2H+1)
        self.coeff_table = mu.reshape(G, M, 2 * self.n_harm + 1).astype(np.float64)

        # Broadband tables (checkpoint units; relative comb/broadband balance
        # is what matters — the chunk is RMS-normalized at the end).
        assert gp.bb_rps_grid is not None and gp.bb_sigma is not None
        assert gp.bb_freqs is not None and gp.bb_mag is not None
        self.bb_rps_grid = np.asarray(gp.bb_rps_grid, dtype=np.float64)
        self.bb_sigma = np.asarray(gp.bb_sigma, dtype=np.float64)
        self.bb_freqs = np.asarray(gp.bb_freqs, dtype=np.float64)
        self.bb_mag = np.asarray(gp.bb_mag, dtype=np.float64)
        self.gp_fs = float(GP_FS)
        del gp  # nothing torch/gpytorch survives past init

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> GPRotorNoisePool:
        def g(key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        rps = g("rps", {}) or {}
        rps_kind = rps.get("kind", "synthetic_intermittent")
        if rps_kind != "synthetic_intermittent":
            raise ValueError(
                "gp noise supports only rps.kind 'synthetic_intermittent' (the GP's "
                f"operating-point support is ~40..85 rev/s), got {rps_kind!r}"
            )
        checkpoint = g("checkpoint")
        if not checkpoint:
            raise ValueError("gp noise source requires 'checkpoint'")
        table = g("table", {}) or {}
        return cls(
            checkpoint=checkpoint,
            drone=g("drone", "dregon"),
            sample_rate=sample_rate,
            duration_s=duration_s,
            broadband=bool(g("broadband", True)),
            rotor_mode=str(g("rotor_mode", "per_rotor")),
            mic_mode=str(g("mic_mode", "shell")),
            aggressiveness=float(rps.get("aggressiveness", 1.0)),
            drone_profile=g("drone_profile"),
            render_fs=int(g("render_fs", 24000)),
            table_rps_min=float(table.get("rps_min", 35.0)),
            table_rps_max=float(table.get("rps_max", 92.0)),
            table_rps_step=float(table.get("rps_step", 0.5)),
            target_rms=float(g("target_rms", 0.1)),
            dregon_dir=str(g("dregon_dir", "data/DREGON")),
        )

    def close(self) -> None:  # interface parity with the other pools
        return None

    # -- synthesis ---------------------------------------------------------------

    def _coeffs_at(self, rps_mean: float) -> np.ndarray:
        """Linear rps-interpolation of the coefficient table -> ``(M, 2H+1)``.

        Clamped at the grid edges (beyond them the GP posterior mean-reverts
        anyway — its rps lengthscale is only ~1-3 rev/s)."""
        grid = self.rps_grid
        x = float(np.clip(rps_mean, grid[0], grid[-1]))
        i = int(np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2))
        a = (x - grid[i]) / (grid[i + 1] - grid[i])
        return (1.0 - a) * self.coeff_table[i] + a * self.coeff_table[i + 1]

    def _tonal(self, w: np.ndarray, traj_rf: np.ndarray, phi0: float) -> np.ndarray:
        """FM comb from Fourier coefficients: ``(M, n)`` at ``render_fs``.

        Same construction as ``EgonoiseGPModel.synthesize``'s trajectory branch
        (phase = 2*pi*cumsum(n_blades*rps)/fs; rows [1, sin k, cos k]), with an
        added initial fundamental phase and sin/cos recurrences instead of an
        explicit ``(2H+1, n)`` design matrix.
        """
        phase = phi0 + 2.0 * np.pi * self.n_blades * np.cumsum(traj_rf) / self.render_fs
        s1, c1 = np.sin(phase), np.cos(phase)
        out = np.repeat(w[:, 0:1], phase.shape[0], axis=1)  # DC
        sk, ck = s1, c1
        for k in range(1, self.n_harm + 1):
            out += np.outer(w[:, 2 * k - 1], sk) + np.outer(w[:, 2 * k], ck)
            if k < self.n_harm:
                sk, ck = sk * c1 + ck * s1, ck * c1 - sk * s1
        return out

    def _broadband(self, n: int, rps_mean: float, rng: np.random.Generator) -> np.ndarray:
        """Colored broadband at ``render_fs``: the checkpoint's global magnitude
        shape with random phases, RMS set by the sigma_b(rps) table -> (M, n)."""
        sigma = float(np.interp(rps_mean, self.bb_rps_grid, self.bb_sigma))
        f = np.fft.rfftfreq(n, 1.0 / self.render_fs)
        mag = np.interp(f, self.bb_freqs, self.bb_mag)
        out = np.empty((self.n_mics, n))
        for m in range(self.n_mics):
            spec = mag * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, len(mag)))
            sig = np.fft.irfft(spec, n)
            out[m] = sig / (np.sqrt(np.mean(sig**2)) + 1e-12) * sigma
        return out

    def render(self, rng: np.random.Generator, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Render ``(audio (M, T), rps (R, T))`` at ``sample_rate``."""
        T = int(round(duration_s * self.sample_rate))
        rps = rps_synthesis.generate_intermittent_batch(
            1,
            duration_s,
            self.sample_rate,
            drone_profile=self.blend,
            aggressiveness=self.aggressiveness,
            rng=rng,
        )[0]  # (R, T)
        if self.rotor_mode == "mean":
            mean_traj = rps.mean(axis=0)
            label = np.repeat(mean_traj[None, :], rps.shape[0], axis=0)
            trajs, scale = [mean_traj], 1.0
        else:
            label = rps
            trajs, scale = list(rps), 1.0 / rps.shape[0]

        n_rf = int(round(duration_s * self.render_fs))
        t16 = np.arange(T) / self.sample_rate
        t_rf = np.arange(n_rf) / self.render_fs
        audio_rf = np.zeros((self.n_mics, n_rf))
        for traj16 in trajs:
            traj_rf = np.interp(t_rf, t16, traj16)
            w = self._coeffs_at(float(traj16.mean()))
            audio_rf += scale * self._tonal(w, traj_rf, float(rng.uniform(0.0, 2.0 * np.pi)))
        if self.broadband:
            audio_rf += self._broadband(n_rf, float(rps.mean()), rng)

        from math import gcd

        g = gcd(self.sample_rate, self.render_fs)
        audio = resample_poly(audio_rf, self.sample_rate // g, self.render_fs // g, axis=-1)
        audio = np.asarray(audio, dtype=np.float32)[:, :T]
        if audio.shape[-1] < T:
            audio = np.pad(audio, ((0, 0), (0, T - audio.shape[-1])))
        rms = float(np.sqrt(np.mean(audio**2))) or 1.0
        audio = (audio / rms * self.target_rms).astype(np.float32)
        return audio, label.astype(np.float32)

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        audio, rps = self.render(rng, duration_s)
        audio_us = td.uniform(
            np.ascontiguousarray(audio), self.sample_rate, dims=("mic", "time"), t_start=0.0
        )
        t = np.arange(audio.shape[-1], dtype=np.float64) / self.sample_rate
        rps_es = td.events(t, np.ascontiguousarray(rps), dims=("rotor", "time"), t_start=0.0)
        return make_recording_frame(
            {"audio": audio_us, "rps": rps_es},
            meta={"recording_id": f"gp_{self.drone}"},
            mic_pos=self.mic_pos,
            rotor_pos=self.rotor_pos,
        )
