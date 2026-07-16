r"""Faithful replication of the JASA GP rotor-noise model on ``jasa-flyovers``.

Lee, Ko, Seshadri & Rauleder, "Bayesian machine learning framework for
time-domain prediction of multirotor vehicle noise", JASA 159(4):3418-3435
(2026), DOI 10.1121/10.0043469.

Unlike the sibling :mod:`gp_rotor_noise` module (which was wired to Michael's
*stationary* 8-mic ring), this module targets the paper's actual construct on
the CONA-generated ``jasa-flyovers`` dataset: a NASA 1-Pax quadrotor flying a
level edgewise flyover along ``+x`` over a 256-mic ground array, 44.1 kHz x 1 s
per case, speeds ``V in {1..10} m/s``.

Model construct (paper Sec. II-III), view we implement
------------------------------------------------------
The paper's GP predicts the pressure time series at ``z = (x, y, V, t)`` with a
kernel ``k = k_Matern52(x) . k_Matern52(y) . k_Fourier(t) . k_Matern52(V)``
(Eqs. 10-16).  Because ``k_Fourier`` is ``F^T D F`` with a *fixed* BPF-informed
design ``F`` and a diagonal prior ``D``, the tonal Fourier-coefficient vector
``w`` has a closed-form Gaussian posterior (Eq. 15).  We therefore implement the
mathematically equivalent, far more usable factorisation:

  1. **Pre-process** each ``(mic, V)`` case: de-Dopplerize (Eqs. 1-2), take the
     tonal component (paper: 4-level db4 DWT approximation; the dataset already
     ships ``tonal``/``broadband`` split, which is exactly that separation), and
     phase-align the first BPF to a global baseline (mic ``(-30, 0)``, ``V=1``).
  2. **Project** the tonal signal onto the BPF-informed Fourier design ``F``
     (harmonics ``k * BPF``, ``k = 1..H``) by least squares -> coefficient
     vector ``w in R^(2H+1)`` (Eq. 13/15 MAP).
  3. **Regress** ``w`` over the operating point ``(x, y, V)`` with a Matern-5/2
     GP (Eqs. 11-12, 16) -- one exact GP per coefficient, sharing the ARD
     construction; the per-coefficient signal variance is the diagonal ``D``.
  4. **Broadband**: per-mic ``sigma_b`` (std of the broadband component) is the
     Gaussian likelihood noise floor; ``eps ~ N(0, sigma_b^2)`` is sampled at
     synthesis (Eq. 3, ``Sigma_b ~= sigma_b^2 I``).

Synthesis: predict ``w(x, y, V)``, build ``F`` at the requested BPF/time grid
(the comb frequency may be time-varying -- driven by a supplied rotor-speed
signal), form ``p = F w`` and add the broadband sample.  Because absolute phase
is fixed by the alignment convention, no Doppler needs re-applying for a static
listener.

Train/test split (paper Sec. II C): train ``V in {6, 8, 10}``, validate
``V in {7, 9}``; extrapolation demos at ``V in {4, 5}``.  Spatial region: the
aft grid ``x in [-140, -30], y in [0, 70]`` (the down-selection region where the
de-Dopplerization factor ``alpha < 1``).
"""

# pyright: reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOptionalSubscript=false, reportOptionalCall=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportCallIssue=false
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# ── physical / dataset constants ────────────────────────────────────────────
FS: float = 44100.0
C0: float = 340.294  # ISA sea-level speed of sound [m/s] (dataset meta)
BPF_HZ: float = 33.55  # NASA 1-Pax hover blade-passing frequency (3 * 671/60)
N_BLADES: int = 3
SRC_POS: tuple[float, float, float] = (0.0, 0.0, 30.0)  # de-Doppler source anchor

TRAIN_SPEEDS = (6.0, 8.0, 10.0)
TEST_SPEEDS = (7.0, 9.0)
EXTRAP_SPEEDS = (4.0, 5.0)

AFT_X_RANGE = (-140.0, -30.0)
AFT_Y_RANGE = (0.0, 70.0)

BASELINE_MIC = (-30.0, 0.0)  # phase-alignment reference (paper baseline)


# ════════════════════════════════════════════════════════════════════════════
# Data loading (dload ``jasa-flyovers``)
# ════════════════════════════════════════════════════════════════════════════
def load_flyovers(dataset: str = "jasa-flyovers") -> dict[int, dict]:
    """Load every flyover from the dload dataset, keyed by ``round(speed)``.

    Each value holds ``tonal``/``broadband``/``audio`` ``(O, N)``, ``mics``
    ``(O, 3)`` and the scenario ``meta`` dict.
    """
    import dload

    from data_processing.streams import open_repository

    repo = open_repository()
    ds = repo.dataset(dataset)
    out: dict[int, dict] = {}
    for _key, fields in ds.samples():
        meta = dload.codecs.json_from(fields["meta"])
        arr = np.load(io.BytesIO(fields["arrays"]))
        out[round(float(meta["speed"]))] = {
            "meta": meta,
            "tonal": np.asarray(arr["tonal"], dtype=np.float64),
            "broadband": np.asarray(arr["broadband"], dtype=np.float64),
            "audio": np.asarray(arr["audio"], dtype=np.float64),
            "mics": np.asarray(arr["mics"], dtype=np.float64),
            "band_centers": np.asarray(arr["band_centers"], dtype=np.float64),
            "speed": float(meta["speed"]),
        }
    return out


def aft_mask(mics: np.ndarray, x_range=AFT_X_RANGE, y_range=AFT_Y_RANGE) -> np.ndarray:
    """Boolean mask selecting mics inside the aft training region."""
    return (
        (mics[:, 0] >= x_range[0])
        & (mics[:, 0] <= x_range[1])
        & (mics[:, 1] >= y_range[0])
        & (mics[:, 1] <= y_range[1])
    )


# ════════════════════════════════════════════════════════════════════════════
# De-Dopplerization (paper Eqs. 1-2)
# ════════════════════════════════════════════════════════════════════════════
def doppler_alpha(mic_xy: tuple[float, float], v_inf: float, src=SRC_POS, c0=C0) -> float:
    r"""Doppler time-scale factor ``alpha = c / (c + v_D)`` for one mic (Eq. 1-2).

    ``v_D = -v_inf . (r_mic - r_src) / |r_mic - r_src|`` with ``v_inf`` along
    ``+x``.  In the aft region (``x < 0``) the source recedes -> ``v_D < 0`` ->
    ``alpha < 1``.
    """
    r_src = np.asarray(src, dtype=np.float64)
    r_mic = np.array([mic_xy[0], mic_xy[1], 0.0], dtype=np.float64)
    los = r_mic - r_src
    los = los / (np.linalg.norm(los) + 1e-12)
    v_d = -float(np.array([v_inf, 0.0, 0.0]) @ los)
    return c0 / (c0 + v_d)


def dedoppler(
    sig: np.ndarray, mic_xy, v_inf: float, src=SRC_POS, c0=C0
) -> tuple[np.ndarray, float]:
    """Resample a 1-s signal from ``t in [0, 1]`` to the de-compressed ``[0, alpha]``."""
    a = doppler_alpha(mic_xy, v_inf, src, c0)
    n = sig.shape[-1]
    t = np.arange(n) / FS
    t_new = np.linspace(0.0, a * (n - 1) / FS, n)
    return np.interp(t_new, t, sig), a


# ════════════════════════════════════════════════════════════════════════════
# BPF-informed Fourier design (paper Eq. 13) + phase alignment (Sec. II C)
# ════════════════════════════════════════════════════════════════════════════
def fourier_design(t: np.ndarray, n_harm: int, bpf: float = BPF_HZ) -> np.ndarray:
    """Design matrix ``F`` with rows ``[1, sin(w_k t), cos(w_k t), ...]`` (2H+1, N).

    Column layout matches the coefficient vector ``w`` = (mean, then per harmonic
    ``k=1..H`` a ``sin`` then a ``cos`` term).
    """
    rows = [np.ones_like(t)]
    for k in range(1, n_harm + 1):
        w = 2 * np.pi * k * bpf
        rows.append(np.sin(w * t))
        rows.append(np.cos(w * t))
    return np.stack(rows, axis=0)  # (2H+1, N)


def first_bpf_phase(sig: np.ndarray, bpf: float = BPF_HZ) -> float:
    """Phase [rad] of the fundamental BPF component via a one-bin DFT."""
    n = sig.shape[-1]
    t = np.arange(n) / FS
    return float(np.angle(np.sum(sig * np.exp(-2j * np.pi * bpf * t))))


def phase_align(sig: np.ndarray, target_phase: float, bpf: float = BPF_HZ) -> np.ndarray:
    """Circularly shift ``sig`` so its first-BPF phase equals ``target_phase``."""
    ph = first_bpf_phase(sig, bpf)
    dphi = (target_phase - ph + np.pi) % (2 * np.pi) - np.pi
    n_shift = int(round(dphi / (2 * np.pi * bpf) * FS))
    return np.roll(sig, n_shift)


def fit_coeffs(tonal: np.ndarray, n_harm: int, bpf: float = BPF_HZ) -> np.ndarray:
    """Least-squares Fourier coefficient vector ``w in R^(2H+1)`` for one signal."""
    n = tonal.shape[-1]
    t = np.arange(n) / FS
    A = fourier_design(t, n_harm, bpf).T  # (N, 2H+1)
    w, *_ = np.linalg.lstsq(A, tonal, rcond=None)
    return w


# ════════════════════════════════════════════════════════════════════════════
# GP over (x, y, V) predicting the Fourier-coefficient vector
# ════════════════════════════════════════════════════════════════════════════
def _make_gp(train_x: torch.Tensor, train_y: torch.Tensor):
    """Batched exact Matern-5/2 GP: one independent GP per coefficient output.

    ``train_x`` (N, 3) standardized ``(x, y, V)``; ``train_y`` (N, T) standardized
    coefficients.  Returns ``(model, likelihood)``.  Each of the ``T`` outputs
    gets its own ARD Matern-5/2 lengthscales (Eq. 11-12 spatial + Eq. 16
    velocity) and its own signal variance (the diagonal ``D``) and noise.
    """
    import gpytorch

    n_tasks = train_y.shape[-1]
    batch = torch.Size([n_tasks])

    class BatchGP(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=3, batch_shape=batch),
                batch_shape=batch,
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch)
    # tx shared across the task batch: (T, N, 3); ty: (T, N)
    tx = train_x.unsqueeze(0).expand(n_tasks, *train_x.shape).contiguous()
    ty = train_y.t().contiguous()
    model = BatchGP(tx, ty, likelihood)
    return model, likelihood, tx, ty


@dataclass
class JasaGPConfig:
    n_harm: int = 24
    bpf: float = BPF_HZ
    train_speeds: tuple[float, ...] = TRAIN_SPEEDS
    aft_x_range: tuple[float, float] = AFT_X_RANGE
    aft_y_range: tuple[float, float] = AFT_Y_RANGE
    iters: int = 300
    lr: float = 0.05
    verbose: bool = True
    broadband_bands: int = 24  # for the colored-broadband synth model


class JasaGPModel:
    """Faithful JASA GP: fit on flyovers, predict Fourier coeffs, synthesize audio."""

    def __init__(self, cfg: JasaGPConfig | None = None):
        self.cfg = cfg or JasaGPConfig()
        self.model = None
        self.likelihood = None
        # standardisation + state needed at inference
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.target_phase: float = 0.0
        # broadband: per-(mic,V) sigma_b table + averaged colored magnitude spectrum
        self.sigma_b_mean: float = 0.0
        self.bb_freqs: np.ndarray | None = None  # rfft freqs for colored synth
        self.bb_mag: np.ndarray | None = None  # averaged broadband |rfft| shape

    # ── training ────────────────────────────────────────────────────────────
    def _assemble(self, flyovers: dict[int, dict]):
        """Build (X, W, sigma_b) design points from the aft-region train speeds."""
        cfg = self.cfg
        # global baseline phase (mic (-30,0), lowest available train speed's signal)
        base_speed = min(cfg.train_speeds)
        base = flyovers[round(base_speed)]
        b_i = int(
            (
                (base["mics"][:, 0] - BASELINE_MIC[0]) ** 2
                + (base["mics"][:, 1] - BASELINE_MIC[1]) ** 2
            ).argmin()
        )
        b_dd, _ = dedoppler(base["tonal"][b_i], BASELINE_MIC, base_speed)
        self.target_phase = first_bpf_phase(b_dd, cfg.bpf)

        xs, ws, sbs = [], [], []
        bb_accum = None
        bb_count = 0
        for v in cfg.train_speeds:
            fl = flyovers[round(v)]
            mics = fl["mics"]
            mask = aft_mask(mics, cfg.aft_x_range, cfg.aft_y_range)
            idx = np.where(mask)[0]
            for i in idx:
                mic_xy = (float(mics[i, 0]), float(mics[i, 1]))
                dd, _ = dedoppler(fl["tonal"][i], mic_xy, v)
                dd = phase_align(dd, self.target_phase, cfg.bpf)
                w = fit_coeffs(dd, cfg.n_harm, cfg.bpf)
                xs.append([mic_xy[0], mic_xy[1], v])
                ws.append(w)
                sbs.append(float(np.std(fl["broadband"][i])))
                # accumulate colored broadband magnitude (de-Dopplered) for synth
                bb_dd, _ = dedoppler(fl["broadband"][i], mic_xy, v)
                mag = np.abs(np.fft.rfft(bb_dd))
                bb_accum = mag if bb_accum is None else bb_accum + mag
                bb_count += 1
        X = np.asarray(xs, dtype=np.float64)
        W = np.asarray(ws, dtype=np.float64)
        sb = np.asarray(sbs, dtype=np.float64)
        self.sigma_b_mean = float(sb.mean())
        n = flyovers[round(base_speed)]["tonal"].shape[-1]
        self.bb_freqs = np.fft.rfftfreq(n, 1.0 / FS)
        self.bb_mag = bb_accum / max(bb_count, 1)
        return X, W, sb

    def fit(self, flyovers: dict[int, dict]) -> None:
        import gpytorch

        cfg = self.cfg
        X, W, _sb = self._assemble(flyovers)
        if cfg.verbose:
            print(
                f"[jasa-gp] design points: X={X.shape} W={W.shape} sigma_b~{self.sigma_b_mean:.4f}"
            )

        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-9
        self.y_mean, self.y_std = W.mean(0), W.std(0) + 1e-9
        Xs = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        Ws = torch.tensor((W - self.y_mean) / self.y_std, dtype=torch.float32)

        model, likelihood, tx, ty = _make_gp(Xs, Ws)
        self.model, self.likelihood = model, likelihood
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for it in range(cfg.iters):
            opt.zero_grad()
            out = model(tx)
            loss = -mll(out, ty).sum()
            loss.backward()
            opt.step()
            if cfg.verbose and (it % 50 == 0 or it == cfg.iters - 1):
                print(f"[jasa-gp] it={it:4d} loss={loss.item():.4f}")
        model.eval()
        likelihood.eval()
        self._tx = tx  # cache training inputs for prediction batch shape
        self._ty = ty

    # ── prediction of coefficients ────────────────────────────────────────────
    def predict_coeffs(self, x: float, y: float, v: float, with_std=False):
        """Posterior mean (and optional std) Fourier coefficients at ``(x, y, V)``."""
        import gpytorch

        assert self.model is not None
        xq = np.array([[x, y, v]], dtype=np.float64)
        xqs = torch.tensor((xq - self.x_mean) / self.x_std, dtype=torch.float32)
        n_tasks = self.y_mean.shape[0]
        xqb = xqs.unsqueeze(0).expand(n_tasks, 1, 3).contiguous()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(xqb))
            mu = pred.mean.squeeze(-1).numpy()  # (T,)
            sd = pred.stddev.squeeze(-1).numpy()
        w = mu * self.y_std + self.y_mean
        if with_std:
            return w, sd * self.y_std
        return w

    # ── synthesis ─────────────────────────────────────────────────────────────
    def _fourier_from_rps(self, t: np.ndarray, rps: np.ndarray | None, bpf: float):
        """Design matrix at possibly time-varying comb frequency.

        ``rps`` (rev/s, same length as ``t``) drives an instantaneous
        ``bpf(t) = N_blades * rps(t)``; the harmonic phases integrate the
        instantaneous frequency (FM).  ``rps=None`` -> constant ``bpf``.
        """
        H = self.cfg.n_harm
        if rps is None:
            return fourier_design(t, H, bpf)
        inst_bpf = N_BLADES * np.asarray(rps, dtype=np.float64)
        phase = 2 * np.pi * np.cumsum(inst_bpf) / FS  # integral of freq
        rows = [np.ones_like(t)]
        for k in range(1, H + 1):
            rows.append(np.sin(k * phase))
            rows.append(np.cos(k * phase))
        return np.stack(rows, axis=0)

    def synthesize(
        self,
        x: float,
        y: float,
        v: float,
        duration: float = 1.0,
        rps: np.ndarray | None = None,
        broadband: str = "colored",
        seed: int = 0,
    ) -> np.ndarray:
        """Render a pressure time series at listener ``(x, y)`` for flight speed ``v``.

        ``rps`` optionally supplies a per-sample rotor speed [rev/s] (length
        ``round(duration*FS)``) to frequency-modulate the comb; otherwise the
        hover BPF is held.  ``broadband``: ``"none"``, ``"white"`` (paper
        ``N(0, sigma_b^2)``) or ``"colored"`` (shaped by the learned broadband
        magnitude spectrum).
        """
        n = int(round(duration * FS))
        t = np.arange(n) / FS
        w = self.predict_coeffs(x, y, v)
        F = self._fourier_from_rps(t, rps, self.cfg.bpf)  # (2H+1, n)
        tonal = F.T @ w  # (n,)
        out = tonal.copy()
        if broadband != "none":
            out = out + self._broadband(n, seed, kind=broadband)
        return out.astype(np.float32)

    def _broadband(self, n: int, seed: int, kind: str) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if kind == "white":
            return rng.normal(0.0, self.sigma_b_mean, n)
        # colored: random-phase spectrum shaped by the averaged broadband magnitude
        assert self.bb_mag is not None
        base_n = 2 * (len(self.bb_mag) - 1)
        mag = self.bb_mag
        if n != base_n:
            f_new = np.fft.rfftfreq(n, 1.0 / FS)
            mag = np.interp(f_new, self.bb_freqs, self.bb_mag)
        phase = rng.uniform(0, 2 * np.pi, len(mag))
        spec = mag * np.exp(1j * phase)
        sig = np.fft.irfft(spec, n)
        # normalise to the learned per-mic-average broadband RMS (sigma_b)
        rms = np.sqrt(np.mean(sig**2)) + 1e-12
        return sig / rms * self.sigma_b_mean

    # ── (de)serialisation ─────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cfg": asdict(self.cfg),
            "model": self.model.state_dict() if self.model is not None else None,
            "likelihood": self.likelihood.state_dict() if self.likelihood is not None else None,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "target_phase": self.target_phase,
            "sigma_b_mean": self.sigma_b_mean,
            "bb_freqs": self.bb_freqs,
            "bb_mag": self.bb_mag,
            "train_x": self._tx.detach().cpu() if self.model is not None else None,
            "train_y": self._ty.detach().cpu() if self.model is not None else None,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> JasaGPModel:
        state = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(JasaGPConfig(**state["cfg"]))
        for k in (
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "target_phase",
            "sigma_b_mean",
            "bb_freqs",
            "bb_mag",
        ):
            setattr(m, k, state[k])
        tx = state["train_x"]
        ty = state["train_y"]
        model, likelihood, _tx, _ty = _make_gp(
            torch.zeros(tx.shape[1], tx.shape[2]), torch.zeros(tx.shape[1], ty.shape[0])
        )
        # rebuild with the real training data so prediction conditioning is exact
        model.set_train_data(inputs=tx, targets=ty, strict=False)
        model.load_state_dict(state["model"])
        likelihood.load_state_dict(state["likelihood"])
        model.eval()
        likelihood.eval()
        m.model, m.likelihood, m._tx = model, likelihood, tx
        return m


# ════════════════════════════════════════════════════════════════════════════
# CLI training entry
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/jasa_gp/best.pt")
    p.add_argument("--n_harm", type=int, default=24)
    p.add_argument("--iters", type=int, default=300)
    args = p.parse_args()

    flyovers = load_flyovers()
    cfg = JasaGPConfig(n_harm=args.n_harm, iters=args.iters)
    model = JasaGPModel(cfg)
    model.fit(flyovers)
    model.save(args.out)
    print(f"[jasa-gp] saved -> {args.out}")


if __name__ == "__main__":
    main()
