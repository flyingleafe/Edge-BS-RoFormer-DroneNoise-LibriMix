r"""Per-drone GP rotor-noise model on the CONA-auralized ``drone-egonoise`` sweep.

The "GP trained on setup-matched simulated data" required by the blind-eval
campaign: one GP per drone (``dregon`` / ``matrice100``), trained on the
onboard-array ego-noise RPS sweep (rps 40..85 step 5, 2 s @ 44.1 kHz, 64 mics
randomly re-sampled around the airframe per case, all 4 rotors at the same rps).

Model — the JASA construct (:mod:`.jasa_gp`) with **rps replacing flight speed**:

    k = k_Fourier(t; harmonics k * BPF(rps), k=1..H)  x  k_Matern52(rps)
        x  k_Matern52(mic x, y, z)

As in :mod:`.jasa_gp` we use the closed-form factorisation: per (case, mic) the
tonal signal is least-squares-projected onto the BPF-informed Fourier design
(``BPF = n_blades * rps``, comb spacing refined by :func:`jasa_gp.estimate_f0`),
the fundamental phase is zeroed (:func:`jasa_gp.align_coeffs`), and one batched
exact Matern-5/2 ARD GP (4 input dims: mic xyz in the body frame + rps) regresses
each of the ``2H+1`` coefficients over the operating point.  Broadband: per-rps
sigma_b table (interpolated over rps) + a global colored magnitude shape.

Mic positions are randomly re-sampled per case (seed 0 vs seed 1 differ), so the
seed-1 validation queries the spatial kernel at genuinely unseen positions.

Memory (documented for the kaggle submission, H=60 -> T=2H+1=121 outputs):
  - GP fit: N <= 10 rps x 64 mics = 640 design points; batched covar
    (T, N, N) fp32 = 121 * 640^2 * 4 B ~= 198 MB; with autograd graph and
    Cholesky work buffers the peak is ~1.5-2 GB.
  - Coefficient extraction (per case, streamed): design A (88200, 121) fp64
    = 85 MB + SVD workspace ~= 300-400 MB, released between cases.
  Total well under kaggle's ~13 GiB; H is capped by usefulness (tonal energy
  of the n_fft=2048 Griffin-Lim synthesis dies out well below Nyquist), not RAM.
  H=60 covers 4.8 kHz at rps=40, 7.2 kHz at rps=60, 10.2 kHz at rps=85.

Validation:
  - seed 1 of every training rps (train on seed 0 only),
  - rps=60 held out entirely (both seeds) -> interpolation check.
Outputs under ``--out`` (default ``results/gp_egonoise/<drone>/``):
``best.pt``, ``eval_metrics.json``, ``overlay_R*_s*.png`` spectral/time overlays.

Usage::

    python src/experiments/gp_rotor_noise/train_egonoise_gp.py --drone dregon
    python src/experiments/gp_rotor_noise/train_egonoise_gp.py --smoke  # local

Remote (kaggle)::

    omnirun submit --backend kaggle --gpus 1 --time 4h --yes -- \
        python src/experiments/gp_rotor_noise/train_egonoise_gp.py --drone dregon
"""

# pyright: reportOptionalMemberAccess=false, reportOptionalOperand=false, reportOptionalSubscript=false, reportOptionalCall=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportCallIssue=false
from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.gp_rotor_noise import jasa_gp as J

FS = 44100.0
DRONES = ("dregon", "matrice100")


# ════════════════════════════════════════════════════════════════════════════
# Data loading (dload ``drone-egonoise``) — streamed, one case in RAM at a time
# ════════════════════════════════════════════════════════════════════════════
def iter_cases(
    drone: str,
    dataset: str = "drone-egonoise",
    version: str | None = None,
    arrays: tuple[str, ...] = ("tonal",),
    max_mics: int | None = None,
):
    """Yield per-case dicts for one drone, streaming (never all cases in RAM).

    Each yielded dict: ``key, rps (float), seed (int), bpf_hz, n_blades,
    mics (M, 3) body-frame`` plus the requested ``arrays`` (fp32, (M, N)).
    ``max_mics`` truncates the mic set (smoke mode).
    """
    import dload

    from data_processing.streams import open_repository

    repo = open_repository()
    ds = repo.dataset(dataset, version)
    for key, fields in ds.samples():
        if not key.startswith(drone + "_"):
            continue
        meta = dload.codecs.json_from(fields["meta"])
        arr = np.load(io.BytesIO(fields["arrays"]))  # NpzFile: lazy per array
        m_sl = slice(None) if max_mics is None else slice(max_mics)
        rec = {
            "key": key,
            "rps": float(meta["rps"]),
            "seed": int(meta["seed"]),
            "bpf_hz": float(meta["bpf_hz"]),
            "n_blades": int(meta["n_blades"]),
            "mics": np.asarray(arr["mics_body"], dtype=np.float64)[m_sl],
        }
        for name in arrays:
            rec[name] = np.asarray(arr[name], dtype=np.float32)[m_sl]
        yield rec


def sweep_inventory(drone: str, dataset: str, version: str | None) -> dict[tuple[float, int], str]:
    """Map ``(rps, seed) -> key`` without decoding audio (meta field only)."""
    import dload

    from data_processing.streams import open_repository

    repo = open_repository()
    ds = repo.dataset(dataset, version)
    out: dict[tuple[float, int], str] = {}
    for key, fields in ds.samples():
        if not key.startswith(drone + "_"):
            continue
        meta = dload.codecs.json_from(fields["meta"])
        out[(float(meta["rps"]), int(meta["seed"]))] = key
    return out


# ════════════════════════════════════════════════════════════════════════════
# Per-case coefficient extraction (batched over mics)
# ════════════════════════════════════════════════════════════════════════════
def extract_coeffs(
    tonal: np.ndarray, bpf_hz: float, n_harm: int, f0_rel: float = 0.02
) -> tuple[np.ndarray, float]:
    """LSQ Fourier coefficients for all mics of one case -> ((M, 2H+1), f0).

    The comb spacing is refined around the prescribed BPF (Griffin-Lim
    re-synthesis can shift it slightly); one shared f0 per case (no Doppler on
    the onboard array).  Each mic's coefficient vector is fundamental-phase
    aligned (:func:`jasa_gp.align_coeffs`), the same convention the GP predicts.
    """
    f0 = J.estimate_f0(tonal[0].astype(np.float64), bpf_hz, rel=f0_rel)
    n = tonal.shape[-1]
    t = np.arange(n) / FS
    A = J.fourier_design(t, n_harm, f0).T  # (N, 2H+1)
    W, *_ = np.linalg.lstsq(A, tonal.T.astype(np.float64), rcond=None)  # (2H+1, M)
    W = W.T
    return np.stack([J.align_coeffs(w, n_harm) for w in W], axis=0), f0


def aligned_tonal(tonal: np.ndarray, f0: float) -> np.ndarray:
    """Per-mic fundamental-phase-aligned tonal (ground truth for overlays)."""
    return np.stack([J.align_fundamental_time(s.astype(np.float64), f0) for s in tonal], axis=0)


# ════════════════════════════════════════════════════════════════════════════
# Batched exact GP over (mic_x, mic_y, mic_z, rps)
# ════════════════════════════════════════════════════════════════════════════
def _make_gp(train_x: torch.Tensor, train_y: torch.Tensor):
    """One independent ARD Matern-5/2 GP per Fourier coefficient (batched).

    ``train_x`` (N, D) standardized; ``train_y`` (N, T).  Same construction as
    :func:`jasa_gp._make_gp` but with ``D`` input dims (here D=4: xyz + rps).
    """
    import gpytorch

    n_tasks = train_y.shape[-1]
    n_dims = train_x.shape[-1]
    batch = torch.Size([n_tasks])

    class BatchGP(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_dims, batch_shape=batch),
                batch_shape=batch,
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch)
    tx = train_x.unsqueeze(0).expand(n_tasks, *train_x.shape).contiguous()
    ty = train_y.t().contiguous()
    model = BatchGP(tx, ty, likelihood)
    return model, likelihood, tx, ty


@dataclass
class EgonoiseGPConfig:
    drone: str = "dregon"
    n_harm: int = 60
    iters: int = 400
    lr: float = 0.05
    train_seed: int = 0
    holdout_rps: tuple[float, ...] = (60.0,)
    n_blades: int = 2  # overwritten from meta at fit time
    verbose: bool = True
    extra: dict = field(default_factory=dict)


class EgonoiseGPModel:
    """Per-drone GP: fit on the rps sweep, predict coeffs at (mic xyz, rps)."""

    def __init__(self, cfg: EgonoiseGPConfig | None = None):
        self.cfg = cfg or EgonoiseGPConfig()
        self.model = None
        self.likelihood = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        # broadband model: sigma_b(rps) table + global colored magnitude shape
        self.bb_rps_grid: np.ndarray | None = None
        self.bb_sigma: np.ndarray | None = None
        self.bb_freqs: np.ndarray | None = None
        self.bb_mag: np.ndarray | None = None
        self._tx: torch.Tensor | None = None
        self._ty: torch.Tensor | None = None

    # ── training ────────────────────────────────────────────────────────────
    def fit(self, cases) -> dict:
        """Train on an iterable of case dicts (must include ``tonal`` and
        ``broadband`` arrays).  Returns a small summary dict."""
        import gpytorch

        cfg = self.cfg
        xs, ws = [], []
        bb_sigma: dict[float, list[float]] = {}
        bb_accum, bb_count = None, 0
        f0s = []
        n_cases = 0
        for c in cases:
            w, f0 = extract_coeffs(c["tonal"], c["bpf_hz"], cfg.n_harm)
            cfg.n_blades = c["n_blades"]
            xs.append(np.concatenate([c["mics"], np.full((len(w), 1), c["rps"])], axis=1))
            ws.append(w)
            f0s.append(f0)
            n_cases += 1
            sig = c["broadband"].astype(np.float64)
            bb_sigma.setdefault(c["rps"], []).extend(np.std(sig, axis=-1).tolist())
            mag = np.abs(np.fft.rfft(sig, axis=-1)).mean(0)
            bb_accum = mag if bb_accum is None else bb_accum + mag
            bb_count += 1
            if cfg.verbose:
                print(
                    f"[fit] {c['key']}: f0={f0:.3f} Hz (bpf {c['bpf_hz']:.1f}) "
                    f"|w|~{np.abs(w).mean():.4f}",
                    flush=True,
                )
        if n_cases == 0:
            raise RuntimeError(f"no training cases found for drone={cfg.drone}")
        X = np.concatenate(xs, axis=0)  # (N, 4)
        W = np.concatenate(ws, axis=0)  # (N, 2H+1)
        grid = sorted(bb_sigma)
        self.bb_rps_grid = np.asarray(grid, dtype=np.float64)
        self.bb_sigma = np.asarray([float(np.mean(bb_sigma[r])) for r in grid])
        n = 2 * (len(bb_accum) - 1)
        self.bb_freqs = np.fft.rfftfreq(n, 1.0 / FS)
        self.bb_mag = bb_accum / max(bb_count, 1)

        if cfg.verbose:
            print(f"[fit] design: X={X.shape} W={W.shape} ({n_cases} cases)", flush=True)

        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-9
        self.y_mean, self.y_std = W.mean(0), W.std(0) + 1e-9
        Xs = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        Ws = torch.tensor((W - self.y_mean) / self.y_std, dtype=torch.float32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, likelihood, tx, ty = _make_gp(Xs, Ws)
        model, likelihood = model.to(device), likelihood.to(device)
        tx, ty = tx.to(device), ty.to(device)
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        t0 = time.time()
        loss_val = float("nan")
        for it in range(cfg.iters):
            opt.zero_grad()
            out = model(tx)
            loss = -mll(out, ty).sum()
            loss.backward()
            opt.step()
            loss_val = loss.item()
            if cfg.verbose and (it % 25 == 0 or it == cfg.iters - 1):
                print(f"[fit] it={it:4d} loss={loss_val:.4f}", flush=True)
        model.eval()
        likelihood.eval()
        self.model, self.likelihood = model.cpu(), likelihood.cpu()
        self._tx, self._ty = tx.cpu(), ty.cpu()
        self.model.set_train_data(inputs=self._tx, targets=self._ty, strict=False)
        return {
            "n_cases": n_cases,
            "n_points": int(X.shape[0]),
            "n_tasks": int(W.shape[1]),
            "final_loss": loss_val,
            "fit_seconds": time.time() - t0,
            "f0_over_bpf": float(
                np.mean([f / (cfg.n_blades * r) for f, r in zip(f0s, [x[0, 3] for x in xs])])
            ),
        }

    # ── prediction / synthesis ──────────────────────────────────────────────
    def predict_coeffs(self, mics: np.ndarray, rps: float) -> np.ndarray:
        """Posterior-mean Fourier coefficients at each mic for one rps -> (M, 2H+1)."""
        import gpytorch

        assert self.model is not None
        M = mics.shape[0]
        xq = np.concatenate([np.asarray(mics, np.float64), np.full((M, 1), rps)], axis=1)
        xqs = torch.tensor((xq - self.x_mean) / self.x_std, dtype=torch.float32)
        n_tasks = self.y_mean.shape[0]
        xqb = xqs.unsqueeze(0).expand(n_tasks, M, xqs.shape[-1]).contiguous()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mu = self.model(xqb).mean.numpy()  # (T, M)
        return mu.T * self.y_std + self.y_mean

    def synthesize(
        self,
        mics: np.ndarray,
        rps,
        duration: float = 2.0,
        broadband: str = "none",
        seed: int = 0,
        f0: float | None = None,
    ) -> np.ndarray:
        """Render (M, T) pressure at the given mics.

        ``rps``: scalar (constant comb at ``f0`` or ``n_blades*rps``) or a
        per-sample trajectory (FM comb; coefficients frozen at the mean rps).
        """
        rps_arr = np.atleast_1d(np.asarray(rps, dtype=np.float64))
        n = int(round(duration * FS)) if rps_arr.size == 1 else rps_arr.size
        t = np.arange(n) / FS
        w = self.predict_coeffs(mics, float(rps_arr.mean()))  # (M, 2H+1)
        H = self.cfg.n_harm
        if rps_arr.size == 1:
            comb = f0 if f0 is not None else self.cfg.n_blades * float(rps_arr[0])
            F = J.fourier_design(t, H, comb)  # (2H+1, n)
        else:
            phase = 2 * np.pi * np.cumsum(self.cfg.n_blades * rps_arr) / FS
            rows = [np.ones(n)]
            for k in range(1, H + 1):
                rows.append(np.sin(k * phase))
                rows.append(np.cos(k * phase))
            F = np.stack(rows, axis=0)
        out = w @ F  # (M, n)
        if broadband != "none":
            out = out + self._broadband(mics.shape[0], n, float(rps_arr.mean()), seed)
        return out.astype(np.float32)

    def _broadband(self, n_mics: int, n: int, rps: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sigma = float(np.interp(rps, self.bb_rps_grid, self.bb_sigma))
        f_new = np.fft.rfftfreq(n, 1.0 / FS)
        mag = np.interp(f_new, self.bb_freqs, self.bb_mag)
        out = np.empty((n_mics, n))
        for m in range(n_mics):
            spec = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, len(mag)))
            sig = np.fft.irfft(spec, n)
            out[m] = sig / (np.sqrt(np.mean(sig**2)) + 1e-12) * sigma
        return out

    # ── (de)serialisation ───────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "model": self.model.state_dict(),
                "likelihood": self.likelihood.state_dict(),
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "bb_rps_grid": self.bb_rps_grid,
                "bb_sigma": self.bb_sigma,
                "bb_freqs": self.bb_freqs,
                "bb_mag": self.bb_mag,
                "train_x": self._tx,
                "train_y": self._ty,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> EgonoiseGPModel:
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["cfg"]
        cfg["holdout_rps"] = tuple(cfg.get("holdout_rps", ()))
        m = cls(EgonoiseGPConfig(**cfg))
        for k in (
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "bb_rps_grid",
            "bb_sigma",
            "bb_freqs",
            "bb_mag",
        ):
            setattr(m, k, state[k])
        tx, ty = state["train_x"], state["train_y"]
        model, likelihood, _, _ = _make_gp(
            torch.zeros(tx.shape[1], tx.shape[2]), torch.zeros(tx.shape[1], ty.shape[0])
        )
        model.set_train_data(inputs=tx, targets=ty, strict=False)
        model.load_state_dict(state["model"])
        likelihood.load_state_dict(state["likelihood"])
        model.eval()
        likelihood.eval()
        m.model, m.likelihood, m._tx, m._ty = model, likelihood, tx, ty
        return m


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════
def _loudness_db(x: np.ndarray) -> float:
    ac = x - np.mean(x)
    return float(20.0 * np.log10(np.sqrt(np.mean(ac**2)) / 20e-6 + 1e-12))


def evaluate_case(model: EgonoiseGPModel, case: dict) -> tuple[dict, dict]:
    """Metrics + overlay payload for one held-out case."""
    w_true, f0 = extract_coeffs(case["tonal"], case["bpf_hz"], model.cfg.n_harm)
    w_pred = model.predict_coeffs(case["mics"], case["rps"])
    coeff_rmse = float(np.sqrt(np.mean((w_pred - w_true) ** 2)))
    rel_resid = float(((w_pred - w_true) ** 2).sum() / ((w_true**2).sum() + 1e-12))
    # time/spectral comparison vs the phase-aligned true tonal
    truth = aligned_tonal(case["tonal"], f0)  # (M, N)
    pred = model.synthesize(case["mics"], case["rps"], truth.shape[-1] / FS, f0=f0)
    corrs = [float(np.corrcoef(truth[m], pred[m])[0, 1]) for m in range(truth.shape[0])]
    dloud = [_loudness_db(pred[m]) - _loudness_db(truth[m]) for m in range(truth.shape[0])]
    row = {
        "key": case["key"],
        "rps": case["rps"],
        "seed": case["seed"],
        "f0": round(f0, 4),
        "coeff_rmse": round(coeff_rmse, 6),
        "rel_resid": round(rel_resid, 4),
        "corr_mean": round(float(np.mean(corrs)), 4),
        "corr_min": round(float(np.min(corrs)), 4),
        "dLoud_dB_mean": round(float(np.mean(dloud)), 3),
    }
    payload = {"truth": truth, "pred": pred, "f0": f0, "case": case}
    return row, payload


def overlay_png(payload: dict, out_path: Path, n_show: int = 3) -> None:
    """Spectral + time overlays for a few mics (nearest / median / farthest)."""
    truth, pred, f0 = payload["truth"], payload["pred"], payload["f0"]
    case = payload["case"]
    r = np.linalg.norm(case["mics"], axis=1)
    order = np.argsort(r)
    picks = [order[0], order[len(order) // 2], order[-1]][:n_show]
    n = truth.shape[-1]
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    han = np.hanning(n)
    fig, axs = plt.subplots(len(picks), 2, figsize=(13, 2.6 * len(picks)), squeeze=False)
    for row_i, m in enumerate(picks):
        st = 20 * np.log10(np.abs(np.fft.rfft(truth[m] * han)) + 1e-9)
        sp = 20 * np.log10(np.abs(np.fft.rfft(pred[m] * han)) + 1e-9)
        ax = axs[row_i, 0]
        ax.plot(freqs, st, "k", lw=0.6, label="true tonal")
        ax.plot(freqs, sp, "r", lw=0.6, alpha=0.8, label="GP")
        ax.set_xlim(0, 6000)
        ax.set_ylim(np.percentile(st, 5) - 10, st.max() + 10)
        ax.set_title(f"{case['key']} mic{m} (|r|={r[m]:.2f} m) spectrum [dB]", fontsize=8)
        t = np.arange(n) / FS
        sl = (t >= 0.5) & (t <= 0.5 + 3.0 / f0)
        ax2 = axs[row_i, 1]
        ax2.plot(t[sl], truth[m][sl], "k", lw=0.8, label="true tonal")
        ax2.plot(t[sl], pred[m][sl], "r", lw=0.8, label="GP")
        ax2.set_title("time (3 BPF periods)", fontsize=8)
    axs[0, 0].legend(fontsize=7)
    axs[0, 1].legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drone", choices=DRONES, default="dregon")
    p.add_argument("--out", default=None, help="default: results/gp_egonoise/<drone>")
    p.add_argument("--dataset", default="drone-egonoise")
    p.add_argument("--version", default=None, help="dload version (default: latest)")
    p.add_argument("--n_harm", type=int, default=60)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--holdout_rps", type=float, nargs="*", default=[60.0])
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="4 mics, H=8, <=2 rps, few iters")
    args = p.parse_args()

    max_mics = None
    if args.smoke:
        args.n_harm, args.iters = 8, 20
        max_mics = 4

    out = Path(args.out or f"results/gp_egonoise/{args.drone}")
    out.mkdir(parents=True, exist_ok=True)

    inv = sweep_inventory(args.drone, args.dataset, args.version)
    if not inv:
        raise SystemExit(f"no samples for drone={args.drone} in {args.dataset}")
    all_rps = sorted({r for r, _ in inv})
    print(f"[data] {args.drone}: {len(inv)} cases, rps={all_rps}", flush=True)

    holdout = set(args.holdout_rps)
    if args.smoke:
        train_rps = [r for r in all_rps if r not in holdout][:2] or all_rps[:1]
        holdout = holdout & set(all_rps) if len(all_rps) > 2 else set()
    else:
        train_rps = [r for r in all_rps if r not in holdout]

    def is_train(c):
        return c["rps"] in train_rps and c["seed"] == args.train_seed

    def is_val(c):
        return (c["rps"] in holdout) or (c["rps"] in train_rps and c["seed"] != args.train_seed)

    cfg = EgonoiseGPConfig(
        drone=args.drone,
        n_harm=args.n_harm,
        iters=args.iters,
        lr=args.lr,
        train_seed=args.train_seed,
        holdout_rps=tuple(sorted(holdout)),
    )
    model = EgonoiseGPModel(cfg)
    train_stream = (
        c
        for c in iter_cases(
            args.drone, args.dataset, args.version, arrays=("tonal", "broadband"), max_mics=max_mics
        )
        if is_train(c)
    )
    summary = model.fit(train_stream)
    print(f"[fit] {json.dumps(summary)}", flush=True)
    model.save(out / "best.pt")
    print(f"[fit] saved -> {out / 'best.pt'}", flush=True)

    rows = []
    for c in iter_cases(
        args.drone, args.dataset, args.version, arrays=("tonal",), max_mics=max_mics
    ):
        if not is_val(c):
            continue
        row, payload = evaluate_case(model, c)
        row["split"] = "rps_interp" if c["rps"] in holdout else "seed_holdout"
        rows.append(row)
        print(f"[eval] {json.dumps(row)}", flush=True)
        # overlays: every interp case + the lowest/highest seed-holdout rps
        if c["rps"] in holdout or c["rps"] in (min(train_rps), max(train_rps)):
            overlay_png(payload, out / f"overlay_R{c['rps']:.0f}_s{c['seed']}.png")
    if not rows:
        print("[eval] WARNING: no validation cases available (partial dataset?)", flush=True)

    def agg(split):
        sel = [r for r in rows if r["split"] == split]
        if not sel:
            return None
        return {
            "n": len(sel),
            "coeff_rmse_mean": float(np.mean([r["coeff_rmse"] for r in sel])),
            "rel_resid_mean": float(np.mean([r["rel_resid"] for r in sel])),
            "corr_mean": float(np.mean([r["corr_mean"] for r in sel])),
            "dLoud_dB_mean_abs": float(np.mean([abs(r["dLoud_dB_mean"]) for r in sel])),
        }

    metrics = {
        "drone": args.drone,
        "config": asdict(cfg),
        "fit": summary,
        "train_rps": train_rps,
        "holdout_rps": sorted(holdout),
        "seed_holdout": agg("seed_holdout"),
        "rps_interp": agg("rps_interp"),
        "cases": rows,
    }
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[done] metrics -> {out / 'eval_metrics.json'}", flush=True)
    for split in ("seed_holdout", "rps_interp"):
        print(f"[done] {split}: {json.dumps(metrics[split])}", flush=True)


if __name__ == "__main__":
    main()
