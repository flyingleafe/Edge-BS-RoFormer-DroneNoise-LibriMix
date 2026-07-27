"""CKLA mechanistic activation analysis (docs/ckla-design.md §6 diagnostics kit).

Instruments the trained CKLA models on REAL validation clips and answers six
mechanistic questions (results write-up:
``docs/experiments/ckla-activation-analysis.md``):

1. Precision-gating adaptivity — is the evidence precision λ_v(t) actually
   modulated within a clip (CV, correlation with frame energy / speech-band
   fraction), or is the layer a fixed smoother?
2. State precision λ_t dynamics — do the per-slot precisions saturate to a
   constant (⇒ fixed EMA bank) and what is the effective Kalman gain φ/λ?
3. Readout horizon selection — does the query redistribute readout mass
   across the multi-scale slot bank over time, or is the slot mix static?
4. Rotation usage on real data — actual ω_t−ω0 excursions, their correlation
   with GT RPS level/derivative, plus the causal 3-arm test: intact vs
   rotation-zeroed vs imaginary-readout-zeroed PIT-MAE.
5. Where does RPS become decodable — ridge probes at trunk / after block 1 /
   after block 2, CKLA vs the g2_if transformer comparator.
6. Amplitude-shortcut sensitivity — prediction shifts under spectral
   recoloring / global gain / frequency scaling ×1.02 for both architectures
   (timbre-reader vs comb-reader hypothesis).

Protocol: N clips (default 12 = 8 dregon_cruise + 4 fly124_cruise, seeded)
from the vk_valid_comparison clip table (``scripts/rps_predictor_vk_eval.py``),
mic channel 0, CPU, fp32, no_grad. Checkpoints via the same Hydra-compose
route as the vk eval.

Run (laptop CPU, ~10 min):
  python scripts/ckla_activation_analysis.py \
      --data <path-to>/DREGON-LM-V4-michaels-full/valid
Outputs under ``--out`` (default results/ckla_activation_analysis/):
``summary.txt`` (all tables), ``report.json`` (all numbers),
``ablation_per_clip.csv``, ``fig_a1..fig_a6*.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))  # for importing rps_predictor_vk_eval

SR = 16000
HOP = 512
N_FFT = 2048
FRAME_S = HOP / SR
N_ROTORS = 4
SPEECH_BAND = (300.0, 3000.0)

# name -> (experiment config, checkpoint URI)
MODELS: dict[str, tuple[str, str]] = {
    "ckla_p1": ("ckla_p1_if", "r2://ml-data/artifacts/ckla_p1_if/checkpoints/best.ckpt"),
    "ckla_p0": (
        "ckla_p0_staticcomb",
        "r2://ml-data/artifacts/ckla_p0_staticcomb/checkpoints/best.ckpt",
    ),
    "g2_if": (
        "g2_if_transformer",
        "r2://ml-data/artifacts/g2_if_transformer/checkpoints/best.ckpt",
    ),
    # Mechanistic-lever arms (docs/experiments/ckla.md): the λ-gain probe
    # (analysis 2) on ckla_pnoise verifies the gain stays alive; the
    # scale-response probe (analysis 6) on ckla_freqscale verifies the
    # amplitude anchor broke.
    "ckla_pnoise": (
        "ckla_p1_pnoise",
        "r2://ml-data/artifacts/ckla_p1_pnoise/checkpoints/best.ckpt",
    ),
    # Freq-scale v2 (p=1.0, alpha [0.7,1.3]) and v3 synthesis-first arms —
    # the scale-response probe (analysis 6) is their success criterion.
    "g2_if_freqscale_v2": (
        "g2_if_freqscale_v2",
        "r2://ml-data/artifacts/g2_if_freqscale_v2/checkpoints/best.ckpt",
    ),
    "ckla_pnoise_fs_v2": (
        "ckla_pnoise_fs_v2",
        "r2://ml-data/artifacts/ckla_pnoise_fs_v2/checkpoints/best.ckpt",
    ),
    "g2_if_v3synth": (
        "g2_if_v3synth",
        "r2://ml-data/artifacts/g2_if_v3synth/checkpoints/best.ckpt",
    ),
    "ckla_pnoise_v3synth": (
        "ckla_pnoise_v3synth",
        "r2://ml-data/artifacts/ckla_pnoise_v3synth/checkpoints/best.ckpt",
    ),
    "ckla_freqscale": (
        "ckla_p1_freqscale",
        "r2://ml-data/artifacts/ckla_p1_freqscale/checkpoints/best.ckpt",
    ),
}
# Names routed through the CKLA instrumented forward. Membership is also
# checked structurally at dispatch (any model whose head is a
# TemporalCKLAHead) so newly-registered CKLA arms (ckla_pnoise,
# ckla_freqscale, ...) are not silently sent down the transformer-taps path
# — that mis-dispatch crashed the first lever-probe run (Slurm 20928550).
CKLA_MODELS = ("ckla_p1", "ckla_p0", "ckla_pnoise", "ckla_freqscale")


def _is_ckla(model) -> bool:
    from models.ckla import TemporalCKLAHead

    return isinstance(getattr(model, "head", None), TemporalCKLAHead)


# ─── helpers ─────────────────────────────────────────────────────────────────


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def select_clips(n_clips: int, seed: int) -> list[tuple]:
    """Seeded selection: ~2/3 dregon_cruise, ~1/3 fly124_cruise (pool ratio)."""
    import rps_predictor_vk_eval as vk

    dregon = [c for c in vk.CLIPS if c[1] in vk.DREGON_RECS and c[3] == "cruise"]
    fly = [c for c in vk.CLIPS if c[1] == "michaels_FLY124" and c[3] == "cruise"]
    n_fly = max(1, round(n_clips * len(fly) / (len(fly) + len(dregon))))
    n_dregon = n_clips - n_fly
    rng = np.random.default_rng(seed)
    sel_d = sorted(rng.choice(len(dregon), size=min(n_dregon, len(dregon)), replace=False))
    sel_f = sorted(rng.choice(len(fly), size=min(n_fly, len(fly)), replace=False))
    return [dregon[i] for i in sel_d] + [fly[i] for i in sel_f]


def frame_features(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Frame-grid acoustic covariates on the model's STFT grid (2048/512,
    center=True — same grid as the stft_mag_if front-end, so T matches the
    captured internals). Returns log_energy (T,), speech_frac (T,) — energy
    fraction in 300–3000 Hz — and the power spectrogram for plotting."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    spec = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        window=torch.hann_window(N_FFT),
        center=True,
        return_complex=True,
    )
    power = (spec.real**2 + spec.imag**2).numpy()  # (F, T)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    total = power.sum(axis=0) + 1e-12
    band = (freqs >= SPEECH_BAND[0]) & (freqs <= SPEECH_BAND[1])
    return {
        "log_energy": np.log10(total),
        "speech_frac": power[band].sum(axis=0) / total,
        "power": power,
        "freqs": freqs,
    }


def ckla_forward_instrumented(model, audio: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
    """One captured forward: returns (pred (4,T), per-layer summary dicts,
    probe taps {trunk (T,128), block1 (T,dm), block2 (T,dm)}).

    Per-layer summaries reduce the full (1,T,N,D) internals immediately to
    keep memory light: lam_slot/phi_slot/p_slot/omega (T,N), lv/phi_mean (T,),
    horizon (N,) in frames.
    """
    import torch

    mixers = [blk.mixer for blk in model.head.blocks]
    taps: dict[str, np.ndarray] = {}
    hooks = []

    def tap_pool(_m, _i, out):
        taps["trunk"] = out.detach().numpy()[0].T  # (B,128,T) -> (T,128)

    hooks.append(model.freq_pool.register_forward_hook(tap_pool))
    for bi, blk in enumerate(model.head.blocks):

        def tap_block(_m, _i, out, bi=bi):
            taps[f"block{bi + 1}"] = out.detach().numpy()[0]  # (B,T,dm) -> (T,dm)

        hooks.append(blk.register_forward_hook(tap_block))

    for m in mixers:
        m.capture = []
        m.capture_state = True
    try:
        with torch.no_grad():
            pred = model(torch.from_numpy(audio[None].astype(np.float32)))
        layers = []
        for m in mixers:
            cap = m.capture[-1]
            k = cap["k"].numpy()[0]  # (T, N)
            lam_v = cap["lam_v"].numpy()[0]  # (T, D)
            lam = cap["lam"].numpy()[0]  # (T, N, D)
            contrib = cap["contrib"].numpy()[0]  # (T, N)
            omega = cap["omega"].numpy()[0]  # (T, N)
            abar = cap["abar_mag"].numpy()  # (N, D)
            lv_mean = lam_v.mean(axis=-1)  # (T,)
            phi_slot = (k**2) * lv_mean[:, None]  # (T, N) — mean over D of k²λv
            mass = np.abs(contrib)
            p_slot = mass / (mass.sum(axis=-1, keepdims=True) + 1e-12)
            layers.append(
                {
                    "lv": lv_mean,
                    "phi_mean": phi_slot.mean(axis=-1),
                    "lam_slot": lam.mean(axis=-1),  # (T, N)
                    "phi_slot": phi_slot,
                    "p_slot": p_slot,
                    "omega": omega,
                    "horizon": np.median(1.0 / np.maximum(1.0 - abar, 1e-6), axis=-1),  # (N,)
                }
            )
    finally:
        for m in mixers:
            m.capture = None
            m.capture_state = False
        for h in hooks:
            h.remove()
    return pred.numpy()[0], layers, taps


def transformer_forward_taps(model, audio: np.ndarray) -> tuple[np.ndarray, dict]:
    """Plain forward + probe taps for SimpleConvV2Transformer (trunk,
    after transformer encoder layer 1, after layer 2)."""
    import torch

    taps: dict[str, np.ndarray] = {}
    hooks = []

    def tap_pool(_m, _i, out):
        taps["trunk"] = out.detach().numpy()[0].T

    hooks.append(model.freq_pool.register_forward_hook(tap_pool))
    for li, layer in enumerate(model.head.transformer.layers):

        def tap_layer(_m, _i, out, li=li):
            taps[f"block{li + 1}"] = out.detach().numpy()[0]  # (B,T,dm)

        hooks.append(layer.register_forward_hook(tap_layer))
    try:
        with torch.no_grad():
            pred = model(torch.from_numpy(audio[None].astype(np.float32)))
    finally:
        for h in hooks:
            h.remove()
    return pred.numpy()[0], taps


def plain_forward(model, audio: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(audio[None].astype(np.float32))).numpy()[0]


def pit_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    from tasks.rps_prediction import align_rps_to_gt

    return float(np.mean(np.abs(align_rps_to_gt(pred, gt) - gt)))


# ─── perturbations (analysis 6) ──────────────────────────────────────────────


def spectral_tilt(audio: np.ndarray, tilt_db: float) -> np.ndarray:
    """Linear-in-frequency gain tilt across 0–8 kHz, ±tilt_db/2 at the edges
    (0 dB at 4 kHz), applied in the STFT domain (1024/256, hann, iSTFT)."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    win = torch.hann_window(1024)
    spec = torch.stft(x, n_fft=1024, hop_length=256, window=win, center=True, return_complex=True)
    freqs = torch.from_numpy(np.fft.rfftfreq(1024, 1.0 / SR).astype(np.float32))
    gain = 10.0 ** ((tilt_db * (freqs / 8000.0 - 0.5)) / 20.0)
    out = torch.istft(
        spec * gain[:, None], n_fft=1024, hop_length=256, window=win, length=len(audio)
    )
    return out.numpy()


def freq_scale(audio: np.ndarray, factor_num: int = 51, factor_den: int = 50) -> np.ndarray:
    """Playback-rate change: all frequencies ×(factor_num/factor_den) = ×1.02."""
    from scipy.signal import resample_poly

    return resample_poly(audio.astype(np.float64), factor_den, factor_num).astype(np.float32)


# ─── ridge probe (analysis 5) ────────────────────────────────────────────────


def ridge_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Closed-form ridge on standardized features, alpha picked on an inner
    80/20 split of the train frames. Targets: the per-frame SORTED (ascending)
    GT RPS vector — removes rotor permutation ambiguity (documented choice)."""
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    xt = (x_train - mu) / sd
    xs = (x_test - mu) / sd
    ym = y_train.mean(axis=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(xt))
    n_in = int(0.8 * len(idx))
    tr, va = idx[:n_in], idx[n_in:]

    def fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        d = x.shape[1]
        return np.linalg.solve(x.T @ x + alpha * np.eye(d), x.T @ (y - ym))

    best_alpha, best_err = 1.0, np.inf
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        w = fit(xt[tr], y_train[tr], alpha)
        err = float(np.mean(np.abs(xt[va] @ w + ym - y_train[va])))
        if err < best_err:
            best_alpha, best_err = alpha, err
    w = fit(xt, y_train, best_alpha)
    pred = xs @ w + ym
    resid = pred - y_test
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_test - y_test.mean(axis=0)) ** 2))
    return {
        "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "mae": float(np.mean(np.abs(resid))),
        "alpha": best_alpha,
    }


# ─── figures ─────────────────────────────────────────────────────────────────


def fig_lamv_overlay(out_dir: Path, clip_id: str, feats: dict, layers: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(layers[0]["lv"])) * FRAME_S
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 3.2 * len(layers)), sharex=True)
    axes = np.atleast_1d(axes)
    fmax_bin = int(4000 / (SR / N_FFT))
    for li, (ax, lay) in enumerate(zip(axes, layers)):
        ax.imshow(
            10 * np.log10(feats["power"][:fmax_bin] + 1e-10),
            origin="lower",
            aspect="auto",
            extent=(0, t[-1], 0, 4000),
            cmap="magma",
        )
        ax.set_ylabel(f"layer {li + 1}\nfreq (Hz)")
        ax2 = ax.twinx()
        ax2.plot(t, lay["lv"], color="cyan", lw=1.5, label="mean λ_v(t)")
        ax2.set_ylabel("λ_v (mean over ch)", color="cyan")
        ax2.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"A1 — evidence precision λ_v(t) over spectrogram — {clip_id}")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_a1_lamv_{clip_id}.png", dpi=130)
    plt.close(fig)


def fig_lambda_traj(out_dir: Path, model_name: str, per_layer: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(per_layer), figsize=(6 * len(per_layer), 7), sharex=True)
    axes = np.atleast_2d(axes)
    for li, lay in enumerate(per_layer):
        lam = lay["lam_mean"]  # (T, N) mean over clips/channels
        gain = lay["gain_mean"]
        order = np.argsort(lay["horizon"])
        t = np.arange(lam.shape[0]) * FRAME_S
        cmap = plt.get_cmap("viridis")
        for rank, n in enumerate(order):
            c = cmap(rank / max(len(order) - 1, 1))
            axes[0, li].plot(t, lam[:, n], color=c, lw=1)
            axes[1, li].plot(t, gain[:, n], color=c, lw=1)
        axes[0, li].set_yscale("log")
        axes[0, li].set_title(f"{model_name} layer {li + 1} — λ_t per slot (color=horizon rank)")
        axes[0, li].set_ylabel("λ (mean ch, clips)")
        axes[1, li].set_yscale("log")
        axes[1, li].set_ylabel("gain φ/λ")
        axes[1, li].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_a2_lambda_gain_{model_name}.png", dpi=130)
    plt.close(fig)


def fig_slotmass(out_dir: Path, clip_id: str, layers: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 2.6 * len(layers)), sharex=True)
    axes = np.atleast_1d(axes)
    for li, (ax, lay) in enumerate(zip(axes, layers)):
        order = np.argsort(lay["horizon"])
        t_end = lay["p_slot"].shape[0] * FRAME_S
        im = ax.imshow(
            lay["p_slot"][:, order].T,
            origin="lower",
            aspect="auto",
            extent=(0, t_end, -0.5, len(order) - 0.5),
            cmap="viridis",
        )
        ax.set_ylabel(f"layer {li + 1}\nslot (horizon rank)")
        fig.colorbar(im, ax=ax, label="readout mass")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"A3 — readout mass over slots — {clip_id}")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_a3_slotmass_{clip_id}.png", dpi=130)
    plt.close(fig)


def fig_rotation(out_dir: Path, exc_std: list[np.ndarray], ablation: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for li, es in enumerate(exc_std):
        ax1.plot(np.arange(len(es)), es, "o-", label=f"layer {li + 1}")
    ax1.set_xlabel("slot")
    ax1.set_ylabel("std_t(ω_t − ω0) (rad)")
    ax1.set_title("A4 — rotation excursion per slot (mean over clips)")
    ax1.legend()
    pools = list(next(iter(ablation.values())).keys())
    arms = list(ablation.keys())
    width = 0.8 / len(arms)
    for ai, arm in enumerate(arms):
        vals = [ablation[arm][p] for p in pools]
        ax2.bar(np.arange(len(pools)) + ai * width, vals, width, label=arm)
    ax2.set_xticks(np.arange(len(pools)) + 0.4 - width / 2)
    ax2.set_xticklabels(pools)
    ax2.set_ylabel("PIT-MAE (rev/s)")
    ax2.set_title("A4 — causal ablations")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_a4_rotation.png", dpi=130)
    plt.close(fig)


def fig_bars(
    out_dir: Path, fname: str, title: str, ylabel: str, groups: dict[str, dict[str, float]]
) -> None:
    """Grouped bar chart: groups[series][category] = value."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = list(next(iter(groups.values())).keys())
    series = list(groups.keys())
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(1.6 * len(cats) + 3, 4))
    for si, s in enumerate(series):
        vals = [groups[s].get(c, np.nan) for c in cats]
        ax.bar(np.arange(len(cats)) + si * width, vals, width, label=s)
    ax.set_xticks(np.arange(len(cats)) + 0.4 - width / 2)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=130)
    plt.close(fig)


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--data", default="dload:DREGON-LM-V4-michaels-valid-full")
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=["ckla_p1", "ckla_p0", "g2_if"],
        help="ckla_p0 adds the static-comb P0 model to analyses 1-3 as a reference",
    )
    ap.add_argument("--out", default="results/ckla_activation_analysis")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import rps_predictor_vk_eval as vk

    clips = select_clips(args.clips, args.seed)
    clip_ids = [c[0] for c in clips]
    pool_of = {
        c[0]: ("dregon_cruise" if c[1] in vk.DREGON_RECS else "fly124_cruise") for c in clips
    }
    print(f"[ckla_aa] clips ({len(clips)}): {clip_ids}", flush=True)

    audio_all, gt_all = vk.load_clip_data(args.data)
    audio = {cid: audio_all[cid][0] for cid in clip_ids}  # ch0, (128000,)
    gt = {cid: gt_all[cid] for cid in clip_ids}  # (4, 251)
    feats = {cid: frame_features(audio[cid]) for cid in clip_ids}

    models: dict[str, Any] = {}
    for name in args.models:
        experiment, uri = MODELS[name]
        print(f"[ckla_aa] loading {name} ({experiment})", flush=True)
        models[name] = vk.load_model(experiment, uri, "cpu")

    report: dict[str, Any] = {
        "clips": clip_ids,
        "pools": pool_of,
        "seed": args.seed,
        "data": args.data,
    }
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    # ── instrumented forwards (CKLA models) + taps (all models) ────────────
    ckla_caps: dict[str, dict[str, list[dict]]] = {}  # model -> clip -> layer summaries
    preds: dict[str, dict[str, np.ndarray]] = {}  # model -> clip -> (4, T)
    taps: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, model in models.items():
        preds[name], taps[name] = {}, {}
        if name in CKLA_MODELS or _is_ckla(model):
            ckla_caps[name] = {}
            for cid in clip_ids:
                p, layers, tp = ckla_forward_instrumented(model, audio[cid])
                preds[name][cid], ckla_caps[name][cid], taps[name][cid] = p, layers, tp
        else:
            for cid in clip_ids:
                p, tp = transformer_forward_taps(model, audio[cid])
                preds[name][cid], taps[name][cid] = p, tp
        maes = [pit_mae(preds[name][cid], gt[cid]) for cid in clip_ids]
        emit(f"[forward] {name}: mean PIT-MAE over {len(clip_ids)} clips = {np.mean(maes):.3f}")

    example_clips = [
        next(cid for cid in clip_ids if pool_of[cid] == "dregon_cruise"),
        next(cid for cid in clip_ids if pool_of[cid] == "fly124_cruise"),
    ]

    # ═══ A1: precision-gating adaptivity ═══════════════════════════════════
    emit("\n═══ A1 — precision-gating adaptivity (λ_v) ═══")
    emit(f"{'model':<10}{'layer':<7}{'CV(λ_v)':>10}{'r(logE)':>10}{'r(speech)':>11}{'CV(φ)':>10}")
    a1: dict[str, Any] = {}
    for name in ckla_caps:
        a1[name] = []
        n_layers = len(next(iter(ckla_caps[name].values())))
        for li in range(n_layers):
            cvs, r_e, r_s, cvp = [], [], [], []
            for cid in clip_ids:
                lay = ckla_caps[name][cid][li]
                lv, phi = lay["lv"], lay["phi_mean"]
                cvs.append(lv.std() / (lv.mean() + 1e-12))
                cvp.append(phi.std() / (phi.mean() + 1e-12))
                r_e.append(pearson(lv, feats[cid]["log_energy"]))
                r_s.append(pearson(lv, feats[cid]["speech_frac"]))
            row = {
                "layer": li + 1,
                "cv_lam_v": float(np.mean(cvs)),
                "cv_lam_v_std": float(np.std(cvs)),
                "r_log_energy": float(np.nanmean(r_e)),
                "r_speech_frac": float(np.nanmean(r_s)),
                "cv_phi": float(np.mean(cvp)),
            }
            a1[name].append(row)
            emit(
                f"{name:<10}{li + 1:<7}{row['cv_lam_v']:>10.4f}{row['r_log_energy']:>10.3f}"
                f"{row['r_speech_frac']:>11.3f}{row['cv_phi']:>10.4f}"
            )
    report["a1_precision_gating"] = a1
    for cid in example_clips:
        if "ckla_p1" in ckla_caps:
            fig_lamv_overlay(out_dir, cid, feats[cid], ckla_caps["ckla_p1"][cid])

    # ═══ A2: state precision dynamics ══════════════════════════════════════
    emit("\n═══ A2 — state precision λ_t dynamics ═══")
    emit(
        f"{'model':<10}{'layer':<7}{'med t_sat(s)':>13}{'max t_sat(s)':>13}"
        f"{'CV(λ) post-sat':>15}{'med gain(end)':>15}{'gain end/start':>15}"
    )
    a2: dict[str, Any] = {}
    for name in ckla_caps:
        a2[name] = []
        n_layers = len(next(iter(ckla_caps[name].values())))
        for li in range(n_layers):
            lam_mean = np.mean([ckla_caps[name][cid][li]["lam_slot"] for cid in clip_ids], axis=0)
            phi_mean = np.mean([ckla_caps[name][cid][li]["phi_slot"] for cid in clip_ids], axis=0)
            horizon = ckla_caps[name][clip_ids[0]][li]["horizon"]
            gain = phi_mean / (lam_mean + 1e-12)
            n_slots = lam_mean.shape[1]
            t_sat = np.zeros(n_slots)
            cv_post = np.zeros(n_slots)
            for n in range(n_slots):
                traj = lam_mean[:, n]
                final = traj[-1]
                idx = np.nonzero(traj >= 0.95 * final)[0]
                t0 = int(idx[0]) if len(idx) else len(traj) - 1
                t_sat[n] = t0 * FRAME_S
                post = traj[t0:]
                cv_post[n] = post.std() / (post.mean() + 1e-12)
            g_start = gain[:10].mean(axis=0)
            g_end = gain[-50:].mean(axis=0)
            row = {
                "layer": li + 1,
                "t_sat_median_s": float(np.median(t_sat)),
                "t_sat_max_s": float(np.max(t_sat)),
                "cv_lam_post_sat_mean": float(np.mean(cv_post)),
                "gain_end_median": float(np.median(g_end)),
                "gain_end_over_start_median": float(np.median(g_end / (g_start + 1e-12))),
                "t_sat_per_slot_s": t_sat.tolist(),
                "horizon_per_slot_frames": horizon.tolist(),
                "gain_end_per_slot": g_end.tolist(),
            }
            a2[name].append(row)
            emit(
                f"{name:<10}{li + 1:<7}{row['t_sat_median_s']:>13.2f}{row['t_sat_max_s']:>13.2f}"
                f"{row['cv_lam_post_sat_mean']:>15.4f}{row['gain_end_median']:>15.3g}"
                f"{row['gain_end_over_start_median']:>15.3g}"
            )
        per_layer_fig = [
            {
                "lam_mean": np.mean([ckla_caps[name][cid][li]["lam_slot"] for cid in clip_ids], 0),
                "gain_mean": np.mean(
                    [
                        ckla_caps[name][cid][li]["phi_slot"]
                        / (ckla_caps[name][cid][li]["lam_slot"] + 1e-12)
                        for cid in clip_ids
                    ],
                    0,
                ),
                "horizon": ckla_caps[name][clip_ids[0]][li]["horizon"],
            }
            for li in range(n_layers)
        ]
        fig_lambda_traj(out_dir, name, per_layer_fig)
    report["a2_lambda_dynamics"] = a2

    # ═══ A3: readout horizon selection ═════════════════════════════════════
    emit("\n═══ A3 — readout horizon selection (contrib mass over slots) ═══")
    emit(
        f"{'model':<10}{'layer':<7}{'H (bits)':>12}{'H std':>8}"
        f"{'std_t(p) mean':>14}{'r(longmass,speech)':>20}"
    )
    a3: dict[str, Any] = {}
    for name in ckla_caps:
        a3[name] = []
        n_layers = len(next(iter(ckla_caps[name].values())))
        for li in range(n_layers):
            ents, stds, r_long = [], [], []
            n_slots = ckla_caps[name][clip_ids[0]][li]["p_slot"].shape[1]
            for cid in clip_ids:
                lay = ckla_caps[name][cid][li]
                p = lay["p_slot"]  # (T, N)
                ent = -(p * np.log2(p + 1e-12)).sum(axis=-1)
                ents.append(ent)
                stds.append(p.std(axis=0).mean())
                order = np.argsort(lay["horizon"])
                long_mass = p[:, order[len(order) // 2 :]].sum(axis=-1)
                r_long.append(pearson(long_mass, feats[cid]["speech_frac"]))
            ent_all = np.concatenate(ents)
            row = {
                "layer": li + 1,
                "entropy_bits_mean": float(ent_all.mean()),
                "entropy_bits_std": float(ent_all.std()),
                "entropy_max_bits": float(np.log2(n_slots)),
                "std_t_p_mean": float(np.mean(stds)),
                "r_long_mass_speech": float(np.nanmean(r_long)),
            }
            a3[name].append(row)
            emit(
                f"{name:<10}{li + 1:<7}{row['entropy_bits_mean']:>12.3f}"
                f"{row['entropy_bits_std']:>8.3f}{row['std_t_p_mean']:>14.4f}"
                f"{row['r_long_mass_speech']:>20.3f}"
            )
    report["a3_readout_selection"] = a3
    for cid in example_clips:
        if "ckla_p1" in ckla_caps:
            fig_slotmass(out_dir, cid, ckla_caps["ckla_p1"][cid])

    # ═══ A4: rotation usage on real data (ckla_p1) ═════════════════════════
    a4: dict[str, Any] = {}
    if "ckla_p1" in models:
        emit("\n═══ A4 — rotation usage (ckla_p1) ═══")
        n_layers = len(next(iter(ckla_caps["ckla_p1"].values())))
        mixers = [blk.mixer for blk in models["ckla_p1"].head.blocks]
        exc_std_layers, exc_rows = [], []
        for li in range(n_layers):
            omega0 = mixers[li].omega0.detach().numpy()  # (N,)
            exc_all, gt_mean_all, gt_deriv_all = [], [], []
            per_slot_std = []
            for cid in clip_ids:
                omega = ckla_caps["ckla_p1"][cid][li]["omega"]  # (T, N)
                exc = omega - omega0[None, :]
                per_slot_std.append(exc.std(axis=0))
                exc_all.append(np.abs(exc).mean(axis=-1))  # (T,)
                g_mean = gt[cid].mean(axis=0)
                gt_mean_all.append(g_mean)
                gt_deriv_all.append(np.abs(np.gradient(g_mean) / FRAME_S))
            slot_std = np.mean(per_slot_std, axis=0)  # (N,)
            exc_std_layers.append(slot_std)
            exc_cat = np.concatenate(exc_all)
            row = {
                "layer": li + 1,
                "exc_std_per_slot_rad": slot_std.tolist(),
                "exc_std_median_rad": float(np.median(slot_std)),
                "exc_std_max_rad": float(np.max(slot_std)),
                "r_exc_gt_rps": pearson(exc_cat, np.concatenate(gt_mean_all)),
                "r_exc_gt_rps_deriv": pearson(exc_cat, np.concatenate(gt_deriv_all)),
            }
            exc_rows.append(row)
            emit(
                f"layer {li + 1}: std(ω−ω0) median {row['exc_std_median_rad']:.4f} rad, "
                f"max {row['exc_std_max_rad']:.4f} rad | r(|exc|, GT rps) "
                f"{row['r_exc_gt_rps']:.3f} | r(|exc|, |dGT/dt|) {row['r_exc_gt_rps_deriv']:.3f}"
            )
        a4["excursions"] = exc_rows

        # Causal 3-arm test.
        emit("\ncausal arms (per-pool mean PIT-MAE, rev/s):")
        model_rot0 = deepcopy(models["ckla_p1"])
        with torch.no_grad():
            for blk in model_rot0.head.blocks:
                m = blk.mixer
                m.s.zero_()
                m.omega0.zero_()
                m.omega_proj.weight.zero_()
                m.omega_proj.bias.zero_()
        model_im0 = deepcopy(models["ckla_p1"])
        with torch.no_grad():
            for blk in model_im0.head.blocks:
                m = blk.mixer
                dm = m.d_model
                m.mix.weight[:, dm:].zero_()
        arms = {"intact": None, "rot_zero": model_rot0, "im_zero": model_im0}
        per_clip_mae: dict[str, dict[str, float]] = {a: {} for a in arms}
        for arm, mdl in arms.items():
            for cid in clip_ids:
                p = preds["ckla_p1"][cid] if mdl is None else plain_forward(mdl, audio[cid])
                per_clip_mae[arm][cid] = pit_mae(p, gt[cid])
        pools = ("dregon_cruise", "fly124_cruise", "all")
        ablation_pooled: dict[str, dict[str, float]] = {}
        emit(f"{'arm':<10}" + "".join(f"{p:>16}" for p in pools))
        for arm in arms:
            ablation_pooled[arm] = {}
            for pool in pools:
                vals = [
                    per_clip_mae[arm][cid]
                    for cid in clip_ids
                    if pool == "all" or pool_of[cid] == pool
                ]
                ablation_pooled[arm][pool] = float(np.mean(vals))
            emit(f"{arm:<10}" + "".join(f"{ablation_pooled[arm][p]:>16.3f}" for p in pools))
        a4["ablation_pooled"] = ablation_pooled
        a4["ablation_per_clip"] = per_clip_mae
        with open(out_dir / "ablation_per_clip.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "pool"] + list(arms))
            for cid in clip_ids:
                w.writerow([cid, pool_of[cid]] + [f"{per_clip_mae[a][cid]:.4f}" for a in arms])
        fig_rotation(out_dir, exc_std_layers, ablation_pooled)
    report["a4_rotation"] = a4

    # ═══ A5: linear probes ═════════════════════════════════════════════════
    emit("\n═══ A5 — ridge probes (target: sorted GT RPS vector, ascending) ═══")
    rng = np.random.default_rng(args.seed + 1)
    train_ids, test_ids = [], []
    for pool in ("dregon_cruise", "fly124_cruise"):
        ids = [cid for cid in clip_ids if pool_of[cid] == pool]
        perm = rng.permutation(len(ids))
        half = (len(ids) + 1) // 2
        train_ids += [ids[i] for i in perm[:half]]
        test_ids += [ids[i] for i in perm[half:]]
    emit(f"probe split: train {sorted(train_ids)} | test {sorted(test_ids)}")
    y_sorted = {cid: np.sort(gt[cid], axis=0).T for cid in clip_ids}  # (T, 4)
    a5: dict[str, Any] = {"train_clips": sorted(train_ids), "test_clips": sorted(test_ids)}
    emit(f"{'model':<10}{'tap':<9}{'R2':>8}{'MAE':>8}{'alpha':>8}")
    for name in models:
        a5[name] = {}
        for tap in ("trunk", "block1", "block2"):
            xtr = np.concatenate([taps[name][cid][tap] for cid in train_ids])
            ytr = np.concatenate([y_sorted[cid] for cid in train_ids])
            xte = np.concatenate([taps[name][cid][tap] for cid in test_ids])
            yte = np.concatenate([y_sorted[cid] for cid in test_ids])
            res = ridge_probe(xtr, ytr, xte, yte)
            a5[name][tap] = res
            emit(f"{name:<10}{tap:<9}{res['r2']:>8.3f}{res['mae']:>8.3f}{res['alpha']:>8.1f}")
    report["a5_probes"] = a5
    fig_bars(
        out_dir,
        "fig_a5_probe_r2.png",
        "A5 — linear decodability of RPS (ridge probe R² on held-out clips)",
        "R²",
        {
            name: {tap: a5[name][tap]["r2"] for tap in ("trunk", "block1", "block2")}
            for name in models
        },
    )

    # ═══ A6: amplitude-shortcut sensitivity ════════════════════════════════
    emit("\n═══ A6 — amplitude-shortcut sensitivity ═══")
    # A6 runs on every loaded model (was a hard-coded ("ckla_p1", "g2_if")
    # pair, which silently dropped the lever arms in the 20930287 probe run —
    # the freqscale model's scale response is the aug's success metric).
    sens_models = list(models)
    a6: dict[str, Any] = {}
    emit(
        f"{'model':<10}{'recolor+6':>11}{'recolor-6':>11}{'gain+6':>9}{'gain-6':>9}"
        f"{'scale ratio':>12}{'scale dev%':>11}"
    )
    for name in sens_models:
        d: dict[str, list[float]] = {k: [] for k in ("rc_up", "rc_dn", "g_up", "g_dn")}
        ratios = []
        for cid in clip_ids:
            base = preds[name][cid]
            variants = {
                "rc_up": spectral_tilt(audio[cid], +6.0),
                "rc_dn": spectral_tilt(audio[cid], -6.0),
                "g_up": audio[cid] * (10.0 ** (6.0 / 20.0)),
                "g_dn": audio[cid] * (10.0 ** (-6.0 / 20.0)),
            }
            for key, wav in variants.items():
                p = plain_forward(models[name], wav)
                p = vk.perm_align(p.astype(np.float64), base.astype(np.float64))
                d[key].append(float(np.mean(np.abs(p - base))))
            p_scaled = plain_forward(models[name], freq_scale(audio[cid]))
            ratios.append(float(p_scaled.mean() / base.mean()))
        ratio_mean = float(np.mean(ratios))
        row = {
            "recolor_up_dabs": float(np.mean(d["rc_up"])),
            "recolor_dn_dabs": float(np.mean(d["rc_dn"])),
            "gain_up_dabs": float(np.mean(d["g_up"])),
            "gain_dn_dabs": float(np.mean(d["g_dn"])),
            "scale_ratio_mean": ratio_mean,
            "scale_ratio_ideal": 1.02,
            "scale_deviation_pct": float(100.0 * (ratio_mean / 1.02 - 1.0)),
            "scale_ratio_per_clip": ratios,
        }
        a6[name] = row
        emit(
            f"{name:<10}{row['recolor_up_dabs']:>11.3f}{row['recolor_dn_dabs']:>11.3f}"
            f"{row['gain_up_dabs']:>9.3f}{row['gain_dn_dabs']:>9.3f}"
            f"{row['scale_ratio_mean']:>12.4f}{row['scale_deviation_pct']:>11.2f}"
        )
    report["a6_perturbations"] = a6
    if sens_models:
        fig_bars(
            out_dir,
            "fig_a6_perturbation.png",
            "A6 — mean |ΔRPS| under input perturbations (scale: % deviation from ×1.02)",
            "mean |ΔRPS| (rev/s) / dev %",
            {
                name: {
                    "recolor +6dB": a6[name]["recolor_up_dabs"],
                    "recolor −6dB": a6[name]["recolor_dn_dabs"],
                    "gain +6dB": a6[name]["gain_up_dabs"],
                    "gain −6dB": a6[name]["gain_dn_dabs"],
                    "scale dev %": abs(a6[name]["scale_deviation_pct"]),
                }
                for name in sens_models
            },
        )

    # ── write outputs ───────────────────────────────────────────────────────
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[ckla_aa] wrote {out_dir}/report.json, summary.txt, figures", flush=True)


if __name__ == "__main__":
    main()
