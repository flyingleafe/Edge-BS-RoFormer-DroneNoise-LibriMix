"""Side-by-side generalization probe for DCUNet / Edge-BS-RoFormer / MP-SENet.

The point of this module is to make one thing visible: *DCUNet fits the noise it
was trained on and fails on noise it was not*, while the two stronger
architectures degrade far more gracefully.

To see that you need an **in-distribution** condition, which none of the
published valid sets provide (they all hold noise out by design). So the three
conditions below walk a single axis — how far the noise is from what the model
trained on — while holding *everything else* fixed, including the same 25
held-out LibriSpeech speakers on all three:

===================  =====================================================
condition            noise provenance
===================  =====================================================
``seen``             the F1 Pass-A **training** pool itself
                     (``drone_seen``: DREGON minus the held-out recording,
                     michaels FLY125, and the *train* side of every shard
                     holdout) — noise these checkpoints were fitted on.
``unseen_rec``       ``drone``: the *valid* side — held-out recordings of
                     the **same** drone datasets.
``unseen_drone``     ``avq_ego``: the AVQ quadrotor, which appears nowhere
                     in the F1 training pool at all.
===================  =====================================================

All three checkpoints compared here are Pass A — trained on the *same* drone
pool — so the comparison is architecture-vs-architecture, not recipe-vs-recipe.

Usage in a notebook::

    from generalization_lib import CONDITIONS, MODELS, run, plot_summary, listen
    df, samples = run(n_per_snr=3, snrs=[-15, -5])
    plot_summary(df)
    listen(samples, condition="unseen_drone", snr=-15, index=0)

Everything runs on CPU; keep ``n_per_snr`` small (2–5) if you have no GPU.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

SR = 16000

# condition key -> (derivations.SE_CATEGORY_NOISE category, human label, short blurb)
CONDITIONS: dict[str, tuple[str, str, str]] = {
    "seen": ("drone_seen", "SEEN noise", "the training pool itself"),
    "unseen_rec": ("drone", "UNSEEN recordings", "held-out recordings, same datasets"),
    "unseen_drone": ("avq_ego", "AVQ — absent from training", "AVQ — absent from training"),
    # Part B — the narrow-trained probe. `f2_dcunet_avq_heldout` saw ONLY AVQ
    # session 1, so these two are literally seen/unseen for it, same drone.
    "avq_seen": ("avq_ego_s1", "AVQ session 1", "TRAINED on (probe)"),
    "avq_unseen": ("avq_ego_s2", "AVQ session 2", "never seen (probe)"),
}

# display name -> (experiment/checkpoint name, conf/model yaml)
MODELS: dict[str, tuple[str, str]] = {
    "DCUNet": ("f1_dcunet_a", "conf/model/a1_baseline_dcunet.yaml"),
    "Edge-BS-RoFormer": ("f1_edge_bs_rof_a", "conf/model/a1_edge_bs_rof_fa.yaml"),
    "MP-SENet": ("f1_mpsenet_a", "conf/model/f1_mpsenet.yaml"),
}

# Part B: DCUNet trained on a NARROW pool. Pass-A checkpoints above were trained
# on the broad drone pool, so their "in-distribution" condition is itself broad —
# which is why they show "weak everywhere" rather than "fits in, fails out".
# Only a narrowly-trained model can show the latter, and this is it.
MODELS_NARROW: dict[str, tuple[str, str]] = {
    "DCUNet (AVQ-only)": ("f2_dcunet_avq_survey", "conf/model/f2_dcunet_survey.yaml"),
    "DCUNet (AVQ sess.1)": ("f2_dcunet_avq_heldout", "conf/model/f2_dcunet_survey.yaml"),
}
ALL_MODELS = {**MODELS, **MODELS_NARROW}

COLOURS = {
    "DCUNet": "#9467bd",
    "Edge-BS-RoFormer": "#2ca02c",
    "MP-SENet": "#d62728",
    "DCUNet (AVQ-only)": "#1f77b4",
    "DCUNet (AVQ sess.1)": "#ff7f0e",
}


@dataclass
class Sample:
    """One mixture with its clean target and every model's estimate."""

    condition: str
    snr: float
    index: int
    mixture: np.ndarray
    target: np.ndarray
    estimates: dict[str, np.ndarray]


# ── data ────────────────────────────────────────────────────────────────────
def draw_samples(
    condition: str, *, n_per_snr: int, snrs: list[int], duration_s: float = 2.0, seed: int = 12345
) -> list[tuple[float, int, np.ndarray, np.ndarray]]:
    """Generate mixtures for one condition. Returns (snr, idx, mixture, target)."""
    from data_processing.derivations import (  # noqa: PLC0415
        PARENTS,
        SE_CATEGORY_NOISE,
        SE_HELDOUT_SPEAKERS,
        iter_se_valid_category,
    )

    category = CONDITIONS[condition][0]
    out = []
    for _, frame in iter_se_valid_category(
        category,
        SE_CATEGORY_NOISE[category],
        per_snr=n_per_snr,
        snr_grid=[float(s) for s in snrs],
        duration_s=duration_s,
        sample_rate=SR,
        seed=seed,
        heldout_speakers=SE_HELDOUT_SPEAKERS,
        librispeech=PARENTS["librispeech"],
    ):
        snr = float(frame["meta"]["input_snr"])
        mix = np.asarray(frame["mixture"].data, np.float32).reshape(-1)
        tgt = np.asarray(frame["target"].data, np.float32).reshape(-1)
        out.append((snr, len([o for o in out if o[0] == snr]), mix, tgt))
    return out


# ── models ──────────────────────────────────────────────────────────────────
def load_model(display_name: str, device: str = "cpu") -> tuple[Any, Any]:
    """Load one checkpoint, self-fetching ``best.ckpt`` from R2 when absent."""
    from omegaconf import OmegaConf  # noqa: PLC0415
    from se_eval import fetch_checkpoint  # noqa: PLC0415

    from training.config import build_task_and_codec, instantiate_model  # noqa: PLC0415

    exp, cfg_path = ALL_MODELS[display_name]
    model_cfg = OmegaConf.load(ROOT / cfg_path)
    if device == "cpu":  # flash-attn kernels are CUDA-only
        with contextlib.suppress(Exception):
            model_cfg.params.config.model.flash_attn = False
    _, codec = build_task_and_codec(model_cfg)
    model = instantiate_model(model_cfg).to(device)
    ckpt = fetch_checkpoint(exp, ROOT / "results" / exp / "best.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model, codec


def enhance(model, codec, mixtures: list[np.ndarray], device: str = "cpu") -> list[np.ndarray]:
    batch = torch.stack([torch.as_tensor(m, dtype=torch.float32) for m in mixtures]).to(device)
    with torch.no_grad():
        est = codec.call_model(model, {"mixture": batch})
    est = np.asarray(est.detach().cpu()).reshape(len(mixtures), -1)
    return [est[i] for i in range(est.shape[0])]


# ── metrics ─────────────────────────────────────────────────────────────────
def score(target: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    from metrics.separation import pesq, si_sdr, stoi  # noqa: PLC0415

    n = min(target.shape[0], estimate.shape[0])
    ref, est = target[:n], estimate[:n]
    out = {"si_sdr": float(si_sdr(ref[None, :], est[None, :]))}
    try:
        out["estoi"] = float(stoi(ref, est, SR, extended=True))
    except Exception:
        out["estoi"] = float("nan")
    try:
        out["pesq"] = float(pesq(ref, est, SR))
    except Exception:
        out["pesq"] = float("nan")
    return out


def run(
    *,
    n_per_snr: int = 3,
    snrs: list[int] | None = None,
    models: list[str] | None = None,
    conditions: list[str] | None = None,
    duration_s: float = 2.0,
    device: str = "cpu",
    seed: int = 12345,
) -> tuple[pd.DataFrame, list[Sample]]:
    """Score every model on every condition; return tidy metrics + raw audio.

    The ``noisy`` anchor is included as a model so every number can be read as
    "better or worse than doing nothing" without a second lookup.
    """
    snrs = [-20, -15, -10, -5] if snrs is None else snrs
    models = list(MODELS) if models is None else models
    conditions = list(CONDITIONS) if conditions is None else conditions

    loaded = {}
    for name in models:
        print(f"loading {name} …", flush=True)
        loaded[name] = load_model(name, device=device)

    rows: list[dict] = []
    samples: list[Sample] = []
    for cond in conditions:
        print(f"drawing {cond} …", flush=True)
        drawn = draw_samples(cond, n_per_snr=n_per_snr, snrs=snrs, duration_s=duration_s, seed=seed)
        mixes = [m for _, _, m, _ in drawn]
        ests: dict[str, list[np.ndarray]] = {"noisy": mixes}
        for name in models:
            print(f"  {name} …", flush=True)
            model, codec = loaded[name]
            ests[name] = enhance(model, codec, mixes, device=device)

        for i, (snr, idx, mix, tgt) in enumerate(drawn):
            per_model = {}
            for name, series in ests.items():
                m = score(tgt, series[i])
                rows.append({"condition": cond, "model": name, "snr": snr, "index": idx, **m})
                per_model[name] = series[i]
            samples.append(
                Sample(
                    condition=cond,
                    snr=snr,
                    index=idx,
                    mixture=mix,
                    target=tgt,
                    estimates=per_model,
                )
            )
    return pd.DataFrame(rows), samples


# ── views ───────────────────────────────────────────────────────────────────
def summary_table(df: pd.DataFrame, metric: str = "estoi") -> pd.DataFrame:
    """Mean `metric` per (model, condition), plus the delta against `noisy`."""
    piv = df.pivot_table(index="model", columns="condition", values=metric, aggfunc="mean")
    order = [c for c in CONDITIONS if c in piv.columns]
    piv = piv[order]
    if "noisy" in piv.index:
        delta = piv.subtract(piv.loc["noisy"], axis=1)
        delta.index = [f"{i}  (Δ vs noisy)" for i in delta.index]
        piv = pd.concat([piv, delta.drop(index="noisy  (Δ vs noisy)", errors="ignore")])
    return pd.DataFrame(piv).round(3)


def plot_summary(df: pd.DataFrame, metric: str = "estoi"):
    """One panel per condition; the collapse shows up as a shrinking bar."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    conds = [c for c in CONDITIONS if c in set(df.condition)]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    models = [m for m in ALL_MODELS if m in set(df.model)]
    for ax, cond in zip(axes, conds, strict=True):
        sub = pd.DataFrame(df[df.condition == cond])
        base = pd.DataFrame(sub[sub.model == "noisy"]).groupby("snr")[metric].mean()
        for name in models:
            g = pd.DataFrame(sub[sub.model == name]).groupby("snr")[metric].mean()
            delta = [float(g[s]) - float(base[s]) for s in g.index]
            ax.plot(list(g.index), delta, marker="o", lw=2.5, color=COLOURS.get(name), label=name)
        ax.axhline(0, color="black", lw=1.2)
        label, blurb = CONDITIONS[cond][1], CONDITIONS[cond][2]
        ax.set_title(f"{label}\n{blurb}", fontsize=11)
        ax.set_xlabel("input SNR (dB)")
    axes[0].set_ylabel(f"Δ{metric} vs doing nothing")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    return fig


def listen(samples: list[Sample], *, condition: str, snr: float, index: int = 0):
    """Audio players + spectrograms for one clip across all models."""
    import librosa  # noqa: PLC0415
    import librosa.display  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from IPython.display import Audio, display  # noqa: PLC0415

    hits = [
        s
        for s in samples
        if s.condition == condition and abs(s.snr - snr) < 1e-6 and s.index == index
    ]
    if not hits:
        raise ValueError(f"no sample for {condition} @ {snr} dB index {index}")
    s = hits[0]

    tracks = [("clean target", s.target), ("mixture", s.mixture)]
    tracks += [(k, v) for k, v in s.estimates.items() if k != "noisy"]

    print(f"{CONDITIONS[condition][1]} — {CONDITIONS[condition][2]} — {snr:+.0f} dB input")
    for name, audio in tracks:
        sc = score(s.target, audio)
        print(f"  {name:<20s} SI-SDR {sc['si_sdr']:7.2f} dB   eSTOI {sc['estoi']:.3f}")
        display(Audio(audio, rate=SR))

    fig, axes = plt.subplots(1, len(tracks), figsize=(3.1 * len(tracks), 3.0), sharey=True)
    for ax, (name, audio) in zip(np.atleast_1d(axes), tracks, strict=True):
        d = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=512, hop_length=128)), ref=1.0)
        librosa.display.specshow(d, sr=SR, hop_length=128, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 4000)
    fig.tight_layout()
    return fig
