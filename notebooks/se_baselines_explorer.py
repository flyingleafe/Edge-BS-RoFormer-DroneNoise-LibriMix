"""Interactive per-SNR explorer for the F1 SE blind-baselines.

Backing data: the per-clip CSVs written by ``scripts/se_eval.py`` (one
row per validation clip, carrying ``category`` / ``input_snr`` + SI-SDR / SDR /
PESQ / eSTOI). Because the data is per-clip, ANY subset — a chosen set of noise
categories, a chosen set of models, one valid set — aggregates on the fly to
per-SNR means; nothing is baked in at eval time.

Usage (in a notebook)::

    from se_baselines_explorer import load_perclip, explorer
    df = load_perclip()
    explorer(df)          # ipywidgets UI: pick valid / categories / models / metrics

or headless::

    fig = plot_metrics(df, valid="SE-valid-harmonic",
                       categories=["drone", "motors"],
                       methods=["f1_mpsenet_b", "noisy", "wiener"])
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# repo root = parent of notebooks/
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    ROOT / "results" / "f1_perclip",
    ROOT / "writing" / "reports" / "2026-07-22_se-blind-baselines" / "data" / "perclip",
]

METRICS = ["si_sdr", "estoi", "pesq"]
METRIC_LABEL = {"si_sdr": "SI-SDR (dB)", "estoi": "eSTOI", "pesq": "PESQ", "sdr": "SDR (dB)"}
ARCH_LABEL = {
    "dcunet": "DCUNet",
    "edge_bs_rof": "Edge-BS-RoFormer",
    "mpsenet": "MP-SENet",
    "tfgridnet": "TF-GridNet",
    "sgmse": "SGMSE+",
}
ANCHORS = ("noisy", "wiener")
# anchors drawn as dashed grey/black reference lines, not colored model curves
ANCHOR_STYLE = {"noisy": dict(color="0.4", ls="--"), "wiener": dict(color="0.0", ls=":")}


def _parse_method(method: str) -> tuple[str, str]:
    """method -> (readable label, group) — 'f1_dcunet_a' -> ('DCUNet (A)', 'model')."""
    if method in ANCHORS:
        return method.capitalize(), "anchor"
    body = method[len("f1_") :] if method.startswith("f1_") else method
    passid = ""
    if body.endswith(("_a", "_b")):
        passid = body[-1].upper()
        body = body[:-2]
    label = ARCH_LABEL.get(body, body)
    return (f"{label} ({passid})" if passid else label), "model"


def load_perclip(dirs: list[Path] | None = None) -> pd.DataFrame:
    """Load and concatenate every per-clip CSV found under ``dirs``.

    Adds ``label`` (readable) and ``kind`` ('model'|'anchor') columns. Safe to
    re-run as more eval CSVs land (globs fresh each call)."""
    dirs = dirs or DEFAULT_DIRS
    frames = []
    for d in dirs:
        for path in sorted(glob.glob(str(Path(d) / "*.csv"))):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"no per-clip CSVs under {[str(d) for d in dirs]} — run scripts/se_eval.py "
            "or `omnirun pull f1-perclip-eval-*` first"
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["method", "valid", "clip_id"], keep="last")
    labels = df["method"].map(_parse_method)
    df["label"] = labels.map(lambda t: t[0])
    df["kind"] = labels.map(lambda t: t[1])
    return df


def aggregate(
    df: pd.DataFrame, valid: str, categories: list[str], methods: list[str]
) -> pd.DataFrame:
    """Per-(method, SNR) mean over the chosen valid/categories/methods subset.

    Returns long-format columns: method, label, kind, input_snr, n, <metrics>."""
    sub: pd.DataFrame = df.loc[(df["valid"] == valid) & (df["method"].isin(methods))]
    if categories:
        sub = sub.loc[sub["category"].isin(categories)]
    agg = (
        sub.groupby(["method", "label", "kind", "input_snr"])
        .agg(n=("clip_id", "size"), **{m: (m, "mean") for m in METRICS + ["sdr"]})
        .reset_index()
        .sort_values(["kind", "label", "input_snr"])
    )
    return agg


def plot_metrics(
    df: pd.DataFrame,
    valid: str,
    categories: list[str],
    methods: list[str],
    metrics: list[str] | None = None,
    show_anchors: bool = True,
):
    """3-panel per-SNR plot (one panel per metric) over the chosen subset."""
    metrics = metrics or METRICS
    methods = list(methods)
    if show_anchors:
        methods = methods + [a for a in ANCHORS if a not in methods]
    agg = aggregate(df, valid, categories, methods)
    if agg.empty:
        raise ValueError("no rows for this selection (are the eval CSVs present yet?)")

    models_agg: pd.DataFrame = agg.loc[agg["kind"] == "model"]
    model_labels = sorted(models_agg["label"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {str(lab): cmap(i % 10) for i, lab in enumerate(model_labels)}

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.6), squeeze=False)
    ncats = agg["n"].groupby(agg["input_snr"]).max()
    title_n = int(ncats.max()) if len(ncats) else 0
    for ax, metric in zip(axes[0], metrics):
        for method, g in agg.groupby("method"):
            g = g.sort_values("input_snr")
            label = str(g["label"].iloc[0])
            if g["kind"].iloc[0] == "anchor":
                ax.plot(
                    g["input_snr"],
                    g[metric],
                    label=label,
                    lw=1.6,
                    marker=None,
                    **ANCHOR_STYLE.get(str(method), {}),
                )
            else:
                ax.plot(
                    g["input_snr"],
                    g[metric],
                    label=label,
                    lw=2,
                    marker="o",
                    ms=4,
                    color=colors.get(label),
                )
        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel(METRIC_LABEL.get(metric, metric))
        ax.set_title(METRIC_LABEL.get(metric, metric))
        ax.grid(True, alpha=0.3)
    cats_txt = ", ".join(categories) if categories else "all categories"
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"{valid} · {cats_txt} · ≤{title_n} clips/SNR", fontsize=11)
    fig.tight_layout()
    return fig


def explorer(df: pd.DataFrame | None = None):
    """Build the ipywidgets UI. Returns the container widget (display it)."""
    import ipywidgets as w
    from IPython.display import display

    df = load_perclip() if df is None else df

    models_df: pd.DataFrame = df.loc[df["kind"] == "model"]
    valids = sorted(df["valid"].unique())
    model_methods = sorted(models_df["method"].unique())

    valid_w = w.Dropdown(
        options=valids,
        description="Valid:",
        value="SE-valid-harmonic" if "SE-valid-harmonic" in valids else valids[0],
    )
    cat_w = w.SelectMultiple(description="Categories:", rows=7)
    method_w = w.SelectMultiple(
        options=[(_parse_method(m)[0], m) for m in model_methods],
        value=tuple(model_methods),
        description="Models:",
        rows=min(10, len(model_methods) or 1),
    )
    metric_w = w.SelectMultiple(
        options=METRICS, value=tuple(METRICS), description="Metrics:", rows=3
    )
    anchor_w = w.Checkbox(value=True, description="show anchors (noisy/Wiener)")
    out = w.Output()

    def _refresh_cats(*_):
        vsub: pd.DataFrame = df.loc[df["valid"] == valid_w.value]
        cats = sorted(vsub["category"].unique())
        cat_w.options = cats
        cat_w.value = tuple(cats)  # default: all categories

    def _redraw(*_):
        with out:
            out.clear_output(wait=True)
            try:
                fig = plot_metrics(
                    df,
                    valid_w.value,
                    list(cat_w.value),
                    list(method_w.value),
                    list(metric_w.value),
                    anchor_w.value,
                )
                plt.show()
                plt.close(fig)
            except Exception as e:  # noqa: BLE001 - surface selection errors in-cell
                print(f"[no plot] {e}")

    valid_w.observe(lambda *_a: (_refresh_cats(), _redraw()), names="value")
    for widget in (cat_w, method_w, metric_w, anchor_w):
        widget.observe(_redraw, names="value")

    _refresh_cats()
    _redraw()
    ui = w.VBox([w.HBox([valid_w, anchor_w]), w.HBox([cat_w, method_w, metric_w]), out])
    display(ui)
    return ui
