#!/usr/bin/env python3
"""Generate figures and CSV tables for the RPS-predictor architecture-sweep report.

All numbers are transcribed from the autoresearch session artifacts
(session summarized in docs/experiments/simpleconv-rps-architecture-search.md;
raw session.json/scripts remain under
autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/).

Two training series share the same 26 model keys and the same fixed
DREGON-LM-V4-michaels/valid evaluation set:
  - offline : fixed offline train set, 50 epochs, patience 10
  - online  : online mixture stream, 200 max epochs, patience 50, aug after 50k

Metrics (all on the same fixed validation set):
  pit_mse  PIT-matched MSE (primary, lower better)
  rmse     PIT-matched RMSE
  mae_f    per-frame MAE
  mae_c    per-clip MAE
  r2       coefficient of determination (higher better)
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASELINE = "simple_conv_v2"

# Thematic family for each model key (for grouping / colour).
FAMILY = {
    "simple_conv_v2": "Baseline",
    "simple_conv_v2_transformer": "Temporal head",
    "simple_conv_v2_local_attn": "Temporal head",
    "simple_conv_v2_gru96": "Temporal head",
    "simple_conv_v2_tcn": "Temporal head",
    "simple_conv_v2_multires": "Input features",
    "simple_conv_v2_dwt": "Input features",
    "simple_conv_v2_magphase": "Input features",
    "simple_conv_v2_dual_pool": "Pooling",
    "simple_conv_v2_uni_gru": "Causal / streaming",
    "simple_conv_v2_causal_gru": "Causal / streaming",
    "simple_conv_v2_causal_gru96": "Causal / streaming",
    "simple_conv_v2_uni_gru128": "Causal / streaming",
    "simple_conv_v2_uni_gru128_norm": "Causal / streaming",
    "simple_conv_v2_uni_gru128_norm_do03": "Causal / streaming",
    "simple_conv_v2_uni_gru96_norm_do03": "Causal / streaming",
    "simple_conv_v2_uni_gru96_norm_do02": "Causal / streaming",
    "simple_conv_v2_uni_gru64_norm_do03": "Causal / streaming",
    "simple_conv_v2_causal_tcn": "Causal / streaming",
    "simple_conv_tcn": "SMoLnet / freq-dilated",
    "smolnet_rps_tcn": "SMoLnet / freq-dilated",
    "smolnet_rps_causal_tcn": "SMoLnet / freq-dilated",
    "simple_conv_v2_smol_tcn": "SMoLnet / freq-dilated",
    "simple_conv_v2_smol_causal_tcn": "SMoLnet / freq-dilated",
    "smolnet_rps_simple_head": "SMoLnet / freq-dilated",
    "simple_conv_v2_smol_bigru": "SMoLnet / freq-dilated",
}

# key: (pit_mse, rmse, mae_f, mae_c, r2)
OFFLINE = {
    "simple_conv_v2": (7.8920, 2.81, 2.08, 1.62, 0.8183),
    "simple_conv_v2_transformer": (43.5184, 6.60, 5.03, 4.58, -0.6571),
    "simple_conv_v2_local_attn": (18.5846, 4.31, 3.25, 2.71, 0.5213),
    "simple_conv_v2_multires": (8.9704, 3.00, 2.17, 1.62, 0.8088),
    "simple_conv_v2_dwt": (8.8957, 2.98, 2.12, 1.70, 0.8133),
    "simple_conv_v2_magphase": (10.4266, 3.23, 2.39, 1.85, 0.7466),
    "simple_conv_v2_dual_pool": (9.8217, 3.13, 2.37, 1.93, 0.7462),
    "simple_conv_v2_gru96": (8.6612, 2.94, 2.07, 1.62, 0.8216),
    "simple_conv_v2_uni_gru": (228.6723, 15.12, 9.82, 9.13, -10.4445),
    "simple_conv_v2_causal_gru": (83.5143, 9.14, 5.50, 5.08, -2.8866),
    "simple_conv_v2_causal_gru96": (253.5939, 15.92, 8.14, 7.73, -12.6410),
    "simple_conv_v2_uni_gru128": (39.8099, 6.31, 4.12, 3.80, -0.5486),
    "simple_conv_v2_uni_gru128_norm": (20.2943, 4.50, 2.34, 1.81, 0.7391),
    "simple_conv_v2_uni_gru128_norm_do03": (218.0722, 14.77, 7.49, 7.22, -11.7088),
    "simple_conv_v2_uni_gru96_norm_do03": (13.1309, 3.62, 2.64, 2.12, 0.7340),
    "simple_conv_v2_uni_gru96_norm_do02": (65.4811, 8.09, 4.43, 3.88, -1.2644),
    "simple_conv_v2_uni_gru64_norm_do03": (95.4777, 9.77, 4.96, 4.60, -3.7860),
    "simple_conv_tcn": (24.5623, 4.96, 3.35, 2.68, 0.3952),
    "simple_conv_v2_tcn": (10.7799, 3.28, 2.22, 1.72, 0.7606),
    "simple_conv_v2_causal_tcn": (14.1444, 3.76, 2.70, 2.19, 0.6025),
    "smolnet_rps_tcn": (17.4362, 4.18, 2.95, 2.35, 0.4048),
    "smolnet_rps_causal_tcn": (49.6064, 7.04, 5.97, 4.86, -0.3305),
    "simple_conv_v2_smol_tcn": (9.0751, 3.01, 1.89, 1.31, 0.8318),
    "simple_conv_v2_smol_causal_tcn": (8.3806, 2.89, 1.93, 1.46, 0.8331),
    "smolnet_rps_simple_head": (141.0523, 11.88, 10.03, 9.58, -3.8919),
    "simple_conv_v2_smol_bigru": (11.3410, 3.37, 2.34, 1.73, 0.7461),
}

# key: (pit_mse, rmse, mae_f, mae_c, r2, best_epoch, timed_out)
ONLINE = {
    "simple_conv_v2": (8.5349, 2.92, 2.00, 1.56, 0.8332, 42, False),
    "simple_conv_v2_transformer": (8.4629, 2.91, 2.16, 1.68, 0.8085, 15, False),
    "simple_conv_v2_local_attn": (10.0549, 3.17, 2.37, 1.86, 0.7637, 17, False),
    "simple_conv_v2_multires": (8.7521, 2.96, 2.27, 1.87, 0.7710, 57, False),
    "simple_conv_v2_dwt": (8.8512, 2.98, 2.32, 1.87, 0.7828, 13, False),
    "simple_conv_v2_magphase": (8.1824, 2.86, 2.06, 1.54, 0.8348, 37, False),
    "simple_conv_v2_dual_pool": (8.4940, 2.91, 2.29, 1.80, 0.7888, 16, False),
    "simple_conv_v2_gru96": (10.7942, 3.29, 2.51, 2.12, 0.7886, 6, False),
    "simple_conv_v2_uni_gru": (8.7301, 2.95, 2.28, 1.88, 0.8030, 52, False),
    "simple_conv_v2_causal_gru": (14.6395, 3.83, 2.48, 2.04, 0.7703, 25, False),
    "simple_conv_v2_causal_gru96": (11.4611, 3.39, 2.50, 2.03, 0.7657, 15, False),
    "simple_conv_v2_uni_gru128": (7.3264, 2.71, 2.04, 1.55, 0.8224, 17, False),
    "simple_conv_v2_uni_gru128_norm": (7.9864, 2.83, 2.11, 1.62, 0.8024, 25, False),
    "simple_conv_v2_uni_gru128_norm_do03": (8.2826, 2.88, 2.15, 1.69, 0.8057, 17, False),
    "simple_conv_v2_uni_gru96_norm_do03": (8.3325, 2.89, 2.26, 1.87, 0.8059, 25, False),
    "simple_conv_v2_uni_gru96_norm_do02": (9.3224, 3.05, 2.31, 1.88, 0.7946, 21, False),
    "simple_conv_v2_uni_gru64_norm_do03": (88.8055, 9.42, 7.60, 7.27, -1.8765, 10, False),
    "simple_conv_tcn": (14.1221, 3.76, 2.77, 2.17, 0.6832, 15, False),
    "simple_conv_v2_tcn": (12.4689, 3.53, 2.80, 2.34, 0.6924, 21, False),
    "simple_conv_v2_causal_tcn": (11.3739, 3.37, 2.35, 1.73, 0.7458, 26, False),
    "smolnet_rps_tcn": (11.8746, None, 2.60, 1.99, 0.7270, 48, True),
    "smolnet_rps_causal_tcn": (12.9773, 3.60, 2.59, 1.94, 0.6755, 50, False),
    "simple_conv_v2_smol_tcn": (12.6057, 3.55, 2.49, 1.83, 0.6863, 38, False),
    "simple_conv_v2_smol_causal_tcn": (8.9874, 3.00, 2.02, 1.56, 0.8237, 63, False),
    "smolnet_rps_simple_head": (13.7097, 3.70, 2.51, 1.92, 0.6900, 43, False),
    "simple_conv_v2_smol_bigru": (9.5475, 3.09, 2.38, 2.01, 0.8052, 10, False),
}

FAMILY_COLOURS = {
    "Baseline": "#d62728",
    "Temporal head": "#1f77b4",
    "Input features": "#2ca02c",
    "Pooling": "#9467bd",
    "Causal / streaming": "#ff7f0e",
    "SMoLnet / freq-dilated": "#8c564b",
}

UNI_GRU_KEYS = [
    "simple_conv_v2_uni_gru",
    "simple_conv_v2_uni_gru128",
    "simple_conv_v2_uni_gru128_norm",
    "simple_conv_v2_uni_gru128_norm_do03",
    "simple_conv_v2_uni_gru96_norm_do03",
    "simple_conv_v2_uni_gru96_norm_do02",
    "simple_conv_v2_uni_gru64_norm_do03",
]

# Follow-up rerun: grad_clip=0.5, 200 epochs, patience 50.
# key: (pit_mse, rmse, mae_f, mae_c, r2, best_epoch, nan_observed)
CLIPPED_OFFLINE = {
    "simple_conv_v2_uni_gru128": (10.4019, 3.23, 2.03, 1.56, 0.7818, 25, False),
    "simple_conv_v2_uni_gru96_norm_do03": (18.9526, 4.35, 2.54, 1.82, 0.6640, 21, True),
    "simple_conv_v2_uni_gru128_norm_do03": (178.0662, 13.34, 6.93, 6.64, -9.2014, 3, False),
    "simple_conv_v2_uni_gru96_norm_do02": (194.3177, 13.94, 6.94, 6.59, -10.2839, 4, True),
    "simple_conv_v2_uni_gru64_norm_do03": (212.2095, 14.57, 7.77, 7.36, -10.5720, 5, True),
    "simple_conv_v2_uni_gru128_norm": (228.8797, 15.13, 9.93, 9.23, -10.4075, 2, True),
    "simple_conv_v2_uni_gru": (573.9627, 23.96, 23.29, 22.97, -15.7796, 3, True),
}

CLIPPED_ONLINE = {
    "simple_conv_v2_uni_gru": (8.5433, 2.92, 2.17, 1.72, 0.8018, 69, False),
    "simple_conv_v2_uni_gru96_norm_do02": (8.8912, 2.98, 2.28, 1.82, 0.7822, 16, False),
    "simple_conv_v2_uni_gru128": (9.1313, 3.02, 2.33, 1.87, 0.7674, 25, False),
    "simple_conv_v2_uni_gru96_norm_do03": (9.1714, 3.03, 2.32, 1.80, 0.7668, 20, True),
    "simple_conv_v2_uni_gru128_norm_do03": (11.4313, 3.38, 2.46, 1.98, 0.7532, 20, True),
    "simple_conv_v2_uni_gru64_norm_do03": (17.0005, 4.12, 2.68, 2.24, 0.7084, 15, True),
    "simple_conv_v2_uni_gru128_norm": (36.1067, 6.01, 3.41, 2.84, -0.2631, 9, True),
}


def _fmt(v):
    return "—" if v is None else f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"


def write_csv(path, series, ranked):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#", "Model", "PIT MSE", "RMSE", "MAE/f", "MAE/c", "R2"])
        for rank, key in enumerate(ranked, 1):
            pit, rmse, maef, maec, r2 = series[key][:5]
            star = "*" if (len(series[key]) > 6 and series[key][6]) else ""
            w.writerow(
                [
                    rank,
                    key.replace("simple_conv_v2", "scv2").replace("smolnet_rps", "smol"),
                    f"{pit:.4f}{star}",
                    _fmt(rmse) if rmse is not None else "—",
                    f"{maef:.2f}",
                    f"{maec:.2f}",
                    f"{r2:.4f}",
                ]
            )


def barh_leaderboard(path, series, title):
    ranked = sorted(
        series, key=lambda k: series[k][0], reverse=True
    )  # worst at bottom -> best at top
    vals = [series[k][0] for k in ranked]
    colours = [FAMILY_COLOURS[FAMILY[k]] for k in ranked]
    labels = [k.replace("simple_conv_v2", "scv2").replace("smolnet_rps", "smol") for k in ranked]

    fig, ax = plt.subplots(figsize=(9, 9))
    y = range(len(ranked))
    ax.barh(list(y), vals, color=colours, edgecolor="black", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(min(vals) * 0.85, max(vals) * 2.2)  # headroom for value labels
    ax.set_xlabel("PIT MSE (log scale, lower is better)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    baseline_val = series[BASELINE][0]
    ax.axvline(
        baseline_val,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"baseline = {baseline_val:.2f}",
    )
    for yi, v in zip(y, vals):
        ax.text(v * 1.03, yi, f"{v:.2f}", va="center", fontsize=8)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOURS.values()]
    ax.legend(
        handles + [ax.lines[0]],
        list(FAMILY_COLOURS) + [f"baseline = {baseline_val:.2f}"],
        fontsize=8,
        loc="lower right",
    )
    ax.grid(axis="x", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def offline_vs_online(path):
    fig, ax = plt.subplots(figsize=(8, 8))
    keys = [k for k in OFFLINE if k in ONLINE]
    for k in keys:
        xo = OFFLINE[k][0]
        yo = ONLINE[k][0]
        ax.scatter(
            xo,
            yo,
            color=FAMILY_COLOURS[FAMILY[k]],
            s=45,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    lim_lo, lim_hi = 6, 300
    ax.plot(
        [lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, alpha=0.6, label="offline = online"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Offline PIT MSE (log)", fontsize=11)
    ax.set_ylabel("Online-mix PIT MSE (log)", fontsize=11)
    ax.set_title(
        "Same model, offline vs. online-mixed training\n(points below the diagonal improved with online mixing)",
        fontsize=12,
        fontweight="bold",
    )
    # annotate the most extreme movers
    for k in [
        "simple_conv_v2_uni_gru",
        "simple_conv_v2_uni_gru128",
        "simple_conv_v2_uni_gru128_norm_do03",
        "smolnet_rps_simple_head",
        "simple_conv_v2_magphase",
    ]:
        ax.annotate(
            k.replace("simple_conv_v2", "scv2").replace("smolnet_rps", "smol"),
            (OFFLINE[k][0], ONLINE[k][0]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOURS.values()]
    ax.legend(handles, list(FAMILY_COLOURS), fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pit_vs_r2(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, (series, name) in zip(axes, [(OFFLINE, "Offline"), (ONLINE, "Online-mix")]):
        for k in series:
            ax.scatter(
                series[k][0],
                series[k][4],
                color=FAMILY_COLOURS[FAMILY[k]],
                s=45,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("PIT MSE (log, lower better)", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("R² (higher better)", fontsize=11)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOURS.values()]
    axes[1].legend(handles, list(FAMILY_COLOURS), fontsize=8, loc="lower left")
    fig.suptitle("PIT MSE vs. R² across all 26 models", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_clipped_csv(path, series):
    ranked = sorted(series, key=lambda k: series[k][0])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#", "Model", "NaN?", "PIT MSE", "RMSE", "MAE/f", "MAE/c", "R2", "Best epoch"])
        for rank, key in enumerate(ranked, 1):
            pit, rmse, maef, maec, r2, best_epoch, nan_observed = series[key]
            label = key.replace("simple_conv_v2", "scv2")
            w.writerow(
                [
                    rank,
                    label,
                    "yes" if nan_observed else "no",
                    f"{pit:.4f}",
                    f"{rmse:.2f}",
                    f"{maef:.2f}",
                    f"{maec:.2f}",
                    f"{r2:.4f}",
                    best_epoch,
                ]
            )


def clipped_uni_gru_comparison(path):
    labels = [k.replace("simple_conv_v2", "scv2") for k in UNI_GRU_KEYS]
    series_defs = [
        ("Offline 50ep", OFFLINE, "#9ecae1", None),
        ("Offline + clip", CLIPPED_OFFLINE, "#1f77b4", CLIPPED_OFFLINE),
        ("Online", ONLINE, "#fdd0a2", None),
        ("Online + clip", CLIPPED_ONLINE, "#ff7f0e", CLIPPED_ONLINE),
    ]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = list(range(len(UNI_GRU_KEYS)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (name, data, colour, nan_source), off in zip(series_defs, offsets):
        vals = [data[k][0] for k in UNI_GRU_KEYS]
        bars = ax.bar(
            [xi + off for xi in x],
            vals,
            width=width,
            label=name,
            color=colour,
            edgecolor="black",
            linewidth=0.4,
        )
        if nan_source is not None:
            for bar, key in zip(bars, UNI_GRU_KEYS):
                if nan_source[key][6]:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.08,
                        "NaN",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

    ax.axhline(
        OFFLINE[BASELINE][0], color="#d62728", linestyle="--", linewidth=1, label="offline baseline"
    )
    ax.axhline(
        ONLINE[BASELINE][0], color="#d62728", linestyle=":", linewidth=1, label="online baseline"
    )
    ax.set_yscale("log")
    ax.set_ylabel("PIT MSE (log, lower better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Unidirectional-GRU follow-up: grad_clip=0.5 does not reproduce the online winner")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    assets = pathlib.Path("assets")
    assets.mkdir(exist_ok=True)

    offline_ranked = sorted(OFFLINE, key=lambda k: OFFLINE[k][0])
    online_ranked = sorted(ONLINE, key=lambda k: ONLINE[k][0])

    write_csv(assets / "offline.csv", OFFLINE, offline_ranked)
    write_csv(assets / "online.csv", ONLINE, online_ranked)
    write_clipped_csv(assets / "clipped_offline.csv", CLIPPED_OFFLINE)
    write_clipped_csv(assets / "clipped_online.csv", CLIPPED_ONLINE)

    barh_leaderboard(
        assets / "fig_offline_leaderboard.png",
        OFFLINE,
        "Offline fixed-train sweep — validation PIT MSE",
    )
    barh_leaderboard(
        assets / "fig_online_leaderboard.png", ONLINE, "Online-mixed rerun — validation PIT MSE"
    )
    offline_vs_online(assets / "fig_offline_vs_online.png")
    pit_vs_r2(assets / "fig_pit_vs_r2.png")
    clipped_uni_gru_comparison(assets / "fig_clipped_uni_gru.png")
    print("Wrote figures and CSVs to", assets.resolve())


if __name__ == "__main__":
    main()
