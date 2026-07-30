"""Pre-training CPU de-risk for the wind-wake channel (design stage W0).

The wind-wake model (``models.generative.wind_wake_gen``) claims the *physics*
decides which microphones sit in a rotor's downwash and therefore carry flow
noise — before any parameter is fit from audio. This script tests that claim
against DREGON's constant-speed single-motor recordings, which are the cleanest
possible probe: exactly one rotor spins, so its geometric wake gate predicts a
per-mic exposure pattern that we can compare to the measured low-frequency
broadband floor (a proxy for incoherent flow noise) across the 8-mic array.

It also runs the design's central *negative* test: Michael's array sits above /
forward of the rotor disk, so the same geometric gate must predict ≈ 0 wind
there — the property that lets one model generalize across arrays with no
per-array tuning.

Everything here is CPU-light: a handful of single-motor clips, capped in length,
Welch PSDs, and a closed-form gate. No training, no GPU. Figures are written to
the scratchpad; a summary table + correlations are printed.

Run (from the worktree root)::

    python scripts/wind_wake_validation.py [--speeds 90 80] [--seconds 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

_ROOT = Path(__file__).resolve().parents[1]
# The `models` namespace resolves to the main checkout via the editable install;
# prepend this worktree's src so we pick up wind_wake_gen from here.
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "notebooks"))

from stage0_rtf_utils import find_dregon_dir, load_motor  # noqa: E402

from data_processing.sources import dregon, michaels  # noqa: E402
from models.generative.wind_wake_gen import wake_flow_speed  # noqa: E402

# Physically-motivated gate parameters (NOT fit — the point is that geometry
# alone predicts exposure). Gate *shape* across mics is rps-independent, so the
# absolute k is irrelevant here.
GATE = dict(k=1.0, alpha=1.0, beta=0.5, gate_softness=0.05)
ROTOR_RADIUS = 0.127  # ~10-inch prop, DREGON MikroKopter scale
DOWNWASH_AXIS = torch.tensor([0.0, 0.0, -1.0])  # rotors thrust up ⇒ downwash −z
FLOOR_BAND = (15.0, 120.0)  # Hz — low-band flow-noise proxy
NPERSEG = 8192


def gate_exposure(mic_pos: np.ndarray, rotor_xyz: np.ndarray) -> np.ndarray:
    """Per-mic geometric wake exposure for a single spinning rotor (module B).

    Returns the length-``M`` flow-speed vector ``U_m`` (m/s) from the closed-form
    gate; only its *shape* across mics matters for the de-risk.
    """
    mic = torch.as_tensor(mic_pos, dtype=torch.float32).unsqueeze(0)  # [1, M, 3]
    rotor = torch.as_tensor(rotor_xyz, dtype=torch.float32).reshape(1, 1, 3)
    axis = DOWNWASH_AXIS.reshape(1, 1, 3)
    rps = torch.full((1, 1, 4), 80.0)  # magnitude irrelevant to the per-mic shape
    u = wake_flow_speed(
        mic,
        rotor,
        axis,
        ROTOR_RADIUS,
        rps,
        k=GATE["k"],
        alpha=GATE["alpha"],
        beta=GATE["beta"],
        gate_softness=GATE["gate_softness"],
    )  # [1, M, 4]
    return u.mean(-1).squeeze(0).numpy()


def inv_distance_exposure(mic_pos: np.ndarray, rotor_xyz: np.ndarray) -> np.ndarray:
    """Control predictor: plain isotropic ``1/r`` acoustic proximity per mic.

    The wake gate and simple proximity both peak near the rotor, so this control
    says how much of the gate's predictive power is *wake-specific* vs merely
    "closer mics are louder". If gate ≈ 1/r, this test cannot separate flow noise
    from near-field acoustics (expected here: all DREGON mics lie below the rotor,
    so the downstream cone barely discriminates).
    """
    d = np.linalg.norm(mic_pos - rotor_xyz[None, :], axis=1)
    return 1.0 / np.clip(d, 1e-3, None)


def measured_floor(audio: np.ndarray, sr: int) -> np.ndarray:
    """Per-mic low-band broadband floor: median PSD in ``FLOOR_BAND`` (dB).

    The median across the band is robust to the sparse tonal harmonics, so it
    tracks the incoherent broadband floor between them — the flow-noise proxy.
    """
    freqs, psd = welch(audio, fs=sr, nperseg=NPERSEG, axis=-1)  # psd [M, F]
    lo, hi = FLOOR_BAND
    band = (freqs >= lo) & (freqs <= hi)
    floor = np.median(psd[:, band], axis=1)  # [M]
    return 10.0 * np.log10(floor + 1e-20)


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_dregon(speeds: list[int], seconds: float):
    dregon_dir = find_dregon_dir()
    mic_pos, rotor_pos = dregon.get_geometry(dregon_dir)  # corrected geometry
    print(f"DREGON geometry (corrected): {mic_pos.shape[0]} mics, {rotor_pos.shape[0]} rotors")
    print(
        f"  mic z-range [{mic_pos[:, 2].min():.3f}, {mic_pos[:, 2].max():.3f}] m; "
        f"rotor z {rotor_pos[:, 2].mean():.3f} m\n"
    )

    rows = []  # (motor, speed, gate[M], floor[M])
    for motor_id in (1, 2, 3, 4):
        rotor_xyz = rotor_pos[motor_id - 1]  # Motor k ↔ rotor row (k−1), per Stage-0
        gate = gate_exposure(mic_pos, rotor_xyz)
        for speed in speeds:
            audio, sr = load_motor(dregon_dir, motor_id, speed, max_seconds=seconds)
            floor = measured_floor(audio, sr)
            rows.append((motor_id, speed, gate, floor))

    # Per-(motor,speed) rank agreement vs the wake gate AND a plain 1/r control.
    print(f"{'motor':>5} {'speed':>5} {'gate_r':>7} {'1/r_r':>6}  gate-argmax vs loudest-floor mic")
    pooled_g, pooled_f, pooled_d = [], [], []
    per_gate, per_dist = [], []
    for motor_id, speed, gate, floor in rows:
        dist = inv_distance_exposure(mic_pos, rotor_pos[motor_id - 1])
        r_gate = _spearman(gate, floor)
        r_dist = _spearman(dist, floor)
        per_gate.append(r_gate)
        per_dist.append(r_dist)
        pooled_g.append(_zscore(gate))
        pooled_f.append(_zscore(floor))
        pooled_d.append(_zscore(dist))
        print(
            f"{motor_id:>5} {speed:>5} {r_gate:>7.3f} {r_dist:>6.3f}  "
            f"mic{int(np.argmax(gate))} vs mic{int(np.argmax(floor))}"
        )

    pg, pf, pd = (np.concatenate(x) for x in (pooled_g, pooled_f, pooled_d))
    pooled_gate = _spearman(pg, pf)
    pooled_dist = _spearman(pd, pf)
    print(
        f"\nPOOLED spearman (n={len(pg)} mic-obs): wake-gate {pooled_gate:.3f}  |  "
        f"1/r control {pooled_dist:.3f}  |  pearson(gate) {_pearson(pg, pf):.3f}"
    )
    print(
        f"mean per-clip spearman: gate {np.mean(per_gate):.3f} ± {np.std(per_gate):.3f}  |  "
        f"1/r {np.mean(per_dist):.3f} ± {np.std(per_dist):.3f}"
    )
    return rows, mic_pos, rotor_pos, (pooled_gate, pooled_dist, float(np.mean(per_gate)))


def run_michaels():
    mic_pos, rotor_pos = michaels.get_geometry()
    print(
        f"\nMichael's geometry: mic z-range "
        f"[{mic_pos[:, 2].min():.3f}, {mic_pos[:, 2].max():.3f}] m; "
        f"rotor z {rotor_pos[:, 2].mean():.3f} m"
    )
    total = np.zeros(mic_pos.shape[0])
    for r in range(rotor_pos.shape[0]):
        total = np.sqrt(total**2 + gate_exposure(mic_pos, rotor_pos[r]) ** 2)
    print(
        f"Michael's predicted per-mic wind exposure (all rotors): "
        f"max {total.max():.4f} m/s, mean {total.mean():.4f} m/s"
    )
    return mic_pos, rotor_pos, total


def make_figure(rows, michaels_gate, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Average gate/floor per motor across speeds for a clean bar view.
    by_motor: dict[int, list] = {}
    for motor_id, _speed, gate, floor in rows:
        by_motor.setdefault(motor_id, []).append((gate, floor))
    motors = sorted(by_motor)

    fig, axes = plt.subplots(2, len(motors), figsize=(3.2 * len(motors), 6), squeeze=False)
    m = rows[0][2].shape[0]
    x = np.arange(m)
    for j, motor_id in enumerate(motors):
        gates = np.mean([g for g, _ in by_motor[motor_id]], axis=0)
        floors = np.mean([f for _, f in by_motor[motor_id]], axis=0)
        ax = axes[0][j]
        ax.bar(x, _zscore(gates), color="#2563a6", alpha=0.85)
        ax.set_title(f"Motor {motor_id}")
        ax.set_ylabel("predicted gate (z)" if j == 0 else "")
        ax.set_xticks(x)
        ax2 = axes[1][j]
        ax2.bar(x, _zscore(floors), color="#a86413", alpha=0.85)
        ax2.set_ylabel("measured floor (z)" if j == 0 else "")
        ax2.set_xlabel("mic")
        ax2.set_xticks(x)
    fig.suptitle(
        "Wind-wake W0 de-risk — geometric gate (top) vs measured low-band floor (bottom)\n"
        f"Michael's max predicted exposure: {michaels_gate.max():.3f} m/s (negative test)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    print(f"\nFigure written: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speeds", type=int, nargs="+", default=[90, 80])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/claude-1000") / "wind_wake_w0_derisk.png"
        if Path("/tmp/claude-1000").exists()
        else _ROOT / "wind_wake_w0_derisk.png",
    )
    args = ap.parse_args()

    print("=" * 72)
    print("WIND-WAKE W0 DE-RISK — does geometry predict where flow noise lands?")
    print("=" * 72)
    rows, _mic, _rotor, stats = run_dregon(args.speeds, args.seconds)
    _, _, michaels_gate = run_michaels()
    make_figure(rows, michaels_gate, args.out)

    pooled_gate, pooled_dist, mean_gate = stats
    print("\n" + "-" * 72)
    if pooled_gate <= 0.4:
        verdict = (
            "WEAK — the geometric gate does not order the low-band floor; the smooth "
            "straight-column gate misses the hub near-field. Revisit before W0 training."
        )
    elif pooled_gate <= pooled_dist + 0.05:
        verdict = (
            "PASS (geometry-driven, proximity-confounded) — the gate strongly predicts "
            "the per-mic floor, but not measurably better than plain 1/r proximity here: "
            "all DREGON mics sit below the disk, so the wake cone barely discriminates. "
            "Wake-vs-acoustic separation needs the incoherence/directionality test (W1). "
            "Sufficient to proceed to W0 transduction fitting."
        )
    else:
        verdict = (
            "PASS (wake-specific) — the gate beats 1/r proximity, so wake geometry adds "
            "predictive power beyond mere closeness. Proceed to W0 training."
        )
    print(f"VERDICT: {verdict}")
    print(
        f"  pooled spearman: gate={pooled_gate:.3f} vs 1/r={pooled_dist:.3f}; "
        f"mean per-clip gate={mean_gate:.3f}"
    )
    print("-" * 72)


if __name__ == "__main__":
    main()
