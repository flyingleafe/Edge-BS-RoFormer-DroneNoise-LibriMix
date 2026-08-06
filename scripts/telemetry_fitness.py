#!/usr/bin/env python3
"""Goodness of fit of candidate rotor-speed trajectories (GitHub issue 17, §A-D).

A thin ``utils.gridrun`` driver over ``tracking.fitness``. One unit is one
(window, candidate, control); every hold-out, the bootstrap and the residual
decomposition are re-aggregations of that unit's single demodulation, so they
ride along inside it.

The candidate spec is a small language, because "which trajectory" is the whole
question:

  telemetry        the window's raw ``motors_measured``, unmodified
  scale:0.99458    telemetry times a constant rate scale
  lp:5             telemetry low-passed at 5 Hz (the pre-smoothing of issue 17
                   step 1 — the 0.269 rev/s / 49.7 Hz staircase is measurement
                   noise, not signal)
  lp:5+scale:0.996 specs compose with ``+``, left to right — the pre-smoothed
                   carrier under a pure global scale. A profile over that one
                   parameter is the campaign's fixed-DOF scale estimator
  file:PATH:KEY    an ``.npz`` of fitted trajectories on the window's frame grid
                   (the phase 6b hook). ``{key}`` inside PATH is replaced by the
                   window key, so ONE spec scores a whole directory of per-window
                   fits — which is exactly what ``scripts/telemetry_refit.py``
                   writes (``<out>/traj/<arm>/<window>.npz``, entry ``r_fit``).

The four controls of §B all run from the same flag: ``on`` is the measurement,
``offcomb`` the half-integer null, ``mismatch`` a partner window's telemetry on
this window's audio, ``permute`` the candidate's rotor rows rolled by one. The
fourth control of §B is FLY124, whose labels were recalibrated and read
-0.063 %: it is ``--dataset fly124``, one flag, the identical procedure.

Examples:
  python scripts/telemetry_fitness.py --smoke --jobs 6
  python scripts/telemetry_fitness.py --dataset fly124 --candidates telemetry,lp:5
  python scripts/telemetry_fitness.py --windows free-flight_nosource_room1__w01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]  # this checkout (code)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # beatvk_eval, for --build-preps

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402
from utils.paths import get_data_root  # noqa: E402

#: The pulled local prep cache of the beat-VK band-admission job.
PULLED_PREP = get_data_root() / (
    "omnirun-outputs/bandadm-ladder-7fb2e4/results/beatvk_bandadm/vk_arms/prep_cache"
)
#: Where ``--build-preps`` writes when there is no pulled cache — a fresh
#: cluster checkout has neither, and the cache is gitignored.
BUILT_PREP = Path("results/telemetry_prep")
#: The frozen protocol's dataset pin (the manifest of the pulled cache), so a
#: rebuilt window is the same window and not merely a window of the same name.
PREP_PIN = "54849c13ed3a29fa3516503ac291539b3fb2004b22b8b0a549827d0a60c06b86"
OUT_DEFAULT = "results/telemetry_fitness"
CONTROLS = ("on", "offcomb", "mismatch", "permute")
#: The control name the library knows ``on`` by.
LIB_CONTROL = {"on": "none", "offcomb": "offcomb", "mismatch": "mismatch", "permute": "permute"}

DREGON_WINDOWS = tuple(
    f"{rec}__w{i:02d}"
    for rec in (
        "free-flight_nosource_room1",
        "free-flight_speech-low_room1",
        "free-flight_whitenoise-low_room1",
    )
    for i in range(3)
)
FLY124_WINDOWS = tuple(f"FLY124__w{i:02d}" for i in range(6))
ALL_WINDOWS = DREGON_WINDOWS + FLY124_WINDOWS

#: The correspondence-breaking null: telemetry of a DIFFERENT window of the
#: SAME recording, so the spectra are real and only the correspondence is wrong.
PARTNER: dict[str, str] = {}
for _key in ALL_WINDOWS:
    _rec, _idx = _key.split("__w")
    _n = 6 if _rec == "FLY124" else 3
    PARTNER[_key] = f"{_rec}__w{(int(_idx) + 1) % _n:02d}"

#: The frozen DREGON window of the displacement campaign, and the first FLY124
#: cruise window — the smoke pair.
SMOKE_WINDOWS = ("free-flight_nosource_room1__w01", "FLY124__w02")
SMOKE_CANDIDATES = ("telemetry", "scale:0.99458", "lp:5")

PROTOCOL = {
    "dataset": "beatvk-valid-raw@54849c13ed3a",
    "statistic": "four components at FIXED degrees of freedom: broadband "
    "(out-of-DC share of demodulated envelope power), phase noise (k^2-weighted "
    "mean square of the per-harmonic rate opinion about zero, per mic), "
    "magnitude roughness (high-pass share of |z_k| power), and RIDGE "
    "concentration (phase 6d: dB of DC line density over the local floor, "
    "read on its own gate — the only component where more is better)",
    "band_hz_k": "min(b0 k, 0.45 rate_ref) Hz, rate_ref pinned to the window's "
    "telemetry so the band never follows the candidate",
    "admission": "conditioning gate — no other rotor's real line within "
    "gate_band_frac * band + guard for >= 90 % of a block's frames; derived "
    "from the REFERENCE only, so every candidate and control scores the same "
    "cells. Report admit_frac with every number: at b0 = 1 rev/s a DREGON twin "
    "pair (0.42 rev/s apart) collides at every harmonic",
    "controls": {
        "on": "the measurement",
        "offcomb": "half-integer comb (k + 0.5) g(t): no rotor line can exist",
        "mismatch": "telemetry of another window of the same recording",
        "permute": "candidate rotor rows rolled by one — a null of the RESIDUAL "
        "pairing; the acoustic components are permutation-invariant by "
        "construction (see tracking/fitness.py)",
    },
    "holdouts": "none / fit even k / fit odd k / fit mic 0 / fit half the blocks",
}


# ---------------------------------------------------------------------------
# data + candidates


def resolve_prep_dir() -> Path:
    """Where the frozen protocol windows live, in order of preference.

    ``TELEMETRY_PREP_DIR`` wins, then the pulled band-admission cache (this
    laptop), then the ``--build-preps`` output. A cluster worktree has only the
    last one, because the cache is a gitignored artifact.
    """
    env = os.environ.get("TELEMETRY_PREP_DIR")
    if env:
        return Path(env)
    return PULLED_PREP if PULLED_PREP.exists() else BUILT_PREP


def build_preps(keys: list[str], dst: Path, version: str | None = PREP_PIN) -> None:
    """Materialize missing protocol windows from the pinned dataset.

    The same slice as ``scripts/beatvk_vk_arms.build_preps`` — protocol resample
    to 16 kHz, ``tracking.protocols.slice_window`` — minus the rotor/mic weight
    file, which needs the raw DREGON coordinates and which nothing in this
    campaign reads. REQUIRED on a cluster: without it every unit dies on
    ``FileNotFoundError``, and gridrun turns a unit exception into a ``.err``
    file rather than a visible failure.
    """
    import numpy as np

    want: dict[str, set[int]] = {}
    for key in keys:
        rid, _, idx = key.partition("__w")
        want.setdefault(rid, set()).add(int(idx))
    missing = {
        rid: sorted(i for i in idxs if not (dst / f"{rid}__w{i:02d}.npz").exists())
        for rid, idxs in want.items()
    }
    missing = {rid: idxs for rid, idxs in missing.items() if idxs}
    if not missing:
        print(f"[prep] {dst}: complete", flush=True)
        return

    from beatvk_eval import SR, load_recordings

    from data_processing.frames import resample_audio_series
    from tracking.protocols import BEATVK, iter_windows, slice_window

    dst.mkdir(parents=True, exist_ok=True)
    for rec in load_recordings(version, set(missing), keep_audio=True):
        rid = rec["recording_id"]
        specs = {s.index: s for s in iter_windows(BEATVK, {rid: {"windows": rec["windows"]}})}
        audio16 = np.atleast_2d(
            np.asarray(resample_audio_series(rec["audio"], SR).data, dtype=np.float32)
        )
        for widx in missing[rid]:
            spec = specs[widx]
            seg, ft, r_meas, edge = slice_window(audio16, SR, spec, rec["ts"], rec["vals"])
            assert r_meas is not None
            np.savez(
                dst / f"{rid}__w{widx:02d}.npz",
                allow_pickle=False,
                start_s=np.float64(spec.start_s),
                end_s=np.float64(spec.end_s),
                regime=np.str_(spec.regime),
                audio=seg,
                ft=ft,
                r_meas=r_meas,
                edge=edge,
            )
        rec["audio"] = None
        print(f"[prep] {rid}: {len(missing[rid])} windows -> {dst}", flush=True)


def prep_sha1(key: str) -> str:
    """Fingerprint of a window's audio + telemetry.

    A rebuilt cache must be the SAME window as the pulled one; this travels in
    every unit's JSON so the comparison is a diff of two numbers, not a claim.
    """
    import hashlib

    import numpy as np

    with np.load(resolve_prep_dir() / f"{key}.npz") as z:
        h = hashlib.sha1()
        for name in ("audio", "ft", "r_meas"):
            h.update(np.ascontiguousarray(z[name]).tobytes())
    return h.hexdigest()[:12]


def _load(key: str) -> dict[str, Any]:
    """The frozen prep-cache window: audio, frame grid, telemetry, regime."""
    import numpy as np

    with np.load(resolve_prep_dir() / f"{key}.npz") as z:
        return {
            "audio": np.asarray(z["audio"], np.float64),
            "ft": np.asarray(z["ft"], np.float64),
            "r": np.asarray(z["r_meas"], np.float64),
            "regime": str(z["regime"]),
        }


def build_candidate(spec: str, r: Any, ft: Any, key: str = "") -> Any:
    """Materialize one candidate trajectory from its spec string.

    ``{key}`` is substituted with the window key first, so a per-window family
    of fitted trajectories is ONE spec (see the ``file:`` line of the module
    docstring). The substitution happens before parsing, so it never changes
    what a spec without the placeholder means.

    Specs compose with ``+``, applied left to right: ``lp:5+scale:0.996`` is the
    pre-smoothed carrier under a pure global rate scale. That family is the
    campaign's ONE-parameter scale estimator — a profile over ``s`` has a single
    degree of freedom, so unlike a fitted trajectory it cannot buy a better
    score with flexibility.
    """
    import numpy as np

    from tracking.telemetry_refit import presmooth

    spec = spec.replace("{key}", key)
    if "+" in spec:
        out = r
        for part in spec.split("+"):
            out = build_candidate(part, out, ft, key)
        return out
    if spec == "telemetry":
        return r
    kind, _, rest = spec.partition(":")
    if kind == "scale":
        return r * float(rest)
    if kind == "lp":
        return presmooth(r, ft, float(rest))
    if kind == "file":
        path, _, entry = rest.rpartition(":")
        with np.load(path) as z:
            return np.asarray(z[entry], dtype=np.float64)
    raise ValueError(f"unknown candidate spec {spec!r}")


def _uid(spec: str) -> str:
    """A candidate spec as a filename-safe unit id fragment."""
    out = spec
    for ch in ":/{}":
        out = out.replace(ch, "-")
    return out


def worker(unit: Unit) -> dict[str, Any]:
    """One (window, candidate, control) unit."""
    import numpy as np

    from tracking.fitness import FitnessConfig, score_window

    p = unit.params
    key, spec, control = str(p["key"]), str(p["candidate"]), str(p["control"])
    win = _load(key)
    cfg = FitnessConfig(
        k_min=int(p["k_min"]),
        k_max=int(p["k_max"]),
        b0_revs=float(p["b0"]),
        fs_env=float(p["fs_env"]),
        n_blocks=int(p["n_blocks"]),
        gate_band_frac=float(p.get("gate_band_frac", 1.0)),
    )
    cand = build_candidate(spec, win["r"], win["ft"], key)
    partner = None
    if control == "mismatch":
        other = _load(PARTNER[key])
        n = win["ft"].size
        partner = np.stack(
            [
                np.interp(win["ft"], other["ft"][: other["r"].shape[1]], other["r"][i])[:n]
                for i in range(other["r"].shape[0])
            ]
        )
    out = score_window(
        win["audio"],
        win["ft"],
        cand,
        win["r"],
        cfg=cfg,
        control=LIB_CONTROL[control],
        partner=partner,
        n_boot=int(p["boot"]),
        seed=int(p["seed"]),
    )
    # The identity fields go LAST: ``score_window`` also reports a ``control``
    # (the library's name for it), and the unit's own name must win.
    return {
        **out,
        "key": key,
        "recording": key.split("__")[0],
        "regime": win["regime"],
        "candidate": spec,
        "control": control,
        "lib_control": LIB_CONTROL[control],
        "rotor_mean_rev_s": [round(float(v), 3) for v in win["r"].mean(axis=1)],
    }


# ---------------------------------------------------------------------------
# report


#: ``ridge`` is the phase-6d component and the only one where MORE is better
#: (``tracking.fitness.HIGHER_IS_BETTER``).
COMPONENTS = ("broadband", "phase_noise", "roughness", "ridge", "pp_dr")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled components per (dataset, candidate, control, hold-out)."""
    pooled: dict[str, Any] = {}
    for name, is_fly in (("dregon", False), ("fly124", True)):
        sel = [r for r in rows if r["key"].startswith("FLY124") == is_fly and "scores" in r]
        if not sel:
            continue
        block: dict[str, Any] = {}
        for cand in sorted({r["candidate"] for r in sel}):
            for ctl in CONTROLS:
                got = [r for r in sel if r["candidate"] == cand and r["control"] == ctl]
                if not got:
                    continue
                per_ho: dict[str, Any] = {}
                for ho in sorted({h for r in got for h in r["scores"]}):
                    vals = [r["scores"][ho] for r in got if ho in r["scores"]]
                    per_ho[ho] = {c: _mean([v.get(c) for v in vals]) for c in COMPONENTS}
                res = [r["residual"]["pooled"] for r in got if r.get("residual", {}).get("pooled")]
                block[f"{cand}|{ctl}"] = {
                    "n_windows": len(got),
                    "admit_frac": _mean([r["cells"]["admit_frac"] for r in got]),
                    "admit_frac_ridge": _mean([r["cells"].get("admit_frac_ridge") for r in got]),
                    "line_share_gated": _mean([r["cells"].get("line_share_gated") for r in got]),
                    "line_share_ridge": _mean([r["cells"].get("line_share_ridge") for r in got]),
                    "n_cells": _mean(
                        [r["scores"]["none"]["n_cells"] for r in got if "none" in r["scores"]]
                    ),
                    "holdouts": per_ho,
                    "residual": {
                        k: _mean([d.get(k) for d in res])
                        for k in (
                            "scale_pct",
                            "lag_s",
                            "d_rms",
                            "resid_rms",
                            "tach_bound_frac",
                            "tach_flatness",
                        )
                    }
                    if res
                    else None,
                }
        pooled[name] = block
    return {"protocol": PROTOCOL, "pooled": pooled}


def _mean(vals: list[Any]) -> float | None:
    import numpy as np

    v = np.asarray([x for x in vals if isinstance(x, (int, float))], dtype=np.float64)
    v = v[np.isfinite(v)]
    return round(float(v.mean()), 6) if v.size else None


def print_table(summary: dict[str, Any]) -> None:
    """The reading order: components on the left, its own null beside it."""
    for ds, block in summary.get("pooled", {}).items():
        print(f"\n=== {ds} ===")
        head = f"{'candidate':16s} {'control':9s} {'holdout':12s}"
        head += "".join(f"{c:>12s}" for c in COMPONENTS)
        print(head)
        for tag in sorted(block):
            cand, ctl = tag.split("|")
            for ho, vals in block[tag]["holdouts"].items():
                row = f"{cand:16s} {ctl:9s} {ho:12s}"
                row += "".join(
                    f"{vals[c]:12.5f}" if vals[c] is not None else f"{'—':>12s}" for c in COMPONENTS
                )
                print(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="all", choices=("dregon", "fly124", "all"))
    ap.add_argument("--windows", default="", help="comma-separated keys (overrides --dataset)")
    ap.add_argument("--candidates", default=",".join(SMOKE_CANDIDATES))
    ap.add_argument("--controls", default=",".join(CONTROLS))
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--b0", type=float, default=1.0, help="k-scaled band, rev/s of capture")
    ap.add_argument("--fs-env", type=float, default=250.0)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument(
        "--gate-band-frac",
        type=float,
        default=1.0,
        help="conditioning gate: an interferer nearer than this times the band "
        "(plus the guard) gates the cell out; below 1.0 trades purity for coverage",
    )
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="the two smoke windows only")
    ap.add_argument(
        "--build-preps",
        action="store_true",
        help="materialize the protocol windows first (REQUIRED on a cluster: the "
        "prep cache is a gitignored artifact, so a fresh worktree has no windows)",
    )
    ap.add_argument("--out", default=OUT_DEFAULT)
    add_gridrun_args(ap, jobs=6)
    args = ap.parse_args()

    if args.smoke:
        keys = list(SMOKE_WINDOWS)
    elif args.windows:
        keys = [k.strip() for k in args.windows.split(",") if k.strip()]
    else:
        keys = list(
            {"dregon": DREGON_WINDOWS, "fly124": FLY124_WINDOWS, "all": ALL_WINDOWS}[args.dataset]
        )
    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    ctls = [c.strip() for c in args.controls.split(",") if c.strip()]
    for bad in [k for k in keys if k not in ALL_WINDOWS]:
        ap.error(f"unknown window {bad!r}; known: {', '.join(ALL_WINDOWS)}")
    for bad in [c for c in ctls if c not in CONTROLS]:
        ap.error(f"unknown control {bad!r}; known: {', '.join(CONTROLS)}")
    if args.build_preps:
        wanted = sorted({k for key in keys for k in (key, PARTNER[key])})  # mismatch partners too
        build_preps(wanted, resolve_prep_dir())

    common = {
        "k_min": args.k_min,
        "k_max": args.k_max,
        "b0": args.b0,
        "fs_env": args.fs_env,
        "n_blocks": args.n_blocks,
        "gate_band_frac": args.gate_band_frac,
        "boot": args.boot,
        "seed": args.seed,
    }
    units = [
        Unit(
            f"{k}__{_uid(c)}__{ctl}",
            {"key": k, "candidate": c, "control": ctl, **common},
        )
        for k in sorted(keys)
        for c in cands
        for ctl in ctls
    ]
    print(f"[telemetry_fitness] {len(units)} units", flush=True)
    res = gridrun_from_args(args, units, worker, args.out, summarize=summarize)
    print_table(res.summary)
    raise SystemExit(res.exit_code)


if __name__ == "__main__":
    main()
