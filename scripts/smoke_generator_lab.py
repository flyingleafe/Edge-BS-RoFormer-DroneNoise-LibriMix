"""Headless exercise of every code path in notebooks/generator_lab.py.

The notebook itself cannot be run in CI (ipywidgets need a kernel with a live
front end), so this drives the same library functions with the same arguments
the widgets would supply. If this passes, the notebook's cells work; what it
does not check is the widget wiring.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "notebooks"))

import generator_lab as lab  # noqa: E402


def _check(label: str, fn):
    try:
        out = fn()
        print(f"  OK   {label}: {out}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False


def main() -> int:
    ok = True
    print("=== registry ===")
    print(f"  {len(lab.VARIANTS)} variants")

    print("=== excitations ===")
    excs: dict[str, lab.Excitation] = {}

    for dataset in lab.DATASETS:

        def build_real(ds=dataset):
            recs = lab.recordings(ds)
            e = lab.real_slice(ds, recs[0], 20.0, 3.0)  # recordings() filters unusable ones
            excs[ds] = e
            return (
                f"{e.label} audio{e.audio.shape} rps{e.rps.shape} "
                f"mics{e.mic_pos.shape} meanRPS {e.mean_rps:.1f}"
            )

        ok &= _check(f"real_slice({dataset})", build_real)

    for kind in ("intermittent", "full_flight"):

        def build_synth(k=kind):
            e = lab.synth_slice(drone="dregon", kind=k, dur_s=3.0, seed=0)
            excs[f"synth/{k}"] = e
            return f"rps{e.rps.shape} meanRPS {e.mean_rps:.1f}"

        ok &= _check(f"synth_slice({kind})", build_synth)

    ref = excs.get("DREGON-frames") or next(iter(excs.values()))

    print("=== render every variant ===")
    for name, spec in lab.VARIANTS.items():

        def render(n=name):
            a = lab.render(n, ref)
            assert a.ndim == 2, a.shape
            assert np.isfinite(a).all(), "non-finite audio"
            return f"{a.shape} rms {float(np.sqrt(np.mean(a**2))):.2e}"

        got = _check(f"render[{spec.family}] {name}", render)
        # GP/CONA need artifacts that may be absent; the
        # learned families and `real` must work.
        if spec.family in ("deep", "real"):
            ok &= got

    print("=== embedding sweep (the notebook's alpha slider) ===")
    deep = next(n for n, s in lab.VARIANTS.items() if s.family == "deep" and s.conditioned)
    for alpha in (-0.5, 0.0, 0.5, 1.0, 1.5):
        ok &= _check(
            f"alpha={alpha:+.1f}",
            lambda a=alpha: str(lab.render(deep, ref, alpha=a, offset=0.2).shape),
        )
    ok &= _check("wind off", lambda: str(lab.render(deep, ref, wind=False).shape))
    ok &= _check("jitter override", lambda: str(lab.render(deep, ref, jitter_sigma=0.3).shape))

    print("=== analysis helpers ===")

    def analyse():
        audio = lab.render(deep, ref)
        db, freqs, times = lab.spectrogram(audio[0])
        col = lab.spectrum_at(db, times, times[-1] / 2)
        matched = lab.match_rms(audio[0], ref.audio[0])
        return f"spec{db.shape} col{col.shape} rms {float(np.sqrt(np.mean(matched**2))):.2e}"

    ok &= _check("spectrogram/spectrum/match_rms", analyse)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
