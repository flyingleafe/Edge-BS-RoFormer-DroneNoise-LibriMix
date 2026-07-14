"""Prepare figure assets for the 2026-07-13 rps-synthetic-data-status slide deck.

Reads from other reports' assets/ and docs/experiments/, writes only into
this deck's own assets/ dir.
"""

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ROOT = HERE.parents[2]  # repo root

ASSETS.mkdir(exist_ok=True)

FULLFLIGHT_REPORT = ROOT / "writing/reports/2026-07-12_full-flight-sim2real-rps/assets"
REFINEMENT_REPORT = ROOT / "writing/reports/2026-07-10_rps-refinement/assets"

COPIES = [
    (FULLFLIGHT_REPORT / "silence_fade.png", ASSETS / "silence_fade.png"),
    (FULLFLIGHT_REPORT / "fullflight.png", ASSETS / "fullflight.png"),
    (FULLFLIGHT_REPORT / "tracking.png", ASSETS / "tracking.png"),
    (REFINEMENT_REPORT / "method_comb_alignment.png", ASSETS / "method_comb_alignment.png"),
    (REFINEMENT_REPORT / "val_overlay.png", ASSETS / "val_overlay.png"),
    # results_table.typ is NOT auto-copied: this deck's copy has been hand-edited
    # (bolded per-architecture best "All" cells) and must not be clobbered by a
    # re-run of prepare.py.
]


def copy_report_figures():
    for src, dst in COPIES:
        if src.exists():
            shutil.copy(src, dst)
            print(f"copied {src} -> {dst}")
        else:
            print(f"MISSING (skip): {src}")


def generate_interp_strips_and_timewarp():
    """Regenerate the drone-embedding interpolation strips (from the trained E6
    per-drone generator checkpoint) and the time-warp before/after figure (from
    the project's real time_warp code on one DREGON clip). Best-effort: if the
    checkpoint/data are not available on this machine, skip with a message
    rather than failing the whole build (these are the two slowest, most
    fragile figures)."""
    try:
        import runpy

        runpy.run_path(str(HERE / "prepare_interp_strip.py"), run_name="__main__")
        runpy.run_path(str(HERE / "prepare_timewarp_fig.py"), run_name="__main__")
        runpy.run_path(str(HERE / "prepare_jitter_decompose.py"), run_name="__main__")
    except Exception as e:  # noqa: BLE001 - best-effort figure generation
        print(f"SKIPPED interp/time-warp figure generation ({e})")


def generate_regime_eval():
    """Best-effort: CPU inference over the 3 full-flight checkpoints (R2) to
    get per-regime MAE + prediction overlays. Slow (~a few minutes: dload
    stream + 3x CPU forward passes over ~300 windows) — skip gracefully if
    R2/dload is unreachable rather than failing the whole build."""
    try:
        import runpy

        runpy.run_path(str(HERE / "prepare_regime_eval.py"), run_name="__main__")
    except Exception as e:  # noqa: BLE001 - best-effort figure/table generation
        print(f"SKIPPED regime-eval figure/table generation ({e})")


def main():
    copy_report_figures()
    generate_interp_strips_and_timewarp()
    generate_regime_eval()


if __name__ == "__main__":
    main()
